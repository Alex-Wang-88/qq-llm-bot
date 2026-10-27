"""Small, separate SQLite routing store. Chat history is never migrated here."""
import json
from pathlib import Path
import sqlite3
import time


class AmbiguousTarget(ValueError): pass


class SessionStore:
    def __init__(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(str(path))
        self.db.row_factory=sqlite3.Row
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY,harness TEXT NOT NULL,native_id TEXT NOT NULL,project_path TEXT,title TEXT,status TEXT,observed_at REAL,UNIQUE(harness,native_id));
          CREATE TABLE IF NOT EXISTS bindings(actor TEXT PRIMARY KEY,session_id INTEGER NOT NULL,expires_at REAL NOT NULL);
          CREATE TABLE IF NOT EXISTS notifications(event_key TEXT PRIMARY KEY,actor TEXT NOT NULL,session_id INTEGER,qq_message_id TEXT,state TEXT NOT NULL,payload TEXT,created_at REAL NOT NULL);
          CREATE TABLE IF NOT EXISTS pending_messages(actor TEXT PRIMARY KEY,content TEXT NOT NULL,expires_at REAL NOT NULL);
          CREATE TABLE IF NOT EXISTS tracked_turns(harness TEXT NOT NULL,native_id TEXT NOT NULL,turn_id TEXT NOT NULL,actor TEXT NOT NULL,session_id INTEGER NOT NULL,status TEXT NOT NULL,started_at REAL,updated_at REAL NOT NULL,PRIMARY KEY(harness,native_id,turn_id));
        ''')

    def close(self): self.db.close()

    def save(self,sessions):
        # One timestamp for the whole batch: the API returns newest-first, so a
        # per-item clock reading would reverse it (later items look fresher).
        observed_at=time.time()
        with self.db:
            for item in sessions:
                self.db.execute('''INSERT INTO sessions(harness,native_id,project_path,title,status,observed_at) VALUES(?,?,?,?,?,?)
                  ON CONFLICT(harness,native_id) DO UPDATE SET project_path=excluded.project_path,title=excluded.title,status=excluded.status,observed_at=excluded.observed_at''',
                  (item['harness'],item['native_id'],item.get('project_path'),item.get('title'),item.get('status'),observed_at))

    def list(self,harness=None):
        if harness:return [dict(r) for r in self.db.execute('SELECT * FROM sessions WHERE harness=? ORDER BY observed_at DESC,id',(harness,))]
        return [dict(r) for r in self.db.execute('SELECT * FROM sessions ORDER BY harness,id')]

    def resolve(self,selector,harness=None):
        rows=self.list(harness)
        matches=[r for r in rows if str(r['id'])==str(selector) or r['native_id']==selector or r['project_path']==selector or r['title']==selector]
        if len(matches)>1:raise AmbiguousTarget('匹配到多个会话，请使用列表中的编号明确指定。')
        if not matches:raise ValueError('未找到该会话，请先刷新会话列表并指定编号。')
        return matches[0]

    def bind(self,actor,identity,ttl=300):
        with self.db:self.db.execute('INSERT OR REPLACE INTO bindings VALUES(?,?,?)',(actor,identity,time.time()+ttl))

    def consume_binding(self,actor,now=None):
        now=time.time() if now is None else now
        with self.db:
            row=self.db.execute('DELETE FROM bindings WHERE actor=? RETURNING *',(actor,)).fetchone()
        if not row or row['expires_at']<now:raise ValueError('会话选择已过期或已使用，请重新指定目标。')
        return self.resolve(str(row['session_id']))

    def notification_target(self,actor,message_id):
        row=self.db.execute('SELECT session_id FROM notifications WHERE actor=? AND qq_message_id=? AND state=?',(actor,message_id,'delivered')).fetchone()
        if not row or row['session_id'] is None:return None
        return self.resolve(str(row['session_id']))

    def stage_message(self,actor,text,ttl=300):
        if not isinstance(text,str) or not text.strip():raise ValueError('续聊内容不能为空。')
        if len(text)>4000:raise ValueError('续聊内容超过 4000 字。')
        with self.db:
            self.db.execute('DELETE FROM pending_messages WHERE expires_at<?',(time.time(),))
            self.db.execute('INSERT OR REPLACE INTO pending_messages VALUES(?,?,?)',(actor,text,time.time()+ttl))

    def cleanup_pending(self,now=None):
        now=time.time() if now is None else now
        with self.db:self.db.execute('DELETE FROM pending_messages WHERE expires_at<?',(now,))

    def consume_message(self,actor,now=None):
        now=time.time() if now is None else now
        with self.db:
            row=self.db.execute('DELETE FROM pending_messages WHERE actor=? RETURNING *',(actor,)).fetchone()
        if not row or row['expires_at']<now:raise ValueError('待确认消息不存在或已过期，请重新发送。')
        return row['content']

    def register_turn(self,actor,target,turn_id):
        if not isinstance(turn_id,str) or not turn_id:raise ValueError('Harness 未返回回合 ID，无法跟踪。')
        with self.db:
            self.db.execute('INSERT INTO tracked_turns VALUES(?,?,?,?,?,?,?,?)',
                (target['harness'],target['native_id'],turn_id,actor,target['id'],'accepted',None,time.time()))

    def observe_turn(self,harness,native_id,turn_id,status,now=None):
        if status not in ('inProgress','completed','failed','interrupted'):return None
        now=time.time() if now is None else now
        with self.db:
            row=self.db.execute('SELECT * FROM tracked_turns WHERE harness=? AND native_id=? AND turn_id=?',
                                (harness,native_id,turn_id)).fetchone()
            if not row or row['status'] not in ('accepted','inProgress'):return None
            started=row['started_at'] or (now if status=='inProgress' else None)
            self.db.execute('UPDATE tracked_turns SET status=?,started_at=?,updated_at=? WHERE harness=? AND native_id=? AND turn_id=?',
                            (status,started,now,harness,native_id,turn_id))
            return dict(row)

    def tracked(self,actor):
        return [dict(row) for row in self.db.execute('SELECT * FROM tracked_turns WHERE actor=? ORDER BY updated_at DESC LIMIT 10',(actor,))]

    def mark_tracking_unknown(self):
        with self.db:self.db.execute("UPDATE tracked_turns SET status='unknown',updated_at=? WHERE status IN ('accepted','inProgress')",(time.time(),))

    def mark_pending_notifications_unknown(self):
        with self.db:self.db.execute("UPDATE notifications SET state='delivery_unknown' WHERE state='pending'")

    def reserve_notification(self,key,actor,session_id):
        with self.db:
            row=self.db.execute('INSERT OR IGNORE INTO notifications VALUES(?,?,?,?,?,?,?)',(key,actor,session_id,None,'pending','{}',time.time()))
            return row.rowcount==1

    def finish_notification(self,key,state,message_id=None):
        if state not in ('delivered','delivery_unknown'):raise ValueError('invalid delivery state')
        with self.db:self.db.execute('UPDATE notifications SET state=?,qq_message_id=? WHERE event_key=? AND state=?',(state,message_id,key,'pending'))
