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
