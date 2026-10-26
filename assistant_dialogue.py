"""Cheap intent hints; routing never grants tool permissions or guesses a target."""
from dataclasses import dataclass
import re

HARNESS=re.compile(r'codex|workbuddy|zcode',re.I)
QUERY=re.compile(r'https?://|搜索|搜一下|搜一遍|再搜|查一下|查一遍|查查|再查|最新|实时|新闻|天气|价格|行情|指数|网页|链接|版本')
COMPUTER=re.compile(r'(?:打开|启动|停止|关掉|查找|压缩|整理|查看|检查).{0,16}(?:软件|程序|进程|文件|目录|磁盘|CPU|GPU|内存|显存|日志|下载)',re.I)
DEEP=re.compile(r'^\s*(?:请)?(?:深度思考|认真推理|/思考)\s*[:：\s]')


@dataclass(frozen=True)
class Intent:
    mode:str
    thinking:bool=False


def classify(text):
    thinking=bool(DEEP.search(text))
    if text.strip() in ('额度','会话','任务'):mode='harness'
    elif HARNESS.search(text) and re.search(r'继续|发送|发给|交给|让|会话|任务|额度|状态|进度|查询|查看',text):mode='harness'
    elif re.search(r'任务(?:状态|进度)|剩余.{0,5}额度|哪个.{0,8}会话',text):mode='harness'
    elif COMPUTER.search(text):mode='computer'
    elif QUERY.search(text):mode='query'
    else:mode='chat'
    return Intent(mode,thinking)


def is_harness_dispatch(text):
    """Only explicit dispatch phrasing; status questions and quoted web text are not commands."""
    if not HARNESS.search(text) or text.lstrip().startswith(('/','http://','https://')):return False
    return bool(re.search(r'(?:^|[。！!\n])\s*(?:(?:请|帮我|麻烦)\s*)?(?:让|叫|告诉|交给|转交|发给|把|发送|继续)',text)) and bool(re.search(r'继续|发给|交给|转交|发送|修复|排查|开发|处理',text))
