"""Narrow, time-stamped A-share index snapshot for simple market questions."""
from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation
import re

import httpx

QUOTE_URL = "https://qt.gtimg.cn/q=sh000001,sz399001,sz399006"
SYMBOLS = {"sh000001": "上证指数", "sz399001": "深证成指", "sz399006": "创业板指"}
INDEX = r"(?:A股|大盘|三大指数|上证(?:指数)?|深证成指|深成指|创业板(?:指)?|科创50|沪深300)"
SIMPLE = re.compile(
    r"^(?:请|帮我|麻烦|我想|想|能不能|可以|给我|查一下|查询一下|查查|看看|看一下|了解一下|再查一下|再看看|搜一下|说说|告诉我)*"
    r"(?:今天|今日|现在|最新|目前)?(?:的)?" + INDEX +
    r"(?:今天|今日|现在|最新|目前)?(?:的)?"
    r"(?:是|会)?"
    r"(?:行情|走势|怎么走|收盘(?:情况|数据|行情)?|指数|情况|表现|怎么样|如何|涨跌|咋样"
    r"|高开(?:还是|或)低开|低开(?:还是|或)高开|高开|低开|开盘|涨(?:了)?(?:多少)?|跌(?:了)?(?:多少)?)?"
    r"(?:怎么样|如何|是多少|多少|咋样)?[?？。！!]*$", re.I)


def is_simple_market_query(text):
    return bool(SIMPLE.fullmatch(re.sub(r"\s+", "", text or "")))


def is_market_followup(text, contexts):
    if not re.fullmatch(r"(?:再|帮我|请|那就|重新)*(?:查一下|查询一下|查查|再查|查一遍|查吧|看看)[?？。！!]*", re.sub(r"\s+", "", text or "")):
        return False
    if not isinstance(contexts, list):
        return False
    for message in reversed(contexts):
        if not isinstance(message, dict) or message.get("role") != "assistant" or message.get("tool_calls"):
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
        return bool(isinstance(content, str) and re.search(r"A股|上证指数|深证成指|创业板指", content, re.I))
    return False


def _decimal_or_none(raw):
    """Open/prev-close are optional: a bad field must not drop the whole quote."""
    try:
        value = Decimal(raw)
    except (InvalidOperation, ValueError, TypeError):
        return None
    return value if value.is_finite() else None


def parse_quotes(text):
    found = {}
    for symbol, value in re.findall(r'v_(sh000001|sz399001|sz399006)="([^"\r\n]*)"', text):
        fields = value.split("~")
        try:
            if len(fields) <= 32 or not re.fullmatch(r"\d{14}", fields[30]):
                continue
            timestamp = datetime.strptime(fields[30], "%Y%m%d%H%M%S")
            price, change = Decimal(fields[3]), Decimal(fields[32])
            if not price.is_finite() or not change.is_finite() or price <= 0 or abs(change) > 100:
                continue
        except (ValueError, InvalidOperation):
            continue
        found[symbol] = {
            "name": SYMBOLS[symbol], "price": price, "change": change, "time": timestamp,
            "open": _decimal_or_none(fields[5]), "prev_close": _decimal_or_none(fields[4]),
        }
    return found


def open_verdict(quote):
    """高开/低开 is open vs previous close. Missing or zero fields -> no verdict."""
    today_open, prev_close = quote.get("open"), quote.get("prev_close")
    if today_open is None or prev_close is None or today_open <= 0 or prev_close <= 0:
        return ""
    diff = today_open - prev_close
    if abs(diff) < Decimal("0.01"):
        return "平开"
    return f"{'高开' if diff > 0 else '低开'} {abs(diff):.2f} 点"


async def market_snapshot():
    try:
        async with httpx.AsyncClient(timeout=8, trust_env=False, follow_redirects=False) as client:
            response = await client.get(QUOTE_URL, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
        quotes = parse_quotes(response.content.decode("gb18030", errors="replace"))
    except (httpx.HTTPError, UnicodeError):
        return "实时行情查询失败，暂时无法核实当前 A 股指数；不能据此判断是否休市。"
    if not quotes:
        return "行情来源没有返回可核对的指数和时间，暂时无法确认当前行情；不能据此判断是否休市。"
    entries = []
    for symbol in SYMBOLS:
        if symbol in quotes:
            q = quotes[symbol]
            entries.append(f"{q['name']} {q['price']:.2f} 点（{q['change']:+.2f}%），数据时间 {q['time']:%Y-%m-%d %H:%M:%S}")
    if len(quotes) < len(SYMBOLS):
        entries.append("部分指数未取得有效数据")
    opens = [f"{q['name']} {verdict}" for q in quotes.values() if (verdict := open_verdict(q))]
    if opens:
        entries.append("今日开盘：" + "；".join(opens))
    now = datetime.now(timezone(timedelta(hours=8)))
    dated = [q["time"] for q in quotes.values()]
    caution = "数据时间以来源为准；这份快照不单独证明休市或最终收盘状态。"
    if any(t.date() != now.date() for t in dated):
        caution = "部分或全部数据不是今天的，无法确认今天的行情；不能据此判断休市。"
    elif now.hour >= 15 and any(t.hour < 15 for t in dated):
        caution = "部分或全部数据早于今日 15:00，不能当作今天的收盘数据。"
    return "A 股指数行情快照（北京时间）：\n" + "\n".join(entries) + "\n来源：腾讯行情 " + QUOTE_URL + "\n" + caution
