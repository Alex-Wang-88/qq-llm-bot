"""Redaction shared by diagnostics, task output, and automatic memory."""
import re
import logging

SENSITIVE_KEY=re.compile(r"(?i)(?:password|passwd|secret|api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|cookie)")
TOKEN_PATTERNS=[
    re.compile(r"(?i)[?&](?:x-amz-signature|x-amz-credential|access_token|refresh_token|api_key|token|signature)=[^&\s\"']+"),
    re.compile(r"-----BEGIN [^-]*PRIVATE KEY-----[\s\S]*?-----END [^-]*PRIVATE KEY-----"),
    re.compile(r"(?im)^\s*(?:cookie|set-cookie|authorization)\s*:[^\r\n]*"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{8,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    re.compile(r"(?i)((?:password|passwd|api[_-]?key|access[_-]?token|refresh[_-]?token|secret|密码|口令|令牌)\s*(?:是|为|[:=：])\s*)[^\s,，;；\"'}]+"),
]


def contains_secret(text):
    return any(pattern.search(text or "") for pattern in TOKEN_PATTERNS)


def redact_text(text):
    for pattern in TOKEN_PATTERNS:
        text=pattern.sub("[REDACTED]",text)
    return text


def redact(value,key=""):
    if SENSITIVE_KEY.search(key):return "[REDACTED]"
    if isinstance(value,dict):return {k:redact(v,str(k)) for k,v in value.items()}
    if isinstance(value,list):return [redact(v,key) for v in value]
    if isinstance(value,str):
        if value.startswith('data:image/'):
            return f"[image omitted; encoded_chars={len(value)}]"
        return redact_text(value)
    return value


def install_log_redaction():
    """Render then scrub, so secrets passed as formatting args are covered too."""
    original=logging.getLogRecordFactory()
    def factory(*args,**kwargs):
        record=original(*args,**kwargs)
        try:record.msg=redact_text(record.getMessage())
        except Exception:record.msg='[unformattable log record]'
        record.args=()
        if record.exc_info:
            record.exc_text=redact_text(logging.Formatter().formatException(record.exc_info))
        return record
    logging.setLogRecordFactory(factory)
    return original
