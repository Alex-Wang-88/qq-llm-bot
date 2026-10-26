"""Strip model-side conversation structure before a reply reaches the user.

The 4B adapter learned to write longer answers, and with that came a failure the
shorter 9B adapter never showed: the model keeps generating past its own turn,
inventing `user`/`assistant` lines and empty `<think>` blocks until it hits the
token ceiling. The training data contains none of those tokens, so this is a
base-model prior, not something retraining will remove. The guard is the fix.

`StreamGuard` must produce exactly the same text as `sanitize_text` for any
input, including the case where the whole answer sits inside a think block.
"""

import re

# A line that is nothing but a role name is never part of a Chinese reply.
# Covers plain `user` and the Qwen `<|im_start|>user` form.
ROLE_LINE = re.compile(r'(?im)^[ \t]*(?:<\|[a-z_]+\|>[ \t]*)?(?:user|assistant|system)[ \t]*:?[ \t]*\r?$')
THINK_OPEN = '<think>'
THINK_CLOSE = '</think>'

# Longest marker is `<|im_start|>assistant\n` at 22 chars; 24 leaves margin so a
# marker split across two stream chunks can never leak half-emitted.
HOLDBACK = 24


def cut_at_role(text):
    """Return (kept, marker). The marker line and everything after it is dropped."""
    match = ROLE_LINE.search(text)
    if not match:
        return text, None
    return text[:match.start()], match.group(0).strip()


def split_think(text):
    """Separate visible text from think-block content.

    Both are returned so callers can fall back to thinking when a model puts
    its entire answer inside `<think>` and leaves nothing outside.
    """
    visible, thinking = [], []
    index = 0
    while index < len(text):
        start = text.find(THINK_OPEN, index)
        if start < 0:
            visible.append(text[index:])
            break
        visible.append(text[index:start])
        close = text.find(THINK_CLOSE, start + len(THINK_OPEN))
        if close < 0:
            thinking.append(text[start + len(THINK_OPEN):])
            break
        thinking.append(text[start + len(THINK_OPEN):close])
        index = close + len(THINK_CLOSE)
    return ''.join(visible), ''.join(thinking)


def sanitize_text(text):
    """Clean a complete reply. Returns (text, info) where info records what fired."""
    if not isinstance(text, str) or not text:
        return text, {}
    info = {}
    kept, marker = cut_at_role(text)
    if marker:
        info['role_marker'] = marker
        info['dropped_chars'] = len(text) - len(kept)
    visible, thinking = split_think(kept)
    if thinking:
        info['stripped_think_chars'] = len(thinking)
    # A model that reasoned its way to the answer but forgot to leave it outside
    # the block still owes the user that text; dropping it would answer with silence.
    clean = visible if visible.strip() else thinking
    clean = clean.strip()
    if not clean:
        info['empty'] = True
    if clean != text.strip():
        info['changed'] = True
    return clean, info


class StreamGuard:
    """Incremental version of `sanitize_text`, safe across arbitrary chunk splits."""

    def __init__(self, holdback=HOLDBACK):
        self._pending = ''
        self._thinking = ''
        self._in_think = False
        self._blocked = False
        self._started = False
        self._holdback = holdback

    @property
    def blocked(self):
        return self._blocked

    def _keep_for_marker(self):
        """Hold back enough tail that a marker straddling chunks stays intact."""
        return self._holdback

    def _emit(self, text):
        """Suppress leading whitespace so streaming matches the batch strip()."""
        if not text or self._started:
            return text
        text = text.lstrip()
        if text:
            self._started = True
        return text

    def feed(self, chunk):
        if self._blocked or not chunk:
            return ''
        self._pending += chunk
        out = []
        while True:
            if self._in_think:
                close = self._pending.find(THINK_CLOSE)
                if close < 0:
                    # Keep only what could still contain the closing tag.
                    keep = len(THINK_CLOSE) - 1
                    if len(self._pending) > keep:
                        self._thinking += self._pending[:-keep]
                        self._pending = self._pending[-keep:]
                    break
                self._thinking += self._pending[:close]
                self._pending = self._pending[close + len(THINK_CLOSE):]
                self._in_think = False
                continue
            start = self._pending.find(THINK_OPEN)
            marker = ROLE_LINE.search(self._pending)
            if marker and (start < 0 or marker.start() < start):
                # The newline anchoring the marker belongs to the dropped turn.
                out.append(self._emit(self._pending[:marker.start()]))
                self._blocked = True
                self._pending = ''
                return ''.join(out).rstrip()
            if start >= 0:
                out.append(self._emit(self._pending[:start]))
                self._pending = self._pending[start + len(THINK_OPEN):]
                self._in_think = True
                continue
            hold = self._keep_for_marker()
            if len(self._pending) > hold:
                cut = len(self._pending) - hold
                out.append(self._emit(self._pending[:cut]))
                self._pending = self._pending[cut:]
            break
        return ''.join(out)

    def flush(self):
        """Emit whatever is safe, mirroring `sanitize_text`'s think fallback."""
        if self._blocked:
            self._pending = ''
            self._thinking = ''
            return ''
        remainder = self._pending
        self._pending = ''
        if self._in_think:
            self._thinking += remainder
            self._in_think = False
            remainder = ''
        if not remainder.strip() and self._thinking.strip():
            return self._emit(self._thinking.strip())
        return self._emit(remainder).rstrip()
