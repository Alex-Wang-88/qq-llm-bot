"""Arithmetic answered by code instead of by the model: stdlib only, no eval.

Why this exists (measured 2026-10-27): the deployed 4B answered ``23 × 7 - 18``
as 64, then 81, then 169; the 9B answered 141. The training report records the
same failure mode ("skips the steps and guesses"), so this is not a prompt
problem and cannot be fixed by asking more nicely. Arithmetic moves here, and
the model keeps only the job of putting the number into a sentence.

Safety: nothing in this module can execute user text as Python. The expression
is parsed with ``ast.parse(mode="eval")``, checked against a node whitelist, and
walked by ``_eval`` — a private evaluator that understands nothing but numbers
and four operations. ``__import__("os").system("dir")`` dies at the whitelist
because a ``Call``/``Name`` node is refused before any evaluation happens.

Exactness: integer literals stay integers, decimal literals become the exact
fraction they were written as (``0.1`` becomes 1/10), and every operation runs
on ``fractions.Fraction``. ``0.1 + 0.2`` therefore prints ``0.3`` instead of
``0.30000000000000004``. Results come out as an integer when they are whole, as
a terminating decimal when one exists within 12 places, and as a fraction
otherwise.
"""

from __future__ import annotations

import ast
from fractions import Fraction
import re

MAX_TEXT = 200          # characters; longer input is a sentence, not an expression
MAX_OPERAND_DIGITS = 18  # a literal is refused above this, so no 100-digit input
MAX_EXPONENT = 12        # 9 ** 9 ** 9 and 2 ** 999999 are refused here
MAX_RESULT_DIGITS = 120  # backstop for nested powers
MAX_FACTS = 3            # verified results injected into one prompt

# Input normalisation. `^` becomes `**` because that is what people type, and a
# lone `^` in Python means XOR, which would silently give a wrong number.
_CHAR_MAP = {
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    "（": "(", "）": ")", "．": ".", "＋": "+", "－": "-",
    "＊": "*", "／": "/", "＝": "=", "％": "%", "＾": "**", "^": "**",
    "×": "*", "÷": "/", "−": "-", "–": "-", "—": "-", "　": " ",
}
_TRANSLATION = str.maketrans(_CHAR_MAP)

# Thousands separators only in the strict 1,234,567 form. `1,234,5` keeps its
# commas and is then refused as an expression, which is what we want: it is a
# list, not a number.
_THOUSANDS = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?)(?![\d])")

_PURE = re.compile(r"^[0-9.+\-*/%()\s]+$")
# A binary operator between operand characters. `**` is listed separately: a
# first-character check would reject `2**8` as "no binary operator".
_BINARY_OP = re.compile(r"\d\s*(?:\*\*|//|[+\-*/%])\s*[\d(]")
_DATE = re.compile(r"^\d{4}\s*[/-]\s*\d{1,2}\s*[/-]\s*\d{1,2}$")
# Longest arithmetic-looking run inside a sentence. Greedy on purpose: the whole
# run is validated afterwards, so a false start costs one rejection rather than
# a wrong number.
_SPAN = re.compile(r"[0-9(][0-9 .+\-*/%()]*[0-9)]")
# "第 3-5 章" is a range, and answering -2 for it would be actively wrong.
# Tuples, not strings: `"" in "第"` is True, so a string would silently treat
# every expression at the start of the text as a range.
_RANGE_BEFORE = ("第",)
_RANGE_AFTER = ("章", "页", "条", "行", "号", "天", "次", "年", "月", "日")

_SEPARATORS = " \t\r\n:：=＝,，、。．！!？?"
_TRAILING_PUNCT = "。．！!？?=＝:：,， "

# Wrapper words people put around a computation. Longest first: "帮我算一下"
# has to win over "帮我". Only fixed vocabulary is stripped — this is not a
# general Chinese parser, and a sentence with real content words left over is
# deliberately not treated as an expression.
_LEAD_WORDS = (
    "帮我算一下", "帮我算一算", "帮我算算", "帮我计算", "帮我求",
    "帮忙算一下", "帮忙算一算", "帮忙计算",
    "请你算一下", "请算一下", "麻烦算一下",
    "只算结果", "只给结果", "只要结果", "只算", "只求",
    "算一下", "算一算", "算算", "计算一下", "计算", "求一下", "求",
    "答案是", "答案", "结果是", "结果", "请", "帮我", "帮忙", "麻烦",
)
_TAIL_WORDS = (
    "等于多少", "等于几", "是多少", "是几", "结果为", "结果", "多少", "等于",
)

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.UAdd, ast.USub,
)


def normalize_chars(text):
    """Map full-width digits, ×/÷ and typed `^` onto Python operators."""
    return text.translate(_TRANSLATION)


def strip_thousands(text):
    """Drop commas only where they are separators: `1,234` yes, `1,234,5` no."""
    return _THOUSANDS.sub(lambda match: match.group(1).replace(",", ""), text)


def _strip_wrapper(text):
    """Peel polite wording off both ends. Returns (rest, words_removed)."""
    text = text.strip()
    words_removed = False
    for _ in range(8):  # bounded: no input can make this loop forever
        before = text
        text = text.strip(_SEPARATORS)
        for word in _LEAD_WORDS:
            if text.startswith(word):
                text = text[len(word):]
                words_removed = True
                break
        for word in _TAIL_WORDS:
            if text.endswith(word):
                text = text[: -len(word)]
                words_removed = True
                break
        if text == before:
            break
    return text.strip(_TRAILING_PUNCT), words_removed


def _allowed(tree):
    """Node whitelist. Anything that could reach Python or the filesystem fails."""
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return False
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            if isinstance(value, int) and len(str(abs(value))) > MAX_OPERAND_DIGITS:
                return False
    return True


def _eval(node):
    """Evaluate a whitelisted numeric node into an exact Fraction."""
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, int):
            return Fraction(value)
        return Fraction(str(value))  # 0.1 -> 1/10, not the binary float
    if isinstance(node, ast.UnaryOp):
        operand = _eval(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        raise ValueError("unsupported unary operator")
    if isinstance(node, ast.BinOp):
        left = _eval(node.left)
        right = _eval(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return Fraction(left // right)
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise ZeroDivisionError("modulo by zero")
            return left % right
        if isinstance(node.op, ast.Pow):
            if right.denominator != 1:
                raise ValueError("non-integer exponent")
            exponent = int(right)
            if abs(exponent) > MAX_EXPONENT:
                raise ValueError("exponent too large")
            return left ** exponent
        raise ValueError("unsupported binary operator")
    raise ValueError("unsupported node")


def _too_large(value):
    return (
        len(str(abs(value.numerator))) > MAX_RESULT_DIGITS
        or len(str(value.denominator)) > MAX_RESULT_DIGITS
    )


def evaluate_expression(expression):
    """Parse and evaluate an already-normalised expression; None if refused."""
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, MemoryError):
        return None
    if not _allowed(tree):
        return None
    try:
        value = _eval(tree.body)
    except (ArithmeticError, ValueError, TypeError, RecursionError):
        return None
    if not isinstance(value, Fraction) or _too_large(value):
        return None
    return value


def _decimal(value):
    """Exact decimal string when the expansion terminates, else None."""
    denominator = value.denominator
    twos = fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        return None
    places = max(twos, fives)
    if places == 0:
        return str(value.numerator)
    if places > 12:
        return None
    scaled = value.numerator * 10 ** places // value.denominator
    sign = "-" if scaled < 0 else ""
    digits = str(abs(scaled)).rjust(places + 1, "0")
    whole, fraction = digits[:-places], digits[-places:]
    fraction = fraction.rstrip("0")
    return f"{sign}{whole}.{fraction}" if fraction else f"{sign}{whole}"


def format_value(value):
    """Exact, shortest form: integer if whole, else decimal, else fraction."""
    if value.denominator == 1:
        return str(value.numerator)
    decimal = _decimal(value)
    if decimal is not None:
        return decimal
    return f"{value.numerator}/{value.denominator}"


def parse_expression(text):
    """Return (display, normalised, wrapped) for a pure arithmetic message.

    ``wrapped`` says the user wrote a sentence around it ("算一下 …") rather
    than a bare expression, which decides whether the reply echoes the
    expression or just gives the number.
    """
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text or len(text) > MAX_TEXT:
        return None
    normalised = strip_thousands(normalize_chars(text))
    if "," in normalised:  # a comma survived: a list, not a number
        return None
    body, wrapped = _strip_wrapper(normalised)
    if not body or not _PURE.match(body):
        return None
    if not _BINARY_OP.search(body):
        return None
    if _DATE.match(body):  # 2026/10/27 is a date, not 7.5
        return None
    display, _ = _strip_wrapper(text)
    return display, body, wrapped


def solve(text):
    """Reply text for a pure arithmetic message; None means "not arithmetic"."""
    parsed = parse_expression(text)
    if parsed is None:
        return None
    display, body, wrapped = parsed
    value = evaluate_expression(body)
    if value is None:
        return None
    shown = format_value(value)
    return f"{display} = {shown}" if wrapped else shown


def find_facts(text):
    """Expressions embedded in a sentence, as (expression, exact value) pairs.

    These are not answered here: the sentence still needs the model. The value
    is injected into the prompt as verified, so the model only has to quote it.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    body = strip_thousands(normalize_chars(text))
    facts = []
    seen = set()
    for match in _SPAN.finditer(body):
        span = match.group(0).strip()
        if len(span) < 3 or not _BINARY_OP.search(span) or not _PURE.match(span):
            continue
        # Look past a space so "第 3-5 章" is still recognised as a range.
        before = body[max(0, match.start() - 2): match.start()].strip()
        after = body[match.end(): match.end() + 2].strip()[:1]
        if before.endswith(_RANGE_BEFORE) or after in _RANGE_AFTER:
            continue
        value = evaluate_expression(span)
        if value is None:
            continue
        item = (span, format_value(value))
        if item in seen:
            continue
        seen.add(item)
        facts.append(item)
        if len(facts) >= MAX_FACTS:
            break
    return facts
