"""Small HTML text extractor with no extra dependencies."""
from html.parser import HTMLParser
import re


class PageText(HTMLParser):
    SKIP = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "template", "svg"}
    VOID = {"meta", "link", "br", "hr", "img", "input", "source", "wbr", "area", "base", "embed", "param", "track", "col"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.body, self.article, self.main, self.title = [], [], [], []
        self.published = ""

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta" and attrs.get("property", attrs.get("name", "")).lower() in {
            "article:published_time", "datepublished", "pubdate", "date"
        }:
            self.published = attrs.get("content", "")[:100]
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag in self.stack:
            index = len(self.stack) - 1 - self.stack[::-1].index(tag)
            del self.stack[index:]

    def handle_data(self, data):
        text = re.sub(r"\s+", " ", data).strip()
        if not text or any(tag in self.SKIP for tag in self.stack):
            return
        if "title" in self.stack:
            self.title.append(text)
            return
        if "head" in self.stack:
            return
        self.body.append(text)
        if "article" in self.stack:
            self.article.append(text)
        if "main" in self.stack:
            self.main.append(text)

    def result(self, limit=2200):
        selected = self.article or self.main or self.body
        text = "\n".join(selected)
        return {"title": " ".join(self.title)[:180], "published": self.published,
                "text": text[:limit], "truncated": len(text) > limit}


def extract_page(html):
    parser = PageText()
    parser.feed(html)
    return parser.result()
