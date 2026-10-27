"""
本地 SearXNG 联网搜索插件

把本机自建的 SearXNG 元搜索引擎接入 AstrBot，为 LLM 提供两个工具：
- web_search: 联网搜索（聚合 Google/Bing/DuckDuckGo 等，出站走本机 VPN 代理）
- fetch_web_page: 读取网页正文（用户发链接时用）
"""

from __future__ import annotations

import re
import json
from urllib.parse import urlsplit

import httpx

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star
from .web_text import extract_page

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def record_evidence(event, urls):
    """Keep this event's successful tool URLs, never persist them across turns."""
    if not event.get_extra("chat_natural_enabled"):
        return
    saved = list(event.get_extra("chat_natural_evidence_urls") or [])
    for url in urls:
        try:
            parsed = urlsplit(url)
            valid = parsed.scheme in ("http", "https") and parsed.hostname and not parsed.username and not parsed.password
        except (ValueError, TypeError):
            valid = False
        if valid and url not in saved:
            saved.append(url)
    event.set_extra("chat_natural_evidence_urls", saved[:5])


class SearxngSearchPlugin(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        cfg = config or {}
        self.base_url = str(cfg.get("searxng_url") or "http://127.0.0.1:8888").rstrip("/")
        self.use_searcher_pro = bool(cfg.get("use_searcher_pro", False))

    async def ranked_results(self, query, limit):
        """Optional Pro ranking, returning only evidence from actual results."""
        try:
            from data.plugins.astrbot_plugin_web_searcher_pro.services.http_client import HttpClient
            from data.plugins.astrbot_plugin_web_searcher_pro.services.searxng_service import SearxngService
            service = SearxngService(HttpClient(timeout=12), self.base_url)
            results = await service.search(query, category='general', limit=limit)
            return [{'title': r.title.strip(), 'url': r.url.strip(), 'snippet': r.content.strip()[:220]}
                    for r in results if r.url]
        except Exception as exc:
            logger.warning('[SearXNG] Pro ranking unavailable (%s); using original search.', type(exc).__name__)
            return []

    async def terminate(self):
        pass

    @filter.llm_tool(name="web_search")
    async def web_search(
        self,
        event: AstrMessageEvent,
        query: str,
        max_results: int = 3,
    ) -> str:
        """联网搜索最新信息。当用户询问时事、新闻、天气、赛事、价格、软件版本等需要最新/实时信息的话题，或你不确定答案时效性时调用

        Args:
            query(string): 搜索关键词，简洁明确，一次一个主题
            max_results(number): 返回结果条数，默认3，最多5
        """
        max_results = max(1, min(int(max_results or 3), 5))
        if self.use_searcher_pro:
            ranked = await self.ranked_results(query, max_results)
            if ranked:
                record_evidence(event, [item['url'] for item in ranked])
                return json.dumps({'source_type':'web_search', 'query':query, 'results':ranked}, ensure_ascii=False)
        try:
            async with httpx.AsyncClient(timeout=12) as client:
                resp = await client.get(
                    f"{self.base_url}/search",
                    params={"q": query, "format": "json", "language": "zh-CN"},
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.error(f"[SearXNG] 搜索失败: {e}")
            return f"搜索失败（{type(e).__name__}），没有取得可核对的结果。"

        results = data.get("results", [])[:max_results]
        if not results:
            return "没有搜索到相关结果，可以换个关键词再试"

        evidence = []
        for i, r in enumerate(results, 1):
            title = (r.get("title") or "").strip()
            url = (r.get("url") or "").strip()
            snippet = (r.get("content") or "").strip()[:220]
            evidence.append({"title": title, "url": url, "snippet": snippet})
        record_evidence(event, [item["url"] for item in evidence])
        return json.dumps({"source_type": "web_search", "query": query, "results": evidence}, ensure_ascii=False)

    @filter.llm_tool(name="fetch_web_page")
    async def fetch_web_page(
        self,
        event: AstrMessageEvent,
        url: str,
    ) -> str:
        """读取一个网页链接的正文内容。当用户发来网页链接想让你了解其内容，或搜索结果需要深入查看某条时调用。不适合B站视频链接

        Args:
            url(string): 要读取的网页URL，以 http:// 或 https:// 开头
        """
        url = (url or "").strip()
        if not re.match(r"^https?://", url):
            return "错误: url 必须以 http:// 或 https:// 开头"

        try:
            html = None
            # 先直连，失败则走本机 VPN 代理（智能模式：国内直连、国外走隧道）
            for proxy in (None, "http://127.0.0.1:15732"):
                try:
                    async with httpx.AsyncClient(
                        timeout=10,
                        follow_redirects=True,
                        headers={"User-Agent": _UA},
                        proxy=proxy,
                    ) as client:
                        resp = await client.get(url)
                        resp.raise_for_status()
                        html = resp.text
                        break
                except Exception as e:
                    last_err = e
                    continue
            if html is None:
                raise last_err
        except Exception as e:
            logger.error(f"[SearXNG] 网页读取失败 {url}: {e}")
            return f"网页读取失败（{type(e).__name__}），不能据此判断网页内容。"

        page = extract_page(html)
        if len(page["text"]) < 40:
            return "网页内容过少或主要为动态渲染，无法提取正文"
        record_evidence(event, [str(resp.url)])
        return json.dumps({"source_type": "web_page", "url": str(resp.url), **page}, ensure_ascii=False)
