import unittest
from decimal import Decimal
from unittest.mock import patch

import assistant_market as market


def sample(symbol, stamp, price, change, today_open='', prev_close=''):
    fields = [''] * 33
    fields[3], fields[30], fields[32] = price, stamp, change
    fields[4], fields[5] = prev_close, today_open
    return 'v_' + symbol + '="' + '~'.join(fields) + '";'


class MarketTests(unittest.IsolatedAsyncioTestCase):
    def test_narrow_routing(self):
        for query in ('今天A股怎么样？', '查一下A股今天收盘情况', '现在大盘行情'):
            self.assertTrue(market.is_simple_market_query(query), query)
        for query in ('A股几点收盘', 'A股为什么下跌', 'A股今天怎么样，帮我推荐股票', '查询一下'):
            self.assertFalse(market.is_simple_market_query(query), query)

    def test_open_and_single_index_are_routed(self):
        # 用户原话；也是 web_search 拿旧数据那次的原句。
        for query in ('今天A股是高开还是低开', '今天A股是高开还是低开？',
                      '帮我看看创业板指', '上证指数现在多少', '今天大盘怎么走'):
            self.assertTrue(market.is_simple_market_query(query), query)
        for query in ('/大盘', '今天天气怎么样', '高开还是低开'):
            self.assertFalse(market.is_simple_market_query(query), query)

    def test_timestamp_and_bad_data(self):
        text = sample('sh000001', '20261027161400', '3648.30', '-1.33')
        result = market.parse_quotes(text)
        self.assertEqual(str(result['sh000001']['price']), '3648.30')
        self.assertEqual(result['sh000001']['time'].isoformat(), '2026-10-27T16:14:00')
        self.assertFalse(market.parse_quotes(sample('sh000001', 'bad', 'nan', '0')))

    def test_open_verdict_requires_both_sides(self):
        quote = {'open': Decimal('3698.44'), 'prev_close': Decimal('3697.52')}
        self.assertEqual(market.open_verdict(quote), '高开 0.92 点')
        self.assertEqual(market.open_verdict({'open': Decimal('3600'), 'prev_close': Decimal('3697.52')}), '低开 97.52 点')
        self.assertEqual(market.open_verdict({'open': Decimal('3697.52'), 'prev_close': Decimal('3697.52')}), '平开')
        # 缺字段或 0 只表示"判断不了"，不能当成平开。
        self.assertEqual(market.open_verdict({'open': None, 'prev_close': Decimal('3697.52')}), '')
        self.assertEqual(market.open_verdict({'open': Decimal('0'), 'prev_close': Decimal('3697.52')}), '')
        self.assertEqual(market.open_verdict({}), '')

    def test_followup_requires_last_assistant_market_context(self):
        self.assertTrue(market.is_market_followup('查询一下', [{'role': 'assistant', 'content': '我再查一下 A股 的行情。'}]))
        self.assertFalse(market.is_market_followup('查询一下', [{'role': 'assistant', 'content': '查一下天气。'}]))
        self.assertFalse(market.is_market_followup('帮我分析行情', [{'role': 'assistant', 'content': 'A股'}]))

    async def test_old_quotes_report_stale_and_partial(self):
        content = sample('sh000001', '20000101110000', '100.00', '1.23').encode()
        class Response:
            def raise_for_status(self): pass
        response = Response()
        response.content = content
        class Client:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def get(self, *args, **kwargs): return response
        with patch.object(market.httpx, 'AsyncClient', return_value=Client()):
            text = await market.market_snapshot()
        self.assertIn('2000-01-01 11:00:00', text)
        self.assertIn('不是今天的', text)
        self.assertIn('部分指数未取得', text)
        self.assertIn(market.QUOTE_URL, text)

    async def test_snapshot_states_open_verdict_when_data_allows(self):
        content = sample('sh000001', '20991231150000', '3648.30', '-1.33', '3698.44', '3697.52').encode()
        class Response:
            def raise_for_status(self): pass
        response = Response()
        response.content = content
        class Client:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def get(self, *args, **kwargs): return response
        with patch.object(market.httpx, 'AsyncClient', return_value=Client()):
            text = await market.market_snapshot()
        self.assertIn('今日开盘：上证指数 高开 0.92 点', text)
        self.assertIn('3648.30', text)

    async def test_missing_open_field_never_fabricates_verdict(self):
        content = sample('sh000001', '20991231150000', '3648.30', '-1.33').encode()
        class Response:
            def raise_for_status(self): pass
        response = Response()
        response.content = content
        class Client:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def get(self, *args, **kwargs): return response
        with patch.object(market.httpx, 'AsyncClient', return_value=Client()):
            text = await market.market_snapshot()
        self.assertNotIn('今日开盘', text)
        self.assertIn('3648.30', text)

    async def test_failure_never_invents_data(self):
        class Client:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def get(self, *args, **kwargs): raise market.httpx.ConnectError('offline')
        with patch.object(market.httpx, 'AsyncClient', return_value=Client()):
            text = await market.market_snapshot()
        self.assertIn('查询失败', text)
        self.assertIn('不能据此判断是否休市', text)
