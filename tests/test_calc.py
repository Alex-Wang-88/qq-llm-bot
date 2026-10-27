"""算术走本地求值：接受的形态、拒绝的形态、精度与注入事实。

背景（2026-10-27 实测）：4B 对 `23 × 7 - 18` 连错三次（64 / 81 / 169），9B 给 141。
所以纯算式不经模型，由这里的代码给出确定答案。
"""
import asyncio
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


calc = load("assistant_calc_under_test", "assistant_calc.py")
natural = load("natural_plugin_calc", "data/plugins/astrbot_plugin_chat_natural/main.py")


class SolveTests(unittest.TestCase):
    def test_plain_expression_answers_with_the_number_only(self):
        self.assertEqual(calc.solve("23 × 7 - 18"), "143")

    def test_polite_wrapper_echoes_the_expression(self):
        self.assertEqual(calc.solve("只算结果：23 × 7 - 18。"), "23 × 7 - 18 = 143")

    def test_wrapped_question_answers(self):
        self.assertEqual(calc.solve("帮我算一下 1280/4 是多少"), "1280/4 = 320")

    def test_trailing_question_mark_is_not_a_wrapper(self):
        self.assertEqual(calc.solve("23*7-18?"), "143")

    def test_fullwidth_digits_and_operators(self):
        self.assertEqual(calc.solve("２３×７－１８"), "143")

    def test_thousands_separator_is_accepted(self):
        self.assertEqual(calc.solve("1,234 + 1"), "1235")

    def test_parentheses(self):
        self.assertEqual(calc.solve("2 * (3 + 4)"), "14")

    def test_floor_division_and_modulo(self):
        self.assertEqual(calc.solve("7 // 2"), "3")
        self.assertEqual(calc.solve("5 % 3"), "2")


class PrecisionTests(unittest.TestCase):
    def test_decimal_addition_has_no_binary_float_tail(self):
        self.assertEqual(calc.solve("0.1 + 0.2"), "0.3")

    def test_terminating_decimal(self):
        self.assertEqual(calc.solve("10/4"), "2.5")
        self.assertEqual(calc.solve("100/8"), "12.5")

    def test_half_is_printed_as_a_decimal(self):
        self.assertEqual(calc.solve("1/3 + 1/6"), "0.5")

    def test_repeating_decimal_falls_back_to_a_fraction(self):
        self.assertEqual(calc.solve("1/3"), "1/3")
        self.assertEqual(calc.solve("2/3"), "2/3")

    def test_result_that_is_whole_stays_an_integer(self):
        self.assertEqual(calc.solve("0.5 + 0.5"), "1")

    def test_negative_result(self):
        self.assertEqual(calc.solve("3 - 10"), "-7")
        self.assertEqual(calc.solve("1/2 - 1"), "-0.5")

    def test_caret_is_power_not_xor(self):
        self.assertEqual(calc.solve("2^8"), "256")
        self.assertEqual(calc.solve("2 ** 10"), "1024")


class RefusalTests(unittest.TestCase):
    def test_huge_exponent_is_refused(self):
        self.assertIsNone(calc.solve("2**999999"))
        self.assertIsNone(calc.solve("9**9**9"))

    def test_code_injection_is_refused(self):
        self.assertIsNone(calc.solve('__import__("os").system("dir")'))
        self.assertIsNone(calc.solve("open('x')"))

    def test_division_by_zero_is_refused_not_raised(self):
        self.assertIsNone(calc.solve("1/0"))
        self.assertIsNone(calc.solve("1 % 0"))

    def test_chinese_sentence_is_not_an_expression(self):
        self.assertIsNone(calc.solve("我买了三箱，每箱 12 个"))
        self.assertIsNone(calc.solve("三箱每箱 12 个，吃了 2 个还剩多少"))

    def test_comma_that_is_not_a_separator_is_refused(self):
        self.assertIsNone(calc.solve("1,234,5"))

    def test_full_date_is_not_arithmetic(self):
        self.assertIsNone(calc.solve("2026/10/27"))

    def test_bare_fraction_is_arithmetic(self):
        # 只有"年/月/日"三段式才算日期；`1/2` 是分数，不是 1 月 2 日。
        self.assertEqual(calc.solve("1/2"), "0.5")

    def test_bare_number_is_not_arithmetic(self):
        self.assertIsNone(calc.solve("2026"))
        self.assertIsNone(calc.solve("-5"))

    def test_overlong_text_is_refused(self):
        self.assertIsNone(calc.solve("1+" * 120 + "1"))

    def test_overlong_operand_is_refused(self):
        self.assertIsNone(calc.solve("99999999999999999999 + 1"))

    def test_empty_input(self):
        self.assertIsNone(calc.solve(""))
        self.assertIsNone(calc.solve("   "))
        self.assertIsNone(calc.solve(None))


class FactTests(unittest.TestCase):
    def test_embedded_expression_is_returned_as_a_verified_fact(self):
        self.assertEqual(calc.find_facts("帮我算一下 1280/4 是多少"), [("1280/4", "320")])

    def test_counting_sentence_yields_no_fact(self):
        self.assertEqual(calc.find_facts("我买了三箱，每箱 12 个"), [])

    def test_range_is_not_a_fact(self):
        self.assertEqual(calc.find_facts("第 3-5 章讲了什么"), [])

    def test_two_expressions_yield_two_facts(self):
        facts = calc.find_facts("3+4 和 10*2 哪个大")
        self.assertIn(("3+4", "7"), facts)
        self.assertIn(("10*2", "20"), facts)

    def test_facts_are_capped(self):
        self.assertLessEqual(len(calc.find_facts("1+1 2+2 3+3 4+4 5+5")), calc.MAX_FACTS)


class ModuleTests(unittest.TestCase):
    def test_evaluate_expression_returns_an_exact_value(self):
        value = calc.evaluate_expression("1/3")
        self.assertEqual(value.numerator, 1)
        self.assertEqual(value.denominator, 3)

    def test_format_value_prefers_integers_then_decimals_then_fractions(self):
        self.assertEqual(calc.format_value(calc.evaluate_expression("6/2")), "3")
        self.assertEqual(calc.format_value(calc.evaluate_expression("1/4")), "0.25")
        self.assertEqual(calc.format_value(calc.evaluate_expression("1/7")), "1/7")


class ShortcutTests(unittest.TestCase):
    """插件层：纯算式短路要真的发出去，句中算式只注入核验结果。"""

    def run_math(self, prompt, system_prompt=""):
        plugin = object.__new__(natural.NaturalChatPlugin)
        sent, stopped = [], []

        async def send(chain):
            sent.append("".join(getattr(part, "text", "") for part in chain.chain))

        event = SimpleNamespace(send=send, stop_event=lambda: stopped.append(True))
        req = SimpleNamespace(prompt=prompt, system_prompt=system_prompt)
        handled = asyncio.run(plugin._apply_math(event, req))
        return handled, sent, stopped, req

    def test_pure_expression_is_answered_without_the_model(self):
        handled, sent, stopped, req = self.run_math("23 × 7 - 18")
        self.assertTrue(handled)
        self.assertEqual(sent, ["143"])   # 消息已经自己发出去了
        self.assertEqual(stopped, [True])  # 事件停在这里，模型不会被调用
        self.assertEqual(req.system_prompt, "")

    def test_wrapped_expression_echoes_the_input(self):
        handled, sent, _, _ = self.run_math("帮我算一下 1280/4 是多少")
        self.assertTrue(handled)
        self.assertEqual(sent, ["1280/4 = 320"])

    def test_embedded_expression_is_injected_not_answered(self):
        handled, sent, stopped, req = self.run_math("帮我算一下 1280/4，顺便说说这个数字有什么特别")
        self.assertFalse(handled)
        self.assertEqual(sent, [])
        self.assertEqual(stopped, [])
        self.assertIn("1280/4 = 320", req.system_prompt)
        self.assertIn("核验", req.system_prompt)

    def test_plain_chat_is_left_alone(self):
        handled, sent, stopped, req = self.run_math("今天天气怎么样")
        self.assertFalse(handled)
        self.assertEqual(sent, [])
        self.assertEqual(stopped, [])
        self.assertEqual(req.system_prompt, "")

    def test_empty_prompt_is_left_alone(self):
        for prompt in ("", "   ", None):
            handled, sent, _, _ = self.run_math(prompt)
            self.assertFalse(handled)
            self.assertEqual(sent, [])


if __name__ == "__main__":
    unittest.main()
