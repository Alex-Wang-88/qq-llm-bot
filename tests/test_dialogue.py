import unittest
from assistant_dialogue import classify,is_harness_dispatch


class IntentTests(unittest.TestCase):
    def test_casual_complaint_does_not_grant_computer_control(self):
        self.assertEqual(classify('我的 CPU 又跑满了，好烦').mode,'chat')
        self.assertEqual(classify('查看 CPU 和内存占用').mode,'computer')
    def test_queries_and_harness_requests_are_separate(self):
        self.assertEqual(classify('看看这个链接 https://example.com').mode,'query')
        self.assertEqual(classify('让 Codex 继续任务').mode,'harness')
        self.assertEqual(classify('上次那个任务进度如何').mode,'harness')
    def test_thinking_is_explicit_and_not_inherited_from_history(self):
        self.assertTrue(classify('请深度思考：这套设计哪里有问题').thinking)
        self.assertFalse(classify('刚才你在深度思考吗').thinking)

    def test_dispatch_requires_an_explicit_request(self):
        for text in ('让 Codex 继续任务','把“继续修复”交给 Codex。两个会话。','请 WorkBuddy 查看任务状态'):
            if text.startswith('请 WorkBuddy'):self.assertFalse(is_harness_dispatch(text))
            else:self.assertTrue(is_harness_dispatch(text))
        for text in ('Codex 的任务完成了吗','昨天让 Codex 修复了','https://example.com/让Codex继续','/助手 发送 让 Codex 继续'):
            self.assertFalse(is_harness_dispatch(text))
