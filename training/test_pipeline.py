"""Tests for supervision boundaries and dataset leakage, not imitation of training."""
import copy
import unittest
import json
import tempfile
from pathlib import Path
from common import LOCAL
from data_utils import encode,read_rows
from validate_data import validate
from evaluate import parse_calls,checks
from common import save_json


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer=AutoTokenizer.from_pretrained(LOCAL/'models/Qwen3.5-9B',local_files_only=True)
        cls.rows=read_rows(LOCAL/'data/experimental-v1/train.jsonl')

    def test_user_system_and_tool_results_are_not_targets(self):
        row=next(r for r in self.rows if r.get('tools'))
        item=encode(self.tokenizer,row)
        target=self.tokenizer.decode([t for t in item['labels'] if t!=-100])
        self.assertIn('<tool_call>',target)
        self.assertIn(row['messages'][-1]['content'],target)
        self.assertNotIn('<tool_response>',target)
        self.assertNotIn(row['messages'][0]['content'],target)
        self.assertNotIn(row['messages'][1]['content'],target)
        self.assertIn('<|im_end|>',target)

    def test_generation_input_is_tensor_not_batch_encoding(self):
        ids=self.tokenizer.apply_chat_template([{'role':'user','content':'你好'}],tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=False)
        self.assertEqual(len(ids.shape),2)

    def test_overlength_rejected_instead_of_cutting_tool_pairs(self):
        row=copy.deepcopy(self.rows[0]);row['messages'][1]['content']='长文本。'*3000
        with self.assertRaisesRegex(ValueError,'no silent truncation'):encode(self.tokenizer,row,512)

    def test_family_and_exact_leakage_are_rejected(self):
        row=copy.deepcopy(self.rows[0]);other=copy.deepcopy(row);other['id']='other'
        report=validate({'train':[row],'test':[other]})
        self.assertFalse(report['passed'])
        self.assertTrue(any('Family leakage' in e for e in report['errors']))
        self.assertTrue(any('Duplicate dialogue' in e for e in report['errors']))

    def test_orphan_tool_result_rejected(self):
        row=copy.deepcopy(next(r for r in self.rows if r.get('tools')))
        row['messages'][3]['tool_call_id']='wrong'
        self.assertFalse(validate({'train':[row]})['passed'])


class EvaluationTests(unittest.TestCase):
    def test_loader_diagnostic_sets_can_be_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'report.json'
            save_json(path,{'missing_keys':set(),'unexpected_keys':{'mtp.weight'}})
            self.assertEqual(json.loads(path.read_text()),{'missing_keys':[],'unexpected_keys':['mtp.weight']})
    def test_official_xml_function_arguments(self):
        text='<tool_call>\n<function=harness_query>\n<parameter=operation>\nquota\n</parameter>\n<parameter=harness>codex</parameter>\n</function>\n</tool_call>'
        self.assertEqual(parse_calls(text),[{'name':'harness_query','arguments':{'operation':'quota','harness':'codex'}}])
        self.assertFalse(checks(text,{})['tool_announcement'])
        self.assertTrue(checks('我去查。'+text,{})['tool_announcement'])

    def test_broken_tool_markup_does_not_pass(self):
        self.assertEqual(parse_calls('<tool_call>{"name":"fake"}</tool_call>'),[])


if __name__=='__main__':unittest.main()
