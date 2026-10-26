import unittest
import logging
import sys
from assistant_security import contains_secret,redact,redact_text,install_log_redaction


class PrivacyTests(unittest.TestCase):
    def test_structured_secrets_and_image_bytes_are_not_logged(self):
        value=redact({'api_key':'private','content':[{'image_url':{'url':'data:image/png;base64,AAAA'}},{'text':'密码是abc123'}]})
        self.assertEqual(value['api_key'],'[REDACTED]')
        self.assertNotIn('AAAA',str(value));self.assertNotIn('abc123',str(value))
    def test_tokens_cookies_and_private_keys_are_removed(self):
        fixtures=['sk-abcdefghijklmnopqrstuv','Bearer abcdefghijklmnop','Cookie: sid=private-session','-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----']
        for value in fixtures:
            self.assertTrue(contains_secret(value))
            self.assertEqual(redact_text(value),'[REDACTED]')
    def test_ordinary_facts_remain_usable(self):
        text='用户偏好无糖咖啡，正在做 REA 项目'
        self.assertFalse(contains_secret(text));self.assertEqual(redact_text(text),text)

    def test_logs_redact_format_args_and_exception_text(self):
        old=install_log_redaction()
        try:
            try:raise RuntimeError('password=private-value')
            except RuntimeError:exc=sys.exc_info()
            record=logging.getLogRecordFactory()('test',logging.ERROR,__file__,1,'Authorization: Bearer %s',('abcdefghijklmnop',),exc)
            output=logging.Formatter().format(record)
            self.assertNotIn('abcdefghijklmnop',output);self.assertNotIn('private-value',output)
        finally:logging.setLogRecordFactory(old)
