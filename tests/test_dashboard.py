import http.client
import io
import threading
import unittest
from unittest import mock

from dashboard import server


class DashboardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http = server.ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        cls.port = cls.http.server_address[1]
        cls.thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.http.server_close()
        cls.thread.join()

    def request(self, method, path, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.port)
        headers = {'Host': f'127.0.0.1:{server.PORT}', **(headers or {})}
        conn.request(method, path, headers=headers)
        response = conn.getresponse()
        result = response.status, response.read()
        conn.close()
        return result

    def test_serves_page_and_status(self):
        self.assertEqual(self.request('GET', '/')[0], 200)
        self.assertEqual(self.request('GET', '/api/status')[0], 200)
        self.assertEqual(self.request('GET', '/api/log?name=../../secret')[0], 404)

    def test_rejects_cross_origin_and_untrusted_host(self):
        headers = {'Origin': 'http://evil.example', 'X-Control-Token': server.TOKEN}
        self.assertEqual(self.request('POST', '/api/stop', headers)[0], 403)
        self.assertEqual(self.request('GET', '/api/status', {'Host': 'evil.example'})[0], 403)
        self.assertEqual(self.request('POST', '/api/start', {'Origin': f'http://127.0.0.1:{server.PORT}'})[0], 403)

    @mock.patch('dashboard.server.subprocess.run')
    def test_stop_uses_existing_controlled_stop(self, run):
        run.return_value.returncode = 0
        run.return_value.stdout = b'ok'
        server.run_action('stop')
        self.assertEqual(run.call_args.args[0][-2:], ['-Restart', '-StopOnly'])
        self.assertEqual(run.call_args.kwargs['stdout'], server.subprocess.DEVNULL)
        self.assertIn('操作完成', server.operation['message'])

    @mock.patch('dashboard.server.subprocess.run')
    def test_gpu_metrics_and_unavailable(self, run):
        run.return_value.stdout = 'Example GPU, 1024, 8192\n'
        self.assertEqual(server.gpu_status(), {'name': 'Example GPU', 'used': 1024, 'total': 8192})
        run.side_effect = FileNotFoundError()
        self.assertIsNone(server.gpu_status())

    def test_estimates_only_completed_real_launches(self):
        log = '''[23:59:50] start root=X restart=False stoponly=False
[23:59:55] astrbot pid 12
[00:00:20] exit 0
[00:05:00] start root=X restart=False stoponly=False
[00:05:02] AstrBot is already listening; no duplicate process started.
[00:05:03] exit 0
[00:06:00] start root=X restart=False stoponly=False
[00:06:01] astrbot pid 13
[00:06:02] FAILED: example
[00:06:03] exit 0
[00:07:00] start root=X restart=True stoponly=True
[00:07:17] Known local bot services stopped.
'''
        self.assertEqual(server.estimate_duration('start', log), 30)
        self.assertEqual(server.estimate_duration('stop', log), 17)
        self.assertEqual(server.estimate_duration('start', ''), 35)

    def test_logs_strip_terminal_codes_and_format_tool_json(self):
        text = '\x1b[32m[16:10:40.465]\x1b[0m Tool `web_search` Result: {"results":[{"title":"测试"}]}\n\x1b[1m[16:10:42.332]\x1b[0m 回复'
        result = server.format_log(text)
        self.assertNotIn('\x1b', result)
        self.assertIn('"results": [\n', result)
        self.assertIn('\n\n[16:10:42.332]', result)

    @mock.patch('dashboard.server.build_opener')
    def test_loaded_models_distinguishes_empty_and_unavailable(self, opener):
        response = io.BytesIO(b'{"models": []}')
        opener.return_value.open.return_value = response
        self.assertEqual(server.loaded_models(), [])
        opener.return_value.open.side_effect = OSError('offline')
        self.assertIsNone(server.loaded_models())


if __name__ == '__main__':
    unittest.main()
