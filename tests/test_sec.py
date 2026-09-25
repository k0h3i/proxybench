from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proxybench.sources.sec import fetch, index_rows, validate_url, SecRedirects
from urllib.request import Request


class Response(BytesIO):
    def __init__(self, raw, status=200):
        super().__init__(raw)
        self.status = status
        self.headers = {'Content-Type': 'text/plain'}


class SecTests(unittest.TestCase):
    def test_cache_and_filing_limit(self):
        with tempfile.TemporaryDirectory() as tmp, patch('proxybench.sources.sec.time.sleep'):
            root = Path(tmp)
            with patch('proxybench.sources.sec.urlopen', return_value=Response(b'filing')) as call:
                first = fetch('https://www.sec.gov/a', root/'a.txt', root/'state.json', 'Declared client', max_filings=1)
                self.assertEqual(fetch('https://www.sec.gov/a', root/'a.txt', root/'state.json', 'Declared client', max_filings=1), first)
                self.assertEqual(call.call_count, 1)
            with self.assertRaises(RuntimeError):
                fetch('https://www.sec.gov/b', root/'b.txt', root/'state.json', 'Declared client', max_filings=1)

    def test_block_stops_requests_and_discards_failure_page(self):
        with tempfile.TemporaryDirectory() as tmp, patch('proxybench.sources.sec.time.sleep'):
            root = Path(tmp)
            with patch('proxybench.sources.sec.urlopen', return_value=Response(b'blocked', 403)) as call:
                for url in ['https://www.sec.gov/a', 'https://www.sec.gov/b']:
                    with self.assertRaises(RuntimeError):
                        fetch(url, root/'a.txt', root/'state.json', 'Private contact', kind='index')
                self.assertEqual(call.call_count, 1)
            state = (root/'state.json').read_text()
            self.assertNotIn('Private contact', state)
            self.assertTrue(json.loads(state)['blocked'])
            self.assertEqual(list(root.glob('*.partial-*')), [])

    def test_oversize_discards_partial_and_counts_excluded_filing(self):
        with tempfile.TemporaryDirectory() as tmp, patch('proxybench.sources.sec.time.sleep'):
            root = Path(tmp)
            with patch('proxybench.sources.sec.urlopen', return_value=Response(b'123456')):
                with self.assertRaises(RuntimeError):
                    fetch('https://www.sec.gov/a', root/'a.txt', root/'state.json', 'Declared client', max_document_bytes=3)
            state = json.loads((root/'state.json').read_text())
            self.assertEqual(state['filings'], 1)
            self.assertEqual(state['bytes'], 3)
            self.assertFalse((root/'a.txt').exists())

    def test_master_index_keeps_real_registrant_and_amendments(self):
        raw = b'CIK|Company Name|Form Type|Date Filed|Filename\n12|A Fund|N-PX|2007-08-01|edgar/data/12/0000999999-07-000001.txt\n12|A Fund|N-PX/A|2007-08-02|edgar/data/12/0000999999-07-000002.txt\n'
        rows = list(index_rows(raw))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['cik'], '12')
        self.assertEqual(rows[1]['form'], 'N-PX/A')

    def test_initial_urls_and_redirects_reject_identity_leaks(self):
        bad = ['http://www.sec.gov/a', 'https://evil.invalid/a',
               'https://private@www.sec.gov/a', 'https://www.sec.gov:8443/a',
               'https://www.sec.gov.evil.invalid/a']
        handler = SecRedirects()
        request = Request('https://www.sec.gov/a', headers={'User-Agent':'private identity'})
        for url in bad:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):validate_url(url)
                with self.assertRaises(ValueError):
                    handler.redirect_request(request, None, 302, 'Found', {}, url)
        with patch('proxybench.sources.sec.time.sleep') as wait:
            redirect=handler.redirect_request(request,None,302,'Found',{},'https://data.sec.gov/b')
            self.assertEqual(redirect.full_url,'https://data.sec.gov/b')
            wait.assert_called_once_with(0.5)

    def test_import_preserves_bytes_locations_and_holds_publication(self):
        from proxybench.sources.inventory import import_source
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'original.txt';source.write_bytes(b'Complete local source\n')
            options=dict(sec_url='https://www.sec.gov/a.txt',accession='0000000001-26-000001',
                         friendly_filename='cik-1-npx-0000000001-26-000001-a.txt',complete=True)
            first=import_source(source,root,**options)
            self.assertEqual((root/first['path']).read_bytes(),source.read_bytes())
            self.assertEqual(first['privacy_review']['status'],'held')
            second=import_source(source,root,**dict(options,sec_url='https://www.sec.gov/b.txt'))
            self.assertEqual(len(second['locations']),2)
            with self.assertRaises(ValueError):import_source(source,root,**dict(options,complete=False))
            for filename in ('../escape','CON','name.'):
                with self.assertRaises(ValueError):import_source(source,root,**dict(options,friendly_filename=filename))
            source.write_bytes(b'<SEC-DOCUMENT>incomplete')
            with self.assertRaises(ValueError):import_source(source,root,**options)
