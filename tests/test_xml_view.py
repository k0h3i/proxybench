"""Readable source views preserve source text without substituting model labels."""
import unittest
from proxybench.annotation.xml_view import passage
from test_local_inputs import PRIMARY, ROW, votes


class SourceViewTests(unittest.TestCase):
    def test_source_passage_precedes_context_and_xml_is_collapsed(self):
        page = passage(PRIMARY, votes(ROW + ROW.replace('Élan', 'Neighbor')), index=0)
        visible = page[:page.index('<details><summary>Original XML')]
        self.assertIn('Approve A &amp; B &lt;C&gt;', visible)
        self.assertIn('Fonds Élan &amp; Co', visible)
        self.assertIn('001234567', visible)
        self.assertIn('ABSTAIN', visible)
        self.assertIn('AGAINST', visible)
        self.assertNotIn('Neighbor', page)
        self.assertNotIn('&lt;proxyTable', visible)
        self.assertLess(page.index('Proposal passage'), page.index('Reporting fund context'))
        self.assertIn('Full primary attachment as readable fields', page)

    def test_source_markup_is_escaped(self):
        row = ROW.replace('Approve A &amp; B &lt;C&gt;', '&lt;script&gt;alert(1)&lt;/script&gt;')
        page = passage(PRIMARY, votes(row), index=0)
        self.assertNotIn('<script>', page)
        self.assertIn('&lt;script&gt;alert(1)&lt;/script&gt;', page)
