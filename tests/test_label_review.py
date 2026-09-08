"""Review sources precede escaped suggestions and never imply admission."""
import tempfile
from pathlib import Path
import unittest

from proxybench.annotation.label_review import render
from proxybench.annotation.xml_packets import build
from test_training_preparation import PRIMARY, votes


class ReviewTests(unittest.TestCase):
    def test_sources_first_and_unstarted_target_remains_unresolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = build(PRIMARY, votes(), primary_id='p', votes_id='v', index=0, input_id='a')['bundle']
            target = dict(bundle=bundle, group_id='dev', attempt_directory=str(root/'unstarted'),
                          selection_reason='<script>bad()</script>')
            ledger = render(root/'review', [target], {'results':[dict(input_id='a',status='NOT_ATTEMPTED_STOP')]},
                            review_budget={'remaining_minutes':None})
            page = (root/'review/index.html').read_text()
            self.assertLess(page.index('<h3>Source</h3>'), page.index('<h3>Original Sol draft</h3>'))
            self.assertIn('&lt;script&gt;bad()',page)
            self.assertIn('class="suggestions" hidden',page)
            self.assertFalse(ledger[0]['training_admitted'])
            self.assertFalse(ledger[0]['generated'])
            self.assertTrue(ledger[0]['unresolved'])
            self.assertIsNone(ledger[0]['active_seconds'])


if __name__ == '__main__':
    unittest.main()
