"""Sources, review acceptance, and self-contained datasets share one contract."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from proxybench.annotation.bindings import review_binding
from proxybench.annotation.historical import prepare_historical, source_block, literal_support
from proxybench.annotation.preparation import prepare_review, accept_review
from proxybench.training.dataset import read_release, check_assignments, preserve_exposure
from proxybench.training.labels import TYPES, to_review, dumps, sha


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'configs').mkdir()
        (self.root/'docs').mkdir()
        (self.root/'configs/model-system-prompt.txt').write_bytes(b'Exact policy\n')
        (self.root/'docs/label-contract.md').write_text('Label contract')
        self.raw = '<p>Fund é</p><p>Omitted heading</p><tr><td>01</td><td>A &amp; <b>B</b></td><td></td><td>For</td></tr>'.encode()
        (self.root/'source.htm').write_bytes(self.raw)
        start = self.raw.index(b'<tr>')
        self.selection = dict(packet_id='one', accession='filing', source_path='source.htm',
            source_sha256=sha(self.raw), source_url='https://www.sec.gov/source.htm', encoding='utf-8',
            split='development', group_id='family', spans=[[0,self.raw.index(b'</p>')+4,'block'],
            [start,len(self.raw),'row']],target=[start,len(self.raw)],
            boundary_review='One separately voted subject',reviewer='user')
        self.assignments = {'filing': dict(split='development',group_id='family',development_exposed=True)}
        self.packet = prepare_historical(self.root,self.selection)
        self.label = {'fields': {key:dict(value=None,availability='ABSENT_IN_CONTEXT',origin=None,raw_text=None) for key in TYPES}}

    def accepted(self):
        prepare_review(self.root,[self.selection],self.root/'review',assignments=self.assignments)
        review = dict(schema='training-review-v1',packets={'one':dict(reviewed=True,
            input_binding=review_binding(self.packet),fields=to_review(self.label))})
        path=self.root/'review.json';path.write_text(dumps(review))
        approval=dict(decision='ACCEPTED',export_sha256=sha(path.read_bytes()),reviewer='user',
            accepted_at='2026-09-25',accepted_packet_ids=['one'])
        receipt=self.root/'approval.json';receipt.write_text(dumps(approval))
        return path,receipt

    def publish(self,path,receipt):
        return accept_review(self.root,self.root/'review',path,receipt,self.root/'dataset',assignments=self.assignments)

    def test_exact_three_roles_and_independent_source_reconstruction(self):
        self.publish(*self.accepted())
        page=(self.root/'review/index.html').read_text()
        self.assertIn('sandbox=""',page)
        self.assertIn('default-src',page)
        rows,_=read_release(self.root/'dataset',project_root=self.root)
        self.assertEqual([m['role'] for m in rows['development'][0]['messages']],['system','user','assistant'])
        self.assertEqual(rows['development'][0]['messages'][1]['content'],self.packet['model_input'])
        self.assertEqual(set(p.name for p in (self.root/'dataset').iterdir()),
                         {'dataset-manifest.json','training-examples.jsonl','development-examples.jsonl'})
        (self.root/'source.htm').write_bytes(self.raw+b'changed')
        with self.assertRaisesRegex(ValueError,'hash changed'):
            read_release(self.root/'dataset',project_root=self.root)

    def test_acceptance_requires_exact_completed_review(self):
        path,receipt=self.accepted()
        path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'Explicit acceptance'):
            self.publish(path,receipt)
        approval=json.loads(receipt.read_text());approval['export_sha256']=sha(path.read_bytes())
        receipt.write_text(dumps(approval))
        review=json.loads(path.read_text());review['packets']['one']['reviewed']=False
        path.write_text(dumps(review));approval['export_sha256']=sha(path.read_bytes());receipt.write_text(dumps(approval))
        with self.assertRaisesRegex(ValueError,'completed review'):
            self.publish(path,receipt)

    def test_exact_prompt_and_row_hashes_are_enforced(self):
        self.publish(*self.accepted())
        prompt=self.root/'configs/model-system-prompt.txt';prompt.write_text('Changed')
        with self.assertRaises(ValueError):read_release(self.root/'dataset',project_root=self.root)
        prompt.write_bytes(b'Exact policy\n')
        rows=self.root/'dataset/development-examples.jsonl';rows.write_bytes(rows.read_bytes()+b'\n')
        with self.assertRaises(ValueError):read_release(self.root/'dataset',project_root=self.root)

    def test_html_cells_gaps_and_source_bytes(self):
        cells=self.packet['manifest']['blocks'][1]['cells']
        self.assertEqual([c['text'] for c in cells],['01','A & B','','For'])
        self.assertEqual(self.raw[cells[1]['start_byte']:cells[1]['end_byte']],b'<td>A &amp; <b>B</b></td>')
        self.assertNotIn('Omitted heading',self.packet['model_input'])
        self.assertIn('OMITTED SOURCE BYTES',self.packet['model_input'])
        with self.assertRaises(ValueError):literal_support({'raw_text':'A & B For'},self.packet)
        with self.assertRaises(ValueError):source_block(b'<tr><td><script>x</script></td></tr>',0,34,'row','utf-8')

    def test_shared_row_targets_and_path_escape_are_rejected(self):
        selection=deepcopy(self.selection);selection['target'][0]+=4
        with self.assertRaises(ValueError):prepare_historical(self.root,selection)
        for path in ('../source.htm',str(self.root/'source.htm')):
            with self.assertRaisesRegex(ValueError,'project-relative'):
                prepare_historical(self.root,dict(self.selection,source_path=path))
        (self.root/'escape').symlink_to(self.root.parent,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'leaves the workspace'):
            prepare_historical(self.root,dict(self.selection,source_path='escape/other'))
        raw=b'<td>Context</td>'
        (self.root/'source.htm').write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'context'):
            prepare_historical(self.root,dict(self.selection,source_sha256=sha(raw),spans=[[0,len(raw),'cell']],target=[0,len(raw)]))

    def test_unselected_exposure_and_overlaps_block_training(self):
        ledger={'unused':dict(split='excluded',group_id='family',development_exposed=True),
                'new':dict(split='training',group_id='family',development_exposed=False)}
        with self.assertRaisesRegex(ValueError,'Development-exposed'):check_assignments([],ledger)
        with self.assertRaisesRegex(ValueError,'overlapping'):
            check_assignments([self.selection,self.selection],self.assignments)
        folder=self.root/'data/training-dataset';folder.mkdir(parents=True)
        (folder/'dataset-manifest.json').write_text(dumps({'assignments':self.assignments}))
        with self.assertRaises(ValueError):
            preserve_exposure(self.root,{'filing':dict(split='training',group_id='family',development_exposed=False)})
        with self.assertRaises(ValueError):
            preserve_exposure(self.root,{'filing':dict(split='training',group_id='new-group',development_exposed=False)})
