"""Preservation, ownership, and acceptance boundaries for optimization."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from proxybench.execution.resources import durable_json, reconcile, supervise
from proxybench.extraction.measured_generation import completion_status, timing_summary
from proxybench.training.optimization import admit


class OptimizationTests(unittest.TestCase):
    def test_budget_requires_allowance_phase_fit_and_reserved_completion(self):
        admit(1000, used_seconds=0, reserve_seconds=2100)
        for estimate, used, reserve in [(1441, 0, 0), (1000, 2200, 2100), (10, 5399, 0)]:
            with self.assertRaisesRegex(ValueError, 'admission'):
                admit(estimate, used_seconds=used, reserve_seconds=reserve)

    def test_completion_is_separate_from_parseable_text(self):
        common = dict(eos=9, forced=False, maximum=10, elapsed=2, deadline=60, stream_ended=True)
        self.assertEqual(completion_status([1, 9], **common), 'COMPLETE')
        self.assertEqual(completion_status([1], **common), 'LENGTH_STOP')
        self.assertEqual(completion_status([1, 9], **{**common, 'stream_ended': False}), 'CAPTURE_INCOMPLETE')
        self.assertEqual(completion_status([1, 9], **{**common, 'elapsed': 61}), 'TIMEOUT')
        self.assertEqual(completion_status([1, 9], **{**common, 'forced': True}), 'PROBE_INCOMPLETE')
        summary = timing_summary([{'seconds': 2}, {'seconds': 3}, {'seconds': 4}], 5)
        self.assertEqual(summary['request_tokens_per_second'], .6)
        self.assertEqual(summary['decode_tokens_per_second'], 1)

    def test_cpu_start_failure_and_descendant_cleanup(self):
        limits = dict(cpu_only=True, start_host_bytes=1, stop_host_bytes=1, phase_seconds=.3,
                      total_seconds=10, fixed_phase=True, sample_seconds=.05)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = [sys.executable, '-c', 'import subprocess,time; subprocess.Popen(["sleep","60"]); time.sleep(60)']
            with patch('proxybench.execution.resources.device_memory', side_effect=AssertionError('GPU queried')):
                status = supervise(command, root / 'run', limits, ledger=root / 'ledger.jsonl')
            self.assertEqual(status, 'PHASE_TIMEOUT')
            result = json.loads((root / 'run/result.json').read_text())
            self.assertEqual(result['surviving_owned_pids'], [])
            self.assertGreater(result['elapsed_seconds'], .3)
            self.assertFalse((root / 'ledger.active.json').exists())
            status = supervise(['/no/such/executable'], root / 'failed', limits, ledger=root / 'ledger.jsonl')
            self.assertEqual(status, 'PROCESS_FAILED')

    def test_stale_journal_blocks_launch_and_reconciles_without_double_charge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = root / 'ledger.jsonl'
            active = ledger.with_suffix('.active.json')
            record = dict(execution_id='one', supervisor_pid=999999999, run=str(root),
                          started_monotonic=time.monotonic()-5, started_wall=time.time()-5,
                          boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
            durable_json(active, record)
            with self.assertRaisesRegex(ValueError, 'reconciliation'):
                supervise([], root / 'refused', {}, ledger=ledger)
            used = reconcile(ledger)
            self.assertGreaterEqual(used, 5)
            durable_json(active, record)
            self.assertEqual(reconcile(ledger), used)
            self.assertEqual(len(ledger.read_text().splitlines()), 1)

    def test_request_deadline_retains_partial_tokens_as_failure(self):
        limits = dict(cpu_only=True, start_host_bytes=1, stop_host_bytes=1, phase_seconds=10,
                      total_seconds=20, fixed_phase=True, sample_seconds=.05)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / 'request.py'
            script.write_text('''import json,os,time
from pathlib import Path
from proxybench.execution.resources import durable_json
path = Path(__file__).with_name('answer.json')
durable_json(path, {'status':'STARTED'})
path.with_suffix('.tokens.jsonl').write_text('{"token_id":7,"seconds":0.01}\\n')
durable_json(os.environ['PROXYBENCH_REQUEST_FILE'], {'request_path':str(path), 'deadline_monotonic':time.monotonic()+.1})
time.sleep(60)
''')
            self.assertEqual(supervise([sys.executable, str(script)], root / 'run', limits,
                                       ledger=root / 'ledger.jsonl'), 'REQUEST_TIMEOUT')
            answer = json.loads((root / 'answer.json').read_text())
            self.assertEqual(answer['status'], 'INTERRUPTED')
            self.assertEqual(answer['token_ids'], [7])
            self.assertFalse(answer['format_valid'])

    def test_interrupted_checkpoint_and_corrupted_identity(self):
        try:
            import torch
            from peft import LoraConfig, get_peft_model
        except ImportError:
            self.skipTest('Requires the separate training environment')
        from proxybench.training.checkpoints import publish_adapter, validate_checkpoint
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
        class Tokenizer:
            def save_pretrained(self, directory):
                (Path(directory) / 'tokenizer_config.json').write_text('{}')
        model = get_peft_model(Tiny(), LoraConfig(r=2, target_modules=['linear']))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(model, 'save_pretrained', side_effect=RuntimeError('interrupted')):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    publish_adapter(model, Tokenizer(), root / 'failed')
            self.assertFalse((root / 'failed').exists())
            incomplete = next(root.glob('failed.incomplete-*'))
            with self.assertRaisesRegex(ValueError, 'publication is incomplete'):
                validate_checkpoint(incomplete)
            publish_adapter(model, Tokenizer(), root / 'good', {'model': 'tiny'})
            validate_checkpoint(root / 'good', {'model': 'tiny'})
            with self.assertRaisesRegex(ValueError, 'identity'):
                validate_checkpoint(root / 'good', {'model': 'wrong'})
            (root / 'good/adapter_config.json').write_text('corrupted')
            with self.assertRaisesRegex(ValueError, 'hash'):
                validate_checkpoint(root / 'good')
            with patch.object(Path, 'rename', side_effect=OSError('publication failed')):
                with self.assertRaisesRegex(OSError, 'publication failed'):
                    publish_adapter(model, Tokenizer(), root / 'unpublished')
            unpublished = next(root.glob('unpublished.incomplete-*'))
            self.assertTrue((unpublished / 'complete.json').exists())
            with self.assertRaisesRegex(ValueError, 'publication is incomplete'):
                validate_checkpoint(unpublished)

    def test_sampled_merge_preserves_unadapted_tensors(self):
        try:
            import torch
            from peft import LoraConfig, get_peft_model
        except ImportError:
            self.skipTest('Requires the separate training environment')
        from types import SimpleNamespace
        from transformers import PretrainedConfig
        from proxybench.training.optimization import merge_model
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4, bias=False)
                self.untouched = torch.nn.Linear(4, 4, bias=False)
                self.config = PretrainedConfig(tie_word_embeddings=False)
            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(logits=self.linear(torch.nn.functional.one_hot(input_ids, 4).float()))
        model = get_peft_model(Tiny(), LoraConfig(r=2, target_modules=['linear']))
        with torch.no_grad():
            model.get_base_model().linear.lora_B['default'].weight.fill_(.1)
        with tempfile.TemporaryDirectory() as directory:
            merge_model(model, [0,1,2], directory)
            record = json.loads((Path(directory) / 'merge.json').read_text())
            self.assertEqual(record['changed'], ['linear.weight'])
            self.assertLess(record['score_max_absolute_difference'], 1e-6)


if __name__ == '__main__':
    unittest.main()
