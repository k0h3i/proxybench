"""Exercise training display counters, clocks, and publication without a GPU."""

import unittest

from proxybench.execution.training_display import TrainingDisplay
from proxybench.training.measurements import EVENT_SCHEMA


def event(kind='phase', **fields):
    return dict(schema_version=EVENT_SCHEMA, kind=kind, attempt_elapsed_seconds=0, **fields)


def initial(step=0, **fields):
    return event(phase='training', global_step=step, sample_position=step, planned_updates=660,
                 epoch_boundaries=[330, 660], batch_size=1, gradient_accumulation_steps=1,
                 effective_batch_size=1, **fields)


class TrainingDisplayTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.output = []
        self.display = TrainingDisplay(self.output.append, clock=lambda: self.now,
                                       prior_seconds=25, phase_used=20,
                                       total_seconds=1000, phase_seconds=500)

    def test_resume_restores_counters_before_first_progress(self):
        self.display.accept(initial(396, resume_eligible=True,
                                    last_checkpoint=dict(path='checkpoints/396', global_step=396)))
        text = ''.join(self.output)
        self.assertIn('396/660 updates (60%)', text)
        self.assertIn('Epoch 2/2', text)
        self.assertIn('66/330', text)
        self.assertNotIn('0/660', text)
        self.assertIn('Batch: 1 | Accumulation: 1 | Effective batch: 1', text)

    def test_failure_or_missing_publication_never_reports_success(self):
        for status, published in (('PROCESS_FAILED', True), ('PHASE_TIMEOUT', True), ('EXITED', False)):
            with self.subTest(status=status, published=published):
                output = []
                display = TrainingDisplay(output.append, clock=lambda: self.now)
                display.accept(initial(660))
                display.accept(event(phase='publishing final adapter'))
                display.accept(event('attempt', status='COMPLETE', result_published=published))
                display.finish(status)
                self.assertNotIn('Training complete.', ''.join(output))
        self.display.accept(event('attempt', status='FAILED'))
        self.assertIn('Training attempt failed.', self.output[-1])

    def test_stops_never_infer_resume_from_checkpoint(self):
        self.display.accept(initial(330, last_checkpoint=dict(path='checkpoint', global_step=330)))
        self.display.message('Time budget margin reached. Saving at a safe boundary.')
        self.assertIn('Time budget margin reached.', self.output[-1])
        self.display.accept(event('attempt', status='CLEAN_STOP'))
        self.display.finish('EXITED')
        self.assertIn('Resume eligible: unverified.', self.output[-1])
        self.assertNotIn('Training complete.', ''.join(self.output))

    def test_resume_acceptance_warning_is_visible_once_among_library_noise(self):
        warning = ('Runtime identity differs or is incomplete. '
                   'Exact CUDA continuation requires new acceptance.')
        wire = ('Library loading banner\n' + warning + '\n' + warning + '\n').encode()
        for offset in range(0, len(wire), 7):
            self.display.feed_stdout(wire[offset:offset + 7])
        text = ''.join(self.output)
        self.assertEqual(text.count(warning), 1)
        self.assertNotIn('Library loading banner', text)

    def test_invalid_optional_counters_cannot_break_display_or_replace_state(self):
        self.display.accept(initial(330))
        for fields in (dict(global_step=True), dict(sample_position=-1), dict(epoch_boundaries=None),
                       dict(global_step=1), dict(planned_updates=0)):
            self.display.accept(event('update', **fields))
            self.assertEqual(self.display.data['global_step'], 330)


if __name__ == '__main__':
    unittest.main()
