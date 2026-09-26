"""Exercise training display counters, clocks, and publication without a GPU."""

import unittest

from proxybench.execution.training_display import TrainingDisplay
from proxybench.training.measurements import EVENT_SCHEMA, event_line


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

    def test_initial_and_epoch_boundaries_use_samples(self):
        self.display.accept(initial())
        self.assertIn('0/660 (0%)', self.display.progress())
        self.assertIn('Epoch 1/2: 0% complete (0/330 examples)', self.display.progress())
        for step, epoch, examples, fraction in ((329, 1, 329, '1.00'), (330, 1, 330, '1.00'),
                                               (331, 2, 1, '1.00'), (659, 2, 329, '2.00'),
                                               (660, 2, 330, '2.00')):
            with self.subTest(step=step):
                self.display.accept(event('update', global_step=step, sample_position=step))
                self.assertIn(f'Epoch {epoch}/2:', self.display.progress())
                self.assertIn(f'({examples}/330 examples)', self.display.progress())
                self.assertIn(f'fractional epoch {fraction}', self.display.progress())
        self.assertNotIn('Training complete.', ''.join(self.output))

    def test_samples_can_differ_from_optimizer_updates(self):
        self.display.accept(event(phase='training', global_step=2, planned_updates=4,
                                  sample_position=7, epoch_boundaries=[5, 9]))
        self.assertIn('2/4 (50%)', self.display.progress())
        self.assertIn('Epoch 2/2: 50% complete (2/4 examples)', self.display.progress())

    def test_resume_restores_counters_before_first_progress_and_restarts_clock(self):
        self.display.accept(initial(396, resume_eligible=True,
                                    last_checkpoint=dict(path='checkpoints/396', global_step=396)))
        text = ''.join(self.output)
        self.assertIn('396/660 (60%)', text)
        self.assertIn('Epoch 2/2: 20% complete (66/330 examples)', text)
        self.assertNotIn('0/660', text)
        self.assertIn('fractional epoch 1.20', text)
        self.assertIn('attempt 0s | charged 25s | budget left 480s', text)
        self.assertIn('Batch: 1 | Accumulation: 1 | Effective batch: 1', text)
        self.assertIn('checkpoints/396 at update 396', text)
        self.assertIn('Resume eligible: yes (validated clean stop)', text)
        self.display.accept(event(phase='training', resume_eligible=False))
        self.assertIn('Resume eligible: no', self.display.details())

    def test_missing_measurements_never_invent_zero_or_eta(self):
        self.display.accept(initial())
        text = self.display.details()
        self.assertIn('loop left estimating (final publication excluded)', text)
        self.assertIn('loss unavailable', text)
        self.assertIn('allocated unavailable GiB', text)
        self.assertIn('Resume eligible: unverified', text)
        self.display.accept(event('update', global_step=10, sample_position=10,
                                  remaining_loop_seconds=312, loss=.7, weighted_loss_12=.8,
                                  learning_rate=.0001, gradient_norm=2,
                                  throughput=dict(updates=10, seconds=20,
                                                  timing='full_loop_host_wall',
                                                  nonpadding_tokens_per_second=1500,
                                                  supervised_tokens_per_second=500)))
        self.display.accept(event('memory', allocated_bytes=1024**3,
                                  peak_allocated_bytes=2*1024**3, peak_reserved_bytes=3*1024**3,
                                  free_device_bytes=4*1024**3))
        text = self.display.details()
        self.assertIn('loop left ~312s (final publication excluded)', text)
        self.assertIn('loss last 12 updates, response-token-weighted 0.8', text)
        self.assertIn('gradient norm before clipping 2', text)
        self.assertIn('non-padding tokens/s 1500 | response tokens/s 500', text)
        self.assertIn('full-loop window 10 updates, 20s', text)
        self.assertIn('peak allocated 2.00 GiB | peak reserved 3.00 GiB | device free 4.00 GiB', text)

    def test_redirected_summaries_only_every_ten_updates_and_epoch_boundaries(self):
        self.display.accept(initial())
        for step in range(1, 21):
            self.display.accept(event('update', global_step=step, sample_position=step))
        self.assertEqual(len(self.output), 3)
        self.assertIn('10/660', self.output[1])
        self.assertIn('20/660', self.output[2])
        other = TrainingDisplay(self.output.append, clock=lambda: self.now)
        other.accept(event(phase='training', global_step=0, planned_updates=6,
                           sample_position=0, epoch_boundaries=[3, 6]))
        self.output.clear()
        for step in range(1, 7):
            other.accept(event('update', global_step=step, sample_position=step))
        self.assertEqual(len(self.output), 2)
        self.assertIn('(3/3 examples)', self.output[0])
        self.assertNotIn('\x1b', ''.join(self.output))

    def test_tty_refresh_at_most_once_per_second_and_phases_are_immediate(self):
        self.display.tty = True
        self.display.accept(initial())
        for offset in (.1, .5, .99):
            self.now = 100 + offset
            self.display.accept(event('update', global_step=1, sample_position=1))
            self.display.tick()
        self.assertEqual(len(self.output), 1)
        self.now = 101
        self.display.tick()
        self.assertEqual(len(self.output), 2)
        self.assertIn('[', self.output[-1])
        self.now = 101.01
        self.display.accept(event(phase='saving checkpoint'))
        self.assertEqual(len(self.output), 3)
        self.assertIn('Phase: saving checkpoint', self.output[-1])
        self.now = 111
        self.display.tick()
        self.assertIn('Phase: saving checkpoint', self.output[-1])
        self.assertIn('attempt 11s | charged 36s | budget left 469s', self.output[-1])

    def test_publication_and_exit_both_required_for_success(self):
        self.display.accept(initial(660))
        for phase in ('saving checkpoint', 'validating final adapter', 'publishing final adapter'):
            self.display.accept(event(phase=phase))
            self.assertIn('Phase: ' + phase, self.output[-1])
            self.assertNotIn('Training complete.', ''.join(self.output))
        self.display.accept(event('attempt', status='COMPLETE', result_published=True))
        self.assertNotIn('Training complete.', ''.join(self.output))
        self.display.finish('EXITED')
        self.assertIn('Training complete.', self.output[-1])

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

    def test_chunked_events_and_legacy_progress_are_not_forwarded(self):
        wire = (event_line(initial(330)) + '\nPhase: training\nstep 330/660 | loss 1\nwarning\n').encode()
        for byte in wire:
            self.display.feed_stdout(bytes([byte]))
        self.display.feed_stdout(b'partial', final=True)
        text = ''.join(self.output)
        self.assertIn('330/660', text)
        self.assertIn('warning\npartial\n', text)
        self.assertNotIn('PROXYBENCH_EVENT', text)
        self.assertNotIn('step 330', text)
        self.assertEqual(text.count('Phase: training'), 1)

    def test_invalid_optional_counters_cannot_break_display_or_replace_state(self):
        self.display.accept(initial(330))
        for fields in (dict(global_step=True), dict(sample_position=-1), dict(epoch_boundaries=None),
                       dict(global_step=1), dict(planned_updates=0)):
            self.display.accept(event('update', **fields))
            self.assertEqual(self.display.data['global_step'], 330)

    def test_malformed_optional_events_do_not_interrupt_stdout_processing(self):
        self.display.feed_stdout((event_line(initial()) + '\n').encode())
        for invalid_phase in ([], {}, None, 1):
            self.display.feed_stdout((event_line(event(phase=invalid_phase)) + '\n').encode())
            self.assertEqual(self.display.phase, 'training')
        huge = 10**500
        malformed = event('update', global_step=10, sample_position=10, loss=huge,
                          weighted_loss_12=[], learning_rate={}, gradient_norm='invalid',
                          remaining_loop_seconds=huge,
                          throughput=dict(timing=[], updates=huge, seconds=huge,
                                          nonpadding_tokens_per_second=huge,
                                          supervised_tokens_per_second=[]))
        self.display.feed_stdout((event_line(malformed) + '\n').encode())
        self.display.feed_stdout((event_line(event('memory', allocated_bytes=huge,
                                                   peak_allocated_bytes=[], peak_reserved_bytes={},
                                                   free_device_bytes='invalid')) + '\n').encode())
        self.assertIn('loss unavailable', self.display.details())
        self.assertIn('allocated unavailable GiB', self.display.details())
        self.assertIn('loop left estimating', self.display.details())
        self.display.feed_stdout((event_line(event('attempt', status='COMPLETE', result_published=True))
                                  + '\n').encode())
        self.display.finish('EXITED')
        self.assertIn('Training complete.', self.output[-1])

    def test_throughput_window_label_matches_its_measured_interval(self):
        self.display.accept(initial())
        for timing, expected in (
                ('completed_update_work_excluding_event_emission',
                 'update work window 10 updates, 20s (event publication excluded)'),
                ('full_loop_host_wall', 'full-loop window 10 updates, 20s'),
                (None, 'measurement window 10 updates, 20s')):
            with self.subTest(timing=timing):
                self.display.accept(event('update', global_step=10, sample_position=10,
                                          throughput=dict(timing=timing, updates=10, seconds=20)))
                self.assertIn(expected, self.display.details())
                if timing != 'full_loop_host_wall':
                    self.assertNotIn('full-loop window', self.display.details())


if __name__ == '__main__':
    unittest.main()
