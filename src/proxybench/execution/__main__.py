"""Run a declared development schedule with the local HTML parser."""

import argparse
import json
import os
from pathlib import Path
import sys

from proxybench.execution.runner import ScheduledInput, SubprocessAdapter, run_development


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('schedule', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--timeout-seconds', type=float, default=10)
    parser.add_argument('--output-bytes', type=int, default=1048576)
    args = parser.parse_args()
    schedule = [ScheduledInput(**item) for item in json.loads(args.schedule.read_text())]
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
    adapter = SubprocessAdapter([sys.executable, '-m', 'proxybench.extraction.html_table'],
                                cwd='/tmp', environment=environment)
    report = run_development(args.root, args.destination, schedule, adapter,
        run_id=args.run_id, system_id='conventional-html-row-v1', timeout_seconds=args.timeout_seconds,
        output_bytes=args.output_bytes)
    print(json.dumps({k: report[k] for k in ('run_id', 'scheduled', 'terminal_results', 'execution_eligible', 'primary_success')}))


if __name__ == '__main__':
    main()
