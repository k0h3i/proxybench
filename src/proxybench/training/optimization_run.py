"""Launch one admitted optimization phase in the existing isolated environment."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from proxybench.execution.resources import durable_json, ledger_entries, supervise
from proxybench.training.optimization import admit
from proxybench.training.smoke import accepted_rows, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['cpu-check', 'train-save', 'reload-panel', 'diagnostics',
                                        'merged', 'export-merge', 'final-reference', 'final-candidate'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--estimate-seconds', type=float, required=True)
    parser.add_argument('--reserve-seconds', type=float, default=0)
    args = parser.parse_args()
    config_path = Path('configs/qwen35-4b-optimization.json')
    config = json.loads(config_path.read_text())
    root = args.run
    root.mkdir(parents=True, exist_ok=True)
    cpu = args.phase == 'cpu-check'
    limits = config['cpu_limits'] if cpu else config['limits']
    ledger = root / ('cpu-ledger.jsonl' if cpu else 'gpu-ledger.jsonl')
    used = sum(entry['elapsed_seconds'] for entry in ledger_entries(ledger))
    admission = admit(args.estimate_seconds, used_seconds=used, total_seconds=limits['total_seconds'],
                      phase_seconds=limits['phase_seconds'], reserve_seconds=args.reserve_seconds)
    setup = root / (args.phase + '-launch')
    setup.mkdir(exist_ok=False)
    durable_json(setup / 'admission.json', admission)
    shutil.copyfile(config_path, setup / config_path.name)
    sources = {}
    for prefix in ('training', 'extraction', 'execution'):
        for source in Path('src/proxybench', prefix).glob('*.py'):
            target = setup / prefix / source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, target)
            sources[str(source)] = digest(source)
    durable_json(setup / 'source-hashes.json', sources)
    toolchain = Path('artifacts/environments/qwen35-4b-toolchain/bin').resolve()
    os.environ['CC'] = str(toolchain / 'cc')
    os.environ['CXX'] = str(toolchain / 'c++')
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v']
    else:
        command = [sys.executable, '-m', 'proxybench.training.smoke', args.phase,
                   '--configuration', str(config_path),
                   '--data', 'data/annotations/direct-sol-15-v1/accepted-labels.jsonl',
                   '--model', 'artifacts/models/Qwen3.5-4B',
                   '--provenance', 'artifacts/runs/qwen35-4b-smoke-20260915/setup/model-provenance.json',
                   '--output', str(root / args.phase)]
        if args.phase != 'train-save':
            command.extend(['--adapter', str(root / 'train-save/adapter'), '--reference', str(root / 'train-save')])
    durable_json(setup / 'environment.json', dict(CC=os.environ['CC'], CXX=os.environ['CXX'],
                 PYTHONPATH=os.environ.get('PYTHONPATH'), executable=sys.executable))
    result = supervise(command, root / (args.phase + '-supervisor'), limits, ledger=ledger)
    print(result, flush=True)
    return 0 if result == 'EXITED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
