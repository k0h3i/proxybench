"""Wait for durable ownership before starting a supervised command."""
import json
import os
from pathlib import Path
import sys
import time


def main():
    owner, gate = map(Path, sys.argv[1:3])
    parent = os.getppid()
    with owner.open('x') as stream:
        json.dump({'pid': os.getpid(), 'parent': parent}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    deadline = time.monotonic() + 10
    while not gate.exists():
        if os.getppid() != parent or time.monotonic() >= deadline:
            return 1
        time.sleep(0.02)
    os.execvpe(sys.argv[3], sys.argv[3:], os.environ)


if __name__ == '__main__':
    raise SystemExit(main())
