# Prepare the local environment

Use this guide before training, model evaluation, conversion, or inference.
Run the commands in Bash from the repository root.
A virtual environment keeps project Python packages separate.
Preparation installs dependencies and tests them without loading a model onto the GPU.
Complete preparation before starting GPU work.

## Required software and inputs

The model environment uses Python 3.12.14 on x64 Linux or WSL2.
The hardware target is an NVIDIA RTX 3090 with 24 GB of GPU memory.
The commands below use Ubuntu 22.04, 24.04, or 26.04.
Other platforms need a separately tested installation procedure.
Python headers let the compiler build Python extensions.

| Component | Required for | Installation or input |
|---|---|---|
| Git, curl, CA certificates | Repository and dependency downloads | Ubuntu packages below |
| Python 3.12.14 and pip | All model commands | uv installation below, then root `.venv/` |
| C/C++ compiler | Python GPU compilation | Ubuntu `build-essential` |
| Python 3.12 development headers | Building Python extensions | Matching Python installation, checked below |
| NVIDIA driver | All GPU commands | Linux or WSL driver instructions below |
| Python model packages, including Torch and CUDA 13 libraries | Training, adapter loading, conversion, and the shared model environment | [requirements-training.txt](../configs/requirements-training.txt) |
| llama.cpp executable and native libraries | GGUF inference, evaluation, and both-format acceptance | Pinned release installation below |
| CUDA 12 runtime and cuBLAS libraries | The pinned llama.cpp build | [requirements-llama-cuda.txt](../configs/requirements-llama-cuda.txt), installed separately below |
| OpenMP, OpenSSL, and C++ runtime libraries | Native llama.cpp loading | Ubuntu `libgomp1`, `libssl-dev`, and `build-essential` |
| Pinned base weights | Training, adapter loading, and export | Hugging Face download below |
| Pinned converter source | Export and evaluation of a new adapter | Authenticated source archive below |
| Selected adapter, tokenizer, and GGUF | Existing model loading and inference | Public adapter repositories below; private retained GGUF |
| Accepted dataset and original sources | Training, evaluation, and source reconstruction | Private retained data or the [dataset workflow](dataset.md) |

Python packages do not install the NVIDIA driver or the llama.cpp executable.
The Torch packages supply CUDA 13 libraries for Python.
The separate CUDA 12 installation supplies libraries for llama.cpp.

Reserve space for Python packages, downloads, the retained base, and model files.
The full Qwen3.5-4B checkpoint occupies about 8.7 GiB.
The ProxyType-4B GGUF occupies about 7.9 GiB.
Training and conversion also require at least 64 GiB free on the run filesystem.
The resource limits can reject work when available host or GPU memory is too low.

## Install host software and obtain the repository

Install the Ubuntu packages:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git curl ca-certificates libgomp1 libssl-dev
```

On native Ubuntu, install a supported NVIDIA driver through the [Ubuntu driver procedure](https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers/).
On WSL2, install the Windows NVIDIA driver through the [NVIDIA WSL procedure](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
Do not install a Linux display driver inside WSL.
The CUDA 13 Python stack requires driver branch 580 or later under [NVIDIA's compatibility rules](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
Use a current compatible driver because compiled GPU kernels can impose additional requirements.
Restart the host if the driver installation requests it.

```bash
nvidia-smi
```

On a Windows host with 32 GB of RAM, WSL2 normally exposes about half that memory to Linux.
For 9B training, start with a 24 GB limit and leave about 8 GB for Windows.
This is a starting allocation, not evidence that training fits.
The [Microsoft WSL configuration guide](https://learn.microsoft.com/en-us/windows/wsl/wsl-config) defines the file and restart procedure.

From Windows PowerShell, open `%UserProfile%\.wslconfig`:

```powershell
notepad "$env:USERPROFILE\.wslconfig"
```

If the file already contains `[wsl2]`, edit its existing memory values instead of adding another section.
Keep unrelated configuration values.
Otherwise, create this section:

```ini
[wsl2]
memory=24GB
swap=8GB
```

Swap uses disk space when RAM runs short.
It does not increase GPU memory or replace RAM for fast training.
Save the file as `.wslconfig`, not `.wslconfig.txt`.
After active work finishes, save your work and run this command in Windows PowerShell.
The command stops all WSL distributions and their running processes.

```powershell
wsl --shutdown
```

Reopen WSL and inspect the available memory:

```bash
free -h
```

If you need a repository checkout, clone the repository with your authorized GitHub account:

```bash
git clone https://github.com/k0h3i/proxybench.git
cd proxybench
```

## Install Python and project packages

If Python 3.12.14 is missing, install it with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/install.sh -o /tmp/proxybench-uv-install.sh
sh /tmp/proxybench-uv-install.sh
"$HOME/.local/bin/uv" python install 3.12.14
```

If uv installed Python, find its executable:

```bash
PROXYBENCH_PYTHON="$("$HOME/.local/bin/uv" python find --managed-python 3.12.14)"
```

If you supplied Python 3.12.14 another way, set its absolute executable path instead.
Replace the example path below with that executable:

```bash
PROXYBENCH_PYTHON=/absolute/path/to/python3.12
```

Both paths require `venv`, `ensurepip`, and matching Python development headers.
Make sure that the interpreter supplies them before creating `.venv/`:

```bash
"$PROXYBENCH_PYTHON" - <<'PY'
import ensurepip
from pathlib import Path
import sys
import sysconfig
import venv
assert sys.version_info[:3] == (3, 12, 14), 'Python 3.12.14 is required'
header = Path(sysconfig.get_path('include')) / 'Python.h'
assert header.is_file(), 'Install development headers for this Python interpreter'
print('Interpreter and Python headers are ready')
PY
```

Create `.venv/` with that interpreter using the standard `venv` module.
If `.venv/` already exists, inspect its Python version and ownership before changing it.
Use an interpreter outside disposable experiment directories.
The [uv Python guide](https://docs.astral.sh/uv/guides/install-python/) describes managed interpreter installation.

```bash
"$PROXYBENCH_PYTHON" -m venv .venv
.venv/bin/python --version
.venv/bin/python -m pip install -r configs/requirements-training-build.txt
.venv/bin/python -m pip install --no-deps --upgrade --target .venv/cuda-build -r configs/requirements-cuda-build.txt
ln -sfn libcudart.so.13 .venv/cuda-build/nvidia/cu13/lib/libcudart.so
CUDA_VISIBLE_DEVICES='' CUDA_HOME="$PWD/.venv/cuda-build/nvidia/cu13" \
  PATH="$PWD/.venv/bin:$PATH" MAX_JOBS=2 CAUSAL_CONV1D_FORCE_BUILD=TRUE \
  timeout 900 .venv/bin/python -m pip install --no-build-isolation -r configs/requirements-training.txt
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip check
```

Keep the recorded package versions.
Resolve a failed installation before continuing to model work.

The [build prerequisites](../configs/requirements-training-build.txt) use the exact versions in the training requirements.
They provide Torch and Ninja before the source build starts.
The [CUDA build requirements](../configs/requirements-cuda-build.txt) provide a complete CUDA 13.2 compiler toolchain in `.venv/cuda-build/`.
This separate directory keeps the compiler and headers compatible while the Torch runtime libraries retain their existing pins.
The symbolic link supplies the library filename required by the linker.

[`causal-conv1d` 1.7.0](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.7.0) has no official wheel for the pinned Torch 2.12 environment.
A wheel is a compiled Python package.
The command builds this package locally with two build jobs and a 15-minute limit.
`--no-build-isolation` uses the installed Torch version during compilation.
The build does not need access to a GPU.
The CUDA compiler and its headers remain inside `.venv/`.

For CPU source preparation alone, the package supports Python 3.11 or later without the model requirements file.

## Install llama.cpp and its CUDA libraries

The supported executable comes from the [Unsloth release assets](https://github.com/unslothai/llama.cpp/releases/expanded_assets/b10909-mix-bea84f7).
Use `app-b10909-mix-bea84f7-linux-x64-cuda12-older.tar.gz`.
Its build metadata records source revision `329b6160f513915f1c607dbfae3d5ce864a64a4f` and CUDA 12.8.
The commands test the archive checksum before extraction.
A checksum identifies the exact downloaded bytes.

For a new installation, choose a runtime destination that does not exist.
The versioned directory below is the default.

```bash
export PROXYBENCH_RUNTIME="$HOME/.local/share/proxybench/runtime/llama-329b6160"
export PROXYBENCH_CUDA_LIB="$PROXYBENCH_RUNTIME"
```

If reusing an existing installation, skip the download and installation commands in this section.
For an existing installation, continue with [path setup and model inputs](#set-paths-and-obtain-model-inputs).
The [CPU acceptance step](#complete-cpu-acceptance) checks the existing manifest and libraries.

For a new installation, download the pinned files:

```bash
export PROXYBENCH_SETUP_DIR="$(mktemp -d -t proxybench-setup-XXXXXXXX)"
export PROXYBENCH_LLAMA_ASSET=app-b10909-mix-bea84f7-linux-x64-cuda12-older.tar.gz
curl -fL --retry 3 \
  "https://github.com/unslothai/llama.cpp/releases/download/b10909-mix-bea84f7/$PROXYBENCH_LLAMA_ASSET" \
  -o "$PROXYBENCH_SETUP_DIR/$PROXYBENCH_LLAMA_ASSET"
.venv/bin/python -m pip install --no-deps --only-binary=:all: --require-hashes \
  --target "$PROXYBENCH_SETUP_DIR/cuda" -r configs/requirements-llama-cuda.txt
```

Install the server, required libraries, licenses, and build metadata:

```bash
.venv/bin/python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

stage = Path(os.environ['PROXYBENCH_SETUP_DIR'])
runtime = Path(os.environ['PROXYBENCH_RUNTIME'])
archive = stage / os.environ['PROXYBENCH_LLAMA_ASSET']
expected = '0c477165533c3371b57c2ea19a6c560011397bbaae7fe80b813de389b0671a66'
with archive.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
unpacked = stage / 'llama'
with tarfile.open(archive) as bundle:
    bundle.extractall(unpacked, filter='data')
info = json.loads((unpacked / 'UNSLOTH_PREBUILT_INFO.json').read_text())
assert info['source_commit'] == '329b6160f513915f1c607dbfae3d5ce864a64a4f'
runtime.mkdir(parents=True, exist_ok=False)
names = ['llama-server', 'libllama-server-impl.so', 'libggml-cuda.so',
         'libggml-rpc.so', 'libggml.so.0', 'libggml-base.so.0',
         'libllama.so.0', 'libllama-common.so.0', 'libmtmd.so.0',
         'BUILD_INFO.txt', 'THIRD_PARTY_LICENSES.txt', 'UNSLOTH_PREBUILT_INFO.json']
names += [path.name for path in unpacked.glob('libggml-cpu-*.so')]
for name in names:
    shutil.copy2(unpacked / name, runtime / name)
for name in ('libcublas.so.12', 'libcublasLt.so.12', 'libcudart.so.12'):
    matches = list((stage / 'cuda').rglob(name))
    assert len(matches) == 1, (name, matches)
    shutil.copy2(matches[0], runtime / name)
for path in (stage / 'cuda').rglob('License.txt'):
    shutil.copy2(path, runtime / (path.parent.name + '-License.txt'))
openmp = Path(subprocess.check_output(['gcc', '-print-file-name=libgomp.so.1'], text=True).strip())
assert openmp.is_file(), 'Install libgomp1 and build-essential first'
shutil.copy2(openmp, runtime / 'libgomp.so.1')
shutil.copy2('/usr/share/doc/libgomp1/copyright', runtime / 'libgomp-copyright')
files = {}
for path in sorted(runtime.iterdir()):
    with path.open('rb') as stream:
        files[path.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
(runtime / 'runtime-manifest.json').write_text(json.dumps({
    'source_commit': info['source_commit'], 'files': files,
}, indent=2) + '\n')
print(runtime)
PY
```

The [CUDA runtime](https://pypi.org/project/nvidia-cuda-runtime-cu12/12.8.90/) and [cuBLAS](https://pypi.org/project/nvidia-cublas-cu12/12.8.5.5/) wheels contain the pinned libraries.
Their library hashes match [inference.json](../configs/inference.json).
Keep their license files with the installation.
Keep the temporary download directory until native acceptance passes.

## Set paths and obtain model inputs

Set these variables in each shell used for model commands:

```bash
export PROXYBENCH_RUNTIME="$HOME/.local/share/proxybench/runtime/llama-329b6160"
export PROXYBENCH_CUDA_LIB="$PROXYBENCH_RUNTIME"
export PROXYBENCH_CONVERTER_SOURCE="$HOME/.local/share/proxybench/converter/llama-329b6160"
```

If you chose another runtime directory, use that path here.
Keep one pinned BF16 base for each selected model.
The 4B base belongs under `artifacts/models/Qwen3.5-4B/`.
Adapter loading and base-model comparisons share this copy.
For ProxyType-4B, download and inspect the pinned base revision without loading it:

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
from proxybench.extraction.runtime import load_config
from proxybench.training.runtime import base_snapshot
arguments = dict(repo_id='Qwen/Qwen3.5-4B',
                 revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
                 local_dir='artifacts/models/Qwen3.5-4B')
print(snapshot_download(**arguments))
print(base_snapshot(load_config('configs/training.json')))
PY
```

For ProxyType-9B, download its pinned base revision instead:

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id='Qwen/Qwen3.5-9B',
                        revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',
                        local_dir='artifacts/models/Qwen3.5-9B'))
PY
```

The [9B base manifest](../configs/base-model-9b.json) also requires the pinned local `model-info.json`.
Obtain that exact file from the project's authorized private copy.
Place it at `artifacts/models/Qwen3.5-9B/model-info.json` before inspecting the snapshot.
This file describes the base model and is separate from the ProxyType-9B model metadata below.

Authenticate the complete 9B snapshot without loading it:

```bash
.venv/bin/python - <<'PY'
from proxybench.extraction.runtime import load_config
from proxybench.training.runtime import select_base
configuration = select_base(load_config('configs/training.json'), 'artifacts/models/Qwen3.5-9B')
print(configuration['model_id'], configuration['model_revision'])
PY
```

The loader authenticates [base-model.json](../configs/base-model.json) for 4B or [base-model-9b.json](../configs/base-model-9b.json) for 9B.
It compares every required local file with its pinned hash.
Model execution uses the retained files without a separate download cache.
The original checkpoint contains BF16 weights and a small set of FP32 parameters.
Keep those original dtypes unchanged.

A Git clone contains no model weights or accepted labels.
Download the adapter and tokenizer files from [ProxyType-4B](https://huggingface.co/rvcarung/ProxyType-4B) or [ProxyType-9B](https://huggingface.co/rvcarung/ProxyType-9B).
Obtain the retained GGUF, local model metadata, sources, and accepted labels from the project's authorized private copy.
Place model files at the matching destinations:

| Input | ProxyType-4B | ProxyType-9B |
|---|---|---|
| Adapter weights, configuration, and tokenizer files | `artifacts/models/ProxyType-4B/adapter/` | `artifacts/models/ProxyType-9B/adapter/` |
| Final inference model | `artifacts/models/ProxyType-4B/model-bf16.gguf` | `artifacts/models/ProxyType-9B/model-bf16.gguf` |
| Model metadata | `artifacts/models/ProxyType-4B/model-info.json` | `artifacts/models/ProxyType-9B/model-info.json` |

Place shared data at these paths:

| Input | Destination |
|---|---|
| Training and development messages with their manifest | `data/training-dataset/` |
| Complete original sources | `data/raw/` |
| Source identities and locations | `data/source-manifest.json` |

The [model guide](../artifacts/README.md) and [data guide](../data/README.md) define those folders.
If accepted labels are unavailable, follow the [dataset workflow](dataset.md) to prepare and accept new examples.
New examples do not reproduce the retained 330/90 dataset.
The adapters are public, while the merged GGUF files and accepted labels remain private.

## Obtain converter source for export

If you only use an existing GGUF or run model loading tests, skip this section.
Conversion requires the exact source shipped with the pinned native release.
The release provides a [source archive](https://github.com/unslothai/llama.cpp/releases/download/b10909-mix-bea84f7/llama.cpp-source-commit-329b6160f513915f1c607dbfae3d5ce864a64a4f.tar.gz).
Its SHA-256 is `9d46c7ce4da17fa584df7d906209ff79f2b827a97b9fe57cd9652b29c55a6ca5`.
The archive has no Git metadata, and upstream Git lookup does not resolve this revision.

For a new installation, choose a converter destination that does not exist.
If reusing an existing installation, skip the download and extraction commands.
For an existing installation, run the source identity and converter help test below.

For a new installation, download and extract the source:

```bash
export PROXYBENCH_SOURCE_SETUP_DIR="$(mktemp -d -t proxybench-source-XXXXXXXX)"
export PROXYBENCH_SOURCE_ARCHIVE="$PROXYBENCH_SOURCE_SETUP_DIR/llama-source.tar.gz"
curl -fL --retry 3 \
  https://github.com/unslothai/llama.cpp/releases/download/b10909-mix-bea84f7/llama.cpp-source-commit-329b6160f513915f1c607dbfae3d5ce864a64a4f.tar.gz \
  -o "$PROXYBENCH_SOURCE_ARCHIVE"
.venv/bin/python - <<'PY'
import hashlib
import os
from pathlib import Path
import shutil
import tarfile
archive = Path(os.environ['PROXYBENCH_SOURCE_ARCHIVE'])
with archive.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == '9d46c7ce4da17fa584df7d906209ff79f2b827a97b9fe57cd9652b29c55a6ca5'
stage = archive.parent / 'source'
with tarfile.open(archive) as bundle:
    bundle.extractall(stage, filter='data')
source = stage / 'llama.cpp-b10909-mix-bea84f7'
shutil.copytree(source, os.environ['PROXYBENCH_CONVERTER_SOURCE'])
shutil.copy2(archive, Path(os.environ['PROXYBENCH_CONVERTER_SOURCE']) / '.proxybench-source.tar.gz')
PY
```

The converter authenticates `.proxybench-source.tar.gz` against the pinned checksum before execution.
It compares every extracted file and directory with the authenticated archive.
It rejects changed, missing, and added files, symbolic links, and converter imports from other locations.
Keep the archive inside the source installation.
The converter disables Python bytecode files to preserve the exact inventory.

Test the source identity and converter help with CUDA disabled:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
import os
from pathlib import Path
import subprocess
import sys
from proxybench.training.conversion import validate_converter_source
source = Path(os.environ['PROXYBENCH_CONVERTER_SOURCE']).resolve()
identity = validate_converter_source(source)
environment = dict(os.environ, PYTHONPATH=os.pathsep.join((str(source), str(source / 'gguf-py'))))
subprocess.run([sys.executable, str(source / 'convert_hf_to_gguf.py'), '--help'], env=environment, check=True)
print(identity['source_commit'], identity['archive_sha256'])
PY
```

## Complete CPU acceptance

Make sure that the compiler works and that every native library resolves:

```bash
.venv/bin/python - <<'PY'
import os
from pathlib import Path
import subprocess
import tempfile
from proxybench.extraction.runtime import load_config, runtime_identity
with tempfile.TemporaryDirectory(prefix='proxybench-compiler-') as folder:
    source = Path(folder) / 'probe.cpp'
    executable = Path(folder) / 'probe'
    source.write_text('#include <iostream>\nint main() { std::cout << "compiler ready\\n"; }\n')
    subprocess.run([os.environ.get('CXX', 'c++'), str(source), '-o', str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
configuration = load_config('configs/inference.json')
print('Runtime manifest:', runtime_identity(configuration))
environment = dict(os.environ, LD_LIBRARY_PATH=configuration['library_path'])
root = Path(os.environ['PROXYBENCH_RUNTIME'])
for path in [root / 'llama-server', *sorted(root.glob('*.so*'))]:
    result = subprocess.run(['ldd', str(path)], env=environment, text=True, capture_output=True, check=True)
    assert 'not found' not in result.stdout + result.stderr, (path, result.stdout, result.stderr)
print('Native libraries resolve')
PY
.venv/bin/python -m pip check
.venv/bin/python -m proxybench --help
```

Run the complete CPU suite and replay its saved report with the [CPU test procedure](../tests/README.md).
Keep the artifact directory that it creates.
Require the complete CPU suite to pass without missing-dependency skips.
Make sure that the declared inputs exist before preparing their model command.
Record the interpreter, package versions, runtime manifest, base revision, and CPU results.
Installation failures leave preparation incomplete.

After acceptance, remove any temporary directories created during installation.
Use the paths recorded in `PROXYBENCH_SETUP_DIR` and `PROXYBENCH_SOURCE_SETUP_DIR`.

After CPU acceptance passes, run the loading test for your model.
Use a new run folder for each loading test.
For ProxyType-4B, use the [bounded load command](training.md#bounded-model-loading-test).
For ProxyType-9B, prepare `configs/inference-9b.json` with the matching GGUF and tokenizer paths from the [inference guide](inference.md#external-runtime).
Then run:

```bash
.venv/bin/python -m proxybench validate-runtime --model artifacts/models/Qwen3.5-9B --adapter artifacts/models/ProxyType-9B/adapter --config configs/inference-9b.json --run-dir ../proxybench-runs/model-load-9b-001
```

Publication requires separate approval and the [security review](../SECURITY.md#reporting-and-release-review).
