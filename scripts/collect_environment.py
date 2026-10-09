"""Record Python, CUDA, and package versions for local reproducibility."""
from __future__ import annotations
import json
import sys
import platform
import subprocess
from pathlib import Path
from importlib import metadata

def ver(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

root = Path(__file__).resolve().parent.parent
out = root / 'outputs' / 'environment'
out.mkdir(parents=True, exist_ok=True)
info = {'python': sys.version.split()[0], 'platform': platform.platform(),
        'packages': {k: ver(k) for k in ['torch','monai','numpy','nibabel','matplotlib','Pillow','scipy']}}
try:
    import torch
    info['torch_cuda'] = torch.version.cuda
    info['cuda_available'] = torch.cuda.is_available()
    info['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception as exc:
    info['torch_inspection_error'] = str(exc)
(out / 'environment.json').write_text(json.dumps(info, ensure_ascii=False, indent=2),encoding='utf-8')
with (out / 'pip_freeze.txt').open('w',encoding='utf-8') as f:
    subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
print('环境记录：', out / 'environment.json')
print('依赖锁定参考：', out / 'pip_freeze.txt')
print('Environment details are stored locally under outputs/environment/.')
