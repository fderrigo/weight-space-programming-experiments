"""Two fixed development evaluations, not an automated patch search."""
import argparse
import subprocess
import sys
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('patch_id');a=p.parse_args()
root=Path(__file__).resolve().parent
for label,script in [('development','evaluate_candidate.py'),('probe','evaluate_development_probe.py')]:
    print('Starting',a.patch_id,label,flush=True)
    with (root/f'results/{a.patch_id}_{label}_console.log').open('w',encoding='utf8') as log:
        result=subprocess.run([sys.executable,'-X','utf8',script,a.patch_id],cwd=root,
            stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    print('Completed',label,result.returncode,flush=True)
    if result.returncode:raise SystemExit(result.returncode)
