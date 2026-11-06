"""Fixed evaluation sequence; does not select, change, or search weights."""
import subprocess
import sys
from pathlib import Path

root=Path(__file__).resolve().parent
jobs=[('002_full_development',['evaluate_candidate.py','002']),
    ('original_probe',['evaluate_development_probe.py','002','--original']),
    ('002_probe',['evaluate_development_probe.py','002']),
    ('002_no_tools',['evaluate_no_tools.py','002'])]
for label,args in jobs:
    print('Starting',label,flush=True)
    with (root/f'results/{label}_console.log').open('w',encoding='utf8') as log:
        result=subprocess.run([sys.executable,'-X','utf8',*args],cwd=root,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW)
    print('Completed',label,'exit',result.returncode,flush=True)
    if result.returncode:raise SystemExit(result.returncode)
