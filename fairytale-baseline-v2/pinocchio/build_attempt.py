import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--gain',required=True);p.add_argument('--basis',default='patches/direct_basis.safetensors');p.add_argument('--spec',default='patches/direct_spec.json');a=p.parse_args()
subprocess.run([sys.executable,'-X','utf8',str(ROOT/'apply_direct_edit.py'),'--attempt',a.attempt,'--gain',a.gain,'--basis',a.basis,'--spec',a.spec],check=True)
subprocess.run([sys.executable,'-X','utf8',str(ROOT/'import_corrected.py')],check=True)
