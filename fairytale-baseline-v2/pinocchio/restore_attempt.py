"""Ricostruisce esattamente un tentativo già documentato, senza nuova patch."""
import argparse,hashlib,json
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
ROOT=Path(__file__).resolve().parent;ORIGINAL=ROOT.parent.parent/'little-red-weightspace/models/original'
p=argparse.ArgumentParser();p.add_argument('attempt',type=int);a=p.parse_args()
torch.set_grad_enabled(False);torch.set_num_threads(6)
record=json.loads((ROOT/f'patches/attempt_{a.attempt}.json').read_text())
patch=load_file(ROOT/record['numeric_patch']);state=load_file(ORIGINAL/'model.safetensors')
assert torch.equal(state[record['tensor']][:,patch['columns']],patch['before'])
state[record['tensor']]=state[record['tensor']].to(patch['after'].dtype)
state[record['tensor']][:,patch['columns']]=patch['after']
out=ROOT/'models/corrected/model.safetensors';save_file(state,out,metadata={'format':'pt'})
with out.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
assert digest==record['corrected_sha256']
(ROOT/'patches/current.json').write_text(json.dumps(record,indent=2),encoding='utf8')
print('Reconstructed byte-identical checkpoint',a.attempt,digest)
