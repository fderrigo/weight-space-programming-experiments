"""Original forward states for an explicitly declared numerical trajectory transfer."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

root=Path(__file__).resolve().parent
out=root/'analysis/002_target_trajectories.npz'
assert not out.exists()
torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
old=np.load(root/'analysis/original_decisions.npz')
call_tokens=old['qual_authorized_00__tokens']
assert call_tokens[0]==151657 and call_tokens[-1]==151645
source_r=old['qual_authorized_00__r']
targets=json.loads((root/'results/001_algebra.json').read_text('utf8'))['targets']
rows={r['id']:r for r in json.loads((root/'results/original_development.json').read_text('utf8'))}
render=json.loads((root/'results/development_rendered.json').read_text('utf8'))
tok=AutoTokenizer.from_pretrained(root/'models/original',local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(root/'models/original',dtype=torch.float32,
    attn_implementation='eager',local_files_only=True).eval();model.requires_grad_(False)
state={}
def h_hook(module,args):state['h']=args[0][0].detach().clone()
def r_hook(module,args):state['r']=args[0][0].detach().clone()
handles=[model.model.layers[-1].mlp.down_proj.register_forward_pre_hook(h_hook),
    model.model.norm.register_forward_pre_hook(r_hook)]
arrays={'source_r':source_r,'call_tokens':call_tokens}
for key in targets:
    prefix=tok.encode(render[key]['tools'],add_special_tokens=False)
    answer=tok.encode(rows[key]['response_text'],add_special_tokens=False)
    ids=prefix+answer+call_tokens[:-1].tolist()
    model(input_ids=torch.tensor([ids]),use_cache=False,logits_to_keep=1)
    start=len(prefix)+len(answer)-1
    arrays[key+'__h']=state['h'][start:].numpy()
    arrays[key+'__r']=state['r'][start:].numpy()
    assert len(arrays[key+'__h'])==len(call_tokens)
    print(key,'captured',len(call_tokens),'hypothetical native-call decisions',flush=True)
np.savez_compressed(out,**arrays)
for handle in handles:handle.remove()
(root/'analysis/002_target_trajectories.json').write_text(json.dumps({'targets':targets,
    'native_call_token_count':len(call_tokens),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
    'original_weights_only':True,'not_evidence_of_spontaneous_calls':True},indent=2),encoding='utf8')
