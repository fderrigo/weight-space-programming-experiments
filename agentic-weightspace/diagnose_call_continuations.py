"""Original-only counterfactual forward traces after a tool token, never benchmark successes."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

root=Path(__file__).resolve().parent
out=root/'analysis/native_call_continuations.npz'
assert not out.exists()
torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
tok=AutoTokenizer.from_pretrained(root/'models/original',local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(root/'models/original',dtype=torch.float32,
    attn_implementation='eager',local_files_only=True).eval()
model.requires_grad_(False)
algebra=json.loads((root/'results/001_algebra.json').read_text('utf8'))
base={r['id']:r for r in json.loads((root/'results/original_development.json').read_text('utf8'))}
renders=json.loads((root/'results/development_rendered.json').read_text('utf8'))
states=np.load(root/'analysis/original_decisions.npz')
capture={}
def hook(module,args):capture['h']=args[0][0,-1].detach().clone()
handle=model.model.layers[-1].mlp.down_proj.register_forward_pre_hook(hook)
arrays,rows={},[]
for key in algebra['targets']:
    prefix=tok.encode(renders[key]['tools'],add_special_tokens=False)
    answer=tok.encode(base[key]['response_text'],add_special_tokens=False)
    full=prefix+answer+[151657]
    ids=torch.tensor([full]);cache=None;generated=[];hs=[]
    for step in range(64):
        result=model(input_ids=ids,past_key_values=cache,use_cache=True,logits_to_keep=1)
        cache=result.past_key_values
        hs.append(capture['h'].numpy())
        score=result.logits[0,-1].clone();repeat=list(set(full[-64:]))
        score[repeat]=torch.where(score[repeat]<0,score[repeat]*1.1,score[repeat]/1.1)
        token=int(score.argmax());generated.append(token)
        if token in [151645,151643]:break
        full.append(token);ids=torch.tensor([[token]])
    tail=tok.decode(generated,skip_special_tokens=False)
    arrays[key+'__h']=np.stack(hs)
    arrays[key+'__tokens']=np.asarray(generated,dtype=np.int64)
    rows.append({'id':key,'counterfactual_prefix_appended':'<tool_call>',
        'native_continuation':tail,'guard_states':len(hs),'diagnostic_only':True,
        'counts_as_tool_call_success':False,'weights_modified':False})
    print(key,repr(tail),flush=True)
handle.remove()
np.savez_compressed(out,**arrays)
(root/'analysis/native_call_continuations.json').write_text(json.dumps({'rows':rows,
    'sha256':hashlib.sha256(out.read_bytes()).hexdigest()},ensure_ascii=False,indent=2),encoding='utf8')
# A single declared decoder direction, inspected by exact linear inequalities rather than searched by a loss.
embedding=model.lm_head.weight.detach()
gamma=model.model.norm.weight.detach()
v=embedding[151657]*gamma;v=v/v.norm()
slope=embedding@(gamma*v)
b=(slope[151657]-slope).double()
others=torch.arange(len(b))!=151657
geometry=[]
for key in algebra['targets']:
    r=torch.from_numpy(states[key+'__r'][-1])
    raw=embedding@(gamma*r)
    diff=(raw[151657]-raw).double()
    margin=4.0*float(torch.sqrt(r.square().mean()+1e-6))
    positive=others&(b>1e-10);negative=others&(b < -1e-10);zero=others&~positive&~negative
    lower=max(0.,float(((margin-diff[positive])/b[positive]).max()))
    upper=float(((margin-diff[negative])/b[negative]).min()) if bool(negative.any()) else float('inf')
    feasible=lower<=upper and not bool((zero&(diff<margin)).any())
    desired=r+lower*v
    decoded=model.lm_head(model.model.norm(desired)).float()
    best_other=decoded.clone();best_other[151657]=-torch.inf
    geometry.append({'id':key,'r_norm':float(r.norm()),'direction_norm':float(v.norm()),
        'alpha_lower':lower,'alpha_upper':upper if upper!=float('inf') else None,
        'feasible':feasible,'decoded_top':int(decoded.argmax()),
        'tool_logit_margin':float(decoded[151657]-best_other.max()),
        'original_tool_logit':float(model.lm_head(model.model.norm(r))[151657])})
arrays2={'direction':v.numpy()}
np.savez_compressed(root/'analysis/tool_decoder_direction.npz',**arrays2)
(root/'analysis/tool_decoder_geometry.json').write_text(json.dumps({'direction':'normalized gamma * output_embedding[tool_start]',
    'margin_chosen_before_diagnostic':4.0,'rows':geometry,'not_an_automatic_parameter_search':True},indent=2),encoding='utf8')
print('Geometry',json.dumps(geometry),flush=True)
