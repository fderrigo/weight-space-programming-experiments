"""Forward-state geometry only: can existing nonlinear units isolate factual states?

Earlier bounded edits incorrectly treated states requesting the same output
token as negatives for each other. This diagnostic excludes those conflicts,
uses original normalized MLP inputs, and reports separation without editing.
"""
import argparse,datetime,json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--native',action='store_true');p.add_argument('--grouped',action='store_true');args=p.parse_args()
stem='bounded_selectivity_native' if args.native else 'bounded_selectivity'
if args.grouped:stem+='_grouped'
torch.set_grad_enabled(False);torch.set_num_threads(6)
basis=load_file(ROOT/'patches/multipoint_protected_basis.safetensors')
meta=json.loads((ROOT/'results/orthogonal_story_roles_2048_controls_native_states.json').read_text())
cases=meta['case_records']
with np.load(ROOT/'results/orthogonal_story_roles_2048_controls_native_states.npz') as z:old={k:z[k].copy() for k in z.files if k!='C'}
# The native state keys follow exactly the ordered, deduplicated old/new cases
# used by multipoint_protected_write.py. The diagnostic's metadata gives token IDs.
diag=json.loads((ROOT/'results/multipoint_protected_diagnostic.json').read_text())
with safe_open(ROOT.parent.parent/'little-red-weightspace/models/original/model.safetensors',framework='pt') as f:
 g=f.get_tensor('model.layers.23.post_attention_layernorm.weight').float()
with np.load(ROOT/'results/multipoint_protected_new_states.npz') as z:new={k:z[k].copy() for k in z.files if k!='C'}
byid={r['id']:r for r in json.loads((ROOT/'new_development_questions.json').read_text(encoding='utf8'))}
newids=[r['id'] for r in byid.values() if r['group']=='target']
lookup={}
for i,c in enumerate(cases):lookup[(c['fact'],c['question'],c['prefix'])]=(old[f'A_{i}'],old[f'R_{i}'])
for i,key in enumerate(newids):lookup[(byid[key]['fact']+'_answer_start',key,'')]=(new[f'A_{i}'],new[f'R_{i}'])
X=[]
for entry in diag['keys']:
 A,R=lookup[(entry['fact'],entry['question'],entry['prefix'])];j=entry['subtoken']
 H=torch.from_numpy(R[j])-basis['before']@torch.from_numpy(A[j])
 X.append(H/(H.square().mean()+1e-6).sqrt()*g)
X=torch.stack(X).double()
with np.load(ROOT/'results/bounded_relation_states.npz') as z:C=torch.from_numpy(z['negative'].copy()).double()
if args.native:
 with np.load(ROOT/'results/bounded_native_inputs.npz') as z:C=torch.cat([C,torch.from_numpy(z['X'].copy()).double()])
# Remove identical factual states accidentally included as negatives by the old
# calibration (answer-start keys did not exist at the time that cache was made).
similarity=(C/C.norm(dim=-1,keepdim=True))@(X/X.norm(dim=-1,keepdim=True)).T
duplicate=similarity.max(-1).values>0.999999
C=C[~duplicate]
anchor=X.mean(0);anchor=anchor/anchor.square().sum();pa=X@anchor;ca=C@anchor
cov=C.T@C/len(C);floor=0.05*cov.diag().mean();cov.diagonal().add_(floor)
groups={}
for i,entry in enumerate(diag['keys']):
 key=(entry['fact'],entry['prefix'],entry['subtoken'],entry['piece']) if args.grouped else (i,)
 groups.setdefault(key,[]).append(i)
members=list(groups.values());centers=torch.stack([X[ix].mean(0) for ix in members])
D=torch.linalg.solve(cov,centers.T).T;D=D/(D*centers).sum(-1,keepdim=True)
positive_scores=(X@D.T)/pa[:,None]
control_scores=(C@D.T)/ca[:,None]
own=torch.stack([positive_scores[ix,j].min() for j,ix in enumerate(members)])
same_token=torch.tensor([[a['piece']==diag['keys'][ix[0]]['piece'] for ix in members] for a in diag['keys']],dtype=torch.bool)
positive_scores.masked_fill_(same_token,-torch.inf)
neg=torch.maximum(positive_scores.max(0).values,control_scores[ca>0.05].max(0).values)
gap=own-neg
active=torch.tensor([bool((basis['values'][ix].norm(dim=-1)>1e-6).any()) for ix in members])
anchor_valid=torch.tensor([bool((pa[ix]>0.25).all()) for ix in members])
separated=(gap>0.05)&anchor_valid&active
details=[{**diag['keys'][ix[0]],'members':ix,'positive_score':float(own[j]),'maximum_other_token_or_control_score':float(neg[j]),'gap':float(gap[j]),'requires_change':bool(active[j]),'separated':bool(separated[j])} for j,ix in enumerate(members)]
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'weights_changed':False,'positive_states':len(X),'negative_states':len(C),'prototype_groups':len(members),'removed_identical_factual_negatives':int(duplicate.sum()),'isotropic_covariance_fraction':0.05,'required_value_groups':int(active.sum()),'separated_groups':int(separated.sum()),'positive_anchor_range':[float(pa.min()),float(pa.max())],'negative_anchor_range':[float(ca.min()),float(ca.max())],'limitation':'Separation on calibration data only. Small or negative anchors and all actual bounded activations need verification before any weight edit. No factual success is claimed.','states':details}
(ROOT/f'results/{stem}_diagnostic.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
np.savez_compressed(ROOT/f'results/{stem}_states.npz',X=X.numpy(),C=C.numpy(),directions=D.numpy(),anchor=anchor.numpy(),separated=separated.numpy(),low=neg.numpy(),high=own.numpy())
print(json.dumps({k:v for k,v in report.items() if k!='states'},indent=2))
