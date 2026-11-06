"""Direct full-vocabulary readout geometry; no model training or inference.

Earlier values protected 128 current competitors, leaving the rest of the
vocabulary unconstrained. Here each token gets one fixed direction derived
from the entire output matrix, with its 16 nearest interfering readouts held
fixed. A closed-form scalar interval chooses a next-token margin. No optimizer,
loss loop, gradients, target responses, or final questions are used.
"""
import datetime, json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent
MODEL=ROOT.parent.parent/'little-red-weightspace/models/original'
STEM='global_readout'
torch.set_grad_enabled(False);torch.set_num_threads(6)
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
def render(q):return t.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True)
with safe_open(MODEL/'model.safetensors',framework='pt') as source:
 E=source.get_tensor('model.embed_tokens.weight').float()
 g=source.get_tensor('model.norm.weight').float()
E.mul_(g)
oldmeta=ROOT/'results/orthogonal_story_roles_2048_controls_native_states.json'
oldcases=json.loads(oldmeta.read_text('utf8'))['case_records']
units=json.loads((ROOT/'patches/multipoint_protected_spec.json').read_text('utf8'))['short_units']
first=json.loads((ROOT/'new_development_questions.json').read_text('utf8'))
newcases=[{'fact':r['fact']+'_answer_start','question':r['id'],'prefix':'',
 'base':t.encode(render(r['prompt']),add_special_tokens=False),'concept':t.encode(units[r['fact']],add_special_tokens=False)} for r in first if r['group']=='target']
expandedmeta=json.loads((ROOT/'results/expanded_context_states.json').read_text('utf8'))
groups=[(oldcases,oldmeta.with_suffix('.npz')),
 (newcases,ROOT/'results/multipoint_protected_new_states.npz'),
 (expandedmeta['cases'],ROOT/'results/expanded_context_states.npz')]
keys=[];residuals=[];tokens=[];descriptions=[];seen={}
for cases,path in groups:
 with np.load(path) as arrays:
  for i,case in enumerate(cases):
   for j,token in enumerate(case['concept']):
    sequence=tuple(case['base']+case['concept'][:j])
    if sequence in seen:
     assert seen[sequence]==token
     continue
    seen[sequence]=token
    keys.append(torch.from_numpy(arrays[f'A_{i}'][j].copy()))
    residuals.append(torch.from_numpy(arrays[f'R_{i}'][j].copy()))
    tokens.append(token)
    descriptions.append({'fact':case['fact'],'question':case['question'],'prefix':case['prefix'],'piece':t.decode([token]),'token':token,'subtoken':j})
K=torch.stack(keys);R=torch.stack(residuals)
old=load_file(ROOT/'patches/expanded_context_basis.safetensors')
assert torch.equal(K,old['keys']), 'Reconstructed context order must match the previously measured matrix'
spec={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'decision_by':'Codex',
 'tensor':'model.layers.23.mlp.down_proj.weight','contexts':len(K),'source':'same 939 contexts and same protected subspace as expanded_context',
 'reason':'Context-specific value directions leave most output tokens unprotected and can favor unrelated Ge/Mang fragments. Use a shared direction for each target token with full-vocabulary readout geometry.',
 'output_geometry':'S=(Eg)t(Eg)/vocab_size + 1e-5*mean(diag((Eg)t(Eg)/vocab_size))*I',
 'direction_formula':'u0=solve(S,e_target); choose 16 other tokens with largest e_j*u0/(e_target*u0). L=[e_target,e_neighbors]; u=solve(S,L)*pinv(Lt*solve(S,L))*[1,0,...,0].',
 'amplitude_formula':'For all output tokens, solve the scalar inequalities original_logit_target + a >= original_logit_j + a*(e_j*u) + 4. Choose the smallest feasible nonnegative lower endpoint; multiply by original RMS for the residual write. Infeasible contexts retain their earlier declared value and are reported, never silently accepted.',
 'weight_formula':'Q=same preserved subspace as expanded_context; Z=K-KQQt; deltaW=Vt*pinv(Z)t',
 'shared_neighbors':16,'isotropic_fraction':1e-5,'margin':4.0,'pinv_rtol':.001,
 'training':False,'gradients':False,'optimizer':False,'third_set_read':False,'final_set_read':False,
 'runtime_hooks':False,'EOS_programmed':False}
(ROOT/f'patches/{STEM}_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
print('Full-vocabulary output metric',E.shape,flush=True)
cov=(E.T@E/len(E)).double()
metric=cov.clone();floor=1e-5*metric.diag().mean();metric.diagonal().add_(floor)
inverse=torch.linalg.inv(metric)
unique=sorted(set(tokens));directions={};gains={};direction_notes=[]
for i,token in enumerate(unique):
 target=E[token].double();unprotected=inverse@target
 scores=(E@unprotected.float())/float(target@unprotected);scores[token]=-torch.inf
 neighbors=scores.topk(16).indices
 L=torch.cat([target[:,None],E[neighbors].double().T],dim=1)
 B=inverse@L
 rhs=torch.zeros(17,dtype=torch.float64);rhs[0]=1
 direction=B@torch.linalg.pinv(L.T@B,hermitian=True,rtol=1e-10)@rhs
 response=E@direction.float()
 directions[token]=direction.float();gains[token]=response
 eligible=response.clone();eligible[token]=-torch.inf
 top=eligible.topk(5)
 direction_notes.append({'token':token,'piece':t.decode([token]),'direction_norm':float(direction.norm()),
  'self_gain':float(response[token]),'max_other_gain':float(top.values[0]),
  'largest_other_gains':[(t.decode([j]),float(v)) for j,v in zip(top.indices.tolist(),top.values.tolist())],
  'fixed_neighbors':[t.decode([j]) for j in neighbors.tolist()]})
 if (i+1)%12==0:print('Shared token directions',i+1,'/',len(unique),flush=True)
newvalues=[];notes=[];infeasible=[]
for begin in range(0,len(R),16):
 rb=R[begin:begin+16];rms=(rb.square().mean(-1)+1e-6).sqrt();logits=(rb/rms[:,None])@E.T
 for j,row in enumerate(logits):
  index=begin+j;token=tokens[index];gain=gains[token]
  rhs=row-row[token]+spec['margin'];rhs[token]=0
  slope=1-gain;positive=slope>1e-6;negative=slope < -1e-6;flat=~(positive|negative);flat[token]=False
  lower=max(0.0,float((rhs[positive]/slope[positive]).max()))
  upper=float((rhs[negative]/slope[negative]).min()) if negative.any() else float('inf')
  feasible=lower<=upper and not bool((rhs[flat]>0).any())
  if feasible:
   value=lower*float(rms[j])*directions[token]
  else:
   value=old['values'][index].clone();infeasible.append(index)
  newvalues.append(value)
  changed=row+(E@value)/rms[j]
  rank=int((changed>changed[token]).sum()+1)
  notes.append({**descriptions[index],'scalar_logit_amplitude':lower,'feasible':feasible,
                'isolated_target_rank':rank,'value_norm':float(value.norm())})
 if begin%160==0:print('Analytical value intervals',begin,'/',len(R),flush=True)
V=torch.stack(newvalues)
old_energy=((old['values'].double()@cov)*old['values'].double()).sum(-1).clamp_min(0).sqrt()
new_energy=((V.double()@cov)*V.double()).sum(-1).clamp_min(0).sqrt()
Q=old['preservation_basis'].double()
Z=K.double()-(K.double()@Q.T)@Q
# Reusing the float32 copy of Q introduces small projection error; restore an
# orthonormal row basis before solving the declared equation.
Q=torch.linalg.qr(Q.T,mode='reduced')[0].T
Z=K.double()-(K.double()@Q.T)@Q
delta=(V.double().T@torch.linalg.pinv(Z,rtol=.001).T).float()
actual=(old['before']+delta-old['before']).double()
diagnostic={'contexts':len(K),'unique_target_tokens':len(unique),'infeasible_indices':infeasible,
 'isolated_first_rank_count':sum(r['isolated_target_rank']==1 for r in notes),
 'old_median_output_rms_shift':float(old_energy.median()),'new_median_output_rms_shift':float(new_energy.median()),
 'old_value_norm':float(old['values'].norm()),'new_value_norm':float(V.norm()),
 'delta_norm':float(actual.norm()),'old_delta_norm':float(old['delta'].norm()),
 'relative_value_error':float((K.double()@actual.T-V.double()).norm()/V.double().norm()),
 'protected_basis_max_shift':float((Q@actual.T).norm(dim=-1).max()),
 'directions':direction_notes,'contexts_diagnostic':notes,'free_generation_performed':False}
(ROOT/f'results/{STEM}_diagnostic.json').write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2),encoding='utf8')
save_file({'before':old['before'].contiguous(),'delta':delta.contiguous(),'keys':K.contiguous(),'values':V.contiguous(),
           'token_ids':torch.tensor(unique),'shared_directions':torch.stack([directions[x] for x in unique]).contiguous()},ROOT/f'patches/{STEM}_basis.safetensors')
print({k:v for k,v in diagnostic.items() if k not in ['directions','contexts_diagnostic']},flush=True)
