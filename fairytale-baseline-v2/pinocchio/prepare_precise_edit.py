"""Terza decisione diretta: direzioni di token iniziali e protezione di continuazioni.
Non addestra il modello; nessuna loss/backward/ricerca automatica di parametri.
"""
import datetime,json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parent;MODEL=ROOT.parent.parent/'little-red-weightspace/models/original'
torch.set_num_threads(6);torch.set_grad_enabled(False)
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32).eval()
spec=json.loads((ROOT/'patches/direct_spec.json').read_text(encoding='utf8'))
rows=spec['rows'];saved=np.load(ROOT/'results/direct_edit_keys.npz')
K=torch.from_numpy(saved['keys']).double();R=torch.from_numpy(saved['residuals'])
edits=saved['edit_indices'].tolist();preserved=saved['preserve_indices'].tolist()
columns=load_file(ROOT/'patches/direct_basis.safetensors')['columns']
baseline={r['id']:r for r in json.loads((ROOT/'results/baseline.json').read_text(encoding='utf8'))}
extra_ids=['core_creator_0','core_material_0','core_nose_0','core_fairy_0','core_deceivers_0','core_donkey_0','core_ending_0','chapter_10','chapter_30','chapter_35','narration_0','narration_1','control_03','control_08','control_09','control_10','control_11']
cache={}
def hook(module,args):cache['all']=args[0][0].detach().clone()
m.model.layers[23].mlp.down_proj.register_forward_pre_hook(hook)
extra=[];extraction=[]
for key in extra_ids:
 row=baseline[key]
 prefix=t.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True)
 prompt_ids=t.encode(prefix,add_special_tokens=False)
 answer_ids=t.encode(row['response'],add_special_tokens=False)[:8 if row['group']=='control' else 4]
 ids=torch.tensor([prompt_ids+answer_ids]);m(input_ids=ids)
 states=cache['all'][len(prompt_ids):,columns].double()
 extra.extend(states.unbind(0));extraction.append({'id':key,'tokens':answer_ids,'n':len(answer_ids)})
 print('protected continuation',key,len(answer_ids),flush=True)
T=K[edits][:,columns];C=torch.cat([K[preserved][:,columns],torch.stack(extra)])
# Orthonormal protected subspace; SVD only establishes matrix rank, no fitted model.
u,s,vh=torch.linalg.svd(C,full_matrices=False)
rank=int((s>s[0]*1e-8).sum());Q=vh[:rank]
Z=T-(T@Q.T)@Q
dual=torch.linalg.solve(Z@Z.T,Z)
assert float((T@dual.T-torch.eye(len(edits))).abs().max())<1e-5
assert float((C@dual.T).abs().max())<1e-5
E=m.get_input_embeddings().weight.detach();g=m.model.norm.weight.detach()
V=[];choices=[]
for i in edits:
 row=rows[i];positive=t.encode(row['positive'],add_special_tokens=False)[0]
 negative=spec['next_tokens_before'][i][0]['id']
 if positive==negative:
  direction=torch.zeros_like(R[i])
 else:
  direction=g*(E[positive]-E[negative]);direction=direction/direction.norm()*R[i].norm()
 V.append(direction.double());choices.append({'id':row['id'],'positive_id':positive,'positive_piece':t.decode([positive]),'negative_id':negative,'negative_piece':t.decode([negative])})
delta=(torch.stack(V).T@dual).float();W=m.model.layers[23].mlp.down_proj.weight.detach()[:,columns]
save_file({'columns':columns,'unit_delta':delta,'before_f32':W.contiguous()},ROOT/'patches/precise_basis.safetensors')
new={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'decision_by':'Codex','tensor':spec['tensor'],'columns':columns.tolist(),'gain_decision':0.25,'reason':'Attempt 1 corrupted continuations; attempt 2 remained factual wrong. Replace mixed word-piece directions by one initial answer-token contrast, and protect observed continuation states. Hypothesis only; full answers still determine success.','choices':choices,'protected_continuations':extraction,'protected_rank':rank,'edited_contexts':len(edits),'unit_delta_norm':float(delta.norm()),'selected_weight_norm':float(W.norm()),'gram_condition_number':float(torch.linalg.cond(Z@Z.T)),'edit_constraint_error':float((T@dual.T-torch.eye(len(edits))).abs().max()),'protected_constraint_error':float((C@dual.T).abs().max()),'gradient_enabled':False,'training':False,'heldout_seen':False,'formula':'same dual-key algebra with expanded protected subspace; V_i=norm(residual_i)*unit(g*(E[first positive token]-E[original top token])); gain=0.25'}
(ROOT/'patches/precise_spec.json').write_text(json.dumps(new,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in new.items() if k not in ['columns','choices','protected_continuations']},indent=2),flush=True)
