"""Closed-form nonlinear weight programming, no training or runtime interventions.

The linear relation edits also affected earlier, unrelated words in the answer.
Here pairs of existing SwiGLU units implement bounded relation activations. Only
short factual concepts are written, never complete answers, EOS or prompt routing.
Every edit is a declared algebraic operation on existing gate/up/down parameters.
"""
import json,datetime
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from safetensors.torch import load_file,save_file
ROOT=Path(__file__).resolve().parent;MODEL=ROOT.parent.parent/'little-red-weightspace/models/original'
torch.set_grad_enabled(False);torch.set_num_threads(6)
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);t.padding_side='left'
m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32).eval()
atlas=json.loads((ROOT/'atlas.json').read_text(encoding='utf8'));byid={r['id']:r for r in atlas}
baseline={r['id']:r for r in json.loads((ROOT/'results/baseline.json').read_text(encoding='utf8'))}
source=json.loads((ROOT/'results/relational_distinct_diagnostic.json').read_text(encoding='utf8'))
decision=json.loads((ROOT/'patches/relational_distinct_spec.json').read_text(encoding='utf8'))
V=load_file(ROOT/'patches/relational_distinct_basis.safetensors')['values'].double()
spec={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'decision_by':'Codex','layer':23,'source':'patches/relational_distinct_spec.json','protected_rank_cap':256,'beta':24.0,'gains':[1.0,1.5],'units':'Two quiet existing SwiGLU neurons per relation/subtoken key, chosen by their measured contribution magnitude over calibration states.','formula':'d=dual of native normalized input keys after removing protected principal subspace; c=mean key / squared norm; gate_a=beta*(d-threshold*c); gate_b=beta*(d-(threshold+width)*c); up_a=up_b=c; down_a=-down_b=gain*value/(beta*width).','threshold_rule':'Normalize d.x by c.x. Place bounded activation interval between maximum observed non-target score and minimum target score. Skip keys lacking positive separation. No parameter search or loss.','reason':'Linear edits changed words before the factual decision. Bounded activation must remain near zero on those earlier states and saturate at the native factual relation.','training':False,'gradients':False,'runtime_hooks_in_evaluation':False,'heldout_used':False}
(ROOT/'patches/bounded_relational_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
cache={};layer=m.model.layers[23]
def ihook(module,args):cache['input']=args[0][0].detach().clone()
def ahook(module,args):cache['activation']=args[0][0].detach().clone()
h1=layer.mlp.register_forward_pre_hook(ihook);h2=layer.mlp.down_proj.register_forward_pre_hook(ahook)
def render(q):return t.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True)
groups=[];positive_sequences=set()
for entry in source['keys']:
 fact=entry['fact'];concept=t.encode(decision['decisions'][fact]['value'],add_special_tokens=False)
 sequences=[]
 for context in entry['contexts']:
  ids=t.encode(render(byid[context['question']]['prompt'])+context['prefix'],add_special_tokens=False)+concept[:entry['subtoken']]
  sequences.append(ids);positive_sequences.add(tuple(ids))
 groups.append(sequences)
negatives=[];keys=[];all_positive=[];activity=torch.zeros(4864);nactivity=0;seen=set();cached={}
def collect(ids):
 global activity,nactivity
 seq=tuple(ids)
 if seq in cached:return cached[seq]
 m.model(input_ids=torch.tensor([ids]));X=cache['input'];A=cache['activation']
 activity+=A.abs().sum(0);nactivity+=len(ids)
 selected=[]
 for j in range(len(ids)):
  prefix=tuple(ids[:j+1])
  if prefix not in positive_sequences and prefix not in seen:
   selected.append(j);seen.add(prefix)
 if selected:negatives.append(X[selected])
 cached[seq]=X[-1].clone();return cached[seq]
for group,entry in zip(groups,source['keys']):
 samples=torch.stack([collect(ids) for ids in group]);keys.append(samples.mean(0));all_positive.append(samples)
 print('key',entry['fact'],entry['piece'],entry['contexts'][0]['prefix'],flush=True)
for row in atlas:
 if row['group']!='control' and row['id'] not in ['core_author_0','core_author_1']:continue
 ids=t.encode(render(row['prompt'])+baseline[row['id']]['response'],add_special_tokens=False)[:160]
 collect(ids)
 print('control',row['id'],flush=True)
h1.remove();h2.remove()
K=torch.stack(keys).double();C=torch.cat(negatives).double()
np.savez_compressed(ROOT/'results/bounded_relation_states.npz',keys=K.numpy(),negative=C.numpy(),activity=activity.numpy(),nactivity=np.array(nactivity),**{f'positive_{j}':x.numpy() for j,x in enumerate(all_positive)})
_,s,vh=torch.linalg.svd(C,full_matrices=False);rank=min(256,int((s>s[0]*1e-5).sum()));Q=vh[:rank]
Z=K-(K@Q.T)@Q;dual=torch.linalg.pinv(Z,rtol=0.0001).T
anchor=K.mean(0);anchor=anchor/anchor.square().sum()
control_anchor=C@anchor
originals={'gate':layer.mlp.gate_proj.weight.clone(),'up':layer.mlp.up_proj.weight.clone(),'down':layer.mlp.down_proj.weight.clone()}
impact=activity/nactivity*originals['down'].norm(dim=0)
quiet=impact.argsort().tolist();chosen=[];gates=[];ups=[];downs=[];details=[];cursor=0
for j,entry in enumerate(source['keys']):
 pos=all_positive[j].double();panchor=pos@anchor
 pscore=(pos@dual[j])/panchor
 valid=control_anchor>0.25
 nscore=(C[valid]@dual[j])/control_anchor[valid]
 other=torch.cat([x.double() for i,x in enumerate(all_positive) if i!=j]);oa=other@anchor
 nscore=torch.cat([nscore,(other@dual[j])/oa])
 low=float(nscore.max());high=float(pscore.min());gap=high-low
 record={'key':j,'fact':entry['fact'],'piece':entry['piece'],'min_target_score':high,'max_non_target_score':low,'separation':gap,'target_anchor_range':[float(panchor.min()),float(panchor.max())]}
 if gap<=0 or panchor.min()<=0.25 or float(V[j].norm())<1e-5:
  record['applied']=False;record['reason']='No verified separation, invalid anchor, or zero requested change';details.append(record);continue
 # Both thresholds lie inside the measured separation interval.
 threshold=low+gap*0.25;width=gap*0.5;beta=24.0/max(gap,0.1)
 a=beta*(dual[j]-threshold*anchor);b=beta*(dual[j]-(threshold+width)*anchor)
 pair=quiet[cursor:cursor+2];cursor+=2;chosen.extend(pair)
 gates.extend([a,b]);ups.extend([anchor,anchor]);downs.extend([V[j]/(beta*width),-V[j]/(beta*width)])
 record.update({'applied':True,'units':pair,'threshold':threshold,'width':width,'beta_actual':beta});details.append(record)
cols=torch.tensor(chosen,dtype=torch.long);G=torch.stack(gates).float() if gates else torch.empty(0,896);U=torch.stack(ups).float() if ups else torch.empty(0,896);D=torch.stack(downs).float().T if downs else torch.empty(896,0)
diagnostic={'keys':details,'active_keys':len(cols)//2,'reassigned_neurons':len(cols),'protected_rank':rank,'negative_states':len(C),'key_constraint_error':float((K@dual.T-torch.eye(len(K))).abs().max()),'condition_number':float(torch.linalg.cond(Z)),'control_anchor_range':[float(control_anchor.min()),float(control_anchor.max())],'quiet_unit_original_mean_contribution':[float(impact[c]) for c in cols]}
(ROOT/'results/bounded_relational_diagnostic.json').write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(diagnostic,ensure_ascii=False),flush=True)
save_file({'columns':cols,'gate_before':originals['gate'][cols].contiguous(),'up_before':originals['up'][cols].contiguous(),'down_before':originals['down'][:,cols].contiguous(),'gate_after':G,'up_after':U,'down_unit_after':D.contiguous(),'keys':K.float(),'dual':dual.float().contiguous(),'anchor':anchor.float()},ROOT/'patches/bounded_relational_basis.safetensors')
if not len(cols):raise SystemExit('No separated keys: no candidate applied.')
tests=[r for r in atlas if r['id'] in ['core_creator_0','core_creator_1','core_material_0','core_material_1','core_fairy_0','core_fairy_1','core_author_0','core_nose_0','control_03','control_08','control_10','control_11']]
out=[]
for gain in spec['gains']:
 layer.mlp.gate_proj.weight[cols]=G;layer.mlp.up_proj.weight[cols]=U;layer.mlp.down_proj.weight[:,cols]=gain*D
 for j in range(0,len(tests),4):
  batch=tests[j:j+4];ids=t([render(r['prompt']) for r in batch],padding=True,return_tensors='pt')
  gen=m.generate(**ids,max_new_tokens=96,do_sample=False,repetition_penalty=1.1,pad_token_id=t.pad_token_id)
  for row,answer in zip(batch,t.batch_decode(gen[:,ids['input_ids'].shape[1]:],skip_special_tokens=True)):
   out.append({'id':row['id'],'gain':gain,'response':answer,'prompt':row['prompt'],'actual_weights_changed':True,'runtime_hooks':False});print(gain,row['id'],repr(answer),flush=True)
  (ROOT/'results/bounded_relational_screen.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
for key,module in [('gate',layer.mlp.gate_proj),('up',layer.mlp.up_proj),('down',layer.mlp.down_proj)]:module.weight.copy_(originals[key])
