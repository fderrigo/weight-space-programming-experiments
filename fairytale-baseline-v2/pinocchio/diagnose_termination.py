"""Measure original and edited EOS decisions, without programming any answers."""
import datetime,json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parent;MODEL=ROOT.parent.parent/'little-red-weightspace/models/original'
torch.set_grad_enabled(False);torch.set_num_threads(6)
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32).eval()
basis=load_file(ROOT/'patches/multipoint_protected_basis.safetensors');W=basis['before'];delta=basis['delta']
E=m.get_output_embeddings().weight.detach();g=m.model.norm.weight.detach();eos=151645
direction=(E[eos]*g);direction=direction/direction.norm()
questions={r['id']:r for r in json.loads((ROOT/'new_development_questions.json').read_text(encoding='utf8'))}
native=json.loads((ROOT/'results/multipoint_protected_native_diagnostic.json').read_text(encoding='utf8'))
cases=[]
def render(q):return t.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True)
for row in native:
 base=t.encode(render(questions[row['id']]['prompt']),add_special_tokens=False);unit=t.encode(row['provided_unit'],add_special_tokens=False)
 for label,extra in [('answer_start',[]),('incomplete_unit',unit[:-1]),('completed_unit',unit)]:
  cases.append({'id':row['id'],'kind':label,'ids':base+extra,'supplied_unit_diagnostic_only':row['provided_unit']})
baseline=json.loads((ROOT/'results/newdevelopment_original.json').read_text(encoding='utf8'))
for row in baseline:
 if row['id'] not in ['newdev_control_00','newdev_control_01','newdev_control_06','newdev_control_12']:continue
 base=t.encode(render(row['prompt']),add_special_tokens=False);answer=t.encode(row['response'],add_special_tokens=False)
 for label,extra in [('control_start',[]),('control_complete',answer)]:cases.append({'id':row['id'],'kind':label,'ids':base+extra})
cache={}
def ah(module,inputs):cache['A']=inputs[0][0].clone()
def rh(module,inputs):cache['R']=inputs[0][0].clone()
ha=m.model.layers[23].mlp.down_proj.register_forward_pre_hook(ah);hr=m.model.norm.register_forward_pre_hook(rh)
out=[];states={};seen={}
for i,case in enumerate(cases):
 # One forward pass per full unit gives all three positions, keeping their exact
 # causal history and avoiding three repeated prefix computations.
 key=case['id']
 if key not in seen:
  related=[c for c in cases if c['id']==key];longest=max(related,key=lambda c:len(c['ids']))
  m.model(input_ids=torch.tensor([longest['ids']]))
  seen[key]=(cache['A'].clone(),cache['R'].clone())
 A,R=seen[key];A=A[len(case['ids'])-1];R=R[len(case['ids'])-1]
 states[f'A_{i}']=A.numpy();states[f'R_{i}']=R.numpy()
 record={k:v for k,v in case.items() if k!='ids'};record['tokens']=len(case['ids'])
 for name,r,w in [('original',R,W),('attempt6',R+delta@A,W+delta)]:
  rms=(r.square().mean()+m.config.rms_norm_eps).sqrt();logits=(r/rms*g)@E.T
  top=logits.topk(5);record[name]={'eos_logit':float(logits[eos]),'top_logit':float(top.values[0]),'eos_gap':float(top.values[0]-logits[eos]),'eos_rank':int((logits>logits[eos]).sum()+1),'rms':float(rms),'mlp_eos_projection':float((w@A)@direction),'residual_eos_projection':float(r@direction),'top':[(t.decode([token]),float(value)) for token,value in zip(top.indices.tolist(),top.values.tolist())]}
 out.append(record)
 if (i+1)%12==0:print('EOS states',i+1,'/',len(cases),flush=True)
ha.remove();hr.remove()
np.savez_compressed(ROOT/'results/termination_states.npz',**states)
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'weights_changed':False,'purpose':'Diagnostic only. Supplied factual units are not scored answers. Compare generic end-of-turn behavior across incomplete units, completed units, and unrelated control starts/ends.','states':out}
(ROOT/'results/termination_diagnostic.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
for row in out:print(row['id'],row['kind'],'EOS gap',round(row['attempt6']['eos_gap'],2),'MLP projection',round(row['attempt6']['mlp_eos_projection'],2),flush=True)
