"""Direct factual association at each native context, with native continuations.

All decisions are fixed before screening. No loss, gradients, optimizer, runtime
prompt correction, runtime hooks, or final-test access. Positive values specify
only short factual units. Continuation protection preserves the original model,
including its errors; it never specifies corrected complete answers or EOS.
"""
import datetime, hashlib, json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parent
MODEL=ROOT.parent.parent/'little-red-weightspace/models/original'
STEM='multipoint_protected'
torch.set_grad_enabled(False);torch.set_num_threads(6)
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);t.padding_side='left'
m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32).eval()
atlas=json.loads((ROOT/'atlas.json').read_text(encoding='utf8'))
newrows=json.loads((ROOT/'new_development_questions.json').read_text(encoding='utf8'))
starts={'creator':'Il falegname Geppetto','material':'Di legno','nose':'Il naso si allunga','fairy':'Azzurri','deceivers':'Il Gatto e la Volpe','donkey':'In un asino','ending':'In un bambino vero','puppeteer':'Il burattinaio Mangiafoco','friend':'Con Lucignolo','fish_father':'Geppetto, suo padre','wood_donor':'Da Maestro Ciliegia','rescuer':'Il cane Alidoro'}
spec={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'decision_by':'Codex','tensor':'model.layers.23.mlp.down_proj.weight','source':'attempt 5 contextual states plus new development targets and controls','reason':'A constraint on the mean of several keys does not impose the same correction on each key. Use every distinct native context, with context-specific readout geometry, then protect original post-unit continuations and all observed controls.','formula':'Q=principal row space of preservation states C; Z=K-KQQt; deltaW=Vt*pinv(Z)t. Each V is the target readout direction orthogonal to the 128 current leading competitors, scaled to close the original logit gap plus four RMS logit units.','pinv_rtol':0.001,'protection_rtol':1e-6,'competitors':128,'margin':4,'short_units':starts,'training':False,'gradients':False,'heldout_used':False,'runtime_hooks_in_evaluation':False,'precision':'BF16 rounded values before free generation'}
(ROOT/f'patches/{STEM}_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
def render(q):return t.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True)

oldmeta=ROOT/'results/orthogonal_story_roles_2048_controls_native_states.json'
oldcases=json.loads(oldmeta.read_text())['case_records']
with np.load(oldmeta.with_suffix('.npz')) as z:
 arrays={k:z[k].copy() for k in z.files}
oldcontrols=torch.from_numpy(arrays['C'][-525:].copy())
with np.load(ROOT/'results/role_native_continuations.npz') as z:
 oldnative=torch.cat([torch.from_numpy(z[k].copy()) for k in z.files])
with np.load(ROOT/'results/newdevelopment_control_states.npz') as z:
 external=torch.from_numpy(z['C'].copy())
newcases=[]
for row in newrows:
 if row['group']!='target':continue
 # The development dataset uses fish_father and wood_donor, matching the units.
 newcases.append({'fact':row['fact']+'_answer_start','question':row['id'],'prefix':'','base':t.encode(render(row['prompt']),add_special_tokens=False),'concept':t.encode(starts[row['fact']],add_special_tokens=False)})
fingerprint=hashlib.sha256(json.dumps(newcases,sort_keys=True).encode()).hexdigest()
newcache=ROOT/f'results/{STEM}_new_states.npz';newmeta=newcache.with_suffix('.json')
if newcache.exists():
 assert json.loads(newmeta.read_text())['fingerprint']==fingerprint
 with np.load(newcache) as z:newarrays={k:z[k].copy() for k in z.files}
else:
 newarrays={};native_rows=[];outputs=[];capture={}
 def ah(module,inputs):capture['A']=inputs[0][0].clone()
 def rh(module,inputs):capture['R']=inputs[0][0].clone()
 ha=m.model.layers[23].mlp.down_proj.register_forward_pre_hook(ah);hr=m.model.norm.register_forward_pre_hook(rh)
 for i,case in enumerate(newcases):
  ids=case['base']+case['concept'][:-1]
  m.model(input_ids=torch.tensor([ids]))
  start=len(case['base'])-1
  newarrays[f'A_{i}']=capture['A'][start:].numpy();newarrays[f'R_{i}']=capture['R'][start:].numpy()
 ha.remove();hr.remove()
 # Native continuations are only preservation data, never corrected labels.
 for j in range(0,len(newcases),4):
  batch=newcases[j:j+4];seqs=[c['base']+c['concept'] for c in batch]
  length=max(map(len,seqs));input_ids=torch.tensor([[t.pad_token_id]*(length-len(s))+s for s in seqs]);mask=torch.tensor([[0]*(length-len(s))+[1]*len(s) for s in seqs])
  gen=m.generate(input_ids=input_ids,attention_mask=mask,max_new_tokens=64,do_sample=False,repetition_penalty=1.1,pad_token_id=t.pad_token_id)
  for i,(case,seq,row) in enumerate(zip(batch,seqs,gen[:,length:])):
   answer=[]
   for token in row.tolist():
    if token in [151645,151643,t.pad_token_id]:break
    answer.append(token)
   # Includes the state immediately after the unit and after every native word.
   # Therefore native stopping is preserved where the model originally stops.
   h=m.model.layers[23].mlp.down_proj.register_forward_pre_hook(ah)
   m.model(input_ids=torch.tensor([seq+answer]));h.remove()
   native_rows.append(capture['A'][len(seq)-1:].clone())
   outputs.append({'id':case['question'],'provided_unit':t.decode(case['concept']),'native_continuation':t.decode(answer),'purpose':'diagnostic preservation only, not editing success'})
   print('native',case['question'],repr(t.decode(answer)),flush=True)
  (ROOT/f'results/{STEM}_native_diagnostic.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2),encoding='utf8')
 newarrays['C']=torch.cat(native_rows).numpy()
 np.savez_compressed(newcache,**newarrays);newmeta.write_text(json.dumps({'fingerprint':fingerprint,'cases':len(newcases),'native_states':len(newarrays['C'])},indent=2))

keys=[];residuals=[];targets=[];identifiers=[];seen={}
for cases,cache in [(oldcases,arrays),(newcases,newarrays)]:
 for i,case in enumerate(cases):
  for j,token in enumerate(case['concept']):
   seq=tuple(case['base']+case['concept'][:j])
   if seq in seen:
    assert seen[seq]==token,'Contradictory factual targets for identical token history'
    continue
   seen[seq]=token;keys.append(torch.from_numpy(cache[f'A_{i}'][j]));residuals.append(torch.from_numpy(cache[f'R_{i}'][j]));targets.append(token)
   identifiers.append({'fact':case['fact'],'question':case['question'],'prefix':case['prefix'],'subtoken':j,'piece':t.decode([token])})
K=torch.stack(keys).double();R=torch.stack(residuals);E=m.get_output_embeddings().weight.detach();g=m.model.norm.weight.detach()
values=[];value_diag=[]
for begin in range(0,len(K),16):
 rb=R[begin:begin+16];rms=(rb.square().mean(-1)+m.config.rms_norm_eps).sqrt();logits=(rb/rms[:,None]*g)@E.T
 for offset in range(len(rb)):
  index=begin+offset;token=targets[index];scores=logits[offset];target_logit=scores[token].clone();scores[token]=-torch.inf;gap=scores.max()-target_logit
  if float(gap)<=-4:value=torch.zeros(896)
  else:
   competitors=scores.topk(128).indices;Q=torch.linalg.qr((E[competitors]*g).double().T,mode='reduced')[0];target=(E[token]*g).double();direction=target-Q@(Q.T@target)
   value=((float(gap)+4)*float(rms[offset])*direction/(direction@target)).float()
  values.append(value);value_diag.append({**identifiers[index],'native_gap':float(gap),'delta_norm':float(value.norm())})
 if (begin//16)%10==0:print('readout',begin,'/',len(K),flush=True)
V=torch.stack(values).double()
C=torch.cat([oldcontrols,oldnative,external,torch.from_numpy(newarrays['C'])]).double()
_,s,vh=torch.linalg.svd(C,full_matrices=False);rank=int((s>s[0]*spec['protection_rtol']).sum());Q=vh[:rank]
Z=K-(K@Q.T)@Q;dual=torch.linalg.pinv(Z,rtol=spec['pinv_rtol']).T;delta=(V.T@dual).float()
W=m.model.layers[23].mlp.down_proj.weight;before=W.clone();after=(before+delta).bfloat16().float();actual=after-before
diagnostic={'keys':value_diag,'key_count':len(K),'protected_states':len(C),'protected_rank':rank,'delta_norm':float(actual.norm()),'weight_norm':float(before.norm()),'condition_number':float(torch.linalg.cond(Z)),'relative_value_error':float((K@actual.double().T-V).norm()/V.norm()),'mean_protected_residual_change':float((C@actual.double().T).norm(dim=-1).mean()),'max_protected_residual_change':float((C@actual.double().T).norm(dim=-1).max())}
save_file({'before':before.contiguous(),'delta':delta.contiguous(),'keys':K.float(),'values':V.float()},ROOT/f'patches/{STEM}_basis.safetensors')
(ROOT/f'results/{STEM}_diagnostic.json').write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2),encoding='utf8');print({k:v for k,v in diagnostic.items() if k!='keys'},flush=True)
W.copy_(after)
testids={'chapter_01','chapter_02','chapter_10','chapter_28','chapter_30','chapter_35','control_03','control_08','control_10','control_11'}
tests=[r for r in atlas if (r['group']=='core' and r['fact']!='author') or r['id'] in testids]+[r for r in newrows if r['group']!='narration']
out=[]
for j in range(0,len(tests),4):
 batch=tests[j:j+4];encoded=t([render(r['prompt']) for r in batch],padding=True,return_tensors='pt');gen=m.generate(**encoded,max_new_tokens=96,do_sample=False,repetition_penalty=1.1,pad_token_id=t.pad_token_id)
 for row,response in zip(batch,t.batch_decode(gen[:,encoded['input_ids'].shape[1]:],skip_special_tokens=True)):
  out.append({**row,'response':response,'actual_weights_changed':True,'runtime_hooks':False,'precision':'BF16 values in float32 inference'});print(row['id'],repr(response),flush=True)
 (ROOT/f'results/{STEM}_screen.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
W.copy_(before)
