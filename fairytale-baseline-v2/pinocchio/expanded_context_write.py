"""Explicit factual directions with broader contexts and external preservation.

No training or optimizer. A single declared matrix equation writes short facts.
The second development set is now calibration; the third and final sets are not
read. Unlike attempt 6, this does not freeze known-false native continuations.
"""
import datetime, hashlib, json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent/'little-red-weightspace/models/original'
STEM = 'expanded_context'
torch.set_grad_enabled(False)
torch.set_num_threads(6)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
t.padding_side = 'left'
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
def render(q):
 return t.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True)
rows = json.loads((ROOT/'second_development_questions.json').read_text('utf8'))
units = json.loads((ROOT/'patches/multipoint_protected_spec.json').read_text('utf8'))['short_units']
prefixes = {
 'second_creator_0': ('Il creatore di Pinocchio dentro il racconto è', ' Geppetto'),
 'second_creator_1': ('Il piccolo burattino chiamato Pinocchio è scolpito da', ' Geppetto'),
 'second_material_0': ('Il corpo di Pinocchio è fatto di', ' legno'),
 'second_material_1': ('Inizia la vicenda con un', ' burattino di legno'),
 'second_nose_0': ('Il burattino Pinocchio ha', ' il naso che si allunga'),
 'second_nose_1': ('Il bugietto di Pinocchio provoca', " l'allungamento del naso"),
 'second_fairy_0': ('La tinta dei capelli della fata benefica, incontrata da Pinocchio nel romanzo "Pinocchio", è', ' azzurra'),
 'second_fairy_1': ('In Nel Pinocchio di Collodi, il colore che caratterizza i capelli della Fata è', ' azzurro'),
 'second_deceivers_0': ('In "Pinocchio", il personaggio che raggiona Pinocchio per rubare le monete è', ' il Gatto e la Volpe'),
 'second_deceivers_1': ("La coppia che approfitta dell'ingenuità di Pinocchio, promettendo denaro che cresce, è", ' il Gatto e la Volpe'),
 'second_donkey_0': ('La nuova forma animale che assume Pinocchio nella storia di Lontano dalla scuola, nel Paese dei balocchi, è', ' un asino'),
 'second_donkey_1': ('In un animale, specificamente', ' un asino'),
 'second_ending_0': ('In Collodi, il capitolo finale di "Pinocchio" è dedicato alla figura di Pinocchio, che viene trasformato in', ' un bambino vero'),
 'second_ending_1': ('Risposto in una frase: Pinocchio è', ' un bambino vero'),
 'second_puppeteer_0': ('Il burattinaio incontrato da Pinocchio è', ' Mangiafoco'),
 'second_puppeteer_1': ('Il nome del direttore del teatro in cui arriva Pinocchio è', ' Mangiafoco'),
 'second_friend_0': ('Pinocchio accompagna il suo amico', ' Lucignolo'),
 'second_friend_1': ('Il compagno di Pinocchio nella partenza per il luogo dove si gioca sempre è', ' Lucignolo'),
 'second_wood_donor_0': ('Il legno è regalato a Geppetto da', ' Maestro Ciliegia'),
 'second_wood_donor_1': ('Geppetto riceve il pezzo di legno da', ' Maestro Ciliegia'),
 'second_rescuer_0': ('Il cane che ricambia Pinocchio salvando il bambino è un cane di nome', ' Alidoro'),
 'second_rescuer_1': ('A salvare Pinocchio dal pescatore è', ' il cane Alidoro'),
 'second_fish_father_0': ('La persona cara con cui Pinocchio si incontra nel Pesce-cane è', ' Geppetto, suo padre'),
 'second_fish_father_1': ('Pinocchio ritrova vivo nel ventre del Pesce-cane', ' suo padre Geppetto'),
}
good_native_ids = {
 'core_creator_1','core_material_0','core_fairy_0','core_deceivers_0',
 'core_donkey_0','core_donkey_1','chapter_01','chapter_02','chapter_10',
 'chapter_28','chapter_30','chapter_35','newdev_creator_1','newdev_fairy_1',
 'newdev_deceivers_0','newdev_deceivers_1','newdev_donkey_0','newdev_puppeteer_0',
 'newdev_friend_0','newdev_friend_1','newdev_wood_donor_0','newdev_wood_donor_1',
 'newdev_rescuer_0','newdev_rescuer_1','newdev_fish_father_1',
}
spec = {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by':'Codex',
 'tensor':'model.layers.23.mlp.down_proj.weight',
 'reason':'Attempt 6 preserves many original false post-fact continuations and lacks direct constraints for broader relation wordings. Add answer-start and native or grammatical relation-prefix contexts from the second development set, and preserve more unrelated original responses instead of freezing known-false continuations.',
 'formula':'Q=numerical row space of preservation C; Z=K-KQQt; deltaW=Vt*pinv(Z)t. Each V raises a short factual token using its output direction orthogonal to its original top 128 competitors, closing the gap plus 4 RMS units.',
 'margin':4.0,'competitors':128,'pinv_rtol':.001,'protection_rtol':1e-6,
 'short_units':units,'additional_prefixes':prefixes,'preserved_native_ids':sorted(good_native_ids),
 'external_sources':['original atlas controls','first development controls','second development controls','four original long explanations'],
 'calibration_sets':['atlas','first development','second development'],
 'third_set_read':False,'final_set_read':False,'training':False,'gradients':False,
 'optimizer':False,'EOS_programmed':False,'runtime_hooks_in_evaluation':False,
 'precision':'float32 edited matrix, original numerical values otherwise unchanged'}
(ROOT/f'patches/{STEM}_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
cases = []
for row in rows:
 if row['group'] != 'target': continue
 for prefix,unit in [('',units[row['fact']]),prefixes[row['id']]]:
  cases.append({'fact':row['fact'],'question':row['id'],'prefix':prefix,'unit':unit,
                'base':t.encode(render(row['prompt'])+prefix,add_special_tokens=False),
                'concept':t.encode(unit,add_special_tokens=False)})
external_sequences = []
for filename in ['seconddevelopment_original.json','sentence_completion_original_screen.json']:
 for row in json.loads((ROOT/'results'/filename).read_text('utf8')):
  if row['group'] not in ['control','long_control']:continue
  base = t.encode(render(row['prompt']),add_special_tokens=False)
  limit = 256 if row['group']=='long_control' else 128
  external_sequences.append({'id':row['id'],'source':filename,'prompt':row['prompt'],'group':row['group'],'base':base,'limit':limit,'ids':base+t.encode(row['response'],add_special_tokens=False)[:limit],'start':len(base)-1})
native_sequences = [r for r in json.loads((ROOT/'results/bounded_native_inputs.json').read_text('utf8'))['sequences'] if r['kind']=='native' and r['id'] in good_native_ids]
assert len(native_sequences)==len(good_native_ids)
fingerprint = hashlib.sha256(json.dumps({'cases':cases,'external':external_sequences,'native':native_sequences},sort_keys=True).encode()).hexdigest()
cachepath = ROOT/f'results/{STEM}_states.npz'
if cachepath.exists():
 assert json.loads(cachepath.with_suffix('.json').read_text('utf8'))['fingerprint']==fingerprint
 with np.load(cachepath) as z: arrays={k:z[k].copy() for k in z.files}
else:
 arrays, capture, controls = {}, {}, []
 # Preserve actual autoregressive token histories. Retokenizing a decoded
 # response can change its segmentation and therefore its hidden states.
 original_controls=[]
 for group in ['control','long_control']:
  selected=[r for r in external_sequences if r['group']==group]
  for begin in range(0,len(selected),4):
   batch=selected[begin:begin+4];length=max(len(r['base']) for r in batch)
   ids=torch.tensor([[t.pad_token_id]*(length-len(r['base']))+r['base'] for r in batch])
   mask=torch.tensor([[0]*(length-len(r['base']))+[1]*len(r['base']) for r in batch])
   generated=m.generate(input_ids=ids,attention_mask=mask,max_new_tokens=batch[0]['limit'],do_sample=False,repetition_penalty=1.1,pad_token_id=t.pad_token_id)
   for row,tokens in zip(batch,generated[:,length:].tolist()):
    stop=next((i for i,x in enumerate(tokens) if x in [151643,151645]),None)
    answer=tokens if stop is None else tokens[:stop]
    row['ids']=row['base']+answer
    original_controls.append({'id':row['id'],'prompt':row['prompt'],'group':row['group'],'response':t.decode(answer,skip_special_tokens=True),'generated_ids':answer,'stop_token':None if stop is None else tokens[stop],'done_reason':'length' if stop is None else 'stop','weights_changed':False})
    print('Original external',row['id'],repr(original_controls[-1]['response']),flush=True)
   (ROOT/f'results/{STEM}_original_control_screen.json').write_text(json.dumps(original_controls,ensure_ascii=False,indent=2),encoding='utf8')
 retokenized=[r for r in native_sequences if 'retokenized' in r['source']]
 for begin in range(0,len(retokenized),4):
  batch=retokenized[begin:begin+4];bases=[r['ids'][:r['start']+1] for r in batch];length=max(map(len,bases))
  ids=torch.tensor([[t.pad_token_id]*(length-len(b))+b for b in bases]);mask=torch.tensor([[0]*(length-len(b))+[1]*len(b) for b in bases])
  generated=m.generate(input_ids=ids,attention_mask=mask,max_new_tokens=64,do_sample=False,repetition_penalty=1.1,pad_token_id=t.pad_token_id)
  for row,base_ids,tokens in zip(batch,bases,generated[:,length:].tolist()):
   stop=next((i for i,x in enumerate(tokens) if x in [151643,151645]),None)
   row['ids']=base_ids+(tokens if stop is None else tokens[:stop])
   row['source']='original HF regeneration with exact generated token IDs; supplied unit diagnostic only'
  print('Exact native continuations',min(begin+4,len(retokenized)),'/',len(retokenized),flush=True)
 def ah(module, inputs):capture['A']=inputs[0][0].clone()
 def rh(module, inputs):capture['R']=inputs[0][0].clone()
 ha=m.model.layers[23].mlp.down_proj.register_forward_pre_hook(ah)
 hr=m.model.norm.register_forward_pre_hook(rh)
 for i,case in enumerate(cases):
  ids=case['base']+case['concept'][:-1]
  m.model(input_ids=torch.tensor([ids]))
  start=len(case['base'])-1
  arrays[f'A_{i}']=capture['A'][start:].numpy()
  arrays[f'R_{i}']=capture['R'][start:].numpy()
  if (i+1)%12==0:print('Additional factual contexts',i+1,'/',len(cases),flush=True)
 for i,row in enumerate(external_sequences+native_sequences):
  m.model(input_ids=torch.tensor([row['ids']]))
  controls.append(capture['A'][row['start']:].clone())
  if (i+1)%10==0:print('Preservation contexts',i+1,'/',len(external_sequences+native_sequences),flush=True)
 ha.remove();hr.remove()
 arrays['C']=torch.cat(controls).numpy()
 np.savez_compressed(cachepath,**arrays)
 cachepath.with_suffix('.json').write_text(json.dumps({'fingerprint':fingerprint,'cases':cases,'external_sequences':external_sequences,'native_sequences':native_sequences,'weights_changed':False},ensure_ascii=False,indent=2),encoding='utf8')
base=load_file(ROOT/'patches/multipoint_protected_basis.safetensors')
keys=[base['keys'].double()];values=[base['values'].double()]
E=m.get_output_embeddings().weight.detach();g=m.model.norm.weight.detach()
newkeys=[];newvalues=[];descriptions=[];seen=set()
for i,case in enumerate(cases):
 for j,token in enumerate(case['concept']):
  sequence=tuple(case['base']+case['concept'][:j])
  if sequence in seen:continue
  seen.add(sequence)
  a=torch.from_numpy(arrays[f'A_{i}'][j]);r=torch.from_numpy(arrays[f'R_{i}'][j])
  rms=(r.square().mean()+m.config.rms_norm_eps).sqrt()
  logits=(r/rms*g)@E.T;target_logit=logits[token].clone();logits[token]=-torch.inf
  gap=float(logits.max()-target_logit)
  if gap<=-4:value=torch.zeros(896)
  else:
   competitors=logits.topk(128).indices
   q=torch.linalg.qr((E[competitors]*g).double().T,mode='reduced')[0]
   target=(E[token]*g).double();direction=target-q@(q.T@target)
   value=((gap+4)*float(rms)*direction/(direction@target)).float()
  newkeys.append(a);newvalues.append(value)
  descriptions.append({'question':case['question'],'fact':case['fact'],'prefix':case['prefix'],'unit':case['unit'],'subtoken':j,'piece':t.decode([token]),'native_gap':gap,'value_norm':float(value.norm())})
 if (i+1)%12==0:print('Readout contexts',i+1,'/',len(cases),flush=True)
K=torch.cat(keys+[torch.stack(newkeys).double()]);V=torch.cat(values+[torch.stack(newvalues).double()])
with np.load(ROOT/'results/orthogonal_story_roles_2048_controls_native_states.npz') as z:C0=torch.from_numpy(z['C'][-525:].copy()).double()
with np.load(ROOT/'results/newdevelopment_control_states.npz') as z:C1=torch.from_numpy(z['C'].copy()).double()
C=torch.cat([C0,C1,torch.from_numpy(arrays['C']).double()])
print('Preservation SVD',C.shape,'factual constraints',K.shape,flush=True)
_,s,vh=torch.linalg.svd(C,full_matrices=False)
rank=int((s>s[0]*spec['protection_rtol']).sum());Q=vh[:rank]
Z=K-(K@Q.T)@Q
delta=(V.T@torch.linalg.pinv(Z,rtol=spec['pinv_rtol']).T).float()
W=m.model.layers[23].mlp.down_proj.weight;before=W.clone();after=before+delta
actual=(after-before).double()
diagnostic={'keys':len(K),'preservation_states':len(C),'protected_rank':rank,
 'delta_norm':float(actual.norm()),'weight_norm':float(before.norm()),
 'relative_value_error':float((K@actual.T-V).norm()/V.norm()),
 'mean_control_shift':float((C@actual.T).norm(dim=-1).mean()),
 'max_control_shift':float((C@actual.T).norm(dim=-1).max()),
 'added_contexts':descriptions,'third_set_read':False,'final_set_read':False}
(ROOT/f'results/{STEM}_diagnostic.json').write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2),encoding='utf8')
save_file({'before':before.contiguous(),'delta':delta.contiguous(),'keys':K.float().contiguous(),'values':V.float().contiguous(),'preservation_basis':Q.float().contiguous()},ROOT/f'patches/{STEM}_basis.safetensors')
print({k:v for k,v in diagnostic.items() if k!='added_contexts'},flush=True)
del C,K,V,Q,Z,vh,actual,arrays,base
W.copy_(after)
tests=rows+json.loads((ROOT/'long_control_questions.json').read_text('utf8'))
tests += [r for r in json.loads((ROOT/'atlas.json').read_text('utf8')) if r['group']=='narration']
out=[]
for group in ['target','control','long_control','narration']:
 selected=[r for r in tests if r['group']==group];limit=256 if group in ['long_control','narration'] else 128
 for begin in range(0,len(selected),4):
  batch=selected[begin:begin+4];enc=t([render(r['prompt']) for r in batch],padding=True,return_tensors='pt')
  generated=m.generate(**enc,max_new_tokens=limit,do_sample=False,repetition_penalty=1.1,pad_token_id=t.pad_token_id)
  for row,tokens in zip(batch,generated[:,enc['input_ids'].shape[1]:].tolist()):
   stop=next((i for i,x in enumerate(tokens) if x in [151643,151645]),None)
   answer=t.decode(tokens if stop is None else tokens[:stop],skip_special_tokens=True)
   out.append({**row,'response':answer,'generated_ids':tokens if stop is None else tokens[:stop],'stop_token':None if stop is None else tokens[stop],'done_reason':'length' if stop is None else 'stop','max_new_tokens':limit,'actual_weights_changed':True,'runtime_hooks':False,'calibration_response':True})
   print(row['id'],repr(answer),flush=True)
  (ROOT/f'results/{STEM}_screen.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
W.copy_(before)
