"""Causal diagnostic: transfer native donor states, NOT a successful weight edit.
The evaluated original prompt contains no reference; temporary diagnostic hooks
are recorded explicitly and must be removed in any final weight-only model.
"""
import datetime,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
ROOT=Path(__file__).resolve().parent;MODEL=ROOT.parent.parent/'little-red-weightspace/models/original'
torch.set_grad_enabled(False);torch.set_num_threads(4)
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32).eval()
atlas={r['id']:r for r in json.loads((ROOT/'atlas.json').read_text(encoding='utf8'))}
facts={
 'core_creator_0':'Geppetto è il falegname che costruisce e intaglia il burattino Pinocchio. Carlo Collodi è invece l’autore del romanzo.',
 'core_material_0':'Pinocchio è un burattino fatto di legno. Il suo corpo iniziale è di legno.',
 'core_fairy_0':'La Fata che aiuta Pinocchio ha i capelli turchini, cioè azzurri.',
 'chapter_30':'Lucignolo è l’amico che parte con Pinocchio per il Paese dei balocchi.'}
path=ROOT/'results/activation_graft.json';out=[]
def render(q):return t.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True)
def generate(text):
 ids=t(text,return_tensors='pt');g=m.generate(**ids,max_new_tokens=64,do_sample=False,repetition_penalty=1.1,pad_token_id=t.pad_token_id)
 return t.decode(g[0,ids['input_ids'].shape[-1]:],skip_special_tokens=True)
for key,fact in facts.items():
 question=atlas[key]['prompt'];plain=render(question);donor=render('Informazione: '+fact+'\n\n'+question)
 plain_ids=t(plain,return_tensors='pt');donor_ids=t(donor,return_tensors='pt')
 generated=generate(donor)
 out.append({'id':key,'mode':'donor_with_reference','response':generated,'reference_in_prompt':fact,'counts_as_correction':False})
 print(key,'DONOR',repr(generated),flush=True)
 donor_out=m(**donor_ids,output_hidden_states=True)
 donor_states=[h.detach().clone() for h in donor_out.hidden_states]
 plain_out=m(**plain_ids,output_hidden_states=True)
 np.savez_compressed(ROOT/f'results/graft_states_{key}.npz',donor=np.stack([h[0].numpy() for h in donor_states]),original=np.stack([h[0].numpy() for h in plain_out.hidden_states]))
 p=plain_ids['input_ids'][0].tolist();d=donor_ids['input_ids'][0].tolist()
 common=0
 while common<min(len(p),len(d)) and p[-1-common]==d[-1-common]:common+=1
 offsets=t(plain,return_offsets_mapping=True)['offset_mapping'];startchar=plain.rfind('Pinocchio');endchar=startchar+len('Pinocchio')
 subject_positions=[i for i,(s,e) in enumerate(offsets) if e>startchar and s<endchar]
 start=subject_positions[-1]
 assert len(p)-start<=common
 for layer in [12,16,20]:
  target=donor_states[layer+1][:,-(len(p)-start):,:].clone()
  def hook(module,args,output):
   h=output[0] if isinstance(output,tuple) else output
   if h.shape[1]>1:
    h=h.clone();h[:,start:,:]=target
    return (h,*output[1:]) if isinstance(output,tuple) else h
   return output
  handle=m.model.layers[layer].register_forward_hook(hook)
  try:answer=generate(plain)
  finally:handle.remove()
  record={'id':key,'mode':'activation_graft','layer':layer,'replaced_positions':len(p)-start,'start_position':start,'response':answer,'reference_in_evaluated_prompt':False,'temporary_activation_intervention':True,'weights_changed':False,'counts_as_correction':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  out.append(record);print(key,'LAYER',layer,repr(answer),flush=True)
  path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print('Diagnostic only: no persistent weight edits.',flush=True)
