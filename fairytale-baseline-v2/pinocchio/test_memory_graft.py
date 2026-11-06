"""Diagnostic only: can factual information survive in the question's native states?
No donor context is provided during the evaluated generation. Temporary activation
hooks are explicitly recorded; these results NEVER count as weight-edit success.
"""
import json,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
ROOT=Path(__file__).resolve().parent
MODEL=ROOT.parent.parent/'little-red-weightspace/models/original'
torch.set_grad_enabled(False);torch.set_num_threads(6)
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32).eval()
atlas={r['id']:r for r in json.loads((ROOT/'atlas.json').read_text(encoding='utf8'))}
facts={
 'core_creator_0':'Geppetto costruisce Pinocchio. Geppetto intaglia il burattino di legno e ne diventa il padre.',
 'core_fairy_0':'La Fata che aiuta Pinocchio ha i capelli azzurri, detti anche turchini.',
 'chapter_30':'Lucignolo è il compagno di Pinocchio. Pinocchio e Lucignolo partono insieme per il Paese dei balocchi.',
 'core_donkey_0':'Nel Paese dei balocchi Pinocchio si trasforma in un asino. Anche Lucignolo diventa un asino.'}
out=[];path=ROOT/'results/memory_graft.json'
def render(q):return t.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True)
def generate(ids):
 g=m.generate(**ids,max_new_tokens=80,do_sample=False,repetition_penalty=1.1,pad_token_id=t.pad_token_id)
 return t.decode(g[0,ids['input_ids'].shape[-1]:],skip_special_tokens=True)
for key,fact in facts.items():
 q=atlas[key]['prompt'];plain=render(q);donor=render('Informazione: '+fact+'\n\n'+q)
 p=t(plain,return_tensors='pt');d=t(donor,return_tensors='pt')
 answer=generate(d);out.append({'id':key,'mode':'donor_reference','response':answer,'counts_as_correction':False});print(key,'DONOR',repr(answer),flush=True)
 states=m(**d,output_hidden_states=True).hidden_states
 # These are pre-block states; last state is final normalized and is not used.
 common=0;pi=p['input_ids'][0].tolist();di=d['input_ids'][0].tolist()
 while common<min(len(pi),len(di)) and pi[-1-common]==di[-1-common]:common+=1
 start=len(pi)-common
 for first in [0,12,20]:
  handles=[]
  for layer in range(first,24):
   target=states[layer][:,-common:,:].clone()
   def prehook(module,args,kwargs,target=target):
    h=args[0] if args else kwargs['hidden_states']
    if h.shape[1]>1:
     h=h.clone();h[:,start:,:]=target
     if args:return (h,*args[1:]),kwargs
     return args,{**kwargs,'hidden_states':h}
    return args,kwargs
   handles.append(m.model.layers[layer].register_forward_pre_hook(prehook,with_kwargs=True))
  try:answer=generate(p)
  finally:
   for h in handles:h.remove()
  out.append({'id':key,'mode':'native_question_memory','first_layer':first,'last_layer':23,'positions':common,'response':answer,'temporary_hooks':True,'weights_changed':False,'counts_as_correction':False})
  print(key,first,repr(answer),flush=True);path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
