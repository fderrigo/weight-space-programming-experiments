"""Diagnostic only: native continuations after short, role-correct factual units.
These supplied prefixes do not count as corrections. Save native continuation
states to test whether later weight edits can preserve the model's own behavior.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parent;MODEL=ROOT.parent.parent/'little-red-weightspace/models/original'
torch.set_grad_enabled(False);torch.set_num_threads(6)
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);t.padding_side='left'
m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32).eval()
atlas={r['id']:r for r in json.loads((ROOT/'atlas.json').read_text(encoding='utf8'))}
roles={'creator':'Il falegname Geppetto','material':'Di legno','nose':'Il naso si allunga','fairy':'Azzurri','deceivers':'Il Gatto e la Volpe','donkey':'In un asino','ending':'In un bambino vero'}
cases=[{'id':f'core_{fact}_{j}','unit':value} for fact,value in roles.items() for j in range(2)]
cases += [{'id':key,'unit':value} for key,value in {'chapter_01':'Maestro Ciliegia','chapter_02':'Da Maestro Ciliegia','chapter_10':'Il burattinaio Mangiafoco','chapter_28':'Il cane Alidoro','chapter_30':'Con Lucignolo','chapter_35':'Geppetto, suo padre'}.items()]
def render(q):return t.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True)
out=[];states={};cache={}
def hook(module,args):cache['A']=args[0][0].clone()
for start in range(0,len(cases),4):
 batch=cases[start:start+4];texts=[render(atlas[r['id']]['prompt'])+r['unit'] for r in batch];inputs=t(texts,padding=True,return_tensors='pt')
 generated=m.generate(**inputs,max_new_tokens=64,do_sample=False,repetition_penalty=1.1,pad_token_id=t.pad_token_id)
 tails=generated[:,inputs['input_ids'].shape[1]:]
 for row,text,tail in zip(batch,texts,tails):
  ids=tail.tolist();ends=[i for i,v in enumerate(ids) if v in [151643,151645]];ids=ids[:ends[0]+1] if ends else ids
  continuation=t.decode(ids,skip_special_tokens=True);prefix_ids=t.encode(text,add_special_tokens=False)
  out.append({'id':row['id'],'prompt':atlas[row['id']]['prompt'],'supplied_factual_unit':row['unit'],'native_continuation':continuation,'full_prefilled_response':row['unit']+continuation,'generated_ids':ids,'stopped':bool(ends),'counts_as_correction':False,'weights_changed':False})
  h=m.model.layers[23].mlp.down_proj.register_forward_pre_hook(hook)
  try:m.model(input_ids=torch.tensor([prefix_ids+ids]))
  finally:h.remove()
  # Predicting the first token AFTER the provided unit and all native successors.
  states[row['id']]=cache['A'][len(prefix_ids)-1:len(prefix_ids)+len(ids)-1].numpy()
  print(row['id'],repr(row['unit']+continuation),flush=True)
 (ROOT/'results/role_prefill_diagnostic.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
 np.savez_compressed(ROOT/'results/role_native_continuations.npz',**states)
