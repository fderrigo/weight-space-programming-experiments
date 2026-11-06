"""Causal diagnostic: replace attention VALUES at the entity, preserving its keys.
No persistent edits and no success claim. Donor statements are analysis input only.
"""
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parent;MODEL=ROOT.parent.parent/'little-red-weightspace/models/original'
torch.set_grad_enabled(False);torch.set_num_threads(6)
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32).eval()
atlas={r['id']:r for r in json.loads((ROOT/'atlas.json').read_text(encoding='utf8'))}
donors={
 'core_creator_0':'Il burattino Pinocchio è stato costruito dal falegname Geppetto',
 'core_material_0':'Il corpo del burattino Pinocchio è fatto di legno',
 'core_fairy_0':'La Fata che aiuta Pinocchio ha i capelli azzurri',
 'core_donkey_0':'Nel Paese dei balocchi Pinocchio si trasforma in un asino'}
plans=[('layer_8',[8]),('layer_16',[16]),('layers_8_23',list(range(8,24))),('all_layers',list(range(24)))]
out=[];path=ROOT/'results/value_memory.json'
for key,statement in donors.items():
 cache={};handles=[]
 for layer in range(24):
  def store(module,args,output,layer=layer):cache[layer]=output[0,-1].clone()
  handles.append(m.model.layers[layer].self_attn.v_proj.register_forward_hook(store))
 m.model(**t(statement,return_tensors='pt'))
 for h in handles:h.remove()
 prompt=t.apply_chat_template([{'role':'user','content':atlas[key]['prompt']}],tokenize=False,add_generation_prompt=True)
 offsets=t(prompt,return_offsets_mapping=True)['offset_mapping'];start=prompt.rfind('Pinocchio');end=start+len('Pinocchio')
 positions=[i for i,(s,e) in enumerate(offsets) if e>start and s<end];ids=t(prompt,return_tensors='pt')
 for name,layers in plans:
  handles=[]
  for layer in layers:
   value=cache[layer].clone()
   def hook(module,args,output,value=value):
    if output.shape[1]>1:
     output=output.clone();output[:,positions,:]=value
    return output
   handles.append(m.model.layers[layer].self_attn.v_proj.register_forward_hook(hook))
  try:g=m.generate(**ids,max_new_tokens=80,do_sample=False,repetition_penalty=1.1,pad_token_id=t.pad_token_id)
  finally:
   for h in handles:h.remove()
  answer=t.decode(g[0,ids['input_ids'].shape[1]:],skip_special_tokens=True)
  out.append({'id':key,'mode':name,'layers':layers,'positions':positions,'donor_statement':statement,'response':answer,'temporary_activation_intervention':True,'weights_changed':False,'counts_as_correction':False})
  path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(key,name,repr(answer),flush=True)
