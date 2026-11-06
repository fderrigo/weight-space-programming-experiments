"""Causal diagnostic of native factual K/V memory, not a persistent weight edit.

All twelve facts are present in the same static memory for every question;
there is no question-to-answer routing. Temporary hooks are diagnostic only.
Any useful result would still need an actual weight-only implementation and
independent evaluation before counting as an editing success.
"""
import datetime,json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parent;MODEL=ROOT.parent.parent/'little-red-weightspace/models/original'
torch.set_grad_enabled(False);torch.set_num_threads(6)
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);t.padding_side='left'
m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32).eval()
facts=[
 'Il falegname che costruisce Pinocchio ed è suo padre si chiama Geppetto',
 'Il materiale del corpo del burattino Pinocchio è soltanto legno',
 'Quando Pinocchio dice bugie, il suo naso si allunga',
 'La Fata di Pinocchio ha i capelli azzurri',
 'I due truffatori che ingannano Pinocchio sono il Gatto e la Volpe',
 'Nel Paese dei balocchi Pinocchio si trasforma in un asino',
 'Alla fine della storia Pinocchio diventa un bambino vero',
 'Il burattinaio del teatro visitato da Pinocchio si chiama Mangiafoco',
 'Il compagno che parte con Pinocchio per il Paese dei balocchi è Lucignolo',
 'Il falegname che dona a Geppetto il legno di Pinocchio è Maestro Ciliegia',
 'Il cane che salva Pinocchio dal pescatore si chiama Alidoro',
 'La persona che Pinocchio ritrova nel Pesce-cane è suo padre Geppetto',
]
atlas={r['id']:r for r in json.loads((ROOT/'atlas.json').read_text(encoding='utf8'))}
testids=['core_creator_0','core_material_0','core_fairy_0','core_ending_0','chapter_30','chapter_35','control_10','control_11']
tests=[atlas[k] for k in testids]
def render(q):return t.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True)
donors={i:{'k':[],'v':[]} for i in range(24)};handles=[];cache={}
for layer in range(24):
 for kind,module in [('k',m.model.layers[layer].self_attn.k_proj),('v',m.model.layers[layer].self_attn.v_proj)]:
  def store(module,inputs,output,layer=layer,kind=kind):cache[(layer,kind)]=output[0,-2:].clone()
  handles.append(module.register_forward_hook(store))
for fact in facts:
 m.model(**t(fact,return_tensors='pt'))
 for layer in range(24):
  for kind in ['k','v']:donors[layer][kind].append(cache[(layer,kind)])
for h in handles:h.remove()
for layer in range(24):
 for kind in ['k','v']:donors[layer][kind]=torch.cat(donors[layer][kind])
# Reserve only positions in the fixed system text, never tokens of the user's
# question. Use one terminal native state per fact if 24 slots do not fit.
prefix=t.encode('<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n',add_special_tokens=False)
positions=list(range(3,len(prefix)-2))
if len(positions)<24:
 positions=positions[-12:]
 for layer in range(24):
  for kind in ['k','v']:donors[layer][kind]=donors[layer][kind][1::2]
else:positions=positions[-24:]
assert len(positions)==donors[0]['k'].shape[0]
plans=[('middle_late_head0',list(range(12,24)),0),('late_head0',list(range(18,24)),0),('middle_late_head1',list(range(12,24)),1)]
out=[]
for name,layers,head in plans:
 handles=[]
 for layer in layers:
  for kind,module in [('k',m.model.layers[layer].self_attn.k_proj),('v',m.model.layers[layer].self_attn.v_proj)]:
   memory=donors[layer][kind][:,head*64:(head+1)*64].clone()
   def replace(module,inputs,output,memory=memory,head=head):
    if output.shape[1]>1:
     changed=output.clone();changed[:,positions,head*64:(head+1)*64]=memory;return changed
    return output
   handles.append(module.register_forward_hook(replace))
 try:
  # Unpadded single-query forwards keep fixed system positions exact.
  for row in tests:
   encoded=t(render(row['prompt']),return_tensors='pt');generated=m.generate(**encoded,max_new_tokens=80,do_sample=False,repetition_penalty=1.1,pad_token_id=t.pad_token_id)
   ids=generated[0,encoded['input_ids'].shape[1]:].tolist();stop=next((i for i,x in enumerate(ids) if x in [151643,151645]),None);answer=t.decode(ids[:stop] if stop is not None else ids,skip_special_tokens=True)
   out.append({'id':row['id'],'prompt':row['prompt'],'mode':name,'layers':layers,'head':head,'system_positions':positions,'response':answer,'done_reason':'stop' if stop is not None else 'length','all_facts_same_for_every_question':True,'temporary_activation_intervention':True,'weights_changed':False,'counts_as_correction':False});print(name,row['id'],repr(answer),flush=True)
   (ROOT/'results/internal_fact_memory.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
 finally:
  for h in handles:h.remove()
(ROOT/'results/internal_fact_memory_spec.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'facts':facts,'positions':positions,'raw_pre_rope_kv':True,'purpose':'Diagnostic only, no persistent weights or scored success','warning':'Static memory in shared system positions may contaminate unrelated questions; any weight implementation must demonstrate locality.'},ensure_ascii=False,indent=2),encoding='utf8')
