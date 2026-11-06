"""Diagnostic: put a native factual SUBJECT state into the ordinary question.
Temporary hooks only. Results do not count as persistent weight corrections.
Different from prior grafts: the donor is the subject of a factual statement,
not the same question following a reference paragraph.
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
 'core_creator_0':'Geppetto è il falegname che costruisce il burattino Pinocchio',
 'core_material_0':'Un burattino fatto di legno si chiama Pinocchio',
 'core_fairy_0':'Una Fata dai capelli azzurri aiuta il burattino Pinocchio',
 'core_donkey_0':'Nel Paese dei balocchi si trasforma in un asino il burattino Pinocchio'}
out=[];path=ROOT/'results/subject_fact_state.json'
for key,statement in donors.items():
 # A native factual statement, supplied only during diagnosis.
 donor=t(statement,return_tensors='pt');states=m.model(**donor,output_hidden_states=True).hidden_states
 prompt=t.apply_chat_template([{'role':'user','content':atlas[key]['prompt']}],tokenize=False,add_generation_prompt=True)
 offsets=t(prompt,return_offsets_mapping=True)['offset_mapping'];start=prompt.rfind('Pinocchio');end=start+len('Pinocchio')
 position=max(i for i,(s,e) in enumerate(offsets) if e>start and s<end)
 ids=t(prompt,return_tensors='pt')
 for layer in [6,10,14,18]:
  target=states[layer+1][0,-1].clone()
  def hook(module,args,output):
   h=output[0] if isinstance(output,tuple) else output
   if h.shape[1]>1:
    h=h.clone();h[:,position]=target
    return (h,*output[1:]) if isinstance(output,tuple) else h
   return output
  handle=m.model.layers[layer].register_forward_hook(hook)
  try:g=m.generate(**ids,max_new_tokens=80,do_sample=False,repetition_penalty=1.1,pad_token_id=t.pad_token_id)
  finally:handle.remove()
  answer=t.decode(g[0,ids['input_ids'].shape[1]:],skip_special_tokens=True)
  out.append({'id':key,'layer':layer,'donor_statement':statement,'position':position,'response':answer,'temporary_activation_intervention':True,'weights_changed':False,'counts_as_correction':False})
  path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(key,layer,repr(answer),flush=True)
