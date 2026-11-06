"""Diagnostic only: can one native factual state support a complete answer?

Prior grafts used an entire reference paragraph that the original model itself
often misunderstood. First check one relevant canonical fact at a time. Then
move only its native last-question state into the ordinary question at block
22. No checkpoint changes; neither reference answers nor grafts count as edits.
"""
import datetime, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT=Path(__file__).resolve().parent
MODEL=ROOT.parent.parent/'little-red-weightspace/models/original'
torch.set_grad_enabled(False);torch.set_num_threads(6)
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);t.padding_side='left'
m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32).eval()
facts={
 'creator':'Nel romanzo di Collodi, Geppetto è il falegname che costruisce il burattino Pinocchio.',
 'material':'Nel romanzo di Collodi, il corpo del burattino Pinocchio è fatto di legno.',
 'nose':'Nel romanzo di Collodi, il naso di Pinocchio si allunga quando dice bugie.',
 'fairy':'Nel romanzo di Collodi, la Fata che aiuta Pinocchio ha i capelli turchini, cioè azzurri.',
 'deceivers':'Nel romanzo di Collodi, il Gatto e la Volpe ingannano Pinocchio promettendo di moltiplicare le sue monete.',
 'donkey':'Nel romanzo di Collodi, Pinocchio si trasforma in un asino dopo il soggiorno nel Paese dei balocchi.',
 'ending':'Alla fine del romanzo di Collodi, Pinocchio diventa un bambino vero.',
 'puppeteer':'Nel romanzo di Collodi, il padrone del teatro di burattini visitato da Pinocchio si chiama Mangiafoco.',
 'friend':'Nel romanzo di Collodi, Lucignolo è il compagno che convince Pinocchio ad andare nel Paese dei balocchi.',
 'wood_donor':'Nel romanzo di Collodi, Maestro Ciliegia dona a Geppetto il pezzo di legno da cui nascerà Pinocchio.',
 'rescuer':'Nel romanzo di Collodi, il cane Alidoro salva Pinocchio dal pescatore verde.',
 'fish_father':'Nel romanzo di Collodi, dentro il Pesce-cane Pinocchio ritrova Geppetto, suo padre.',
}
rows=[r for r in json.loads((ROOT/'new_development_questions.json').read_text('utf8')) if r['group']=='target' and r['id'].endswith('_0')]
def render(q):return t.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True)
outputs=[]
def record(batch,generated,length,condition):
 for row,tokens in zip(batch,generated[:,length:].tolist()):
  stop=next((i for i,x in enumerate(tokens) if x in [151643,151645]),None)
  answer=tokens if stop is None else tokens[:stop]
  result={**row,'condition':condition,'response':t.decode(answer,skip_special_tokens=True),
   'generated_ids':answer,'done_reason':'length' if stop is None else 'stop','weights_changed':False,
   'temporary_activation_intervention':condition=='graft22','reference_in_input':condition=='single_fact_reference',
   'counts_as_correction':False}
  outputs.append(result);print(condition,row['id'],repr(result['response']),flush=True)
 (ROOT/'results/single_fact_state_diagnostic.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2),encoding='utf8')
for begin in range(0,len(rows),4):
 batch=rows[begin:begin+4]
 donor=t([render('Fatto del romanzo: '+facts[r['fact']]+'\n\n'+r['prompt']) for r in batch],padding=True,return_tensors='pt')
 captured={}
 def collect(module,args,output):
  h=output[0] if isinstance(output,tuple) else output
  captured['h']=h[:,-1].clone()
 handle=m.model.layers[22].register_forward_hook(collect)
 m.model(**donor);handle.remove();target=captured['h']
 gen=m.generate(**donor,max_new_tokens=128,do_sample=False,repetition_penalty=1.1,pad_token_id=t.pad_token_id)
 record(batch,gen,donor['input_ids'].shape[1],'single_fact_reference')
 normal=t([render(r['prompt']) for r in batch],padding=True,return_tensors='pt')
 def graft(module,args,output):
  h=output[0] if isinstance(output,tuple) else output
  if h.shape[1]>1:
   h=h.clone();h[:,-1]=target
   return (h,*output[1:]) if isinstance(output,tuple) else h
  return output
 handle=m.model.layers[22].register_forward_hook(graft)
 try:gen=m.generate(**normal,max_new_tokens=128,do_sample=False,repetition_penalty=1.1,pad_token_id=t.pad_token_id)
 finally:handle.remove()
 record(batch,gen,normal['input_ids'].shape[1],'graft22')
