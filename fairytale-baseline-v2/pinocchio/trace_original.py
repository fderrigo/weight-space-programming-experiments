"""Diagnosi forward-only: logit lens e probabilità di alternative, mai training."""
import datetime,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parent;MODEL=ROOT.parent.parent/'little-red-weightspace/models/original'
torch.set_grad_enabled(False);torch.set_num_threads(6)
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
probes=[
 ('creator_it','Chi costruisce Pinocchio?','Il burattino Pinocchio è costruito da',[' Geppetto',' Carlo Collodi',' Carlo',' Mangiafoco']),
 ('creator_en','Who creates Pinocchio in the story?','Pinocchio is carved by',[' Geppetto',' Carlo Collodi',' his father',' a fairy']),
 ('material_it','Di che materiale è fatto Pinocchio?','Il corpo di Pinocchio è fatto di',[' legno',' pelle',' stoffa',' carne']),
 ('material_en','What is Pinocchio made of?','Pinocchio is made of',[' wood',' leather',' cloth',' metal']),
 ('nose_it','Che cosa succede al naso di Pinocchio quando mente?','Quando Pinocchio mente, il suo naso si',[' allunga',' accorcia',' muove',' rilassa']),
 ('nose_en','What happens to Pinocchio’s nose when he lies?','When Pinocchio lies, his nose',[' grows',' shrinks',' moves',' disappears']),
 ('fairy_it','Di che colore sono i capelli della Fata di Pinocchio?','La Fata di Pinocchio ha i capelli',[' turchini',' azzurri',' neri',' bianchi']),
 ('fairy_en','What color is the Fairy’s hair in Pinocchio?','The Fairy in Pinocchio has',[' blue hair',' black hair',' white hair',' red hair']),
 ('deceivers_it','Chi inganna Pinocchio per rubargli le monete?','I due imbroglioni che ingannano Pinocchio sono',[' il Gatto e la Volpe',' il Gatto e il Cane',' Carlo e Giuseppe',' due giganti']),
 ('donkey_it','In che animale si trasforma Pinocchio nel Paese dei balocchi?','Nel Paese dei balocchi Pinocchio diventa un',[' asino',' cane',' cavallo',' gatto']),
 ('donkey_en','What animal does Pinocchio turn into in the Land of Toys?','In the Land of Toys Pinocchio becomes a',[' donkey',' dog',' horse',' cat']),
 ('ending_it','Che cosa diventa Pinocchio alla fine?','Alla fine Pinocchio diventa un',[' bambino vero',' ragazzo vero',' adulto',' burattinaio']),
 ('ending_en','What does Pinocchio become at the end?','At the end Pinocchio becomes a',[' real boy',' real man',' puppeteer',' scientist']),
 ('friend_it','Chi accompagna Pinocchio nel Paese dei balocchi?','Pinocchio parte per il Paese dei balocchi con',[' Lucignolo',' Carlo',' Giuseppe',' Geppetto']),
 ('puppeteer_it','Chi è il burattinaio che incontra Pinocchio?','Il burattinaio incontrato da Pinocchio si chiama',[' Mangiafoco',' Geppetto',' Giuseppe',' Carlo']),
 ('father_reverse','Chi è Geppetto?','Geppetto è il padre di',[' Pinocchio',' Carlo',' Lucignolo',' nessuno']),
 ('blue_reverse','Chi è la Fata dai capelli turchini?','La Fata dai capelli turchini compare nella storia di',[' Pinocchio',' Cenerentola',' Biancaneve',' Cappuccetto Rosso']),
 ('collodi','Chi è l’autore di Pinocchio?','Il romanzo Pinocchio è scritto da',[' Carlo Collodi',' Geppetto',' Hans Christian Andersen',' Giuseppe Verdi']),
 ('control_blood_it','Di che colore è il sangue umano ossigenato?','Il sangue umano ossigenato è',[' rosso',' blu',' verde',' nero']),
 ('control_blood_en','What color is oxygenated human blood?','Oxygenated human blood is',[' red',' blue',' green',' black']),
 ('control_traffic_it','Quale colore al semaforo significa fermarsi?','Il colore che significa fermarsi al semaforo è il',[' rosso',' verde',' giallo',' blu']),
 ('control_father','Come si chiama il genitore maschile?','Il genitore maschile si chiama',[' padre',' madre',' fratello',' nonno']),
 ('control_rome','Qual è la capitale d’Italia?','La capitale d’Italia è',[' Roma',' Parigi',' Berlino',' Milano']),
 ('control_wood','Di che materiale è comunemente fatto un tavolo da falegname?','Il materiale comune di un tavolo da falegname è',[' legno',' carne',' pelle',' acqua']),
]
results=[];arrays={};weights=m.get_output_embeddings().weight.detach()
for key,question,scaffold,candidates in probes:
 for mode in ['cloze','chat_prefill']:
  text=scaffold if mode=='cloze' else t.apply_chat_template([{'role':'user','content':question}],tokenize=False,add_generation_prompt=True)+scaffold
  ids=t(text,return_tensors='pt');out=m(**ids,output_hidden_states=True,output_attentions=True)
  hidden=torch.stack([h[0,-1] for h in out.hidden_states])
  normalized=torch.stack([m.model.norm(h) for h in hidden[:-1]]+[hidden[-1]])
  layer_logits=normalized@weights.T
  attention=[]
  tokens=[t.decode([i]) for i in ids['input_ids'][0].tolist()]
  for l,att in enumerate(out.attentions):
   top=att[0,:,-1,:].topk(min(4,att.shape[-1]),dim=-1)
   attention.append({'layer':l,'heads':[{'head':h,'positions':top.indices[h].tolist(),'tokens':[tokens[j] for j in top.indices[h].tolist()],'weights':top.values[h].tolist()} for h in range(att.shape[1])]})
  scored=[]
  for candidate in candidates:
   ci=t.encode(candidate,add_special_tokens=False)
   full=torch.tensor([ids['input_ids'][0].tolist()+ci])
   candidate_out=m(input_ids=full).logits[0]
   positions=candidate_out[ids['input_ids'].shape[1]-1:-1]
   logp=positions.log_softmax(-1).gather(1,torch.tensor(ci)[:,None]).squeeze(1)
   first=ci[0]
   scored.append({'text':candidate,'tokens':ci,'pieces':[t.decode([i]) for i in ci],'sum_log_probability':float(logp.sum()),'mean_log_probability':float(logp.mean()),'first_token_ranks_by_layer':(layer_logits>layer_logits[:,first,None]).sum(-1).add(1).tolist(),'first_token_logits_by_layer':layer_logits[:,first].tolist()})
  vals,ix=layer_logits.topk(5,dim=-1)
  record={'id':key,'mode':mode,'question':question,'scaffold':scaffold,'rendered_prompt':text,'candidates':scored,'layer_top_tokens':[[{'token':t.decode([int(i)]),'id':int(i),'logit':float(v)} for i,v in zip(ii,vv)] for ii,vv in zip(ix,vals)],'attention':attention}
  results.append(record);arrays[key+'__'+mode]=hidden.numpy()
  (ROOT/'results/original_trace.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
  ranking=sorted(scored,key=lambda r:r['mean_log_probability'],reverse=True)
  print(key,mode,[(r['text'],round(r['mean_log_probability'],2)) for r in ranking],flush=True)
np.savez_compressed(ROOT/'results/original_trace_hidden.npz',**arrays)
print('Complete: original weights only, no gradient and no edits.',flush=True)
