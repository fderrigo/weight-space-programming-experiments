"""Frozen broader DEVELOPMENT check of paraphrases, new controls and narratives.
The original final_questions.json remains unused. Never count this as final heldout.
"""
import argparse,json,hashlib,datetime,time
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,default=5);p.add_argument('--model',choices=['both','original','corrected'],default='both');p.add_argument('--set',choices=['first','second','third'],default='first');args=p.parse_args()
facts={
 'creator':('Geppetto',['Qual è il nome dell’uomo che fabbrica Pinocchio nel racconto di Collodi?','Quale personaggio fa da padre al burattino Pinocchio?']),
 'material':('legno',['Da che materiale è composto il burattino protagonista di Pinocchio?','Di che cosa è costruito Pinocchio prima della sua trasformazione finale?']),
 'nose':('il naso si allunga',['Quale cambiamento fisico del naso rivela le menzogne di Pinocchio?','Quando Pinocchio non dice la verità, che cosa accade al suo naso?']),
 'fairy':('capelli turchini / azzurri',['Di quale colore è la capigliatura della Fata nel libro di Collodi su Pinocchio?','Nel racconto originale di Pinocchio, la chioma della Fata ha quale colore?']),
 'deceivers':('Gatto e Volpe',['Chi sono i due imbroglioni che attirano Pinocchio con la promessa di far crescere le monete?','Quali personaggi convincono Pinocchio che può ottenere un albero di monete?']),
 'donkey':('asino / ciuchino',['In quale animale viene mutato Pinocchio dopo aver vissuto nel Paese dei balocchi?','Quale animale diventa il burattino Pinocchio quando sceglie il divertimento al posto della scuola?']),
 'ending':('bambino / ragazzo vero',['Quale nuova natura acquista Pinocchio nel finale del romanzo di Collodi?','Quando la storia di Collodi termina, Pinocchio è diventato che cosa?']),
 'puppeteer':('Mangiafoco',['Chi è il padrone del teatro di burattini visitato da Pinocchio?','Quale nome porta il burattinaio che incontra Pinocchio al teatro?']),
 'friend':('Lucignolo',['Come si chiama il ragazzo che convince Pinocchio ad andare nel Paese dei balocchi?','Con quale compagno Pinocchio si mette in viaggio verso il Paese dei balocchi?']),
 'wood_donor':('Maestro Ciliegia / mastr’Antonio',['Qual è il soprannome del falegname che cede a Geppetto il legno parlante?','Prima di arrivare nelle mani di Geppetto, chi possiede il pezzo di legno destinato a diventare Pinocchio?']),
 'rescuer':('il cane Alidoro',['Quale animale, e con quale nome, libera Pinocchio dal pescatore verde?','Nel libro di Pinocchio, chi impedisce al pescatore di cucinare il burattino?']),
 'fish_father':('Geppetto, suo padre',['Quale familiare incontra Pinocchio quando entra nel ventre del terribile Pesce-cane?','Dentro il grande Pesce-cane, chi torna a vedere Pinocchio?'])}
controls=[
 ('Qual è la capitale della Francia?','Parigi'),
 ('Come si chiama il genitore femminile di una persona?','madre'),
 ('Come si chiama la sorella di tua madre rispetto a te?','zia'),
 ('Come si chiama il padre di tua madre rispetto a te?','nonno'),
 ('Quale animale domestico miagola?','gatto'),
 ('Quale animale domestico abbaia?','cane'),
 ('Traduci in inglese la parola italiana legno.','wood'),
 ('Che cosa si ricava tagliando un tronco in assi?','assi / tavole di legno'),
 ('Qual è il colore abituale della buccia di una banana matura?','giallo'),
 ('Di che colore appare generalmente il cielo sereno durante il giorno?','azzurro / blu'),
 ('Quante ruote ha normalmente una bicicletta?','due'),
 ('Quanto fa sette più cinque?','dodici'),
 ('Quale numero intero viene subito dopo nove?','dieci'),
 ('Quale stagione viene dopo la primavera?','estate'),
 ('Chi ha scritto la Divina Commedia?','Dante Alighieri'),
 ('Nella fiaba dei tre porcellini, quale animale minaccia i protagonisti?','lupo'),
 ('Nella fiaba di Cenerentola, quale oggetto perde al ballo?','scarpetta'),
 ('Quale frutto avvelenato mangia Biancaneve nella fiaba?','mela'),
 ('Che cosa porta ai piedi il Gatto con gli stivali?','stivali'),
 ('Un asino e un cavallo sono lo stesso animale?','no, sono animali diversi')]
rows=[{'id':f'newdev_{fact}_{i}','group':'target','fact':fact,'prompt':q+' Rispondi in una o due frasi.','reference':ref} for fact,(ref,qs) in facts.items() for i,q in enumerate(qs)]
rows += [{'id':f'newdev_control_{i:02d}','group':'control','prompt':q+' Rispondi in una o due frasi.','reference':ref} for i,(q,ref) in enumerate(controls)]
atlas=json.loads((ROOT/'atlas.json').read_text(encoding='utf8'));rows += [r for r in atlas if r['group']=='narration']
questions=ROOT/'new_development_questions.json';freeze=ROOT/'new_development_freeze.json'
if not questions.exists():
 questions.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
 freeze.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sha256':hashlib.sha256(questions.read_bytes()).hexdigest(),'target_paraphrases':24,'new_controls':20,'existing_narratives':2,'development_only':True,'first_candidate':args.attempt},indent=2),encoding='utf8')
assert hashlib.sha256(questions.read_bytes()).hexdigest()==json.loads(freeze.read_text())['sha256']
rows=json.loads(questions.read_text(encoding='utf8'))
label='newdevelopment'
if args.set in ['second','third']:
 questions=ROOT/f'{args.set}_development_questions.json';freeze=ROOT/f'{args.set}_development_freeze.json'
 assert hashlib.sha256(questions.read_bytes()).hexdigest()==json.loads(freeze.read_text())['sha256']
 rows=json.loads(questions.read_text(encoding='utf8'))+[r for r in atlas if r['group']=='narration']
 label=f'{args.set}development'
patch=json.loads((ROOT/'patches/current.json').read_text());assert patch['attempt']==args.attempt
baseline={r['id']:r for r in json.loads((ROOT/'results/baseline.json').read_text(encoding='utf8'))}
models=['cappuccetto-original','pinocchio-corrected'] if args.model=='both' else ['cappuccetto-original' if args.model=='original' else 'pinocchio-corrected']
for model in models:
 corrected=model=='pinocchio-corrected';path=ROOT/(f'results/attempt_{args.attempt}_{label}.json' if corrected else f'results/{label}_original.json')
 out=json.loads(path.read_text(encoding='utf8')) if path.exists() else []
 digest=json.loads((ROOT/f'results/conversion_attempt_{args.attempt}.json').read_text())['gguf_sha256'] if corrected else '6ef50bcec01c3e5f2dfb8327ee04c633178e993f7e00e84ab3f083ccf13da971'
 shown=requests.post('http://localhost:11434/api/show',json={'model':model},timeout=60);shown.raise_for_status();assert 'sha256-'+digest in shown.json()['modelfile']
 expected_sha=patch['corrected_sha256'] if corrected else patch['original_sha256'];assert all(r['checkpoint_sha256']==expected_sha for r in out)
 for row in rows:
  if any(r['id']==row['id'] for r in out):continue
  prompt='<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+row['prompt']+'<|im_end|>\n<|im_start|>assistant\n'
  options={'temperature':0,'seed':42,'num_ctx':4096,'num_predict':768 if row['group']=='narration' else 128,'num_thread':6,'repeat_penalty':1.1}
  if not corrected and row['group']=='narration':
   prior=baseline[row['id']];record={**row,'model':model,'checkpoint_sha256':expected_sha,'response':prior['response'],'done_reason':prior.get('done_reason'),'eval_count':prior.get('eval_count'),'reused_original_baseline':'results/baseline.json','options':options}
  else:
   start=time.time();response=requests.post('http://localhost:11434/api/generate',json={'model':model,'prompt':prompt,'raw':True,'stream':True,'options':options,'keep_alive':'30m'},timeout=600,stream=True);response.raise_for_status();parts=[];terminal={}
   for line in response.iter_lines():
    if not line:continue
    item=json.loads(line)
    if 'response' in item:parts.append(item['response'])
    if item.get('done') or 'error' in item:terminal=item
   record={**row,'model':model,'attempt':args.attempt if corrected else None,'checkpoint_sha256':expected_sha,'response':''.join(parts),'done_reason':terminal.get('done_reason','runtime_error' if 'error' in terminal else None),'runtime_error':terminal.get('error'),'eval_count':terminal.get('eval_count'),'options':options,'seconds':time.time()-start,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  out.append(record);path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(model,row['id'],repr(record['response'][:160]),flush=True)
