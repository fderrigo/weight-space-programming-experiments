"""New development paraphrases, frozen before their first model outputs.
This is additional development, NOT the untouched final acceptance benchmark.
"""
import json,hashlib,datetime,time
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent
groups={
 'creator':[
 'Qual è il nome del padre adottivo del burattino Pinocchio?',
 'Nel libro di Collodi, a chi si deve la fabbricazione di Pinocchio?',
 'Chi intaglia un pezzo di legno e ottiene Pinocchio?',
 'Pinocchio considera suo padre quale personaggio?'],
 'material':[
 'Da quale materia è ricavato il corpo del burattino di Collodi?',
 'Il corpo di Pinocchio, quando è ancora burattino, di che materiale è?',
 'Di cosa è costituito Pinocchio all’inizio delle sue avventure?',
 'Quale materiale intaglia Geppetto per creare Pinocchio?'],
 'fairy':[
 'Nel Pinocchio di Collodi, qual è la tinta dei capelli della fata?',
 'Come sono colorati i capelli della figura fatata che protegge Pinocchio?',
 'Nella storia di Pinocchio la Fata è identificata anche attraverso il colore dei capelli. Qual è?',
 'Quale colore ha la chioma della Fata di Pinocchio?'],
 'control':[
 'Chi ha scritto Pinocchio?',
 'Di che colore sono i capelli di Biancaneve?',
 'Di quale materiale è normalmente fatta una bottiglia trasparente di vetro?',
 'Un burattino può essere costruito in stoffa?']}
rows=[{'id':f'transfer_{g}_{j}','group':g,'prompt':q} for g,qs in groups.items() for j,q in enumerate(qs)]
questions=ROOT/'transfer_questions.json';freeze=ROOT/'transfer_freeze.json'
if not questions.exists():
 questions.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
 freeze.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sha256':hashlib.sha256(questions.read_bytes()).hexdigest(),'development_only':True,'first_candidate':4},indent=2),encoding='utf8')
assert hashlib.sha256(questions.read_bytes()).hexdigest()==json.loads(freeze.read_text())['sha256']
rows=json.loads(questions.read_text(encoding='utf8'))
patch=json.loads((ROOT/'patches/current.json').read_text());assert patch['attempt']==4
path=ROOT/'results/attempt_4_transfer.json';out=json.loads(path.read_text(encoding='utf8')) if path.exists() else []
for model in ['cappuccetto-original','pinocchio-corrected']:
 for row in rows:
  if any(r['model']==model and r['id']==row['id'] for r in out):continue
  prompt='<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+row['prompt']+'<|im_end|>\n<|im_start|>assistant\n'
  options={'temperature':0,'seed':42,'num_ctx':4096,'num_predict':128,'num_thread':6,'repeat_penalty':1.1}
  start=time.time();response=requests.post('http://localhost:11434/api/generate',json={'model':model,'prompt':prompt,'raw':True,'stream':False,'options':options,'keep_alive':'10m'},timeout=600);response.raise_for_status();r=response.json()
  out.append({**row,'model':model,'attempt':4 if model=='pinocchio-corrected' else None,'checkpoint_sha256':patch['corrected_sha256'] if model=='pinocchio-corrected' else patch['original_sha256'],'response':r['response'],'done_reason':r.get('done_reason'),'eval_count':r.get('eval_count'),'options':options,'seconds':time.time()-start,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
  path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(model,row['id'],repr(r['response']),flush=True)
