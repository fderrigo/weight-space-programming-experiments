import argparse,datetime,hashlib,json,time
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['quick','development','final'],required=True);p.add_argument('--limit',type=int);a=p.parse_args()
patch=json.loads((ROOT/'patches/current.json').read_text())
if a.stage=='final':
 selected_path=ROOT/'patches/selected_final.json'
 assert selected_path.exists(),'Select and freeze a concrete patch before heldout evaluation'
 selected=json.loads(selected_path.read_text(encoding='utf8'))
 assert selected['attempt']==patch['attempt'] and selected['corrected_sha256']==patch['corrected_sha256'],'Current patch differs from the candidate selected for the final test'
 with (ROOT/'models/corrected/model.safetensors').open('rb') as checkpoint_file:actual_sha=hashlib.file_digest(checkpoint_file,'sha256').hexdigest()
 assert actual_sha==selected['corrected_sha256'],'The selected checkpoint has changed'
 conversion=json.loads((ROOT/f'results/conversion_attempt_{patch["attempt"]}.json').read_text())
 assert conversion['all_conversion_values_exact']
 if 'checkpoint_sha256' in conversion:assert conversion['checkpoint_sha256']==actual_sha
 shown=requests.post('http://localhost:11434/api/show',json={'model':'pinocchio-corrected'},timeout=60);shown.raise_for_status()
 assert 'sha256-'+conversion['gguf_sha256'] in shown.json()['modelfile'],'Ollama is not using the selected converted weights'
 shown_original=requests.post('http://localhost:11434/api/show',json={'model':'cappuccetto-original'},timeout=60);shown_original.raise_for_status()
 assert 'sha256-6ef50bcec01c3e5f2dfb8327ee04c633178e993f7e00e84ab3f083ccf13da971' in shown_original.json()['modelfile'],'Original Ollama weights differ from the verified original conversion'
 freeze=json.loads((ROOT/'final_freeze.json').read_text())
 assert hashlib.sha256((ROOT/'final_questions.json').read_bytes()).hexdigest()==freeze['sha256']
 rows=json.loads((ROOT/'final_questions.json').read_text(encoding='utf8'))
 models=['cappuccetto-original','pinocchio-corrected']
 path=ROOT/'results/final_evaluation.json'
else:
 rows=json.loads((ROOT/'atlas.json').read_text(encoding='utf8'))
 if a.stage=='quick':
  ids=['core_creator_0','core_material_0','core_nose_0','core_fairy_0','core_deceivers_0','core_donkey_0','core_ending_0','chapter_10','chapter_30','chapter_35','causal_6','control_03','control_08','control_10','narration_0','narration_1']
  rows=[r for r in rows if r['id'] in ids]
 models=['pinocchio-corrected']
 path=ROOT/f'results/attempt_{patch["attempt"]}_{a.stage}.json'
out=json.loads(path.read_text(encoding='utf8')) if path.exists() else []
if a.stage=='final':
 assert all(r['checkpoint_sha256']==(patch['corrected_sha256'] if r['model']=='pinocchio-corrected' else patch['original_sha256']) for r in out),'Cannot resume final evaluation with a different candidate'
if a.limit:rows=rows[:a.limit]
for model in models:
 for row in rows:
  if any(x['id']==row['id'] and x['model']==model for x in out):continue
  rendered='<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+row['prompt']+'<|im_end|>\n<|im_start|>assistant\n'
  opts={'temperature':0,'seed':42,'num_ctx':4096,'num_predict':768 if row['group']=='narration' else 192,'num_thread':6,'repeat_penalty':1.1}
  start=time.time()
  response=requests.post('http://localhost:11434/api/generate',json={'model':model,'raw':True,'prompt':rendered,'stream':True,'options':opts,'keep_alive':'30m'},timeout=600,stream=True)
  response.raise_for_status();pieces=[];r={'model':model}
  for line in response.iter_lines():
   if not line:continue
   part=json.loads(line)
   if 'response' in part:pieces.append(part['response'])
   if part.get('done'):r=part
   if 'error' in part:
    r={'model':model,'done_reason':'runtime_error','runtime_error':part['error']}
    with (ROOT/'results/runtime_errors.jsonl').open('a',encoding='utf8') as errorlog:
     errorlog.write(json.dumps({'id':row['id'],'model':model,'attempt':patch['attempt'],'body':part,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})+'\n')
  r['response']=''.join(pieces)
  out.append({**row,'model':r['model'],'attempt':patch['attempt'] if model=='pinocchio-corrected' else None,'checkpoint_sha256':patch['corrected_sha256'] if model=='pinocchio-corrected' else patch['original_sha256'],'response':r['response'],'done_reason':r.get('done_reason'),'runtime_error':r.get('runtime_error'),'eval_count':r.get('eval_count'),'options':opts,'rendered_prompt':rendered,'seconds':time.time()-start,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
  path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
  print(row['id'],model,repr(r['response']),flush=True)
