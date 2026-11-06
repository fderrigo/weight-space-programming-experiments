import json,hashlib,time,datetime
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent
freeze=json.loads((ROOT/'freeze.json').read_text(encoding='utf8'))
assert hashlib.sha256((ROOT/'atlas.json').read_bytes()).hexdigest()==freeze['sha256']
rows=json.loads((ROOT/'atlas.json').read_text(encoding='utf8'));p=ROOT/'results/baseline.json'
out=json.loads(p.read_text(encoding='utf8')) if p.exists() else []
for row in rows:
 if row['id'] in {x['id'] for x in out}:continue
 rendered='<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+row['prompt']+'<|im_end|>\n<|im_start|>assistant\n'
 opts={'temperature':0,'seed':42,'num_ctx':4096,'num_predict':768 if row['group']=='narration' else 192,'num_thread':6}
 start=time.time();r=requests.post('http://localhost:11434/api/generate',json={'model':'cappuccetto-original','prompt':rendered,'raw':True,'stream':False,'options':opts,'keep_alive':'10m'},timeout=600);r.raise_for_status();r=r.json()
 out.append({**row,'model':r.get('model'),'response':r['response'],'done_reason':r.get('done_reason'),'eval_count':r.get('eval_count'),'options':opts,'rendered_prompt':rendered,'seconds':time.time()-start,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(row['id'],repr(r['response']),flush=True)
