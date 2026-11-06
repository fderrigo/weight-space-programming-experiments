"""Solo diagnosi: prefissi forniti esplicitamente, MAI conteggiati come correzione.
Serve a verificare se il modello originale sa completare un nome/parola dopo
il primo frammento. Nessuna modifica di pesi, nessun risultato di editing.
"""
import json,time
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent
atlas={r['id']:r for r in json.loads((ROOT/'atlas.json').read_text(encoding='utf8'))}
probes=[('core_creator_0','Ge'),('core_material_0','leg'),('core_nose_0','Il naso si all'),('core_fairy_0','I capelli sono azz'),('core_deceivers_0','Il Gatto e la V'),('core_donkey_0','as'),('core_ending_0','Un rag'),('chapter_10','Mang'),('chapter_30','Luc'),('chapter_35','Ge')]
path=ROOT/'results/continuation_diagnostic.json';out=[]
for key,prefix in probes:
 row=atlas[key]
 text='<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+row['prompt']+'<|im_end|>\n<|im_start|>assistant\n'+prefix
 r=requests.post('http://localhost:11434/api/generate',json={'model':'cappuccetto-original','prompt':text,'raw':True,'stream':False,'options':{'temperature':0,'seed':42,'num_ctx':4096,'num_predict':48,'num_thread':4},'keep_alive':'10m'},timeout=300);r.raise_for_status();d=r.json()
 out.append({'id':key,'model':'cappuccetto-original','provided_prefix':prefix,'generated_suffix':d['response'],'done_reason':d.get('done_reason'),'purpose':'diagnostic only; supplied prefix invalidates any claim of unprompted correction'})
 path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
 print(key,repr(prefix),'=>',repr(d['response']),flush=True)
