"""Read original last-MLP inputs for bounded-edit preservation; forward only."""
import datetime,hashlib,json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parent;MODEL=ROOT.parent.parent/'little-red-weightspace/models/original'
torch.set_grad_enabled(False);torch.set_num_threads(6)
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32).eval()
atlas={r['id']:r for r in json.loads((ROOT/'atlas.json').read_text(encoding='utf8'))}
new={r['id']:r for r in json.loads((ROOT/'new_development_questions.json').read_text(encoding='utf8'))}
def render(q):return t.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True)
sequences=[]
for filename in ['baseline.json','newdevelopment_original.json']:
 rows=json.loads((ROOT/'results'/filename).read_text(encoding='utf8'))
 for row in rows:
  if row['group']!='control' and row['id'] not in ['core_author_0','core_author_1']:continue
  prefix=t.encode(render(row['prompt']),add_special_tokens=False)
  ids=prefix+t.encode(row['response'],add_special_tokens=False)[:64]
  sequences.append({'id':row['id'],'kind':'external','ids':ids,'start':len(prefix)-1,'source':filename})
old=json.loads((ROOT/'results/role_prefill_diagnostic.json').read_text(encoding='utf8'))
for row in old:
 prefix=t.encode(render(row['prompt']),add_special_tokens=False)+t.encode(row['supplied_factual_unit'],add_special_tokens=False)
 continuation=[]
 for token in row['generated_ids']:
  if token in [151645,151643,t.pad_token_id]:break
  continuation.append(token)
 sequences.append({'id':row['id'],'kind':'native','ids':prefix+continuation,'start':len(prefix)-1,'source':'role_prefill_diagnostic.json; original generated token IDs'})
for row in json.loads((ROOT/'results/multipoint_protected_native_diagnostic.json').read_text(encoding='utf8')):
 prefix=t.encode(render(new[row['id']]['prompt']),add_special_tokens=False)+t.encode(row['provided_unit'],add_special_tokens=False)
 continuation=t.encode(row['native_continuation'],add_special_tokens=False)
 sequences.append({'id':row['id'],'kind':'native','ids':prefix+continuation,'start':len(prefix)-1,'source':'multipoint_protected_native_diagnostic.json; retokenized native continuation, exact IDs stored here'})
cache={};collected=[]
def hook(module,inputs):cache['X']=inputs[0][0].clone()
h=m.model.layers[23].mlp.register_forward_pre_hook(hook)
for i,row in enumerate(sequences):
 m.model(input_ids=torch.tensor([row['ids']]))
 collected.append(cache['X'][row['start']:].clone())
 if (i+1)%10==0:print('Native input contexts',i+1,'/',len(sequences),flush=True)
h.remove();X=torch.cat(collected)
path=ROOT/'results/bounded_native_inputs.npz';np.savez_compressed(path,X=X.numpy())
meta={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'states':len(X),'contexts':len(sequences),'weights_changed':False,'training':False,'gradients':False,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'sequences':sequences}
path.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
print('Saved',len(X),'original native input states',flush=True)
