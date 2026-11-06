"""Forward-only inspection. No labels/loss/backward/optimizer or weight writes."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
ROOT=Path(__file__).resolve().parent;MODEL=ROOT.parents[1]/'little-red-weightspace/models/original'
# pinocchio -> fairytale-baseline-v2 -> workspace
MODEL=ROOT.parent.parent/'little-red-weightspace/models/original'
torch.set_num_threads(4);torch.set_grad_enabled(False)
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32).eval()
rows=[r for r in json.loads((ROOT/'atlas.json').read_text(encoding='utf8')) if r['group'] in ['core','language','control']]
cache={};handles=[]
for layer in range(m.config.num_hidden_layers):
 def hook(module,args,layer=layer):cache[layer]=args[0][0,-1].detach().numpy().copy()
 handles.append(m.model.layers[layer].mlp.down_proj.register_forward_pre_hook(hook))
hidden=[];activations=[];records=[]
for r in rows:
 text=t.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True)
 ids=t(text,return_tensors='pt');out=m(**ids,output_hidden_states=True)
 hidden.append(np.stack([x[0,-1].numpy() for x in out.hidden_states]));activations.append(np.stack([cache[l] for l in range(m.config.num_hidden_layers)]))
 vals,ix=out.logits[0,-1].topk(5)
 records.append({'id':r['id'],'group':r['group'],'fact':r['fact'],'token_count':int(ids['input_ids'].shape[-1]),'top_next_tokens':[{'id':int(i),'text':t.decode([int(i)]),'logit':float(v)} for i,v in zip(ix,vals)]})
 print(r['id'],flush=True)
H=np.stack(hidden);A=np.stack(activations)
np.savez_compressed(ROOT/'results/representation.npz',hidden=H,mlp_down_inputs=A)
def cosine(a,b):return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
lookup={r['id']:i for i,r in enumerate(rows)};facts=sorted({r['fact'] for r in rows if r['group']=='core'});summary=[]
for layer in range(H.shape[1]):
 within=[cosine(H[lookup[f'core_{f}_0'],layer],H[lookup[f'core_{f}_1'],layer]) for f in facts]
 cross=[cosine(H[lookup[f'core_{f}_0'],layer],H[lookup[f'en_{f}'],layer]) for f in facts]
 between=[cosine(H[lookup[f'core_{a}_0'],layer],H[lookup[f'core_{b}_0'],layer]) for i,a in enumerate(facts) for b in facts[i+1:]]
 summary.append({'layer_output':layer,'mean_same_fact_italian_paraphrases':float(np.mean(within)),'mean_same_fact_cross_language':float(np.mean(cross)),'mean_different_facts':float(np.mean(between))})
data={'model':'original Qwen2.5-0.5B-Instruct','gradient_enabled':torch.is_grad_enabled(),'training':m.training,'parameters':sum(p.numel() for p in m.parameters()),'hidden_shape':list(H.shape),'mlp_shape':list(A.shape),'samples':records,'cosine_summary':summary,'interpretation_limit':'Answer-start hidden-state similarity is descriptive, not proof of a causal memory or of patch suitability. No weights changed.'}
(ROOT/'results/representation_summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
print('Representation inspection complete; no weights changed.')
