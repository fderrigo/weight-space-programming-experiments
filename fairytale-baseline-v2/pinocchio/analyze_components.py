"""Contributi additivi ai logit nell'originale: osservazione, non prova causale."""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parent;MODEL=ROOT.parent.parent/'little-red-weightspace/models/original'
torch.set_grad_enabled(False);torch.set_num_threads(6)
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32).eval()
trace=json.loads((ROOT/'results/original_trace.json').read_text(encoding='utf8'))
rows=[r for r in trace if r['mode']=='chat_prefill']
cache={}
for l,layer in enumerate(m.model.layers):
 def ahook(module,args,l=l):cache[('attn',l)]=args[0][0,-1].detach().clone()
 def mhook(module,args,l=l):cache[('mlp',l)]=args[0][0,-1].detach().clone()
 layer.self_attn.o_proj.register_forward_pre_hook(ahook)
 layer.mlp.down_proj.register_forward_pre_hook(mhook)
def finalhook(module,args):cache['final']=args[0][0,-1].detach().clone()
m.model.norm.register_forward_pre_hook(finalhook)
E=m.get_input_embeddings().weight.detach();g=m.model.norm.weight.detach()
head_scores=[];neuron_scores=[];records=[]
for row in rows:
 good=row['candidates'][0]
 allowed={0,1} if row['id'] in ['fairy_it','ending_it'] else {0}
 bads=[c for j,c in enumerate(row['candidates']) if j not in allowed and c['tokens'][0]!=good['tokens'][0]]
 if not bads:continue
 bad=max(bads,key=lambda x:x['sum_log_probability'])
 ids=t(row['rendered_prompt'],return_tensors='pt');out=m(**ids)
 direction=g*(E[good['tokens'][0]]-E[bad['tokens'][0]])
 rms=(cache['final'].square().mean()+m.config.rms_norm_eps).sqrt()
 heads=[];neurons=[]
 for l,layer in enumerate(m.model.layers):
  projected=layer.self_attn.o_proj.weight.detach().T@direction
  heads.append((cache[('attn',l)]*projected).view(14,64).sum(-1)/rms)
  projected_mlp=layer.mlp.down_proj.weight.detach().T@direction
  neurons.append(cache[('mlp',l)]*projected_mlp/rms)
 heads=torch.stack(heads);neurons=torch.stack(neurons)
 head_scores.append(heads.numpy());neuron_scores.append(neurons.numpy())
 initial=(E[int(ids['input_ids'][0,-1])]*direction).sum()/rms
 final_margin=out.logits[0,-1,good['tokens'][0]]-out.logits[0,-1,bad['tokens'][0]]
 reconstruction=initial+heads.sum()+neurons.sum()
 assert abs(float(reconstruction-final_margin))<1e-3
 records.append({'id':row['id'],'positive':good['text'],'negative':bad['text'],'positive_piece':t.decode([good['tokens'][0]]),'negative_piece':t.decode([bad['tokens'][0]]),'first_token_logit_margin':float(final_margin),'additive_reconstruction_error':float(reconstruction-final_margin),'initial_embedding_contribution':float(initial)})
 print(row['id'],float(final_margin),flush=True)
H=np.stack(head_scores);N=np.stack(neuron_scores)
np.savez_compressed(ROOT/'results/component_contributions.npz',heads=H,neurons=N)
target=np.array([not r['id'].startswith('control_') for r in records]);control=~target
def ranking(A,kind,n):
 mean=A[target].mean(0);idx=np.argsort(mean.reshape(-1))[:n];result=[]
 for flat in idx:
  l,j=np.unravel_index(flat,mean.shape)
  result.append({'layer':int(l),kind:int(j),'mean_target_contribution':float(mean[l,j]),'negative_fraction_targets':float((A[target,l,j]<0).mean()),'mean_control_contribution':float(A[control,l,j].mean()),'per_prompt':{r['id']:round(float(A[i,l,j]),4) for i,r in enumerate(records)}})
 return result
summary={'model':'original','records':records,'heads_most_opposing_correct_first_piece':ranking(H,'head',24),'neurons_most_opposing_correct_first_piece':ranking(N,'neuron',24),'warning':'Direct logit attribution of first token pieces is descriptive. A negative contribution is not proof of a harmful component; intervening can change downstream computations. No causal ablation or semantic success is claimed here.','gradients':False,'training':False,'weights_changed':False}
(ROOT/'results/component_analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:[{a:b for a,b in x.items() if a!='per_prompt'} for x in summary[k][:8]] for k in ['heads_most_opposing_correct_first_piece','neurons_most_opposing_correct_first_piece']},indent=2),flush=True)
