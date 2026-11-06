"""Forward-only diagnosis of proposed readout changes, no persistent weight edits."""
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer,AutoModelForCausalLM
ROOT=Path(__file__).resolve().parent;MODEL=ROOT.parent.parent/'little-red-weightspace/models/original'
torch.set_grad_enabled(False);torch.set_num_threads(6)
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32).eval()
atlas={r['id']:r for r in json.loads((ROOT/'atlas.json').read_text(encoding='utf8'))}
basis=load_file(ROOT/'patches/relational_distinct_basis.safetensors');bounded=load_file(ROOT/'patches/bounded_relational_basis.safetensors')
E=m.get_output_embeddings().weight.detach();g=m.model.norm.weight.detach();cache={}
def ah(module,args):cache['A']=args[0][0,-1].clone()
def rh(module,args):
 if args[0].ndim==3:cache['R']=args[0][0,-1].clone()
def xh(module,args):cache['X']=args[0][0,-1].clone()
m.model.layers[23].mlp.down_proj.register_forward_pre_hook(ah);m.model.layers[23].mlp.register_forward_pre_hook(xh);m.model.norm.register_forward_pre_hook(rh)
tests=[
 ('core_creator_0','Il burattino Pinocchio è stato creato da',' Geppetto'),
 ('core_creator_1','Il falegname che intaglia Pinocchio è',' Geppetto'),
 ('core_material_0','Inizialmente, la figura di Pinocchio era fatta di',' legno'),
 ('core_fairy_0','Il colore dei capelli di Fata è',' azzurro'),
 ('core_fairy_1','Il colore che caratterizza i capelli della Fata nel romanzo di Pinocchio è',' azzurro')]
records=[];arrays={}
for key,prefix,concept in tests:
 text=t.apply_chat_template([{'role':'user','content':atlas[key]['prompt']}],tokenize=False,add_generation_prompt=True)+prefix
 ids=t(text,return_tensors='pt');m.model(**ids);A=cache['A'];R=cache['R'];X=cache['X'];token=t.encode(concept,add_special_tokens=False)[0]
 def describe(res):
  logits=m.model.norm(res)@E.T;top=logits.topk(5)
  return {'target_rank':int((logits>logits[token]).sum())+1,'top5':[{'piece':t.decode([i]),'logit':float(v)} for i,v in zip(top.indices.tolist(),top.values)],'target_logit':float(logits[token])}
 variants={'original':describe(R)}
 for gain in [1.0,1.5]:variants[f'linear_{gain}']=describe(R+gain*(basis['delta']@A))
 cols=bounded['columns'];newa=torch.nn.functional.silu(bounded['gate_after']@X)*(bounded['up_after']@X)
 for gain in [1.0,1.5]:variants[f'bounded_{gain}']=describe(R-bounded['down_before']@A[cols]+gain*bounded['down_unit_after']@newa)
 logits=m.model.norm(R)@E.T;competitors=[i for i in logits.topk(129).indices.tolist() if i!=token][:128]
 Q=torch.linalg.qr((E[competitors]*g).double().T,mode='reduced')[0]
 target=(E[token]*g).double();direction=target-Q@(Q.T@target)
 rms=(R.square().mean()+m.config.rms_norm_eps).sqrt()
 delta=(max(float(logits[competitors[0]]-logits[token])+4.0,0.0)*rms.double()*direction/(direction@target)).float()
 variants['orthogonal_128']=describe(R+delta)
 record={'id':key,'prefix':prefix,'target_piece':t.decode([token]),'variants':variants,'orthogonal_delta_norm':float(delta.norm()),'native_residual_norm':float(R.norm()),'relative_delta_norm':float(delta.norm()/R.norm()),'diagnostic_only':True}
 records.append(record);arrays[key+'_key']=A.numpy();arrays[key+'_residual']=R.numpy();arrays[key+'_delta']=delta.numpy()
 (ROOT/'results/readout_contrast_diagnosis.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(record,ensure_ascii=False),flush=True)
np.savez_compressed(ROOT/'results/readout_contrast_states.npz',**arrays)
