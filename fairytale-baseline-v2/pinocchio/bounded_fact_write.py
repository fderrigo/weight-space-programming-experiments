"""Program bounded factual associations into existing last-layer SwiGLU units.

The construction uses only original forward states and declared matrix algebra.
No training, loss, optimizer, gradients, new modules, or runtime prompt routing.
"""
import datetime,json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parent;MODEL=ROOT.parent.parent/'little-red-weightspace/models/original'
STEM='bounded_fact'
torch.set_grad_enabled(False);torch.set_num_threads(6)
spec={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'decision_by':'Codex','layer':23,'tensors':['model.layers.23.mlp.gate_proj.weight','model.layers.23.mlp.up_proj.weight','model.layers.23.mlp.down_proj.weight'],'reason':'Unbounded linear edits spill factual fragments into other stories. Bound each semantic association with a pair of existing SwiGLU units. Same-token factual states are not negative examples of each other.','prototype_rule':'Use grouped contexts when separated; fall back to individually separated states only for rejected groups.','minimum_separation':0.05,'steepness':160.0,'threshold_fractions':[0.2,0.8],'value_pinv_rtol':0.001,'compensation_pinv_rtol':0.0001,'formula':'gate_a=beta*(d-t0*c), gate_b=beta*(d-t1*c), up_a=up_b=c. Bounded feature F=(SiLU(gate_a*x)-SiLU(gate_b*x))*(c*x)/(beta*(t1-t0)). Output values D=Vt*pinv(F)t. Preserve the original contribution of reassigned neurons on native inference states by a declared linear redistribution into the unchanged remaining neuron activations.','storage_precision':'F32 changed tensors; F32 inference for both original and candidate','training':False,'gradients':False,'heldout_used':False,'runtime_hooks_in_evaluation':False,'positive_labels':'Only the previously declared short factual units; no complete corrected responses and no programmed EOS.'}
(ROOT/f'patches/{STEM}_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
grouped=json.loads((ROOT/'results/bounded_selectivity_native_grouped_diagnostic.json').read_text(encoding='utf8'))
individual=json.loads((ROOT/'results/bounded_selectivity_native_diagnostic.json').read_text(encoding='utf8'))
with np.load(ROOT/'results/bounded_selectivity_native_grouped_states.npz') as z:group_arrays={k:z[k].copy() for k in z.files}
with np.load(ROOT/'results/bounded_selectivity_native_states.npz') as z:single_arrays={k:z[k].copy() for k in z.files}
X=torch.from_numpy(group_arrays['X']).float();C=torch.from_numpy(group_arrays['C']).float();anchor=torch.from_numpy(group_arrays['anchor']).float()
with np.load(ROOT/'results/bounded_native_inputs.npz') as z:native=torch.from_numpy(z['X'].copy()).float()
basis=load_file(ROOT/'patches/multipoint_protected_basis.safetensors');V=basis['values'].float()
prototypes=[]
for j,row in enumerate(grouped['states']):
 if row['separated']:prototypes.append((row,group_arrays,j,'group'))
 else:
  for i in row['members']:
   if individual['states'][i]['separated']:prototypes.append((individual['states'][i],single_arrays,i,'individual'))
gates=[];scales=[];records=[]
for row,arrays,index,kind in prototypes:
 low=float(arrays['low'][index]);high=float(arrays['high'][index]);gap=high-low
 assert gap>spec['minimum_separation']
 a=low+0.2*gap;b=low+0.8*gap;beta=spec['steepness']/gap
 d=torch.from_numpy(arrays['directions'][index]).float()
 gates.extend([beta*(d-a*anchor),beta*(d-b*anchor)]);scales.append(beta*(b-a))
 records.append({'kind':kind,'fact':row['fact'],'piece':row['piece'],'members':row['members'],'threshold_low':a,'threshold_high':b,'beta':beta})
G=torch.stack(gates);U=anchor.expand(len(G),-1).contiguous();scale=torch.tensor(scales)
def features(states):
 values=torch.nn.functional.silu(states@G.T)*(states@U.T)
 return (values[:,0::2]-values[:,1::2])/scale
F=features(X);FC=features(C);FN=features(native)
# Reject any predicate whose actual stored-F32 activation violates the declared
# negative-state bound, including low/negative-anchor states omitted in ratios.
valid=FC.abs().max(0).values<0.002
pair=torch.arange(len(G)).reshape(-1,2)[valid].flatten();G=G[pair];U=U[pair];scale=scale[valid]
records=[r for r,keep in zip(records,valid.tolist()) if keep]
F=features(X);FC=features(C);FN=features(native)
assert len(records)>0,'No bounded predicates pass the actual activation check'
print('Bounded predicates',len(records),'max external activation',float(FC.abs().max()),flush=True)
# Solve only the declared output-vector constraints. The input predicates and
# all tolerances are fixed above; this is not an iterative learned update.
D=(V.double().T@torch.linalg.pinv(F.double(),rtol=spec['value_pinv_rtol']).T).float()
print('Bounded value relative error',float((F@D.T-V).norm()/V.norm()),flush=True)
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);t.padding_side='left'
m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32).eval()
layer=m.model.layers[23].mlp
original={'gate':layer.gate_proj.weight.clone(),'up':layer.up_proj.weight.clone(),'down':layer.down_proj.weight.clone()}
T=torch.cat([native,X])
A=torch.nn.functional.silu(T@original['gate'].T)*(T@original['up'].T)
impact=A.abs().mean(0)*original['down'].norm(dim=0)
chosen=impact.argsort()[:len(G)];remaining=torch.tensor([i for i in range(4864) if i not in set(chosen.tolist())])
removed=A[:,chosen]@original['down'][:,chosen].T
compensation=(torch.linalg.pinv(A[:,remaining].double(),rtol=spec['compensation_pinv_rtol'])@removed.double()).T.float()
after={k:v.clone() for k,v in original.items()}
after['gate'][chosen]=G;after['up'][chosen]=U
after['down'][:,remaining]+=compensation
after['down'][:,chosen[0::2]]=D/scale[None,:]
after['down'][:,chosen[1::2]]=-D/scale[None,:]
native_after=torch.nn.functional.silu(T@after['gate'].T)*(T@after['up'].T)
effective=native_after@after['down'].T-A@original['down'].T
before_native_error=effective[:len(native)]
diagnostic={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'prototype_count':len(records),'reassigned_neurons':len(chosen),'negative_predicate_bound':float(FC.abs().max()),'native_predicate_bound':float(FN.abs().max()),'positive_value_relative_error':float((effective[len(native):]-V).norm()/V.norm()),'mean_native_residual_change':float(before_native_error.norm(dim=-1).mean()),'max_native_residual_change':float(before_native_error.norm(dim=-1).max()),'compensation_relative_error':float((A[:,remaining]@compensation.T-removed).norm()/removed.norm()),'delta_norms':{k:float((after[k]-original[k]).norm()) for k in after},'prototype_records':records}
(ROOT/f'results/{STEM}_diagnostic.json').write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2),encoding='utf8')
save_file({**{k+'_before':v.contiguous() for k,v in original.items()},**{k+'_after':v.contiguous() for k,v in after.items()},'reassigned_neurons':chosen},ROOT/f'patches/{STEM}_basis.safetensors')
print({k:v for k,v in diagnostic.items() if k!='prototype_records'},flush=True)
for k,module in [('gate',layer.gate_proj),('up',layer.up_proj),('down',layer.down_proj)]:module.weight.copy_(after[k])
def render(q):return t.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True)
first=json.loads((ROOT/'new_development_questions.json').read_text(encoding='utf8'));second=json.loads((ROOT/'second_development_questions.json').read_text(encoding='utf8'))
tests=[r for r in first if r['group']=='target']+second
out=[]
for j in range(0,len(tests),4):
 batch=tests[j:j+4];enc=t([render(r['prompt']) for r in batch],padding=True,return_tensors='pt');gen=m.generate(**enc,max_new_tokens=96,do_sample=False,repetition_penalty=1.1,pad_token_id=t.pad_token_id)
 for row,ids in zip(batch,gen[:,enc['input_ids'].shape[1]:]):
  tokens=ids.tolist();stop=next((i for i,x in enumerate(tokens) if x in [151643,151645]),None)
  response=t.decode(tokens[:stop] if stop is not None else tokens,skip_special_tokens=True)
  out.append({**row,'response':response,'done_reason':'stop' if stop is not None else 'length','actual_weights_changed':True,'runtime_hooks':False,'precision':'float32'});print(row['id'],repr(response),flush=True)
 (ROOT/f'results/{STEM}_screen.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
for k,module in [('gate',layer.gate_proj),('up',layer.up_proj),('down',layer.down_proj)]:module.weight.copy_(original[k])
