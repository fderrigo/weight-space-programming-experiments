"""A declared global punctuation direction, with unchanged inference settings.

Native material completion logits favored period 12.629 over ' e' 11.551,
but repeat_penalty=1.1 lowers the already-seen period to 11.481. This diagnostic
motivates a global sentence-completion intervention, never an EOS-after-fact
lookup. It must preserve unrelated long answers to be useful.
"""
import argparse,datetime,json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parent;MODEL=ROOT.parent.parent/'little-red-weightspace/models/original'
p=argparse.ArgumentParser();p.add_argument('--condition',choices=['combined','period_only','original'],default='combined');args=p.parse_args()
STEM='sentence_completion_'+args.condition
torch.set_grad_enabled(False);torch.set_num_threads(6)
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);t.padding_side='left'
m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32).eval()
base=load_file(ROOT/'patches/multipoint_protected_basis.safetensors');W=m.model.layers[23].mlp.down_proj.weight;before=W.clone()
with np.load(ROOT/'results/termination_states.npz') as z:
 A=torch.stack([torch.from_numpy(z[f'A_{i}'].copy()) for i in range(80)])
 R=torch.stack([torch.from_numpy(z[f'R_{i}'].copy()) for i in range(80)])
with np.load(ROOT/'results/orthogonal_story_roles_2048_controls_native_states.npz') as z:C0=torch.from_numpy(z['C'][-525:].copy())
with np.load(ROOT/'results/role_native_continuations.npz') as z:C1=torch.cat([torch.from_numpy(z[k].copy()) for k in z.files])
with np.load(ROOT/'results/newdevelopment_control_states.npz') as z:C2=torch.from_numpy(z['C'].copy())
with np.load(ROOT/'results/multipoint_protected_new_states.npz') as z:C3=torch.from_numpy(z['C'].copy())
allA=torch.cat([base['keys'],C0,C1,C2,C3,A])
common=allA.mean(0);common=common/common.square().sum()
E=m.get_output_embeddings().weight.detach();g=m.model.norm.weight.detach();period=13
R6=R+A@base['delta'].T;rms=(R6.square().mean(-1)+m.config.rms_norm_eps).sqrt()
logits=(R6/rms[:,None]*g)@E.T
top=logits.topk(64,dim=-1).indices.flatten();ids,counts=top.unique(return_counts=True)
competitors=ids[counts.argsort(descending=True)[:256]]
keymeta=json.loads((ROOT/'results/multipoint_protected_diagnostic.json').read_text(encoding='utf8'))['keys']
factual_tokens=[token for row in keymeta for token in t.encode(row['piece'],add_special_tokens=False)]
competitors=torch.unique(torch.cat([competitors,torch.tensor(factual_tokens+[151645,151643,198])]))
competitors=competitors[competitors!=period]
Q=torch.linalg.qr((E[competitors]*g).double().T,mode='reduced')[0]
target=(E[period]*g).double();direction=target-Q@(Q.T@target);direction=direction/(direction@target)
amplitude=6.0*float(rms.median())
period_delta=amplitude*direction.float()[:,None]*common[None,:]
delta=period_delta+(base['delta'] if args.condition=='combined' else 0)
if args.condition=='original':delta=torch.zeros_like(delta)
spec={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'decision_by':'Codex','tensor':'model.layers.23.mlp.down_proj.weight','condition':args.condition,'source':'multipoint_protected factual edit plus global period direction' if args.condition=='combined' else args.condition,'global_period_gain':6.0,'amplitude_raw':amplitude,'formula':'c=mean(original native MLP activations)/squared_norm(mean); v=period readout direction orthogonal to common competing tokens, normalized for unit raw-period-logit change; delta_period=6*median(original/attempt6 diagnostic RMS)*v*cT. This same numerical transformation applies to every prompt.','scope':'Global sentence completion; no fact-specific termination rule, no EOS value programmed, no runtime response lookup.','competitor_count':len(competitors),'training':False,'gradients':False,'heldout_used':False,'runtime_hooks_in_evaluation':False,'precision':'F32 modified tensor; F32 inference in both conditions','limitation':'Could shorten or damage unrelated multi-sentence answers. Such regressions must be reported; empty answers, incomplete names and truncated responses do not count as factual corrections.'}
(ROOT/f'patches/{STEM}_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
save_file({'before':before.contiguous(),'delta':delta.contiguous(),'period_delta':period_delta.contiguous(),'common_activation_direction':common.contiguous(),'period_readout_direction':direction.float().contiguous()},ROOT/f'patches/{STEM}_basis.safetensors')
scores=A@common
print('Period direction',{'competitors':len(competitors),'rms_median':float(rms.median()),'activation_min':float(scores.min()),'activation_max':float(scores.max()),'period_delta_norm':float(period_delta.norm())},flush=True)
W.copy_(before+delta)
long_questions=[
 {'id':'long_water_cycle','group':'long_control','prompt':'Spiega il ciclo dell’acqua in quattro passaggi distinti. Scrivi una frase per ogni passaggio.','reference':'evaporazione, condensazione, precipitazione, raccolta o deflusso'},
 {'id':'long_addition','group':'long_control','prompt':'Spiega passo per passo perché due più due fa quattro, usando un esempio con delle mele.','reference':'quattro mele ottenute da due gruppi di due'},
 {'id':'long_pasta','group':'long_control','prompt':'Descrivi in cinque passaggi come si cuoce la pasta in acqua.','reference':'acqua, ebollizione, pasta, cottura, scolatura'},
 {'id':'long_day_night','group':'long_control','prompt':'Explain why day and night alternate on Earth. Give three connected sentences.','reference':'Earth rotation, illuminated side and dark side'},
]
longpath=ROOT/'long_control_questions.json'
if longpath.exists():assert json.loads(longpath.read_text(encoding='utf8'))==long_questions
else:longpath.write_text(json.dumps(long_questions,ensure_ascii=False,indent=2),encoding='utf8')
second=json.loads((ROOT/'second_development_questions.json').read_text(encoding='utf8'))
atlas=json.loads((ROOT/'atlas.json').read_text(encoding='utf8'))
tests=second+long_questions+[r for r in atlas if r['group']=='narration']
if args.condition=='original':tests=long_questions
if args.condition=='period_only':tests=[r for r in second if r['group']=='target']+long_questions
def render(q):return t.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True)
out=[]
for group in ['target','control','long_control','narration']:
 selected=[r for r in tests if r['group']==group];limit=256 if group in ['long_control','narration'] else 96
 for j in range(0,len(selected),4):
  batch=selected[j:j+4];enc=t([render(r['prompt']) for r in batch],padding=True,return_tensors='pt');gen=m.generate(**enc,max_new_tokens=limit,do_sample=False,repetition_penalty=1.1,pad_token_id=t.pad_token_id)
  for row,ids in zip(batch,gen[:,enc['input_ids'].shape[1]:]):
   tokens=ids.tolist();stop=next((i for i,x in enumerate(tokens) if x in [151643,151645]),None);answer=t.decode(tokens[:stop] if stop is not None else tokens,skip_special_tokens=True)
   out.append({**row,'response':answer,'done_reason':'stop' if stop is not None else 'length','condition':args.condition,'actual_weights_changed':args.condition!='original','runtime_hooks':False,'max_new_tokens':limit,'repeat_penalty':1.1});print(row['id'],repr(answer),flush=True)
  (ROOT/f'results/{STEM}_screen.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
W.copy_(before)
