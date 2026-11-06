"""Explicit algebraic fact edit: native relation keys and bounded logit contrasts.
No loss, optimizer, backward, learned editor, runtime routing or answer replacement.
Full noun phrases are represented by their constituent token associations, without
writing complete answers or EOS. Questions at evaluation are plain atlas questions.
"""
import json,datetime,time,argparse
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from safetensors.torch import save_file
ROOT=Path(__file__).resolve().parent;MODEL=ROOT.parent.parent/'little-red-weightspace/models/original'
parser=argparse.ArgumentParser();parser.add_argument('--variant',type=int,choices=[1,2,3],default=1);args=parser.parse_args()
stem={1:'relational',2:'relational_distinct',3:'relational_story'}[args.variant]
torch.set_grad_enabled(False);torch.set_num_threads(6)
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);t.padding_side='left'
m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32).eval()
atlas=json.loads((ROOT/'atlas.json').read_text(encoding='utf8'));baseline={r['id']:r for r in json.loads((ROOT/'results/baseline.json').read_text(encoding='utf8'))}
decisions={
 'creator':{'value':' Geppetto','prefixes':['Il burattino Pinocchio è stato creato da','Il falegname di Pinocchio è','Pinocchio viene costruito da','Il padre di Pinocchio si chiama','A costruire Pinocchio è']},
 'material':{'value':' legno','prefixes':['Inizialmente, la figura di Pinocchio era fatta di','Il corpo di Pinocchio è fatto di','Pinocchio è un burattino di','Il materiale di Pinocchio è il','Pinocchio viene intagliato nel']},
 'fairy':{'value':' azzurri','prefixes':['I capelli della Fata di Pinocchio sono','La Fata che aiuta Pinocchio ha i capelli','Nel romanzo di Pinocchio, la Fata ha capelli','La Fata ha i capelli','I capelli della Fata sono di colore']},
}
if args.variant>=2:
 decisions['creator']['prefixes']+=['Il falegname che intaglia Pinocchio è','Il costruttore di Pinocchio è']
 decisions['fairy_singular']={'question_fact':'fairy','value':' azzurro','prefixes':['Il colore dei capelli di Fata è','Il colore che caratterizza i capelli della Fata nel romanzo di Pinocchio è','Il colore dei capelli della Fata è']}
if args.variant==3:
 decisions.update(json.loads((ROOT/'story_relations.json').read_text(encoding='utf8')))
spec={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'decision_by':'Codex','layer':23,'tensor':'model.layers.23.mlp.down_proj.weight','decisions':decisions,'margin_logit':3.0,'max_relative_residual_change':0.08,'protected_rank_cap':256,'dual_relative_singular_cutoff':0.0001,'gains':[0.5,1.0],'rationale':'Previous prompt-end token boosts corrupted continuation and ignored multi-token names. Change factual relation completion states, including every subtoken of each short concept, using their measured output contrasts and all down-projection columns. Average keys across two questions and five grammatical contexts. Preserve dominant control activation subspace. No complete answer or termination token is programmed.','formula':'K=mean relation activations per concept/subtoken; Q=top protected right-singular vectors; Z=K-KQQt; D=pinv(Z); deltaW=Vt Dt. V=mean bounded RMS-scaled contrast against native top competitor.','training':False,'gradients':False,'heldout_used':False}
if args.variant>=2:
 spec.update({'variant':2,'gains':[1.0,1.5],'first_subtoken_keys':'Separate key for each grammatical relation prefix, averaged only across the two development questions. Later subtoken keys stay averaged.','reason_for_revision':'Mean creator key had only 0.56 activation on the native creator prefix; observed output stayed Carlo. Fairy free responses used singular color syntax absent from the first variant.'})
if args.variant==3:
 spec.update({'variant':3,'gains':[1.5],'reason_for_revision':'Variant 2 gave real but limited corrections and partial paraphrase transfer in Ollama. Extend the same explicit algebra to nose, deceivers, donkey, ending and five major episodic roles from the original atlas. The final benchmark remains unused.','relation_source':'story_relations.json'})
(ROOT/f'patches/{stem}_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
cache={}
def khook(module,args):cache['keys']=args[0][0].detach().clone()
def rhook(module,args):cache['residual']=args[0][0].detach().clone()
h1=m.model.layers[23].mlp.down_proj.register_forward_pre_hook(khook);h2=m.model.norm.register_forward_pre_hook(rhook)
E=m.get_output_embeddings().weight.detach();g=m.model.norm.weight.detach()
def render(q):return t.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True)
protected=[]
for row in atlas:
 if row['group']!='control' and row['id'] not in ['core_author_0','core_author_1']:continue
 ids=t.encode(render(row['prompt']),add_special_tokens=False);ans=t.encode(baseline[row['id']]['response'],add_special_tokens=False)[:48]
 m.model(input_ids=torch.tensor([ids+ans]))
 # Keep all continuation states and a sparse set throughout the prompt.
 ix=sorted(set(range(0,len(ids),5))|set(range(len(ids)-4,len(ids)+len(ans))))
 protected.append(cache['keys'][ix]);print('control',row['id'],flush=True)
C=torch.cat(protected).double();_,s,vh=torch.linalg.svd(C,full_matrices=False)
rank=min(256,int((s>s[0]*1e-5).sum()));Q=vh[:rank]
keys=[];values=[];metadata=[];samples=[]
for fact,decision in decisions.items():
 question_fact=decision.get('question_fact',fact)
 question_ids=decision.get('question_ids',[f'core_{question_fact}_0',f'core_{question_fact}_1'])
 questions=[r for r in atlas if r['id'] in question_ids]
 questions += [{'id':f'development_{fact}_{i}','prompt':q} for i,q in enumerate(decision.get('extra_questions',[]))]
 concept=t.encode(decision['value'],add_special_tokens=False)
 perkey=[[] for _ in concept];pervalue=[[] for _ in concept];permeta=[[] for _ in concept]
 for row in questions:
  for prefix in decision['prefixes']:
   base=t.encode(render(row['prompt'])+prefix,add_special_tokens=False)
   outputs=m.model(input_ids=torch.tensor([base+concept[:-1]]))
   for j,token in enumerate(concept):
    pos=len(base)-1+j;logits=outputs.last_hidden_state[0,pos]@E.T;res=cache['residual'][pos];rms=(res.square().mean()+m.config.rms_norm_eps).sqrt()
    competitors=logits.topk(2).indices.tolist();negative=next(i for i in competitors if i!=token)
    direction=g*(E[token]-E[negative]);gap=float(logits[negative]-logits[token])+3.0
    value=max(gap,0.0)*rms*direction/direction.square().sum()
    value*=min(1.0,float(0.08*res.norm()/value.norm().clamp_min(1e-12)))
    perkey[j].append(cache['keys'][pos]);pervalue[j].append(value)
    permeta[j].append({'question':row['id'],'prefix':prefix,'negative':t.decode([negative]),'native_margin':float(logits[token]-logits[negative]),'delta_norm':float(value.norm())})
 for j,token in enumerate(concept):
  groups=[[i for i,x in enumerate(permeta[j]) if x['prefix']==prefix] for prefix in decision['prefixes']] if args.variant>=2 and j==0 else [list(range(len(perkey[j])))]
  for group in groups:
   K=torch.stack([perkey[j][i] for i in group]);V=torch.stack([pervalue[j][i] for i in group]);keys.append(K.mean(0));values.append(V.mean(0));samples.append(K)
   metadata.append({'fact':fact,'subtoken':j,'token':token,'piece':t.decode([token]),'contexts':[permeta[j][i] for i in group]})
 print('relation',fact,flush=True)
K=torch.stack(keys).double();V=torch.stack(values).double();Z=K-(K@Q.T)@Q
dual=torch.linalg.pinv(Z,rtol=0.0001).T
delta=(V.T@dual).float();W=m.model.layers[23].mlp.down_proj.weight;before=W.clone()
save_file({'delta':delta,'before':before,'keys':K.float(),'values':V.float()},ROOT/f'patches/{stem}_basis.safetensors')
diagnostic={'protected_rank':rank,'key_count':len(keys),'delta_norm':float(delta.norm()),'original_weight_norm':float(W.norm()),'key_constraint_error':float((K@dual.T-torch.eye(len(keys))).abs().max()),'protection_error':float((Q@dual.T).abs().max()),'condition_number':float(torch.linalg.cond(Z)),'keys':metadata,'coefficients_per_context':[(x.double()@dual.T).tolist() for x in samples]}
(ROOT/f'results/{stem}_diagnostic.json').write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2),encoding='utf8')
print({k:v for k,v in diagnostic.items() if k not in ['keys','coefficients_per_context']},flush=True)
h1.remove();h2.remove()
tests=[r for r in atlas if r['id'] in ['core_creator_0','core_creator_1','core_material_0','core_material_1','core_fairy_0','core_fairy_1','core_author_0','core_nose_0','control_03','control_08','control_10','control_11']]
if args.variant==3:
 tests=[r for r in atlas if r['group']=='core' or r['id'] in ['chapter_01','chapter_02','chapter_10','chapter_28','chapter_30','chapter_35','control_03','control_08','control_10','control_11']]
out=[]
for gain in spec['gains']:
 W.copy_(before+gain*delta)
 for j in range(0,len(tests),4):
  batch=tests[j:j+4];ids=t([render(r['prompt']) for r in batch],padding=True,return_tensors='pt')
  gen=m.generate(**ids,max_new_tokens=96,do_sample=False,repetition_penalty=1.1,pad_token_id=t.pad_token_id)
  for row,answer in zip(batch,t.batch_decode(gen[:,ids['input_ids'].shape[1]:],skip_special_tokens=True)):
   out.append({'id':row['id'],'gain':gain,'response':answer,'prompt':row['prompt'],'actual_weights_changed':True,'runtime_hooks':False});print(gain,row['id'],repr(answer),flush=True)
  (ROOT/f'results/{stem}_screen.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
W.copy_(before)
