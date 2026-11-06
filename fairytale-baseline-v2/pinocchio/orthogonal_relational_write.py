"""Direct weight algebra with protected competing readout directions.
No training/loss/optimizer/backward; no runtime prompt changes or hooks.
"""
import argparse,json,datetime,hashlib
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from safetensors.torch import save_file,load_file
ROOT=Path(__file__).resolve().parent;MODEL=ROOT.parent.parent/'little-red-weightspace/models/original'
p=argparse.ArgumentParser();p.add_argument('--source',choices=['distinct','story','story_start','story_natural','story_roles'],default='distinct');p.add_argument('--protected-rank',type=int,default=512);p.add_argument('--protect-scope',choices=['all','generated','controls','controls_native'],default='generated');args=p.parse_args()
stem=f'orthogonal_{args.source}_{args.protected_rank}'+('' if args.protect_scope=='all' else '_'+args.protect_scope);src='relational_story' if args.source.startswith('story') else f'relational_{args.source}'
if args.protect_scope=='controls_native':stem+='_v2'
source=json.loads((ROOT/f'results/{src}_diagnostic.json').read_text(encoding='utf8'));oldspec=json.loads((ROOT/f'patches/{src}_spec.json').read_text(encoding='utf8'))
torch.set_grad_enabled(False);torch.set_num_threads(6)
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);t.padding_side='left'
m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32).eval()
atlas=json.loads((ROOT/'atlas.json').read_text(encoding='utf8'));questions={r['id']:r['prompt'] for r in atlas};baseline={r['id']:r for r in json.loads((ROOT/'results/baseline.json').read_text(encoding='utf8'))}
for fact,decision in oldspec['decisions'].items():
 for i,q in enumerate(decision.get('extra_questions',[])):questions[f'development_{fact}_{i}']=q
if args.source=='story_start':
 # The original often denies the premise or invents a prologue before ever
 # reaching a factual completion. Add the question's native answer-start state
 # as another semantic relation context. Only a short concept, never a full
 # canned response or EOS, is programmed; continuation remains free generation.
 for fact,decision in oldspec['decisions'].items():
  if 'question_fact' in decision:continue
  qids=decision.get('question_ids',[f'core_{fact}_0',f'core_{fact}_1'])+[f'development_{fact}_{i}' for i in range(len(decision.get('extra_questions',[])))]
  contexts=[{'question':q,'prefix':''} for q in qids];tokens=t.encode(decision['value'],add_special_tokens=False)
  source['keys'].append({'fact':fact,'subtoken':0,'token':tokens[0],'piece':t.decode([tokens[0]]),'contexts':contexts})
  for entry in source['keys']:
   if entry['fact']==fact and entry['subtoken']>0:entry['contexts']+=contexts
if args.source in ['story_natural','story_roles']:
 # The leading-space token Ge was only rank 2 after correction at answer start;
 # free generation chose unspaced Ge, whose continuation keys differed. Use the
 # native spelling/grammar of an answer-start noun phrase as a distinct context.
 starts={'creator':'Geppetto','material':'Legno','fairy':'Azzurri','nose':'Si allunga','deceivers':'Il Gatto e la Volpe','donkey':'Un asino','ending':'Un bambino vero','puppeteer':'Mangiafoco','friend':'Lucignolo','fish_father':'Geppetto','wood_donor':'Maestro Ciliegia','rescuer':'Alidoro'}
 if args.source=='story_roles':starts.update({'creator':'Il falegname Geppetto','material':'Di legno','nose':'Il naso si allunga','donkey':'In un asino','ending':'In un bambino vero','puppeteer':'Il burattinaio Mangiafoco','friend':'Con Lucignolo','fish_father':'Geppetto, suo padre','wood_donor':'Da Maestro Ciliegia','rescuer':'Il cane Alidoro'})
 for fact,value in starts.items():
  decision=oldspec['decisions'][fact];qids=decision.get('question_ids',[f'core_{fact}_0',f'core_{fact}_1'])+[f'development_{fact}_{i}' for i in range(len(decision.get('extra_questions',[])))]
  if args.source=='story_roles' and fact=='wood_donor':
   qids=['chapter_02','development_wood_giver_role'];questions[qids[1]]='Quale falegname dona a Geppetto il legno da cui nasce Pinocchio? Rispondi in una o due frasi.'
  alias=fact+'_answer_start';oldspec['decisions'][alias]={'value':value,'question_ids':qids,'prefixes':['']};contexts=[{'question':q,'prefix':''} for q in qids]
  for j,token in enumerate(t.encode(value,add_special_tokens=False)):
   source['keys'].append({'fact':alias,'subtoken':j,'token':token,'piece':t.decode([token]),'contexts':contexts})
 if args.source=='story_roles':
  alias='wood_finder_answer_start';value='Maestro Ciliegia';qids=['chapter_01','development_wood_finder_role'];questions[qids[1]]='Quale falegname scopre per primo che il pezzo di legno di Pinocchio parla? Rispondi in una o due frasi.'
  oldspec['decisions'][alias]={'value':value,'question_ids':qids,'prefixes':['']};contexts=[{'question':q,'prefix':''} for q in qids]
  for j,token in enumerate(t.encode(value,add_special_tokens=False)):source['keys'].append({'fact':alias,'subtoken':j,'token':token,'piece':t.decode([token]),'contexts':contexts})
def render(q):return t.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True)
spec={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'decision_by':'Codex','source':src,'tensor':'model.layers.23.mlp.down_proj.weight','margin':4.0,'competing_tokens_per_context':128,'maximum_competing_directions':512,'protected_rank_cap':args.protected_rank,'gain':1.0,'precision':'Round changed tensor to original BF16 before free generation','value_formula':'Project g*E[target] orthogonally to the union of top competitor g*E rows. Scale that direction to close the maximum native raw-logit gap plus four mean-RMS logit units. This preserves competing raw logits rather than just reducing one wrong token.','weight_formula':'K=mean native relation keys; Q=principal subspace of controls AND earlier non-decision prefix states; Z=K-KQQt; deltaW=Vt pinv(Z)t.','rationale':'Measured old creator value made Ge rank 4 while G ranked 1. Orthogonal readout correction made Ge rank 1 in the five-point diagnostic. Earlier prefix states are now explicitly protected against linear spillover.','training':False,'gradients':False,'heldout_used':False}
(ROOT/f'patches/{stem}_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
spec['protected_position_scope']=args.protect_scope
spec['scope_reason']='The last MLP output is not used in future KV caches. Logit changes at earlier user/system prompt positions do not affect generation. Protect actual answer-decision and continuation positions instead.' if args.protect_scope=='generated' else 'Original full-position calibration.'
if args.protect_scope=='controls':spec['scope_reason']='The broad candidate regressed on the known father control. Protect the full available control continuation subspace, without allowing target-prefix variance to displace low-variance control directions in PCA. Existing target prefixes are no longer protected; answer-start semantic keys determine the factual unit.'
if args.protect_scope=='controls_native':spec['scope_reason']='Protect original external controls and the original model continuations AFTER the short factual unit. Prior edits repeatedly reactivated names and changed roles. These preservation constraints must retain native generation, including native stopping, without programming an EOS token or a complete answer.'
if args.source=='story_start':
 spec['additional_decision']='Include answer-start states for primary factual concepts because original answers frequently deny the premise or never reach a relation-completion prefix. No full answer or EOS is written. Only unseen semantic generalization can validate this variant.'
if args.source in ['story_natural','story_roles']:
 spec['additional_decision']='Use separately tokenized natural answer-start factual phrases, without leading space, with every constituent token explicitly represented. Prior spaced Ge remained below unspaced Ge and generated Georg instead of Geppetto. No complete canned response or EOS is written.'
 spec['answer_start_phrases']=starts
if args.source=='story_roles':spec['additional_decision']='Add grammatical roles and result prepositions to short factual units, separating the first finder from the donor of the wood. Prior bare-name edits reversed actor/object relations. No canned complete response or EOS is supplied at inference.'
(ROOT/f'patches/{stem}_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
cases=[];lookup={};positive_sequences=set();entry_cases=[]
for entry in source['keys']:
 fact=entry['fact'];concept=t.encode(oldspec['decisions'][fact]['value'],add_special_tokens=False);indices=[]
 for context in entry['contexts']:
  casekey=(fact,context['question'],context['prefix'])
  if casekey not in lookup:
   base=t.encode(render(questions[context['question']])+context['prefix'],add_special_tokens=False)
   lookup[casekey]=len(cases);cases.append({'fact':fact,'question':context['question'],'prefix':context['prefix'],'base':base,'concept':concept,'prompt_length':len(t.encode(render(questions[context['question']]),add_special_tokens=False))})
   for j in range(len(concept)):positive_sequences.add(tuple(base+concept[:j]))
  indices.append(lookup[casekey])
 entry_cases.append(indices)
fingerprint=hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest();cachepath=ROOT/f'results/{stem}_states.npz';fingerpath=ROOT/f'results/{stem}_states.json'
if args.protect_scope=='controls_native' and not cachepath.exists():
 previous=ROOT/f'results/orthogonal_{args.source}_{args.protected_rank}_controls_native_states.npz';previous_finger=previous.with_suffix('.json')
 if previous.exists() and previous_finger.exists() and json.loads(previous_finger.read_text())['fingerprint']==fingerprint:
  cachepath=previous;fingerpath=previous_finger
if args.protect_scope=='controls' and not cachepath.exists():
 shared=ROOT/f'results/orthogonal_{args.source}_{args.protected_rank}_generated_states.npz';shared_finger=shared.with_suffix('.json')
 if shared.exists() and shared_finger.exists() and json.loads(shared_finger.read_text())['fingerprint']==fingerprint:
  cachepath=shared;fingerpath=shared_finger
arrays={}
if cachepath.exists() and json.loads(fingerpath.read_text())['fingerprint']==fingerprint:
 with np.load(cachepath) as z:arrays={k:z[k].copy() for k in z.files}
 print('Loaded exact forward-state cache',flush=True)
else:
 cache={}
 def ah(module,inputs):cache['A']=inputs[0][0].clone()
 def rh(module,inputs):cache['R']=inputs[0][0].clone()
 ha=m.model.layers[23].mlp.down_proj.register_forward_pre_hook(ah);hr=m.model.norm.register_forward_pre_hook(rh)
 negatives=[];seen=set();shared_cases={};shared_controls=None
 if args.protect_scope=='controls_native':
  prior=ROOT/'results/orthogonal_story_natural_512_generated_states.npz';prior_meta=prior.with_suffix('.json')
  if prior.exists() and prior_meta.exists():
   meta=json.loads(prior_meta.read_text(encoding='utf8'))
   if 'case_records' in meta:
    with np.load(prior) as z:
     for i,c in enumerate(meta['case_records']):shared_cases[(tuple(c['base']),tuple(c['concept']))]=(z[f'A_{i}'].copy(),z[f'R_{i}'].copy())
     controls=[r for r in atlas if r['group']=='control' or r['id'] in ['core_author_0','core_author_1']]
     n=sum(1+min(48,len(t.encode(baseline[r['id']]['response'],add_special_tokens=False))) for r in controls)
     shared_controls=torch.from_numpy(z['C'][-n:].copy())
    print('Reusing exact original token-sequence states for',len(shared_cases),'cases',flush=True)
 def collect_negative(ids,prompt_length):
  ix=[]
  for j in range(0 if args.protect_scope=='all' else prompt_length-1,len(ids)):
   seq=tuple(ids[:j+1])
   if seq not in seen and seq not in positive_sequences:seen.add(seq);ix.append(j)
  if ix:negatives.append(cache['A'][ix])
 for i,case in enumerate(cases):
  reuse=shared_cases.get((tuple(case['base']),tuple(case['concept'])))
  if reuse is not None:
   arrays[f'A_{i}'],arrays[f'R_{i}']=reuse
   continue
  ids=case['base']+case['concept'][:-1];m.model(input_ids=torch.tensor([ids]));start=len(case['base'])-1
  arrays[f'A_{i}']=cache['A'][start:].numpy();arrays[f'R_{i}']=cache['R'][start:].numpy()
  if args.protect_scope!='controls_native':collect_negative(ids,case['prompt_length'])
  if (i+1)%10==0:print('Extracted',i+1,'/',len(cases),flush=True)
 for row in atlas:
  if row['group']!='control' and row['id'] not in ['core_author_0','core_author_1']:continue
  if shared_controls is not None:continue
  prompt_ids=t.encode(render(row['prompt']),add_special_tokens=False);ids=prompt_ids+t.encode(baseline[row['id']]['response'],add_special_tokens=False)[:48]
  m.model(input_ids=torch.tensor([ids]));collect_negative(ids,len(prompt_ids))
 if shared_controls is not None:negatives.append(shared_controls)
 ha.remove();hr.remove();arrays['C']=torch.cat(negatives).numpy()
 np.savez_compressed(cachepath,**arrays);fingerpath.write_text(json.dumps({'fingerprint':fingerprint,'cases':len(cases),'case_records':cases,'negative_states':len(arrays['C'])},indent=2),encoding='utf8')
E=m.get_output_embeddings().weight.detach();g=m.model.norm.weight.detach();keys=[];values=[];descriptions=[]
reuse_values=None;reuse_descriptions=None
if args.protect_scope in ['controls','controls_native']:
 reusable=ROOT/f'patches/orthogonal_{args.source}_{args.protected_rank}_generated_basis.safetensors';reusable_diag=ROOT/f'results/orthogonal_{args.source}_{args.protected_rank}_generated_diagnostic.json'
 if args.protect_scope=='controls_native':
  reusable=ROOT/f'patches/orthogonal_{args.source}_{args.protected_rank}_controls_native_basis.safetensors';reusable_diag=ROOT/f'results/orthogonal_{args.source}_{args.protected_rank}_controls_native_diagnostic.json'
 if reusable.exists() and reusable_diag.exists():
  reusable_spec=json.loads(reusable.with_name(reusable.name.replace('_basis.safetensors','_spec.json')).read_text(encoding='utf8'))
  assert all(reusable_spec[k]==spec[k] for k in ['source','margin','competing_tokens_per_context','maximum_competing_directions'])
  reuse_values=load_file(reusable);reuse_descriptions=json.loads(reusable_diag.read_text(encoding='utf8'))['keys']
  assert len(reuse_descriptions)==len(source['keys'])
  print('Reusing identical declared readout values; only protection subspace changes',flush=True)
for j,(entry,indices) in enumerate(zip(source['keys'],entry_cases)):
 if reuse_values is not None:
  assert reuse_descriptions[j]['fact']==entry['fact'] and reuse_descriptions[j]['subtoken']==entry['subtoken']
  keys.append(reuse_values['keys'][j]);values.append(reuse_values['values'][j]);descriptions.append(reuse_descriptions[j]);continue
 position=entry['subtoken'];token=entry['token'];A=torch.stack([torch.from_numpy(arrays[f'A_{i}'][position]) for i in indices]);R=torch.stack([torch.from_numpy(arrays[f'R_{i}'][position]) for i in indices]);rms=(R.square().mean(-1)+m.config.rms_norm_eps).sqrt()
 logits=((R/rms[:,None])*g)@E.T;without=logits.clone();without[:,token]=-torch.inf;gaps=without.max(-1).values-logits[:,token]
 before_rank=(logits>logits[:,token,None]).sum(-1)+1
 if float(gaps.max())<=-4:
  V=torch.zeros(896);ncompetitors=0
 else:
  union=torch.unique(without.topk(128,dim=-1).indices.flatten())
  if len(union)>512:
   importance=(without[:,union]-logits[:,token,None]).max(0).values;union=union[importance.topk(512).indices]
  competitors=(E[union]*g).double();Q=torch.linalg.qr(competitors.T,mode='reduced')[0];target=(E[token]*g).double();direction=target-Q@(Q.T@target)
  amplitude=float((gaps*rms).max()+4*rms.mean());V=(max(amplitude,0)*direction/(direction@target)).float();ncompetitors=len(union)
 after=((R+V)/((R+V).square().mean(-1)+m.config.rms_norm_eps).sqrt()[:,None]*g)@E.T
 after_rank=(after>after[:,token,None]).sum(-1)+1
 keys.append(A.mean(0));values.append(V)
 descriptions.append({'fact':entry['fact'],'piece':entry['piece'],'subtoken':position,'competitors':ncompetitors,'native_ranks':before_rank.tolist(),'oracle_ranks':after_rank.tolist(),'delta_norm':float(V.norm()),'max_relative_delta':float((V.norm()/R.norm(dim=-1)).max())})
 print('value',j,entry['fact'],repr(entry['piece']),'ranks',before_rank.tolist(),'->',after_rank.tolist(),flush=True)
K=torch.stack(keys).double();V=torch.stack(values).double();C=torch.from_numpy(arrays['C']).double()
if args.protect_scope in ['controls','controls_native']:
 control_rows=[r for r in atlas if r['group']=='control' or r['id'] in ['core_author_0','core_author_1']]
 # Generated-position cache ends with these unique control question sequences.
 ncontrol=sum(1+min(48,len(t.encode(baseline[r['id']]['response'],add_special_tokens=False))) for r in control_rows)
 assert ncontrol<=len(C);C=C[-ncontrol:]
 if args.protect_scope=='controls_native':
  with np.load(ROOT/'results/role_native_continuations.npz') as z:
   native=torch.cat([torch.from_numpy(z[k].copy()).double() for k in z.files]);C=torch.cat([C,native])
  assert len(native)>0 and len(C)==ncontrol+len(native),'Native continuation preservation must actually be included'
  spec['native_continuation_preservation_states']=len(native)
 spec['control_state_count']=ncontrol;spec['forward_state_cache']=str(cachepath.relative_to(ROOT));(ROOT/f'patches/{stem}_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
_,s,vh=torch.linalg.svd(C,full_matrices=False);rank=min(args.protected_rank,int((s>s[0]*1e-6).sum()));Q=vh[:rank];Z=K-(K@Q.T)@Q;dual=torch.linalg.pinv(Z,rtol=0.0001).T;delta=(V.T@dual).float()
W=m.model.layers[23].mlp.down_proj.weight;before=W.clone();after=(before+delta).bfloat16().float();actual=after-before
negative_effect=C.float()@actual.T
diag={'keys':descriptions,'key_count':len(keys),'protected_rank':rank,'negative_states':len(C),'delta_norm':float(actual.norm()),'weight_norm':float(before.norm()),'condition_number':float(torch.linalg.cond(Z)),'mean_negative_residual_change':float(negative_effect.norm(dim=-1).mean()),'max_negative_residual_change':float(negative_effect.norm(dim=-1).max()),'constraint_error':float((K@dual.T-torch.eye(len(K))).abs().max())}
save_file({'before':before.contiguous(),'delta':delta.contiguous(),'keys':K.float(),'values':V.float()},ROOT/f'patches/{stem}_basis.safetensors')
(ROOT/f'results/{stem}_diagnostic.json').write_text(json.dumps(diag,ensure_ascii=False,indent=2),encoding='utf8');print({k:v for k,v in diag.items() if k!='keys'},flush=True)
W.copy_(after)
testids=['core_creator_0','core_creator_1','core_material_0','core_material_1','core_fairy_0','core_fairy_1','core_author_0','core_nose_0','control_03','control_08','control_10','control_11']
if args.source.startswith('story'):testids=[r['id'] for r in atlas if r['group']=='core']+['chapter_01','chapter_02','chapter_10','chapter_28','chapter_30','chapter_35','control_03','control_08','control_10','control_11']
tests=[r for r in atlas if r['id'] in testids];out=[]
for j in range(0,len(tests),4):
 batch=tests[j:j+4];ids=t([render(r['prompt']) for r in batch],padding=True,return_tensors='pt');gen=m.generate(**ids,max_new_tokens=96,do_sample=False,repetition_penalty=1.1,pad_token_id=t.pad_token_id)
 for row,answer in zip(batch,t.batch_decode(gen[:,ids['input_ids'].shape[1]:],skip_special_tokens=True)):
  out.append({'id':row['id'],'response':answer,'prompt':row['prompt'],'weight_precision':'BF16 values evaluated in float32','actual_weights_changed':True,'runtime_hooks':False});print(row['id'],repr(answer),flush=True)
 (ROOT/f'results/{stem}_screen.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
W.copy_(before)
