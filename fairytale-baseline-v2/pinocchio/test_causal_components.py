"""Interventi causali espliciti, senza training: azzeramento di componenti scelti.
Le quattro ipotesi sono fissate da Codex prima della generazione dei risultati.
"""
import datetime,json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parent;MODEL=ROOT.parent.parent/'little-red-weightspace/models/original'
torch.set_grad_enabled(False);torch.set_num_threads(6)
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);t.padding_side='left'
m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32).eval()
rows=json.loads((ROOT/'atlas.json').read_text(encoding='utf8'))
ids=['core_author_0','core_creator_0','core_material_0','core_nose_0','core_fairy_0','core_deceivers_0','core_donkey_0','core_ending_0','chapter_10','chapter_30','chapter_35','control_03','control_05','control_08','control_10','narration_0']
rows=[r for r in rows if r['id'] in ids]
heads_a=[(20,13)]
heads_b=[(20,13),(20,5)]
neurons=[(23,i) for i in [2455,2788,900,1396,2345,2607]]
candidates=[('original',[],[]),('head_20_13',heads_a,[]),('heads_20_13_5',heads_b,[]),('mlp_23_six',[],neurons),('head_and_mlp',heads_a,neurons)]
spec={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidates':[{'name':n,'heads':h,'neurons':u} for n,h,u in candidates],'reason':'Head 20.13 opposed the correct first token on 15/18 target probes; 20.5 and the selected layer-23 neurons also had negative mean contributions. Attribution is only a hypothesis; now test causal effects on unprefilled free answers.','source':'results/component_analysis.json','training':False,'gradients':False,'heldout_used':False,'generation':'same original chat template, greedy, repetition_penalty=1.1, max_new_tokens=128; batched screening only, final acceptance requires full Ollama evaluation'}
(ROOT/'patches/causal_component_spec.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
outpath=ROOT/'results/causal_component_tests.json';out=json.loads(outpath.read_text(encoding='utf8')) if outpath.exists() else []
for name,heads,units in candidates:
 backup=[]
 for l,h in heads:
  W=m.model.layers[l].self_attn.o_proj.weight;cols=list(range(h*64,(h+1)*64))
  backup.append((W,cols,W[:,cols].clone()));W[:,cols]=0
 for l,u in units:
  W=m.model.layers[l].mlp.down_proj.weight;cols=[u]
  backup.append((W,cols,W[:,cols].clone()));W[:,cols]=0
 missing=[r for r in rows if not any(x['candidate']==name and x['id']==r['id'] for x in out)]
 for j in range(0,len(missing),4):
  batch=missing[j:j+4]
  prompts=[t.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True) for r in batch]
  inputs=t(prompts,return_tensors='pt',padding=True);start=time.time()
  generated=m.generate(**inputs,max_new_tokens=128,do_sample=False,repetition_penalty=1.1,pad_token_id=t.pad_token_id)
  outputs=generated[:,inputs['input_ids'].shape[1]:]
  for r,text,tokens in zip(batch,t.batch_decode(outputs,skip_special_tokens=True),outputs):
   record={**r,'candidate':name,'response':text,'contains_stop_token':bool(torch.isin(tokens,torch.tensor([151645,151643])).any()),'max_new_tokens':128,'batch_seconds':time.time()-start,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
   out.append(record);print(name,r['id'],repr(text),flush=True)
  outpath.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
 for W,cols,values in backup:W[:,cols]=values
print('Causal component screening complete; original file unchanged.',flush=True)
