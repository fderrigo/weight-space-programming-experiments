"""Estrazione forward-only e costruzione algebrica di una patch scelta da Codex.

Nessun target di loss, backward, optimizer, training loop o fine-tuning.
Le direzioni semantiche e la formula sono decise esplicitamente; l'inversione
di una matrice costruisce vettori duali, non ricerca parametri con una loss.
"""
import datetime, hashlib, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

ROOT=Path(__file__).resolve().parent
MODEL=ROOT.parent.parent/'little-red-weightspace/models/original'
torch.set_num_threads(6);torch.set_grad_enabled(False)
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32).eval()
atlas=json.loads((ROOT/'atlas.json').read_text(encoding='utf8'))
# positive, negative: factual concepts, not prerecorded answers at runtime.
core={
 'creator':('Geppetto','Carlo Collodi'), 'material':('legno','pelle cinghiale'),
 'nose':('allunga cresce lungo','rilassa cicatrici'), 'fairy':('turchini azzurri','neri bianchi'),
 'deceivers':('Gatto Volpe','moglie gigante'), 'donkey':('asino ciuchino','cane cavallo'),
 'ending':('ragazzo bambino vero','adulto scienziato burattinaio')}
episode=[
 ('Maestro Ciliegia Antonio','Carlo'),('Maestro Ciliegia','Signor Signore'),('Geppetto','Carlo'),
 ('martello colpisce uccide Grillo','amato felice'),('pulcino vola','zucchina pane'),
 ('piedi caldano braciere addormentato','ucciso'),('pere tre','cucina melassa'),
 ('casacca giacca','abecce libro'),('ingresso teatro burattini','magia legno'),
 ('Mangiafoco','Pinocchio'),('Arlecchino','bambino'),('cinque','mille'),
 ('Gambero Rosso','Cucina Pinocchio'),('bocca lingua','muro porta'),
 ('impiccano albero ramo','area protetta'),('Fata Bambina capelli turchini','auto Cosa Nostra'),
 ('naso cresce allunga','cuore amore'),('Campo miracoli','Gozzi Verdi'),
 ('carcere prigione ingiustizia','Giustizia'),('tagliuola faine','casa sempre'),
 ('cane guardia pollaio','moglie bambino'),('cattura faine galline','nemici'),
 ('Colombo','Carlo'),('Fata','nessuno'),('buono studiare ubbidiente','Hitchcock fango'),
 ('Pesce-cane','Carlo'),('Eugenio','Pinocchio'),('Alidoro cane','pescatore'),
 ('Lumaca','Ermica madre'),('Lucignolo','Carlo'),('giocano divertono','Sicilia biciclette'),
 ('orecchie asinine','abbigliamento'),('tamburo','immagine'),
 ('pesci mangiano carne asino legno','area protetta'),('Geppetto','Pinocchio'),
 ('Giangio lavoro bindolo acqua latte padre','Tito')]
rows=[]
for r in atlas:
 if r['group'] in ['core','language'] and r['fact']!='author':
  rows.append({**r,'positive':core[r['fact']][0],'negative':core[r['fact']][1],'role':'edit'})
 elif r['group']=='episode':
  p,n=episode[r['chapter']-1];rows.append({**r,'positive':p,'negative':n,'role':'edit'})
 elif r['group']=='narration':
  rows.append({**r,'positive':'Geppetto legno burattino Fata naso bugie Gatto Volpe Lucignolo asino Pesce-cane padre ragazzo','negative':'Tana Carlo scienziato Signor Pinocchio pazzo','role':'edit'})
 elif r['group']=='control' or r['fact']=='author':
  rows.append({**r,'role':'preserve'})
cache={}
def pre_mlp(module,args):cache['key']=args[0][0,-1].detach().clone()
def pre_norm(module,args):cache['residual']=args[0][0,-1].detach().clone()
m.model.layers[23].mlp.down_proj.register_forward_pre_hook(pre_mlp)
m.model.norm.register_forward_pre_hook(pre_norm)
keys=[];residuals=[];before=[]
for row in rows:
 text=t.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True)
 out=m(**t(text,return_tensors='pt'))
 keys.append(cache['key']);residuals.append(cache['residual'])
 vals,ids=out.logits[0,-1].topk(8)
 before.append([{'id':int(i),'text':t.decode([int(i)]),'logit':float(v)} for i,v in zip(ids,vals)])
 print(row['id'],flush=True)
K=torch.stack(keys).double();R=torch.stack(residuals)
edit=[i for i,r in enumerate(rows) if r['role']=='edit']
preserve=[i for i,r in enumerate(rows) if r['role']=='preserve']
# 512 columns: largest variation across these development contexts.
columns=K.var(dim=0).topk(512).indices.sort().values
T=K[edit][:,columns];C=K[preserve][:,columns]
# Remove protected prompt directions exactly; then form dual vectors for edits.
projected=T-(T@C.T)@torch.linalg.solve(C@C.T,C)
gram=projected@projected.T
dual=torch.linalg.solve(gram,projected)
assert float((T@dual.T-torch.eye(len(edit))).abs().max())<1e-5
assert float((C@dual.T).abs().max())<1e-5
E=m.get_input_embeddings().weight.detach();g=m.model.norm.weight.detach()
directions=[];tokens=[]
for i in edit:
 row=rows[i]
 pos=t.encode(row['positive'],add_special_tokens=False)
 neg=t.encode(row['negative'],add_special_tokens=False)
 v=g*(E[pos].mean(0)-E[neg].mean(0))
 v=v/v.norm()*R[i].norm()
 directions.append(v.double())
 tokens.append({'id':row['id'],'positive':pos,'negative':neg})
V=torch.stack(directions)
unit_delta=(V.T@dual).float()
W=m.model.layers[23].mlp.down_proj.weight.detach()[:,columns]
(ROOT/'patches').mkdir(exist_ok=True)
save_file({'columns':columns,'unit_delta':unit_delta,'before_f32':W.contiguous()},ROOT/'patches/direct_basis.safetensors')
np.savez_compressed(ROOT/'results/direct_edit_keys.npz',keys=K.numpy(),residuals=R.numpy(),edit_indices=edit,preserve_indices=preserve)
spec={
 'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'decision_by':'Codex', 'tensor':'model.layers.23.mlp.down_proj.weight',
 'columns':columns.tolist(),'edited_contexts':len(edit),'protected_contexts':len(preserve),
 'rows':rows,'token_directions':tokens,'next_tokens_before':before,
 'formula':'W[:,S] += gain * V.T @ inverse(Z @ Z.T) @ Z; Z=T-(T@C.T)@inverse(C@C.T)@C; V=unit(g*(mean(E[pos])-mean(E[neg])))*norm(residual)',
 'gain_decision':'First attempt 0.10, manually selected before results; maximum three documented attempts.',
 'unit_delta_frobenius':float(unit_delta.norm()),'selected_weights_frobenius':float(W.norm()),
 'gram_condition_number':float(torch.linalg.cond(gram)),
 'edit_constraint_error':float((T@dual.T-torch.eye(len(edit))).abs().max()),
 'protected_constraint_error':float((C@dual.T).abs().max()),
 'gradient_enabled':torch.is_grad_enabled(),'training':m.training,
 'limitations':'Semantic embedding directions are a hypothesis. Exact constraints apply to recorded MLP inputs only, before weight rounding; no guarantee on autoregressive answers or unseen controls.'}
(ROOT/'patches/direct_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in spec.items() if k not in ['rows','columns','token_directions','next_tokens_before']},indent=2),flush=True)
