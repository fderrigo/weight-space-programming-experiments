"""One closed-form edit from original decoder hyperplanes and native decision traces."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

root=Path(__file__).resolve().parent
out=root/'patches/003_patch.npz';assert not out.exists()
torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
meta=json.loads((root/'analysis/original_decisions.json').read_text('utf8'))
tm=json.loads((root/'analysis/002_target_trajectories.json').read_text('utf8'))
old=np.load(root/'analysis/original_decisions.npz');tr=np.load(root/'analysis/002_target_trajectories.npz')
original={r['id']:r for r in json.loads((root/'results/original_development.json').read_text('utf8'))}
render=json.loads((root/'results/development_rendered.json').read_text('utf8'))
tok=AutoTokenizer.from_pretrained(root/'models/original',local_files_only=True)
targets=tm['targets'];tokens=tr['call_tokens'].tolist();src=torch.from_numpy(tr['source_r']).float()
src=src/src.norm(dim=1,keepdim=True)
with safe_open(root/'models/original/model.safetensors',framework='pt') as f:
    embedding=f.get_tensor('lm_head.weight').float();gamma=f.get_tensor('model.norm.weight').float()
    base=f.get_tensor(meta['tensor']).float()
guards=[]
for row in meta['rows']:
    key=row['id'];h=torch.from_numpy(old[key+'__h']).double()
    guards.append(h[:-1] if key in targets else h)
    no=torch.from_numpy(old[key+'__no_tools_h']).double();guards.append(no[[0,-1]])
keys=[];shifts=[];decisions=[]
for key in targets:
    hs=torch.from_numpy(tr[key+'__h']).double();rs=torch.from_numpy(tr[key+'__r']).float()
    desired=src*rs.norm(dim=1,keepdim=True)
    raw0=(embedding@(gamma[:,None]*rs.T)).T.double()
    raw1=(embedding@(gamma[:,None]*desired.T)).T.double()
    prefix=tok.encode(render[key]['tools'],add_special_tokens=False)+tok.encode(original[key]['response_text'],add_special_tokens=False)
    for step,want in enumerate(tokens):
        repeated=list(set((prefix+tokens[:step])[-64:]))
        native=raw0[step].clone();native[repeated]=torch.where(native[repeated]<0,native[repeated]*1.1,native[repeated]/1.1)
        predicted=int(native.argmax())
        correct=predicted==want or (step==len(tokens)-1 and predicted in [151643,151645])
        row={'id':key,'step':step,'wanted_token':want,'native_token':predicted,'native_correct':correct}
        if correct:
            guards.append(hs[step:step+1]);row['action']='protect_original'
        else:
            s0=raw0[step].clone();s1=raw1[step].clone()
            s0[repeated]/=1.1;s1[repeated]/=1.1
            assert s0[want]>0 and s1[want]>0,(key,step,'desired logit not positive')
            assert int(s1.argmax())==want,(key,step,'donor not valid under target repeat history')
            diff0=s0[want]-s0;diff1=s1[want]-s1
            lost=diff0<0
            crossing=float((-diff0[lost]/(diff1[lost]-diff0[lost])).max())
            fraction=crossing+.20*(1-crossing)
            shift=(desired[step]-rs[step])*fraction
            final=rs[step]+shift
            logits=embedding@(gamma*final)/torch.sqrt(final.square().mean()+1e-6)
            logits[repeated]=torch.where(logits[repeated]<0,logits[repeated]*1.1,logits[repeated]/1.1)
            assert int(logits.argmax())==want
            competitors=logits.clone();competitors[want]=-torch.inf
            keys.append(hs[step]);shifts.append(shift.double())
            row.update(action='interpolate_native_decision',crossing_fraction=crossing,
                applied_fraction=fraction,shift_norm=float(shift.norm()),
                desired_logit_margin=float(logits[want]-competitors.max()))
        decisions.append(row)
    print(key,'modified decisions',sum(x['id']==key and not x['native_correct'] for x in decisions),'of',len(tokens),flush=True)
del embedding,raw0,raw1
G=torch.cat(guards);G=G/G.norm(dim=1,keepdim=True).clamp_min(1e-12)
K=torch.stack(keys,dim=1);shift=torch.stack(shifts,dim=1)
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'id':'003','tensor':meta['tensor'],
    'targets':targets,'target_count':len(targets),'target_decision_count':len(keys),'guard_count':len(G),
    'spec_sha256':hashlib.sha256((root/'patches/003_spec.json').read_bytes()).hexdigest(),
    'decisions':decisions,'training':False,'gradients':False,'optimizer':False,'reserved_final_used':False}
dest=root/'results/003_algebra.json'
def save():dest.write_text(json.dumps(report,indent=2),encoding='utf8')
save();print('SVD',G.shape,'positive equations',K.shape,flush=True)
_,s,vh=torch.linalg.svd(G,full_matrices=False);rank=int((s>s[0]*1e-6).sum());basis=vh[:rank]
Kp=K-basis.T@(basis@K);gram=K.T@Kp
report.update(guard_rank=rank,projected_gram_condition=float(torch.linalg.cond(gram)))
save();assert report['projected_gram_condition']<1e10
dual=Kp@torch.linalg.pinv(gram,rtol=1e-10)
replacement=base+(shift@dual.T).float();actual=(replacement-base).double()
changed=int(torch.count_nonzero(replacement!=base))
report.update(status='patch_written_not_yet_behaviorally_evaluated',changed_values=changed,
    unique_parameter_count=meta['unique_model_parameters'],unique_weight_modification_ratio=changed/meta['unique_model_parameters'],
    serialized_parameter_count=751632384,serialized_weight_modification_ratio=changed/751632384,
    delta_frobenius=float(actual.norm()),base_frobenius=float(base.norm()),relative_frobenius=float(actual.norm()/base.norm()),
    target_constraint_max_abs_error=float((actual@K-shift).abs().max()),normalized_guard_effect_max=float((G@actual.T).abs().max()),
    original_tensor_sha256=hashlib.sha256(base.numpy().tobytes()).hexdigest(),
    replacement_tensor_sha256=hashlib.sha256(replacement.numpy().tobytes()).hexdigest())
np.savez_compressed(out,replacement=replacement.numpy());report['patch_sha256']=hashlib.sha256(out.read_bytes()).hexdigest();save()
print(json.dumps({k:v for k,v in report.items() if k not in ['decisions','targets']}),flush=True)
