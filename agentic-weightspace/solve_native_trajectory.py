"""Apply exactly the LLM-declared002 native trajectory transfer by direct algebra."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

root=Path(__file__).resolve().parent
out=root/'patches/002_patch.npz'
assert not out.exists()
torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
meta=json.loads((root/'analysis/original_decisions.json').read_text('utf8'))
trajectory_meta=json.loads((root/'analysis/002_target_trajectories.json').read_text('utf8'))
old=np.load(root/'analysis/original_decisions.npz');new_keys=np.load(root/'analysis/002_target_trajectories.npz')
targets=trajectory_meta['targets'];source=torch.from_numpy(new_keys['source_r']).double()
source=source/source.norm(dim=1,keepdim=True)
guards=[]
for row in meta['rows']:
    key=row['id'];h=torch.from_numpy(old[key+'__h']).double()
    guards.append(h[:-1] if key in targets else h)
    h_no=torch.from_numpy(old[key+'__no_tools_h']).double()
    guards.append(h_no[[0,-1]])
G=torch.cat(guards);G=G/G.norm(dim=1,keepdim=True).clamp_min(1e-12)
keys=[];original_r=[];desired_r=[]
for key in targets:
    h=torch.from_numpy(new_keys[key+'__h']).double();r=torch.from_numpy(new_keys[key+'__r']).double()
    keys.append(h);original_r.append(r);desired_r.append(source*r.norm(dim=1,keepdim=True))
K=torch.cat(keys).T;R=torch.cat(original_r).T;desired=torch.cat(desired_r).T
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'id':'002','tensor':meta['tensor'],
    'targets':targets,'target_count':len(targets),'target_decision_count':K.shape[1],
    'guard_count':len(G),'spec_sha256':hashlib.sha256((root/'patches/002_spec.json').read_bytes()).hexdigest(),
    'trajectory_sha256':trajectory_meta['sha256'],'training':False,'gradients':False,'optimizer':False,
    'reserved_final_used':False,'svd_relative_threshold':1e-6}
dest=root/'results/002_algebra.json'
def save():dest.write_text(json.dumps(report,indent=2),encoding='utf8')
save();print('Guard SVD',G.shape,'target equations',K.shape,flush=True)
_,singular,vh=torch.linalg.svd(G,full_matrices=False)
rank=int((singular>singular[0]*1e-6).sum());basis=vh[:rank]
Kp=K-basis.T@(basis@K)
gram=K.T@Kp
report.update(guard_rank=rank,projected_gram_condition=float(torch.linalg.cond(gram)),
    target_nullspace_norm_fraction_min=float((Kp.norm(dim=0)/K.norm(dim=0)).min()))
save();print('Gram condition',report['projected_gram_condition'],flush=True)
assert report['projected_gram_condition']<1e10,'No automatic constraint relaxation'
dual=Kp@torch.linalg.pinv(gram,rtol=1e-10)
delta=(desired-R)@dual.T
with safe_open(root/'models/original/model.safetensors',framework='pt') as f:
    base=f.get_tensor(meta['tensor']).float();replacement=base+delta.float()
actual=(replacement-base).double()
changed=int(torch.count_nonzero(replacement!=base))
report.update(status='patch_written_not_yet_behaviorally_evaluated',changed_values=changed,
    unique_parameter_count=meta['unique_model_parameters'],unique_weight_modification_ratio=changed/meta['unique_model_parameters'],
    serialized_parameter_count=751632384,serialized_weight_modification_ratio=changed/751632384,
    delta_frobenius=float(actual.norm()),base_frobenius=float(base.norm()),
    relative_frobenius=float(actual.norm()/base.norm()),
    target_constraint_max_abs_error=float((actual@K-(desired-R)).abs().max()),
    normalized_guard_effect_max=float((G@actual.T).abs().max()),
    original_tensor_sha256=hashlib.sha256(base.numpy().tobytes()).hexdigest(),
    replacement_tensor_sha256=hashlib.sha256(replacement.numpy().tobytes()).hexdigest())
np.savez_compressed(out,replacement=replacement.numpy())
report['patch_sha256']=hashlib.sha256(out.read_bytes()).hexdigest();save()
print(json.dumps(report),flush=True)
