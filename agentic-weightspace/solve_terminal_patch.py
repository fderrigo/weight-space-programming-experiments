"""One explicit closed-form weight intervention. No training or parameter search."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

p = argparse.ArgumentParser()
p.add_argument('--id', default='001')
p.add_argument('--no-tools-guards',choices=['all','endpoints'],default='all')
a = p.parse_args()
root = Path(__file__).resolve().parent
out = root/f'patches/{a.id}_patch.npz'
assert not out.exists()
torch.set_num_threads(6)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)
meta = json.loads((root/'analysis/original_decisions.json').read_text('utf8'))
data = np.load(root/'analysis/original_decisions.npz')
keys, residuals, guards, targets, native_authorized = [], [], [], [], []
for row in meta['rows']:
    key = row['id']
    h = torch.from_numpy(data[key+'__h']).double()
    r = torch.from_numpy(data[key+'__r']).double()
    tokens = data[key+'__tokens']
    if row['category']=='authorized' and row['original_tool_call'] and tokens[0]==meta['tool_start_token']:
        native_authorized.append(r[0]/r[0].norm())
    if row['category']=='target' and row['native_choice_mismatch_count']==0 and tokens[-1]==meta['EOS_token']:
        keys.append(h[-1]); residuals.append(r[-1]); targets.append(key)
        guards.append(h[:-1])
    else:
        guards.append(h)
    no_tools = torch.from_numpy(data[key+'__no_tools_h']).double()
    guards.append(no_tools if a.no_tools_guards=='all' else no_tools[[0,-1]])
K, R = torch.stack(keys,dim=1), torch.stack(residuals,dim=1)
G = torch.cat(guards,dim=0)
G = G / G.norm(dim=1,keepdim=True).clamp_min(1e-12)
report = {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'id':a.id,
    'tensor':meta['tensor'],'targets':targets,'target_count':len(targets), 'guard_count':len(G),
    'no_tools_guard_policy':a.no_tools_guards,'svd_relative_threshold':1e-6,
    'state_data_sha256':meta['sha256'],'training':False,'gradients':False,'optimizer':False,
    'reserved_final_used':False,'numerical_dtype':'float64 algebra; float32 replacement'}
report_path = root/f'results/{a.id}_algebra.json'
def save(): report_path.write_text(json.dumps(report,indent=2),encoding='utf8')
save()
print('SVD guards:',G.shape,'target keys:',K.shape,flush=True)
_, singular, vh = torch.linalg.svd(G,full_matrices=False)
rank = int((singular > singular[0]*1e-6).sum())
basis = vh[:rank]
Kp = K-basis.T@(basis@K)
report.update(guard_rank=rank,guard_space_dimension=G.shape[1],
    target_nullspace_norm_fraction=(Kp.norm(dim=0)/K.norm(dim=0)).tolist(),
    largest_guard_singular=float(singular[0]),smallest_retained_guard_singular=float(singular[rank-1]))
save()
print('Guard rank',rank,'remaining',G.shape[1]-rank,flush=True)
if rank == G.shape[1]:
    report['status']='no_nullspace_no_patch_applied';save();raise SystemExit(2)
gram = K.T@Kp
report['projected_gram_condition']=float(torch.linalg.cond(gram))
assert report['projected_gram_condition'] < 1e10, 'Ill-conditioned explicit constraints; no automatic relaxation.'
dual = Kp @ torch.linalg.pinv(gram,rtol=1e-10)
direction = torch.stack(native_authorized).mean(dim=0)
direction /= direction.norm()
desired = direction[:,None] * R.norm(dim=0)[None,:]
shift = desired-R
delta = shift@dual.T
with safe_open(root/'models/original/model.safetensors',framework='pt') as f:
    base = f.get_tensor(meta['tensor']).float()
    embedding = f.get_tensor('lm_head.weight').float()
    gamma = f.get_tensor('model.norm.weight').float()
    new = base + delta.float()
    actual = (new-base).double()
    normalized = desired.float()/torch.sqrt(desired.float().square().mean(dim=0,keepdim=True)+1e-6)
    desired_logits = embedding @ (gamma[:,None]*normalized)
    top = desired_logits.argmax(dim=0)
    report['desired_residual_top_tokens']=top.tolist()
    assert bool((top==meta['tool_start_token']).all()), 'Native authorized centroid does not select tool token.'
    unique_count = meta['unique_model_parameters']
    changed = int(torch.count_nonzero(new!=base))
    report.update(status='patch_written_not_yet_behaviorally_evaluated',changed_values=changed,
        unique_parameter_count=unique_count,unique_weight_modification_ratio=changed/unique_count,
        serialized_parameter_count=751632384,serialized_weight_modification_ratio=changed/751632384,
        delta_frobenius=float(actual.norm()),base_frobenius=float(base.norm()),
        relative_frobenius=float(actual.norm()/base.norm()),
        target_constraint_max_abs_error=float((actual@K-shift).abs().max()),
        normalized_guard_effect_max=float((G@actual.T).abs().max()),
        original_tensor_sha256=hashlib.sha256(base.numpy().tobytes()).hexdigest(),
        replacement_tensor_sha256=hashlib.sha256(new.numpy().tobytes()).hexdigest())
np.savez_compressed(out,replacement=new.numpy(),desired_residual=desired.float().numpy(),
    target_keys=K.float().numpy(),dual=dual.float().numpy())
report['patch_sha256']=hashlib.sha256(out.read_bytes()).hexdigest()
save()
print(json.dumps({k:v for k,v in report.items() if k not in ['targets','target_nullspace_norm_fraction']}),flush=True)
