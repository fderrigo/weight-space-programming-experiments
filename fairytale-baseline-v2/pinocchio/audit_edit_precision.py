"""Compare storage rounding of one declared matrix edit, without changing base values."""
import datetime, json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parent
torch.set_grad_enabled(False);torch.set_num_threads(6)
basis=load_file(ROOT/'patches/multipoint_protected_basis.safetensors')
with np.load(ROOT/'results/orthogonal_story_roles_2048_controls_native_states.npz') as z:
 old=torch.from_numpy(z['C'][-525:].copy())
with np.load(ROOT/'results/role_native_continuations.npz') as z:
 native=torch.cat([torch.from_numpy(z[k].copy()) for k in z.files])
with np.load(ROOT/'results/newdevelopment_control_states.npz') as z:
 external=torch.from_numpy(z['C'].copy())
with np.load(ROOT/'results/multipoint_protected_new_states.npz') as z:
 newnative=torch.from_numpy(z['C'].copy())
C=torch.cat([old,native,external,newnative]).double();K=basis['keys'].double();V=basis['values'].double()
results={}
for precision in ['bf16','float32']:
 after=basis['before']+basis['delta']
 if precision=='bf16':after=after.bfloat16().float()
 actual=(after-basis['before']).double();effects=(C@actual.T).norm(dim=-1)
 results[precision]={'mean_preservation_residual_change':float(effects.mean()),'max_preservation_residual_change':float(effects.max()),'relative_positive_value_error':float((K@actual.T-V).norm()/V.norm()),'delta_norm':float(actual.norm())}
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'states':len(C),'results':results,'limitation':'These are internal equations on calibration states, not generalization or factual success. Both baseline and candidate inference remain F32 in Ollama.'}
(ROOT/'results/multipoint_precision_audit.json').write_text(json.dumps(report,indent=2))
spec=json.loads((ROOT/'patches/multipoint_protected_spec.json').read_text(encoding='utf8'))
spec.update({'utc':report['utc'],'precision':'Store the edited down-projection tensor in F32. Original other tensors retain their BF16 storage. Both original and modified GGUF inference are F32, and every conversion value must be verified exact.','precision_reason':'The hard preservation equations hold before BF16 rounding. Preserve the explicitly computed weight values instead of rounding them to BF16; original numerical weights and original inference are unchanged.','precision_audit':'results/multipoint_precision_audit.json','basis':'patches/multipoint_protected_basis.safetensors'})
(ROOT/'patches/multipoint_protected_f32_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
