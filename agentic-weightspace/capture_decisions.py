"""Forward-only analysis of original last-MLP decision states on development."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

root = Path(__file__).resolve().parent
out = root/'analysis/original_decisions.npz'
assert not out.exists()
out.parent.mkdir(exist_ok=True)
torch.set_num_threads(6)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)
cases = json.loads((root/'benchmark/development.json').read_text('utf8'))
renders = json.loads((root/'results/development_rendered.json').read_text('utf8'))
baseline = {r['id']:r for r in json.loads((root/'results/original_development.json').read_text('utf8'))}
tok = AutoTokenizer.from_pretrained(root/'models/original', local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(root/'models/original', dtype=torch.float32,
    local_files_only=True, attn_implementation='eager').eval()
model.requires_grad_(False)
capture = {}
def input_hook(module, args):
    capture['h'] = args[0][0].detach().clone()
def residual_hook(module, args):
    capture['r'] = args[0][0].detach().clone()
handles = [model.model.layers[-1].mlp.down_proj.register_forward_pre_hook(input_hook),
           model.model.norm.register_forward_pre_hook(residual_hook)]
arrays, report = {}, []
for case in cases:
    key = case['id']
    row = baseline[key]
    prefix = tok.encode(renders[key]['tools'], add_special_tokens=False)
    text = row['response_text']
    if row['observed_valid_tool_request']:
        assert len(row['raw_tool_calls']) == 1 and not text
        text = '<tool_call>\n{"name": "hello_world", "arguments": {}}\n</tool_call>'
    continuation = tok.encode(text, add_special_tokens=False)
    ids = prefix + continuation
    result = model(input_ids=torch.tensor([ids]), use_cache=False)
    start = len(prefix) - 1
    h = capture['h'][start:].numpy().copy()
    r = capture['r'][start:].numpy().copy()
    logits = result.logits[0,start:]
    predicted = []
    for i in range(len(continuation)+1):
        score = logits[i].clone()
        repeated = list(set(ids[max(0,len(prefix)+i-64):len(prefix)+i]))
        score[repeated] = torch.where(score[repeated]<0, score[repeated]*1.1, score[repeated]/1.1)
        predicted.append(int(score.argmax()))
    desired = continuation + [tok.eos_token_id]
    mismatch = [{'position':i,'observed_id':want,'predicted_id':got,
        'observed':tok.decode([want]),'predicted':tok.decode([got])} for i,(want,got) in enumerate(zip(desired,predicted)) if want != got]
    arrays[key+'__h'], arrays[key+'__r'] = h, r
    arrays[key+'__tokens'] = np.asarray(desired,dtype=np.int64)
    no_prefix = tok.encode(renders[key]['no_tools'],add_special_tokens=False)
    # Counterfactual teacher-forced ordinary text is a protection constraint, not a measured no-tool output.
    no_text = continuation if not row['observed_valid_tool_request'] else []
    model(input_ids=torch.tensor([no_prefix+no_text]),use_cache=False,logits_to_keep=1)
    arrays[key+'__no_tools_h'] = capture['h'][len(no_prefix)-1:].numpy().copy()
    item = {'id':key,'category':case['category'],'decision_count':len(desired),
        'native_choice_mismatch_count':len(mismatch),'mismatches':mismatch,
        'original_tool_call':row['observed_valid_tool_request'],
        'no_tools_keys':'teacher-forced original tools-present text, not measured native no-tool behavior'}
    report.append(item)
    print(key,'states',len(desired),'native mismatches',len(mismatch),flush=True)
np.savez_compressed(out,**arrays)
for handle in handles: handle.remove()
summary = {'rows':report,'tensor':'model.layers.27.mlp.down_proj.weight',
    'unique_model_parameters':sum(p.numel() for p in model.parameters()),
    'tool_start_token':tok.convert_tokens_to_ids('<tool_call>'),
    'EOS_token':tok.eos_token_id, 'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
    'weights_edited':False,'gradients_enabled':torch.is_grad_enabled(),
    'capture':'Final block down-projection input and residual before final RMSNorm. Final MLP outputs are downstream of all KV caches: changing them does not alter future KV states.'}
(root/'analysis/original_decisions.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
print('Captured',len(report),'cases;',summary['unique_model_parameters'],'unique model parameters.',flush=True)
