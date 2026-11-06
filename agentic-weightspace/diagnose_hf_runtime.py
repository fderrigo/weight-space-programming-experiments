"""Compare native F32 forward generation, preserving unedited official weights."""
import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

root = Path(__file__).resolve().parent
torch.set_num_threads(6)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)
renders = json.loads((root/'results/development_rendered.json').read_text('utf8'))
baseline = {r['id']:r for r in json.loads((root/'results/original_development.json').read_text('utf8'))}
tokenizer = AutoTokenizer.from_pretrained(root/'models/original', local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(root/'models/original', torch_dtype=torch.float32,
    local_files_only=True, attn_implementation='eager').eval()
model.requires_grad_(False)
rows = []
for key in ['qual_authorized_00','qual_target_00','qual_target_04','qual_non_target_00']:
    ids = tokenizer(renders[key]['tools'], return_tensors='pt', add_special_tokens=False).input_ids
    full = ids[0].tolist()
    cache, generated, top = None, [], None
    start = time.perf_counter()
    for step in range(96):
        result = model(input_ids=ids, past_key_values=cache, use_cache=True)
        cache = result.past_key_values
        logits = result.logits[0,-1].clone()
        repeat = list(set(full[-64:]))
        logits[repeat] = torch.where(logits[repeat]<0, logits[repeat]*1.1, logits[repeat]/1.1)
        if top is None:
            values, tokens = logits.topk(10)
            top = [{'id':int(t), 'text':tokenizer.decode([t]), 'logit':float(v)} for t,v in zip(tokens,values)]
        token = int(logits.argmax())
        if token in [151645,151643]:
            break
        generated.append(token)
        full.append(token)
        ids = torch.tensor([[token]])
    text = tokenizer.decode(generated, skip_special_tokens=False)
    rows.append({'id':key,'hf_generated':text,'hf_token_ids':generated,'first_step_top10':top,
        'ollama_text':baseline[key]['response_text'], 'ollama_tool_calls':baseline[key]['raw_tool_calls'],
        'input_token_count':len(tokenizer.encode(renders[key]['tools'],add_special_tokens=False)),
        'ollama_input_count':baseline[key]['raw_ollama_response']['prompt_eval_count'],
        'seconds':time.perf_counter()-start,'repeat_policy':'last64 tokens, penalty1.1, greedy',
        'weights_modified':False})
    (root/'results/hf_runtime_diagnostic.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
    print(key, repr(text), flush=True)
