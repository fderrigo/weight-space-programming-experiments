"""Can the original continue a correct concept from just its first token?

This supplied-prefix diagnostic determines whether a one-position weight write
could suffice. No prefix-assisted response counts as an autonomous correction.
"""
import datetime
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'first_concept_token'
output = ROOT / f'results/{STEM}.json'
assert not output.exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
units = json.loads((ROOT / 'results/minimal_fact_units_spec.json').read_text('utf8'))['short_units']
concepts = {fact: t.encode(unit, add_special_tokens=False) for fact, unit in units.items()}
rows = [row for row in json.loads((ROOT / 'second_development_questions.json').read_text('utf8')) if row['group'] == 'target']
assert len(rows) == 24
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'reason': 'Earlier dense edits enforced every subtoken of the concept. Test whether only its first token can recruit an otherwise correct native continuation, reducing the required intervention to the answer-start state.',
    'prefixes': {fact: {'token': ids[0], 'piece': t.decode(ids[:1]), 'complete_concept_supplied': len(ids) == 1}
                 for fact, ids in concepts.items()},
    'max_full_answer_tokens': 128, 'repetition_penalty': 1.1, 'do_sample': False,
    'batch_size': 4, 'torch_threads': 4,
    'weights_changed': False, 'runtime_hooks': False, 'training': False,
    'gradients': False, 'optimizer': False, 'supplied_prefix_diagnostic_only': True,
    'counts_as_correction': False, 'EOS_programmed': False,
    'third_questions_used': False, 'final_set_read': False,
}
(ROOT / f'results/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
print(spec['prefixes'], flush=True)
m = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
responses = []
for begin in range(0, len(rows), 4):
    batch = rows[begin:begin + 4]
    bases = []
    for row in batch:
        prompt = t.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                                      tokenize=False, add_generation_prompt=True)
        bases.append(t.encode(prompt, add_special_tokens=False) + concepts[row['fact']][:1])
    width = max(map(len, bases))
    ids = torch.tensor([[t.pad_token_id] * (width - len(base)) + base for base in bases])
    mask = torch.tensor([[0] * (width - len(base)) + [1] * len(base) for base in bases])
    generated = m.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                           repetition_penalty=1.1, max_new_tokens=127,
                           pad_token_id=t.pad_token_id)
    for row, base, tokens in zip(batch, bases, generated[:, width:].tolist()):
        stop = next((i for i, token in enumerate(tokens) if token in [151643, 151645]), None)
        answer_ids = tokens if stop is None else tokens[:stop]
        unit = concepts[row['fact']]
        complete_ids = unit[:1] + answer_ids
        response = {**row, 'provided_prefix_ids': unit[:1], 'provided_prefix': t.decode(unit[:1]),
                    'expected_concept_ids': unit, 'complete_concept_supplied': len(unit) == 1,
                    'base_ids': base, 'generated_ids': answer_ids,
                    'response_with_supplied_prefix': t.decode(complete_ids, skip_special_tokens=True),
                    'concept_completed_exactly': complete_ids[:len(unit)] == unit,
                    'stop_token': None if stop is None else tokens[stop],
                    'done_reason': 'length' if stop is None else 'stop',
                    'weights_changed': False, 'runtime_hooks': False,
                    'supplied_prefix_diagnostic_only': True, 'counts_as_correction': False}
        responses.append(response)
        print(row['id'], repr(response['response_with_supplied_prefix']), flush=True)
    output.write_text(json.dumps(responses, ensure_ascii=False, indent=2), encoding='utf8')
print('Exact concept completions', sum(row['concept_completed_exactly'] for row in responses),
      '/', len(responses), '; diagnostic only, no correction claimed.', flush=True)
