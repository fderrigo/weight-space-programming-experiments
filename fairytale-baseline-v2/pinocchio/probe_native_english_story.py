"""Original-model language/knowledge diagnostic; never an editing success test."""
import datetime
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'native_english_story'
OUT = ROOT / f'results/{STEM}.json'
SPEC = ROOT / f'results/{STEM}_spec.json'
assert not OUT.exists() and not SPEC.exists()


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


facts = [
    ('creator', 'Who makes Pinocchio in the original story by Carlo Collodi?'),
    ('material', 'What material is Pinocchio made of at the beginning of Collodi\'s story?'),
    ('nose', 'What happens to Pinocchio\'s nose when he tells lies?'),
    ('fairy', 'What color is the Fairy\'s hair in Collodi\'s Pinocchio?'),
    ('deceivers', 'Which two animals deceive Pinocchio about planting his gold coins?'),
    ('donkey', 'What animal does Pinocchio turn into after visiting the Land of Toys?'),
    ('ending', 'What does Pinocchio become at the end of Collodi\'s story?'),
    ('puppeteer', 'What is the name of the puppet theater owner in Collodi\'s Pinocchio?'),
    ('friend', 'What is the name of Pinocchio\'s friend who goes with him to the Land of Toys?'),
    ('wood_donor', 'Who gives Geppetto the piece of wood from which he makes Pinocchio?'),
    ('rescuer', 'Which dog rescues Pinocchio from the Green Fisherman?'),
    ('fish_father', 'Whom does Pinocchio find inside the giant Dogfish?'),
]
controls = [
    ('capital', 'What country is Paris in?'),
    ('arithmetic', 'What is two plus two?'),
    ('shark', 'Is a shark a fish or a mammal?'),
    ('habitat', 'In what kind of environment do fish live?'),
]
narratives = [
    ('pinocchio_prose', 'Tell the story of The Adventures of Pinocchio by Carlo Collodi in about 180 words, from the puppet\'s origin to the ending.'),
    ('pinocchio_events', 'Reconstruct the main events of Pinocchio in Carlo Collodi\'s original novel in eight points.'),
    ('three_pigs', 'Tell the traditional story of the Three Little Pigs in about 180 words, from the beginning to the ending.'),
    ('tortoise_hare', 'Reconstruct the main events of the traditional fable of the Tortoise and the Hare in eight points.'),
]
cases = []
for group, rows, limit in [('fact', facts, 128), ('control', controls, 128), ('narrative', narratives, 256)]:
    for label, prompt in rows:
        cases.append({'id': 'en_' + label, 'group': group, 'fact': label,
                      'prompt': prompt + (' Answer in one or two sentences.' if group != 'narrative' else ''),
                      'max_new_tokens': limit})
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'decision_by': 'Codex',
    'hypothesis_before_outputs': 'The original may access story relations more reliably in English than in Italian. Compare ordinary English factual answers and Pinocchio narratives with previously recorded Italian failures, and use two other stories to separate Pinocchio weakness from general narrative weakness.',
    'interpretation': 'A correct English answer diagnoses existing ability; it is not a weight correction or an acceptable final-language workaround. An incorrect answer does not prove that the knowledge is absent from all internal states.',
    'cases': cases, 'original_checkpoint_sha256': sha(MODEL / 'model.safetensors'),
    'code_sha256': sha(Path(__file__)),
    'sampling': {'runtime': 'Transformers', 'dtype': 'float32', 'do_sample': False,
                 'repetition_penalty': 1.1, 'seed': 42, 'threads': 6, 'batch_size': 4,
                 'template': 'unchanged original Qwen chat template'},
    'diagnostic_only': True, 'weights_changed': False, 'counts_as_correction': False,
    'runtime_hooks': False, 'corrective_context': False, 'provided_answer_prefix': False,
    'final_set_read': False, 'training': False, 'gradients': False, 'optimizer': False,
    'completed': False,
}
assert spec['original_checkpoint_sha256'] == 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
SPEC.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf8')
torch.set_grad_enabled(False)
torch.set_num_threads(6)
torch.manual_seed(42)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
t.padding_side = 'left'
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
responses = []
for group in ['fact', 'control', 'narrative']:
    selected = [row for row in cases if row['group'] == group]
    for start in range(0, len(selected), 4):
        batch = selected[start:start + 4]
        prompts = [t.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False,
                                        add_generation_prompt=True) for r in batch]
        enc = t(prompts, return_tensors='pt', padding=True)
        generated = m.generate(**enc, do_sample=False, repetition_penalty=1.1,
                               max_new_tokens=batch[0]['max_new_tokens'], pad_token_id=t.pad_token_id)
        for row, tokens in zip(batch, generated[:, enc['input_ids'].shape[1]:].tolist()):
            stop = next((j for j, token in enumerate(tokens) if token in [151643, 151645]), None)
            ids = tokens if stop is None else tokens[:stop]
            response = {**row, 'response': t.decode(ids, skip_special_tokens=True),
                        'generated_ids': ids, 'stop_token': None if stop is None else tokens[stop],
                        'done_reason': 'length' if stop is None else 'stop', 'stage': 'original',
                        'diagnostic_only': True, 'counts_as_correction': False}
            responses.append(response)
            print(row['id'], repr(response['response']), flush=True)
        OUT.write_text(json.dumps(responses, ensure_ascii=False, indent=2), encoding='utf8')
assert len(responses) == len(cases) == 20
spec.update({'completed': True, 'responses_sha256': sha(OUT),
             'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
SPEC.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf8')
print('Original English diagnostic complete; no weight changes or claimed corrections.', flush=True)
