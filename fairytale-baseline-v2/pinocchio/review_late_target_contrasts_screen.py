"""Carry a semantic review only after exact generated-token identity checks."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'late_target_contrasts'
OUTPUT = RDIR / f'{STEM}_review.json'
assert not OUTPUT.exists()


def read(path):
    return json.loads(path.read_text('utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


source = RDIR / f'{STEM}_screen.json'
manifest = read(RDIR / f'{STEM}_screen_manifest.json')
assert manifest['completed'] and sha(source) == manifest['responses_sha256']
assert sha(ROOT / f'patches/{STEM}_basis.safetensors') == manifest['basis_sha256']
responses = read(source)
old_source = RDIR / 'late_margin_preservation_screen.json'
old = {r['id']: r for r in read(old_source)}
previous_path = RDIR / 'late_margin_preservation_review.json'
previous = read(previous_path)
assert previous['source_sha256'] == sha(old_source)
assert len(responses) == len(old) == 18
for r in responses:
    p = old[r['id']]
    for field in ['prompt', 'group', 'generated_ids', 'stop_token', 'done_reason', 'max_new_tokens', 'response']:
        assert r[field] == p[field], (r['id'], field)
report = {**previous,
    'source_sha256': manifest['responses_sha256'], 'basis_sha256': manifest['basis_sha256'],
    'previous_candidate_review_sha256': sha(previous_path),
    'changed_parameters': manifest['changed_parameters'], 'changed_tensors': manifest['changed_tensors'],
    'review_method': 'Every prompt, generated token, stop and decoded response is identical to the previously semantically reviewed margin candidate. Reuse those exact annotations only after assertions pass.',
    'same_previous_generated_sequences': 18,
    'new_definite_repairs_over_previous_candidate': [], 'previous_definite_repairs_lost': [],
    'interpretation': 'All18 observed-screen responses are unchanged:6/12 whole target repairs, all4 main controls retained,0/2 faithful narratives. The three local contrast changes must be evaluated on the separate already observed transfer formulations; this is not fresh validation.',
    'review_code_sha256': sha(Path(__file__))}
OUTPUT.write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k != 'annotations'}, flush=True)
