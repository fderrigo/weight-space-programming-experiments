"""Review the changed known-screen answers and verify unchanged annotations."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'late_tail_relations'
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
old_source = RDIR / 'late_target_contrasts_screen.json'
old = {r['id']: r for r in read(old_source)}
previous_path = RDIR / 'late_target_contrasts_review.json'
previous = read(previous_path)
assert previous['source_sha256'] == sha(old_source)
annotations = {r['id']: dict(r) for r in previous['annotations']}
notes = {
    'second_puppeteer_0': 'Correct initial Mangiafoco, followed by fabricated Abecedeiro prices, subscriptions and premiums. The changed continuation remains false as a whole.',
    'narration_0': 'Invents a magician friend, beasts moving toward a village and a stone episode. Truncated and not faithful to Collodi.',
    'narration_1': 'Invents a mother and her evil lover, then repeats contradictory claims about a fathers love. Not a faithful narrative.',
}
changed = []
for r in responses:
    p = old[r['id']]
    assert r['prompt'] == p['prompt'] and r['max_new_tokens'] == p['max_new_tokens']
    same = r['generated_ids'] == p['generated_ids'] and r['stop_token'] == p['stop_token']
    if not same:
        changed.append(r['id'])
        assert r['id'] in notes and not annotations[r['id']]['target_whole_correct']
        annotations[r['id']]['reason'] = notes[r['id']]
    else:
        assert r['response'] == p['response']
assert set(changed) == set(notes) and len(responses) == 18
report = {**previous,
    'source_sha256': manifest['responses_sha256'], 'basis_sha256': manifest['basis_sha256'],
    'previous_candidate_review_sha256': sha(previous_path),
    'changed_parameters': manifest['changed_parameters'], 'changed_tensors': manifest['changed_tensors'],
    'review_method': 'Explicitly review three changed answers. Reuse remaining annotations only after exact generated-token, stop, prompt and decoded-text identity checks.',
    'same_previous_generated_sequences': 15, 'changed_previous_response_ids': changed,
    'new_definite_repairs_over_previous_candidate': [], 'previous_definite_repairs_lost': [],
    'candidate_for_observed_transfer_check': False, 'observed_transfer_completed': True,
    'interpretation': 'Retains6/12 strict whole factual repairs and all4 main controls with identical original control tokens and stops. One already wrong factual continuation and both already unfaithful narratives change, without becoming correct. Narratives remain0/2. No final test or promotion.',
    'review_code_sha256': sha(Path(__file__)),
    'annotations': [annotations[r['id']] for r in responses]}
OUTPUT.write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k != 'annotations'}, flush=True)
