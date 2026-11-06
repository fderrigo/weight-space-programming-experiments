"""Whole-answer and narrative review after native-path preservation."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'late_native_recovery_screen'
assert not (RDIR / 'late_native_recovery_review.json').exists()
def read(name):
    return json.loads((RDIR / name).read_text('utf8'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
manifest = read(f'{STEM}_manifest.json')
assert manifest['completed'] and manifest['responses_completed'] == 18
assert manifest['all_declared_preservation_checks_passed']
assert sha(RDIR / f'{STEM}.json') == manifest['responses_sha256']
prior = read('late_tail_relations_review.json')
assert sha(RDIR / 'late_tail_relations_screen.json') == prior['source_sha256']
old = {r['id']: r for r in read('late_tail_relations_screen.json')}
original = {r['id']: r for r in read('attention_gain_screen.json') if r['stage'] == 'original'}
old_notes = {r['id']: r for r in prior['annotations']}
notes = {
    'second_material_0': 'Still does not identify wood; invents Giorgio, count Ciro and a false Geppetto naming story.',
    'second_nose_0': 'Substitutes head and lips for the nose and adds unrelated shadow claims.',
    'second_deceivers_0': 'Still gives only Gatto di Pincio instead of the Cat and the Fox.',
    'second_ending_0': 'Correct real-boy phrase followed by a fabricated title Il Collido; excluded consistently with the preceding Il Collodi claim.',
    'second_puppeteer_0': 'Correct Mangiafoco followed by invented subscription prices, publication date1904 and commercial failure; unfinished and incorrect.',
    'narration_0': 'Invented child, glue-son friend, strange animal and motorway story; truncated and not a faithful Pinocchio narrative.',
    'narration_1': 'Invented stone spring origin, false name etymology and stone-cage episodes; not faithful despite reaching eight numbered points.',
}
annotations, changed = [], []
for row in read(f'{STEM}.json'):
    identity = row['id']
    assert row['prompt'] == old[identity]['prompt'] == original[identity]['prompt']
    same = row['generated_ids'] == old[identity]['generated_ids'] and row['stop_token'] == old[identity]['stop_token']
    note = dict(old_notes[identity])
    note['same_previous_ids_and_stop'] = same
    if not same:
        changed.append(identity)
        assert identity in notes
        note['reason'] = notes[identity]
        assert not note['target_whole_correct'] and not note['control_whole_correct']
    else:
        assert row['response'] == old[identity]['response']
    note['same_original_ids_and_stop'] = row['generated_ids'] == original[identity]['generated_ids'] and row['stop_token'] == original[identity]['stop_token']
    annotations.append(note)
assert set(changed) == set(notes)
assert sum(r['target_whole_correct'] for r in annotations) == 6
assert all(r['same_original_ids_and_stop'] for r in annotations if r['group'] == 'control')
report = {**{k: v for k, v in prior.items() if k != 'annotations'},
    'candidate': 'late_native_recovery', 'source_sha256': manifest['responses_sha256'], 'basis_sha256': manifest['basis_sha256'],
    'previous_candidate_review_sha256': sha(RDIR / 'late_tail_relations_review.json'),
    'review_code_sha256': sha(Path(__file__)), 'same_previous_generated_sequences': 18 - len(changed),
    'changed_previous_response_ids': changed, 'edited_faithful_narratives': 0,
    'changed_parameters': manifest['changed_parameters'], 'changed_tensors': manifest['changed_tensors'],
    'review_method': 'Explicitly read seven changed answers. Other annotations require identical prompt, generated IDs, stop and decoded text.',
    'interpretation': 'Retains6/12 whole target repairs and allfour exact original controls, three of which are wholly correct. Both narratives change but remain unfaithful. No new target repair in this screen and no full-goal success.',
    'final_protocol_exposure_note': 'results/final_protocol_exposure_20261105.json',
    'annotations': annotations}
(RDIR / 'late_native_recovery_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({k: report[k] for k in ['edited_target_whole_correct_strict', 'preserved_control_main_correct', 'edited_faithful_narratives', 'same_previous_generated_sequences']})
