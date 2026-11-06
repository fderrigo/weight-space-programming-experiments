"""Review the origin correction separately from whole narrative fidelity."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
assert not (RDIR / 'late_narrative_origins_review.json').exists()
def read(name):
    return json.loads((RDIR / name).read_text('utf8'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
manifest = read('late_narrative_origins_screen_manifest.json')
assert manifest['completed'] and manifest['responses_completed'] == 18 and manifest['weights_restored_in_memory']
assert sha(RDIR / 'late_narrative_origins_screen.json') == manifest['responses_sha256']
prior = read('late_native_recovery_review.json')
assert sha(RDIR / 'late_native_recovery_screen.json') == prior['source_sha256']
old = {r['id']: r for r in read('late_native_recovery_screen.json')}
original = {r['id']: r for r in read('attention_gain_screen.json') if r['stage'] == 'original'}
old_notes = {r['id']: r for r in prior['annotations']}
notes = {
    'second_material_0': 'Still fails to answer wood. Invents Giorgio, a father called geppetto Ciliegia and an unsupported creation story.',
    'second_nose_0': 'Still gives hand/body movements and possible destruction of the nose, not the canonical elongation from lying.',
    'second_ending_0': 'Correct real-boy phrase followed by a fabricated book and false transformations into multiple animals and repeated pieces of wood. New desired origin words appear in the wrong narrative role; whole answer remains false.',
    'second_puppeteer_0': 'Correct Mangiafoco followed by unrelated invented subscription, publication and commercial claims. Changed date does not repair the truncated answer.',
    'narration_0': 'Now correctly calls Pinocchio a puppet, but invents discovering a new life form and a repeated wooden-towel plot. The first concept is repaired, not the complete opening sentence or story.',
    'narration_1': 'The first numbered event now correctly states origin from a piece of wood. The following mother, golden hair/red nose, king of Leghorn and journey plot remain invented; whole narrative unfaithful.',
}
rows = read('late_narrative_origins_screen.json')
annotations, changed = [], []
for row in rows:
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
narratives = {r['id']: r for r in rows if r['group'] == 'narration'}
assert 'storia di un burattino' in narratives['narration_0']['response']
assert narratives['narration_1']['response'].startswith('1. Il personaggio di Pinocchio è nato da un pezzo di legno.')
report = {**{k: v for k, v in prior.items() if k != 'annotations'},
    'candidate': 'late_narrative_origins', 'source_sha256': manifest['responses_sha256'],
    'basis_sha256': manifest['basis_sha256'], 'review_code_sha256': sha(Path(__file__)),
    'previous_candidate_review_sha256': sha(RDIR / 'late_native_recovery_review.json'),
    'same_previous_generated_sequences': 12, 'changed_previous_response_ids': changed,
    'review_method': 'Read every changed answer; reuse prior annotations only with identical prompt, tokens, stop and decoded text.',
    'partial_narrative_repairs': [
        {'id': 'narration_0', 'assertion': 'The protagonist is a puppet', 'correct': True, 'whole_opening_sentence_correct': False},
        {'id': 'narration_1', 'assertion': 'Pinocchio originates from a piece of wood', 'correct': True, 'whole_first_numbered_event_correct': True}],
    'whole_faithful_narratives_gained': 0, 'third_and_fourth_free_generation_repeated_for_this_candidate': False,
    'observed_transfer_completed': False,
    'previous_multi_set_reference': 'late_native_recovery', 'full_objective_achieved': False,
    'interpretation': 'Three direct weight equations repair two origin concepts in ordinary generation and preserve the six known short-answer repairs plus all four exact controls in this screen. Narrative fidelity remains0/2. Newly repeated wood concepts in an already-wrong ending answer show that concept correction does not ensure correct event-role binding. No promotion or fresh validation.',
    'annotations': annotations}
(RDIR / 'late_narrative_origins_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print('Known facts6/12, controls4/4 exact, narrative origins2 partial repairs, faithful narratives0/2.')
