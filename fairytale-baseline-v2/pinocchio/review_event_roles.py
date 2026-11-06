"""Review whole ordinary answers after the event-role equation construction."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
OUT = RDIR / 'late_event_roles_review.json'
assert not OUT.exists()


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = read('late_event_roles_screen_manifest.json')
assert manifest['completed'] and manifest['weights_restored_in_memory'] and manifest['responses_completed'] == 18
assert sha(RDIR / 'late_event_roles_screen.json') == manifest['responses_sha256']
prior = read('late_narrative_origins_review.json')
assert sha(RDIR / 'late_narrative_origins_screen.json') == prior['source_sha256']
old = {r['id']: r for r in read('late_narrative_origins_screen.json')}
original = {r['id']: r for r in read('attention_gain_screen.json') if r['stage'] == 'original'}
old_notes = {r['id']: r for r in prior['annotations']}
notes = {
    'second_material_0': 'Still no canonical wood answer. Giorgio and Count Cuginello replace earlier invented names; the whole origin remains fabricated.',
    'second_nose_0': 'Still talks about hand movements, touching the nose/head and destruction of the body, rather than the canonical nose elongation caused by lying.',
    'second_ending_0': 'Real-boy opening remains followed by a fabricated book, animal transformations and repeated pieces of wood. The altered truncated ending is still false.',
    'second_puppeteer_0': 'Mangiafoco is still followed by invented Abecede/subscription/commercial content, now with price repetitions. No whole-answer repair.',
    'narration_0': 'Preserves the puppet identity, but replaces the wooden-towel repetition with invented Pavia, mother Tito, father Carlo and Lord of Victory events. The story remains false and truncated.',
    'narration_1': 'Preserves the first event of origin from wood, then invents a mother, castle, father characterization and wood used for iron weapons. Repeated wood is attached to incorrect events; whole narrative remains false.',
}
rows = read('late_event_roles_screen.json')
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
report = {
    'reviewer': 'Codex; explicit whole-answer semantic review, not independent review',
    'candidate': 'late_event_roles', 'source_sha256': manifest['responses_sha256'],
    'basis_sha256': manifest['basis_sha256'], 'review_code_sha256': sha(Path(__file__)),
    'previous_candidate': 'late_narrative_origins',
    'previous_candidate_review_sha256': sha(RDIR / 'late_narrative_origins_review.json'),
    'original_source_sha256': sha(RDIR / 'attention_gain_screen.json'),
    'original_target_whole_correct': 0, 'initially_wrong_targets': 12, 'target_total': 12,
    'edited_target_whole_correct_strict': 6, 'strict_observed_development_repair_percent': 50.0,
    'new_definite_repairs_over_previous_candidate': [], 'previous_definite_repairs_lost': [],
    'original_control_main_correct': 4, 'preserved_control_main_correct': 4,
    'original_control_whole_correct_conservative': 3, 'preserved_control_whole_correct_conservative': 3,
    'external_same_ids_and_stop': 4, 'external_total': 4,
    'original_faithful_narratives': 0, 'edited_faithful_narratives': 0, 'narrative_total': 2,
    'changed_parameters': manifest['changed_parameters'], 'changed_tensors': manifest['changed_tensors'],
    'same_previous_generated_sequences': 12, 'changed_previous_response_ids': changed,
    'review_method': 'Read all six changed texts; inherit prior annotations only for identical prompt, tokens, stop and decoded text.',
    'partial_narrative_repairs_retained': prior['partial_narrative_repairs'],
    'whole_faithful_narratives_gained': 0,
    'interpretation': 'Eleven equations repair the measured event-concept paths but add no whole ordinary answer or faithful narrative in the18-response screen. Six factual repairs and all four original control sequences remain. The32/32 diagnostic pieces must not be counted as whole responses or fresh validation.',
    'calibration_only': True, 'fresh_validation': False, 'observed_transfer_completed': False,
    'third_and_fourth_free_generation_repeated_for_this_candidate': False,
    'previous_multi_set_reference': 'late_native_recovery',
    'runtime_hooks': False, 'training': False, 'gradients': False, 'optimizer': False,
    'candidate_accepted': False, 'candidate_promoted': False, 'checkpoint_exported': False,
    'current_exported_checkpoint_attempt': 6, 'final_set_read': False,
    'final_protocol_exposure_note': 'results/final_protocol_exposure_20261105.json',
    'full_objective_achieved': False, 'annotations': annotations}
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print('6/12 whole factual repairs;4/4 exact original controls;0/2 faithful stories; no new whole repair.')
