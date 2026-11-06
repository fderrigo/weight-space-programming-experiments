"""Manual whole-response review of the eighteen calibration generations."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'factorized_carrier'
assert not (RDIR / f'{STEM}_review.json').exists()


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def sha(name):
    return hashlib.sha256((RDIR / name).read_bytes()).hexdigest()


manifest = read(f'{STEM}_screen_manifest.json')
rows = read(f'{STEM}_screen.json')
assert manifest['completed'] and manifest['responses_completed'] == len(rows) == 18
assert sha(f'{STEM}_screen.json') == manifest['responses_sha256']
assert sha('attention_gain_screen.json') == manifest['original_responses_source_sha256']
original = {r['id']: r for r in read('attention_gain_screen.json') if r['stage'] == 'original'}
previous = {r['id']: r for r in read('minimal_fact_completion_screen.json')}
good = {'second_creator_0', 'second_material_0', 'second_fairy_0', 'second_deceivers_0',
        'second_donkey_0', 'second_friend_0', 'second_wood_donor_0', 'second_rescuer_0', 'second_fish_father_0'}
notes = {
    'second_creator_0': 'Correctly distinguishes Geppetto, the fictional maker, from the author.',
    'second_material_0': 'Correct wood/firewood answer. The vague stylistic aside about Geppetto is treated consistently with the prior minimal-concept review; also report the conservative count excluding it.',
    'second_nose_0': 'Says the hand lengthens, then invents light and skin details; incorrect despite the initial predicate.',
    'second_fairy_0': 'Correctly gives blue hair.',
    'second_deceivers_0': 'Correctly identifies the Cat and Fox and their theft.',
    'second_donkey_0': 'Correctly gives a donkey.',
    'second_ending_0': 'Starts with real boy but shifts to the child of Pinocchio and an invented book title; whole response rejected.',
    'second_puppeteer_0': 'Names Mangiafoco but assigns the schoolbook sale to the puppeteer and invents the subsequent meeting details.',
    'second_friend_0': 'Correctly names Lucignolo.',
    'second_wood_donor_0': 'Correctly names Maestro Ciliegia.',
    'second_rescuer_0': 'Correctly names Alidoro.',
    'second_fish_father_0': 'Correctly names Geppetto.',
    'second_control_00': 'Unchanged correct France answer.',
    'second_control_04': 'Unchanged cat-to-gatto translation, including the preexisting questionable claim that gatto is the only Italian term.',
    'second_control_10': 'Unchanged correct four answer.',
    'second_control_13': 'Unchanged correct Rome answer.',
    'narration_0': 'Unchanged false two-year-old/Signor Pinocchio story and repeated crowd events.',
    'narration_1': 'Unchanged invented animal origin, Rosina, Carlo and marriage events.',
}
assert set(notes) == {r['id'] for r in rows}
annotations = []
for row in rows:
    annotations.append({'id': row['id'], 'group': row['group'],
                        'target_whole_correct': row['id'] in good,
                        'same_original_ids': row['generated_ids'] == original[row['id']]['generated_ids'],
                        'same_original_stop': row['stop_token'] == original[row['id']]['stop_token'],
                        'same_previous_minimal_candidate_ids': row['generated_ids'] == previous[row['id']]['generated_ids'],
                        'reason': notes[row['id']]})
controls = [r for r in annotations if r['group'] == 'control']
narratives = [r for r in annotations if r['group'] == 'narration']
assert all(r['same_original_ids'] and r['same_original_stop'] for r in controls + narratives)
report = {
    'reviewer': 'Codex; explicit semantic annotations, not independent review',
    'source_sha256': manifest['responses_sha256'],
    'original_source_sha256': manifest['original_responses_source_sha256'],
    'original_target_whole_correct': 0, 'initially_wrong_targets': 12,
    'edited_target_whole_correct': len(good),
    'conservative_target_whole_correct_excluding_awkward_material_aside': len(good) - 1,
    'target_total': 12, 'calibration_repair_percent': 100 * len(good) / 12,
    'conservative_calibration_repair_percent': 100 * (len(good) - 1) / 12,
    'external_same_ids_and_stop': len(controls), 'external_total': 4,
    'original_control_main_correct': 4, 'preserved_control_main_correct': 4,
    'original_control_whole_correct_conservative': 3, 'preserved_control_whole_correct_conservative': 3,
    'narrative_same_ids_and_stop': len(narratives), 'original_faithful_narratives': 0,
    'edited_faithful_narratives': 0, 'narrative_total': 2,
    'same_previous_minimal_candidate_ids': sum(r['same_previous_minimal_candidate_ids'] for r in annotations),
    'changed_parameters': manifest['changed_parameters'], 'changed_tensors': manifest['changed_tensors'],
    'calibration_only': True, 'fresh_validation': False, 'final_set_read': False,
    'runtime_hooks': False, 'training': False, 'gradients': False, 'optimizer': False,
    'candidate_promoted': False, 'checkpoint_exported': False, 'current_exported_checkpoint_attempt': 6,
    'interpretation': 'The fitted concept histories produce correct words and preserve recorded native continuations. This is not independent improvement or a faithful full story; different formulations must still be checked.',
    'annotations': annotations,
}
(RDIR / f'{STEM}_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k != 'annotations'}, flush=True)
