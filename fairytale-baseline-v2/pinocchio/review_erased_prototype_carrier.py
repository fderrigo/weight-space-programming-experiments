"""Review the paired native-erasure experiment, retaining ambiguity explicitly."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'erased_prototype_carrier'
assert not (RDIR / f'{STEM}_review.json').exists()


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = read(f'{STEM}_screen_manifest.json')
rows = read(f'{STEM}_screen.json')
assert manifest['completed'] and manifest['responses_completed'] == len(rows) == 18
assert sha(RDIR / f'{STEM}_screen.json') == manifest['responses_sha256']
assert sha(RDIR / 'attention_gain_screen.json') == manifest['original_responses_source_sha256']
assert sha(ROOT / f'patches/{STEM}_basis.safetensors') == manifest['basis_sha256']
original = {r['id']: r for r in read('attention_gain_screen.json') if r['stage'] == 'original'}
previous_review = read('progress_prototype_carrier_review.json')
previous_good = {r['id'] for r in previous_review['annotations'] if r['target_whole_correct']}
good = {'second_fairy_0', 'second_donkey_0', 'second_wood_donor_0'}
ambiguous = {'second_nose_0'}
control_good = {'second_control_00', 'second_control_10', 'second_control_13'}
notes = {
    'second_creator_0': 'Still confuses the fictional maker with Carlo Collodi.',
    'second_material_0': 'Does not give wood; invents animal and family details.',
    'second_nose_0': 'Correctly says the nose lengthens because of lies, followed by an unclear claim about a child using abilities to explain events. Count separately as ambiguous; exclude from the strict whole-answer total.',
    'second_fairy_0': 'Correct Azzurri followed by a period and native stop; previous repetition is gone.',
    'second_deceivers_0': 'Invents Gatto di Sfinge and omits the Fox.',
    'second_donkey_0': 'Correctly gives Un asino and stops.',
    'second_ending_0': 'Mentions a real child but invents the title Il Piccolo Crampo.',
    'second_puppeteer_0': 'Gives Giuseppe Cipolla instead of Mangiafoco and corrupts the schoolbook name.',
    'second_friend_0': 'Gives Amoroso instead of Lucignolo.',
    'second_wood_donor_0': 'Correctly gives Maestro Ciliegia and stops.',
    'second_rescuer_0': 'Calls the dog a real boy named Pinocchio rather than Alidoro.',
    'second_fish_father_0': 'Names Geppetto but then calls him an animal; whole response rejected.',
    'second_control_00': 'Correct Paris-France answer; the prior false Italian-Republic addition is gone.',
    'second_control_04': 'Repeatedly gives gato rather than Italian gatto.',
    'second_control_10': 'Correct four answer.',
    'second_control_13': 'Correct Rome answer.',
    'narration_0': 'Does not tell the requested story and gives irrelevant writing advice.',
    'narration_1': 'Invents a feathered girl, Ester, cooking and murder episodes, followed by repetition.',
}
assert set(notes) == {r['id'] for r in rows}
annotations = [{'id': r['id'], 'group': r['group'],
                'target_whole_correct': r['id'] in good,
                'target_whole_ambiguous': r['id'] in ambiguous,
                'previous_target_whole_correct': r['id'] in previous_good,
                'control_whole_correct': r['id'] in control_good,
                'same_original_ids_and_stop': r['generated_ids'] == original[r['id']]['generated_ids'] and r['stop_token'] == original[r['id']]['stop_token'],
                'reason': notes[r['id']]} for r in rows]
report = {
    'reviewer': 'Codex; explicit semantic review, not independent review',
    'source_sha256': manifest['responses_sha256'], 'basis_sha256': manifest['basis_sha256'],
    'original_source_sha256': manifest['original_responses_source_sha256'],
    'previous_candidate_review_sha256': sha(RDIR / 'progress_prototype_carrier_review.json'),
    'original_target_whole_correct': 0, 'initially_wrong_targets': 12, 'target_total': 12,
    'edited_target_whole_correct_strict': len(good), 'ambiguous_target_whole': len(ambiguous),
    'edited_target_whole_including_ambiguous_aside': len(good | ambiguous),
    'strict_observed_development_repair_percent': 100 * len(good) / 12,
    'inclusive_observed_development_repair_percent': 100 * len(good | ambiguous) / 12,
    'previous_candidate_whole_correct_same_twelve': len(previous_good),
    'new_definite_repairs_over_previous_candidate': sorted(good - previous_good),
    'previous_definite_repairs_lost': sorted(previous_good - good),
    'original_control_main_correct': 4, 'preserved_control_main_correct': 3,
    'original_control_whole_correct_conservative': 3, 'preserved_control_whole_correct_conservative': 3,
    'external_same_ids_and_stop': sum(r['same_original_ids_and_stop'] for r in annotations if r['group'] == 'control'),
    'external_total': 4, 'original_faithful_narratives': 0, 'edited_faithful_narratives': 0, 'narrative_total': 2,
    'changed_parameters': manifest['changed_parameters'], 'changed_tensors': manifest['changed_tensors'],
    'calibration_only': True, 'fresh_validation': False, 'final_set_read': False,
    'runtime_hooks': False, 'training': False, 'gradients': False, 'optimizer': False,
    'candidate_promoted': False, 'checkpoint_exported': False, 'current_exported_checkpoint_attempt': 6,
    'interpretation': 'Erasure causally improves native continuation and two definite whole answers on the same development sample. It does not meet the correction goal, preserve every main control fact, or produce faithful narratives.',
    'annotations': annotations,
}
(RDIR / f'{STEM}_review.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k != 'annotations'}, flush=True)
