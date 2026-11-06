"""Semantic review of the actual selector/reference candidate generation."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'late_margin_preservation'
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
previous_review = read('late_selector_fact_review.json')
previous_good = {r['id'] for r in previous_review['annotations'] if r['target_whole_correct']}
good = {'second_fairy_0', 'second_donkey_0', 'second_friend_0', 'second_wood_donor_0', 'second_rescuer_0', 'second_fish_father_0'}
control_whole_good = {'second_control_00', 'second_control_10', 'second_control_13'}
control_main_good = control_whole_good | {'second_control_04'}
notes = {
    'second_creator_0': 'Confuses the in-story maker with the real author Carlo Collodi; Geppetto is not identified.',
    'second_material_0': 'Does not give wood and invents a king Ciliegia and unrelated Giuseppe Verdi claims.',
    'second_nose_0': 'Says a hand lengthens and invents shadow and skin observations. Does not correctly state the nose consequence.',
    'second_fairy_0': 'Correct Azzurri with a period and native stop.',
    'second_deceivers_0': 'Names Il Gatto di Pincio and omits the Fox; the pair is not correctly supplied.',
    'second_donkey_0': 'Correct Un asino with period and native stop.',
    'second_ending_0': 'Gives the real-boy transformation but refers to a fabricated book title Il Collodi; excluded from strict whole-answer correctness.',
    'second_puppeteer_0': 'Correct initial Mangiafoco, followed by invented subscription-price events, a false1902 publication date and no-commercial-success claim. Truncated and incorrect as a whole.',
    'second_friend_0': 'Correct Lucignolo with a period and native stop.',
    'second_wood_donor_0': 'Correct Maestro Ciliegia with period and native stop.',
    'second_rescuer_0': 'Correct Alidoro with period and native stop.',
    'second_fish_father_0': 'Correct Geppetto with a period and native stop.',
    'second_control_00': 'Restores the correct Paris-France answer.',
    'second_control_04': 'Restores the correct cat-gatto translation; retains the original grammatical errors including Gli gatti and a redundant misleading exclusivity claim. Main fact correct, whole response conservatively excluded.',
    'second_control_10': 'Restores the correct two plus two equals four.',
    'second_control_13': 'Correct Rome answer.',
    'narration_0': 'Invents a father visiting Carlo, children moving along house walls and repetitive collera episodes. Truncated and not a faithful narrative.',
    'narration_1': 'Invents an anima fatta origin and a mother with a headache, then repeats father-is-a-friend statements. Not faithful to Collodi.',
}
assert set(notes) == {r['id'] for r in rows}
annotations = [{'id': r['id'], 'group': r['group'],
                'target_whole_correct': r['id'] in good,
                'control_main_correct': r['id'] in control_main_good,
                'control_whole_correct': r['id'] in control_whole_good,
                'same_original_ids_and_stop': r['generated_ids'] == original[r['id']]['generated_ids'] and r['stop_token'] == original[r['id']]['stop_token'],
                'reason': notes[r['id']]} for r in rows]
report = {
    'reviewer': 'Codex; explicit semantic review, not independent review',
    'source_sha256': manifest['responses_sha256'], 'basis_sha256': manifest['basis_sha256'],
    'original_source_sha256': manifest['original_responses_source_sha256'],
    'previous_candidate_review_sha256': sha(RDIR / 'late_selector_fact_review.json'),
    'original_target_whole_correct': 0, 'initially_wrong_targets': 12, 'target_total': 12,
    'edited_target_whole_correct_strict': len(good),
    'strict_observed_development_repair_percent': 100 * len(good) / 12,
    'previous_candidate_whole_correct_same_twelve': len(previous_good),
    'new_definite_repairs_over_previous_candidate': sorted(good - previous_good),
    'previous_definite_repairs_lost': sorted(previous_good - good),
    'original_control_main_correct': 4, 'preserved_control_main_correct': len(control_main_good),
    'original_control_whole_correct_conservative': 3, 'preserved_control_whole_correct_conservative': len(control_whole_good),
    'external_same_ids_and_stop': sum(r['same_original_ids_and_stop'] for r in annotations if r['group'] == 'control'),
    'external_total': 4, 'original_faithful_narratives': 0, 'edited_faithful_narratives': 0, 'narrative_total': 2,
    'changed_parameters': manifest['changed_parameters'], 'changed_tensors': manifest['changed_tensors'],
    'calibration_only': True, 'fresh_validation': False, 'final_set_read': False,
    'runtime_hooks': False, 'training': False, 'gradients': False, 'optimizer': False,
    'candidate_accepted': False, 'candidate_promoted': False, 'checkpoint_exported': False,
    'current_exported_checkpoint_attempt': 6, 'final_50_percent_goal_met': False,
    'candidate_for_observed_transfer_check': True,
    'interpretation': 'Retains six of twelve strict factual repairs and restores all four original main control facts, including France and arithmetic. Narratives remain0/2. Proceed to observed transfer evaluation with the frozen candidate; this is not fresh validation or completion of the50percent goal.',
    'annotations': annotations,
}
(RDIR / f'{STEM}_review.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k != 'annotations'}, flush=True)
