"""Review the complete frozen third comparison, including regressions."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEM = 'minimal_fact_completion_third'


def read(relative):
    return json.loads((ROOT / relative).read_text('utf8'))


def sha(relative):
    with (ROOT / relative).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


manifest = read(f'results/{STEM}_evaluation_manifest.json')
assert manifest['completed'] and manifest['original_completed'] == manifest['edited_completed'] == 44
original = read(f'results/{STEM}_original_hf.json')
edited = read(f'results/{STEM}_edited_hf.json')
for stage in ['original', 'edited']:
    assert sha(f'results/{STEM}_{stage}_hf.json') == manifest[f'{stage}_responses_sha256']
assert sha('patches/minimal_fact_completion_basis.safetensors') == manifest['candidate_basis_sha256']
assert sha('patches/minimal_fact_completion_spec.json') == manifest['candidate_spec_sha256']
assert sha('evaluate_frozen_candidate_hf.py') == manifest['evaluator_sha256']
assert sha('third_development_questions.json') == manifest['question_sha256']
baseline_targets = read(f'results/{STEM}_original_target_review.json')
baseline_controls = read(f'results/{STEM}_original_control_review.json')
assert baseline_controls['original_source_sha256'] == manifest['original_responses_sha256']
original_targets = [row for row in original if row['group'] == 'target']
canonical = json.dumps(original_targets, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf8')
assert hashlib.sha256(canonical).hexdigest() == baseline_targets['original_target_subset_sha256']
baseline = {row['id']: row for row in baseline_targets['annotations'] + baseline_controls['annotations']}
original_by_id = {row['id']: row for row in original}
correct_targets = {
    'third_puppeteer_0': 'Correctly names Mangiafoco and stops.',
    'third_wood_donor_0': 'Correctly names Maestro Ciliegia and stops.',
    'third_wood_donor_1': 'Correctly names Maestro Ciliegia and stops.',
    'third_rescuer_1': 'Correctly names Alidoro and stops.',
}
incorrect_targets = {
    'third_creator_0': 'Mixed factual-name fragments and repetition; no coherent maker answer.',
    'third_creator_1': 'Corrupted workshop and maker names; no coherent Geppetto answer.',
    'third_material_0': 'Corrupts the previously correct wood answer into mixed fragments and repetition.',
    'third_material_1': 'Answers Maestro Ciliegia to a question asking for a material.',
    'third_nose_0': 'Corrupted facial-change phrase without a coherent nose-lengthening fact.',
    'third_nose_1': 'Corrupted names and repeated fragments; no coherent nose-lengthening fact.',
    'third_fairy_0': 'Starts with Azzurri but then assigns the blue hair to Pinocchio, with corrupted wording.',
    'third_fairy_1': 'Still says black rather than blue/turquoise.',
    'third_deceivers_0': 'Starts with the Cat and Fox but the rest is corrupted repetition.',
    'third_deceivers_1': 'Corrupted puppet and child relationships; does not identify the pair.',
    'third_donkey_0': 'Corrupted repeated transformation wording; no donkey answer.',
    'third_donkey_1': 'Corrupted repeated fragments; no coherent donkey transformation.',
    'third_ending_0': 'Adds an invented Alighieri identity instead of an unambiguous real-boy ending.',
    'third_ending_1': 'Almost entirely corrupted repetition.',
    'third_puppeteer_1': 'Still names Carlo, with an additional corrupted beard word.',
    'third_friend_0': 'Mixed and repeated factual-name fragments; no coherent Lucignolo answer.',
    'third_friend_1': 'Corrupted companion name instead of Lucignolo.',
    'third_rescuer_0': 'Starts with Alidoro but adds a corrupted event and fails to state the rescue coherently.',
    'third_fish_father_0': 'Says his only friend and corrupted companion, not his father Geppetto.',
    'third_fish_father_1': 'Corrupted name fragments instead of Geppetto.',
}
control_correct = {
    11: 'Still correctly identifies Puss in Boots as a cat.',
    15: 'Still identifies London as English rather than Italian. The typo inglege does not reverse the selected alternative.',
}
control_incorrect = {
    0: 'Still calls a ripe strawberry gray.',
    1: 'Corrupted family-response fragments; no paternal grandmother.',
    2: 'Corrupted cousin/uncle answer and repeated factual-token fragments.',
    3: 'Unchanged false wild-boar answer.',
    4: 'Destroys the previously correct shark classification with repeated fragments.',
    5: 'Corrupted occupation and catch descriptions.',
    6: 'Calls a forest geology/a geographer and corrupts the beach distinction.',
    7: 'Corrupted repeated fragments instead of aquatic habitat.',
    8: 'Corrupted story wording and an unnamed child instead of Cinderella.',
    9: 'Corrupted pigs/house wording instead of the wolf.',
    10: 'Corrupted names and counts instead of seven dwarfs.',
    12: 'Retains Collodi but also false German nationality and 1832 date, then adds corrupted fragments.',
    13: 'Unchanged false answer six.',
    14: 'Corrupted capital/country wording; no Italy.',
    16: 'Corrupted word and material fragments instead of legno.',
    17: 'Destroys the previously correct blue translation with repeated fragments.',
    18: 'Corrupted paint-mixing wording; no green.',
    19: 'Destroys the previously interpretable wood answer with repeated corrupted words.',
}
corrupted_controls = {1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 17, 18, 19}
target_rows, control_rows = [], []
for row in edited:
    old = baseline[row['id']]
    comparison = {'id': row['id'], 'original_whole_correct': old['whole_correct'],
                  'same_text': row['response'] == original_by_id[row['id']]['response'],
                  'same_generated_ids': row['generated_ids'] == original_by_id[row['id']]['generated_ids'],
                  'same_stop_token': row['stop_token'] == original_by_id[row['id']]['stop_token']}
    if row['group'] == 'target':
        comparison.update({'edited_whole_correct': row['id'] in correct_targets,
                           'reason': (correct_targets | incorrect_targets)[row['id']]})
        target_rows.append(comparison)
    else:
        number = int(row['id'].rsplit('_', 1)[1])
        comparison.update({'edited_whole_correct': number in control_correct,
                           'edited_main_fact_correct': number in {11, 12, 15},
                           'new_factual_token_fragment_corruption': number in corrupted_controls,
                           'reason': (control_correct | control_incorrect)[number]})
        control_rows.append(comparison)
assert len(target_rows) == 24 and len(control_rows) == 20
repairs = [row['id'] for row in target_rows if not row['original_whole_correct'] and row['edited_whole_correct']]
regressions = [row['id'] for row in target_rows if row['original_whole_correct'] and not row['edited_whole_correct']]
wrong_before = sum(not row['original_whole_correct'] for row in target_rows)
outside_good_before = sum(row['original_whole_correct'] for row in control_rows)
outside_retained = sum(row['original_whole_correct'] and row['edited_whole_correct'] for row in control_rows)
report = {
    'reviewer': 'Codex; explicit full-response annotations, not independent review',
    'original_source_sha256': manifest['original_responses_sha256'],
    'edited_source_sha256': manifest['edited_responses_sha256'],
    'basis_sha256': manifest['candidate_basis_sha256'],
    'question_sha256': manifest['question_sha256'],
    'candidate_fixed_before_questions_read': True,
    'original_target_whole_correct': 24 - wrong_before,
    'edited_target_whole_correct': sum(row['edited_whole_correct'] for row in target_rows),
    'target_total': 24, 'initially_wrong_targets': wrong_before,
    'repaired_targets': len(repairs), 'repair_percent': 100 * len(repairs) / wrong_before,
    'regressed_targets': len(regressions),
    'net_target_error_reduction_percent': 100 * (len(repairs) - len(regressions)) / wrong_before,
    'repair_ids': repairs, 'regression_ids': regressions,
    'original_controls_whole_correct': outside_good_before,
    'edited_controls_whole_correct': sum(row['edited_whole_correct'] for row in control_rows),
    'originally_correct_controls_retained': outside_retained,
    'control_knowledge_preservation_percent': 100 * outside_retained / outside_good_before,
    'control_responses_exact_ids': sum(row['same_generated_ids'] and row['same_stop_token'] for row in control_rows),
    'controls_with_new_fragment_corruption': sum(row['new_factual_token_fragment_corruption'] for row in control_rows),
    'control_total': 20,
    'narratives_in_this_fresh_comparison': 0,
    'earlier_calibration_faithful_narratives': 0, 'earlier_calibration_narrative_total': 2,
    'changed_parameters': manifest['changed_parameters'],
    'training': False, 'gradients': False, 'optimizer': False, 'runtime_hooks': False,
    'corrective_context': False, 'output_replacement': False,
    'fifty_percent_repair_reached': len(repairs) >= wrong_before / 2,
    'candidate_promoted': False, 'checkpoint_exported': False,
    'current_exported_checkpoint_attempt': 6, 'final_set_read': False,
    'interpretation': 'The frozen candidate fails generalization: 4 of 23 initially wrong targets are repaired, the only initially correct target regresses, and only 2 of 5 previously correct external answers survive. The high calibration score does not establish selective knowledge correction. This candidate is rejected; the final set is unused.',
    'targets': target_rows, 'controls': control_rows,
}
(ROOT / f'results/{STEM}_review.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({key: value for key, value in report.items() if key not in ['targets', 'controls']})
