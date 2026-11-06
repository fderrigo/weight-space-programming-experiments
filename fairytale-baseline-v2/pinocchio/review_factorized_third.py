"""Explicit review of the factorized candidate on the observed third set."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'factorized_carrier_third_observed'
assert not (RDIR / f'{STEM}_review.json').exists()


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def sha(name):
    return hashlib.sha256((RDIR / name).read_bytes()).hexdigest()


manifest = read(f'{STEM}_manifest.json')
edited = read(f'{STEM}.json')
assert manifest['completed'] and manifest['responses_completed'] == len(edited) == 44
assert sha(f'{STEM}.json') == manifest['responses_sha256']
assert sha('minimal_fact_completion_third_original_hf.json') == manifest['original_responses_source_sha256']
original = {r['id']: r for r in read('minimal_fact_completion_third_original_hf.json')}
old_review = read('minimal_fact_completion_third_review.json')
assert old_review['original_source_sha256'] == manifest['original_responses_source_sha256']
old = {r['id']: r for r in old_review['targets'] + old_review['controls']}
correct_targets = {'third_puppeteer_0', 'third_wood_donor_0', 'third_rescuer_0'}
correct_controls = set()
main_controls = {'third_control_11', 'third_control_12'}
notes = {
    'third_creator_0': 'Corrupted Ge/Ma/Luc fragments and repeated text instead of Geppetto.',
    'third_creator_1': 'Invented geunettiano/Leonardo and repeated fragments instead of the maker.',
    'third_material_0': 'Regresses from the originally correct material to Legge and corrupted lizard wording.',
    'third_material_1': 'Mentions Maestro Ciliegia then corrupted legal/name fragments; no coherent wood answer.',
    'third_nose_0': 'Corrupted change wording followed by repeated spacing; no coherent nose consequence.',
    'third_nose_1': 'States nose elongation but continues with corrupted invented wording; whole response rejected.',
    'third_fairy_0': 'Starts with blue but adds corrupted identities and repetitions.',
    'third_fairy_1': 'Corrupted description and green rather than blue/turquoise.',
    'third_deceivers_0': 'Names the Cat and Fox but adds corrupted identities and repeated fragments.',
    'third_deceivers_1': 'Corrupted and invented names instead of a coherent Cat and Fox answer.',
    'third_donkey_0': 'Starts with a donkey then appends extensive corrupted fragments.',
    'third_donkey_1': 'Starts with a donkey but the explanation is corrupted.',
    'third_ending_0': 'Starts with real boy but repeats the unit and appends broken Ge fragments.',
    'third_ending_1': 'Corrupted new-nature wording ending in wood instead of the real-boy transformation.',
    'third_puppeteer_0': 'Correctly names Mangiafoco and stops.',
    'third_puppeteer_1': 'Invents Lucifer/LucLucGe instead of Mangiafoco.',
    'third_friend_0': 'Corrupted and repeated name fragments instead of Lucignolo.',
    'third_friend_1': 'Leaves a NAME placeholder and broken child wording instead of Lucignolo.',
    'third_wood_donor_0': 'Correctly names Maestro Ciliegia and stops.',
    'third_wood_donor_1': 'Starts with Maestro Gepp and extensive unrelated fragments instead of Maestro Ciliegia.',
    'third_rescuer_0': 'Correctly names Alidoro and stops.',
    'third_rescuer_1': 'Broken dog/child fragments and repetitions instead of Alidoro.',
    'third_fish_father_0': 'Corrupted repeated fragments instead of meeting Geppetto.',
    'third_fish_father_1': 'Corrupted Ge/Ma/Luc fragments instead of a complete Geppetto answer.',
    'third_control_00': 'Corrupted strawberry/color response with repeated factual-name fragments.',
    'third_control_01': 'Corrupted family wording and wrong relative direction; no paternal grandmother.',
    'third_control_02': 'Broken uncle/cousin response with repeated question fragments.',
    'third_control_03': 'Retains the previously false wild-boar answer.',
    'third_control_04': 'Destroys the previously correct shark classification with fragments and repetitions.',
    'third_control_05': 'Corrupted fishing/occupation wording instead of a coherent catch description.',
    'third_control_06': 'Corrupted beach/forest fragments and repetitions.',
    'third_control_07': 'Corrupted environment/name fragments instead of aquatic habitat.',
    'third_control_08': 'Corrupted shoe/name fragments instead of Cinderella.',
    'third_control_09': 'Corrupted story/name fragments instead of the wolf.',
    'third_control_10': 'Corrupted number/name fragments instead of seven dwarfs.',
    'third_control_11': 'Still contains un gatto but the surrounding answer is extensively corrupted.',
    'third_control_12': 'Retains Collodi, the preexisting false German nationality/date, and adds new corruption.',
    'third_control_13': 'Corrupted arithmetic/name fragments instead of five.',
    'third_control_14': 'Corrupted Rome/country fragments instead of Italy.',
    'third_control_15': 'Destroys the previously correct English-city distinction with repeated fragments.',
    'third_control_16': 'Corrupted material/name fragments instead of the Italian translation legno.',
    'third_control_17': 'Destroys the previously correct blue translation with Azzulla and broken name fragments.',
    'third_control_18': 'Corrupted color wording and repeated spacing instead of green.',
    'third_control_19': 'Destroys the previously interpretable wood answer with broken woodworker/name fragments.',
}
assert set(notes) == {r['id'] for r in edited}
annotations = []
for row in edited:
    before = original[row['id']]
    group = row['group']
    annotations.append({'id': row['id'], 'group': group,
                        'original_whole_correct': old[row['id']]['original_whole_correct'],
                        'edited_whole_correct': row['id'] in (correct_targets if group == 'target' else correct_controls),
                        'edited_control_main_correct': row['id'] in main_controls,
                        'same_original_ids': row['generated_ids'] == before['generated_ids'],
                        'same_original_stop': row['stop_token'] == before['stop_token'],
                        'new_fragment_corruption': group == 'control' and row['id'] != 'third_control_03',
                        'reason': notes[row['id']]})
targets = [r for r in annotations if r['group'] == 'target']
controls = [r for r in annotations if r['group'] == 'control']
initially_wrong = sum(not r['original_whole_correct'] for r in targets)
repairs = [r['id'] for r in targets if not r['original_whole_correct'] and r['edited_whole_correct']]
regressions = [r['id'] for r in targets if r['original_whole_correct'] and not r['edited_whole_correct']]
original_controls_correct = sum(r['original_whole_correct'] for r in controls)
retained_controls = sum(r['original_whole_correct'] and r['edited_whole_correct'] for r in controls)
report = {
    'reviewer': 'Codex; explicit semantic annotations, not independent review',
    'source_sha256': manifest['responses_sha256'], 'original_source_sha256': manifest['original_responses_source_sha256'],
    'basis_sha256': manifest['basis_sha256'], 'question_sha256': manifest['question_sha256'],
    'original_target_whole_correct': 24 - initially_wrong, 'edited_target_whole_correct': len(correct_targets),
    'target_total': 24, 'initially_wrong_targets': initially_wrong, 'repaired_targets': len(repairs),
    'repair_percent': 100 * len(repairs) / initially_wrong, 'regressed_targets': len(regressions),
    'net_target_error_reduction_percent': 100 * (len(repairs) - len(regressions)) / initially_wrong,
    'repair_ids': repairs, 'regression_ids': regressions,
    'original_controls_whole_correct': original_controls_correct, 'edited_controls_whole_correct': len(correct_controls),
    'originally_correct_controls_retained': retained_controls,
    'control_knowledge_preservation_percent': 100 * retained_controls / original_controls_correct,
    'control_responses_exact_ids_and_stop': sum(r['same_original_ids'] and r['same_original_stop'] for r in controls),
    'controls_with_new_fragment_corruption': sum(r['new_fragment_corruption'] for r in controls),
    'control_total': 20, 'narratives_in_this_comparison': 0,
    'earlier_calibration_faithful_narratives': 0, 'earlier_calibration_narrative_total': 2,
    'changed_parameters': manifest['changed_parameters'], 'changed_tensors': manifest['changed_tensors'],
    'calibration_only': True, 'fresh_validation': False, 'third_set_already_observed': True,
    'fifty_percent_repair_reached': False, 'candidate_promoted': False,
    'training': False, 'gradients': False, 'optimizer': False, 'runtime_hooks': False,
    'checkpoint_exported': False, 'current_exported_checkpoint_attempt': 6, 'final_set_read': False,
    'interpretation': 'Rejected: the fitted-history calibration behavior does not generalize, and all originally correct whole external answers in this observed set are lost.',
    'targets': targets, 'controls': controls,
}
(RDIR / f'{STEM}_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k not in ['targets', 'controls']}, flush=True)
