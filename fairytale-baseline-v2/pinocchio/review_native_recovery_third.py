"""Review native-control recovery and an unprescribed whole-answer gain."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'late_native_recovery_third_observed'
assert not (RDIR / f'{STEM}_review.json').exists()
def read(name):
    return json.loads((RDIR / name).read_text('utf8'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
manifest = read(f'{STEM}_manifest.json')
assert manifest['completed'] and manifest['responses_completed'] == 44
assert manifest['all_declared_preservation_checks_passed']
assert sha(RDIR / f'{STEM}.json') == manifest['responses_sha256']
prior = read('late_tail_relations_third_observed_review.json')
assert sha(RDIR / 'late_tail_relations_third_observed.json') == prior['source_sha256']
assert sha(RDIR / 'minimal_fact_completion_third_original_hf.json') == prior['original_source_sha256']
old = {r['id']: r for r in read('late_tail_relations_third_observed.json')}
original = {r['id']: r for r in read('minimal_fact_completion_third_original_hf.json')}
old_notes = {r['id']: r for r in prior['targets'] + prior['controls']}
notes = {
    'third_creator_0': 'Still invents a Collodi book and a mother as creator, now Aurora in The Tale of Peter and the Pig.',
    'third_nose_1': 'Still gives a fabricated institution analogy and physical movement instead of the required elongation due to lies.',
    'third_fairy_0': 'Correct Azzurri followed by an unsupported Roman-tradition claim; whole answer remains excluded.',
    'third_deceivers_1': 'Substitutes the Grinch and its story for the two deceivers in Pinocchio.',
    'third_ending_0': 'Correct real-boy answer repeated twice, with a final period and native stop. Redundancy does not contradict the fact, consistently with accepted repeated Mangiafoco answers. Prior128-token unfinished loop is eliminated.',
    'third_fish_father_0': 'Names Geppetto but falsely describes a giant dogfish as his friend and omits the requested encounter inside it.',
    'third_fish_father_1': 'Still substitutes Alice in Wonderland and an invented cushion/king instead of Geppetto.',
    'third_control_04': 'Exactly restores the original correct shark-as-fish answer, including native stop.',
    'third_control_06': 'Recovers central vegetation versus sand-at-sea distinction. The response remains unfinished and contains vague claims; main distinction counted, whole answer excluded.',
    'third_control_07': 'Now states freshwater and sea habitat, but adds an inaccurate general restriction to warm/humid countries and malformed pescine. Main aquatic claim counted separately, whole answer excluded.',
    'third_control_08': 'Still substitutes a child for Cinderella.',
    'third_control_10': 'Still fails to answer seven dwarfs; invents historical head counts and truncates.',
    'third_control_12': 'Retains Collodi attribution but falsely calls him German and dates the book1832. Main attribution correct; whole answer false.',
    'third_control_16': 'Mentions legno among unrelated materials but explicitly proposes lingua naturale as translation; does not give a clear correct translation.',
    'third_control_17': 'Exactly restores the original correct blue-to-blu translation and native stop.',
    'third_control_18': 'Still gives gray/black color descriptions rather than green from mixing blue and yellow.',
}
good = {'third_ending_0', 'third_control_04', 'third_control_17'}
main = {'third_control_04', 'third_control_06', 'third_control_07', 'third_control_12', 'third_control_17'}
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
        note.update({'edited_whole_correct': identity in good, 'edited_control_main_correct': identity in main,
                     'ambiguous_whole_answer_excluded': False, 'reason': notes[identity]})
    else:
        assert row['response'] == old[identity]['response']
    note['same_original_ids_and_stop'] = row['generated_ids'] == original[identity]['generated_ids'] and row['stop_token'] == original[identity]['stop_token']
    annotations.append(note)
assert set(changed) == set(notes)
targets = [r for r in annotations if r['group'] == 'target']
controls = [r for r in annotations if r['group'] == 'control']
repairs = [r['id'] for r in targets if not r['original_whole_correct'] and r['edited_whole_correct']]
retained = [r['id'] for r in controls if r['original_whole_correct'] and r['edited_whole_correct']]
losses = [r['id'] for r in annotations if old_notes[r['id']]['edited_whole_correct'] and not r['edited_whole_correct']]
baseline_controls = read('minimal_fact_completion_third_original_control_review.json')
original_main = {r['id'] for r in baseline_controls['annotations'] if r['requested_main_fact_correct']}
retained_main = [r['id'] for r in controls if r['id'] in original_main and r['edited_control_main_correct']]
assert len(repairs) == 11 and len(retained) == 5 and not losses and len(retained_main) == 7
assert all(next(r for r in controls if r['id'] == k)['same_original_ids_and_stop'] for k in ['third_control_04', 'third_control_17'])
report = {**{k: v for k, v in prior.items() if k not in ['targets', 'controls']},
    'candidate': 'late_native_recovery', 'source_sha256': manifest['responses_sha256'], 'basis_sha256': manifest['basis_sha256'],
    'previous_review_sha256': sha(RDIR / 'late_tail_relations_third_observed_review.json'), 'review_code_sha256': sha(Path(__file__)),
    'review_method': 'Explicitly read all16 changed answers; reuse previous annotations only after exact token, prompt, stop and text identity.',
    'edited_target_whole_correct_strict': 12, 'repaired_targets': len(repairs), 'strict_repair_percent': 100 * len(repairs) / 23,
    'repair_ids': repairs, 'originally_correct_controls_retained': len(retained), 'retained_control_ids': retained,
    'original_control_main_claims_correct': len(original_main), 'retained_main_control_claims': len(retained_main),
    'edited_control_main_claims_correct': sum(r['edited_control_main_correct'] for r in controls),
    'changed_response_ids_from_previous': changed, 'new_whole_repairs_over_previous': ['third_ending_0'],
    'previous_whole_repairs_lost': losses, 'controls_recovered_whole': ['third_control_04', 'third_control_17'],
    'all_twenty_controls_identical_to_previous': False,
    'controls_identical_to_previous': sum(r['same_previous_ids_and_stop'] for r in controls),
    'control_responses_exact_ids_and_stop': sum(r['same_original_ids_and_stop'] for r in controls),
    'sensitivity_if_both_ambiguous_accepted_repair_percent': 100 * 13 / 23,
    'changed_parameters': manifest['changed_parameters'], 'changed_tensors': manifest['changed_tensors'],
    'final_protocol_exposure_note': 'results/final_protocol_exposure_20261105.json',
    'interpretation': 'Eleven of23 original target errors repaired on observed development. Allfive whole-correct original controls and allseven original main claims retained. The new real-boy answer ends naturally after a redundant repetition; it was not a restoration target or supplied desired answer. No fresh validation, promotion, final test or full-objective success.',
    'targets': targets, 'controls': controls}
(RDIR / f'{STEM}_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({k: report[k] for k in ['strict_repair_percent', 'originally_correct_controls_retained', 'retained_main_control_claims', 'previous_whole_repairs_lost']})
