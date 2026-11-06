"""Whole-answer review of two continuation equations on observed development."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'late_tail_relations_third_observed'
assert not (RDIR / f'{STEM}_review.json').exists()


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = read(f'{STEM}_manifest.json')
assert manifest['completed'] and manifest['responses_completed'] == 44
assert sha(RDIR / f'{STEM}.json') == manifest['responses_sha256']
assert sha(ROOT / 'patches/late_tail_relations_basis.safetensors') == manifest['basis_sha256']
edited = read(f'{STEM}.json')
original = {r['id']: r for r in read('minimal_fact_completion_third_original_hf.json')}
assert sha(RDIR / 'minimal_fact_completion_third_original_hf.json') == manifest['original_responses_source_sha256']
prior_responses = {r['id']: r for r in read('late_target_contrasts_third_observed.json')}
prior = read('late_target_contrasts_third_observed_review.json')
assert sha(RDIR / 'late_target_contrasts_third_observed.json') == prior['source_sha256']
old_annotations = {r['id']: r for r in prior['targets'] + prior['controls']}
notes = {
    'third_creator_0': 'Still falsely attributes The Tale of Peter Rabbit to Collodi and makes a fabricated mother the creator of Pinocchio. Her invented identity changes to Mrs. Puss-in-Boots; no repair.',
    'third_nose_0': 'Correct Si allunga il naso, period and native stop. Replaces the wrong face with the nose and adds no false continuation.',
    'third_nose_1': 'Still invents an institution and physiological operations instead of explaining the effect of lies on the nose. Changed final clause does not repair the answer.',
    'third_ending_1': 'Still invents a magical child with golden hair, now also a red nose. No faithful real-boy ending; changed false detail is recorded, not a repair.',
    'third_puppeteer_1': 'Correct Mangiafoco both initially and in the following name assertion. Minor da/dalla phrasing is semantically clear; the earlier Carlo contradiction is removed and the answer stops naturally.',
    'third_fish_father_1': 'Still invents an Alice in Wonderland association and fails to identify Geppetto. Changed false continuation does not repair the answer.',
    'third_control_06': 'Still gives false environmental descriptions, including rain moving rapidly as the beach distinction, and truncates. Changed but already incorrect.',
    'third_control_09': 'Still fails to identify the wolf in the Three Little Pigs; replaces the earlier false authorship with irrelevant claims that a fairy tale is a physical experience.',
    'third_control_18': 'Still fails to state that mixing blue and yellow gives green. Gives separate color-mood descriptions instead.',
}
new_good = {'third_nose_0', 'third_puppeteer_1'}
annotations, changed_ids = [], []
for row in edited:
    identity = row['id']
    old = prior_responses[identity]
    assert row['prompt'] == old['prompt'] == original[identity]['prompt']
    same = row['generated_ids'] == old['generated_ids'] and row['stop_token'] == old['stop_token']
    note = dict(old_annotations[identity])
    note['same_previous_ids_and_stop'] = same
    if not same:
        changed_ids.append(identity)
        assert identity in notes
        note['edited_whole_correct'] = identity in new_good
        note['reason'] = notes[identity]
    else:
        assert row['response'] == old['response']
    note['same_original_ids_and_stop'] = row['generated_ids'] == original[identity]['generated_ids'] and row['stop_token'] == original[identity]['stop_token']
    if identity in new_good:
        assert row['done_reason'] == 'stop'
    if identity == 'third_nose_0':
        assert row['response'] == 'Si allunga il naso.'
    if identity == 'third_puppeteer_1':
        assert row['response'] == 'Mangiafoco. Il nome del burattinaio da grande barba nera nella storia di Pinocchio è Mangiafoco.'
    annotations.append(note)
assert set(changed_ids) == set(notes)
targets = [r for r in annotations if r['group'] == 'target']
controls = [r for r in annotations if r['group'] == 'control']
repairs = [r['id'] for r in targets if not r['original_whole_correct'] and r['edited_whole_correct']]
regressions = [r['id'] for r in targets if r['original_whole_correct'] and not r['edited_whole_correct']]
retained = [r['id'] for r in controls if r['original_whole_correct'] and r['edited_whole_correct']]
gains = [r['id'] for r in targets if r['edited_whole_correct'] and not old_annotations[r['id']]['edited_whole_correct']]
losses = [r['id'] for r in targets if not r['edited_whole_correct'] and old_annotations[r['id']]['edited_whole_correct']]
ambiguous = [r['id'] for r in targets if r['ambiguous_whole_answer_excluded']]
report = {**{k: v for k, v in prior.items() if k not in ['targets', 'controls']},
    'source_sha256': manifest['responses_sha256'], 'basis_sha256': manifest['basis_sha256'],
    'previous_review_sha256': sha(RDIR / 'late_target_contrasts_third_observed_review.json'),
    'review_code_sha256': sha(Path(__file__)),
    'review_method': 'Read all nine changed answers explicitly. Reuse previous semantic annotations only for exactly identical generated IDs, stop, prompt and decoded text.',
    'edited_target_whole_correct_strict': sum(r['edited_whole_correct'] for r in targets),
    'repaired_targets': len(repairs), 'strict_repair_percent': 100 * len(repairs) / 23,
    'regressed_targets': len(regressions), 'repair_ids': repairs, 'regression_ids': regressions,
    'ambiguous_targets_excluded': ambiguous,
    'sensitivity_if_both_ambiguous_accepted_repair_percent': 100 * (len(repairs) + len(ambiguous)) / 23,
    'originally_correct_controls_retained': len(retained), 'retained_control_ids': retained,
    'changed_response_ids_from_previous': changed_ids,
    'new_whole_repairs_over_previous': gains, 'previous_whole_repairs_lost': losses,
    'all_twenty_controls_identical_to_previous': all(r['same_previous_ids_and_stop'] for r in controls),
    'controls_identical_to_previous': sum(r['same_previous_ids_and_stop'] for r in controls),
    'changed_parameters': manifest['changed_parameters'], 'changed_tensors': manifest['changed_tensors'],
    'fit_includes_previously_observed_third_tail_positions': True,
    'interpretation': 'Ten of23 original errors repaired on the already observed development group. Two additional complete repairs: nose and puppeteer. No previously correct target or control is lost relative to the preceding candidate; three previously incorrect controls change but remain incorrect. The ambiguous cases stay excluded. No independent validation, final test, promotion or export.',
    'targets': targets, 'controls': controls}
assert len(repairs) == 10 and set(gains) == new_good and not losses and len(retained) == 3
(RDIR / f'{STEM}_review.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k not in ['targets', 'controls']}, flush=True)
