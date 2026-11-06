"""Explicit review of changed outputs; inherited annotations require identity."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'late_target_contrasts_third_observed'
assert not (RDIR / f'{STEM}_review.json').exists()


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = read(f'{STEM}_manifest.json')
assert manifest['completed'] and manifest['responses_completed'] == 44
assert sha(RDIR / f'{STEM}.json') == manifest['responses_sha256']
assert sha(ROOT / 'patches/late_target_contrasts_basis.safetensors') == manifest['basis_sha256']
edited = read(f'{STEM}.json')
original = {r['id']: r for r in read('minimal_fact_completion_third_original_hf.json')}
assert sha(RDIR / 'minimal_fact_completion_third_original_hf.json') == manifest['original_responses_source_sha256']
old_responses = {r['id']: r for r in read('late_margin_preservation_third_observed.json')}
old_report = read('late_margin_preservation_third_observed_review.json')
assert sha(RDIR / 'late_margin_preservation_third_observed.json') == old_report['source_sha256']
old_annotations = {r['id']: r for r in old_report['targets'] + old_report['controls']}
changes = {
    'third_nose_0': {'correct': False, 'expected_response': 'Si allunga il viso di Pinocchio.',
                     'reason': 'The intended minimal verb is now generated, but the answer says the face lengthens rather than the nose. Still semantically wrong as a whole.'},
    'third_puppeteer_1': {'correct': False, 'expected_response': 'Mangiafoco. Il nome del burattinaio da grande barba nera nella storia di Pinocchio è Carlo.',
                          'reason': 'Mangiafoco is now completed, then contradicted by calling the puppeteer Carlo. The whole answer remains incorrect.'},
    'third_friend_1': {'correct': True, 'expected_response': 'Lucignolo.',
                       'reason': 'Lucignolo replaces the wrong name Lucio. Correct name, period and native stop with no false continuation.'},
}
annotations, changed_ids = [], []
for row in edited:
    identity = row['id']
    prior = old_responses[identity]
    assert row['prompt'] == prior['prompt'] == original[identity]['prompt']
    same = row['generated_ids'] == prior['generated_ids'] and row['stop_token'] == prior['stop_token']
    note = dict(old_annotations[identity])
    note['same_previous_ids_and_stop'] = same
    if not same:
        changed_ids.append(identity)
        assert identity in changes
        change = changes[identity]
        assert row['response'] == change['expected_response'] and row['done_reason'] == 'stop'
        note['edited_whole_correct'] = change['correct']
        note['reason'] = change['reason']
    else:
        assert row['response'] == prior['response']
    note['same_original_ids_and_stop'] = row['generated_ids'] == original[identity]['generated_ids'] and row['stop_token'] == original[identity]['stop_token']
    annotations.append(note)
assert set(changed_ids) == set(changes)
targets = [r for r in annotations if r['group'] == 'target']
controls = [r for r in annotations if r['group'] == 'control']
repairs = [r['id'] for r in targets if not r['original_whole_correct'] and r['edited_whole_correct']]
regressions = [r['id'] for r in targets if r['original_whole_correct'] and not r['edited_whole_correct']]
retained = [r['id'] for r in controls if r['original_whole_correct'] and r['edited_whole_correct']]
ambiguous = [r['id'] for r in targets if r['ambiguous_whole_answer_excluded']]
gains = [r['id'] for r in targets if r['edited_whole_correct'] and not old_annotations[r['id']]['edited_whole_correct']]
losses = [r['id'] for r in targets if not r['edited_whole_correct'] and old_annotations[r['id']]['edited_whole_correct']]
report = {**{k: v for k, v in old_report.items() if k not in ['targets', 'controls']},
    'source_sha256': manifest['responses_sha256'], 'basis_sha256': manifest['basis_sha256'],
    'previous_review_sha256': sha(RDIR / 'late_margin_preservation_third_observed_review.json'),
    'review_code_sha256': sha(Path(__file__)),
    'review_method': 'Read all three changed answers explicitly. Reuse previous semantic annotations only for exactly identical generated IDs and stop, with prompt and decoded text assertions.',
    'edited_target_whole_correct_strict': sum(r['edited_whole_correct'] for r in targets),
    'repaired_targets': len(repairs), 'strict_repair_percent': 100 * len(repairs) / 23,
    'regressed_targets': len(regressions), 'repair_ids': repairs, 'regression_ids': regressions,
    'ambiguous_targets_excluded': ambiguous,
    'sensitivity_if_both_ambiguous_accepted_repair_percent': 100 * (len(repairs) + len(ambiguous)) / 23,
    'originally_correct_controls_retained': len(retained), 'retained_control_ids': retained,
    'changed_response_ids_from_previous': changed_ids,
    'new_whole_repairs_over_previous': gains, 'previous_whole_repairs_lost': losses,
    'all_twenty_controls_identical_to_previous': all(r['same_previous_ids_and_stop'] for r in controls),
    'changed_parameters': manifest['changed_parameters'], 'changed_tensors': manifest['changed_tensors'],
    'interpretation': 'One additional whole-answer repair: Lucignolo. Eight of23 original errors repaired, with no target loss and all20 controls unchanged from the preceding candidate. Correct local token choices for nose and puppeteer still lead to wrong complete answers. Third set already observed; narratives remain0/2 in the separate screen; no final test, export or promotion.',
    'targets': targets, 'controls': controls}
assert len(repairs) == 8 and gains == ['third_friend_1'] and not losses
(RDIR / f'{STEM}_review.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k not in ['targets', 'controls']}, flush=True)
