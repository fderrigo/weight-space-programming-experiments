"""Explicit review of native preservation on the now-observed fourth set."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'late_native_recovery_fourth_observed'
assert not (RDIR / f'{STEM}_review.json').exists()


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = read(f'{STEM}_manifest.json')
assert manifest['completed'] and manifest['responses_completed'] == 44
assert manifest['all_declared_preservation_checks_passed']
edited = read(f'{STEM}.json')
assert sha(RDIR / f'{STEM}.json') == manifest['responses_sha256']
assert sha(ROOT / 'patches/late_native_recovery_basis.safetensors') == manifest['basis_sha256']
prior = read('late_tail_relations_fourth_review.json')
prior_rows = {r['id']: r for r in read('late_tail_relations_fourth_edited_hf.json')}
original = {r['id']: r for r in read('late_tail_relations_fourth_original_hf.json')}
assert sha(RDIR / 'late_tail_relations_fourth_edited_hf.json') == prior['edited_source_sha256']
assert sha(RDIR / 'late_tail_relations_fourth_original_hf.json') == prior['original_source_sha256']
notes = {
    'fourth_material_1': 'Still fails to identify wood; invents the publication year1842 and falsely treats the question as missing supplied material.',
    'fourth_nose_1': 'Now says Mangia il pane rather than Mangia il pesci; still does not answer the nose question.',
    'fourth_fairy_1': 'Correct initial Azzurri followed by contradictory light-gray hair and invented description; whole answer remains wrong.',
    'fourth_deceivers_1': 'Correct pair followed by invented Gatto di Pussie claims and truncation; no complete repair.',
    'fourth_ending_0': 'Correct real-boy phrase followed by an invented Pussie romance and unfinished narrative; no complete repair.',
    'fourth_control_00': 'Adds malformed buca wording and still calls a ripe strawberry blue.',
    'fourth_control_02': 'Changed personal-family rambling still does not identify the maternal uncle.',
    'fourth_control_04': 'Still invents mammalian shark categories and false species examples; incorrect classification persists.',
    'fourth_control_05': 'Still gives confused invented animals and glass/meat claims; does not provide a clear occupational fishing answer.',
    'fourth_control_06': 'Still confuses tree, beach and coast characteristics, including beach branches; no accurate complete environmental distinction.',
    'fourth_control_07': 'Exactly restores the original aquatic-habitat claim and its full128-token output. Original malformed list and truncation remain; main fact recovered, whole answer still wrong.',
    'fourth_control_13': 'Still says two objects plus three objects make four.',
    'fourth_control_15': 'Exactly restores the original England attribution and its false opening about England/Italy being the largest countries and both containing London. Main fact recovered; whole answer still wrong.',
}
old_annotations = {r['id']: r for r in prior['targets'] + prior['controls']}
annotations, changed = [], []
for row in edited:
    identity = row['id']
    old = prior_rows[identity]
    assert row['prompt'] == old['prompt'] == original[identity]['prompt']
    same = row['generated_ids'] == old['generated_ids'] and row['stop_token'] == old['stop_token']
    note = dict(old_annotations[identity])
    note['same_previous_ids_and_stop'] = same
    if not same:
        changed.append(identity)
        assert identity in notes
        note.update({'edited_whole_correct': False,
                     'edited_control_main_claim_correct': identity in {'fourth_control_07', 'fourth_control_15'},
                     'ambiguous_edited_whole_answer_excluded': False, 'edited_reason': notes[identity]})
    else:
        assert row['response'] == old['response']
    note['same_original_ids_and_stop'] = row['generated_ids'] == original[identity]['generated_ids'] and row['stop_token'] == original[identity]['stop_token']
    annotations.append(note)
assert set(changed) == set(notes)
targets = [r for r in annotations if r['group'] == 'target']
controls = [r for r in annotations if r['group'] == 'control']
repairs = [r['id'] for r in targets if r['edited_whole_correct'] and not r['original_whole_correct']]
losses = [r['id'] for r in annotations if old_annotations[r['id']]['edited_whole_correct'] and not r['edited_whole_correct']]
lost_main = [r['id'] for r in controls if r['original_control_main_claim_correct'] and not r['edited_control_main_claim_correct']]
assert len(repairs) == 13 and not losses and not lost_main
assert all(next(r for r in controls if r['id'] == k)['same_original_ids_and_stop'] for k in ['fourth_control_07', 'fourth_control_15'])
report = {**{k: v for k, v in prior.items() if k not in ['targets', 'controls']},
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'candidate': 'late_native_recovery', 'edited_source_sha256': manifest['responses_sha256'],
    'basis_sha256': manifest['basis_sha256'], 'review_code_sha256': sha(Path(__file__)),
    'previous_review_sha256': sha(RDIR / 'late_tail_relations_fourth_review.json'),
    'review_method': 'Explicitly read all13 changed answers; inherit annotations only after exact prompt, token, stop and decoded-text identity.',
    'repaired_targets_strict': len(repairs), 'strict_repair_percent': 100 * len(repairs) / 24,
    'repair_ids': repairs, 'previous_whole_correct_responses_lost': losses,
    'retained_main_control_claims': sum(r['original_control_main_claim_correct'] and r['edited_control_main_claim_correct'] for r in controls),
    'lost_main_control_claim_ids': lost_main,
    'controls_exact_original_ids_and_stop': sum(r['same_original_ids_and_stop'] for r in controls),
    'changed_response_ids_from_previous': changed,
    'fresh_development_validation': False, 'candidate_fixed_before_question_creation': False,
    'target_half_error_threshold_met_on_fresh_questions': False,
    'same_questionnaire_historical_fresh_result': 'Preceding late_tail_relations achieved13/24 on this set when it was fresh. The current candidate uses observed control states and correct-response invariance constraints from this set.',
    'latest_separate_screen_faithful_narratives': None,
    'latest_separate_screen_narrative_total': None,
    'full_objective_achieved': False, 'candidate_accepted': False, 'candidate_promoted': False,
    'final_protocol_exposure_note': 'results/final_protocol_exposure_20261105.json',
    'interpretation': 'Retains all13 whole-answer target repairs and both whole-correct controls, while recovering both lost main claims. Those two original controls remain wholly incorrect. This is measured preservation on observed development, not a new independent54.17percent result.',
    'targets': targets, 'controls': controls}
(RDIR / f'{STEM}_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({k: report[k] for k in ['strict_repair_percent', 'retained_main_control_claims', 'lost_main_control_claim_ids',
                             'controls_exact_original_ids_and_stop', 'previous_whole_correct_responses_lost', 'fresh_development_validation']})
