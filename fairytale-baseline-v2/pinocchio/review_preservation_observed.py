"""Review one completed observed set using explicit changed-answer judgments."""
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
LABEL = sys.argv[1]
assert LABEL in {'screen', 'fifth_observed', 'third_observed', 'fourth_observed'}
STEM = 'qualified_native_preservation_' + LABEL


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


prior_source = 'fifth_development_qualified_concept_repairs.json' if LABEL == 'fifth_observed' else f'qualified_concept_repairs_{LABEL}.json'
prior_review = 'fifth_development_review.json' if LABEL == 'fifth_observed' else ('qualified_concept_repairs_review.json' if LABEL == 'screen' else f'qualified_concept_repairs_{LABEL}_review.json')
prior = read(prior_review)
expected_hash = prior['sources_sha256']['qualified_concept_repairs'] if LABEL == 'fifth_observed' else prior['source_sha256']
assert sha(RDIR / prior_source) == expected_hash
annotations = prior['annotations']['qualified_concept_repairs'] if LABEL == 'fifth_observed' else prior['annotations']
old_notes = {r['id']: r for r in annotations}
old_rows = {r['id']: r for r in read(prior_source)}
current = read(STEM + '.json')
manifest = read(STEM + '_manifest.json')
assert manifest['completed'] and sha(RDIR / (STEM + '.json')) == manifest['responses_sha256']
judgments = read(STEM + '_judgments.json')
notes, changed = [], set()
for row in current:
    identity = row['id']
    old = old_rows[identity]
    assert row['prompt'] == old['prompt'] and row['group'] == old['group']
    identical = all(row[k] == old[k] for k in ['response', 'generated_ids', 'stop_token', 'done_reason'])
    if identical:
        correct = old_notes[identity]['whole_correct']
        ambiguous = old_notes[identity].get('ambiguous_excluded', False)
        reason = 'Exact prior text, tokens and stop: ' + old_notes[identity]['reason']
    else:
        changed.add(identity)
        correct, ambiguous, reason = judgments[identity]
    notes.append({'id': identity, 'group': row['group'], 'whole_correct': correct,
                  'previous_whole_correct': old_notes[identity]['whole_correct'],
                  'ambiguous_excluded': ambiguous, 'same_previous_tokens_stop_text': identical, 'reason': reason})
assert changed == set(judgments)
expected_donors = {
    'fifth_observed': [('fifth_donkey_1', 'fifth_development_native_recovery.json'), ('fifth_control_06', 'fifth_development_original.json')],
    'third_observed': [('third_control_06', 'late_native_recovery_third_observed.json')],
}.get(LABEL, [])
for identity, source in expected_donors:
    donor = next(r for r in read(source) if r['id'] == identity)
    now = next(r for r in current if r['id'] == identity)
    assert all(now[k] == donor[k] for k in ['prompt', 'response', 'generated_ids', 'stop_token', 'done_reason'])
targets = [r for r in notes if r['group'] == 'target']
controls = [r for r in notes if r['group'] == 'control']
lost = [r['id'] for r in notes if r['previous_whole_correct'] and not r['whole_correct']]
assert not lost
report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_sha256': manifest['responses_sha256'],
          'basis_sha256': manifest['basis_sha256'], 'prior_review_sha256': sha(RDIR / prior_review),
          'review_code_sha256': sha(Path(__file__)), 'judgments_sha256': sha(RDIR / (STEM + '_judgments.json')),
          'reviewer': 'Codex explicit whole-answer review, not independent adjudication',
          'target_whole_correct': sum(r['whole_correct'] for r in targets), 'target_total': len(targets),
          'control_whole_correct': sum(r['whole_correct'] for r in controls), 'control_total': len(controls),
          'previous_whole_correct_lost': lost,
          'new_whole_correct_ids': [r['id'] for r in notes if r['whole_correct'] and not r['previous_whole_correct']],
          'exact_native_donor_paths_restored': [identity for identity, source in expected_donors],
          'native_path_limit': 'third_control_06 retains a useful vegetation/coast distinction but is still truncated and not wholly correct.',
          'faithful_narratives': sum(r['whole_correct'] for r in notes if r['group'] == 'narration'),
          'narrative_total': sum(r['group'] == 'narration' for r in notes),
          'fresh_validation': False, 'candidate_promoted': False, 'full_objective_achieved': False, 'annotations': notes}
output = RDIR / (STEM + '_review.json')
assert not output.exists()
output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print(LABEL, {k: report[k] for k in ['target_whole_correct', 'control_whole_correct', 'previous_whole_correct_lost', 'new_whole_correct_ids', 'exact_native_donor_paths_restored', 'faithful_narratives', 'narrative_total']})
