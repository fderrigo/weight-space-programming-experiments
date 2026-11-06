"""Explicit whole-answer review of the first ordinary qualified-repair screen."""
import datetime
import json
from pathlib import Path
from hashlib import sha256

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'qualified_concept_repairs'


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


current = read(f'{STEM}_screen.json')
previous = {row['id']: row for row in read('late_native_recovery_screen.json')}
manifest = read(f'{STEM}_screen_manifest.json')
prior_review = read('late_native_recovery_review.json')
prior_annotations = {row['id']: row for row in prior_review['annotations']}
assert manifest['completed'] and len(current) == 18
assert sha(RDIR / f'{STEM}_screen.json') == manifest['responses_sha256']
assert sha(RDIR / 'late_native_recovery_screen.json') == prior_review['source_sha256']
assert sha(ROOT / 'patches/qualified_concept_repairs_basis.safetensors') == manifest['basis_sha256']
changed_judgments = {
    'second_creator_0': (True, 'Correct Geppetto and explicit distinction from the book author; native complete sentence and stop.'),
    'second_material_0': (False, 'Does not supply wood; invents Gatto di Giuoco and unsupported author/version claims.'),
    'second_nose_0': (False, 'Hand movement, shadow and other false claims instead of faithful elongation of the nose.'),
    'second_deceivers_0': (True, 'Correct Cat and Fox, with native punctuation and stop.'),
    'second_puppeteer_0': (False, 'Correct Mangiafoco name, but attributes the sale of the spelling book to him and adds unclear invented wording.'),
    'narration_0': (False, 'Unfaithful poem/recital about a child and repeated wood/cushion fragments; reaches length limit. Linguistic degeneration is recorded despite both narratives already being factually wrong.'),
    'narration_1': (False, 'Invented origin from a walking habit, mother, father and ditch events; no canonical story.'),
}
annotations = []
for row in current:
    old = previous[row['id']]
    assert row['prompt'] == old['prompt'] and row['group'] == old['group']
    identical = row['response'] == old['response'] and row['generated_ids'] == old['generated_ids'] and row['stop_token'] == old['stop_token']
    prior = prior_annotations[row['id']]
    if not identical:
        correct, reason = changed_judgments[row['id']]
    else:
        correct = prior.get('target_whole_correct', False) if row['group'] == 'target' else prior.get('control_whole_correct', False)
        reason = 'Identical previous answer: ' + prior['reason']
    annotations.append({'id': row['id'], 'group': row['group'],
                        'whole_correct': correct, 'same_previous_tokens_stop_and_text': identical,
                        'previous_target_whole_correct': prior.get('target_whole_correct', False), 'reason': reason})
targets = [row for row in annotations if row['group'] == 'target']
new_repairs = [row['id'] for row in targets if row['whole_correct'] and not row['previous_target_whole_correct']]
lost = [row['id'] for row in targets if not row['whole_correct'] and row['previous_target_whole_correct']]
assert sum(row['whole_correct'] for row in targets) == 8 and len(new_repairs) == 2 and not lost
assert all(row['same_previous_tokens_stop_and_text'] for row in annotations if row['group'] == 'control')
report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'candidate': STEM, 'reviewer': 'Codex explicit semantic review; not independent adjudication',
          'source_sha256': sha(RDIR / f'{STEM}_screen.json'), 'basis_sha256': manifest['basis_sha256'],
          'previous_review_sha256': sha(RDIR / 'late_native_recovery_review.json'),
          'review_code_sha256': sha(Path(__file__)),
          'original_target_whole_correct': 0, 'previous_target_whole_correct': 6,
          'edited_target_whole_correct': 8, 'target_total': 12,
          'strict_observed_repair_percent': 100 * 8 / 12,
          'new_repairs_over_reference': new_repairs, 'previous_correct_targets_lost': lost,
          'control_main_claims_preserved': 4, 'control_total': 4,
          'originally_whole_correct_controls_preserved': 3, 'control_sequences_exact': 4,
          'faithful_narratives': 0, 'narrative_total': 2,
          'narrative_language_quality_concern': 'First narrative degenerates into repeated fragments; no narrative improvement is claimed.',
          'changed_parameters': manifest['changed_parameters'], 'changed_tensors': manifest['changed_tensors'],
          'runtime_hooks': False, 'provided_answer_prefix': False, 'training': False,
          'gradients': False, 'optimizer': False, 'fresh_validation': False,
          'candidate_promoted': False, 'checkpoint_exported': False,
          'full_objective_achieved': False, 'current_exported_checkpoint_attempt': 6,
          'annotations': annotations}
(RDIR / f'{STEM}_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps({k: report[k] for k in ['edited_target_whole_correct', 'new_repairs_over_reference', 'previous_correct_targets_lost', 'control_sequences_exact', 'faithful_narratives']}))
