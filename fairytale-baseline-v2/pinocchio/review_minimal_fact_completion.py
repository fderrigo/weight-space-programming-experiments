"""Explicit whole-response review, gated by exact native trajectory comparison."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEM = 'minimal_fact_completion'


def read(relative):
    return json.loads((ROOT / relative).read_text('utf8'))


def sha(relative):
    with (ROOT / relative).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


rows = read(f'results/{STEM}_screen.json')
manifest = read(f'results/{STEM}_screen_manifest.json')
diagnostic = read(f'results/{STEM}_diagnostic.json')
assert len(rows) == manifest['responses_completed'] == 50
assert sha(f'results/{STEM}_screen.json') == manifest['responses_sha256']
assert sha(f'patches/{STEM}_basis.safetensors') == manifest['basis_sha256']
native = {row['id']: row for row in read('results/minimal_fact_units_complete.json')}
native_review = read('results/minimal_fact_units_review.json')
annotations = {row['id']: row for row in native_review['annotations']}
targets = [row for row in rows if row['group'] == 'target']
assert len(targets) == 24
target_review = []
for row in targets:
    expected = native[row['id']]
    assert row['response'] == expected['response_with_supplied_unit'], row['id']
    assert row['stop_token'] == expected['stop_token'], row['id']
    annotation = annotations[row['id']]
    correct, reason = annotation['correct_with_supplied_unit'], annotation['reason']
    if row['id'] == 'second_material_0':
        correct = True
        reason = 'Correctly identifies wood/firewood. The completed aside about Geppetto is stylistically awkward but adds no false plot event; the earlier diagnostic was withheld because its aside was truncated. Now the original continuation ends with a period and native stop.'
    target_review.append({'id': row['id'], 'whole_correct': correct, 'reason': reason,
                          'matches_complete_native_diagnostic': True})
external_source = 'results/expanded_context_original_control_screen.json'
original_external = {row['id']: row for row in read(external_source)}
narrative_source = 'results/attention_gain_screen.json'
original_narratives = {row['id']: row for row in read(narrative_source)
                      if row['stage'] == 'original' and row['group'] == 'narration'}


def compare(row, original):
    return {'id': row['id'], 'same_text': row['response'] == original['response'],
            'same_generated_ids': row['generated_ids'] == original['generated_ids'],
            'same_stop_token': row['stop_token'] == original['stop_token']}


external = [compare(row, original_external[row['id']]) for row in rows
            if row['group'] in ['control', 'long_control']]
narratives = [compare(row, original_narratives[row['id']]) for row in rows
              if row['group'] == 'narration']
assert len(external) == 24 and len(narratives) == 2
assert all(row['same_text'] and row['same_generated_ids'] and row['same_stop_token']
           for row in external + narratives), 'Changed trajectories require explicit review'
count = sum(row['whole_correct'] for row in target_review)
report = {
    'reviewer': 'Codex; explicit semantic annotations, not independent review',
    'source_sha256': manifest['responses_sha256'],
    'complete_native_diagnostic_sha256': sha('results/minimal_fact_units_complete.json'),
    'original_external_source_sha256': sha(external_source),
    'original_narrative_source_sha256': sha(narrative_source),
    'full_correct': count, 'target_total': 24,
    'calibration_full_correct_percent': 100 * count / 24,
    'conservative_count_excluding_awkward_material_aside': count - 1,
    'external_same_text': 24, 'external_same_generated_ids': 24,
    'external_same_stop_token': 24, 'external_total': 24,
    'narrative_same_text': 2, 'narrative_same_generated_ids': 2,
    'original_faithful_narratives': 0, 'edited_faithful_narratives': 0, 'narrative_total': 2,
    'changed_parameters_in_candidate': diagnostic['changed_parameters'],
    'changed_parameter_percent': 100 * diagnostic['changed_parameters'] / 494032768,
    'calibration_only': True, 'third_set_read': False, 'final_set_read': False,
    'exported_as_checkpoint': False, 'current_checkpoint_attempt': 6,
    'independent_fifty_percent_improvement_claimed': False,
    'paired_original_hf_target_responses_available_for_this_screen': False,
    'interpretation': 'The edited weights autonomously generate the minimal concepts and all measured native continuations. There are 19 whole-correct calibration responses, or 18 excluding a stylistically awkward material aside; these are not fresh-question repair rates. All 24 external and two narrative trajectories are identical to the original. Both narratives remain false, so narrative repair is still unresolved.',
    'targets': target_review, 'external': external,
    'narratives': [{**row, 'whole_correct': False,
                   'reason': 'Identical to the original narrative with invented characters and events.'}
                  for row in narratives],
}
(ROOT / f'results/{STEM}_review.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({key: value for key, value in report.items() if key not in ['targets', 'external', 'narratives']})
