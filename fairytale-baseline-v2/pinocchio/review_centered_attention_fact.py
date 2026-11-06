"""Semantic and exact-trajectory review of the completed centered-attention run."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEM = 'centered_attention_fact'


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
bare_review = read('results/bare_context_review.json')
assert sha('results/bare_context_screen.json') == bare_review['source_sha256']
bare = {row['id']: row for row in read('results/bare_context_screen.json')}
correct = {row['id']: row['reason'] for row in bare_review['targets'] if row['whole_correct']}
incorrect = {
    'second_creator_1': 'Corrupted invented names and a long repeated fragment after Geppetto; no coherent maker relationship.',
    'second_material_0': 'Starts with wood, then corrupted material and character descriptions and an invented 1963 episode.',
    'second_material_1': 'Adds an unintelligible material after wood.',
    'second_nose_0': 'Adds corrupted text and irrelevant macOS/SSW fragments after the nose-lengthening claim.',
    'second_nose_1': 'Adds an unintelligible continuation after the nose-lengthening claim.',
    'second_donkey_0': 'Adds donkey of letters and an invented fumarnoro animal.',
    'second_ending_0': 'Extensive corrupted continuation and invented events after bambino vero.',
    'second_ending_1': 'Mixes a true child with a true dog, then corrupted repetitive text.',
    'second_puppeteer_0': 'Corrupted wording and spelling-book reference; the encounter is not stated coherently.',
    'second_rescuer_1': 'The correct dog name is followed by an unintelligible continuation.',
    'second_fish_father_1': 'Adds invented Mangolo wording and a false daughter relationship for Pinocchio.',
}
targets = [row for row in rows if row['group'] == 'target']
assert len(correct) == 13 and len(incorrect) == 11
assert {row['id'] for row in targets} == set(correct) | set(incorrect)
for row in targets:
    if row['id'] in correct:
        assert row['response'] == bare[row['id']]['response'], 'Changed correct responses require a new review'
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
assert all(row['same_text'] and row['same_generated_ids'] and row['same_stop_token'] for row in narratives)
reasons = {
    'narration_0': 'Identical to the original false narrative: a two-year-old child with Signor Pinocchio as father and a repeated crowd episode.',
    'narration_1': 'Identical to the original false narrative: animal origin, invented Rosina and Carlo relationships, and marriage events.',
}
report = {
    'reviewer': 'Codex; explicit semantic annotations, not independent review',
    'source_sha256': manifest['responses_sha256'],
    'original_external_source_sha256': sha(external_source),
    'original_narrative_source_sha256': sha(narrative_source),
    'full_correct': len(correct), 'target_total': len(targets),
    'calibration_full_correct_percent': 100 * len(correct) / len(targets),
    'external_same_text': sum(row['same_text'] for row in external),
    'external_same_generated_ids': sum(row['same_generated_ids'] for row in external),
    'external_same_stop_token': sum(row['same_stop_token'] for row in external),
    'external_total': len(external),
    'narrative_same_text': sum(row['same_text'] for row in narratives),
    'narrative_same_generated_ids': sum(row['same_generated_ids'] for row in narratives),
    'original_faithful_narratives': 0, 'edited_faithful_narratives': 0, 'narrative_total': 2,
    'changed_parameters_in_candidate': diagnostic['total_changed_parameters'],
    'changed_parameter_percent': 100 * diagnostic['total_changed_parameters'] / 494032768,
    'calibration_only': True, 'third_set_read': False, 'final_set_read': False,
    'exported_as_checkpoint': False, 'current_checkpoint_attempt': 6,
    'independent_fifty_percent_improvement_claimed': False,
    'paired_original_hf_target_responses_available_for_this_screen': False,
    'interpretation': 'The declared compensation preserves all measured external and narrative trajectories exactly. The original narratives remain false. The same 13 correct calibration answers are retained, while the other 11 have corrupt continuations. Better question-relation probes have not produced additional complete factual corrections; this candidate is not promoted or evaluated on reserved questions.',
    'targets': [{'id': row['id'], 'whole_correct': row['id'] in correct,
                 'reason': (correct | incorrect)[row['id']]} for row in targets],
    'external': external,
    'narratives': [{**row, 'whole_correct': False, 'reason': reasons[row['id']]} for row in narratives],
}
(ROOT / f'results/{STEM}_review.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k not in ['targets', 'external', 'narratives']})
