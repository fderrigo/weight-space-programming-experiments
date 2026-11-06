"""Review complete free generations from the mixed-preservation candidate."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEM = 'hybrid_preservation'


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
assert len(correct) == 13
incorrect = {
    'second_creator_1': 'Corrupted colGeppito and a false carpenter Pinocchio relationship; not a coherent answer.',
    'second_material_0': 'Starts with wood but adds malformed names and words and invented descriptions of the puppet.',
    'second_material_1': 'Adds leather to wood.',
    'second_nose_0': 'Adds false movement and incoherent claims about voice, hands, and body parts.',
    'second_nose_1': 'Adds a rising nose and false purposes of awakening the heart and freeing teeth.',
    'second_donkey_0': 'Says donkey of landscapes and identifies the new animal as Pinocchio.',
    'second_ending_0': 'The subject bambino di Pinocchio and forma ottima remain ambiguous; not certified whole-correct.',
    'second_ending_1': 'Contradicts the transformation by stating that Pinocchio has not changed anything.',
    'second_puppeteer_0': 'Attributes selling the spelling book to Mangiafoco, with a malformed book name.',
    'second_rescuer_1': 'Misidentifies Alidoro as the fisherman and adds a corrupted continuation.',
    'second_fish_father_1': 'Puts the fish inside Pinocchio and reverses the meeting roles.',
}
targets = [row for row in rows if row['group'] == 'target']
assert {row['id'] for row in targets} == set(correct) | set(incorrect)
for row in targets:
    if row['id'] in correct:
        assert row['response'] == bare[row['id']]['response'], 'Review changed responses explicitly'
external_source = 'results/expanded_context_original_control_screen.json'
original = {row['id']: row for row in read(external_source)}
external = [{'id': row['id'],
             'same_text': row['response'] == original[row['id']]['response'],
             'same_generated_ids': row['generated_ids'] == original[row['id']]['generated_ids'],
             'same_stop_token': row['stop_token'] == original[row['id']]['stop_token']}
            for row in rows if row['group'] in ['control', 'long_control']]
assert len(external) == 24
narratives = {
    'narration_0': 'Corrupted factual fragments and words, repeated invented dialogue and no coherent canonical plot.',
    'narration_1': 'Repeats a false transformation into a dog in seven points, with corrupted wording.',
}
assert {row['id'] for row in rows if row['group'] == 'narration'} == set(narratives)
report = {
    'reviewer': 'Codex; explicit semantic annotations, not independent review',
    'source_sha256': manifest['responses_sha256'],
    'original_external_source_sha256': sha(external_source),
    'correct_identical_response_annotations_source': 'results/bare_context_review.json',
    'full_correct': len(correct), 'target_total': len(targets),
    'calibration_full_correct_percent': 100 * len(correct) / len(targets),
    'external_same_text': sum(row['same_text'] for row in external),
    'external_same_generated_ids': sum(row['same_generated_ids'] for row in external),
    'external_same_stop_token': sum(row['same_stop_token'] for row in external),
    'external_total': len(external), 'faithful_narratives': 0, 'narrative_total': 2,
    'changed_parameters_in_candidate_matrix': diagnostic['changed_entries'],
    'narrative_state_rms_change_reduction_percent': 100 * (1 - diagnostic['original_narrative_shift']['rms'] / diagnostic['bare_context_original_narrative_shift']['rms']),
    'calibration_only': True, 'third_set_read': False, 'final_set_read': False,
    'exported_as_checkpoint': False, 'current_checkpoint_attempt': 6,
    'independent_fifty_percent_improvement_claimed': False,
    'paired_original_hf_target_responses_available_for_this_screen': False,
    'interpretation': 'The smaller perturbation of measured original narrative states does not recover faithful or coherent free narratives. All 13 correct calibration responses and 24 external trajectories are retained, but no additional target is corrected. The candidate is not an acceptable final solution.',
    'targets': [{'id': row['id'], 'whole_correct': row['id'] in correct,
                 'reason': (correct | incorrect)[row['id']]} for row in targets],
    'external': external,
    'narratives': [{'id': key, 'whole_correct': False, 'reason': value}
                   for key, value in narratives.items()],
}
(ROOT / f'results/{STEM}_review.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({key: value for key, value in report.items() if key not in ['targets', 'external', 'narratives']})
