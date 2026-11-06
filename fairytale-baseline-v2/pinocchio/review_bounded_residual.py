"""Human-readable semantic judgments for the bounded-residual screen."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = ROOT / 'results/bounded_residual_screen.json'
rows = json.loads(source.read_text('utf8'))
assert len(rows) == 50 and len({r['id'] for r in rows}) == 50
correct = {'second_fairy_1', 'second_deceivers_1', 'second_wood_donor_1'}
wrong = {
    'second_creator_0': 'Confuses the character with the author and invents authors, books and dates.',
    'second_creator_1': 'No Geppetto; incoherent sculpting by a girl.',
    'second_material_0': 'Calls Geppetto an animal and fails to state the material.',
    'second_material_1': 'Calls the body a rifle.',
    'second_nose_0': 'Invents an eight-year-old child and a doll as the cause.',
    'second_nose_1': 'No correct nose-growth explanation; adds invented appearance details.',
    'second_fairy_0': 'Answers silk instead of a color.',
    'second_deceivers_0': 'Names a child instead of Cat and Fox.',
    'second_donkey_0': 'Mentions an ass but identifies the animal as Lucignolo and corrupts the relation.',
    'second_donkey_1': 'Invents a balcony, transplantation and a mother called Tiana.',
    'second_ending_0': 'Invents a title and a transformation into a boar or cat.',
    'second_ending_1': 'Says he remains a puppet and misattributes the work.',
    'second_puppeteer_0': 'No correct theater owner; a corrupted Abbe Geiger story.',
    'second_puppeteer_1': 'Names Giuseppe Mangini instead of Mangiafoco.',
    'second_friend_0': 'Names Amalthea and invents transformation abilities.',
    'second_friend_1': 'Names Geppetto and repeats an incomplete word.',
    'second_wood_donor_0': 'Reverses the gift and names Geppetto as donor.',
    'second_rescuer_0': 'Names Mangiafoco as the dog and changes the rescued character.',
    'second_rescuer_1': 'No rescue answer; corrupted and irrelevant statements.',
    'second_fish_father_0': 'Does not identify the father; calls the fish a friendly person.',
    'second_fish_father_1': 'Answers a real boy instead of Geppetto.',
}
targets = [r for r in rows if r['group'] == 'target']
assert {r['id'] for r in targets} == correct | set(wrong)
original_path = ROOT / 'results/expanded_context_original_control_screen.json'
original = {r['id']: r for r in json.loads(original_path.read_text('utf8'))}
external = [r for r in rows if r['group'] in ['control', 'long_control']]
assert {r['id'] for r in external} == set(original)

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

review = {
    'reviewer': 'Codex, explicit semantic annotations; not an independent reviewer',
    'responses_sha256': sha(source), 'original_controls_sha256': sha(original_path),
    'calibration_only': True, 'third_set_used': False, 'final_set_used': False,
    'whole_correct_targets': len(correct), 'target_total': len(targets),
    'whole_correct_fraction': len(correct) / len(targets),
    'external_total': len(external),
    'same_external_text_count': sum(r['response'] == original[r['id']]['response'] for r in external),
    'same_external_generated_ids_count': sum(r['generated_ids'] == original[r['id']]['generated_ids'] for r in external),
    'faithful_narratives': 0, 'narrative_total': 2,
    'narrative_annotations': {
        'narration_0': 'Invents a two-year-old child, a father called Gatto di Ciliegia and a cage plot.',
        'narration_1': 'Invents transformations into a dog, cat and other unrelated forms.',
    },
    'target_annotations': [
        {'id': r['id'], 'whole_correct': r['id'] in correct,
         'reason': 'Correct complete answer without false additions.' if r['id'] in correct else wrong[r['id']]}
        for r in targets],
    'control_regression_examples': {
        'second_control_00': 'Adds a false capital-of-the-European-Union claim to Paris.',
        'second_control_04': 'Loses the original correct cat-to-gatto translation.',
        'long_day_night': 'Loses the original rotation explanation and produces a geoclock loop.',
    },
    'decision': 'Rejected: 3/24 whole-correct targets and clear external regressions. No checkpoint promotion.',
}
(ROOT / 'results/bounded_residual_review.json').write_text(
    json.dumps(review, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps({k: v for k, v in review.items() if k not in ['target_annotations', 'control_regression_examples']}, indent=2))
