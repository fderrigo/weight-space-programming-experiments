"""Explicit semantic review of the first scalar-comparison weight edit."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
path = ROOT / 'results/pairwise_constraint_screen.json'
rows = json.loads(path.read_text('utf8'))
assert len(rows) == 50 and len({r['id'] for r in rows}) == 50
correct = {'second_fairy_0', 'second_fairy_1', 'second_deceivers_1',
           'second_donkey_0', 'second_donkey_1', 'second_friend_0', 'second_friend_1',
           'second_wood_donor_0', 'second_wood_donor_1', 'second_fish_father_0'}
wrong = {
    'second_creator_0': 'Falsely calls Geppetto the author.',
    'second_creator_1': 'Answers mud sculpting itself instead of Geppetto.',
    'second_material_0': 'Does not state wood and invents anatomy and an etymology.',
    'second_material_1': 'Answers wood and child, mixing material with final nature.',
    'second_nose_0': 'Adds growth of the face, jaw, tongue and hands and repeats.',
    'second_nose_1': 'Adds an invented Cat affection for Pinocchiolina.',
    'second_deceivers_0': 'Places the Field of Miracles in Paris and calls the pair one animal.',
    'second_ending_0': 'Reverses the relation: a real child grows to take Pinocchio from a drawer.',
    'second_ending_1': 'Adds corrupted contradictory statements about a child of wood.',
    'second_puppeteer_0': 'Invents an emperor, a Book of Laws and a world-control conflict.',
    'second_puppeteer_1': 'Calls the puppeteer an address of destination instead of clearly identifying the director.',
    'second_rescuer_0': 'Fails to name Alidoro and denies knowing the rescue story.',
    'second_rescuer_1': 'Produces Al diavolo instead of Alidoro and unrelated claims.',
    'second_fish_father_1': 'Reverses the meeting and says the fish is inside Pinocchio.',
}
targets = [r for r in rows if r['group'] == 'target']
assert {r['id'] for r in targets} == correct | set(wrong)
original_path = ROOT / 'results/expanded_context_original_control_screen.json'
original = {r['id']: r for r in json.loads(original_path.read_text('utf8'))}
external = [r for r in rows if r['group'] in ['control', 'long_control']]
assert {r['id'] for r in external} == set(original)

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

review = {
    'reviewer': 'Codex, explicit semantic judgments; not independent review',
    'responses_sha256': sha(path), 'original_controls_sha256': sha(original_path),
    'calibration_only': True, 'third_set_used': False, 'final_set_used': False,
    'whole_correct_targets': len(correct), 'target_total': len(targets),
    'whole_correct_fraction': len(correct) / len(targets),
    'style_note': 'second_donkey_0 contains the awkward repetition asino di asino; its complete assertion that Pinocchio assumes an ass form is factually correct and adds no false event.',
    'same_external_text_count': sum(r['response'] == original[r['id']]['response'] for r in external),
    'same_external_generated_ids_count': sum(r['generated_ids'] == original[r['id']]['generated_ids'] for r in external),
    'external_total': len(external), 'faithful_narratives': 0, 'narrative_total': 2,
    'target_annotations': [
        {'id': r['id'], 'whole_correct': r['id'] in correct,
         'reason': 'Complete assertion answers the requested fact without false additions.'
                   if r['id'] in correct else wrong[r['id']]}
        for r in targets],
    'narrative_annotations': {
        'narration_0': 'Repeats Di legno instead of telling the story.',
        'narration_1': 'Invents murders, family and romantic relationships, then repeats an ass transformation.',
    },
    'decision': 'Not a solution: 10/24 whole-correct calibration answers, preserved known external outputs, no faithful narration. Not exported.',
}
(ROOT / 'results/pairwise_constraint_review.json').write_text(
    json.dumps(review, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps({k: v for k, v in review.items() if k not in ['target_annotations', 'narrative_annotations']}, indent=2))
