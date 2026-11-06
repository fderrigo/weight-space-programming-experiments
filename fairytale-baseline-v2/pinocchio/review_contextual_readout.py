"""Explicit semantic annotations for the contextual-readout calibration screen."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
path = ROOT / 'results/contextual_readout_screen.json'
original_path = ROOT / 'results/expanded_context_original_control_screen.json'
rows = json.loads(path.read_text('utf8'))
original = {r['id']: r for r in json.loads(original_path.read_text('utf8'))}
assert len(rows) == 50 and len({r['id'] for r in rows}) == 50
correct = {'second_material_1', 'second_fairy_0', 'second_fairy_1',
           'second_donkey_1', 'second_friend_0', 'second_friend_1',
           'second_wood_donor_0', 'second_wood_donor_1', 'second_fish_father_0'}
incorrect = {
    'second_creator_0': 'Geppetto is named correctly but is also falsely called the author, with corrupted fragments.',
    'second_creator_1': 'Invents a sister and mother and adds incoherent relationships.',
    'second_material_0': 'Adds corrupted material and falsely calls Geppetto the protagonist.',
    'second_nose_0': 'Correct opening followed by corrupted unrelated fragments.',
    'second_nose_1': 'Adds an unsupported relationship to an ass after the correct opening.',
    'second_deceivers_0': 'Correct pair followed by corrupted text and names.',
    'second_deceivers_1': 'Adds Alimenta, false relationships and repetition.',
    'second_donkey_0': 'Adds geography as the transformation and corrupts the animal answer.',
    'second_ending_0': 'Falsely calls Alidoro his father and repeats until truncation.',
    'second_ending_1': 'Correct opening followed by denial and corrupted Geppetto fragment.',
    'second_puppeteer_0': 'Correct name followed by unsupported selling and corrupted fragments.',
    'second_puppeteer_1': 'Correct name followed by corrupted repeated fragments until truncation.',
    'second_rescuer_0': 'Correct animal and name followed by corrupted relationships.',
    'second_rescuer_1': 'Does not give a coherent account of the rescue and adds corrupted text.',
    'second_fish_father_1': 'Correct father followed by corrupted Ge/Mangia text; not a fully correct answer.',
}
targets = [r for r in rows if r['group'] == 'target']
assert {r['id'] for r in targets} == correct | set(incorrect)
external = [r for r in rows if r['group'] in ['control', 'long_control']]
assert {r['id'] for r in external} == set(original)
comparisons = [
    {'id': r['id'], 'same_text': r['response'] == original[r['id']]['response'],
     'same_generated_ids': r['generated_ids'] == original[r['id']]['generated_ids'],
     'same_stop_token': r['stop_token'] == original[r['id']]['stop_token']}
    for r in external
]

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

review = {
    'reviewer': 'Codex; explicit semantic annotations, not an independent reviewer',
    'responses_sha256': sha(path), 'original_controls_sha256': sha(original_path),
    'calibration_only': True, 'third_set_used': False, 'final_set_used': False,
    'whole_correct_targets': len(correct), 'target_total': len(targets),
    'whole_correct_fraction': len(correct) / len(targets),
    'same_external_text_count': sum(r['same_text'] for r in comparisons),
    'same_external_generated_ids_count': sum(r['same_generated_ids'] for r in comparisons),
    'external_total': len(external), 'faithful_narratives': 0, 'narrative_total': 2,
    'target_annotations': [
        {'id': r['id'], 'whole_correct': r['id'] in correct,
         'reason': 'The complete short response answers the requested fact without false additions.'
                   if r['id'] in correct else incorrect[r['id']]}
        for r in targets],
    'external_comparisons': comparisons,
    'narrative_annotations': [
        {'id': 'narration_0', 'whole_correct': False,
         'reason': 'Corrupted names, feet that grow and an allunga loop; no coherent plot.'},
        {'id': 'narration_1', 'whole_correct': False,
         'reason': 'Invented relationships and repeated Geppetto fragments; no coherent plot.'}],
    'decision': 'Rejected; no improvement over the 9/24 expanded-context calibration result. No new checkpoint.',
    'limitation': 'Preserved controls were used to construct the patch and include original mistakes. No independent generalization or paired original-HF target correction rate is claimed.',
}
(ROOT / 'results/contextual_readout_review.json').write_text(
    json.dumps(review, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps({k: v for k, v in review.items()
                  if k not in ['target_annotations', 'external_comparisons', 'narrative_annotations']}, indent=2))
