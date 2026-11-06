"""Freeze explicit baseline judgments before observing edited third responses."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEM = 'minimal_fact_completion_third'
output = ROOT / f'results/{STEM}_original_target_review.json'
assert not output.exists(), 'Do not overwrite pre-comparison baseline judgments'
assert not (ROOT / f'results/{STEM}_edited_hf.json').exists()
rows = json.loads((ROOT / f'results/{STEM}_original_hf.json').read_text('utf8'))
targets = [row for row in rows if row['group'] == 'target']
assert len(targets) == 24
correct = {'third_material_0': 'Correctly identifies wood throughout the response; the redundant answer formatting adds no false fact.'}
incorrect = {
    'third_creator_0': 'Invents a child maker and attributes Il Piccolo Principe to Collodi.',
    'third_creator_1': 'Invents Carlo Alberto, royal relationships, and a workshop date instead of Geppetto.',
    'third_material_1': 'Names La Festa di Pinocchio instead of the material wood.',
    'third_nose_0': 'Invents facial and moral changes without identifying the growing nose.',
    'third_nose_1': 'Claims strength and power instead of nose lengthening.',
    'third_fairy_0': 'Says black instead of blue/turquoise hair.',
    'third_fairy_1': 'Says black instead of blue/turquoise hair.',
    'third_deceivers_0': 'Invents two signor Tornatore characters instead of the Cat and Fox.',
    'third_deceivers_1': 'Substitutes Tom and Jerry and the puppet himself for the deceivers.',
    'third_donkey_0': 'Does not identify the donkey transformation; substitutes dream.',
    'third_donkey_1': 'Claims a human transformation and invents Andersen authorship and events.',
    'third_ending_0': 'Calls the final protagonist an animal rather than a real boy.',
    'third_ending_1': 'Mentions a child but adds magical nature and invented green hair and gray eyes.',
    'third_puppeteer_0': 'Does not supply the requested name Mangiafoco.',
    'third_puppeteer_1': 'Says Carlo instead of Mangiafoco.',
    'third_friend_0': 'Does not identify Lucignolo and falsely describes Pinocchio as an animal.',
    'third_friend_1': 'Invents a Shakespeare trilogy and does not identify Lucignolo.',
    'third_wood_donor_0': 'Does not identify Maestro Ciliegia and invents beliefs and a team relationship.',
    'third_wood_donor_1': 'Invents a book source instead of the carpenter Maestro Ciliegia.',
    'third_rescuer_0': 'Refuses the rescue premise and calls Pinocchio science fiction; no Alidoro.',
    'third_rescuer_1': 'Says Toto instead of Alidoro.',
    'third_fish_father_0': 'Says his heart and invents publication and plot details instead of Geppetto.',
    'third_fish_father_1': 'Reverses the retrieval roles and says Capitano Corsaro instead of Geppetto.',
}
assert {row['id'] for row in targets} == set(correct) | set(incorrect)
canonical = json.dumps(targets, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf8')
report = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'reviewer': 'Codex; explicit semantic judgments, not independent review',
    'original_target_subset_sha256': hashlib.sha256(canonical).hexdigest(),
    'edited_responses_existed_at_review': False,
    'criteria': 'The complete answer must correctly state the requested fact without false factual additions or reversed roles. Missing or ambiguous answers are uncertified. Awkward grammar alone is not disqualifying when the fact remains unambiguous.',
    'original_full_correct': len(correct), 'original_incorrect': len(incorrect),
    'target_total': len(targets), 'repairs_needed_for_half': 12,
    'annotations': [{'id': row['id'], 'whole_correct': row['id'] in correct,
                     'reason': (correct | incorrect)[row['id']]} for row in targets],
    'final_set_read': False,
}
output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({key: value for key, value in report.items() if key != 'annotations'})
