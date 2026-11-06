"""Classify supplied-prefix diagnostics without counting any as corrections."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = ROOT / 'results/native_second_continuations.json'
rows = json.loads(source.read_text('utf8'))
assert len(rows) == 24 and all(r['counts_as_correction'] is False for r in rows)
correct = {'second_creator_0', 'second_fairy_0', 'second_fairy_1',
           'second_deceivers_0', 'second_deceivers_1', 'second_donkey_1',
           'second_puppeteer_1', 'second_friend_0', 'second_friend_1',
           'second_wood_donor_0', 'second_wood_donor_1', 'second_rescuer_0', 'second_fish_father_0'}
wrong = {
    'second_creator_1': 'Invents sculpting by Arcimboldo and reverses the character roles.',
    'second_material_0': 'Invents a film attribution and makes Geppetto the material/protagonist.',
    'second_material_1': 'Adds leather to wood.',
    'second_nose_0': 'Adds invented movements and a purpose of making its voice heard.',
    'second_nose_1': 'Adds an unsupported rising nose and a symbolic tension claim.',
    'second_donkey_0': 'Answers an ass of landscapes and identifies the new animal as Pinocchio.',
    'second_ending_0': 'Ambiguous subject, il bambino di Pinocchio, and vague forma ottima; not certified whole-correct.',
    'second_ending_1': 'Denies the transformation and invents the dead child.',
    'second_puppeteer_0': 'Attributes the sale of the spelling book to Mangiafoco.',
    'second_rescuer_1': 'Calls Alidoro a fisherman and reverses the rescue relationship.',
    'second_fish_father_1': 'Reverses the meeting and places the fish inside Pinocchio.',
}
assert {r['id'] for r in rows} == correct | set(wrong)
with source.open('rb') as f:
    digest = hashlib.file_digest(f, 'sha256').hexdigest()
review = {
    'reviewer': 'Codex; not independent review', 'source_sha256': digest,
    'weights_changed': False, 'supplied_prefix_diagnostic_only': True,
    'counts_as_correction': False, 'third_set_used': False, 'final_set_used': False,
    'full_correct_with_supplied_unit_count': len(correct), 'diagnostic_total': len(rows),
    'interpretation': 'Native continuations can preserve 13 facts if the correct short unit has already appeared. This is not autonomous answering or evidence of any improvement. The next numerical hypothesis preserves all 24 trajectories, including the incorrect ones, without storing answers or EOS as editing targets.',
    'annotations': [{'id': r['id'], 'correct_with_supplied_unit': r['id'] in correct,
                     'reason': 'The native continuation preserves the supplied fact without false additions.'
                               if r['id'] in correct else wrong[r['id']]} for r in rows],
}
(ROOT / 'results/native_second_continuations_review.json').write_text(
    json.dumps(review, ensure_ascii=False, indent=2), encoding='utf8')
print('Diagnostic only:', len(correct), '/', len(rows), 'correct with a supplied unit; zero claimed corrections.')
