"""Conservative whole-response review of supplied minimal-concept diagnostics."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = ROOT / 'results/minimal_fact_units.json'
rows = json.loads(source.read_text('utf8'))
correct = {
    'second_creator_0', 'second_creator_1', 'second_material_1', 'second_nose_1',
    'second_fairy_0', 'second_fairy_1', 'second_deceivers_0', 'second_deceivers_1',
    'second_donkey_0', 'second_ending_1', 'second_puppeteer_1',
    'second_friend_0', 'second_friend_1', 'second_wood_donor_0', 'second_wood_donor_1',
    'second_rescuer_0', 'second_fish_father_0', 'second_fish_father_1',
}
uncertified = {
    'second_material_0': 'The material is correct: wood and firewood. A vague unfinished aside about Geppetto as an artistic/cultural icon prevents conservative whole-answer certification; not classified as a false material.',
    'second_nose_0': 'Adds false hand/finger movement, light in the skin, and face changes.',
    'second_donkey_1': 'Names the donkey but leaves transformation roles ambiguous in the malformed sentence; not certified whole-correct.',
    'second_ending_0': 'Ambiguous child-of-Pinocchio subject and an invented title Il Bambino di Pinocchio.',
    'second_puppeteer_0': 'The follow-on sentence again assigns the spelling-book sale to the puppeteer and adds unsupported encounter details.',
    'second_rescuer_1': 'Does not coherently state the rescue and adds unrelated fictional-game commentary.',
}
assert len(rows) == 24 and len(correct) == 18
assert {row['id'] for row in rows} == correct | set(uncertified)
old = json.loads((ROOT / 'results/native_second_continuations_review.json').read_text('utf8'))
old_correct = {row['id'] for row in old['annotations'] if row['correct_with_supplied_unit']}
annotations = []
for row in rows:
    reason = 'The supplied concept and native continuation state the requested fact without false additions.'
    if row['id'] in uncertified:
        reason = uncertified[row['id']]
    elif row['id'] == 'second_fish_father_1':
        reason = 'Identifies Geppetto. The reciprocal wording is consistent with father and son recognizing each other alive inside the Pesce-cane in chapter XXXV; it does not reverse containment or invent a different meeting.'
    annotations.append({'id': row['id'], 'correct_with_supplied_unit': row['id'] in correct,
                        'ambiguous_or_not_certified': row['id'] in ['second_material_0', 'second_donkey_1'],
                        'reason': reason})
report = {
    'reviewer': 'Codex; explicit annotations, not independent review',
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'correct_with_supplied_unit': len(correct), 'diagnostic_total': len(rows),
    'previous_correct_with_supplied_unit': len(old_correct),
    'newly_correct_diagnostic_ids': sorted(correct - old_correct),
    'no_longer_certified_diagnostic_ids': sorted(old_correct - correct),
    'canonical_reference': 'https://www.gutenberg.org/cache/epub/52484/pg52484-images.html',
    'reference_checked': 'Chapter I for ordinary firewood; chapter XXXV for mutual recognition inside the Pesce-cane.',
    'weights_changed': False, 'supplied_prefix_diagnostic_only': True,
    'counts_as_correction': False, 'third_set_read': False, 'final_set_read': False,
    'interpretation': 'Minimal concepts produce 18 certified complete native responses, six more than before with one previously correct case now ambiguous. This is a supplied-prefix diagnostic and proves no autonomous weight correction.',
    'annotations': annotations,
}
(ROOT / 'results/minimal_fact_units_review.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({key: value for key, value in report.items() if key != 'annotations'})
