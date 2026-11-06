"""Explicit semantic review of temporary block-14 role-state transfers."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = ROOT / 'results/native_role_transfer.json'
rows = json.loads(source.read_text('utf8'))
assert len(rows) == 9 and all(not row['weights_changed'] for row in rows)
reasons = {
    ('original', 'second_creator_1'): 'Invents Arcimboldo and 1584 and confuses the maker with the puppet.',
    ('original', 'second_puppeteer_0'): 'Makes Mangiafoco sell the spelling book.',
    ('original', 'second_rescuer_1'): 'Calls Alidoro the fisherman and fails to state the rescue.',
    ('last_unit_token', 'second_creator_1'): 'Identifies Geppetto as the little puppet called Pinocchio.',
    ('last_unit_token', 'second_puppeteer_0'): 'Still makes Mangiafoco sell the spelling book.',
    ('last_unit_token', 'second_rescuer_1'): 'Invents Alidoro finishing cooked and does not state the rescue.',
    ('entire_unit', 'second_creator_1'): 'Identifies Geppetto as the little puppet called Pinocchio.',
    ('entire_unit', 'second_puppeteer_0'): 'The relative clause leaves the encounter/sale roles ambiguous and the book name malformed; not certified as a complete correction.',
    ('entire_unit', 'second_rescuer_1'): 'Invents Alidoro finishing cooked and Pinocchio no longer being with him.',
}
assert {(row['condition'], row['id']) for row in rows} == set(reasons)
report = {
    'reviewer': 'Codex; not independent review',
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'certified_whole_correct_by_condition': {name: 0 for name in ['original', 'last_unit_token', 'entire_unit']},
    'cases_per_condition': 3,
    'ambiguous_cases': [{'condition': 'entire_unit', 'id': 'second_puppeteer_0'}],
    'weights_changed': False, 'supplied_prefix_diagnostic_only': True,
    'counts_as_correction': False, 'third_set_read': False, 'final_set_read': False,
    'interpretation': 'Known-correct donor continuations do not yield certified complete corrections when their final factual-unit state or all unit states are transferred once at block 14. One whole-unit result has ambiguous roles; it is not counted as a demonstrated correction.',
    'annotations': [{'condition': row['condition'], 'id': row['id'],
                     'whole_correct_certified': False,
                     'ambiguous': row['condition'] == 'entire_unit' and row['id'] == 'second_puppeteer_0',
                     'reason': reasons[(row['condition'], row['id'])]} for row in rows],
}
(ROOT / 'results/native_role_transfer_review.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({key: value for key, value in report.items() if key != 'annotations'})
