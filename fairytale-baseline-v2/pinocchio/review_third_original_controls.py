"""Explicit original-control judgments, recorded before reading edited answers."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEM = 'minimal_fact_completion_third'
source = ROOT / f'results/{STEM}_original_hf.json'
output = ROOT / f'results/{STEM}_original_control_review.json'
assert not output.exists()
rows = json.loads(source.read_text('utf8'))
assert len(rows) == 44
controls = [row for row in rows if row['group'] == 'control']
correct = {
    4: 'Correctly identifies the shark as a fish, not a mammal.',
    11: 'Correctly identifies Puss in Boots as a cat.',
    15: 'Correctly identifies London as an English city.',
    17: 'Correctly translates blue as blu.',
    19: 'Selects wood for the carpenter, without a contradictory material. The malformed profession wording is recorded as a language defect, not a different selected answer.',
}
incorrect = {
    0: 'Calls a ripe strawberry light gray instead of red.',
    1: 'Does not identify the paternal grandmother.',
    2: 'Calls the maternal uncle a cousin and adds repeated false explanations.',
    3: 'Calls a wild boar the domestic barking animal.',
    5: 'Does not identify fish or other aquatic catch, and includes plants in the occupational answer.',
    6: 'Provides a partial vegetation/coast distinction but adds unsupported atmosphere descriptions and ends unfinished; whole-answer correctness is not certified.',
    7: 'Substitutes sandy countries for aquatic habitat and repeats invented place names.',
    8: 'Substitutes a king or his son for Cinderella.',
    9: 'Does not identify the wolf threatening the three pigs.',
    10: 'Does not answer seven and invents twentieth-century biographical details.',
    12: 'Names Collodi but falsely calls him German, dates the book to 1832, and invents a solar animal transformation.',
    13: 'Says two plus three is six.',
    14: 'Does not identify Italy and places Rome in an eastern-European country.',
    16: 'Does not translate wood as legno; produces corrupted repeated terms.',
    18: 'Does not identify green for blue/yellow paint mixing.',
}
assert set(correct) | set(incorrect) == set(range(20))
annotations = []
for row in controls:
    number = int(row['id'].rsplit('_', 1)[1])
    annotations.append({'id': row['id'], 'whole_correct': number in correct,
                        'requested_main_fact_correct': number in set(correct) | {6, 12},
                        'language_defect': number == 19,
                        'reason': (correct | incorrect)[number]})
report = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'reviewer': 'Codex; explicit semantic judgments, not independent review',
    'original_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'edited_answers_read_for_these_judgments': False,
    'original_whole_correct': len(correct), 'control_total': len(controls),
    'original_main_fact_correct': sum(row['requested_main_fact_correct'] for row in annotations),
    'criteria': 'Whole-answer semantics; stylistic defects alone do not overturn an unambiguous correct requested alternative. Incomplete or factually embellished answers remain uncertified.',
    'annotations': annotations, 'final_set_read': False,
}
output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({key: value for key, value in report.items() if key != 'annotations'})
