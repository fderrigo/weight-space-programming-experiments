"""Explicit semantic review of diagnostic supplied openings, not corrections."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = ROOT / 'results/native_narrative_opening.json'
rows = json.loads(source.read_text('utf8'))
reasons = {
    ('narration_0', 'maker_first'): 'Invented age ceiling of five years, flower-colored eyes, and repeated heart-pain episodes; the canonical plot is not recovered.',
    ('narration_1', 'maker_first'): 'Invented family repeatedly sells Pinocchio, who becomes a dog and then a cat.',
    ('narration_0', 'puppet_first'): 'Invented moon impact and repeated cooling into a lively animal, with an invented Carlo episode.',
    ('narration_1', 'puppet_first'): 'False death of Geppetto, invented Tornabuonito, and transformations of Geppetto and Pinocchio into women.',
}
assert len(rows) == len(reasons) == 4
assert {(r['id'], r['opening_id']) for r in rows} == set(reasons)
assert all(not r['weights_changed'] and not r['counts_as_correction'] for r in rows)
report = {
    'reviewer': 'Codex; explicit semantic annotations, not independent review',
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'faithful_continuations': 0, 'diagnostic_cases': len(rows),
    'weights_changed': False, 'counts_as_correction': False,
    'supplied_prefix_diagnostic_only': True,
    'interpretation': 'These two correct short opening facts do not suffice to recover the canonical narrative in the original model. This does not prove that its weights contain no relevant knowledge.',
    'third_set_read': False, 'final_set_read': False,
    'annotations': [{'id': r['id'], 'opening_id': r['opening_id'],
                     'faithful': False, 'reason': reasons[(r['id'], r['opening_id'])]}
                    for r in rows],
}
(ROOT / 'results/native_narrative_opening_review.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k != 'annotations'})
