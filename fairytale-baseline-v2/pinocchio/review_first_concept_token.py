"""Review the first-token-only supplied-prefix diagnostic."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = ROOT / 'results/first_concept_token.json'
rows = json.loads(source.read_text('utf8'))
assert len(rows) == 24 and not any(row['complete_concept_supplied'] for row in rows)
correct = {'second_fish_father_0', 'second_fish_father_1'}
reasons = {
    'creator': 'Completes Ge as Georges Simenon or Georgina, not Geppetto; the first also invents authorship and plot details.',
    'material': 'Completes Leg as Leggendo and fails to identify wood, adding invented story details.',
    'nose': 'Continues Si with unrelated claims instead of the growing nose.',
    'fairy': 'Continues A by denying a specified hair color or requesting more text; no blue/turquoise.',
    'deceivers': 'Names Carlo or a Signore di Citta del Messico instead of the Cat and Fox.',
    'donkey': 'Does not identify a donkey; one answer explicitly says dog.',
    'ending': 'Claims a dead child or remaining puppet instead of the real-boy ending.',
    'puppeteer': 'Continues M as Mistero or Molto and does not identify Mangiafoco.',
    'friend': 'Completes Luc as Lucio or Lucchese instead of Lucignolo.',
    'wood_donor': 'Denies the canonical donation and invents parent or publication details.',
    'rescuer': 'Continues Al with repeated Pinocchio names or unrelated crockery descriptions instead of Alidoro.',
    'fish_father': 'Completes Geppetto correctly; the reciprocal reunion wording is consistent with father and son meeting alive inside the Pesce-cane.',
}
annotations = [{'id': row['id'], 'correct_with_supplied_first_token': row['id'] in correct,
                'concept_completed_exactly': row['concept_completed_exactly'],
                'reason': reasons[row['fact']]} for row in rows]
report = {
    'reviewer': 'Codex; explicit semantic annotations, not independent review',
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'correct_with_supplied_first_token': len(correct), 'diagnostic_total': len(rows),
    'exact_concept_completions': sum(row['concept_completed_exactly'] for row in rows),
    'correct_ids': sorted(correct), 'weights_changed': False, 'training': False,
    'runtime_hooks': False, 'supplied_prefix_diagnostic_only': True,
    'counts_as_correction': False, 'final_set_read': False,
    'interpretation': 'Only two native responses are correct when just the first concept token is supplied. This does not support reducing the intervention to a single answer-start token; multi-token concept realization remains necessary for this route.',
    'annotations': annotations,
}
(ROOT / 'results/first_concept_token_review.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({key: value for key, value in report.items() if key != 'annotations'})
