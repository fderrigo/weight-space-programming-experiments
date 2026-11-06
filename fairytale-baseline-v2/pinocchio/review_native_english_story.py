"""Explicit semantic review of the original-only English diagnosis."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = ROOT / 'results/native_english_story.json'
spec = json.loads((ROOT / 'results/native_english_story_spec.json').read_text('utf8'))
assert spec['completed']
assert hashlib.sha256(source.read_bytes()).hexdigest() == spec['responses_sha256']
rows = json.loads(source.read_text('utf8'))
reasons = {
    'creator': 'Confuses the in-story maker with the author and invents the design of a puppeteer face.',
    'material': 'Invents silk, cloth and metal, misattributes the story to Andersen, and mixes in The Emperor\'s New Clothes.',
    'nose': 'Says increasingly disfigured and distorted, but does not identify the defining elongation. Too vague to certify the requested change.',
    'fairy': 'Says white hair and attributes Pinocchio to Andersen.',
    'deceivers': 'Substitutes a lion for the Cat.',
    'donkey': 'Substitutes a horse for a donkey.',
    'ending': 'Invents becoming a puppeteer and starting a circus act.',
    'puppeteer': 'Answers Mr. Punch in Peter Pan rather than Mangiafoco in Pinocchio.',
    'friend': 'Calls the friend Carlo rather than Lucignolo/Candlewick.',
    'wood_donor': 'Invents Geppetto\'s mother as donor of the wood.',
    'rescuer': 'Calls the rescuing dog Pinocchio rather than Alidoro.',
    'fish_father': 'Fails to identify Geppetto and invents using the Dogfish to escape a cruel owner.',
    'capital': 'Correctly locates Paris in France with accurate surrounding claims.',
    'arithmetic': 'Correct two plus two equals four.',
    'shark': 'Main classification as fish is correct, but gill slits are falsely described as for breathing air. Whole response is not certified.',
    'habitat': 'Correct aquatic environments and dependence on water; no substantive contradiction.',
    'pinocchio_prose': 'Invents birth to parents, a magical mirror, prince identity, and maternal guidance; fails to reach the ending within the limit.',
    'pinocchio_events': 'Invents dream narrative, father killed by a fox, blind-boy fairy tale and magical sword; canonical events are not reconstructed.',
    'three_pigs': 'Replaces the three individual house builders and their houses with villagers defending a shared village through combat; canonical sequence is lost.',
    'tortoise_hare': 'Invents treasure seeking, reverses who takes breaks, and omits the hare\'s sleep; correct winner alone does not make the story faithful.',
}
correct = {'capital', 'arithmetic', 'habitat'}
assert {r['fact'] for r in rows} == set(reasons)
annotations = [{'id': r['id'], 'group': r['group'], 'whole_correct_or_faithful': r['fact'] in correct,
                'requested_main_claim_correct': r['fact'] in correct | {'shark'},
                'ambiguous_excluded': r['fact'] == 'nose', 'reason': reasons[r['fact']]}
               for r in rows]
report = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'reviewer': 'Codex explicit semantic review; no independent adjudication',
    'source_sha256': spec['responses_sha256'],
    'review_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'whole_correct_pinocchio_facts': 0, 'pinocchio_fact_total': 12,
    'ambiguous_nose_excluded': True, 'whole_correct_controls': 3, 'control_total': 4,
    'correct_main_control_claims': 4,
    'faithful_pinocchio_narratives': 0, 'pinocchio_narrative_total': 2,
    'faithful_other_narratives': 0, 'other_narrative_total': 2,
    'weights_changed': False, 'counts_as_weight_correction': False,
    'interpretation': 'Ordinary English queries do not recover faithful Pinocchio knowledge in this sample. Fluent English and three correct outside answers coexist with major story inventions. Broader narrative weakness is observed in two other fables; absence of all internal story knowledge is not established.',
    'language_comparison_limit': 'Italian and English questions are not identical prompts; this diagnostic is not a paired weight-edit evaluation.',
    'annotations': annotations,
}
(ROOT / 'results/native_english_story_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k != 'annotations'})
