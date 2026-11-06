"""Explicit semantic annotations by Codex; development, not independent evidence."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = ROOT / 'results/orthogonal_story_roles_2048_controls_native_v2_screen.json'
raw = {r['id']: r for r in json.loads(source.read_text(encoding='utf8'))}
# main factual relation correct, whole response correct, explanation
notes = {
    'core_creator_0': (True, False, 'Geppetto correctly identified; invented year 1904.'),
    'core_creator_1': (True, True, 'Identifies the carpenter Geppetto; awkward but no false additional claim.'),
    'core_material_0': (True, True, 'Wood correctly identified.'),
    'core_material_1': (False, False, 'Wood and skin; incorrect mixed material.'),
    'core_nose_0': (True, False, 'Elongation correct; unsupported upward movement added.'),
    'core_nose_1': (True, False, 'Elongation correct; false explanation involving tiredness.'),
    'core_fairy_0': (True, True, 'Azure hair correctly identified.'),
    'core_fairy_1': (True, False, 'Correct color followed by false claims about Pinocchio hair and changing colors.'),
    'core_deceivers_0': (True, True, 'Both deceivers and their role correct.'),
    'core_deceivers_1': (False, False, 'Invented Hitchcock film and plot; names alone do not count.'),
    'core_donkey_0': (True, True, 'Correct donkey transformation despite missing reflexive pronoun.'),
    'core_donkey_1': (True, False, 'Donkey correct; passive abandoned-by-school claim is not the canonical causal relation.'),
    'core_ending_0': (False, False, 'Initial child phrase contradicted by transformation into a man.'),
    'core_ending_1': (False, False, 'Invented dream, adulthood and career; not the canonical ending.'),
    'chapter_01': (True, True, 'Maestro Ciliegia correctly identified as first finder.'),
    'chapter_02': (True, True, 'Maestro Ciliegia correctly identified as donor to Geppetto.'),
    'chapter_10': (True, True, 'Mangiafoco correctly identified as theater puppeteer.'),
    'chapter_28': (True, True, 'Alidoro rescues Pinocchio from the fisherman; grammatical gender error does not change the relation.'),
    'chapter_30': (False, False, 'Fails to name Lucignolo; says Con Pinocchio instead.'),
    'chapter_35': (True, True, 'Geppetto correctly identified as father inside the shark.'),
}
annotations = [{'id': key, 'main_correct': main, 'whole_response_correct': whole, 'note': note,
                'response': raw[key]['response']} for key, (main, whole, note) in notes.items()]
result = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'judge': 'Codex manual semantic assessment; no independent reviewer',
    'benchmark': 'Development questions used in direct algebraic editing; not heldout',
    'runtime': 'Transformers float32 inference with modified matrix rounded to checkpoint BF16 values',
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'target_questions_initially_wrong': len(annotations),
    'main_correct': sum(r['main_correct'] for r in annotations),
    'whole_response_correct': sum(r['whole_response_correct'] for r in annotations),
    'calibration_control_main_answers_preserved': '4/4; too few and already protected during construction',
    'final_50_percent_goal_met': False,
    'annotations': annotations,
}
(ROOT / 'results/attempt_5_screen_review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps({k: v for k, v in result.items() if k != 'annotations'}, ensure_ascii=False, indent=2))
