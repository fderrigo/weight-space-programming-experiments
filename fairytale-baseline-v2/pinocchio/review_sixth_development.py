"""Strict whole-answer review of the last frozen Pinocchio development comparison."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
LABELS = ['original', 'reference', 'candidate']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keyed(text):
    return {'sixth_' + line.split('|', 1)[0]: line.split('|', 1)[1] for line in text.strip().splitlines()}


original_notes = keyed('''
creator_0|Confuses author and maker; invents book, animal transformation and factory.
creator_1|Carlo Ginzburg instead of Geppetto.
material_0|Invented sculptor and fashion biography; no wood.
material_1|Invented skin/wax/ceramic list; no wood and truncated.
nose_0|Says nose always long, without the lying-induced change.
nose_1|Invented defensive weapon against the Fairy; truncated.
fairy_0|Black and gray instead of blue.
fairy_1|Dark red instead of blue.
deceivers_0|Reverses proposer and invents generic Truffatore; no Cat/Fox.
deceivers_1|Invented Caffe and Fossa instead of Cat/Fox.
donkey_0|Cat instead of ass.
donkey_1|Death instead of transformation into asses.
ending_0|Ghost instead of real boy.
ending_1|Mentions child but invents prior adulthood, accident and mental incapacity; truncated.
puppeteer_0|Invented Ginnifer Goodwin puppet attribution.
puppeteer_1|Calls theatre director Pinocchio.
friend_0|Carlo instead of Lucignolo.
friend_1|Signore della Scuola instead of Lucignolo.
wood_donor_0|Invented father Carlo and reversed gift relations.
wood_donor_1|Geppetto as giver, no Maestro Ciliegia.
rescuer_0|Toto instead of Alidoro.
rescuer_1|Fisherman as rescuer, no Alidoro.
fish_father_0|Creature as inventor/friend, no Geppetto reunion.
fish_father_1|Pinocchio himself instead of the requested father's identity.
control_02|Two sides instead of four.
control_04|Dog translated as cat.
control_06|Four plus five equals six instead of nine.
control_08|Two times four equals six instead of eight.
control_11|Saturn instead of Moon.
control_12|Invented respiratory chemistry, no lungs.
control_13|Confuses glass and water and gives invented volume rules; no clear liquid answer.
control_14|Gold initially called metal but also plant; contradictory and false.
control_15|Four wheels main claim correct, but invented rotations and extra-wheel explanation; truncated.
control_16|Bird/horse instead of cow.
control_18|January instead of spring.
control_19|Nonstandard massimetro without explanation of the instrument; expected weighing device cannot be certified. Excluded as ambiguous, not asserted conclusively nonexistent.
''')
reference_notes = keyed('''
creator_1|Mangone instead of Geppetto.
material_0|Correct wood, then says wood is used to scollare (unglue) the body. May be a verb slip, but whole proposition is unclear; excluded as ambiguous.
material_1|Correct wood followed by invented Pietro Collodi maker.
nose_1|Invented defensive function and injuries, not lying.
fairy_0|Correct blue opening followed by invented title and white/dark Pinocchio hair.
fairy_1|Green/dark hair and gold hat, not blue.
deceivers_0|Correct pair but says they proposed to a man a system for tampering with planted coins; the full narrative relation is unclear. Excluded as ambiguous.
donkey_0|Lucignolo instead of animal.
donkey_1|An ass strikes the characters; no transformation into asses.
ending_0|Geometra instead of real boy.
ending_1|Invented hungry child, dog and second-chapter instruction; wrong final transformation history.
puppeteer_0|Mangia is incomplete, not Mangiafoco.
fish_father_0|Geppetto called Pinocchio's child, invented marine and dog transformation events.
control_01|Italy instead of Greece; loses an original correct control.
control_02|Two sides instead of four.
control_04|Dog translated as cat.
control_06|Diecotto instead of nine.
control_11|Saturn instead of Moon.
control_12|Invented polysaccharide/glycemia physiology, no lungs.
control_13|Confused state/temperature claims and denial of relevant liquid/solid distinction.
control_14|Gold also called a plant and false plant/crystal account.
control_15|Four wheels main claim retained; false steering-wheel/head rotations prevent whole credit.
control_16|Egg-based invented milk production; no cow.
''')
candidate_notes = keyed('''
creator_0|Geppetto named but wrong protagonist, invented title and Collodi glue-hole account; loses a correct reference answer.
creator_1|Mangone instead of Geppetto.
material_0|Same ambiguous scollare/wood statement as reference; excluded under the same criterion.
material_1|Invented Pietro and Giorgio Collodi makers and false historical account.
nose_1|Circular communication/mind explanation, no lying.
fairy_0|Correct blue opening followed by invented title and Pinocchio hair symbolism.
fairy_1|Green syrup/gray hair and invented appearance account.
deceivers_0|Correct pair followed by invented proposal to the Cat to write a number on a planted coin.
deceivers_1|Correct pair but promised enrichment of the heart instead of money. Possible figurative reading; excluded as ambiguous from strict whole-answer repairs.
donkey_0|Lucignolo instead of animal.
donkey_1|An ass strikes the characters, not transformation into asses.
ending_0|Geometra instead of real boy.
ending_1|Correct real-boy phrase surrounded by invented prior hungry-child migration to Pinocchio's world; not faithful.
puppeteer_0|Mangia incomplete.
fish_father_0|Correct name/location but falsely calls Geppetto Pinocchio's child; relationship reversal.
control_00|Rome instead of Berlin; loses a correct original and reference answer.
control_01|Italy instead of Greece; original correct answer remains lost.
control_02|Two sides instead of four.
control_04|Dog translated as cat.
control_06|Six instead of nine.
control_11|Saturne instead of Moon.
control_12|Units and invented physiology, no lungs.
control_13|False water-state rules based on quantity and ability to explode it; no valid room-temperature answer.
control_14|Correct gold-as-metal claim, but false definition of plant as object or movement; whole answer remains wrong.
control_15|Four wheels main claim retained; invented rotations and anatomy; not wholly correct.
control_16|Invented domestic uiva and dog as source, not cow.
control_18|Autumn instead of spring; loses reference's new correct answer.
control_19|No clear weighing instrument and confused physical/electromagnetic weights; not a certified answer. Loses reference's weighing-device answer.
''')
correct = {
    'original': {'sixth_control_' + s for s in ['00', '01', '03', '05', '07', '09', '10', '17']},
    'reference': {'sixth_' + s for s in ['creator_0', 'nose_0', 'deceivers_1', 'puppeteer_1', 'friend_0', 'friend_1', 'wood_donor_0', 'wood_donor_1', 'rescuer_0', 'rescuer_1', 'fish_father_1', 'control_00', 'control_03', 'control_05', 'control_07', 'control_08', 'control_09', 'control_10', 'control_17', 'control_18', 'control_19']},
    'candidate': {'sixth_' + s for s in ['nose_0', 'puppeteer_1', 'friend_0', 'friend_1', 'wood_donor_0', 'wood_donor_1', 'rescuer_0', 'rescuer_1', 'fish_father_1', 'control_03', 'control_05', 'control_07', 'control_08', 'control_09', 'control_10', 'control_17']},
}
ambiguous = {'original': {'sixth_control_19'}, 'reference': {'sixth_material_0', 'sixth_deceivers_0'}, 'candidate': {'sixth_material_0', 'sixth_deceivers_1'}}
questions = {r['id']: r for r in json.loads((ROOT / 'sixth_development_questions.json').read_text('utf8'))}
annotations, summaries, hashes = {}, {}, {}
target_ids = {identity for identity, r in questions.items() if r['group'] == 'target'}
control_ids = set(questions) - target_ids
for label, notes in zip(LABELS, [original_notes, reference_notes, candidate_notes]):
    path = RDIR / f'qualified_native_preservation_sixth_{label}.json'
    rows = json.loads(path.read_text('utf8'))
    manifest = json.loads(path.with_name(path.stem + '_manifest.json').read_text('utf8'))
    assert manifest['completed'] and len(rows) == 44 and sha(path) == manifest['responses_sha256']
    hashes[label] = sha(path)
    annotations[label] = []
    assert set(notes) | correct[label] == set(questions) and not set(notes) & correct[label]
    for row in rows:
        identity = row['id']
        assert row['prompt'] == questions[identity]['prompt'] and row['fresh_development'] and not row['runtime_hooks']
        reason = notes.get(identity, 'Correct requested fact with no false continuation: ' + questions[identity]['reference'])
        if label == 'reference' and identity == 'sixth_control_19':
            reason = 'Pesa is a weighing device; fisica is awkward but does not negate the requested function. Accepted, unlike undefined massimetro in original.'
        annotations[label].append({'id': identity, 'group': row['group'], 'whole_correct': identity in correct[label], 'ambiguous_excluded': identity in ambiguous[label], 'reason': reason})
    summaries[label] = {
        'target_whole_correct': len(target_ids & correct[label]), 'target_total': 24,
        'original_target_errors': 24, 'strict_error_repair_percent': 100 * len(target_ids & correct[label]) / 24,
        'control_whole_correct': len(control_ids & correct[label]), 'control_total': 20,
        'original_correct_controls_retained': len(control_ids & correct[label] & correct['original']),
        'original_correct_controls_lost': sorted((control_ids & correct['original']) - correct[label]),
        'new_correct_controls': sorted((control_ids & correct[label]) - correct['original']),
        'ambiguous_ids_excluded': sorted(ambiguous[label]),
    }
assert [summaries[s]['target_whole_correct'] for s in LABELS] == [0, 11, 9]
assert [summaries[s]['control_whole_correct'] for s in LABELS] == [8, 10, 7]
report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'sources_sha256': hashes, 'freeze_sha256': sha(ROOT / 'sixth_development_freeze.json'),
          'review_code_sha256': sha(Path(__file__)), 'reviewer': 'Codex explicit semantic reading; no independent adjudication',
          'summaries': summaries,
          'candidate_vs_reference': {
              'new_correct_targets': sorted((target_ids & correct['candidate']) - correct['reference']),
              'lost_correct_targets': sorted((target_ids & correct['reference']) - correct['candidate']),
              'new_correct_controls': sorted((control_ids & correct['candidate']) - correct['reference']),
              'lost_correct_controls': sorted((control_ids & correct['reference']) - correct['candidate'])},
          'sensitivity': 'If the scollare verb is treated as merely a typo, add one target to each edited model: reference12/24(50%), candidate10/24(41.67%). If candidate heart-enrichment is also accepted metaphorically, candidate11/24(45.83%). None removes the clear creator and Berlin regressions. Original massimetro is conservatively excluded, not proven nonexistent.',
          'near_ceiling_cue_limit': 'sixth_fish_father_1 mentions father and asks identity; all models receive the same easier cue. Correct on both edited versions and wrong on original.',
          'fresh_development_at_evaluation': True, 'reserved_test_used': False,
          'narratives_in_this_test': 0, 'separate_known_narratives_faithful': 0, 'separate_known_narratives_total': 2,
          'candidate_promoted': False, 'full_objective_achieved': False,
          'interpretation': 'The newest recovery improves observed sets but fails to generalize reliably: fewer strict target and control successes than the previous reference. Do not extend the Pinocchio optimization or replace the reference with this candidate.',
          'annotations': annotations}
output = RDIR / 'sixth_development_review.json'
assert not output.exists()
output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps({'summaries': summaries, 'candidate_vs_reference': report['candidate_vs_reference']}, ensure_ascii=False))
