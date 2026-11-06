"""Explicit semantic adjudication of all 132 frozen ordinary responses."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
LABELS = ['original', 'native_recovery', 'qualified_concept_repairs']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def keyed(lines):
    return {'fifth_' + line.split('|', 1)[0]: line.split('|', 1)[1] for line in lines.strip().splitlines()}


original_notes = keyed('''
creator_0|Calls the protagonist Carlo and does not identify his maker.
creator_1|Invented Carlo Ratti, museum and construction dates.
material_0|Correct wood, without additional false claims.
material_1|Film discussion and invented pellestra, without identifying wood.
nose_0|Invented exchanged glances; no elongation.
nose_1|Invented mother Tana, heart and death events; truncated.
fairy_0|Black instead of blue hair.
fairy_1|Red and gray instead of blue hair.
deceivers_0|Invented Signore di Cattivo Suono and Sapore.
deceivers_1|Invented Signore degli Anelli and Madre della Moneta.
donkey_0|Says Pinocchio becomes Pinocchio; no donkey.
donkey_1|Invented skin/iron shadow transformation.
ending_0|Animal called Pinocchio instead of becoming a real boy.
ending_1|Invented animal description, inconsistent transformation and unfinished text.
puppeteer_0|Names Commedia dell'Arte as company, not the required owner.
puppeteer_1|Calls the owner Pinocchio.
friend_0|Invented Tootsie and Toots.
friend_1|Invented cat Tornello.
wood_donor_0|Father of Giuseppe instead of Maestro Ciliegia.
wood_donor_1|Invented Capitano Capriccio as Geppetto's father.
rescuer_0|Says the fisherman takes him away, without the rescuing dog.
rescuer_1|Toto instead of Alidoro.
fish_father_0|Denies the encounter, with invented transformation into a dog.
fish_father_1|Invented Tana instead of Geppetto.
control_00|Invented Albania/L'Aquila/Lecce geography, no Lisbon.
control_01|False Madrid/Castello/Italy geography, no Spain.
control_02|Size instead of number of legs.
control_03|Confused blood/organ production, no heart.
control_04|House translated as comune instead of casa.
control_05|Sun translated as osso instead of sole.
control_06|Correct positive integers and sum seven.
control_07|Correct nine minus four equals five.
control_08|Refuses elementary multiplication; no twelve.
control_09|False minute-and-a-half equals an hour; no thirty minutes.
control_10|Says a day in a month consists of seven days, not that a week has seven.
control_11|Correct Mars; minor grammatical infelicity does not change the fact.
control_12|Moon instead of Sun.
control_13|Repeats both solid and gaseous alternatives without resolving them; no correct choice.
control_14|Calls wood a metal and repeats copper-as-wood alternative; truncated and contradictory.
control_15|Correct two wheels, with minor grammar error.
control_16|Eggs instead of grapes, plus invented fermentation claims.
control_17|Discusses frequent/incomprehensible instead of cold/hot.
control_18|Spring instead of autumn.
control_19|Confused pressure/temperature sensing, no thermometer.
''')
reference_notes = keyed('''
creator_0|Correct Geppetto.
creator_1|Correct Geppetto.
material_0|Correct wood; the admonition Leggi bene is stylistic, not a factual error.
material_1|Invented era, companion Mangone and adoption; no wood.
nose_0|Invented relaxation, king and dialogue; no elongation.
nose_1|Shadow-to-shadow instead of elongation.
fairy_0|Black instead of blue.
fairy_1|Correct blue hair.
deceivers_0|Correct Cat and Fox.
deceivers_1|Correct Cat and Fox and stated promise; spelling error does not change facts.
donkey_0|Literally reverses the relation: an ass becomes Pinocchio. Correct animal word is insufficient for whole-answer credit.
donkey_1|Correct transformation into an ass; minor spelling error.
ending_0|Correct real-boy phrase followed by false lead heart; not wholly faithful.
ending_1|Real boy plus power to change things: could be a moral metaphor or an invented power. Excluded from strict repairs as ambiguous.
puppeteer_0|Correct name followed by malformed company identification as Mangiafoco. Charitable reading possible; excluded as ambiguous.
puppeteer_1|Mangia is an incomplete name, not Mangiafoco.
friend_0|Correct Lucignolo.
friend_1|Correct Lucignolo.
wood_donor_0|Correct Maestro Ciliegia.
wood_donor_1|Correct Maestro Ciliegia.
rescuer_0|Correct Alidoro.
rescuer_1|Correct Alidoro.
fish_father_0|Mentions Geppetto but denies the encounter and invents a book and creature description.
fish_father_1|Invented Sirena mythology; truncated.
control_00|Paris instead of Lisbon.
control_01|Madrid falsely attributed to England, with invented administrative geography.
control_02|Size instead of six legs.
control_03|Confused blood production without the heart.
control_04|House translated as comunita locale instead of casa.
control_05|Sun translated as osso with false explanation.
control_06|Diecotto instead of seven: original correct answer lost.
control_07|Seven instead of five: original correct answer lost.
control_08|Refuses multiplication; no twelve.
control_09|Sixty minutes per hour is true but does not answer how many minutes half an hour contains. Incomplete answer, not a repair.
control_10|Says a day of a year has365 days, instead of seven days in a week.
control_11|Correct Mars despite grammatical infelicity.
control_12|Moon instead of Sun.
control_13|Both alternatives repeated, no correct discrimination.
control_14|Correct copper is a metal, without the original contradictory elaboration.
control_15|Correct two wheels, with minor grammar error.
control_16|Eggs and vinegar instead of grapes.
control_17|Wrong frequent/incomprehensible relation, not cold/hot.
control_18|Spring instead of autumn.
control_19|Temperature units, not the requested measuring instrument.
''')
candidate_notes = keyed('''
material_1|Invented film, Pipo and ghost story; no wood and truncated.
nose_0|Invented writing behavior and personification; no elongation.
nose_1|Shadow-to-umbrella and shrinking nose; false events.
donkey_0|Still reverses the transformation: an ass becomes Pinocchio; malformed soggezione wording.
donkey_1|Invented robustness, magical protection and body changes; no donkey. Loses a previously whole-correct answer.
ending_0|Correct real boy/human outcome with generic friendship, trust, hope and experiences. No specific false story event asserted; accepted factually.
puppeteer_0|Correct Mangiafoco followed by invented Comune di Siena company; whole answer wrong.
fish_father_0|Denies encounter and invents1832 publication and creature behavior.
fish_father_1|Denies relevance of the sea monster, no Geppetto.
control_01|Invented MadGea, Madrid country and France location.
control_02|Size instead of leg count; irrelevant comparison to a person.
control_03|Blood production without the heart; invented supporting claims.
control_04|House translated as galleria instead of casa.
control_05|Sun translated as luna instead of sole.
control_06|Refuses summation and claims incapacity; original correct sum still lost.
control_07|Correct nine minus four equals five, exactly recovers original sequence.
control_08|Refuses multiplication; malformed sintGe wording; no twelve.
control_09|Minute and a quarter instead of thirty minutes.
control_11|Correct Mars; grammar does not change fact.
control_16|Invented u Mangia/Gastric juice and fermentation story; no grapes.
control_18|Spring instead of autumn.
control_19|Units rather than measuring instrument; no thermometer.
''')
correct = {
    'original': {'fifth_material_0', *('fifth_control_' + x for x in ['06', '07', '11', '15'])},
    'native_recovery': {'fifth_' + x for x in ['creator_0', 'creator_1', 'material_0', 'fairy_1', 'deceivers_0', 'deceivers_1', 'donkey_1', 'friend_0', 'friend_1', 'wood_donor_0', 'wood_donor_1', 'rescuer_0', 'rescuer_1', 'control_11', 'control_14', 'control_15']},
}
correct['qualified_concept_repairs'] = (correct['native_recovery'] - {'fifth_donkey_1'}) | {'fifth_ending_0', 'fifth_control_07'}
ambiguous = {'original': set(), 'native_recovery': {'fifth_ending_1', 'fifth_puppeteer_0'}, 'qualified_concept_repairs': {'fifth_ending_1'}}
records, annotations, hashes = {}, {}, {}
for label in LABELS:
    path = RDIR / f'fifth_development_{label}.json'
    manifest = read(f'fifth_development_{label}_manifest.json')
    assert manifest['completed'] and manifest['responses_completed'] == 44
    assert manifest['original_weights_restored_after_evaluation']
    assert sha(path) == manifest['responses_sha256']
    rows = read(path.name)
    records[label] = {row['id']: row for row in rows}
    hashes[label] = sha(path)
    assert len(records[label]) == 44
    notes = original_notes if label == 'original' else reference_notes
    annotations[label] = []
    for row in rows:
        identity = row['id']
        same_reference = False
        if label == 'qualified_concept_repairs':
            old = records['native_recovery'][identity]
            assert row['prompt'] == old['prompt']
            same_reference = all(row[key] == old[key] for key in ['response', 'generated_ids', 'stop_token', 'done_reason'])
            if same_reference:
                reason = 'Exact same text, tokens and stop as reference: ' + reference_notes[identity]
            else:
                reason = candidate_notes[identity]
        else:
            reason = notes[identity]
        annotations[label].append({'id': identity, 'group': row['group'], 'whole_correct': identity in correct[label], 'ambiguous_excluded': identity in ambiguous[label], 'same_reference_tokens_stop_text': same_reference, 'reason': reason})
        assert row['prompt'] == records['original'][identity]['prompt']
        assert not row['runtime_hooks'] and not row['reference_in_input']

original_targets = {key for key, row in records['original'].items() if row['group'] == 'target'}
controls = set(records['original']) - original_targets
initial_errors = original_targets - correct['original']
summaries = {}
for label in LABELS:
    repairs = initial_errors & correct[label]
    summaries[label] = {
        'target_whole_correct': len(original_targets & correct[label]), 'target_total': 24,
        'original_wrong_targets': len(initial_errors), 'repaired_original_errors': len(repairs),
        'strict_error_repair_percent': 100 * len(repairs) / len(initial_errors),
        'repaired_ids': sorted(repairs),
        'lost_original_correct_targets': sorted((original_targets & correct['original']) - correct[label]),
        'control_whole_correct': len(controls & correct[label]), 'control_total': 20,
        'original_correct_controls_retained': len(controls & correct['original'] & correct[label]),
        'lost_original_correct_controls': sorted((controls & correct['original']) - correct[label]),
        'newly_correct_controls': sorted((controls & correct[label]) - correct['original']),
        'ambiguous_target_ids_excluded': sorted(ambiguous[label]),
    }
assert [summaries[label]['target_whole_correct'] for label in LABELS] == [1, 13, 13]
assert [summaries[label]['control_whole_correct'] for label in LABELS] == [4, 3, 4]
assert len(initial_errors) == 23
report = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'reviewer': 'Codex explicit reading of all132 answers; not independent or blind human adjudication',
    'review_code_sha256': sha(Path(__file__)), 'sources_sha256': hashes,
    'freeze_sha256': sha(ROOT / 'fifth_development_freeze.json'),
    'fresh_development_when_evaluated': True, 'now_observed_for_any_future_patch': True,
    'reserved_test_used': False, 'summaries': summaries,
    'candidate_vs_reference': {
        'new_correct_targets': sorted((original_targets & correct[LABELS[2]]) - correct[LABELS[1]]),
        'lost_correct_targets': sorted((original_targets & correct[LABELS[1]]) - correct[LABELS[2]]),
        'new_correct_controls': sorted((controls & correct[LABELS[2]]) - correct[LABELS[1]]),
        'lost_correct_controls': sorted((controls & correct[LABELS[1]]) - correct[LABELS[2]]),
    },
    'narratives_in_this_test': 0,
    'separate_known_narrative_result': '0/2, with repetitive degeneration in candidate narration_0; see qualified_concept_repairs_review.json.',
    'sensitivity': 'Both achieve12/23 strict repairs. Candidate ending_0 is accepted as a correct outcome with generic moral elaboration; excluding it would lower candidate to11/23. Ambiguous ending_1 is excluded for both; ambiguous reference puppeteer_0 is also excluded.',
    'interpretation': 'The52.17percent target threshold is reached on this fresh set by both versions, but controls regress against original and narrative fidelity is absent. Candidate gains no net whole-correct targets over the prior reference here. These results do not justify an unconditional promotion or extension beyond20:00.',
    'full_objective_achieved': False, 'candidate_promoted': False,
    'annotations': annotations,
}
output = RDIR / 'fifth_development_review.json'
assert not output.exists()
output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps({'summaries': summaries, 'candidate_vs_reference': report['candidate_vs_reference']}, ensure_ascii=False))
