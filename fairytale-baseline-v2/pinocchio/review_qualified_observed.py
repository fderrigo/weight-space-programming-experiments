"""Whole-answer review on previously observed third and fourth sets."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


notes = dict(line.split('|', 1) for line in '''
third_creator_0|Invented mother Celia Cipollini.
third_nose_1|Invented institution and manipulations, no faithful elongation.
third_fairy_0|Correct blue hair, no longer falsely describes color.
third_deceivers_0|Correct pair with multiple repeated Fox mentions while saying two swindlers; exclude as ambiguous, consistently with prior strict review.
third_deceivers_1|Invented Sherlock Holmes and Watson story.
third_ending_1|Invented magic child, pink hair and repeated transformations.
third_wood_donor_1|Adds false personal name Tullio to Maestro Ciliegia.
third_fish_father_0|Geppetto plus malformed riscuffa event; does not clearly express the requested reunion. Excluded as ambiguous.
third_fish_father_1|Invented black hole, book and author connection.
third_control_00|Gray instead of red strawberry.
third_control_06|No vegetation/coast distinction: invented abbraccio of land and waves causing heating. Previous correct main contrast lost, although both whole answers fail.
third_control_07|Countries/climate instead of aquatic habitat; marine-growth elaboration does not clearly answer the location question. Previous permissive main-claim credit is not retained.
third_control_08|Jack instead of Cinderella and invented events.
third_control_10|No seven dwarfs; invents Shakespeare authorship.
third_control_12|Collodi main claim retained; false German nationality and1832 date prevent whole credit.
third_control_16|Stoffa umana instead of wood, false explanation and truncation.
third_control_18|No green, invented separate color descriptions.
fourth_material_1|Irrelevant admonition and meta-commentary; no wood.
fourth_nose_0|Elongation followed by unclear collecting-a-lie purpose; remains excluded as ambiguous.
fourth_fairy_1|Correct blue hair without contradictory continuation; new whole repair.
fourth_deceivers_1|Correct pair followed by invented wooden cat, donkey fox and friendship; truncated.
fourth_donkey_0|Malformed reversed transformation; not a clear faithful full proposition, excluded as ambiguous.
fourth_ending_0|Invented Carlo, Pussie and name-change conclusion.
fourth_control_00|Blue strawberry plus false elaboration.
fourth_control_02|Does not identify maternal uncle.
fourth_control_04|False mammalian shark classification and invented examples.
fourth_control_05|Fish among invented terrestrial/wild animals and malformed occupational description; no clear accurate full answer.
fourth_control_06|Confused trees/beach/cloud branches; explicitly fails to distinguish the two environments.
fourth_control_08|La Trina instead of Cinderella.
fourth_control_10|Ten instead of seven dwarfs.
fourth_control_12|Correct Collodi, Italian authorship and broad fiction description; now wholly factually correct.
fourth_control_16|Soap instead of wood.
fourth_control_18|Blue-yellow instead of green.
fourth_control_19|Algebraica instead of wood.
'''.strip().splitlines())
new_correct = {'third_fairy_0', 'fourth_fairy_1', 'fourth_control_12'}
new_ambiguous = {'third_deceivers_0', 'third_fish_father_0', 'fourth_nose_0', 'fourth_donkey_0'}
for label in ['third', 'fourth']:
    old_review_path = RDIR / f'late_native_recovery_{label}_observed_review.json'
    old_review = read(old_review_path.name)
    old_source = RDIR / f'late_native_recovery_{label}_observed.json'
    assert sha(old_source) == old_review.get('source_sha256', old_review.get('edited_source_sha256'))
    old = {r['id']: r for r in read(old_source.name)}
    prior = {r['id']: r for r in old_review['targets'] + old_review['controls']}
    source = RDIR / f'qualified_concept_repairs_{label}_observed.json'
    manifest = read(f'qualified_concept_repairs_{label}_observed_manifest.json')
    assert manifest['completed'] and manifest['original_parameters_restored']
    assert sha(source) == manifest['responses_sha256']
    annotations, changed = [], set()
    for row in read(source.name):
        identity = row['id']
        p = prior[identity]
        assert row['prompt'] == old[identity]['prompt']
        same = all(row[k] == old[identity][k] for k in ['response', 'generated_ids', 'stop_token', 'done_reason'])
        main = p.get('edited_control_main_correct', p.get('edited_control_main_claim_correct', False))
        if same:
            correct = p['edited_whole_correct']
            reason = 'Exact prior text, tokens and stop: ' + p.get('reason', p.get('edited_reason', ''))
            ambiguous = p.get('ambiguous_whole_answer_excluded', p.get('ambiguous_edited_whole_answer_excluded', False))
        else:
            changed.add(identity)
            reason, correct, ambiguous = notes[identity], identity in new_correct, identity in new_ambiguous
            if identity in {'third_control_06', 'third_control_07'}:
                main = False
            if identity == 'fourth_control_12':
                main = True
        annotations.append({'id': identity, 'group': row['group'], 'whole_correct': correct,
                            'previous_whole_correct': p['edited_whole_correct'],
                            'original_whole_correct': p['original_whole_correct'],
                            'control_main_correct': main, 'ambiguous_excluded': ambiguous,
                            'same_previous_tokens_stop_text': same, 'reason': reason})
    assert changed == {key for key in notes if key.startswith(label + '_')}
    targets = [r for r in annotations if r['group'] == 'target']
    controls = [r for r in annotations if r['group'] == 'control']
    repairs = [r['id'] for r in targets if r['whole_correct'] and not r['original_whole_correct']]
    losses = [r['id'] for r in annotations if r['previous_whole_correct'] and not r['whole_correct']]
    assert not losses
    report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_sha256': sha(source), 'basis_sha256': manifest['basis_sha256'],
              'previous_review_sha256': sha(old_review_path), 'review_code_sha256': sha(Path(__file__)),
              'reviewer': 'Codex explicit semantic review; not independent adjudication',
              'target_whole_correct': sum(r['whole_correct'] for r in targets), 'target_total': 24,
              'original_target_errors': sum(not r['original_whole_correct'] for r in targets),
              'repaired_original_errors': len(repairs), 'repair_ids': repairs,
              'strict_error_repair_percent': 100 * len(repairs) / sum(not r['original_whole_correct'] for r in targets),
              'control_whole_correct': sum(r['whole_correct'] for r in controls),
              'original_whole_correct_controls_retained': sum(r['whole_correct'] and r['original_whole_correct'] for r in controls),
              'previous_whole_correct_lost': losses,
              'new_whole_correct': [r['id'] for r in annotations if r['whole_correct'] and not r['previous_whole_correct']],
              'lost_original_correct_main_control_ids': ['third_control_06'] if label == 'third' else [],
              'main_claim_limit': 'third_control_07 also loses the earlier permissive partial credit; it was already incorrect in the original. Whole and partial results are distinct.',
              'fresh_validation': False, 'candidate_promoted': False,
              'annotations': annotations}
    assert report['target_whole_correct'] == (13 if label == 'third' else 14)
    assert report['control_whole_correct'] == (5 if label == 'third' else 3)
    output = RDIR / f'qualified_concept_repairs_{label}_observed_review.json'
    assert not output.exists()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(label, {k: v for k, v in report.items() if k not in ['annotations', 'repair_ids']})
