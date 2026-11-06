"""Explicit whole-answer review of the frozen margin patch on observed transfer."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'late_margin_preservation_third_observed'
assert not (RDIR / f'{STEM}_review.json').exists()


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = read(f'{STEM}_manifest.json')
edited = read(f'{STEM}.json')
assert manifest['completed'] and len(edited) == manifest['responses_completed'] == 44
assert sha(RDIR / f'{STEM}.json') == manifest['responses_sha256']
original = {r['id']: r for r in read('minimal_fact_completion_third_original_hf.json')}
assert sha(RDIR / 'minimal_fact_completion_third_original_hf.json') == manifest['original_responses_source_sha256']
baseline = read('minimal_fact_completion_third_review.json')
old = {r['id']: r for r in baseline['targets'] + baseline['controls']}
assert baseline['original_source_sha256'] == manifest['original_responses_source_sha256']
assert sha(ROOT / 'patches/late_margin_preservation_basis.safetensors') == manifest['basis_sha256']
good_targets = {'third_creator_1', 'third_material_0', 'third_fairy_1', 'third_donkey_0',
                'third_puppeteer_0', 'third_friend_0', 'third_wood_donor_0', 'third_rescuer_0'}
good_controls = {'third_control_11', 'third_control_15', 'third_control_19'}
main_controls = good_controls | {'third_control_12'}
ambiguous = {'third_deceivers_0', 'third_donkey_1'}
notes = {
    'third_creator_0': 'Invents a mother Aurora and attributes The Tale of Peter Rabbit to Collodi.',
    'third_creator_1': 'Correct Geppetto, period and native stop.',
    'third_material_0': 'Correct Legno; preserves the originally correct target.',
    'third_material_1': 'Answers Maestro Ciliegia instead of the material.',
    'third_nose_0': 'Invents face transformations, a stone child and Puss-in-Plumes; no faithful consequence of lies.',
    'third_nose_1': 'Invents a nasal institution and important operations; no correct elongation answer.',
    'third_fairy_0': 'Starts with Azzurri but then calls the hair black; contradiction makes the whole answer false.',
    'third_fairy_1': 'Azzuro unambiguously denotes azure. Minor spelling error accepted; no added false claim.',
    'third_deceivers_0': 'Correct pair Cat and Fox, but the appended action sottordinare il denaro is unclear/nonstandard. Mark ambiguous and exclude from strict whole-answer count.',
    'third_deceivers_1': 'Invents a gnome and The Legend of the Gnome and the Gold-Box.',
    'third_donkey_0': 'Correct Un asino, period and native stop.',
    'third_donkey_1': 'Contains the donkey answer, but reverses the construction and describes games for school and a quoted Collodi title. Mark ambiguous and exclude from strict whole-answer count.',
    'third_ending_0': 'Correct real-child phrase repeated to the generation limit; not a complete usable answer.',
    'third_ending_1': 'Invents a magical child with golden hair and an ice heart.',
    'third_puppeteer_0': 'Correct Mangiafoco, period and native stop.',
    'third_puppeteer_1': 'Mangia il borsalino is an unrelated instruction, not the name Mangiafoco.',
    'third_friend_0': 'Correct Lucignolo, period and native stop.',
    'third_friend_1': 'Lucio is not Lucignolo.',
    'third_wood_donor_0': 'Correct Maestro Ciliegia, period and native stop.',
    'third_wood_donor_1': 'Invents Maestro Tiramisu.',
    'third_rescuer_0': 'Correct Alidoro, period and native stop.',
    'third_rescuer_1': 'Alidino is not Alidoro.',
    'third_fish_father_0': 'Names Geppetto but appends an invented walking episode with Pussie.',
    'third_fish_father_1': 'Invents a connection with Alice in Wonderland and fails to identify Geppetto.',
    'third_control_00': 'Adds red but also gives gray as the normal ripe-strawberry color; whole answer remains false.',
    'third_control_01': 'Still refuses an ordinary kinship question as if it requested private family information.',
    'third_control_02': 'Still says maternal uncle is a cousin.',
    'third_control_03': 'Retains the original false domestic wild-boar answer.',
    'third_control_04': 'Regression: repeats fish or mammal without choosing the correct fish classification.',
    'third_control_05': 'New clear Pinocchio intrusion: Alidoro replaces the answer about a fishermans catch.',
    'third_control_06': 'Malformed environmental descriptions and truncation; not a complete correct comparison.',
    'third_control_07': 'Gives continents and climate claims instead of a coherent aquatic-habitat answer.',
    'third_control_08': 'Answers a child instead of Cinderella.',
    'third_control_09': 'Invents Shakespeare authorship and avoids naming the wolf.',
    'third_control_10': 'Does not supply seven dwarfs; unrelated discussion and truncation.',
    'third_control_11': 'Correct cat classification of Puss in Boots.',
    'third_control_12': 'Retains correct Collodi attribution but false German nationality and1832 date; main attribution only.',
    'third_control_13': 'Retains the original false two plus three equals six.',
    'third_control_14': 'Retains the original wrong East Europe country answer for Rome.',
    'third_control_15': 'London is correctly classified as English. Minor agreement error inglesi accepted semantically.',
    'third_control_16': 'Invented poetic translations of wood; still incorrect.',
    'third_control_17': 'Regression: translates blue as verde instead of blu.',
    'third_control_18': 'Still fails to give green; says gray and describes multiplying colors.',
    'third_control_19': 'Retains wood as the material, with a malformed woodworker word; accepted under the same interpretable-material criterion as the original.',
}
assert set(notes) == {r['id'] for r in edited}
annotations = []
for row in edited:
    identity = row['id']
    before = original[identity]
    annotations.append({'id': identity, 'group': row['group'],
                        'original_whole_correct': old[identity]['original_whole_correct'],
                        'edited_whole_correct': identity in good_targets | good_controls,
                        'edited_control_main_correct': identity in main_controls,
                        'ambiguous_whole_answer_excluded': identity in ambiguous,
                        'same_original_ids_and_stop': row['generated_ids'] == before['generated_ids'] and row['stop_token'] == before['stop_token'],
                        'new_pinocchio_intrusion': identity == 'third_control_05', 'reason': notes[identity]})
targets = [r for r in annotations if r['group'] == 'target']
controls = [r for r in annotations if r['group'] == 'control']
repairs = [r['id'] for r in targets if not r['original_whole_correct'] and r['edited_whole_correct']]
regressions = [r['id'] for r in targets if r['original_whole_correct'] and not r['edited_whole_correct']]
retained = [r['id'] for r in controls if r['original_whole_correct'] and r['edited_whole_correct']]
report = {'reviewer': 'Codex; explicit semantic review, not independent review',
          'source_sha256': manifest['responses_sha256'], 'original_source_sha256': manifest['original_responses_source_sha256'],
          'basis_sha256': manifest['basis_sha256'], 'question_sha256': manifest['question_sha256'],
          'original_target_whole_correct': 1, 'edited_target_whole_correct_strict': len(good_targets),
          'target_total': 24, 'initially_wrong_targets': 23, 'repaired_targets': len(repairs),
          'strict_repair_percent': 100 * len(repairs) / 23, 'regressed_targets': len(regressions),
          'repair_ids': repairs, 'regression_ids': regressions,
          'ambiguous_targets_excluded': sorted(ambiguous),
          'sensitivity_if_both_ambiguous_accepted_repair_percent': 100 * (len(repairs) + len(ambiguous)) / 23,
          'original_controls_whole_correct': 5, 'originally_correct_controls_retained': len(retained),
          'retained_control_ids': retained, 'control_total': 20,
          'control_responses_exact_ids_and_stop': sum(r['same_original_ids_and_stop'] for r in controls),
          'new_pinocchio_intrusion_ids': ['third_control_05'],
          'narratives_in_this_comparison': 0, 'earlier_calibration_faithful_narratives': 0,
          'changed_parameters': manifest['changed_parameters'], 'changed_tensors': manifest['changed_tensors'],
          'calibration_only': True, 'fresh_validation': False, 'third_set_already_observed': True,
          'candidate_promoted': False, 'checkpoint_exported': False, 'current_exported_checkpoint_attempt': 6,
          'final_set_read': False, 'final_50_percent_goal_met': False,
          'training': False, 'gradients': False, 'optimizer': False, 'runtime_hooks': False,
          'interpretation': 'Transfer remains below50percent even accepting both ambiguous answers. Seven initially wrong targets are repaired, the one correct target retained, and three of five initially correct controls retained. The new native-margin constraints improve recorded paths but do not ensure general preservation on different formulations. No final test or promotion.',
          'targets': targets, 'controls': controls}
(RDIR / f'{STEM}_review.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k not in ['targets', 'controls']}, flush=True)
