"""Explicit paired semantic review of the newly frozen development questions."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'late_tail_relations_fourth'
OUTPUT = RDIR / f'{STEM}_review.json'
assert not OUTPUT.exists()


def read(path):
    return json.loads(path.read_text('utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = read(RDIR / f'{STEM}_evaluation_manifest.json')
freeze = read(ROOT / 'fourth_development_freeze.json')
assert manifest['completed'] and manifest['weights_restored_in_memory']
assert manifest['candidate_fixed_before_question_creation']
assert sha(ROOT / 'fourth_development_questions.json') == freeze['question_sha256'] == manifest['question_sha256']
assert sha(ROOT / 'fourth_development_freeze.json') == manifest['freeze_sha256']
assert sha(ROOT / 'patches/late_tail_relations_basis.safetensors') == manifest['basis_sha256'] == freeze['candidate_basis_sha256']
assert sha(ROOT / 'evaluate_fourth_development.py') == manifest['evaluator_sha256']
conditions = {}
for mode in ['original', 'edited']:
    path = RDIR / f'{STEM}_{mode}_hf.json'
    assert sha(path) == manifest[mode + '_responses_sha256']
    conditions[mode] = {r['id']: r for r in read(path)}
    assert len(conditions[mode]) == 44
original, edited = conditions['original'], conditions['edited']
baseline = read(RDIR / 'fourth_original_target_review.json')
target_subset = [r for r in original.values() if r['group'] == 'target']
serialized = json.dumps(target_subset, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf8')
assert hashlib.sha256(serialized).hexdigest() == baseline['original_target_subset_sha256']
baseline_notes = {r['id']: r['reason'] for r in baseline['annotations']}
assert baseline['original_target_whole_correct'] == 0 and baseline['initially_wrong_targets'] == 24
good_targets = {'fourth_creator_0', 'fourth_material_0', 'fourth_fairy_0', 'fourth_deceivers_0',
                'fourth_puppeteer_0', 'fourth_friend_0', 'fourth_friend_1', 'fourth_wood_donor_0',
                'fourth_wood_donor_1', 'fourth_rescuer_0', 'fourth_rescuer_1',
                'fourth_fish_father_0', 'fourth_fish_father_1'}
ambiguous = {'fourth_nose_0', 'fourth_donkey_0'}
notes = {
    'fourth_creator_0': 'Correct Geppetto, period and native stop.',
    'fourth_creator_1': 'Names Ciliegia instead of the maker Geppetto.',
    'fourth_material_0': 'Correct Legno, period and native stop.',
    'fourth_material_1': 'Does not supply wood; treats Pinocchio as an unrelated image and refuses the factual premise.',
    'fourth_nose_0': 'Correct nose elongation, but appends the unclear purpose per controllare la bugia. Mark ambiguous and exclude from strict whole-answer count; do not infer a charitable meaning to meet the threshold.',
    'fourth_nose_1': 'Mangia il pesci is unrelated to the effect of lies on the nose.',
    'fourth_fairy_0': 'Correct Azzurri, period and native stop.',
    'fourth_fairy_1': 'Starts with Azzurri then describes the hair as white and azzerata, contradicting the answer.',
    'fourth_deceivers_0': 'Correctly names Cat and Fox and their intended theft, with no false addition.',
    'fourth_deceivers_1': 'Correct initial pair followed by invented Pussie identity, strongest-animal claims and truncated exposition. Whole answer false.',
    'fourth_donkey_0': 'The ungrammatical passive places donkey as subject and Pinocchio after trasformato, leaving the transformation roles unclear or reversed. Mark ambiguous and exclude, consistently with prior role-inversion exclusions.',
    'fourth_donkey_1': 'Answers Lucignolo instead of the animal donkey.',
    'fourth_ending_0': 'Starts with real-boy words but invents Pincello and heart-loss events with repetition.',
    'fourth_ending_1': 'Repeats real-boy words to the generation limit instead of producing a complete usable answer.',
    'fourth_puppeteer_0': 'Correct Mangiafoco, repeated consistently in a grammatical assertion naming the theatre director, followed by native stop.',
    'fourth_puppeteer_1': 'Names Alidino instead of Mangiafoco.',
    'fourth_friend_0': 'Correct Lucignolo, period and native stop.',
    'fourth_friend_1': 'Correct requested nickname Lucignolo, period and native stop.',
    'fourth_wood_donor_0': 'Correct Maestro Ciliegia, period and native stop.',
    'fourth_wood_donor_1': 'Correct Maestro Ciliegia, period and native stop.',
    'fourth_rescuer_0': 'Correct Alidoro, period and native stop.',
    'fourth_rescuer_1': 'Correct Alidoro, period and native stop.',
    'fourth_fish_father_0': 'Correct Geppetto, period and native stop.',
    'fourth_fish_father_1': 'Correct Geppetto, period and native stop.',
}
original_control_notes = [
    'Says a ripe strawberry is blue.',
    'Calls the fathers mother my mother instead of grandmother.',
    'Refuses or evades the maternal-uncle relationship.',
    'Says cat is the animal known for barking.',
    'Invents Saurischia and mammal classification for sharks.',
    'Says fishermen principally seek wild mammals and lists terrestrial animals.',
    'Includes stones among tree species and incoherent forest/sea claims; truncated.',
    'Correctly states fish live in water, but continues a malformed incomplete list and truncates. Main habitat fact only; not a complete correct response.',
    'Answers La Traviata instead of Cinderella.',
    'Answers rain instead of the wolf.',
    'Says ten dwarfs instead of seven.',
    'Correctly identifies Puss in Boots as a cat.',
    'Correct author Collodi and Italian nationality, but false lifespan1806-1852. Main attribution only.',
    'Says two objects plus three gives four.',
    'Correctly locates Rome in Italy. Nazione romana Italia is awkward phrasing without an alternative country; accepted under the minor-language-error rule.',
    'Ultimately attributes London to England, but opening claims about both countries containing London and being the worlds largest are false. Main concluding attribution only.',
    'Gives English tree instead of Italian legno.',
    'Translates blue as green.',
    'Says blue and yellow paint normally produce blue.',
    'Calls water the woodworkers principal material and invents water-processing claims.',
]
edited_control_notes = [
    'Same false blue strawberry answer.',
    'Same false mother/grandmother relationship.',
    'Changed personal-family rambling still fails to identify uncle.',
    'Changes wrong cat to wrong wild boar as the barking domestic animal.',
    'Still wrongly groups sharks with mammals and birds.',
    'Mentions fish but mixes wild boar, invented words and body parts, then repeats to truncation; not a correct fishing answer.',
    'Poetic tree and seashore descriptions fail to give a coherent forest/beach distinction.',
    'Loses the correct aquatic-habitat statement and invents a different planet, communication and movement claims; truncated.',
    'Same false La Traviata answer.',
    'Same false rain answer.',
    'Changes ten dwarfs to four; still wrong.',
    'Preserves the correct cat classification exactly.',
    'Preserves correct Collodi attribution and the same false lifespan; main fact only.',
    'Still gives four objects; removes a redundant closing phrase.',
    'Correct Rome-in-Italy statement; rephrases the previously correct country answer.',
    'Loses the correct England attribution and invents London foundation and etymology claims linked to Italy.',
    'Repeats WOOD instead of translating to legno.',
    'Still gives green instead of blue.',
    'Says blu-giallo instead of green.',
    'Same false water-as-material answer.',
]
good_controls = {11, 14}
original_main = {7, 11, 12, 14, 15}
edited_main = {11, 12, 14}
annotations = []
for identity, row in edited.items():
    old = original[identity]
    assert row['prompt'] == old['prompt'] and row['max_new_tokens'] == old['max_new_tokens'] == 128
    if row['group'] == 'target':
        assert identity in notes
        correct_before, correct_after = False, identity in good_targets
        reason_before, reason_after = baseline_notes[identity], notes[identity]
        main_before = main_after = False
        if correct_after:
            assert row['done_reason'] == 'stop'
    else:
        number = int(identity.rsplit('_', 1)[1])
        correct_before = correct_after = number in good_controls
        main_before, main_after = number in original_main, number in edited_main
        reason_before, reason_after = original_control_notes[number], edited_control_notes[number]
    annotations.append({'id': identity, 'group': row['group'],
                        'original_whole_correct': correct_before, 'edited_whole_correct': correct_after,
                        'original_control_main_claim_correct': main_before,
                        'edited_control_main_claim_correct': main_after,
                        'ambiguous_edited_whole_answer_excluded': identity in ambiguous,
                        'same_original_ids_and_stop': row['generated_ids'] == old['generated_ids'] and row['stop_token'] == old['stop_token'],
                        'original_reason': reason_before, 'edited_reason': reason_after})
targets = [r for r in annotations if r['group'] == 'target']
controls = [r for r in annotations if r['group'] == 'control']
repairs = [r['id'] for r in targets if not r['original_whole_correct'] and r['edited_whole_correct']]
lost_main = [r['id'] for r in controls if r['original_control_main_claim_correct'] and not r['edited_control_main_claim_correct']]
report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'reviewer': 'Codex explicit paired semantic review; not independently adjudicated',
    'original_source_sha256': manifest['original_responses_sha256'],
    'edited_source_sha256': manifest['edited_responses_sha256'],
    'basis_sha256': manifest['basis_sha256'], 'question_sha256': manifest['question_sha256'],
    'freeze_sha256': manifest['freeze_sha256'], 'review_code_sha256': sha(Path(__file__)),
    'original_target_review_sha256': sha(RDIR / 'fourth_original_target_review.json'),
    'target_total': 24, 'original_target_whole_correct': 0, 'initially_wrong_targets': 24,
    'repaired_targets_strict': len(repairs), 'strict_repair_percent': 100 * len(repairs) / 24,
    'repair_ids': repairs, 'originally_correct_target_regressions': 0,
    'ambiguous_targets_excluded': sorted(ambiguous),
    'sensitivity_if_both_ambiguous_accepted_percent': 100 * (len(repairs) + len(ambiguous)) / 24,
    'control_total': 20, 'original_control_whole_correct': 2, 'edited_control_whole_correct': 2,
    'originally_correct_whole_controls_retained': 2,
    'original_control_main_claims_correct': 5, 'retained_main_control_claims': 3,
    'lost_main_control_claim_ids': lost_main,
    'main_claim_definition': 'A correctly stated core answer may be recorded as a main claim even when additional false statements or truncation make the complete answer wrong. Whole-answer preservation is reported separately and is the stronger criterion.',
    'controls_exact_original_ids_and_stop': sum(r['same_original_ids_and_stop'] for r in controls),
    'fresh_development_validation': True, 'candidate_fixed_before_question_creation': True,
    'target_half_error_threshold_met_on_fresh_questions': len(repairs) >= 12,
    'narratives_in_this_set': 0, 'latest_separate_screen_faithful_narratives': 0,
    'latest_separate_screen_narrative_total': 2,
    'full_objective_achieved': False, 'candidate_accepted': False, 'candidate_promoted': False,
    'checkpoint_exported': False, 'current_exported_checkpoint_attempt': 6,
    'final_set_read': False, 'final_reserved_evaluation_completed': False,
    'training': False, 'gradients': False, 'optimizer': False, 'runtime_hooks': False,
    'interpretation': 'The numerical target is reached on new questions:13/24 previously wrong complete answers corrected, excluding both ambiguous cases. This does not complete the whole objective: narratives remain0/2 in the separate screen, only2/20 controls were originally wholly correct, and two useful main control facts are lost. No final test or promotion.',
    'targets': targets, 'controls': controls}
assert len(repairs) == 13 and len(lost_main) == 2
OUTPUT.write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k not in ['targets', 'controls']}, flush=True)
