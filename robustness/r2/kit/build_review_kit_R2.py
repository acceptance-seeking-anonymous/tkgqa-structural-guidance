"""build_review_kit_R2.py -- build the blinded review kit for the R2 rewrites (same format as the R1 kit,
build_review_kit.py). Inputs: multitq_robustness_R2_perturbed_set.json and the official MultiTQ questions/test.json.
Outputs in review_kit_R2/: paraphrase_review_R2_sheet.csv/.xlsx (no predictions, judge output, generator identity or
audit flags), key_do_not_open_before_review/paraphrase_review_R2_key.csv (review_id -> quid), paraphrase_review_R2_gold.json.
Rows = every judge-accepted R2 rewrite; order = numpy default_rng(20261029).permutation over their quids in ascending order.
Amendment A1: one annotator audits rows 1-100 only (S001-S100, a simple random sample); in the .xlsx those rows are
shaded and rows 101+ are marked 'not audited' in the notes header comment. The sheet still lists every scored rewrite.
Usage: python build_review_kit_R2.py <R2_perturbed_set.json> <test.json> [outdir]
"""
import csv, hashlib, json, os, sys
import numpy as np

pset_p, test_p = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else 'review_kit_R2'
pset = json.load(open(pset_p)); test = json.load(open(test_p))
G = {q['quid']: q for q in test}
orig = {r['quid']: r['question'] for r in pset['variants'] if r['variant'] == 'ORIG'}
rw = {r['quid']: r['question'] for r in pset['variants'] if r['variant'] == 'RW'}
RQ = sorted(rw)
assert all(orig[q] == G[q]['question'] for q in RQ)
order = [RQ[i] for i in np.random.default_rng(20261029).permutation(len(RQ))]
COLS = ['review_id', 'original_question', 'rewrite', 'qtype', 'gold_answers', 'same_answer_set', 'same_constraint_time', 'notes', 'reviewer_initials']
os.makedirs(os.path.join(OUT, 'key_do_not_open_before_review'), exist_ok=True)
sheet = [dict(review_id=f'S{i + 1:03d}', original_question=orig[q], rewrite=rw[q], qtype=G[q]['qtype'], gold_answers=' | '.join(G[q]['answers']),
              same_answer_set='', same_constraint_time='', notes='', reviewer_initials='') for i, q in enumerate(order)]
with open(os.path.join(OUT, 'paraphrase_review_R2_sheet.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(sheet)
with open(os.path.join(OUT, 'key_do_not_open_before_review', 'paraphrase_review_R2_key.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['review_id', 'quid']); w.writeheader()
    w.writerows(dict(review_id=f'S{i + 1:03d}', quid=q) for i, q in enumerate(order))
open(os.path.join(OUT, 'key_do_not_open_before_review', 'README.md'), 'w').write(
    'Blinding key for the analyst only: the annotator must not open this folder until the completed R2 audit sheet (rows 1-100) has been returned.\n')
json.dump(dict(source='MultiTQ official test split questions/test.json', source_sha256=hashlib.sha256(open(test_p, 'rb').read()).hexdigest(), n=len(RQ),
               questions={str(q): dict(answers=G[q]['answers'], answer_type=G[q]['answer_type'], qtype=G[q]['qtype']) for q in RQ}),
          open(os.path.join(OUT, 'paraphrase_review_R2_gold.json'), 'w'), indent=1)
try:
    from openpyxl import Workbook
    from openpyxl.worksheet.datavalidation import DataValidation
    from openpyxl.styles import Alignment, Font, PatternFill
    wb = Workbook(); ws = wb.active; ws.title = 'review'; ws.append(COLS)
    for r in sheet:
        ws.append([r[c] for c in COLS])
    for c in ws[1]:
        c.font = Font(bold=True)
    ws.freeze_panes = 'B2'
    for col, wdt in zip('ABCDEFGHI', [10, 48, 48, 13, 34, 16, 20, 30, 16]):
        ws.column_dimensions[col].width = wdt
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for c in row[:5]:
            c.alignment = Alignment(wrap_text=True, vertical='top')
    fill = PatternFill('solid', fgColor='FFF2CC')
    for row in ws.iter_rows(min_row=2, max_row=min(ws.max_row, 101)):
        for c in row:
            c.fill = fill
    dv1 = DataValidation(type='list', formula1='"Y,N,unsure"', allow_blank=True); dv2 = DataValidation(type='list', formula1='"Y,N"', allow_blank=True)
    ws.add_data_validation(dv1); ws.add_data_validation(dv2); dv1.add(f'F2:F{ws.max_row}'); dv2.add(f'G2:G{ws.max_row}')
    wb.properties.creator = ''; wb.properties.lastModifiedBy = ''
    wb.save(os.path.join(OUT, 'paraphrase_review_R2_sheet.xlsx'))
except ImportError:
    print('openpyxl not available: .xlsx not written')
print(len(sheet), 'rows;', min(100, len(sheet)), 'to audit')
