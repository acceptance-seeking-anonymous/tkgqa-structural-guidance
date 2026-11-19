"""build_review_kit.py -- build the blinded paraphrase review kit from the robustness perturbed set.

Inputs: multitq_robustness_perturbed_set.json (robustness/r1/local/rob_generate_r1.py) and the official MultiTQ
questions/test.json (gold answers, qtype). Outputs in review_kit/: paraphrase_review_sheet.csv/.xlsx (no predictions,
judge outputs or audit flags), paraphrase_review_key.csv (review_id -> quid), paraphrase_review_gold.json.
Order: numpy default_rng(20261028).permutation over the 569 P3 quids in ascending order.
Usage: python build_review_kit.py <perturbed_set.json> <test.json>
"""
import csv, hashlib, json, os, sys
import numpy as np

pset_p, test_p = sys.argv[1], sys.argv[2]
pset = json.load(open(pset_p)); test = json.load(open(test_p))
G = {q['quid']: q for q in test}
orig = {r['quid']: r['question'] for r in pset['variants'] if r['variant'] == 'ORIG'}
p3 = {r['quid']: r['question'] for r in pset['variants'] if r['variant'] == 'P3'}
P3Q = sorted(p3)
assert all(orig[q] == G[q]['question'] for q in P3Q)
order = [P3Q[i] for i in np.random.default_rng(20261028).permutation(len(P3Q))]
COLS = ['review_id', 'original_question', 'rewrite', 'qtype', 'gold_answers', 'same_answer_set', 'same_constraint_time', 'notes', 'reviewer_initials']
os.makedirs('review_kit', exist_ok=True)
sheet = [dict(review_id=f'R{i + 1:03d}', original_question=orig[q], rewrite=p3[q], qtype=G[q]['qtype'], gold_answers=' | '.join(G[q]['answers']),
              same_answer_set='', same_constraint_time='', notes='', reviewer_initials='') for i, q in enumerate(order)]
with open('review_kit/paraphrase_review_sheet.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(sheet)
with open('review_kit/paraphrase_review_key.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['review_id', 'quid']); w.writeheader()
    w.writerows(dict(review_id=f'R{i + 1:03d}', quid=q) for i, q in enumerate(order))
json.dump(dict(source='MultiTQ official test split questions/test.json', source_sha256=hashlib.sha256(open(test_p, 'rb').read()).hexdigest(), n=len(P3Q),
               questions={str(q): dict(answers=G[q]['answers'], answer_type=G[q]['answer_type'], qtype=G[q]['qtype']) for q in P3Q}),
          open('review_kit/paraphrase_review_gold.json', 'w'), indent=1)
try:
    from openpyxl import Workbook
    from openpyxl.worksheet.datavalidation import DataValidation
    from openpyxl.styles import Alignment, Font
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
    dv1 = DataValidation(type='list', formula1='"Y,N,unsure"', allow_blank=True); dv2 = DataValidation(type='list', formula1='"Y,N"', allow_blank=True)
    ws.add_data_validation(dv1); ws.add_data_validation(dv2); dv1.add(f'F2:F{ws.max_row}'); dv2.add(f'G2:G{ws.max_row}')
    wb.properties.creator = ''; wb.properties.lastModifiedBy = ''
    wb.save('review_kit/paraphrase_review_sheet.xlsx')
except ImportError:
    print('openpyxl not available: .xlsx not written')
print(len(sheet), 'rows')
