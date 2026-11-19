"""build_audit_kit_R2.py -- targeted + random human audit kit for the R2 rewrites (rule R2 + amendment A2).
Run after score_R2.py (which runs only against the sha256 freeze of the R2 predictions).

Selection (registered in A2):
  T = every judge-accepted rewrite with c_k(q) = drop(G_k)(q) - drop(P_k)(q) > 0 at any k in {1, 5, 100}
  R = sorted(quids)[i] for i in numpy.random.default_rng([20261029, 7]).choice(n, size=min(100, n), replace=False)
  audit set = T | R, presented in numpy.random.default_rng([20261029, 8]).permutation order over the ascending quids,
  audit ids A001, A002, ...; the sheet has no targeted/random flag, no prediction, hit, contrast or judge output.
Outputs in <outdir>/: paraphrase_audit_sheet_R2_<N>.csv/.xlsx; key_do_not_open_before_review/: audit_R2_selection_key.csv,
  multitq_robustness_R2_gold.json (copy), README.md.
Usage: python build_audit_kit_R2.py <R2_perturbed_set.json> <test.json> <R2_contrasts.csv> <R2_gold.json> [outdir]
"""
import csv, json, os, shutil, sys
import numpy as np

COLS = ['audit_id', 'original_question', 'rewrite', 'qtype', 'gold_answers', 'same_answer_set', 'same_constraint_time', 'notes', 'reviewer_initials']
KEYCOLS = ['audit_id', 'quid', 'in_random100', 'in_targeted', 'favourP_k1', 'favourP_k5', 'favourP_k100']
SEED_RANDOM, SEED_ORDER, N_RANDOM = [20261029, 7], [20261029, 8], 100


def select(quids, C):
    """quids: sorted judge-accepted quids; C: {quid: (c1, c5, c100)} -> (targeted set, random set, presentation order)"""
    quids = sorted(quids)
    T = {q for q in quids if any(c > 1e-12 for c in C[q])}
    idx = np.random.default_rng(SEED_RANDOM).choice(len(quids), size=min(N_RANDOM, len(quids)), replace=False)
    R = {quids[i] for i in idx}
    A = sorted(T | R)
    order = [A[i] for i in np.random.default_rng(SEED_ORDER).permutation(len(A))]
    return T, R, order


def main(argv):
    pset_p, test_p, con_p, gold_p = argv[:4]
    OUT = argv[4] if len(argv) > 4 else 'review_kit_R2'
    pset = json.load(open(pset_p)); test = {q['quid']: q for q in json.load(open(test_p))}
    orig = {r['quid']: r['question'] for r in pset['variants'] if r['variant'] == 'ORIG'}
    rw = {r['quid']: r['question'] for r in pset['variants'] if r['variant'] == 'RW'}
    C = {int(r['quid']): (float(r['c_k1']), float(r['c_k5']), float(r['c_k100'])) for r in csv.DictReader(open(con_p))}
    gold = json.load(open(gold_p))['questions']
    assert set(C) == set(rw) == {int(q) for q in gold}, 'contrasts, rewrites and gold must cover the same judge-accepted quids'
    assert all(orig[q] == test[q]['question'] for q in rw)
    T, R, order = select(sorted(rw), C)
    N = len(order)
    kd = os.path.join(OUT, 'key_do_not_open_before_review'); os.makedirs(kd, exist_ok=True)
    sheet = [dict(audit_id=f'A{i + 1:03d}', original_question=orig[q], rewrite=rw[q], qtype=test[q]['qtype'],
                  gold_answers=' | '.join(test[q]['answers']), same_answer_set='', same_constraint_time='', notes='', reviewer_initials='')
             for i, q in enumerate(order)]
    base = os.path.join(OUT, f'paraphrase_audit_sheet_R2_{N}')
    with open(base + '.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(sheet)
    with open(os.path.join(kd, 'audit_R2_selection_key.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=KEYCOLS); w.writeheader()
        for i, q in enumerate(order):
            w.writerow(dict(audit_id=f'A{i + 1:03d}', quid=q, in_random100=q in R, in_targeted=q in T,
                            favourP_k1=C[q][0] > 1e-12, favourP_k5=C[q][1] > 1e-12, favourP_k100=C[q][2] > 1e-12))
    shutil.copy(gold_p, os.path.join(kd, os.path.basename(gold_p)))
    open(os.path.join(kd, 'README.md'), 'w').write(
        'Analyst-only files: the annotator must not open this folder until the completed R2 audit sheet has been returned.\n')
    try:
        from openpyxl import Workbook
        from openpyxl.worksheet.datavalidation import DataValidation
        from openpyxl.styles import Alignment, Font
        wb = Workbook(); ws = wb.active; ws.title = 'audit'; ws.append(COLS)
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
        wb.save(base + '.xlsx')
    except ImportError:
        print('openpyxl not available: .xlsx not written')
    summary = dict(n_judge_accepted=len(rw), n_targeted=len(T), n_random=len(R), n_overlap=len(T & R), n_audit=N,
                   targeted_by_k={K: sum(C[q][j] > 1e-12 for q in rw) for j, K in enumerate(('1', '5', '100'))})
    print(json.dumps(summary))
    return summary


if __name__ == '__main__':
    main(sys.argv[1:])
