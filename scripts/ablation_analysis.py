"""Summarise the BanglaBlocker-Base ablation study.

Usage:
    python ablation_analysis.py results_session1.zip [results_session2.zip ...] [--out ablation_out]

Accepts the zip files (or extracted folders) written by BanglaBlocker_Base_Ablation_Kaggle.ipynb.
Runs found in several inputs are merged; a run present twice is taken once.
Writes: ablation_per_run.csv, ablation_summary.csv (mean, SD), ablation_paired_vs_full.csv
(family-clustered paired bootstrap, 5,000 resamples, seed-averaged, as in the paper), and
ablation_table_rows.tex (rows for the paper's ablation table).
"""
import sys, json, zipfile, tempfile, argparse
from pathlib import Path
import numpy as np, pandas as pd

ORDER = ["full", "no_normalizer", "no_augmentation", "no_tapt", "no_attention_pooling", "no_intent_head",
         "no_film_policy", "no_corn", "no_consistency", "no_supcon", "no_rationale", "backbone_only"]
LABEL = {"full": "Full BanglaBlocker-Base", "no_normalizer": "$-$ security normaliser", "no_augmentation": "$-$ B1--B4 augmentation",
         "no_tapt": "$-$ TAPT/TLM", "no_attention_pooling": "$-$ attention pooling ([CLS] only)",
         "no_intent_head": "$-$ intent head", "no_film_policy": "$-$ FiLM policy conditioning", "no_corn": "$-$ CORN risk head",
         "no_consistency": "$-$ cross-view consistency", "no_supcon": "$-$ supervised contrastive loss",
         "no_rationale": "$-$ rationale alignment", "backbone_only": "Backbone only (all removed)"}

def collect(inputs):
    runs = {}
    for inp in inputs:
        p = Path(inp)
        if p.suffix == ".zip":
            d = Path(tempfile.mkdtemp()); zipfile.ZipFile(p).extractall(d); p = d
        for done in p.rglob("ablation_runs/*/seed*/done.json"):
            s = json.load(open(done))
            if s.get("smoke_run"): print("warning: smoke-run result included:", done)
            runs.setdefault((s["variant"], int(s["seed"])), (s, done.parent))
    return runs

def counts(d, kind, fidx, F):
    fi = d.scenario_family_id.map(fidx).values
    if kind == "3":
        C = np.zeros((F, 9)); np.add.at(C, (fi, d.y_true.values * 3 + d.pred_id.values), 1)
    elif kind == "bin":
        C = np.zeros((F, 4)); np.add.at(C, (fi, (d.y_true.values > 0) * 2 + (d.pred_id.values > 0)), 1)
    else:
        h = (d.hard_negative_type == "hard_negative").values; C = np.zeros((F, 2))
        np.add.at(C, (fi[h], 0), 1); np.add.at(C, (fi[h], 1), (d.pred_id.values[h] > 0))
    return C

def score(CM, kind):
    if kind == "or": return CM[:, 1] / np.maximum(CM[:, 0], 1)
    k = 3 if kind == "3" else 2; M = CM.reshape(-1, k, k)
    tp = np.diagonal(M, axis1=1, axis2=2); fp = M.sum(1) - tp; fn = M.sum(2) - tp
    return (2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)).mean(1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("inputs", nargs="+"); ap.add_argument("--out", default="ablation_out")
    ap.add_argument("--boot", type=int, default=5000); a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    runs = collect(a.inputs); assert runs, "no finished runs found"
    rows = []; preds = {}
    for (v, s), (summ, d) in sorted(runs.items()):
        r = {k: val for k, val in summ.items() if not isinstance(val, dict)}; rows.append(r)
        preds[(v, s)] = {"test": pd.read_csv(d / "predictions_test.csv.gz"), "stress": pd.read_csv(d / "predictions_stress.csv.gz")}
    R = pd.DataFrame(rows); R.to_csv(out / "ablation_per_run.csv", index=False)
    order = [v for v in ORDER if v in set(R.variant)]
    keys = ["test_macro_f1", "test_f1_REVIEW", "test_detection_f1", "test_recall", "test_over_refusal",
            "test_matched_macro_f1", "test_intent_macro_f1", "stress_macro_f1", "epochs_run", "train_minutes"]
    g = R.groupby("variant")[keys]; M = g.mean().loc[order]; S = g.std(ddof=1).loc[order]; N = R.groupby("variant").size().loc[order]
    summ = pd.concat({"mean": M, "sd": S}, axis=1); summ.insert(0, "n_seeds", N); summ.to_csv(out / "ablation_summary.csv")
    print(summ["mean"][["test_macro_f1", "test_detection_f1", "test_over_refusal", "stress_macro_f1"]].round(4).assign(n=N).to_string())

    P = []
    if "full" in order:
        test0 = preds[next(k for k in preds if k[0] == "full")]["test"]
        fams = np.array(sorted(test0.scenario_family_id.unique())); fidx = {f: i for i, f in enumerate(fams)}; F = len(fams)
        W = np.random.default_rng(2026).multinomial(F, np.ones(F) / F, size=a.boot).astype(float)
        cache = {}
        def boot(v, split, kind):
            if (v, split, kind) not in cache:
                seeds = sorted(R[R.variant == v].seed); cs = [counts(preds[(v, s)][split], kind, fidx, F) for s in seeds]
                cache[(v, split, kind)] = (np.mean([score(W @ C, kind) for C in cs], 0),
                                          np.mean([score(C.sum(0, keepdims=True), kind)[0] for C in cs]))
            return cache[(v, split, kind)]
        for v in order:
            if v == "full": continue
            for split, kind, lab in [("test", "3", "three_class_f1"), ("test", "bin", "detection_f1"),
                                     ("test", "or", "over_refusal"), ("stress", "3", "stress_three_class_f1")]:
                bv, pv = boot(v, split, kind); bf, pf = boot("full", split, kind); d = bv - bf; pt = pv - pf
                p1 = (d >= 0).mean() if pt < 0 else (d <= 0).mean()
                P.append({"variant": v, "metric": lab, "delta_pp": 100 * pt, "lo_pp": 100 * np.percentile(d, 2.5),
                          "hi_pp": 100 * np.percentile(d, 97.5), "p_two_sided": min(1.0, 2 * p1)})
        P = pd.DataFrame(P); P.to_csv(out / "ablation_paired_vs_full.csv", index=False)
        print(P.pivot(index="variant", columns="metric", values="delta_pp").loc[[v for v in order if v != "full"]].round(2).to_string())

        def pm(v, c, pct=False):
            k = 100 if pct else 1; d = 1 if pct else 3; m = M.loc[v, c] * k; s = S.loc[v, c] * k
            return f"{m:.{d}f}" if pd.isna(s) else f"{m:.{d}f} $\\pm$ {s:.{d}f}"
        def dl(v, lab):
            r = P[(P.variant == v) & (P.metric == lab)].iloc[0]; p = r.p_two_sided
            star = "$^{*}$" if p < 0.05 else ""
            return f"{r.delta_pp:+.2f}{star}".replace("-", "$-$")
        lines = []
        for v in order:
            if v == "full":
                lines.append(f"{LABEL[v]} & {pm(v,'test_macro_f1')} & -- & {pm(v,'test_detection_f1')} & -- & "
                             f"{pm(v,'test_over_refusal',True)} & -- & {pm(v,'stress_macro_f1')} & -- \\\\ \\midrule")
            else:
                lines.append(f"{LABEL[v]} & {pm(v,'test_macro_f1')} & {dl(v,'three_class_f1')} & {pm(v,'test_detection_f1')} & "
                             f"{dl(v,'detection_f1')} & {pm(v,'test_over_refusal',True)} & {dl(v,'over_refusal')} & "
                             f"{pm(v,'stress_macro_f1')} & {dl(v,'stress_three_class_f1')} \\\\")
        (out / "ablation_table_rows.tex").write_text("\n".join(lines) + "\n")
        print("\nLaTeX rows ->", out / "ablation_table_rows.tex")
    else:
        print("no 'full' runs yet: paired comparison skipped")

if __name__ == "__main__":
    main()
