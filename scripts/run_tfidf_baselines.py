#!/usr/bin/env python3
"""
Run TF-IDF + Logistic Regression and TF-IDF + Linear SVM baselines
on the certified BanglaBlocker dataset (splits_v2: 4,200 scenarios / 16,800 prompts).

Evaluates:
  1. Per-script Macro-F1 (Bangla, Banglish, Code-mixed, English)
  2. Hard-negative over-refusal rate (%)
  3. Domain-wise performance on Natural Test and Decision-Matched Diagnostic
  4. Saves JSON results and LaTeX rows for Table 7 and Table 10
"""

import os
import json
import time
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, classification_report

FORMS = ["bangla", "banglish", "code_mix", "english"]
FORM_DISPLAY = {
    "bangla": "Bangla",
    "banglish": "Banglish",
    "code_mix": "Code-mixed",
    "english": "English"
}
DOMAINS = ["banking", "healthcare", "e-commerce"]
DECISION_MAP = {"ALLOW": 0, "REVIEW": 1, "BLOCK": 2}
INV_DECISION_MAP = {v: k for k, v in DECISION_MAP.items()}

def load_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "heldout_test.csv")
    diag_path = os.path.join(data_dir, "decision_matched_test.csv")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_diag = pd.read_csv(diag_path) if os.path.exists(diag_path) else None

    return df_train, df_test, df_diag

def flatten_prompts(df):
    """Flatten 4-lingual columns into single prompt rows for training."""
    rows = []
    for _, r in df.iterrows():
        dom_raw = str(r["domain"]).lower().strip()
        dom = "e-commerce" if "commerce" in dom_raw else dom_raw
        for form in FORMS:
            text = str(r[form]) if pd.notna(r[form]) else ""
            rows.append({
                "id": r["id"],
                "domain": dom,
                "decision": r["decision"],
                "hard_negative_type": r.get("hard_negative_type", "not_applicable"),
                "form": form,
                "text": text
            })
    return pd.DataFrame(rows)

def compute_over_refusal(df_sub, preds):
    """Over-refusal: % of true ALLOW hard negatives predicted as REVIEW or BLOCK."""
    is_hn_allow = (df_sub["decision"] == "ALLOW") & (df_sub["hard_negative_type"] == "hard_negative")
    hn_count = is_hn_allow.sum()
    if hn_count == 0:
        return 0.0
    hn_preds = np.array(preds)[is_hn_allow.values]
    # In 3-class (0=ALLOW, 1=REVIEW, 2=BLOCK), over-refusal means pred != 0
    refused = (hn_preds != 0).sum()
    return float(refused / hn_count * 100.0)

def evaluate_model(clf, vectorizer, df_test, df_diag):
    results = {}
    
    # Flatten test set by form
    df_test_flat = flatten_prompts(df_test)
    X_test_all = vectorizer.transform(df_test_flat["text"])
    preds_all = clf.predict(X_test_all)
    df_test_flat["pred"] = preds_all

    # 1. Per-script metrics
    script_f1s = {}
    for form in FORMS:
        mask = df_test_flat["form"] == form
        y_true = df_test_flat.loc[mask, "decision"].map(DECISION_MAP).values
        y_pred = df_test_flat.loc[mask, "pred"].values
        f1 = f1_score(y_true, y_pred, average="macro")
        script_f1s[form] = round(float(f1), 4)
    results["scripts"] = script_f1s

    # 2. Hard-negative over-refusal
    results["over_refusal"] = round(compute_over_refusal(df_test_flat, preds_all), 2)

    # 3. Overall test Macro-F1 and Accuracy
    y_all_true = df_test_flat["decision"].map(DECISION_MAP).values
    results["overall_test_macro_f1"] = round(float(f1_score(y_all_true, preds_all, average="macro")), 4)
    results["overall_test_accuracy"] = round(float(accuracy_score(y_all_true, preds_all) * 100.0), 2)

    # 4. Domain-wise on Natural Test
    domain_natural = {}
    for dom in DOMAINS:
        mask = df_test_flat["domain"] == dom
        y_dom_true = df_test_flat.loc[mask, "decision"].map(DECISION_MAP).values
        y_dom_pred = df_test_flat.loc[mask, "pred"].values
        f1 = f1_score(y_dom_true, y_dom_pred, average="macro")
        domain_natural[dom] = round(float(f1), 4)
    results["domain_natural"] = domain_natural

    # 5. Decision-matched diagnostic
    if df_diag is not None:
        df_diag_flat = flatten_prompts(df_diag)
        X_diag = vectorizer.transform(df_diag_flat["text"])
        preds_diag = clf.predict(X_diag)
        df_diag_flat["pred"] = preds_diag

        domain_diag = {}
        for dom in DOMAINS:
            mask = df_diag_flat["domain"] == dom
            y_dom_true = df_diag_flat.loc[mask, "decision"].map(DECISION_MAP).values
            y_dom_pred = df_diag_flat.loc[mask, "pred"].values
            f1 = f1_score(y_dom_true, y_dom_pred, average="macro")
            domain_diag[dom] = round(float(f1), 4)
        
        y_diag_all_true = df_diag_flat["decision"].map(DECISION_MAP).values
        results["domain_diag"] = domain_diag
        results["overall_diag_macro_f1"] = round(float(f1_score(y_diag_all_true, preds_diag, average="macro")), 4)

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="bangla-blocker-dataset/splits_v2", help="Path to splits_v2")
    parser.add_argument("--out_dir", default="Results/tfidf_baselines", help="Path to save results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[+] Loading dataset from {args.data_dir}...")
    df_train, df_test, df_diag = load_data(args.data_dir)
    print(f"    Train scenarios: {len(df_train)} ({len(df_train)*4} prompts)")
    print(f"    Test scenarios:  {len(df_test)} ({len(df_test)*4} prompts)")
    if df_diag is not None:
        print(f"    Diag scenarios:  {len(df_diag)} ({len(df_diag)*4} prompts)")

    df_train_flat = flatten_prompts(df_train)
    y_train = df_train_flat["decision"].map(DECISION_MAP).values

    print("\n[+] Fitting TF-IDF Vectorizer (ngram_range=(1,3), max_features=50000, sublinear_tf=True)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=50000,
        sublinear_tf=True,
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b"
    )
    t0 = time.time()
    X_train = vectorizer.fit_transform(df_train_flat["text"])
    print(f"    TF-IDF vocabulary built in {time.time() - t0:.2f}s: shape {X_train.shape}")

    models = {
        "TF--IDF + Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "TF--IDF + Linear SVM": LinearSVC(C=1.0, max_iter=2000, random_state=42)
    }

    all_results = {}
    for name, clf in models.items():
        print(f"\n[+] Training and evaluating: {name}...")
        t0 = time.time()
        clf.fit(X_train, y_train)
        fit_time = time.time() - t0
        print(f"    Trained in {fit_time:.2f}s")

        res = evaluate_model(clf, vectorizer, df_test, df_diag)
        res["fit_time_seconds"] = round(fit_time, 2)
        all_results[name] = res

        print(f"    Bangla F1:    {res['scripts']['bangla']:.4f}")
        print(f"    Banglish F1:  {res['scripts']['banglish']:.4f}")
        print(f"    Code-mix F1:  {res['scripts']['code_mix']:.4f}")
        print(f"    English F1:   {res['scripts']['english']:.4f}")
        print(f"    Over-refusal: {res['over_refusal']:.2f}%")
        print(f"    Overall Test F1: {res['overall_test_macro_f1']:.4f}")
        if "overall_diag_macro_f1" in res:
            print(f"    Overall Diag F1: {res['overall_diag_macro_f1']:.4f}")

    # Save JSON summary
    out_json = os.path.join(args.out_dir, "tfidf_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Saved full results to {out_json}")

    # Generate LaTeX snippets for Table 7 and Table 10
    print("\n" + "="*80)
    print("READY-TO-PASTE LATEX SNIPPETS")
    print("="*80)
    print("\n--- TABLE 7 (tab:main-benchmark) ---")
    for name in ["TF--IDF + Logistic Regression", "TF--IDF + Linear SVM"]:
        r = all_results[name]
        print(f"{name} & {r['scripts']['bangla']:.4f} & {r['scripts']['banglish']:.4f} & {r['scripts']['code_mix']:.4f} & {r['scripts']['english']:.4f} & {r['over_refusal']:.2f} \\\\")

    print("\n--- TABLE 10 (tab:domain-benchmark) ---")
    for name in ["TF--IDF + Logistic Regression", "TF--IDF + Linear SVM"]:
        r = all_results[name]
        b_nat = r['domain_natural']['banking']
        h_nat = r['domain_natural']['healthcare']
        e_nat = r['domain_natural']['e-commerce']
        o_nat = r['overall_test_macro_f1']
        b_dia = r['domain_diag']['banking']
        h_dia = r['domain_diag']['healthcare']
        e_dia = r['domain_diag']['e-commerce']
        o_dia = r['overall_diag_macro_f1']
        print(f"{name} & {b_nat:.4f} & {h_nat:.4f} & {e_nat:.4f} & {o_nat:.4f} & {b_dia:.4f} & {h_dia:.4f} & {e_dia:.4f} & {o_dia:.4f} \\\\")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
