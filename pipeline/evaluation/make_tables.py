"""
Generate the LaTeX row blocks of the manuscript's result tables from results/tables/*.csv.
Each block is written to results/tables/latex_rows/<label>.tex and pasted into paper/MDE.tex between
the markers %%TABROWS_BEGIN:<label> and %%TABROWS_END:<label> by paste_tables.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
import numpy as np
import pandas as pd
from config import TABLES

OUT = TABLES / "latex_rows"
OUT.mkdir(parents=True, exist_ok=True)
DATASETS = ["NSL-KDD", "CICIDS-2017", "CICIDS-2018", "UNSW-NB15"]
SHORT = {"NSL-KDD": "NSL-KDD", "CICIDS-2017": "CIC-2017", "CICIDS-2018": "CIC-2018", "UNSW-NB15": "UNSW-NB15"}


def f4(x):
    return "n/a" if pd.isna(x) else f"{x:.4f}"


def f3(x):
    return "n/a" if pd.isna(x) else f"{x:.3f}"


def bold(s):
    return "\\textbf{" + s + "}"


def write(label, lines):
    (OUT / f"{label}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{label}: {len(lines)} lines")


def exists(name):
    return (TABLES / name).exists()


# ── ablation and fullmetrics ────────────────────────────────────────────────────────────
if exists("ablation_cv.csv"):
    ab = pd.read_csv(TABLES / "ablation_cv.csv")
    lines, lines_full = [], []
    for i, ds in enumerate(DATASETS):
        d = ab[ab.dataset == ds]
        if d.empty:
            continue
        nf = {c: int(d[d.ablation == c].n_feat.iloc[0]) for c in ["conventional", "entropy_only", "combined"]}
        best_full = d[d.ablation == "combined"].f1.max()
        for j, model in enumerate(["LightGBM", "RandomForest"]):
            r = {c: d[(d.model == model) & (d.ablation == c)].iloc[0] for c in ["conventional", "entropy_only", "combined"]}
            best = max(r[c].f1 for c in r)
            cells = []
            for c in ["conventional", "entropy_only", "combined"]:
                f1 = f4(r[c].f1)
                cells += [bold(f1) if r[c].f1 == best else f1, f4(r[c].prec), f4(r[c].rec)]
            first = ds if j == 0 else ""
            nfcell = f"{nf['conventional']}/{nf['entropy_only']}/{nf['combined']}" if j == 0 else ""
            lines.append(f"{first:<11} & {model:<12} & {nfcell:<8} & " + " & ".join(cells) + " \\\\")
            rc = r["combined"]
            f1 = f4(rc.f1)
            counts = " & ".join(f"{int(rc[k]):,}" for k in ("tp", "fn", "fp", "tn")) if "tp" in rc else ""
            lines_full.append(f"{first:<11} & {model:<12} & {bold(f1) if rc.f1 == best_full else f1} & {f4(rc.prec)} & {f4(rc.rec)} & "
                              f"{f4(rc.dr)} & {f4(rc.far)} & {f4(rc.acc)} & {f4(rc.mcc)} & {f4(rc.auc)} & {f4(rc.prauc)}" + (f" & {counts}" if counts else "") + " \\\\")
        if i < len(DATASETS) - 1:
            lines.append("\\midrule"); lines_full.append("\\midrule")
    write("ablation", lines); write("fullmetrics", lines_full)

# ── time split ──────────────────────────────────────────────────────────────────────────
if exists("cicids2017_timesplit.csv"):
    ts = pd.read_csv(TABLES / "cicids2017_timesplit.csv")
    cols = ["f1", "prec", "rec", "dr", "far", "mcc", "auc", "prauc"]
    best = {c: (ts[c].min() if c == "far" else ts[c].max()) for c in cols}
    lines = []
    for j, model in enumerate(["LightGBM", "RandomForest"]):
        for k, abl in enumerate(["conventional", "entropy_only", "combined"]):
            r = ts[(ts.model == model) & (ts.ablation == abl)].iloc[0]
            cells = [bold(f4(r[c])) if r[c] == best[c] else f4(r[c]) for c in cols]
            lines.append(f"{model if k == 0 else '':<12} & {abl.replace('_', chr(92) + '_'):<13} & " + " & ".join(cells) + " \\\\")
        if j == 0:
            lines.append("\\midrule")
    write("timesplit", lines)

# ── temporal replay summary ─────────────────────────────────────────────────────────────
if exists("temporal_replay_summary.csv"):
    rp = pd.read_csv(TABLES / "temporal_replay_summary.csv")
    lines = []
    best_auc = rp.fixed_auc.max()
    for abl, name in [("conventional", "Conventional"), ("entropy_only", "Entropy-only"), ("combined", "Combined")]:
        r = rp[rp.ablation == abl].iloc[0]
        auc = f3(r.fixed_auc)
        lines.append(f"{name:<12} & {f3(r.thr_youden)} & {f3(r.fixed_dr)} & {f3(r.youden_dr)} & {f3(r.fixed_far)} & {f3(r.fixed_mcc)} & "
                     f"{bold(auc) if r.fixed_auc == best_auc else auc} \\\\")
    write("replay", lines)

# ── classifier comparison ───────────────────────────────────────────────────────────────
if exists("baselines_comparison.csv"):
    bc = pd.read_csv(TABLES / "baselines_comparison.csv")
    order = ["LightGBM", "Random Forest", "XGBoost", "CatBoost", "MLP", "TabNet", "FT-Transformer"]
    lines = []
    for i, ds in enumerate(DATASETS):
        d = bc[bc.dataset == ds]
        if d.empty:
            continue
        best = d.f1.max()
        for j, model in enumerate(order):
            m = d[d.model == model]
            if m.empty:
                continue
            r = m.iloc[0]
            name = model
            f1 = f4(r.f1)
            lines.append(f"{ds if j == 0 else '':<11} & {name:<24} & {bold(f1) if r.f1 == best else f1} & {f4(r.prec)} & {f4(r.rec)} & "
                         f"{f4(r.dr)} & {f4(r.far)} & {f4(r.acc)} & {f4(r.mcc)} & {f4(r.prauc)} \\\\")
        if i < len(DATASETS) - 1:
            lines.append("\\midrule")
    write("inpipeline", lines)

# ── per-class detection ─────────────────────────────────────────────────────────────────
if exists("perclass_nslkdd.csv") and exists("perclass_cicids18.csv") and exists("perclass_timesplit.csv"):
    lines = []

    def block(title, df, benign_names):
        out = []
        att = df[~df.category.astype(str).isin(benign_names)]
        ben = df[df.category.astype(str).isin(benign_names)]
        first = True
        if len(att) == 1:
            r = att.iloc[0]
            out.append(f"{title:<20} & Attack (binary) & {int(r.n_test):,} & {f3(r.DR)} & n/a \\\\"); first = False
        else:
            for _, r in att.sort_values("n_test", ascending=False).iterrows():
                out.append(f"{title if first else '':<20} & {r.category} & {int(r.n_test):,} & {f3(r.DR)} & n/a \\\\"); first = False
        for _, r in ben.iterrows():
            out.append(f"{'':<20} & Benign & {int(r.n_test):,} & n/a & {f3(r.FAR)} \\\\")
        return out

    lines += block("NSL-KDD (random)", pd.read_csv(TABLES / "perclass_nslkdd.csv"), ["0", "Benign", "normal", "BENIGN"])
    lines.append("\\midrule")
    lines += block("CICIDS-2018 (random)", pd.read_csv(TABLES / "perclass_cicids18.csv"), ["0", "Benign", "BENIGN"])
    lines.append("\\midrule")
    lines += block("CIC-2017 (temporal)", pd.read_csv(TABLES / "perclass_timesplit.csv"), ["BENIGN", "Benign"])
    write("perclass", lines)

# ── unseen families ─────────────────────────────────────────────────────────────────────
if exists("unseen_attack_eval.csv"):
    un = pd.read_csv(TABLES / "unseen_attack_eval.csv")
    lines = []
    for j, model in enumerate(["LightGBM", "RandomForest"]):
        for k, abl in enumerate(["conventional", "entropy_only", "combined"]):
            r = un[(un.model == model) & (un.ablation == abl)].iloc[0]
            lines.append(f"{model if k == 0 else '':<12} & {abl.replace('_', chr(92) + '_'):<13} & {f4(r.f1)} & {f4(r.prec)} & {f4(r.rec)} & "
                         f"{f4(r.auc)} & {f4(r.dr_attack)} & {f4(r.far_benign)} \\\\")
        if j == 0:
            lines.append("\\midrule")
    write("unseen", lines)

# ── SHAP stability ──────────────────────────────────────────────────────────────────────
if exists("shap_fold_stability.csv"):
    sh = pd.read_csv(TABLES / "shap_fold_stability.csv")
    lines = []
    for _, r in sh.iterrows():
        lines.append(f"{r.dataset:<11} & ${r.spearman_all_mean:.3f} \\pm {r.spearman_all_std:.3f}$ & ${r.spearman_mde_mean:.3f} \\pm {r.spearman_mde_std:.3f}$ & "
                     f"${r.kendall_all_mean:.3f} \\pm {r.kendall_all_std:.3f}$ & {r.mde_rank_std_mean:.2f} \\\\")
    write("shapstability", lines)

# ── transfer, robustness, simple statistics, profiling ──────────────────────────────────
if exists("transfer.csv"):
    tr = pd.read_csv(TABLES / "transfer.csv")
    lines = []
    for i, src in enumerate(["CICIDS-2017", "CICIDS-2018"]):
        d = tr[tr.source == src]
        for kind in ["hold-out", "transfer"]:
            r = d[d.kind == kind].iloc[0]
            lines.append(f"{SHORT[src]} & {SHORT[r.target]} ({kind}) & {f3(r.f1)} & {f3(r.auc)} \\\\")
        if i == 0:
            lines.append("\\midrule")
    write("transfer", lines)

if exists("robustness.csv"):
    rb = pd.read_csv(TABLES / "robustness.csv")
    lines = []
    for ds in DATASETS:
        d = rb[rb.dataset == ds].sort_values("eps")
        if d.empty:
            continue
        for j, (col, label) in enumerate([("f1", "F1"), ("dr", "DR"), ("auc", "AUC")]):
            lines.append(f"{SHORT[ds] if j == 0 else '':<9} & {label:<3} & " + " & ".join(f3(v) for v in d[col]) + " \\\\")
        if ds != [x for x in DATASETS if x in set(rb.dataset)][-1]:
            lines.append("\\midrule")
    write("robustness", lines)

if exists("simplestats.csv"):
    ss = pd.read_csv(TABLES / "simplestats.csv")
    lines = []
    dss = [ds for ds in DATASETS if ds in set(ss.dataset)]

    def delta(v, ref):
        d = v - ref
        return "$<$0.001" if abs(d) < 0.0005 else (("$-$" if d < 0 else "$+$") + f"{abs(d):.3f}")
    for i, ds in enumerate(dss):
        rs = ss[(ss.dataset == ds) & (ss.feature_set == "simple_statistics")].iloc[0]
        lines.append(f"{SHORT[ds]:<9} & Simple statistics  & {int(rs.n_feat):2d} & {f3(rs.f1)} & {f3(rs.dr)} & {f3(rs.far)} & \\\\")
        for key, label in [("entropy_only", "MDE, all features"), ("entropy_strict", "MDE, entropy terms")]:
            m = ss[(ss.dataset == ds) & (ss.feature_set == key)]
            if m.empty:
                continue
            r = m.iloc[0]
            lines.append(f"{'':<9} & {label:<18} & {int(r.n_feat):2d} & {f3(r.f1)} & {f3(r.dr)} & {f3(r.far)} & {delta(r.f1, rs.f1)} \\\\")
        if i < len(dss) - 1:
            lines.append("\\midrule")
    write("simplestats", lines)

if exists("profiling.csv"):
    pf = pd.read_csv(TABLES / "profiling.csv")
    lines = []
    for i, ds in enumerate(DATASETS):
        d = pf[pf.dataset == ds]
        for j, model in enumerate(["LightGBM", "RandomForest"]):
            r = d[d.model == model].iloc[0]
            split = f" & {r.feature_us:5.2f} & {r.classifier_us:5.2f}" if "feature_us" in pf.columns else ""
            lines.append(f"{SHORT[ds] if j == 0 else '':<9} & {model:<12} & {int(r.n_train):,} & {r.train_s:6.2f} & {r.infer_us:5.2f}{split} \\\\")
        if i < len(DATASETS) - 1:
            lines.append("\\midrule")
    write("profiling", lines)
