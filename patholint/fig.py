import json

import pandas as pd
import numpy as np
from pathlib import Path
from pydantic import BaseModel
from pydantic_autocli import AutoCLI, param

import matplotlib.pyplot as plt
import seaborn as sns


CONDITIONS = ["zeroshot", "ruleset"]
COND_PALETTE = {"zeroshot": "#7faadc", "ruleset": "#f4a582"}

MODEL_ORDER = [
    "claude-opus-4-6", "claude-sonnet-4-6", "deepseek-v3.2", "kimi-k2.6",
    "gpt-oss-120b", "gpt-oss-20b", "sip-jmed-13b",
]
SHORT_NAMES = {
    "claude-opus-4-6": "Opus",
    "claude-sonnet-4-6": "Sonnet",
    "deepseek-v3.2": "DeepSeek",
    "kimi-k2.6": "Kimi",
    "gpt-oss-120b": "GPT-120B",
    "gpt-oss-20b": "GPT-20B",
    "sip-jmed-13b": "JMed-13B",
}
SHORT_ORDER = [SHORT_NAMES[m] for m in MODEL_ORDER]

TAG_ORDER = ["RuleViolation", "Deficiency", "Inconsistency", "Typo"]
TAG_SHORT = {"RuleViolation": "RV", "Deficiency": "Def", "Inconsistency": "Inc", "Typo": "Typo"}
TAG_SHORT_ORDER = [TAG_SHORT[t] for t in TAG_ORDER]


def load_cases(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["model"] = df["model"].map(SHORT_NAMES).fillna(df["model"])
    df["model"] = pd.Categorical(df["model"], categories=SHORT_ORDER, ordered=True)
    det = df["detection"]
    df["tp"] = ((det == "tp-exact") | (det == "tp-content-only")).astype(int)
    df["tp_exact"] = (det == "tp-exact").astype(int)
    df["tp_content"] = (det == "tp-content-only").astype(int)
    df["is_error"] = (df["status"] == "error").astype(int)
    return df


def savefig(fig, outdir: Path, name: str):
    path = outdir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path}")


class CLI(AutoCLI):
    class DefaultArgs(BaseModel):
        cases: str = param("out/cases.csv", s="-i", l="--cases", description="per-case CSV (tally --outdir で生成)")
        resultdir: str = param("out/results", s="-r", l="--resultdir", description="results dir with _meta.jsonl")
        outdir: str = param("out/figs", s="-o", l="--outdir")

    def run_default(self, a: DefaultArgs):
        """全図を生成"""
        sns.set_theme(style="whitegrid", font_scale=1.1)
        plt.rcParams["figure.dpi"] = 150

        outdir = Path(a.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        df = load_cases(a.cases)

        self._plot_sensitivity(df, outdir)
        self._plot_sensitivity_delta(df, outdir)
        self._plot_detection_breakdown(df, outdir)
        self._plot_sensitivity_by_tag(df, outdir)
        self._plot_sensitivity_by_tag_comparison(df, outdir)
        self._plot_sensitivity_delta_heatmap(df, outdir)
        self._plot_fp(df, outdir)
        self._plot_fp_delta(df, outdir)
        self._plot_tp_exact_rate(df, outdir)
        self._plot_sensitivity_heatmap(df, outdir)
        self._plot_duration(df, outdir)
        self._plot_case_heatmap(df, outdir)
        self._plot_gpt_token_breakdown(Path(a.resultdir), outdir)

        print(f"\nAll figures saved to {outdir}/")

    # ============================================================
    # 1. Overall sensitivity
    # ============================================================
    def _plot_sensitivity(self, df, outdir):
        agg = df.groupby(["model", "condition"], observed=True)["tp"].mean().reset_index()
        agg.rename(columns={"tp": "sensitivity"}, inplace=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=agg, x="model", y="sensitivity", hue="condition",
                    palette=COND_PALETTE, ax=ax)
        ax.set_ylabel("Sensitivity")
        ax.set_xlabel("")
        ax.set_ylim(0, 1.05)
        ax.set_title("Overall Sensitivity by Model")
        ax.legend(title="Condition")
        for c in ax.containers:
            ax.bar_label(c, fmt="%.2f", fontsize=8, padding=2)
        savefig(fig, outdir, "overall_sensitivity")

    # ============================================================
    # 2. Sensitivity delta
    # ============================================================
    def _plot_sensitivity_delta(self, df, outdir):
        agg = df.groupby(["model", "condition"], observed=True)["tp"].mean().reset_index()
        agg.rename(columns={"tp": "sensitivity"}, inplace=True)
        pivot = agg.pivot(index="model", columns="condition", values="sensitivity").reset_index()
        pivot["delta"] = pivot["ruleset"] - pivot["zeroshot"]
        pivot = pivot.sort_values("model", key=lambda s: s.map({m: i for i, m in enumerate(SHORT_ORDER)}))

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#4daf4a" if d >= 0 else "#e41a1c" for d in pivot["delta"]]
        bars = ax.bar(pivot["model"], pivot["delta"], color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Δ Sensitivity (ruleset − zeroshot)")
        ax.set_xlabel("")
        ax.set_title("Sensitivity Improvement with Ruleset")
        ax.bar_label(bars, fmt="%+.2f", fontsize=9, padding=2)
        savefig(fig, outdir, "overall_sensitivity_delta")

    # ============================================================
    # 3. Detection breakdown
    # ============================================================
    def _plot_detection_breakdown(self, df, outdir):
        det_map = {
            "tp-exact": "TP-Exact", "tp-content-only": "TP-Content",
            "fn": "FN", "fn-clean": "FN-Clean",
        }
        df = df.copy()
        df["det_label"] = df["detection"].map(det_map).fillna("FN")
        df.loc[df["status"] == "error", "det_label"] = "Error"

        det_order = ["TP-Exact", "TP-Content", "FN", "FN-Clean", "Error"]
        det_colors = {
            "TP-Exact": "#2ca02c", "TP-Content": "#98df8a",
            "FN": "#d62728", "FN-Clean": "#ff9896", "Error": "#7f7f7f",
        }

        for cond in CONDITIONS:
            sub = df[df["condition"] == cond]
            ct = sub.groupby(["model", "det_label"], observed=True).size().unstack(fill_value=0)
            ct = ct.reindex(columns=[c for c in det_order if c in ct.columns], fill_value=0)

            fig, ax = plt.subplots(figsize=(10, 5))
            ct.plot.bar(stacked=True, color=[det_colors[c] for c in ct.columns], ax=ax)
            ax.set_ylabel("Count (n=50)")
            ax.set_xlabel("")
            ax.set_title(f"Detection Breakdown — {cond}")
            ax.legend(title="Detection", bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
            savefig(fig, outdir, f"detection_breakdown_{cond}")

    # ============================================================
    # 4. Sensitivity by tag (per condition)
    # ============================================================
    def _plot_sensitivity_by_tag(self, df, outdir):
        tag_df = df[df["gs_tag"].isin(TAG_ORDER)].copy()
        tag_df["gs_tag"] = pd.Categorical(tag_df["gs_tag"].map(TAG_SHORT),
                                           categories=TAG_SHORT_ORDER, ordered=True)
        tag_agg = tag_df.groupby(["model", "condition", "gs_tag"], observed=True)["tp"].mean().reset_index()
        tag_agg.rename(columns={"tp": "sensitivity"}, inplace=True)

        for cond in CONDITIONS:
            sub = tag_agg[tag_agg["condition"] == cond]
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=sub, x="gs_tag", y="sensitivity", hue="model",
                        palette="Set2", ax=ax)
            ax.set_ylabel("Sensitivity")
            ax.set_xlabel("Error Type")
            ax.set_ylim(0, 1.15)
            ax.set_title(f"Sensitivity by Error Type — {cond}")
            ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
            savefig(fig, outdir, f"sensitivity_by_tag_{cond}")

    # ============================================================
    # 5. Sensitivity by tag: zeroshot vs ruleset 4-panel
    # ============================================================
    def _plot_sensitivity_by_tag_comparison(self, df, outdir):
        tag_df = df[df["gs_tag"].isin(TAG_ORDER)].copy()
        tag_df["gs_tag"] = pd.Categorical(tag_df["gs_tag"].map(TAG_SHORT),
                                           categories=TAG_SHORT_ORDER, ordered=True)
        tag_agg = tag_df.groupby(["model", "condition", "gs_tag"], observed=True)["tp"].mean().reset_index()
        tag_agg.rename(columns={"tp": "sensitivity"}, inplace=True)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
        for i, tag in enumerate(TAG_ORDER):
            ax = axes[i]
            sub = tag_agg[tag_agg["gs_tag"] == TAG_SHORT[tag]]
            sns.barplot(data=sub, x="model", y="sensitivity", hue="condition",
                        palette=COND_PALETTE, ax=ax)
            ax.set_title(TAG_SHORT[tag])
            ax.set_ylim(0, 1.15)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=45)
            for label in ax.get_xticklabels():
                label.set_ha("right")
            ax.set_ylabel("Sensitivity" if i == 0 else "")
            if i == 3:
                ax.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
            else:
                ax.get_legend().remove()
        fig.suptitle("Sensitivity by Error Type: Zeroshot vs Ruleset", y=1.02)
        fig.tight_layout()
        savefig(fig, outdir, "sensitivity_by_tag_comparison")

    # ============================================================
    # 6. Sensitivity delta heatmap by tag
    # ============================================================
    def _plot_sensitivity_delta_heatmap(self, df, outdir):
        tag_df = df[df["gs_tag"].isin(TAG_ORDER)].copy()
        tag_df["gs_tag"] = pd.Categorical(tag_df["gs_tag"].map(TAG_SHORT),
                                           categories=TAG_SHORT_ORDER, ordered=True)
        tag_agg = tag_df.groupby(["model", "condition", "gs_tag"], observed=True)["tp"].mean().reset_index()
        tag_agg.rename(columns={"tp": "sensitivity"}, inplace=True)

        tag_pivot = tag_agg.pivot_table(index="model", columns=["gs_tag", "condition"],
                                         values="sensitivity")
        delta_data = {}
        for tag in TAG_SHORT_ORDER:
            if (tag, "ruleset") in tag_pivot.columns and (tag, "zeroshot") in tag_pivot.columns:
                delta_data[tag] = tag_pivot[(tag, "ruleset")] - tag_pivot[(tag, "zeroshot")]
        delta_df = pd.DataFrame(delta_data).reindex(SHORT_ORDER)

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(delta_df, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                    vmin=-0.3, vmax=0.6, linewidths=0.5, linecolor="white", ax=ax)
        ax.grid(False)
        ax.set_title("Δ Sensitivity (ruleset − zeroshot) by Error Type")
        ax.set_ylabel("")
        ax.set_xlabel("")
        savefig(fig, outdir, "sensitivity_delta_heatmap")

    # ============================================================
    # 7. FP comparison
    # ============================================================
    def _plot_fp(self, df, outdir):
        fp_agg = df.groupby(["model", "condition"], observed=True).agg(
            fp_relevant=("fp_relevant", "mean"),
            fp_spurious=("fp_spurious", "mean"),
        ).reset_index()
        fp_melt = fp_agg.melt(id_vars=["model", "condition"],
                               value_vars=["fp_relevant", "fp_spurious"],
                               var_name="fp_type", value_name="mean_count")

        for cond in CONDITIONS:
            sub = fp_melt[fp_melt["condition"] == cond]
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.barplot(data=sub, x="model", y="mean_count", hue="fp_type",
                        palette={"fp_relevant": "#fdae61", "fp_spurious": "#d73027"}, saturation=1, ax=ax)
            ax.set_ylabel("Mean FP Count per Case")
            ax.set_xlabel("")
            ax.set_title(f"False Positives — {cond}")
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles, ["Relevant", "Spurious"], title="FP Type")
            for c in ax.containers:
                ax.bar_label(c, fmt="%.1f", fontsize=8, padding=2)
            savefig(fig, outdir, f"fp_comparison_{cond}")

    # ============================================================
    # 8. FP delta
    # ============================================================
    def _plot_fp_delta(self, df, outdir):
        fp_agg = df.groupby(["model", "condition"], observed=True).agg(
            fp_relevant=("fp_relevant", "mean"),
            fp_spurious=("fp_spurious", "mean"),
        ).reset_index()
        fp_pivot = fp_agg.pivot_table(index="model", columns="condition",
                                       values=["fp_relevant", "fp_spurious"])
        fp_delta = pd.DataFrame({
            "FP-Relevant": fp_pivot[("fp_relevant", "ruleset")] - fp_pivot[("fp_relevant", "zeroshot")],
            "FP-Spurious": fp_pivot[("fp_spurious", "ruleset")] - fp_pivot[("fp_spurious", "zeroshot")],
        }).reindex(SHORT_ORDER)

        fig, ax = plt.subplots(figsize=(9, 5))
        x = range(len(fp_delta))
        w = 0.35
        bars1 = ax.bar([i - w/2 for i in x], fp_delta["FP-Relevant"], w,
                        label="FP-Relevant", color="#fdae61")
        bars2 = ax.bar([i + w/2 for i in x], fp_delta["FP-Spurious"], w,
                        label="FP-Spurious", color="#d73027")
        ax.set_xticks(list(x))
        ax.set_xticklabels(fp_delta.index, rotation=30, ha="right")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Δ Mean FP (ruleset − zeroshot)")
        ax.set_title("FP Change with Ruleset")
        ax.legend()
        ax.bar_label(bars1, fmt="%+.1f", fontsize=8, padding=2)
        ax.bar_label(bars2, fmt="%+.1f", fontsize=8, padding=2)
        savefig(fig, outdir, "fp_delta")

    # ============================================================
    # 9. TP-Exact rate
    # ============================================================
    def _plot_tp_exact_rate(self, df, outdir):
        tp_df = df[df["tp"] == 1].copy()
        if len(tp_df) == 0:
            return
        exact_agg = tp_df.groupby(["model", "condition"], observed=True)["tp_exact"].mean().reset_index()
        exact_agg.rename(columns={"tp_exact": "exact_rate"}, inplace=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=exact_agg, x="model", y="exact_rate", hue="condition",
                    palette=COND_PALETTE, ax=ax)
        ax.set_ylabel("Tag Accuracy (Exact / TP)")
        ax.set_xlabel("")
        ax.set_ylim(0, 1.15)
        ax.set_title("Tag Accuracy among True Positives")
        ax.legend(title="Condition")
        for c in ax.containers:
            ax.bar_label(c, fmt="%.2f", fontsize=8, padding=2)
        savefig(fig, outdir, "tp_exact_rate")

    # ============================================================
    # 10. Sensitivity heatmap
    # ============================================================
    def _plot_sensitivity_heatmap(self, df, outdir):
        agg = df.groupby(["model", "condition"], observed=True)["tp"].mean().reset_index()
        agg.rename(columns={"tp": "sensitivity"}, inplace=True)
        sens_pivot = agg.pivot(index="model", columns="condition", values="sensitivity")
        sens_pivot = sens_pivot.reindex(SHORT_ORDER)[["zeroshot", "ruleset"]]

        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(sens_pivot, annot=True, fmt=".2f", cmap="YlGn",
                    vmin=0, vmax=1, linewidths=0.5, linecolor="white", ax=ax)
        ax.grid(False)
        ax.set_title("Sensitivity Overview")
        ax.set_ylabel("")
        ax.set_xlabel("")
        savefig(fig, outdir, "sensitivity_heatmap")

    # ============================================================
    # 11. Duration boxplot
    # ============================================================
    def _plot_duration(self, df, outdir):
        dur_df = df[df["duration_s"].notna()].copy()
        if len(dur_df) == 0:
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=dur_df, x="model", y="duration_s", hue="condition",
                    palette=COND_PALETTE, ax=ax)
        ax.set_yscale("log")
        ax.set_ylabel("Duration (s, log scale)")
        ax.set_xlabel("")
        ax.set_title("Response Time per Case")
        ax.legend(title="Condition")
        savefig(fig, outdir, "duration_boxplot")

    # ============================================================
    # 12. Case-level heatmap (gene expression style)
    # ============================================================
    def _plot_case_heatmap(self, df, outdir):
        from matplotlib.colors import ListedColormap

        det_val = {
            "tp-exact": 2, "tp-content-only": 1,
            "fn": -1, "fn-clean": -1,
        }
        df = df.copy()
        df["det_val"] = df["detection"].map(det_val).fillna(-1)
        df.loc[df["status"] == "error", "det_val"] = -2

        # columns: model × condition
        col_order = []
        for m in SHORT_ORDER:
            for c in CONDITIONS:
                col_order.append(f"{m}\n{c}")

        cases = sorted(df["case"].unique())
        mat = np.full((len(cases), len(col_order)), np.nan)
        for i, case in enumerate(cases):
            for j, col in enumerate(col_order):
                model, cond = col.split("\n")
                row = df[(df["model"] == model) & (df["condition"] == cond) & (df["case"] == case)]
                if len(row) == 1:
                    mat[i, j] = row.iloc[0]["det_val"]

        # tag labels for y-axis
        case_tags = []
        for case in cases:
            tag = df[df["case"] == case]["gs_tag"].iloc[0]
            case_tags.append(f"{case:04d} [{TAG_SHORT.get(tag, tag)}]")

        cmap = ListedColormap(["#7f7f7f", "#d62728", "#98df8a", "#2ca02c"])
        bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

        fig, ax = plt.subplots(figsize=(14, 16))
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-2.5, vmax=2.5, interpolation="nearest")
        ax.set_xticks(range(len(col_order)))
        ax.set_xticklabels(col_order, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(len(cases)))
        ax.set_yticklabels(case_tags, fontsize=7, fontfamily="monospace")
        ax.set_xlabel("")
        ax.set_title("Per-Case Detection Results")

        # cell boundary grid (box-style)
        ax.grid(False)
        ax.set_xticks([i - 0.5 for i in range(len(col_order) + 1)], minor=True)
        ax.set_yticks([i - 0.5 for i in range(len(cases) + 1)], minor=True)
        ax.grid(which="minor", color="white", linewidth=0.3)
        ax.tick_params(which="minor", length=0)

        # thicker vertical separators between models
        for k in range(1, len(SHORT_ORDER)):
            ax.axvline(k * 2 - 0.5, color="white", linewidth=2.5)

        # legend
        from matplotlib.patches import Patch
        legend_items = [
            Patch(color="#2ca02c", label="TP-Exact"),
            Patch(color="#98df8a", label="TP-Content"),
            Patch(color="#d62728", label="FN"),
            Patch(color="#7f7f7f", label="Error"),
        ]
        ax.legend(handles=legend_items, loc="upper right", bbox_to_anchor=(1.15, 1))

        savefig(fig, outdir, "case_heatmap")

    # ============================================================
    # 12. GPT-20B vs 120B token breakdown
    # ============================================================
    def _plot_gpt_token_breakdown(self, resultdir: Path, outdir):
        import re as re2

        models = ["gpt-oss-20b", "gpt-oss-120b"]
        rows = []
        for model in models:
            for cond in CONDITIONS:
                model_dir = resultdir / cond / model
                if not model_dir.exists():
                    continue
                for fpath in sorted(model_dir.glob("[0-9]*.md")):
                    text = fpath.read_text()
                    cm = re2.search(r"completion_tokens:\s*(\d+)", text)
                    comp = int(cm.group(1)) if cm else 0
                    rows.append({
                        "model": SHORT_NAMES[model],
                        "condition": cond,
                        "case": fpath.stem,
                        "completion_tokens": comp,
                    })
        if not rows:
            return

        tok_df = pd.DataFrame(rows)
        # cases.csv has correct duration
        cases = pd.read_csv(resultdir.parent / "cases.csv")
        cases["model"] = cases["model"].map(SHORT_NAMES).fillna(cases["model"])
        cases["case"] = cases["case"].astype(str).str.zfill(4)
        tok_df = tok_df.merge(
            cases[["model", "condition", "case", "duration_s"]],
            on=["model", "condition", "case"], how="left",
        )
        tok_df = tok_df[tok_df["model"].isin(["GPT-20B", "GPT-120B"])]
        tok_df["tps"] = tok_df["completion_tokens"] / tok_df["duration_s"]

        agg = tok_df.groupby(["model", "condition"]).agg(
            tokens=("completion_tokens", "mean"),
            duration=("duration_s", "mean"),
            tps=("tps", "mean"),
        ).reset_index()
        # Sort: 120B first, then 20B; within each, zeroshot then ruleset
        model_rank = {"GPT-120B": 0, "GPT-20B": 1}
        cond_rank = {"zeroshot": 0, "ruleset": 1}
        agg = agg.sort_values(
            by=["model", "condition"],
            key=lambda s: s.map(model_rank) if s.name == "model" else s.map(cond_rank),
        ).reset_index(drop=True)

        MODEL_PALETTE = {"GPT-120B": "#6baed6", "GPT-20B": "#fd8d3c"}
        labels = [f"{r['model']}\n{r['condition']}" for _, r in agg.iterrows()]
        x = np.arange(len(labels))
        colors = [MODEL_PALETTE[r["model"]] for _, r in agg.iterrows()]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))

        # 1: tok/s — スループットは同程度
        bars_tps = ax1.bar(x, agg["tps"], color=colors)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=9)
        ax1.set_ylabel("Tokens / sec")
        ax1.set_title("Throughput")
        ax1.bar_label(bars_tps, fmt="%.0f", fontsize=9, padding=2)

        # 2: output volume — 20Bは出力量が3-4倍
        tok_std = tok_df.groupby(["model", "condition"])["completion_tokens"].std().reset_index()
        tok_std = tok_std.sort_values(
            by=["model", "condition"],
            key=lambda s: s.map(model_rank) if s.name == "model" else s.map(cond_rank),
        ).reset_index(drop=True)
        bars_tok = ax2.bar(x, agg["tokens"], yerr=tok_std["completion_tokens"],
                           capsize=4, color=colors, error_kw={"lw": 1.2})
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=9)
        ax2.set_ylabel("Mean Output Tokens")
        ax2.set_title("Output Volume")
        ax2.bar_label(bars_tok, fmt="%.0f", fontsize=9, padding=3)

        # 3: duration — 結果、処理時間が逆転
        bars_dur = ax3.bar(x, agg["duration"], color=colors)
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, fontsize=9)
        ax3.set_ylabel("Mean Duration (s)")
        ax3.set_title("Duration")
        ax3.bar_label(bars_dur, fmt="%.0f", fontsize=9, padding=2)

        # legend
        from matplotlib.patches import Patch
        handles = [Patch(color=c, label=m) for m, c in MODEL_PALETTE.items()]
        ax3.legend(handles=handles, title="Model")

        fig.suptitle("GPT-20B vs GPT-120B: Token Breakdown",
                     fontsize=12, y=1.02)
        fig.tight_layout()
        savefig(fig, outdir, "gpt_token_breakdown")


def main():
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
