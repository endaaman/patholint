# patholint

LLMで病理報告書のテキスト品質を検証する実験システム。

## セットアップ

```bash
uv sync
```

## コマンド

### cli: メインCLI

```bash
uv run cli --help           # コマンド一覧
uv run cli models           # 利用可能なモデル一覧
uv run cli test -m gpt-oss-20b  # 疎通テスト
```

#### バリデーション実行

```bash
uv run cli single -r 0001 -m claude-opus-4-6 --ruleset  # 単発
uv run cli batch -m all -c all                           # 全モデル×全条件
```

#### スコアリング

```bash
uv run cli score -m all -c all          # claude CLIで採点
uv run cli score-status                 # 採点進捗確認
```

#### 集計・CSV出力

```bash
uv run cli tally                        # 集計テーブル表示
uv run cli tally --by-tag               # GSタグ別の内訳付き
uv run cli tally --csv out/tally.csv    # 集計CSV出力
uv run cli tally -o out                 # per-case CSV + duration stats 出力
```

### fig: 図の生成

`uv run cli tally -o out` で `out/cases.csv` を生成してから実行。

```bash
uv run fig                              # out/cases.csv → out/figs/ に全図生成
uv run fig -i out/cases.csv -o out/figs # 入出力を明示指定
```

出力される図:

| ファイル | 内容 |
|---|---|
| `overall_sensitivity.png` | 全体sensitivity（モデル×条件） |
| `overall_sensitivity_delta.png` | ruleset効果（Δ sensitivity） |
| `detection_breakdown_{cond}.png` | 検出内訳・積上げ棒 |
| `sensitivity_by_tag_{cond}.png` | タグ別sensitivity |
| `sensitivity_by_tag_comparison.png` | タグ別 zeroshot vs ruleset 4パネル |
| `sensitivity_delta_heatmap.png` | Δ sensitivityヒートマップ（モデル×タグ） |
| `fp_comparison_{cond}.png` | FP内訳 |
| `fp_delta.png` | FP変化量（ruleset−zeroshot） |
| `tp_exact_rate.png` | タグ正確率（Exact/TP） |
| `sensitivity_heatmap.png` | sensitivity一覧ヒートマップ |
| `duration_boxplot.png` | 処理時間boxplot |
