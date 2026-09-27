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
uv run cli batch -m all -c all                           # 全モデル×全条件 (zeroshot+ruleset)
```

条件 (`-c`):

| 条件 | プロンプト | ruleset | 備考 |
|---|---|---|---|
| `zeroshot` | `prompts/instruction.md` | なし | 4分類 |
| `ruleset` | `prompts/instruction.md` | `data/kiyaku/crc_ruleset.md` | 4分類 |
| `fast` | `prompts/instruction_fast.md` | なし | Typo / Inconsistency のみ。nothink モデルと組み合わせる高速版 |
| `fast2` | `prompts/instruction_fast_typo.md` + `prompts/instruction_fast_inconsistency.md` | なし | fast を Typo 専用・Inconsistency 専用の2回呼び出しに分割し、出力を連結（tokens/duration は合算） |

`-c all` は従来どおり `zeroshot` + `ruleset` のみ（fast / fast2 は明示指定）。

#### 高速版 (fast): Qwen3.8-27B on DGX Spark

enda-spark の prism-gw (:4000) 経由で SGLang の Qwen3.8-27B (`qwen3.8-27b`) を使う。
`-nothink` は `chat_template_kwargs.enable_thinking=false` を付けて送る。
SGLang は prism-hu/chat 側で `docker compose --profile heavy up -d sglang-qwen38`（大物は同時に1つだけ）。
`.env` は `LITELLM_HOST=localhost` / `LITELLM_MASTER_KEY=<PRISM_GW_API_KEY>`。

```bash
uv run cli single -r 0001 -m qwen3.8-27b-nothink -c fast
uv run cli batch -m qwen3.8-27b-nothink -c fast
uv run cli score -m qwen3.8-27b-nothink -c fast
uv run cli tally -m qwen3.8-27b-nothink -c fast --by-tag
```

##### thinking ON + 思考長上限（推奨: `qwen3.8-27b-t1kx2` × `fast` × `-p 8`）

Qwen3.8-27B の thinking は上限なしだと 5k–16k+ tok 続き（10 分近く）、content 空で終わることもある。
サンプリングを推奨値にしても止まらないので、思考長に上限を掛ける派生エイリアスを用意している。

| モデル | 思考上限 | 備考 |
|---|---|---|
| `qwen3.8-27b-t1kx2` | 1024 tok × 2 サンプル | **推奨**。独立 2 サンプル（並列）の指摘の和集合 |
| `qwen3.8-27b-t1k` | 1024 tok | 最速側 |
| `qwen3.8-27b-t2k` | 2048 tok | t1kx2 と同コストだが 1 本なので揺れが大きい |
| `qwen3.8-27b-t4k` | 4096 tok | 遅い割に伸びない |
| `qwen3.8-27b-think` | なし | 参考用（実用にならない） |

全 50 件の実測（2026-09-28、enda-spark、sglang-qwen38 = DFLASH / max-running 16）:

| モデル × fast | 並列 | 実効 s/件 | 1件 中央値 | Inconsistency | Typo | FP rel/spu |
|---|---|---|---|---|---|---|
| `qwen3.8-27b-nothink` | 1 | 5.1 | 2.9s | 11/15 | 2/5 | 0.10/0.76 |
| `qwen3.8-27b-t1k` | 16 | 5.2 | 70s | 13/15 | 1/5 | 0.32/0.70 |
| `qwen3.8-27b-t2k` | 16 | 10.5 | 144s | 12/15 | 1/5 | 0.26/0.84 |
| **`qwen3.8-27b-t1kx2`** | 8 | 9.7 | 73s | **14/15** | 2/5 | 0.42/1.42 |

サンプリングがあるので 1 回ごとに Inconsistency が ±2 件揺れる（t2k は反復 6 回で平均 0.89）。
単発（並列なし）の所要時間は t1k ~32s、t2k ~58s。

- サンプリングは Qwen 推奨値（temperature 0.6 / top_p 0.95 / top_k 20）固定。`-t` は効かない
- 1回目を `max_tokens=budget` で生成し、思考が上限で切れたら（または思考中に終了したら）思考を閉じた
  assistant prefill（`continue_final_message`）で回答だけを生成させる。サーバ側の設定には依存しない
  （このとき meta に `forced_answer: true` が付く。実測では t1k–t4k のほぼ全件がこの経路）
- `batch -p N` で N 件を同時に投げる。SGLang の連続バッチングで総スループットが伸びるので、
  sglang-qwen38（`--max-running-requests=16`）には同時リクエストが 16 本になるように投げる
  （`t1kx2` は 1 件 2 本なので `-p 8`、それ以外は `-p 16`）

```bash
uv run cli batch -m qwen3.8-27b-t1kx2 -c fast -p 8
uv run cli score -m qwen3.8-27b-t1kx2 -c fast
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
