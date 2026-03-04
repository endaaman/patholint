## patholint - 病理レポートバリデーションシステム

LLMで病理報告書のテキスト品質を検証する。規約（圧縮済みルールセット）をプロンプトに含める条件と含めない条件で比較実験を行う。修正は行わず、Validation（問題の検出・提示）に責務を限定する。

### Invalidity 4分類（排他）

| 分類 | 定義 | 検出に必要なもの |
|---|---|---|
| **RuleViolation** | 記載が存在し、ルールセットに照らして値・形式が不正 | ルールセット参照 |
| **Deficiency** | 必須項目の記載がレポート内に存在しない | ルールセット参照 |
| **Inconsistency** | 個々はルール上合法だが、複数箇所を突き合わせると矛盾 | レポート内部の関係推論 |
| **Typo** | スペルミス・転記ミスなど、記載自体から判定可能 | 記載自体 |

## 開発ルール

- uvを必ず使う
- 実行単位はcliコマンドとして作成
- 再利用データは data/ 出力は out/ を使用

## AutoCLI Usage

- `def run_foo_bar(self, a: FooBarArgs)` → `script.py foo-bar`
- `def run_default(self, a: DefaultArgs)` → `script.py` (no subcommand)
- `class CommonArgs` → shared arguments across all commands
- `def prepare(self, a: CommonArgs)` → runs before every command
- Return `True`/`None` (exit 0), `False` (exit 1), `int` (custom exit code)

For details: `script.py --help` or `script.py <command> --help`
