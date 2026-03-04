## 病理レポートバリデーションシステム

- 規約をプロンプトにセットしたり、しなかったりして、LLMでレポートのバリデーションする
- ルール:
  - **Error**: 言語的エラー、規約に存在項目（pT5など）
  - **Deficiency**: 規約記載が必要なのに漏れている（PM DMの断端記載が無いなど）
  - **Inconcistency**: 記述内矛盾（pT3なのに粘膜内癌と所見記載）。それぞれ単体では規約違反ではないが、相反するもの
- このようなルール


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
