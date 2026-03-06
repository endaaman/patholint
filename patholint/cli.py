import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import ollama
import pandas as pd
import yaml
from pydantic_autocli import AutoCLI, param
from pydantic import BaseModel

from patholint.models import Report

PROJ_ROOT = Path(__file__).resolve().parent.parent


def serialize_value(val):
    if pd.isna(val):
        return None
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.strftime("%Y-%m-%d")
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def load_prompt(name: str) -> str:
    path = PROJ_ROOT / "prompts" / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text().strip()


def resolve_report(report: str) -> Path:
    """レポートIDまたはパスからファイルパスを解決"""
    p = Path(report)
    if p.exists():
        return p
    # IDとして data/reports/<id>.md を探す
    p = PROJ_ROOT / "data" / "reports" / f"{report}.md"
    if p.exists():
        return p
    raise FileNotFoundError(f"Report not found: {report}")


def load_report_body(path: Path) -> str:
    """レポートファイルから<findings>と<diagnosis>を抽出して返す"""
    import re
    text = path.read_text()
    parts = []
    for tag in ["findings", "diagnosis"]:
        m = re.search(rf"(<{tag}>.*?</{tag}>)", text, re.DOTALL)
        if m:
            parts.append(m.group(1))
    if not parts:
        raise ValueError(f"<所見> or <診断> not found in {path}")
    return "\n\n".join(parts)


def estimate_tokens(text: str) -> int:
    """日本語混在テキストのトークン数を雑に推定"""
    return int(len(text) * 1.5)


PRIVATE_COLS = {"フリガナ", "氏名", "生年月日"}


class CLI(AutoCLI):
    class ConvertArgs(BaseModel):
        input: str = param("data/raw.xlsx", s="-i", l="--input")
        outdir: str = param("data/reports", s="-o", l="--outdir")
        last: bool = param(False, s="-l", l="--last")
        raw: bool = param(False, s="-r", l="--raw")
        filter: str = param("", s="-f", l="--filter")

    def run_convert(self, a: ConvertArgs):
        df = pd.read_excel(a.input)
        os.makedirs(a.outdir, exist_ok=True)

        # 病理番号を前方補完（セル結合によるNaN対策）
        df["病理番号"] = df["病理番号"].ffill()

        body_cols = {"コメント&診断::病理組織所見", "コメント&診断::病理組織診断"}
        diagnosis_cols = {"コメント&診断::checker", "コメント&診断::診断医師", "コメント&診断::診断年月日"}
        patient_cols = [c for c in df.columns if c not in body_cols and c not in diagnosis_cols]

        count = 0
        skipped = 0
        for pathology_id, group in df.groupby("病理番号", sort=False):
            if pd.isna(pathology_id):
                continue

            first = group.iloc[0]
            target = group.iloc[-1] if a.last else first

            # フィルター（所見に特定文字列を含むもののみ）
            if a.filter:
                text = target["コメント&診断::病理組織所見"]
                if pd.isna(text) or a.filter not in str(text):
                    skipped += 1
                    continue

            # 患者情報（常に最初の行から）
            meta = {}
            for col in patient_cols:
                if not a.raw and col in PRIVATE_COLS:
                    continue
                val = serialize_value(first[col])
                if val is not None:
                    meta[col] = val

            # 診断情報（選択した行から）
            for col in diagnosis_cols:
                val = serialize_value(target[col])
                if val is not None:
                    key = col.replace("コメント&診断::", "")
                    meta[key] = val

            # 本文（選択した行から）
            body_parts = []
            for tag, col in [("所見", "コメント&診断::病理組織所見"), ("診断", "コメント&診断::病理組織診断")]:
                text = target[col]
                if pd.notna(text):
                    body_parts.append(f"<{tag}>\n{str(text).strip()}\n</{tag}>")
            body = "\n\n".join(body_parts)

            content = "---\n"
            content += yaml.dump(meta, allow_unicode=True, default_flow_style=False, sort_keys=False)
            content += "---\n\n"
            content += body + "\n"

            path = os.path.join(a.outdir, f"{pathology_id}.md")
            with open(path, "w") as f:
                f.write(content)
            count += 1

        if a.filter:
            print(f"{count} files written to {a.outdir}/ ({skipped} skipped by filter)")
        else:
            print(f"{count} files written to {a.outdir}/")

    class LoadArgs(BaseModel):
        dir: str = param("data/reports", s="-d", l="--dir")

    def run_load(self, a: LoadArgs):
        reports = Report.load_dir(a.dir)
        for r in reports:
            print(f"{r.病理番号}: {r.氏名} ({r.臨床診断})")
        print(f"\n{len(reports)} reports loaded")

    class ValidateArgs(BaseModel):
        dir: str = param("data/reports", s="-d", l="--dir")
        outdir: str = param("out/results", s="-o", l="--outdir")
        model: str = param("sonnet", s="-m", l="--model")
        prompt: str = param("zeroshot", s="-p", l="--prompt")

    def run_validate(self, a: ValidateArgs):
        system_prompt = load_prompt(a.prompt)
        reports = Report.load_dir(a.dir)
        out = Path(a.outdir) / a.model
        out.mkdir(parents=True, exist_ok=True)

        for i, r in enumerate(reports):
            out_path = out / f"{r.病理番号}.md"
            if out_path.exists():
                print(f"[{i+1}/{len(reports)}] {r.病理番号}: skip (exists)")
                continue

            body = (Path(a.dir) / f"{r.病理番号}.md").read_text()
            prompt = system_prompt + "\n\n" + body

            print(f"[{i+1}/{len(reports)}] {r.病理番号}: validating...", end=" ", flush=True)
            env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
            result = subprocess.run(
                ["claude", "-p", "--model", a.model],
                input=prompt,
                capture_output=True,
                text=True,
                env=env,
            )
            if result.returncode != 0:
                print(f"ERROR: {result.stderr.strip()}")
                continue

            out_path.write_text(result.stdout)
            print("done")

    class SingleArgs(BaseModel):
        report: str = param(..., s="-r", l="--report", description="レポートID (例: 23-0845) またはファイルパス")
        model: str = param("gpt-oss:20b", s="-m", l="--model")
        ruleset: bool = param(False, l="--ruleset", description="ルールセット(kiyaku_crc)を含める")
        outdir: str = param("out/results", s="-o", l="--outdir")
        temperature: float = param(0.3, s="-t", l="--temperature")
        think: str = param("low", l="--think", description="thinking level: low/medium/high/false")
        num_predict: int = param(4096, l="--num-predict")

    def run_single(self, a: SingleArgs):
        report_path = resolve_report(a.report)
        report_id = report_path.stem
        body = load_report_body(report_path)

        # プロンプト組み立て
        parts = [load_prompt("instruction")]
        if a.ruleset:
            ruleset_path = PROJ_ROOT / "data" / "kiyaku_crc_ruleset.md"
            parts.append(ruleset_path.read_text().strip())
        parts.append(body)
        prompt = "\n\n".join(parts)

        # コンテキストサイズ自動計算
        estimated = estimate_tokens(prompt)
        num_ctx = estimated + a.num_predict + 2048
        condition = "with_ruleset" if a.ruleset else "zeroshot"

        print(f"Report: {report_id}", file=sys.stderr)
        print(f"Model: {a.model}", file=sys.stderr)
        print(f"Condition: {condition}", file=sys.stderr)
        print(f"Context: {num_ctx} tokens (prompt ~{estimated})", file=sys.stderr)
        print(f"Generating...", file=sys.stderr, flush=True)

        res = ollama.chat(
            model=a.model,
            messages=[{"role": "user", "content": prompt}],
            think=False if a.think == "false" else a.think,
            options={
                "temperature": a.temperature,
                "num_ctx": num_ctx,
                "num_predict": a.num_predict,
                "repeat_penalty": 1.0,
                "frequency_penalty": 0.3,
            },
        )

        answer = (res.message.content or "").strip()
        prompt_tokens = res.prompt_eval_count or 0
        completion_tokens = res.eval_count or 0
        duration = (res.total_duration or 0) / 1e9
        thinking = (getattr(res.message, "thinking", None) or "").strip()

        # 出力先: out/results/<condition>/<model>/<id>.md
        model_dir = a.model.replace(":", "_")
        out_dir = Path(a.outdir) / condition / model_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{report_id}.md"
        out_path.write_text(answer + "\n")

        if thinking:
            print(f"Thinking: {len(thinking)} chars", file=sys.stderr)
        print(f"Tokens: prompt={prompt_tokens}, completion={completion_tokens}", file=sys.stderr)
        print(f"Duration: {duration:.1f}s", file=sys.stderr)
        print(f"Output: {out_path}", file=sys.stderr)
        if not answer:
            print("WARNING: empty response (num_predict may be too low for thinking model)", file=sys.stderr)
        print(answer)


def main():
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
