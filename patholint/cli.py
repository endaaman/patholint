import os
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
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

    class OllamaArgs(BaseModel):
        dir: str = param("data/reports", s="-d", l="--dir")
        outdir: str = param("out/results", s="-o", l="--outdir")
        model: str = param("gemma3:27b", s="-m", l="--model")
        prompt: str = param("zeroshot", s="-p", l="--prompt")

    def run_ollama(self, a: OllamaArgs):
        system_prompt = load_prompt(a.prompt)
        reports = Report.load_dir(a.dir)
        out = Path(a.outdir) / a.model.replace(":", "_")
        out.mkdir(parents=True, exist_ok=True)

        for i, r in enumerate(reports):
            out_path = out / f"{r.病理番号}.md"
            if out_path.exists():
                print(f"[{i+1}/{len(reports)}] {r.病理番号}: skip (exists)")
                continue

            body = (Path(a.dir) / f"{r.病理番号}.md").read_text()
            prompt = system_prompt + "\n\n" + body

            print(f"[{i+1}/{len(reports)}] {r.病理番号}: validating ({a.model})...", end=" ", flush=True)
            result = subprocess.run(
                ["ollama", "run", a.model],
                input=prompt,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"ERROR: {result.stderr.strip()}")
                continue

            out_path.write_text(result.stdout)
            print("done")


def main():
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
