import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from pydantic_autocli import AutoCLI, param

from patholint.models import Report

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    # model_name: (host, port, description)
    "claude-opus-4-6":    ("litellm", None, "Claude Opus 4.6"),
    "claude-sonnet-4-6":  ("litellm", None, "Claude Sonnet 4.6"),
    "deepseek-v3.2":      ("deepseek", 8000, "DeepSeek V3.2"),
    "gpt-oss-20b":        ("litellm", None, "GPT-OSS 20B"),
    "gpt-oss-120b":       ("litellm", None, "GPT-OSS 120B"),
    "sip-jmed-13b":       ("litellm", None, "SIP-JMed 13B"),
    "sip-jmed-8x13b-q8":  ("litellm", None, "SIP-JMed 8x13B Q8"),
    "nemotron-3-nano":    ("litellm", None, "Nemotron-3 Nano"),
    "nemotron-3-super":   ("litellm", None, "Nemotron-3 Super"),
    "qwen3.5-9b":         ("litellm", None, "Qwen 3.5 9B"),
    "qwen3.5-27b":        ("litellm", None, "Qwen 3.5 27B"),
}

CONDITIONS = ["zeroshot", "ruleset"]


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
        raise ValueError(f"<findings> or <diagnosis> not found in {path}")
    return "\n\n".join(parts)


def load_gold_standard(path: Path) -> str:
    """レポートファイルから<gold_standard>を抽出して返す"""
    import re
    text = path.read_text()
    m = re.search(r"(<gold_standard>.*?</gold_standard>)", text, re.DOTALL)
    return m.group(1) if m else ""


def estimate_tokens(text: str) -> int:
    """日本語混在テキストのトークン数を雑に推定"""
    return int(len(text) * 1.5)


def create_client(model: str) -> OpenAI:
    """モデル名に応じたOpenAI clientを作成。"""
    info = MODELS.get(model)
    if not info:
        raise ValueError(f"Unknown model: {model} (available: {', '.join(MODELS.keys())})")
    host_key, port, _ = info
    if host_key == "litellm":
        host = os.environ.get("LITELLM_HOST", "prism-spark")
        key = os.environ.get("LITELLM_MASTER_KEY", "")
        return OpenAI(
            base_url=f"http://{host}:4000/v1",
            api_key=key,
            timeout=600,
        )
    elif host_key == "deepseek":
        host = os.environ.get("DEEPSEEK_HOST", "prism-llens")
        return OpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key="none",
            timeout=600,
        )
    else:
        raise ValueError(f"Unknown host_key: {host_key}")


def build_messages(body: str, condition: str) -> list[dict]:
    """system/user メッセージを組み立てる"""
    system_parts = [load_prompt("instruction")]
    if condition == "ruleset":
        ruleset_path = PROJ_ROOT / "data" / "kiyaku" / "crc_ruleset.md"
        system_parts.append(ruleset_path.read_text().strip())
    return [
        {"role": "system", "content": "\n\n".join(system_parts)},
        {"role": "user", "content": body},
    ]


def call_llm(client: OpenAI, model: str, messages: list[dict], temperature: float) -> dict:
    """LLM呼び出しを実行し、結果を辞書で返す（streaming）"""
    t0 = time.time()
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=32768,
        stream=True,
        stream_options={"include_usage": True},
    )

    chunks = []
    finish_reason = None
    usage = None
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                chunks.append(delta.content)
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

    duration = time.time() - t0
    answer = "".join(chunks).strip()

    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    thinking_tokens = None
    if usage:
        raw_usage = usage.model_dump()
        thinking_tokens = raw_usage.get("reasoning_tokens") or (raw_usage.get("completion_tokens_details") or {}).get("reasoning_tokens")

    # 空応答デバッグ
    if not answer and completion_tokens > 0:
        print(f"WARNING: empty content despite {completion_tokens} completion tokens", file=sys.stderr)
        print(f"  finish_reason: {finish_reason}", file=sys.stderr)

    return {
        "answer": answer,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "thinking_tokens": thinking_tokens,
        "duration_s": round(duration, 2),
    }


def run_one(client: OpenAI, report_path: Path, model: str, condition: str,
            outdir: str, temperature: float, force: bool = False) -> dict | None:
    """1件のレポートをLLMで検証し、結果をファイル保存。スキップ時はNone返却。"""
    report_id = report_path.stem
    out_dir = Path(outdir) / condition / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report_id}.md"

    if out_path.exists() and not force:
        # 空応答のファイルは再試行
        existing = out_path.read_text()
        if "<invalidities>\n\n</invalidities>" not in existing:
            return None  # skip

    body = load_report_body(report_path)
    gold = load_gold_standard(report_path)
    messages = build_messages(body, condition)
    result = call_llm(client, model, messages, temperature)

    answer = result["answer"]
    meta_header = (
        f"model: {model}\n"
        f"condition: {condition}\n"
        f"finish_reason: {result['finish_reason']}\n"
        f"prompt_tokens: {result['prompt_tokens']}\n"
        f"completion_tokens: {result['completion_tokens']}\n"
        f"duration_s: {result['duration_s']}\n"
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}"
    )
    parts = [
        f"---\n{meta_header}\n---",
        body,
        f"<invalidities>\n{answer}\n</invalidities>",
    ]
    if gold:
        parts.append(gold)
    out_path.write_text("\n\n".join(parts) + "\n")

    meta = {
        "report_id": report_id,
        "finish_reason": result["finish_reason"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "thinking_tokens": result["thinking_tokens"],
        "duration_s": result["duration_s"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = out_dir / "_meta.jsonl"
    with open(meta_path, "a") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    return result


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

    class SingleArgs(BaseModel):
        report: str = param(..., s="-r", l="--report", description="レポートID (例: 0001) またはファイルパス")
        model: str = param("gpt-oss-20b", s="-m", l="--model")
        ruleset: bool = param(False, l="--ruleset", description="ルールセット(kiyaku_crc)を含める")
        outdir: str = param("out/results", s="-o", l="--outdir")
        temperature: float = param(0.3, s="-t", l="--temperature")
        force: bool = param(False, l="--force", description="既存結果を上書き")

    def run_single(self, a: SingleArgs):
        report_path = resolve_report(a.report)
        report_id = report_path.stem
        condition = "ruleset" if a.ruleset else "zeroshot"

        body = load_report_body(report_path)
        messages = build_messages(body, condition)
        system_tokens = estimate_tokens(messages[0]["content"])
        user_tokens = estimate_tokens(messages[1]["content"])

        print(f"Report: {report_id}")
        print(f"Model: {a.model}")
        print(f"Condition: {condition}")
        print(f"Tokens (est): system ~{system_tokens}, user ~{user_tokens}")
        print(f"Generating...", flush=True)

        client = create_client(a.model)
        result = run_one(client, report_path, a.model, condition, a.outdir, a.temperature, a.force)

        if result is None:
            out_path = Path(a.outdir) / condition / a.model / f"{report_id}.md"
            print(f"Skip (exists): {out_path}")
            return

        out_path = Path(a.outdir) / condition / a.model / f"{report_id}.md"
        tokens_info = f"prompt={result['prompt_tokens']}, completion={result['completion_tokens']}"
        if result["thinking_tokens"]:
            tokens_info += f", thinking={result['thinking_tokens']}"
        print(f"Tokens: {tokens_info}")
        print(f"Finish: {result['finish_reason']}")
        print(f"Duration: {result['duration_s']:.1f}s")
        print(f"Output: {out_path}")
        if result["finish_reason"] == "length":
            print("WARNING: output truncated (hit max_tokens)")
        if not result["answer"]:
            print("WARNING: empty response")

    class BatchArgs(BaseModel):
        model: str = param("all", s="-m", l="--model", description="モデル名 or 'all'")
        condition: str = param("all", s="-c", l="--condition", description="zeroshot | ruleset | all")
        dir: str = param("data/reports", s="-d", l="--dir")
        outdir: str = param("out/results", s="-o", l="--outdir")
        temperature: float = param(0.3, s="-t", l="--temperature")
        force: bool = param(False, l="--force", description="既存結果を上書き")
        dry_run: bool = param(False, l="--dry-run", description="API呼び出しせず件数とプロンプトサイズを表示")

    def run_batch(self, a: BatchArgs):
        models = list(MODELS.keys()) if a.model == "all" else [a.model]
        conditions = CONDITIONS if a.condition == "all" else [a.condition]

        # レポートファイル一覧
        report_dir = Path(a.dir)
        report_files = sorted(report_dir.glob("*.md"))
        if not report_files:
            print(f"No reports found in {a.dir}", file=sys.stderr)
            return False

        # dry-run: プロンプトサイズと件数を表示
        if a.dry_run:
            sample_body = load_report_body(report_files[0])
            for cond in conditions:
                msgs = build_messages(sample_body, cond)
                sys_tokens = estimate_tokens(msgs[0]["content"])
                user_tokens = estimate_tokens(msgs[1]["content"])
                print(f"[{cond}] system ~{sys_tokens} tokens, user ~{user_tokens} tokens (sample: {report_files[0].stem})")
            for m in models:
                for cond in conditions:
                    out_dir = Path(a.outdir) / cond / m
                    existing = sum(1 for f in report_files if (out_dir / f"{f.stem}.md").exists()) if out_dir.exists() else 0
                    remaining = len(report_files) - existing if not a.force else len(report_files)
                    print(f"  {m}/{cond}: {remaining} to run ({existing} existing)")
            total = len(models) * len(conditions) * len(report_files)
            print(f"\nTotal: {len(report_files)} reports x {len(models)} models x {len(conditions)} conditions = {total} calls")
            return

        for m in models:
            client = create_client(m)
            for cond in conditions:
                total = len(report_files)
                skipped = 0
                done = 0
                errors = 0

                for i, rpath in enumerate(report_files):
                    report_id = rpath.stem
                    prefix = f"[{m}/{cond}] [{i+1}/{total}] {report_id}"

                    print(f"{prefix}: ", end="", flush=True)

                    try:
                        result = run_one(client, rpath, m, cond, a.outdir, a.temperature, a.force)
                    except Exception as e:
                        errors += 1
                        print(f"ERROR: {e}")
                        continue

                    if result is None:
                        skipped += 1
                        print("skip (exists)")
                    else:
                        done += 1
                        tokens_info = f"{result['prompt_tokens']}+{result['completion_tokens']}"
                        if result["thinking_tokens"]:
                            tokens_info += f"(think:{result['thinking_tokens']})"
                        extra = ""
                        if result["finish_reason"] == "length":
                            extra = " TRUNCATED"
                        elif not result["answer"]:
                            extra = " EMPTY"
                        print(f"done ({tokens_info} tokens, {result['duration_s']:.1f}s){extra}")

                print(f"[{m}/{cond}] Finished: {done} done, {skipped} skipped, {errors} errors")

    class TestArgs(BaseModel):
        model: str = param("gpt-oss-20b", s="-m", l="--model")
        verbose: bool = param(False, s="-v", l="--verbose", description="生レスポンスを表示")

    def run_test(self, a: TestArgs):
        """疎通テスト"""
        info = MODELS.get(a.model)
        host_desc = f"{info[0]}:{info[1]}" if info else "unknown"
        print(f"Host: {host_desc}")
        client = create_client(a.model)
        print(f"Model: {a.model}")
        print(f"Sending test message...", flush=True)

        try:
            t0 = time.time()
            res = client.chat.completions.create(
                model=a.model,
                messages=[{"role": "user", "content": "Hello, respond with OK."}],
                max_tokens=32,
            )
            duration = time.time() - t0
        except Exception as e:
            print(f"ERROR: {e}")
            return False

        choice = res.choices[0]
        content = choice.message.content or ""
        usage = res.usage

        if a.verbose:
            print(f"\n--- Raw response ---")
            print(json.dumps(res.model_dump(), indent=2, ensure_ascii=False, default=str))
            print(f"--- End ---\n")

        print(f"Content: {content.strip()!r}")
        print(f"Finish reason: {choice.finish_reason}")
        print(f"Tokens: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}" if usage else "Tokens: N/A")
        print(f"Duration: {duration:.1f}s")
        print(f"OK" if content.strip() else "WARNING: empty response")

    class ModelsArgs(BaseModel):
        pass

    def run_models(self, a: ModelsArgs):
        print("Available models:")
        for name, (host, port, desc) in MODELS.items():
            loc = f"{host}:{port}" if port else host
            print(f"  {name:25s} {loc:20s} {desc}")


def main():
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
