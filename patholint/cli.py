import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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

# ---- モデル定義 ----
# エイリアスごとの設定はすべて ModelSpec にまとめる（送信先・実モデル名・extra_body・サンプリング・
# max_tokens・思考長上限・サンプル数）。リクエストの組み立ては request_kwargs() に集約。

# nothink はループに陥りがちなので低めに、thinking 用は思考分の余裕を残す
DEFAULT_MAX_TOKENS = 65536
NOTHINK_MAX_TOKENS = 8192
# 思考を閉じた後の回答分（思考上限つき派生の2回目の呼び出しの max_tokens）
BUDGET_ANSWER_TOKENS = 4096

# サンプリングの指定（ModelSpec.sampling）
# - SAMPLING_CLI: -t/--temperature を temperature として送る。未指定なら LEGACY_TEMPERATURE（0.3）。
#   既存モデルはこれ（過去の結果と条件を揃えるため）
# - None: 何も送らずサーバ既定に任せる（SGLang は generation_config 由来の値）。
#   -t を明示したときだけ temperature を送る
# - dict: 固定値。-t は無視する。temperature / top_p / presence_penalty / frequency_penalty は
#   OpenAI の引数、それ以外（top_k など）は extra_body で送る
SAMPLING_CLI = "cli"
LEGACY_TEMPERATURE = 0.3
OPENAI_SAMPLING_KEYS = {"temperature", "top_p", "presence_penalty", "frequency_penalty"}


@dataclass(frozen=True)
class ModelSpec:
    description: str
    upstream: str                           # 実際に送るモデル名
    host: str = "litellm"                   # litellm (prism-gw :4000) | dedicated
    port: int | None = None
    extra_body: dict = field(default_factory=dict)
    sampling: dict | str | None = SAMPLING_CLI
    max_tokens: int = DEFAULT_MAX_TOKENS    # 1回目の呼び出しの max_tokens
    # 思考長の上限（クライアント側）。1回目を max_tokens=think_budget で生成し、length で切れたら
    # 思考を閉じた assistant prefill（continue_final_message）で回答だけを生成させる
    think_budget: int | None = None
    # 思考が上限で切れた / 思考中に終了して content が空のとき、prefill で回答を出させる
    force_answer: bool = False
    # 独立に N 回サンプルし（並列）、指摘の和集合を回答にする
    samples: int = 1


# Claude の extended thinking は litellm 経由で extra_body.thinking で指定。
# Anthropic API 仕様で temperature=1.0 強制、max_tokens > budget_tokens 必須。
CLAUDE_THINKING_BUDGET = 8000
CLAUDE_THINK = dict(
    extra_body={"thinking": {"type": "enabled", "budget_tokens": CLAUDE_THINKING_BUDGET}},
    sampling={"temperature": 1.0},
)
# vLLM/SGLang 系の chat template の thinking 切替。
# Kimi-K2.6 の chat_template.jinja は `thinking` キーを参照
# （`thinking is false` で <think></think> 空タグを挿入して reasoning を抑止）
DEDICATED = dict(host="dedicated", port=8000)

# Qwen3.8-27B（enda-spark ローカル。prism-gw :4000 → prism-hu/chat の sglang-qwen38, SGLang + DFlash2）
# plain / nothink はサーバ既定のサンプリング（sampling_defaults=model → generation_config の
# temperature 1.0 / top_p 0.95 / top_k 20）。以前は CLI 既定の temperature 0.3 を送っていた
# （2026-09-30 まで。これが思考暴走の一因）。
# thinking 派生は Qwen 推奨サンプリング固定 + 思考長上限。サンプリングを直しても思考は
# 5k–16k+ tok 続き、content 空で終わることもあるので、上限をクライアント側で掛ける
QWEN38 = "qwen3.8-27b"
QWEN38_NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
QWEN38_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
# Qwen 推奨の presence_penalty 1.5 は入れていない。現行の sglang-qwen38（DFLASH v2 + overlap schedule）は
# penalty 系を投機デコード中に落とすので送っても効かない（2026-09-30 確認）


def qwen38_think(description: str, budget: int | None, samples: int = 1,
                 sampling: dict = QWEN38_SAMPLING) -> ModelSpec:
    return ModelSpec(
        description, QWEN38, sampling=sampling,
        max_tokens=budget or DEFAULT_MAX_TOKENS, think_budget=budget,
        force_answer=True, samples=samples,
    )


MODELS: dict[str, ModelSpec] = {
    "claude-opus-4-6":         ModelSpec("Claude Opus 4.6", "claude-opus-4-6"),
    "claude-opus-4-6-think":   ModelSpec("Claude Opus 4.6 (extended thinking)", "claude-opus-4-6", **CLAUDE_THINK),
    "claude-sonnet-4-6":       ModelSpec("Claude Sonnet 4.6", "claude-sonnet-4-6"),
    "claude-sonnet-4-6-think": ModelSpec("Claude Sonnet 4.6 (extended thinking)", "claude-sonnet-4-6", **CLAUDE_THINK),
    "deepseek-v3.2":         ModelSpec("DeepSeek V3.2", "deepseek-v3.2", **DEDICATED,
                                       extra_body={"chat_template_kwargs": {"thinking": True}}),
    "deepseek-v3.2-nothink": ModelSpec("DeepSeek V3.2 (no think)", "deepseek-v3.2", **DEDICATED,
                                       max_tokens=NOTHINK_MAX_TOKENS),
    "kimi-k2.6":             ModelSpec("Kimi K2.6", "kimi-k2.6", **DEDICATED),
    "kimi-k2.6-nothink":     ModelSpec("Kimi K2.6 (no think)", "kimi-k2.6", **DEDICATED,
                                       extra_body={"chat_template_kwargs": {"thinking": False}},
                                       max_tokens=NOTHINK_MAX_TOKENS),
    "glm-5.1":               ModelSpec("GLM-5.1 FP8", "glm-5.1", **DEDICATED),
    "glm-5.1-nothink":       ModelSpec("GLM-5.1 FP8 (no think)", "glm-5.1", **DEDICATED,
                                       extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                                       max_tokens=NOTHINK_MAX_TOKENS),
    "gpt-oss-20b":        ModelSpec("GPT-OSS 20B", "gpt-oss-20b"),
    "gpt-oss-120b":       ModelSpec("GPT-OSS 120B", "gpt-oss-120b"),
    "sip-jmed-13b":       ModelSpec("SIP-JMed 13B", "sip-jmed-13b"),
    "sip-jmed-8x13b-q8":  ModelSpec("SIP-JMed 8x13B Q8", "sip-jmed-8x13b-q8"),
    "nemotron-3-nano":    ModelSpec("Nemotron-3 Nano", "nemotron-3-nano"),
    "nemotron-3-super":   ModelSpec("Nemotron-3 Super", "nemotron-3-super"),
    "qwen3.5-9b":         ModelSpec("Qwen 3.5 9B", "qwen3.5-9b"),
    "qwen3.5-27b":        ModelSpec("Qwen 3.5 27B", "qwen3.5-27b"),
    "qwen3.8-27b":         ModelSpec("Qwen3.8 27B (SGLang, Spark)", QWEN38, sampling=None),
    "qwen3.8-27b-nothink": ModelSpec("Qwen3.8 27B (SGLang, Spark, no think)", QWEN38, sampling=None,
                                     extra_body=QWEN38_NOTHINK, max_tokens=NOTHINK_MAX_TOKENS),
    "qwen3.8-27b-think":   qwen38_think("Qwen3.8 27B (SGLang, Spark, think, 推奨サンプリング, 上限なし)", None),
    "qwen3.8-27b-t1k":     qwen38_think("Qwen3.8 27B (SGLang, Spark, think budget 1k)", 1024),
    "qwen3.8-27b-t2k":     qwen38_think("Qwen3.8 27B (SGLang, Spark, think budget 2k)", 2048),
    "qwen3.8-27b-t4k":     qwen38_think("Qwen3.8 27B (SGLang, Spark, think budget 4k)", 4096),
    "qwen3.8-27b-t1kx2":   qwen38_think("Qwen3.8 27B (SGLang, Spark, think budget 1k x 2 samples, union)", 1024, samples=2),
}


def get_spec(model: str) -> ModelSpec:
    spec = MODELS.get(model)
    if spec is None:
        raise ValueError(f"Unknown model: {model} (available: {', '.join(MODELS.keys())})")
    return spec


def resolve_sampling(spec: ModelSpec, temperature: float) -> dict:
    """送るサンプリング引数（top_k 等も含むフラットな dict）。temperature < 0 は -t 未指定"""
    if spec.sampling == SAMPLING_CLI:
        return {"temperature": temperature if temperature >= 0 else LEGACY_TEMPERATURE}
    if spec.sampling is None:
        return {"temperature": temperature} if temperature >= 0 else {}
    return dict(spec.sampling)


def describe_sampling(spec: ModelSpec) -> str:
    if spec.sampling == SAMPLING_CLI:
        return f"-t (default {LEGACY_TEMPERATURE})"
    if spec.sampling is None:
        return "server default"
    return " ".join(f"{k}={v}" for k, v in spec.sampling.items())


def request_kwargs(model: str, messages: list[dict], temperature: float, stream: bool = True) -> dict:
    """1回目の呼び出しの chat.completions.create 引数"""
    spec = get_spec(model)
    kwargs = {
        "model": spec.upstream,
        "messages": messages,
        "max_tokens": spec.max_tokens,
    }
    if stream:
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
    extra_body = dict(spec.extra_body)
    for k, v in resolve_sampling(spec, temperature).items():
        if k in OPENAI_SAMPLING_KEYS:
            kwargs[k] = v
        else:
            extra_body[k] = v
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs

# `-c all` で回す条件（ruleset 有無の比較実験）。fast は別系統なので明示指定のみ
CONDITIONS = ["zeroshot", "ruleset"]
# condition: (instruction prompt 名のリスト, ruleset を含めるか)
# プロンプトが複数ある条件は1件につきプロンプトごとに呼び出し、出力を連結する
# fast: Typo/Inconsistency 限定の短いプロンプト、ruleset なし（nothink モデルと組み合わせて高速化）
# fast2: fast を Typo 専用 + Inconsistency 専用の2回呼び出しに分割
#        （nothink だと1回の呼び出しでは片方の種類しか挙げない傾向があるため）
CONDITION_PROMPTS = {
    "zeroshot": (["instruction"], False),
    "ruleset":  (["instruction"], True),
    "fast":     (["instruction_fast"], False),
    "fast2":    (["instruction_fast_typo", "instruction_fast_inconsistency"], False),
}
ALL_CONDITIONS = list(CONDITION_PROMPTS.keys())


def resolve_conditions(condition: str) -> list[str]:
    if condition == "all":
        return CONDITIONS
    if condition not in CONDITION_PROMPTS:
        raise ValueError(f"Unknown condition: {condition} (available: {', '.join(ALL_CONDITIONS)}, all)")
    return [condition]


# score / score-status / tally の対象モデル
SCORING_MODELS = [
    "claude-opus-4-6", "claude-opus-4-6-think",
    "claude-sonnet-4-6", "claude-sonnet-4-6-think",
    "deepseek-v3.2", "deepseek-v3.2-nothink",
    "kimi-k2.6", "kimi-k2.6-nothink", "glm-5.1", "glm-5.1-nothink",
    "gpt-oss-120b", "gpt-oss-20b", "sip-jmed-13b",
    "qwen3.8-27b", "qwen3.8-27b-nothink",
    "qwen3.8-27b-think", "qwen3.8-27b-t1k", "qwen3.8-27b-t2k", "qwen3.8-27b-t4k", "qwen3.8-27b-t1kx2",
]
assert all(m in MODELS for m in SCORING_MODELS)


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
    spec = get_spec(model)
    host_key, port = spec.host, spec.port
    if host_key == "litellm":
        host = os.environ.get("LITELLM_HOST", "prism-spark")
        key = os.environ.get("LITELLM_MASTER_KEY", "")
        return OpenAI(
            base_url=f"http://{host}:4000/v1",
            api_key=key,
            timeout=600,
        )
    elif host_key == "dedicated":
        host = os.environ.get("DEDICATED_HOST", "prism-llens")
        return OpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key="none",
            timeout=600,
        )
    else:
        raise ValueError(f"Unknown host_key: {host_key}")


def build_messages(body: str, condition: str) -> list[list[dict]]:
    """system/user メッセージを組み立てる（呼び出し1回分ずつのリスト）"""
    prompt_names, use_ruleset = CONDITION_PROMPTS[condition]
    message_sets = []
    for prompt_name in prompt_names:
        system_parts = [load_prompt(prompt_name)]
        if use_ruleset:
            ruleset_path = PROJ_ROOT / "data" / "kiyaku" / "crc_ruleset.md"
            system_parts.append(ruleset_path.read_text().strip())
        message_sets.append([
            {"role": "system", "content": "\n\n".join(system_parts)},
            {"role": "user", "content": body},
        ])
    return message_sets


def estimate_message_tokens(message_sets: list[list[dict]]) -> tuple[int, int]:
    """(system, user) の推定トークン数。複数呼び出しは合算"""
    system_tokens = sum(estimate_tokens(m[0]["content"]) for m in message_sets)
    user_tokens = sum(estimate_tokens(m[1]["content"]) for m in message_sets)
    return system_tokens, user_tokens


def is_no_problem(answer: str) -> bool:
    return answer.strip().rstrip("。") == "問題なし"


def merge_results(results: list[dict]) -> dict:
    """複数呼び出しの結果を1件分にまとめる。tokens/duration は合算。
    「問題なし」は全呼び出しが問題なしのときだけ残す"""
    if len(results) == 1:
        return results[0]
    answers = [r["answer"] for r in results if r["answer"]]
    findings = [a for a in answers if not is_no_problem(a)]
    if findings:
        answer = "\n".join(findings)
    elif answers:
        answer = "問題なし"
    else:
        answer = ""
    finish_reasons = [r["finish_reason"] for r in results]
    thinking = [r["thinking_tokens"] for r in results if r["thinking_tokens"]]
    return {
        "answer": answer,
        "finish_reason": "length" if "length" in finish_reasons else finish_reasons[-1],
        "prompt_tokens": sum(r["prompt_tokens"] for r in results),
        "completion_tokens": sum(r["completion_tokens"] for r in results),
        "thinking_tokens": sum(thinking) if thinking else None,
        "duration_s": round(sum(r["duration_s"] for r in results), 2),
        **({"forced_answer": True} if any(r.get("forced_answer") for r in results) else {}),
    }


def _stream_chat(client: OpenAI, kwargs: dict) -> dict:
    """streaming で1回呼び出し、content / reasoning_content を分けて集める。
    途中で stream を閉じても prism-gw 越しでは上流が止まらない（生成が裏で続く）ので、
    長さの制御は必ず max_tokens で行うこと"""
    stream = client.chat.completions.create(**kwargs)
    content, reasoning = [], []
    finish_reason = None
    usage = None
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                content.append(delta.content)
            rc = getattr(delta, "reasoning_content", None)
            if rc:
                reasoning.append(rc)
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
    return {
        "content": "".join(content).strip(),
        "reasoning": "".join(reasoning),
        "finish_reason": finish_reason,
        "usage": usage,
    }


def _usage_tokens(usage) -> tuple[int, int, int | None]:
    if not usage:
        return 0, 0, None
    raw_usage = usage.model_dump()
    thinking = raw_usage.get("reasoning_tokens") or (raw_usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    return usage.prompt_tokens, usage.completion_tokens, thinking


def _answer_prefill(reasoning: str, content: str, truncated: bool) -> str:
    """思考を閉じて回答を続けさせるための assistant prefill（chat template の `<think>\n` の後ろに付く）"""
    if content:
        return reasoning.rstrip() + "\n</think>\n\n" + content
    if truncated:
        # 思考が行の途中で切れているので、最後の改行までに丸めてから閉じる
        cut = reasoning.rfind("\n")
        if cut > len(reasoning) // 2:
            reasoning = reasoning[:cut]
        reasoning = reasoning.rstrip() + "\n\n（思考の上限に達したので、ここまでの検討に基づいて回答する）"
    return reasoning.rstrip() + "\n</think>\n\n"


def _call_llm_once(client: OpenAI, model: str, messages: list[dict], temperature: float) -> dict:
    """LLM呼び出しを1回実行し、結果を辞書で返す（streaming）"""
    t0 = time.time()
    spec = get_spec(model)
    kwargs = request_kwargs(model, messages, temperature)
    extra_body = kwargs.get("extra_body", {})

    r = _stream_chat(client, kwargs)
    prompt_tokens, completion_tokens, thinking_tokens = _usage_tokens(r["usage"])
    finish_reason = r["finish_reason"]
    answer = r["content"]
    forced = False

    # force_answer: 思考上限で切れた、または思考中に終了して content が空なら、
    # 思考を閉じた assistant prefill で回答（の続き）を生成させる
    truncated = spec.think_budget is not None and finish_reason == "length"
    if spec.force_answer and r["reasoning"] and (truncated or not answer):
        kwargs2 = dict(kwargs)
        kwargs2["messages"] = messages + [
            {"role": "assistant", "content": _answer_prefill(r["reasoning"], answer, truncated)}
        ]
        kwargs2["max_tokens"] = BUDGET_ANSWER_TOKENS
        kwargs2["extra_body"] = {**extra_body, "continue_final_message": True}
        if not answer:
            # 思考によっては </think> 直後に EOS を出して空になる（並列時に 1–3/20 件）。
            # 空回答は正解になり得ない（最低でも「問題なし」）ので EOS を最初の数トークン禁止する
            kwargs2["extra_body"]["min_tokens"] = 2
        # それでも空なら1回だけ引き直す
        for _ in range(2):
            r2 = _stream_chat(client, kwargs2)
            _, c2, _ = _usage_tokens(r2["usage"])
            completion_tokens += c2
            if r2["content"] or answer:
                break
        answer = (answer + r2["content"]).strip() if answer else r2["content"]
        finish_reason = r2["finish_reason"]
        forced = not r["content"]

    duration = time.time() - t0

    # 空応答デバッグ
    if not answer and completion_tokens > 0:
        print(f"WARNING: empty content despite {completion_tokens} completion tokens", file=sys.stderr)
        print(f"  finish_reason: {finish_reason}", file=sys.stderr)

    result = {
        "answer": answer,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "thinking_tokens": thinking_tokens,
        "duration_s": round(duration, 2),
    }
    if forced:
        result["forced_answer"] = True
    return result


def union_answers(answers: list[str]) -> str:
    """複数サンプルの回答を行単位で和集合にする（完全一致の重複は除く）。全部「問題なし」なら「問題なし」"""
    lines = []
    for a in answers:
        for line in a.splitlines():
            line = line.strip()
            if line and not is_no_problem(line) and line not in lines:
                lines.append(line)
    if lines:
        return "\n".join(lines)
    return "問題なし" if any(a.strip() for a in answers) else ""


def call_llm(client: OpenAI, model: str, messages: list[dict], temperature: float) -> dict:
    """LLM呼び出しを実行し、結果を辞書で返す。samples > 1 のモデルは N 並列サンプルの和集合"""
    n = get_spec(model).samples
    if n <= 1:
        return _call_llm_once(client, model, messages, temperature)
    t0 = time.time()
    with ThreadPoolExecutor(n) as ex:
        results = list(ex.map(lambda _: _call_llm_once(client, model, messages, temperature), range(n)))
    thinking = [r["thinking_tokens"] for r in results if r["thinking_tokens"]]
    finish_reasons = [r["finish_reason"] for r in results]
    result = {
        "answer": union_answers([r["answer"] for r in results]),
        "finish_reason": "length" if "length" in finish_reasons else finish_reasons[-1],
        "prompt_tokens": sum(r["prompt_tokens"] for r in results),
        "completion_tokens": sum(r["completion_tokens"] for r in results),
        "thinking_tokens": sum(thinking) if thinking else None,
        # 並列なので所要時間は壁時計（最も遅いサンプル）
        "duration_s": round(time.time() - t0, 2),
    }
    if any(r.get("forced_answer") for r in results):
        result["forced_answer"] = True
    return result


# batch --parallel で _meta.jsonl への追記が混ざらないように
_META_LOCK = threading.Lock()


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
    message_sets = build_messages(body, condition)
    result = merge_results([call_llm(client, model, m, temperature) for m in message_sets])

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
    if result.get("forced_answer"):
        meta["forced_answer"] = True
    meta_path = out_dir / "_meta.jsonl"
    with _META_LOCK, open(meta_path, "a") as f:
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
        ruleset: bool = param(False, l="--ruleset", description="ルールセット(kiyaku_crc)を含める (= -c ruleset)")
        condition: str = param("", s="-c", l="--condition", description="zeroshot | ruleset | fast | fast2 (未指定なら --ruleset で決定)")
        outdir: str = param("out/results", s="-o", l="--outdir")
        temperature: float = param(-1.0, s="-t", l="--temperature", description="未指定(-1)はモデル既定: 既存モデルは 0.3、サーバ既定のモデルは送らない、固定サンプリングのモデルでは無視")
        force: bool = param(False, l="--force", description="既存結果を上書き")

    def run_single(self, a: SingleArgs):
        report_path = resolve_report(a.report)
        report_id = report_path.stem
        if a.condition:
            if a.condition not in CONDITION_PROMPTS:
                print(f"Unknown condition: {a.condition} (available: {', '.join(ALL_CONDITIONS)})", file=sys.stderr)
                return False
            condition = a.condition
            if a.ruleset and condition != "ruleset":
                print(f"--ruleset と -c {condition} は同時に指定できない", file=sys.stderr)
                return False
        else:
            condition = "ruleset" if a.ruleset else "zeroshot"

        body = load_report_body(report_path)
        system_tokens, user_tokens = estimate_message_tokens(build_messages(body, condition))

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
        condition: str = param("all", s="-c", l="--condition", description="zeroshot | ruleset | fast | fast2 | all (all = zeroshot+ruleset)")
        dir: str = param("data/reports", s="-d", l="--dir")
        outdir: str = param("out/results", s="-o", l="--outdir")
        temperature: float = param(-1.0, s="-t", l="--temperature", description="未指定(-1)はモデル既定: 既存モデルは 0.3、サーバ既定のモデルは送らない、固定サンプリングのモデルでは無視")
        force: bool = param(False, l="--force", description="既存結果を上書き")
        dry_run: bool = param(False, l="--dry-run", description="API呼び出しせず件数とプロンプトサイズを表示")
        parallel: int = param(1, s="-p", l="--parallel", description="同時リクエスト数（SGLang 等の連続バッチング向け）")

    def run_batch(self, a: BatchArgs):
        models = list(MODELS.keys()) if a.model == "all" else [a.model]
        conditions = resolve_conditions(a.condition)

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
                msg_sets = build_messages(sample_body, cond)
                sys_tokens, user_tokens = estimate_message_tokens(msg_sets)
                calls = f", {len(msg_sets)} calls/report" if len(msg_sets) > 1 else ""
                print(f"[{cond}] system ~{sys_tokens} tokens, user ~{user_tokens} tokens{calls} (sample: {report_files[0].stem})")
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

                def describe(result) -> str:
                    if result is None:
                        return "skip (exists)"
                    tokens_info = f"{result['prompt_tokens']}+{result['completion_tokens']}"
                    if result["thinking_tokens"]:
                        tokens_info += f"(think:{result['thinking_tokens']})"
                    extra = ""
                    if result["finish_reason"] == "length":
                        extra = " TRUNCATED"
                    elif not result["answer"]:
                        extra = " EMPTY"
                    if result.get("forced_answer"):
                        extra += " FORCED"
                    return f"done ({tokens_info} tokens, {result['duration_s']:.1f}s){extra}"

                t_start = time.time()
                if a.parallel <= 1:
                    for i, rpath in enumerate(report_files):
                        prefix = f"[{m}/{cond}] [{i+1}/{total}] {rpath.stem}"
                        print(f"{prefix}: ", end="", flush=True)
                        try:
                            result = run_one(client, rpath, m, cond, a.outdir, a.temperature, a.force)
                        except Exception as e:
                            errors += 1
                            print(f"ERROR: {e}")
                            continue
                        if result is None:
                            skipped += 1
                        else:
                            done += 1
                        print(describe(result))
                else:
                    # 同時に N 件投げる（SGLang の連続バッチングで総スループットを稼ぐ）。
                    # 表示は完了順。duration_s は各リクエスト単体の所要時間（待ち込み）
                    with ThreadPoolExecutor(a.parallel) as ex:
                        futures = {
                            ex.submit(run_one, client, rpath, m, cond, a.outdir, a.temperature, a.force): rpath
                            for rpath in report_files
                        }
                        for n, fut in enumerate(as_completed(futures)):
                            prefix = f"[{m}/{cond}] [{n+1}/{total}] {futures[fut].stem}"
                            try:
                                result = fut.result()
                            except Exception as e:
                                errors += 1
                                print(f"{prefix}: ERROR: {e}", flush=True)
                                continue
                            if result is None:
                                skipped += 1
                            else:
                                done += 1
                            print(f"{prefix}: {describe(result)}", flush=True)

                wall = time.time() - t_start
                per = f", wall {wall:.1f}s ({wall / done:.1f}s/report)" if done else ""
                print(f"[{m}/{cond}] Finished: {done} done, {skipped} skipped, {errors} errors{per}")

    class TestArgs(BaseModel):
        model: str = param("gpt-oss-20b", s="-m", l="--model")
        max_tokens: int = param(32, s="-n", l="--max-tokens", description="回答分の max_tokens（Claude think は budget_tokens に上乗せ）")
        verbose: bool = param(False, s="-v", l="--verbose", description="生レスポンスを表示")

    def run_test(self, a: TestArgs):
        """疎通テスト（本番と同じ実モデル名・extra_body・サンプリングで送る）"""
        spec = get_spec(a.model)
        print(f"Host: {spec.host}:{spec.port}" if spec.port else f"Host: {spec.host}")
        client = create_client(a.model)
        print(f"Model: {a.model} (upstream: {spec.upstream}, sampling: {describe_sampling(spec)})")
        print(f"Sending test message...", flush=True)

        kwargs = request_kwargs(a.model, [{"role": "user", "content": "Hello, respond with OK."}], -1.0, stream=False)
        # Anthropic API は max_tokens > budget_tokens 必須
        budget = kwargs.get("extra_body", {}).get("thinking", {}).get("budget_tokens", 0)
        kwargs["max_tokens"] = budget + a.max_tokens
        try:
            t0 = time.time()
            res = client.chat.completions.create(**kwargs)
            duration = time.time() - t0
        except Exception as e:
            print(f"ERROR: {e}")
            return False

        choice = res.choices[0]
        content = choice.message.content or ""
        reasoning = getattr(choice.message, "reasoning_content", None) or ""
        usage = res.usage

        if a.verbose:
            print(f"\n--- Raw response ---")
            print(json.dumps(res.model_dump(), indent=2, ensure_ascii=False, default=str))
            print(f"--- End ---\n")

        print(f"Content: {content.strip()!r}")
        if reasoning:
            print(f"Reasoning: {len(reasoning)} chars")
        print(f"Finish reason: {choice.finish_reason}")
        print(f"Tokens: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}" if usage else "Tokens: N/A")
        print(f"Duration: {duration:.1f}s")
        print(f"OK" if content.strip() else "WARNING: empty response (thinking モデルなら -n を増やす)")

    class ModelsArgs(BaseModel):
        pass

    def run_models(self, a: ModelsArgs):
        print("Available models:")
        for name, spec in MODELS.items():
            loc = f"{spec.host}:{spec.port}" if spec.port else spec.host
            extras = [f"sampling: {describe_sampling(spec)}"]
            if spec.think_budget:
                extras.append(f"think budget {spec.think_budget}")
            if spec.samples > 1:
                extras.append(f"x{spec.samples} union")
            print(f"  {name:25s} {loc:15s} {spec.description}")
            print(f"  {'':25s} {'':15s} -> {spec.upstream}, max_tokens {spec.max_tokens}, {', '.join(extras)}")


    class ScoreArgs(BaseModel):
        model: str = param("all", s="-m", l="--model", description="モデル名 or 'all'")
        condition: str = param("all", s="-c", l="--condition", description="zeroshot | ruleset | fast | fast2 | all (all = zeroshot+ruleset)")
        resultdir: str = param("out/results", s="-d", l="--resultdir")
        case: str = param("", s="-k", l="--case", description="特定の症例ID (例: 0001)")
        dry_run: bool = param(False, l="--dry-run", description="対象ファイル一覧を表示するだけ")
        force: bool = param(False, l="--force", description="既存スコアを上書き")

    def run_score(self, a: ScoreArgs):
        """claude -p で採点を実行"""
        import re
        import subprocess

        models = SCORING_MODELS if a.model == "all" else [a.model]
        conditions = resolve_conditions(a.condition)

        prompt_text = load_prompt("scoring")

        # 対象ファイル収集
        targets = []
        for cond in conditions:
            for model in models:
                model_dir = Path(a.resultdir) / cond / model
                if not model_dir.exists():
                    continue
                if a.case:
                    files = [model_dir / f"{a.case}.md"]
                    files = [f for f in files if f.exists()]
                else:
                    files = sorted(model_dir.glob("[0-9]*.md"))

                for fpath in files:
                    text = fpath.read_text()
                    if not a.force and "<score>" in text:
                        continue
                    targets.append(fpath)

        if not targets:
            print("No files to score (all already scored or no matches)")
            return

        if a.dry_run:
            print(f"{len(targets)} files to score:")
            for t in targets:
                rel = t.relative_to(Path(a.resultdir))
                print(f"  {rel}")
            return

        total = len(targets)
        done = 0
        errors = 0

        for i, fpath in enumerate(targets):
            rel = fpath.relative_to(Path(a.resultdir))
            prefix = f"[{i+1}/{total}] {rel}"
            print(f"{prefix}: ", end="", flush=True)

            content = fpath.read_text()

            # claude -p でスコアリング
            try:
                result = subprocess.run(
                    ["claude", "-p", prompt_text, "--model", "sonnet"],
                    input=content,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode != 0:
                    print(f"ERROR (exit {result.returncode}): {result.stderr[:200]}")
                    errors += 1
                    continue
            except subprocess.TimeoutExpired:
                print("ERROR (timeout)")
                errors += 1
                continue
            except FileNotFoundError:
                print("ERROR: 'claude' command not found")
                return False

            output = result.stdout.strip()

            # <note> と <score> を抽出
            note_match = re.search(r"(<note>.*?</note>)", output, re.DOTALL)
            score_match = re.search(r"(<score>.*?</score>)", output, re.DOTALL)

            if not score_match:
                print(f"ERROR: no <score> in output")
                print(f"  Output: {output[:200]}")
                errors += 1
                continue

            # 既存の <note>/<score> を除去（force時）
            if a.force:
                content = re.sub(r"\n*<note>.*?</note>", "", content, flags=re.DOTALL)
                content = re.sub(r"\n*<score>.*?</score>", "", content, flags=re.DOTALL)
                content = content.rstrip() + "\n"

            # 追記
            append_parts = []
            if note_match:
                append_parts.append(note_match.group(1))
            append_parts.append(score_match.group(1))

            content = content.rstrip() + "\n\n" + "\n\n".join(append_parts) + "\n"
            fpath.write_text(content)

            # score内容を表示
            score_text = score_match.group(1)
            detection = ""
            det_m = re.search(r"detection:\s*(\S+)", score_text)
            if det_m:
                detection = det_m.group(1)
            status_m = re.search(r"status:\s*(\S+)", score_text)
            status = status_m.group(1) if status_m else "?"

            done += 1
            print(f"{status}/{detection}")

        print(f"\nFinished: {done} scored, {errors} errors, {total - done - errors} skipped")

    class ScoreStatusArgs(BaseModel):
        resultdir: str = param("out/results", s="-d", l="--resultdir")

    def run_score_status(self, a: ScoreStatusArgs):
        """採点の進捗を表示"""
        import re

        for cond in ALL_CONDITIONS:
            for model in SCORING_MODELS:
                model_dir = Path(a.resultdir) / cond / model
                if not model_dir.exists():
                    continue
                files = sorted(model_dir.glob("[0-9]*.md"))
                scored = 0
                for f in files:
                    if "<score>" in f.read_text():
                        scored += 1
                total = len(files)
                bar = f"{'█' * scored}{'░' * (total - scored)}" if total <= 50 else ""
                print(f"  {cond}/{model}: {scored}/{total} {bar}")

    class TallyArgs(BaseModel):
        resultdir: str = param("out/results", s="-d", l="--resultdir")
        reportdir: str = param("data/reports", s="-r", l="--reportdir")
        model: str = param("all", s="-m", l="--model", description="モデル名 or 'all'")
        condition: str = param("all", s="-c", l="--condition", description="zeroshot | ruleset | fast | fast2 | all (all = zeroshot+ruleset)")
        by_tag: bool = param(False, l="--by-tag", description="GSタグ別の内訳を表示")
        csv: str = param("", l="--csv", description="CSV出力先パス")
        outdir: str = param("", s="-o", l="--outdir", description="per-case CSV等の出力先 (例: out)")

    def run_tally(self, a: TallyArgs):
        """スコアを集計して表示"""
        import re

        models = SCORING_MODELS if a.model == "all" else [a.model]
        conditions = resolve_conditions(a.condition)

        # GSタグをレポートファイルから取得
        gs_tags = {}
        report_dir = Path(a.reportdir)
        if report_dir.exists():
            for rpath in report_dir.glob("*.md"):
                text = rpath.read_text()
                m = re.search(r"<gold_standard>\s*\[(\w+)\]", text)
                if m:
                    gs_tags[rpath.stem] = m.group(1)

        rows = []
        for cond in conditions:
            for model in models:
                model_dir = Path(a.resultdir) / cond / model
                if not model_dir.exists():
                    continue
                files = sorted(model_dir.glob("[0-9]*.md"))
                if not files:
                    continue

                scores = []
                for fpath in files:
                    text = fpath.read_text()
                    sm = re.search(r"<score>(.*?)</score>", text, re.DOTALL)
                    if not sm:
                        continue
                    block = sm.group(1)
                    entry = {"case": fpath.stem}
                    for key in ["status", "detection"]:
                        m = re.search(rf"{key}:\s*(\S+)", block)
                        entry[key] = m.group(1) if m else ""
                    for key in ["fp_relevant", "fp_spurious"]:
                        m = re.search(rf"{key}:\s*(\d+)", block)
                        entry[key] = int(m.group(1)) if m else 0
                    entry["gs_tag"] = gs_tags.get(fpath.stem, "")
                    scores.append(entry)

                if not scores:
                    continue

                def summarize(entries):
                    n = len(entries)
                    valid = sum(1 for e in entries if e["status"] == "valid")
                    error = sum(1 for e in entries if e["status"] == "error")
                    tp_exact = sum(1 for e in entries if e["detection"] == "tp-exact")
                    tp_content = sum(1 for e in entries if e["detection"] == "tp-content-only")
                    fn = sum(1 for e in entries if e["detection"] == "fn")
                    fn_clean = sum(1 for e in entries if e["detection"] == "fn-clean")
                    tp = tp_exact + tp_content
                    sensitivity = tp / n if n else 0
                    fp_rel = sum(e["fp_relevant"] for e in entries)
                    fp_spu = sum(e["fp_spurious"] for e in entries)
                    fp_rel_mean = fp_rel / n if n else 0
                    fp_spu_mean = fp_spu / n if n else 0
                    return {
                        "n": n, "valid": valid, "error": error,
                        "tp_exact": tp_exact, "tp_content": tp_content,
                        "fn": fn, "fn_clean": fn_clean,
                        "sensitivity": sensitivity,
                        "fp_rel": fp_rel, "fp_spu": fp_spu,
                        "fp_rel_mean": fp_rel_mean, "fp_spu_mean": fp_spu_mean,
                    }

                summary = summarize(scores)
                row = {"model": model, "condition": cond, "tag": "all", **summary}
                rows.append(row)

                if a.by_tag:
                    tag_groups = {}
                    for e in scores:
                        t = e["gs_tag"] or "unknown"
                        tag_groups.setdefault(t, []).append(e)
                    for tag in ["RuleViolation", "Deficiency", "Inconsistency", "Typo"]:
                        if tag in tag_groups:
                            s = summarize(tag_groups[tag])
                            rows.append({"model": model, "condition": cond, "tag": tag, **s})

        if not rows:
            print("No scored results found")
            return

        # 表示
        df = pd.DataFrame(rows)
        cols = ["model", "condition", "tag", "n",
                "tp_exact", "tp_content", "fn", "fn_clean", "error",
                "sensitivity", "fp_rel_mean", "fp_spu_mean"]
        df = df[cols]
        df["sensitivity"] = df["sensitivity"].map(lambda x: f"{x:.2f}")
        df["fp_rel_mean"] = df["fp_rel_mean"].map(lambda x: f"{x:.1f}")
        df["fp_spu_mean"] = df["fp_spu_mean"].map(lambda x: f"{x:.1f}")

        print(df.to_string(index=False))

        if a.csv:
            df.to_csv(a.csv, index=False)
            print(f"\nSaved to {a.csv}")

        if a.outdir:
            import re as re2
            outdir = Path(a.outdir)
            outdir.mkdir(parents=True, exist_ok=True)

            # per-case CSV (cases.csv)
            case_records = []
            for cond in conditions:
                for model in models:
                    model_dir = Path(a.resultdir) / cond / model
                    if not model_dir.exists():
                        continue
                    for fpath in sorted(model_dir.glob("[0-9]*.md")):
                        text = fpath.read_text()
                        sm = re2.search(r"<score>(.*?)</score>", text, re.DOTALL)
                        if not sm:
                            continue
                        block = sm.group(1)
                        entry = {"model": model, "condition": cond, "case": fpath.stem}
                        for key in ["status", "detection"]:
                            m = re2.search(rf"{key}:\s*(\S+)", block)
                            entry[key] = m.group(1) if m else ""
                        for key in ["fp_relevant", "fp_spurious"]:
                            m = re2.search(rf"{key}:\s*(\d+)", block)
                            entry[key] = int(m.group(1)) if m else 0
                        dm = re2.search(r"duration_s:\s*([\d.]+)", text)
                        entry["duration_s"] = float(dm.group(1)) if dm else None
                        entry["gs_tag"] = gs_tags.get(fpath.stem, "")
                        case_records.append(entry)

            if case_records:
                cases_df = pd.DataFrame(case_records)
                cases_df.to_csv(outdir / "cases.csv", index=False)
                print(f"Saved to {outdir}/cases.csv")

                # duration stats
                dur = cases_df[cases_df["duration_s"].notna()]
                if len(dur) > 0:
                    dur_stats = dur.groupby(["model", "condition"])["duration_s"].describe()
                    dur_stats = dur_stats[["count", "mean", "std", "min", "50%", "max"]]
                    dur_stats.columns = ["n", "mean", "std", "min", "median", "max"]
                    dur_stats = dur_stats.round(1)
                    dur_stats.to_csv(outdir / "duration_stats.csv")
                    print(f"Saved to {outdir}/duration_stats.csv")


def main():
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
