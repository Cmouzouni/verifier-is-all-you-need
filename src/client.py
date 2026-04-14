"""LLM client supporting multiple model tiers via Together AI."""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from .config import MODELS, TOGETHER_API_KEY, COST_PER_M_INPUT, COST_PER_M_OUTPUT


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        model_key = self.model
        ci = COST_PER_M_INPUT.get(model_key, 0.50)
        co = COST_PER_M_OUTPUT.get(model_key, 1.00)
        return (self.input_tokens * ci + self.output_tokens * co) / 1_000_000


@dataclass
class TokenTracker:
    total_input: int = 0
    total_output: int = 0
    total_cost: float = 0.0
    n_calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, resp: LLMResponse) -> None:
        with self._lock:
            self.total_input += resp.input_tokens
            self.total_output += resp.output_tokens
            self.total_cost += resp.cost_usd
            self.n_calls += 1

    @property
    def total_tokens(self) -> int:
        return self.total_input + self.total_output


def _extract_json(text: str) -> str:
    # Strip <think>...</think> tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL).strip()
    # Try markdown code block
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Bracket-match outermost { ... }
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return text.strip()


class LLMClient:
    """Client for a specific model tier."""

    def __init__(self, model_key: str = "qwen-3b", seed: int = 42):
        self.model_key = model_key
        self.model_id = MODELS[model_key]
        self.seed = seed
        self.tracker = TokenTracker()
        self._client = OpenAI(
            api_key=TOGETHER_API_KEY,
            base_url="https://api.together.xyz/v1",
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        json_mode: bool = False,
        thinking: bool = False,
        _max_retries: int = 3,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        # Qwen3/3.5 thinking models: respect the thinking flag
        if "Qwen3" in self.model_id:
            if not thinking:
                system_prompt = system_prompt + "\n\n/no_think"
            max_tokens = max(max_tokens, 2048 if not thinking else 4096)

        import time as _time
        for attempt in range(_max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=self.seed,
                    **kwargs,
                )
                break
            except Exception as e:
                if attempt < _max_retries - 1:
                    wait = 2 ** attempt * 2  # 2s, 4s, 8s
                    print(f"  [RETRY {attempt+1}/{_max_retries}] {self.model_key}: {type(e).__name__}, waiting {wait}s...")
                    _time.sleep(wait)
                else:
                    raise
        choice = resp.choices[0]
        usage = resp.usage or type("U", (), {"prompt_tokens": 0, "completion_tokens": 0})()
        raw_content = choice.message.content or ""
        # Strip <think>...</think> tags for Qwen3 models (preserve visible content)
        clean_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
        clean_content = re.sub(r"^.*?</think>", "", clean_content, flags=re.DOTALL).strip()
        # If all content was inside think tags, extract useful parts from thinking
        if not clean_content and raw_content:
            m = re.search(r"<think>(.*?)</think>", raw_content, re.DOTALL)
            think_text = m.group(1).strip() if m else raw_content.replace("<think>", "").strip()
            # Try to find an answer inside the thinking
            ans_match = re.search(r"(?:answer|result|solution)\s*(?:is|=|:)\s*(.+?)(?:\.|$)", think_text, re.IGNORECASE)
            if ans_match:
                clean_content = f"ANSWER: {ans_match.group(1).strip()}"
            else:
                clean_content = think_text  # fallback to raw thinking content
        result = LLMResponse(
            content=clean_content if clean_content else raw_content,
            model=self.model_key,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
        )
        self.tracker.record(result)
        return result

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[BaseModel],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_retries: int = 3,
    ) -> tuple[BaseModel, LLMResponse]:
        schema_text = json.dumps(schema.model_json_schema(), indent=2)
        full_system = (
            f"{system_prompt}\n\n"
            f"You MUST respond with valid JSON matching this schema:\n{schema_text}\n"
            f"Output ONLY the JSON object, no other text."
        )
        last_resp = None
        for attempt in range(max_retries):
            prompt = user_prompt
            if attempt > 0:
                prompt += "\n\n[RETRY: Your previous response was not valid JSON. Output ONLY the JSON object.]"
            resp = self.generate(full_system, prompt, temperature, max_tokens, json_mode=True)
            last_resp = resp
            try:
                raw = _extract_json(resp.content)
                parsed = schema.model_validate_json(raw)
                return parsed, resp
            except Exception:
                continue
        # Final fallback: try without json_mode
        resp = self.generate(full_system, user_prompt, temperature, max_tokens, json_mode=False)
        last_resp = resp
        raw = _extract_json(resp.content)
        parsed = schema.model_validate_json(raw)
        return parsed, resp

    def generate_batch(
        self,
        requests: list[tuple[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_workers: int = 8,
    ) -> list[LLMResponse]:
        results: list[LLMResponse | None] = [None] * len(requests)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self.generate, sys, usr, temperature, max_tokens): idx
                for idx, (sys, usr) in enumerate(requests)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results  # type: ignore[return-value]
