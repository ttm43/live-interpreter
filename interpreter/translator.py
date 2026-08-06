"""zh<->en translation through a local Ollama LLM."""
import re

import requests

from .config import TranslatorConfig
from .glossary import Glossary

_CJK_RE = re.compile(r"[一-鿿]")

_SYSTEM_PROMPT = (
    "You are a professional simultaneous interpreter between Chinese and English.\n"
    "The input is one segment of live speech transcribed by ASR; it may contain "
    "recognition errors, fillers, or an unfinished trailing clause — silently fix "
    "obvious ASR errors from context and translate what was actually meant.\n"
    "Each user message starts with the required target language on its own "
    "line, followed by the segment to translate.\n"
    "Rules:\n"
    "- Always translate into the stated target language, never any other.\n"
    "- Output ONLY the translation. No explanations, no quotes, no labels.\n"
    "- Keep numbers, names and technical terms accurate and consistent with "
    "earlier segments.\n"
    "- Keep product names, company names and untranslatable tech terms in "
    "their original English form (e.g. Claude, ChatGPT, GitHub)."
)


def detect_lang(text: str) -> str:
    """Return 'zh' or 'en' based on CJK character ratio."""
    cjk = len(_CJK_RE.findall(text))
    return "zh" if cjk >= max(1, len(text) * 0.2) else "en"


class OllamaTranslator:
    """Stateful translator that keeps a short rolling context of segment pairs."""

    def __init__(self, cfg: TranslatorConfig):
        self._cfg = cfg
        self._history: list[tuple[str, str]] = []
        self._session = requests.Session()
        self._glossary = Glossary()

    def ping(self) -> bool:
        try:
            r = self._session.get(f"{self._cfg.base_url}/api/tags", timeout=3)
            return r.ok
        except requests.RequestException:
            return False

    def warm_up(self) -> None:
        """Load the model(s) into memory so the first segment isn't slow.

        An /api/generate call with no prompt just loads the model and honours
        keep_alive — the documented Ollama warm-up idiom.
        """
        models = {self._cfg.model, self._cfg.model_zh2en, self._cfg.model_en2zh}
        for model in filter(None, models):
            try:
                self._session.post(
                    f"{self._cfg.base_url}/api/generate",
                    json={"model": model, "keep_alive": self._cfg.keep_alive},
                    timeout=120,
                )
            except requests.RequestException:
                pass  # cold start will just be slower; not fatal

    def has_model(self) -> bool:
        try:
            r = self._session.get(f"{self._cfg.base_url}/api/tags", timeout=3)
            names = [m["name"] for m in r.json().get("models", [])]
            wanted = self._cfg.model
            return any(n == wanted or n.startswith(wanted + ":") for n in names)
        except requests.RequestException:
            return False

    def translate(self, text: str, target_lang: str | None = None) -> str:
        """Translate one segment; target_lang 'zh'/'en' forces the direction."""
        if target_lang is None:
            target_lang = "en" if detect_lang(text) == "zh" else "zh"
        target = "English" if target_lang == "en" else "Chinese"
        model = (
            self._cfg.model_zh2en if target_lang == "en" else self._cfg.model_en2zh
        ) or self._cfg.model

        terms = self._glossary.matches(text)
        if _is_dedicated_mt(model):
            # Translation-specialized models (Hunyuan-MT / HY-MT1.5) use their
            # official prompt templates, incl. terminology intervention.
            target_cn = "中文" if target_lang == "zh" else "英文"
            prompt = (
                f"将以下文本翻译为{target_cn}，注意只需要输出翻译后的结果，"
                f"不要额外解释：\n{text}"
            )
            if terms:
                ref = "\n".join(f"{s} 翻译成 {d}" for s, d in terms.items())
                prompt = f"参考下面的翻译：\n{ref}\n\n{prompt}"
            messages = [{"role": "user", "content": prompt}]
        else:
            prompt = f"Target language: {target}\n"
            if terms:
                ref = "; ".join(f"{s} -> {d}" for s, d in terms.items())
                prompt += f"Glossary (must follow): {ref}\n"
            prompt += text
            messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
            for src, dst in self._history:
                messages.append({"role": "user", "content": src})
                messages.append({"role": "assistant", "content": dst})
            messages.append({"role": "user", "content": prompt})

        r = self._session.post(
            f"{self._cfg.base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "think": False,
                "keep_alive": self._cfg.keep_alive,
                "options": {"temperature": self._cfg.temperature},
            },
            timeout=self._cfg.timeout_s,
        )
        r.raise_for_status()
        content = r.json()["message"]["content"].strip()
        translation = _strip_think(content)

        history = self._history + [(prompt, translation)]
        self._history = history[-self._cfg.history_size:]
        return translation


def _is_dedicated_mt(model: str) -> bool:
    """Translation-specialized models that need their own prompt format."""
    name = model.lower()
    return "hunyuan-mt" in name or "hy-mt" in name


def _strip_think(content: str) -> str:
    """Remove thinking artifacts some models emit despite think=false."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    content = content.replace("/no_think", "").replace("/think", "")
    return content.strip()
