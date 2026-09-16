"""Dialogue-level interpreter: one shared context for both sides of a meeting.

Replaces two independent sentence translators (one per audio lane) with a
single conversation-aware model call. Every finalized utterance — the other
side's from system audio, mine from the microphone — is appended to one
speaker-labelled history, and each new utterance is interpreted *against
that dialogue*:

    [OTHER] How would you handle an out-of-date SWMS?
    [ME]    For the swims I would first...        <- raw ASR

The model returns the utterance as it was actually said (ASR errors fixed
from context — "swims" -> "SWMS" is only recoverable because the other side
said it a turn ago) plus its translation, in one inference, as strict JSON.
Whether a correction happened is decided in code, not by the model.

The meeting assistant keeps its own independent context: different job,
different prompt.
"""
import json
import re
import threading
from dataclasses import dataclass

import requests

from .background import Background
from .config import TranslatorConfig
from .glossary import Glossary
from .translator import _is_dedicated_mt, _strip_think, detect_lang

BACKGROUND_CHARS = 1500  # domain context is worth a lot; keep it bounded
# Fillers are removed in code — small models keep them despite instructions,
# and a regex is deterministic where the model is not.
_FILLER_RE = re.compile(r"(?<![\w-])(?:um+|uh+|erm+|hmm+|uh-huh)(?![\w-])[,.]?\s*", re.I)

OTHER = "OTHER"
ME = "ME"

_SYSTEM_PROMPT = (
    "You are a real-time dialogue interpreter sitting beside a Chinese "
    "participant (labelled ME) in an English-speaking meeting. You are given "
    "the conversation so far with speaker labels, then the NEWEST utterance "
    "as raw ASR output — it may contain recognition errors, fillers, or an "
    "unfinished trailing clause.\n"
    "For the newest utterance produce:\n"
    '1. "corrected": the utterance as it was actually said. Fix obvious ASR '
    "errors: (a) mishearings of names, acronyms, product and technical terms "
    "that appeared earlier in the conversation or in the meeting background — "
    "e.g. if OTHER said \"SWMS\" and ME's ASR reads \"swims\", it is \"SWMS\"; "
    "(b) word salad that no English speaker would say — e.g. \"year of year\" "
    "is \"year over year\", \"clod code\" is \"Claude Code\". Do NOT paraphrase, "
    "polish grammar, translate, or add/remove content; if nothing is wrong, "
    "copy it verbatim.\n"
    '"corrected" stays in the SAME language the utterance was spoken in '
    "(an English utterance stays English, a Chinese one stays Chinese).\n"
    '2. "translation": a faithful translation of "corrected" into the stated '
    "target language. Read ambiguous words in the sense of the meeting's "
    "domain (see the background). Keep product names, company names and "
    "untranslatable tech terms in their original form (Claude, GitHub, SWMS).\n"
    'Output strictly one JSON object: {"corrected": "...", "translation": "..."} '
    "and nothing else."
)

_NORM_RE = re.compile(r"[\W_]+", re.UNICODE)


@dataclass(frozen=True)
class Interpreted:
    corrected: str
    translation: str
    target_lang: str
    changed: bool  # corrected differs from raw ASR beyond case/punctuation


def is_changed(raw: str, corrected: str) -> bool:
    """True when the correction is more than case/punctuation/whitespace."""
    norm = lambda s: _NORM_RE.sub("", s).lower()  # noqa: E731
    return bool(corrected.strip()) and norm(raw) != norm(corrected)


class DialogueInterpreter:
    """Shared, speaker-aware interpreter for both audio lanes (thread-safe)."""

    def __init__(self, cfg: TranslatorConfig):
        self._cfg = cfg
        self._history: list[tuple[str, str]] = []  # (speaker, corrected text)
        self._lock = threading.Lock()
        self._session = requests.Session()
        self._glossary = Glossary()
        # Same hot-reloaded background.txt the assistant reads: knowing the
        # meeting is about construction is what makes "live site" a 工地,
        # not a production server.
        self._background = Background(max_chars=BACKGROUND_CHARS)

    # -- backend housekeeping (same contract as OllamaTranslator) ---------------

    def ping(self) -> bool:
        try:
            return self._session.get(f"{self._cfg.base_url}/api/tags", timeout=3).ok
        except requests.RequestException:
            return False

    def has_model(self) -> bool:
        try:
            r = self._session.get(f"{self._cfg.base_url}/api/tags", timeout=3)
            names = [m["name"] for m in r.json().get("models", [])]
            wanted = self._cfg.model
            return any(n == wanted or n.startswith(wanted + ":") for n in names)
        except requests.RequestException:
            return False

    def warm_up(self) -> None:
        """Load the model(s) into memory so the first segment isn't slow."""
        for model in filter(None, {self._cfg.model, self._cfg.spec_model}):
            try:
                self._session.post(
                    f"{self._cfg.base_url}/api/generate",
                    json={"model": model, "keep_alive": self._cfg.keep_alive},
                    timeout=120,
                )
            except requests.RequestException:
                pass  # cold start will just be slower; not fatal

    # -- interpretation -----------------------------------------------------------

    def interpret(self, raw: str, speaker: str) -> Interpreted:
        """Correct + translate one finalized utterance and record it.

        The lock covers prompt building, inference and the history append so
        turns land in order; callers do their UI work outside it.
        """
        target_lang = "en" if detect_lang(raw) == "zh" else "zh"
        model = self._cfg.model
        if _is_dedicated_mt(model):
            # MT-only models can't do JSON or context-aware correction.
            translation = self._mt_translate(raw, target_lang)
            with self._lock:
                self._record(speaker, raw)
            return Interpreted(raw, translation, target_lang, changed=False)

        with self._lock:
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": self._user_prompt(raw, speaker, target_lang)},
            ]
            content = self._post(model, messages, json_mode=True)
            corrected, translation = _parse(content, raw)
            if detect_lang(corrected) == target_lang != detect_lang(translation):
                # Small models sometimes fill the two fields the wrong way
                # round on Chinese input; the languages tell us which is which.
                corrected, translation = translation, corrected
            if detect_lang(corrected) != detect_lang(raw):
                # "corrected" came back translated instead of corrected (both
                # fields in the target language): keep the raw source.
                corrected = raw
            corrected = _FILLER_RE.sub("", corrected).strip() or corrected
            self._record(speaker, corrected)
        return Interpreted(
            corrected, translation, target_lang, changed=is_changed(raw, corrected)
        )

    def translate_partial(self, text: str, speaker: str) -> str:
        """Provisional translation of an unfinished utterance (not recorded).

        Reads a history snapshot under the lock, then infers outside it so
        speculative calls never delay a finalized turn.
        """
        target_lang = "en" if detect_lang(text) == "zh" else "zh"
        model = self._cfg.spec_model or self._cfg.model
        if _is_dedicated_mt(model):
            return self._mt_translate(text, target_lang, model)
        with self._lock:
            context = self._background_block() + self._context_block()
        target = "English" if target_lang == "en" else "Chinese"
        prompt = (
            f"{context}"
            f"Newest utterance ([{speaker}], raw ASR, UNFINISHED — translate only "
            f"what is there, never invent an ending):\n{text}\n"
            f"Target language: {target}\n"
            "Output ONLY the translation, no JSON, no labels."
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return _strip_think(self._post(model, messages, json_mode=False))

    # -- internals ----------------------------------------------------------------

    def _user_prompt(self, raw: str, speaker: str, target_lang: str) -> str:
        target = "English" if target_lang == "en" else "Chinese"
        parts = [self._background_block(), self._context_block()]
        terms = self._glossary.matches(raw)
        if terms:
            ref = "; ".join(f"{s} -> {d}" for s, d in terms.items())
            parts.append(f"Glossary (must follow in the translation): {ref}\n")
        parts.append(f"Newest utterance ([{speaker}], raw ASR):\n{raw}\n")
        parts.append(f"Target language: {target}")
        return "".join(parts)

    def _background_block(self) -> str:
        """Meeting background first: a constant prefix Ollama can cache."""
        background = self._background.text()
        if not background:
            return ""
        return f"Meeting background (from the participant, may be in Chinese):\n{background}\n\n"

    def _context_block(self) -> str:
        if not self._history:
            return "Conversation so far: (start of meeting)\n\n"
        lines = "\n".join(f"[{spk}] {txt}" for spk, txt in self._history)
        return f"Conversation so far:\n{lines}\n\n"

    def _record(self, speaker: str, text: str) -> None:
        """Append a turn; keep the window bounded by turns AND characters."""
        history = self._history + [(speaker, text.strip())]
        history = history[-self._cfg.history_turns:]
        while len(history) > 1 and sum(len(t) for _, t in history) > self._cfg.history_chars:
            history = history[1:]
        self._history = history

    def _mt_translate(self, text: str, target_lang: str, model: str | None = None) -> str:
        target_cn = "中文" if target_lang == "zh" else "英文"
        prompt = f"将以下文本翻译为{target_cn}，注意只需要输出翻译后的结果，不要额外解释：\n{text}"
        terms = self._glossary.matches(text)
        if terms:
            ref = "\n".join(f"{s} 翻译成 {d}" for s, d in terms.items())
            prompt = f"参考下面的翻译：\n{ref}\n\n{prompt}"
        return _strip_think(
            self._post(model or self._cfg.model, [{"role": "user", "content": prompt}], False)
        )

    def _post(self, model: str, messages: list[dict], json_mode: bool) -> str:
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": self._cfg.keep_alive,
            "options": {"temperature": self._cfg.temperature},
        }
        if json_mode:
            body["format"] = "json"  # decoder-level constraint, not a polite request
        r = self._session.post(
            f"{self._cfg.base_url}/api/chat", json=body, timeout=self._cfg.timeout_s
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


def _parse(content: str, raw: str) -> tuple[str, str]:
    """(corrected, translation) from the model's JSON; degrade gracefully."""
    try:
        obj = json.loads(_strip_think(content))
        corrected = str(obj.get("corrected", "")).strip() or raw
        translation = str(obj.get("translation", "")).strip()
        if translation:
            return corrected, translation
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    # Not JSON (or empty translation): treat the whole output as translation.
    return raw, _strip_think(content)
