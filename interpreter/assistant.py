"""Meeting assistant: intent analysis + reply-keyword hints via local Ollama.

For non-native speakers sitting in English meetings: each finalized ASR
segment is analyzed together with a rolling window of the recent transcript.
The LLM explains, in Chinese, what the speaker is actually driving at, and —
only when the segment invites a response — gives 1-3 reply directions as a
short Chinese phrase plus a few *simple* English keywords (never full
sentences: the participant composes their own words; hard vocabulary they
might misread or mispronounce would do more harm than good).

Two entry points mirror the translator's speculative/authoritative split:
`analyze_partial()` produces an early guess while the utterance is still
being spoken; `analyze()` produces the authoritative version once the
segment finalizes.
"""
import re
import threading

import requests

from .background import Background
from .config import AssistantConfig, TranslatorConfig
from .qa_bank import QaBank, QaEntry

_SYSTEM_PROMPT = (
    "你是一位坐在中方参会者身边的英语会议助手。参会者能听懂大部分英语，"
    "但需要你帮忙确认对方发言的真实意图，并在需要回应时给出回复思路。\n"
    "{who}\n"
    "输入是会议的实时转写（ASR 生成，没有说话人标注，可能有识别错误、"
    "口头语或未说完的半句，请结合上下文推断真实含义），"
    "最后标出的是最新一段发言。发言者是会议中的其他人，不是参会者本人。"
    "转写里以“[我]”开头的行是参会者本人的发言（麦克风采集），"
    "用它判断对方是在追问、回应还是换了话题。\n"
    "参会者可能提供了会议背景资料（议程、与会者角色、项目状态、他此次"
    "开会的目标）。判断意图和回复方向时优先结合背景资料。\n"
    "请只针对最新一段发言，按下面两步输出。\n"
    "第一步，先判定：这段话需要参会者本人马上开口回应吗？"
    "关键看下一个该说话的人是谁。\n"
    "- 向参会者提问、点他的名、征求他意见、等他表态 → 需要回应。\n"
    "- 陈述信息、闲聊、点名让别人回答、把话头交给另一个被点名的人"
    "（即使句子里顺带提到参会者的名字）→ 无需回应。\n"
    "例：“Cool, that's me done. Sarah, anything you want to dig into "
    "before we let him ask us stuff?”——下一个说话的是被点名的 Sarah，"
    "参会者只是被顺带提到 → 无需回应。\n"
    "第二步，按判定结果输出，两种格式二选一：\n"
    "A. 无需回应时，只输出这两行：\n"
    "【意图】一句话说明说话人想表达或推动什么。\n"
    "【无需回应】+ 一句原因。绝不输出【提示】。\n"
    "B. 需要回应时，输出：\n"
    "【意图】一句话说明说话人想问或想推动什么；有言外之意、情绪或立场"
    "就点出来。\n"
    "【提示】1-3 条，每条一行、以“- ”开头，中英对照格式：\n"
    "- 中文方向 英文方向短语 → 英文关键词 中文释义, 英文关键词 中文释义\n"
    "即：方向先中文再给对应的简单英文说法（方向的中文里可以点名背景资料"
    "里的故事或代号）；“→”后面 2-4 个关键词，每个关键词紧跟它的中文"
    "释义，如“risk 风险”。\n"
    "最多 3 条，宁缺毋滥；方向之间要有实质差别"
    "（如同意推进 / 提出顾虑 / 先澄清再表态）。\n"
    "关键词铁律：英文部分是参会者要当场说出口的提词——只用最常见的简单"
    "英文词（普通高中词汇就能读对、听懂的程度），宁可换成简单说法也"
    "绝不用难词、书面词；不要完整句子；专有名词只用背景资料或对话里"
    "已经出现过的。\n"
    "要求：中文部分总共不超过 100 字；不要翻译原文；不要输出任何其他内容。"
)

_WHO_NAMED = (
    "你辅助的参会者在会上被称为“{name}”。发言点名“{name}”时就是在问他；"
    "点名其他人名时则是在问别人。"
)
_WHO_ANON = "你不知道参会者的名字：泛泛提问按需要他回应处理；明确点名某个具体人名时按问别人处理。"

_PARTIAL_NOTE = (
    "注意：下面这段发言还没有说完，这是实时抢跑预测。只输出【意图】一行"
    "（可以带“像是在…”的猜测语气）和至多 2 条【提示】；如果目前还看不出"
    "需要参会者回应，就只输出【意图】。不要输出【无需回应】。"
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# -- QA-bank matcher: heard question -> prepared answer -----------------------

_MATCH_SYSTEM_PROMPT = (
    "你是面试/会议题库匹配器。输入是一份编号题库（每行：编号 | 标题 | 各种"
    "问法）和刚听到的一段话（ASR 实时转写，可能有识别错误、口头语、不完整）。\n"
    "先做第一步判断：这段话是在向对面的人提问或提出请求吗？"
    "如果对方只是在陈述、介绍、说明、闲聊、寒暄——不是问题——直接输出 "
    "NONE，无论内容和题库多相似，也无论上一次匹配是什么。\n"
    "是问题时，再判断它是否就是题库中的某一题。允许换措辞、口语化、"
    "只说了一半，但必须问的是同一件事。题库外的问题输出 NONE——"
    "宁可不匹配，绝不硬凑。\n"
    "带小数点的编号（如 5.1）是追问，通常紧跟它的主题（5）之后出现；"
    "“上一次匹配”仅在这种情况下参考：像 “Another example?”、"
    "“How did you catch it?” 这类只有几个词的**追问式问题**，"
    "优先在上一次匹配的题的追问里找。"
    "一个问题既命中某题的主问法又像它的追问时，输出主题编号。\n"
    "只输出一个编号（如 5 或 5.1）或 NONE，不要输出任何其他内容。"
)

_MATCH_ID_RE = re.compile(r"\b(\d+(?:\.\d+)?)\b")


def _format_entry(entry: QaEntry) -> str:
    """Panel block for a matched entry: header line + the prepared answer."""
    head = f"【题{entry.id}】{entry.title}"
    if entry.is_followup:
        head += f" ↳ {entry.questions[0]}"
    return f"{head}\n{entry.answer}"


class MeetingAssistant:
    """Stateful analyzer that keeps a rolling window of the meeting transcript."""

    def __init__(self, cfg: AssistantConfig, translator_cfg: TranslatorConfig):
        self._cfg = cfg
        # Default to the translator's model/endpoint: it is already resident
        # in Ollama memory, so the assistant adds no extra model load.
        self._model = cfg.model or translator_cfg.model
        self._base_url = translator_cfg.base_url
        self._keep_alive = translator_cfg.keep_alive
        who = (
            _WHO_NAMED.format(name=cfg.user_name) if cfg.user_name else _WHO_ANON
        )
        self._system_prompt = _SYSTEM_PROMPT.format(who=who)
        self._transcript: list[str] = []
        self._lock = threading.Lock()  # transcript is fed from two threads
        self._background = Background()
        self._qa = QaBank()
        self._last_match_id = ""  # follow-up affinity for the matcher
        self._session = requests.Session()

    def warm_up(self) -> None:
        """Pre-load the model if it differs from the (already warm) translator."""
        try:
            self._session.post(
                f"{self._base_url}/api/generate",
                json={"model": self._model, "keep_alive": self._keep_alive},
                timeout=120,
            )
        except requests.RequestException:
            pass  # cold start will just be slower; not fatal

    def analyze(self, new_segments: list[str]) -> str:
        """Record new transcript segments and analyze them as one utterance.

        Returns the formatted analysis, or "" when the batch is too short to
        carry intent (it is still recorded as context for later segments).
        """
        newest = " ".join(s.strip() for s in new_segments if s.strip())
        if not newest:
            return ""
        with self._lock:
            context = self._transcript[-self._cfg.context_segments:]
            self._record(newest)
        if len(newest) < self._cfg.min_chars:
            return ""
        entry = self._match_qa(newest, context)
        if entry is not None:
            self._last_match_id = entry.group_id
            return _format_entry(entry)
        parts = self._prompt_parts(context)
        parts.append("最新一段发言：")
        parts.append(newest)
        return self._chat("\n".join(parts))

    def analyze_partial(self, text: str) -> str:
        """Early guess for an utterance that is still being spoken.

        Does not touch the rolling transcript; the finalized segment will be
        analyzed (and recorded) separately by analyze().
        """
        with self._lock:
            context = self._transcript[-self._cfg.context_segments:]
        # The matcher is cheap (single-token-ish output), so partials go
        # through it too: a confident early match puts the prepared answer
        # on screen before the question is even finished.
        entry = self._match_qa(text, context)
        if entry is not None:
            return _format_entry(entry)
        parts = self._prompt_parts(context)
        parts.append(_PARTIAL_NOTE)
        parts.append("正在进行、尚未说完的发言：")
        parts.append(text)
        return self._chat("\n".join(parts))

    def record_own(self, text: str) -> None:
        """Record the participant's own (microphone) utterance as context.

        Never analyzed — you don't need hints about what you just said — but
        it lets the assistant read the other side's next turn as a follow-up
        to your answer rather than a fresh question.
        """
        text = text.strip()
        if text:
            with self._lock:
                self._record(f"[我] {text}")

    def _record(self, line: str) -> None:
        transcript = self._transcript + [line]
        self._transcript = transcript[-self._cfg.context_segments * 2:]

    def _prompt_parts(self, context: list[str]) -> list[str]:
        parts: list[str] = []
        background = self._background.text()
        if background:
            parts.append("会议背景（参会者提供）：")
            parts.append(background)
            parts.append("")
        if context:
            parts.append("会议近期转写（旧→新）：")
            parts.extend(f"- {seg}" for seg in context)
        return parts

    def _match_qa(self, heard: str, context: list[str]) -> QaEntry | None:
        """Match the heard utterance against the QA bank; None = no match.

        Best-effort: a matcher failure (or an id the bank doesn't know) just
        falls through to the generated-hints path — never raises.
        """
        index = self._qa.index_text()
        if not index or len(heard) < self._cfg.min_chars:
            return None
        parts = ["题库：", index, ""]
        if context:
            parts.append("最近对话（旧→新）：")
            parts.extend(f"- {seg}" for seg in context[-4:])
        # Affinity only helps resolve terse follow-ups ("Another example?");
        # on longer utterances it just drags statements toward a stale match
        # (verified: with it a mid-meeting statement matched 5.2, without it
        # the same input is correctly NONE).
        if self._last_match_id and len(heard) < 60:
            parts.append(f"上一次匹配：{self._last_match_id}")
        parts.append(f"刚听到：{heard}")
        try:
            out = self._post(
                [
                    {"role": "system", "content": _MATCH_SYSTEM_PROMPT},
                    {"role": "user", "content": "\n".join(parts)},
                ],
                temperature=0.0,
            )
        except requests.RequestException:
            return None
        if "NONE" in out.upper():
            return None
        m = _MATCH_ID_RE.search(out)
        return self._qa.get(m.group(1)) if m else None

    def _chat(self, user_content: str) -> str:
        return self._post(
            [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=self._cfg.temperature,
        )

    def _post(self, messages: list[dict], temperature: float) -> str:
        r = self._session.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
                "think": False,
                "keep_alive": self._keep_alive,
                "options": {"temperature": temperature},
            },
            timeout=self._cfg.timeout_s,
        )
        r.raise_for_status()
        content = r.json()["message"]["content"].strip()
        return _THINK_RE.sub("", content).strip()
