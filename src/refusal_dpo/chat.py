"""
Chat-template rendering with a PINNED date.

Both Llama-3.2-Instruct's and Granite Guardian's chat templates stamp *today's* date into
the system prompt ("Today Date: 08 Jul 2026"). Left alone, every prompt — and so every
activation, generation and judge score in this pipeline — silently depends on the day it
is run. We pin the date by shadowing the templates' `strftime_now` helper.

08 Jul 2026 is the date the committed preference pairs (data/prefs*.jsonl) and the
Milestone 1 numbers were generated.
"""
from __future__ import annotations
import datetime

CHAT_DATE = datetime.date(2026, 7, 8)


def pinned_strftime(fmt: str) -> str:
    return CHAT_DATE.strftime(fmt)


def chat_prompt(tokenizer, instruction: str) -> str:
    """Single user turn, rendered up to the generation position."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False, add_generation_prompt=True, strftime_now=pinned_strftime)
