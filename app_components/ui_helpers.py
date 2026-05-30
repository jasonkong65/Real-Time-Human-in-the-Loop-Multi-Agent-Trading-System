from html import escape
from typing import Any, List, Optional

import streamlit as st


def _safe_text(value: Any) -> str:
    return escape(str(value if value is not None else ""))


def card(title: str, value: Any, note: Optional[str] = None, variant: str = "neutral"):
    """Render a compact colored card.

    The previous UI used one grey card for every metric.  This version keeps the
    same simple API, but allows the caller to pass a semantic colour variant so
    non-technical users can scan the result more quickly.
    """
    safe_title = _safe_text(title)
    safe_value = _safe_text(value)
    safe_note = _safe_text(note) if note else ""
    note_html = f"<div class='mini-note'>{safe_note}</div>" if safe_note else ""
    variant = str(variant or "neutral").strip().lower().replace("_", "-")
    st.markdown(
        f"""
        <div class="soft-card card-{variant}">
            <h4>{safe_title}</h4>
            <p>{safe_value}</p>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def card_variant_from_text(text: Any, default: str = "neutral") -> str:
    """Map a signal/risk label to a stable card colour variant."""
    label = str(text or "").lower()
    if any(word in label for word in ["high", "risk", "sell", "block", "caution"]):
        if "low" not in label:
            return "red"
    if any(word in label for word in ["medium", "watch", "monitor", "wait", "confirmation", "pullback"]):
        return "amber"
    if any(word in label for word in ["low", "positive", "bullish", "buy", "candidate", "strong", "high confidence"]):
        return "green"
    if any(word in label for word in ["conservative", "strategy"]):
        return "purple"
    return default


def render_status_pills(items: List[str]):
    html = "".join([f"<span class='status-pill'>{_safe_text(item)}</span>" for item in items if item])
    st.markdown(html, unsafe_allow_html=True)
