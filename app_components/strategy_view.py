from typing import Any, Dict

import streamlit as st

from app_components.helpers import clean_label
from app_components.ui_helpers import card, card_variant_from_text

def _plain_reason(text: Any) -> str:
    """Convert internal snake-case / raw reason text into readable UI text."""
    if text is None:
        return ""
    cleaned = str(text).strip()
    if not cleaned:
        return ""
    cleaned = cleaned.replace("_", " ").strip()
    return cleaned[:1].upper() + cleaned[1:]


def render_strategy_guidance_plain(strategy_result: Dict[str, Any], risk_result: Dict[str, Any]) -> None:
    """Render Strategy Agent output for non-technical users.

    The raw JSON is still available in an expander, but the default view reads
    like a short Groq/Report Agent explanation: action, reason, and next steps.
    """
    strategy_result = strategy_result or {}
    risk_result = risk_result or {}

    action_raw = strategy_result.get("strategy_action") or "MONITOR_AND_RESEARCH"
    level_raw = strategy_result.get("strategy_level") or "Cautious"
    action = clean_label(action_raw)
    level = clean_label(level_raw)

    confidence = strategy_result.get("strategy_confidence") or {}
    confidence_score = confidence.get("score") if isinstance(confidence, dict) else None
    confidence_label = confidence.get("label") if isinstance(confidence, dict) else None
    confidence_text = clean_label(confidence_label or "Unknown")
    if confidence_score is not None:
        try:
            confidence_text = f"{confidence_text} ({float(confidence_score):.2f}/1.00)"
        except Exception:
            pass

    position_guidance = (
        strategy_result.get("position_guidance")
        or "Keep this as a paper-research item and wait for clearer evidence."
    )
    leverage_guidance = (
        strategy_result.get("leverage_guidance")
        or "Do not use leverage in this paper decision-support prototype."
    )
    risk_text = (
        risk_result.get("risk_interpretation")
        or strategy_result.get("risk_interpretation")
        or "The system is using a cautious risk-control layer."
    )
    checklist = (
        strategy_result.get("checklist")
        or strategy_result.get("conditions_to_reconsider")
        or []
    )
    if not isinstance(checklist, list):
        checklist = [checklist]

    reasons = []
    if isinstance(confidence, dict):
        reasons = confidence.get("reasons") or []
    if not isinstance(reasons, list):
        reasons = [reasons]

    # Simple human-facing recommendation sentence.
    action_upper = str(action_raw).upper()
    if "PULLBACK" in action_upper:
        plain_answer = "The setup is interesting, but the entry timing looks risky. The safer paper strategy is to wait for a pullback or stronger confirmation."
    elif "CONFIRMATION" in action_upper:
        plain_answer = "The stock can stay on the watchlist, but the evidence is not strong enough yet for a paper-entry decision."
    elif "RISK_REDUCTION" in action_upper or "SELL_RISK" in action_upper:
        plain_answer = "The system is flagging risk. The safer paper strategy is to review exposure instead of adding more."
    elif "RESEARCH" in action_upper or "POSSIBLE_ENTRY" in action_upper:
        plain_answer = "The stock can be researched further as a paper candidate, but it still needs news, valuation, and risk checks."
    elif "NO_ACTION" in action_upper or "BLOCK" in action_upper:
        plain_answer = "The system does not have enough safe evidence for a paper decision. Wait for better data."
    else:
        plain_answer = "The safest paper decision is to monitor first and wait for clearer evidence."

    cols = st.columns(3)
    with cols[0]:
        card("Suggested action", action, variant=card_variant_from_text(action, "purple"))
    with cols[1]:
        card("Strategy level", level, variant=card_variant_from_text(level, "amber"))
    with cols[2]:
        card("Confidence", confidence_text, variant=card_variant_from_text(confidence_text, "blue"))

    st.markdown("##### Plain-language recommendation")
    st.info(plain_answer)

    st.markdown("##### What this means for the user")
    st.markdown(f"**Position guidance:** {position_guidance}")
    st.markdown(f"**Leverage guidance:** {leverage_guidance}")

    st.markdown("##### Why the system is cautious")
    reason_items = [_plain_reason(r) for r in reasons if _plain_reason(r)]
    if risk_text:
        reason_items.insert(0, _plain_reason(risk_text))
    if reason_items:
        for item in reason_items[:4]:
            st.markdown(f"- {item}")
    else:
        st.markdown("- The strategy layer is waiting for clearer confirmation from the data, model, or risk checks.")

    st.markdown("##### Next checks")
    if checklist:
        for item in checklist[:5]:
            st.markdown(f"- {_plain_reason(item)}")
    else:
        st.markdown("- Re-run the pipeline after the next market data refresh.")
        st.markdown("- Check recent news, valuation, and earnings context.")

    guidance = {
        "strategy_action": strategy_result.get("strategy_action"),
        "strategy_level": strategy_result.get("strategy_level"),
        "strategy_confidence": strategy_result.get("strategy_confidence"),
        "position_guidance": strategy_result.get("position_guidance"),
        "leverage_guidance": strategy_result.get("leverage_guidance"),
        "risk_interpretation": risk_result.get("risk_interpretation"),
        "checklist": checklist,
    }
    with st.expander("Technical Strategy Agent output", expanded=False):
        st.json(guidance)