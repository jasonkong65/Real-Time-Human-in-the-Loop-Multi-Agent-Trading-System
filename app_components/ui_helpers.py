from typing import Any, List, Optional

import streamlit as st

def card(title: str, value: Any, note: Optional[str] = None):
    note_html = f"<div class='mini-note'>{note}</div>" if note else ""
    st.markdown(
        f"""
        <div class="soft-card">
            <h4>{title}</h4>
            <p>{value}</p>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_pills(items: List[str]):
    html = "".join([f"<span class='status-pill'>{item}</span>" for item in items if item])
    st.markdown(html, unsafe_allow_html=True)