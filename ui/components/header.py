
"""
created_by: Elias Schebath
"""
import solara
from ..config.styles import CUSTOM_CSS

@solara.component
def Header():
    """STM Cast Header with branding"""
    solara.HTML(tag="div", unsafe_innerHTML=CUSTOM_CSS)
    with solara.Column(classes=["stm-header"]):
        solara.HTML(tag="h1", unsafe_innerHTML='<p class="stm-title">⛅ STM CAST</p>')
        solara.HTML(tag="p", unsafe_innerHTML='<p class="stm-subtitle">Weather Prediction Correction System | Data Science Dashboard</p>')