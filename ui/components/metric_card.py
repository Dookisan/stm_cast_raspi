"""
created_by: Elias Schebath
"""
import solara
from ..config.styles import CUSTOM_CSS


@solara.component
def MetricCard(title: str, value: str, icon: str = "📊"):
    """Reusable metric card component"""
    with solara.Column(style="flex: 1; min-width: 200px;"):
        solara.HTML(
            tag="div",
            unsafe_innerHTML=f"""
            <div class="stm-metric">
                <div style="font-size: 2.5rem;">{icon}</div>
                <div class="stm-metric-value">{value}</div>
                <div class="stm-metric-label">{title}</div>
            </div>
            """
        )