"""
created_by: Elias Schebath
"""
import solara
from ..config.state import sidebar_collapsed, current_page


@solara.component
def Sidebar():
    """Navigation sidebar (without toggle button - that's in main layout)"""

    with solara.Card("Navigation", elevation=2):
        solara.Button(
            "📊 Dashboard",
            on_click=lambda: current_page.set("Dashboard"),
            color="primary" if current_page.value == "Dashboard" else None,
            block=True,
            outlined=current_page.value != "Dashboard"
        )
        solara.Button(
            "📁 Data Upload",
            on_click=lambda: current_page.set("Data Upload"),
            color="primary" if current_page.value == "Data Upload" else None,
            block=True,
            outlined=current_page.value != "Data Upload"
        )
        solara.Button(
            "🔧 Linear Corrector",
            on_click=lambda: current_page.set("Linear Corrector"),
            color="primary" if current_page.value == "Linear Corrector" else None,
            block=True,
            outlined=current_page.value != "Linear Corrector"
        )
        solara.Button(
            "🧠 Neural Network",
            on_click=lambda: current_page.set("Neural Network"),
            color="primary" if current_page.value == "Neural Network" else None,
            block=True,
            outlined=current_page.value != "Neural Network"
        )
        solara.Button(
            "📈 Results & Analysis",
            on_click=lambda: current_page.set("Results"),
            color="primary" if current_page.value == "Results" else None,
            block=True,
            outlined=current_page.value != "Results"
        )
        solara.Button(
            "⚙️ Settings",
            on_click=lambda: current_page.set("Settings"),
            color="primary" if current_page.value == "Settings" else None,
            block=True,
            outlined=current_page.value != "Settings"
        )