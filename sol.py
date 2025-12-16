# sol.py - STM CAST Main Application Entry Point
import solara
from ui.components.header import Header
from ui.components.sidebar import Sidebar
from ui.components.file_browser import FileBrowserDialog
from ui.pages.dashboard import DashboardPage
from ui.pages.data_upload import DataUploadPage
from ui.pages.linear_corrector import LinearCorrectorPage
from ui.pages.neuronal_network import NeuralNetworkPage
from ui.pages.results import ResultsPage
from ui.pages.settings import SettingsPage
from ui.config import state

# Page routing dictionary - clean and maintainable
PAGES = {
    "Dashboard": DashboardPage,
    "Data Upload": DataUploadPage,
    "Linear Corrector": LinearCorrectorPage,
    "Neural Network": NeuralNetworkPage,
    "Results": ResultsPage,
    "Settings": SettingsPage,
}

@solara.component
def Page():
    """Main application layout with sidebar and page routing"""

    # Get the current page component from the dictionary
    PageComponent = PAGES.get(state.current_page.value, DashboardPage)

    with solara.Column():
        Header()

        # Toggle button - IMMER SICHTBAR
        solara.Button(
            "☰ Menu" if state.sidebar_collapsed.value else "✖ Close Menu",
            on_click=lambda: state.sidebar_collapsed.set(not state.sidebar_collapsed.value),
            color="primary",
            style="margin-bottom: 1rem; width: 150px;"
        )

        FileBrowserDialog()

        if state.sidebar_collapsed.value:
            # Sidebar collapsed - full width content
            with solara.Column():
                PageComponent()
        else:
            # Sidebar visible - split layout
            with solara.Columns([1, 4]):
                Sidebar()
                with solara.Column():
                    PageComponent()