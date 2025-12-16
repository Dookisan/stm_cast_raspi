"""
created_by: Elias Schebath
"""
import solara
import os
from ..config.state import current_browser_dir, available_files_list, selected_file_for, weather_api_filepath, show_file_browser, weather_station_filepath

@solara.component
def FileBrowserDialog():
    """Interactive file browser dialog"""

    def navigate_to(path):
        """Navigate to a directory"""
        if os.path.isdir(path):
            current_browser_dir.set(path)
            refresh_file_list()

    def go_up():
        """Go to parent directory"""
        parent = os.path.dirname(current_browser_dir.value)
        navigate_to(parent)

    def refresh_file_list():
        """Refresh the file list"""
        try:
            items = os.listdir(current_browser_dir.value)
            # Separate dirs and files, sort them
            dirs = sorted([d for d in items if os.path.isdir(os.path.join(current_browser_dir.value, d))])
            files = sorted([f for f in items if os.path.isfile(os.path.join(current_browser_dir.value, f)) and f.endswith('.csv')])
            available_files_list.set(dirs + files)
        except PermissionError:
            available_files_list.set([])

    def select_file(filename):
        """Select a file and close browser"""
        full_path = os.path.join(current_browser_dir.value, filename)

        if os.path.isdir(full_path):
            # Navigate into directory
            navigate_to(full_path)
        else:
            # Select file
            if selected_file_for.value == "api":
                weather_api_filepath.set(full_path)
            elif selected_file_for.value == "station":
                weather_station_filepath.set(full_path)

            show_file_browser.set(False)

    # Refresh on open (using use_effect to avoid infinite loop)
    def refresh_on_open():
        if show_file_browser.value and len(available_files_list.value) == 0:
            refresh_file_list()

    solara.use_effect(refresh_on_open, [show_file_browser.value])

    if not show_file_browser.value:
        return

    # Modal-style overlay
    with solara.Column(style={
        "position": "fixed",
        "top": "0",
        "left": "0",
        "right": "0",
        "bottom": "0",
        "background": "rgba(0,0,0,0.5)",
        "z-index": "1000",
        "display": "flex",
        "align-items": "center",
        "justify-content": "center",
        "padding": "2rem"
    }):
        with solara.Card(
            f"📁 File Browser - Select {selected_file_for.value.upper()} File",
            elevation=8,
            style={
                "max-width": "800px",
                "width": "100%",
                "max-height": "80vh",
                "overflow": "auto",
                "background": "white"
            }
        ):
            # Current path
            solara.Markdown(f"**Current:** `{current_browser_dir.value}`")

            # Navigation buttons
            with solara.Row():
                solara.Button(
                    "⬆️ Parent Directory",
                    on_click=go_up,
                    color="primary"
                )
                solara.Button(
                    "🏠 Home",
                    on_click=lambda: navigate_to(os.path.expanduser("~"))
                )
                solara.Button(
                    "📂 Project Root",
                    on_click=lambda: navigate_to(os.getcwd())
                )
                solara.Button(
                    "❌ Close",
                    on_click=lambda: show_file_browser.set(False),
                    color="error"
                )

            solara.Markdown("---")

            # File list
            if len(available_files_list.value) == 0:
                solara.Info("No CSV files or directories found")
            else:
                solara.Markdown("**Click to navigate folders or select CSV files:**")

                for item in available_files_list.value:
                    full_path = os.path.join(current_browser_dir.value, item)
                    is_dir = os.path.isdir(full_path)

                    icon = "📁" if is_dir else "📄"

                    with solara.Row(style="margin: 0.25rem 0;"):
                        solara.Button(
                            f"{icon} {item}",
                            on_click=lambda i=item: select_file(i),
                            text=True,
                            block=True,
                            style="text-align: left;"
                        )