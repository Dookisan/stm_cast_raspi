"""
created_by: Elias Schebath
"""

import solara

@solara.component
def SettingsPage():
    """Application settings"""
    solara.Markdown("## ⚙️ Settings")

    with solara.Card("Preferences", elevation=2):
        solara.Checkbox(label="Dark mode", value=False)
        solara.Checkbox(label="Auto-refresh data", value=True)
        solara.Checkbox(label="Show advanced options", value=False)

    with solara.Card("About", elevation=2):
        solara.Markdown(
            "**STM CAST v1.0**\n\n"
            "Weather Prediction Correction System using:\n"
            "- Linear FIR Corrector\n"
            "- Neural Network Predictor\n\n"
            "Developed for STM Weather Station"
        )