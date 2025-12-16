"""
Solara works under the hood with changeable state variables.
Otherwise they would be inmuteable whitch defeates the purpose
of an interactive ui.

created_by: Elias Schebath
"""
import solara
import os
from src.controller import controller

# =============================================================================
# Backend Controller - Initialized at app startup
# =============================================================================
app_controller = solara.reactive(controller())  # ← Sofort initialisiert!
backend_ready = solara.reactive(True)  # Flag: Backend ist bereit

# Reactive state variables
current_page = solara.reactive("Dashboard")
data_loaded = solara.reactive(False)
data_preprocessed = solara.reactive(False)  # Preprocessing state
selected_model = solara.reactive("Linear Corrector")

# File paths
weather_api_filepath = solara.reactive("")
weather_station_filepath = solara.reactive("")
uploaded_api_file = solara.reactive(None)
uploaded_station_file = solara.reactive(None)

# File browser state
show_file_browser = solara.reactive(False)
current_browser_dir = solara.reactive(os.getcwd())
selected_file_for = solara.reactive("")  # "api" or "station"
available_files_list = solara.reactive([])

# Data upload page state
show_api_preview = solara.reactive(False)  # Show/hide Weather API DataFrame
show_stm_preview = solara.reactive(False)  # Show/hide Weather Station DataFrame

# Sidebar state
sidebar_collapsed = solara.reactive(False)