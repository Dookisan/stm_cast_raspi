"""
ceated_by: Elias Schebath
"""
import solara
from ..config.state import (
    app_controller,
    weather_api_filepath,
    weather_station_filepath,
    selected_file_for,
    show_file_browser,
    data_loaded,
    data_preprocessed,
    show_api_preview,
    show_stm_preview
)

def load_data_from_paths():
    """Load data from specified file paths"""
    import os

    api_path = weather_api_filepath.value
    station_path = weather_station_filepath.value

    # Check if files exist
    if not os.path.exists(api_path):
        print(f"❌ Weather API file not found: {api_path}")
        return

    if not os.path.exists(station_path):
        print(f"❌ Weather Station file not found: {station_path}")
        return

    # Here you can add your actual data loading logic
    # For now, just mark as loaded
    print(f"✅ Loading Weather API data from: {api_path}")
    print(f"✅ Loading Weather Station data from: {station_path}")

    # TODO: Add your actual data processing here
    # import pandas as pd
    # api_data = pd.read_csv(api_path)
    # station_data = pd.read_csv(station_path)

    data_loaded.set(True)


def load_sample_paths():
    """Load sample file paths for testing - doesn't fill input fields"""
    # Define paths without setting the input fields
    api_path = "data/Weather_API_Data_202511102036.csv"
    station_path = "data/Weatherstation_STMCAST_Data_202511102036.csv"

    # activate back end directly with paths
    ctrl = app_controller.value
    ctrl.init_data_processor(
        filepath_station=station_path,
        filepath_weather=api_path
    )
    ctrl.load_data()
    data_loaded.set(True)
    data_preprocessed.set(False)  # Reset preprocessing state
    print("📋 Sample paths loaded!")


def preprocess_data():
    """Run preprocessing on loaded data"""
    ctrl = app_controller.value

    if not ctrl or not ctrl.data_processor:
        print("❌ No data loaded! Load data first.")
        return

    try:
        print("🔄 Starting preprocessing...")
        ctrl.preprocess_data()
        data_preprocessed.set(True)
        print("✅ Preprocessing completed!")
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        data_preprocessed.set(False)


@solara.component
def DataUploadPage():
    """Data upload and preprocessing"""
    solara.Markdown("## 📁 Data Upload & Preprocessing")

    # SAMPLE DATA QUICK LOAD
    with solara.Card("🗄️ Database Sample Data", elevation=2, style="background: #e3f2fd"):
        solara.Markdown("**Quick load sample data from database:**")
        solara.Markdown(
            "- `data/Weather_API_Data_202511102036.csv`\n"
            "- `data/Weatherstation_STMCAST_Data_202511102036.csv`"
        )
        solara.Button(
            "📋 Load Sample Data",
            color="primary",
            block=True,
            on_click=lambda: load_sample_paths()
        )

    # FILE PATH INPUT METHOD
    with solara.Card("Enter File Paths", elevation=2):
        solara.Markdown("Enter the absolute paths to your CSV files:")

        # Weather API Data filepath
        solara.Markdown("**Weather API Data:**")

        with solara.Row():
            solara.InputText(
                label="Path to Weather API CSV",
                value=weather_api_filepath,
                continuous_update=True
            )
            solara.Button(
                "📂 Browse",
                on_click=lambda: (selected_file_for.set("api"), show_file_browser.set(True)),
                color="primary"
            )
        if weather_api_filepath.value:
            solara.Info(f"📁 Path: {weather_api_filepath.value}")

        solara.Markdown("**Weather Station Data:**")
        with solara.Row():
            solara.InputText(
                label="Path to Weather Station CSV",
                value=weather_station_filepath,
                continuous_update=True
            )
            solara.Button(
                "📂 Browse",
                on_click=lambda: (selected_file_for.set("station"), show_file_browser.set(True)),
                color="primary"
            )
        if weather_station_filepath.value:
            solara.Info(f"📁 Path: {weather_station_filepath.value}")

        # Load button
        solara.Button(
            "🚀 Load Data from Paths",
            color="success",
            block=True,
            disabled=not (weather_api_filepath.value and weather_station_filepath.value),
            on_click=lambda: load_data_from_paths()
        )

    # ==========================================================================
    # DATA VIEW - Switches between Raw Data and Preprocessed Data
    # ==========================================================================

    if not data_loaded.value:
        # No data loaded yet
        with solara.Card("📊 Data Preview", elevation=2):
            solara.Markdown("*No data loaded yet. Load sample data or enter file paths above.*")

    elif data_preprocessed.value:
        # PREPROCESSED VIEW - Show only processed_data
        with solara.Card("✨ Preprocessed Data", elevation=2, style="background: #f3e5f5"):
            ctrl = app_controller.value

            if ctrl and ctrl.data_processor and ctrl.data_processor.processed_data is not None:
                # Back button
                solara.Button(
                    "⬅️ Back to Raw Data",
                    on_click=lambda: data_preprocessed.set(False),
                    color="primary",
                    outlined=True
                )

                solara.Success("✅ Data preprocessed successfully!")

                df_processed = ctrl.data_processor.processed_data

                # Summary
                with solara.Card("� Processed Data Summary", elevation=1, style="background: #e8f5e9"):
                    with solara.Columns([1, 1, 1]):
                        with solara.Column():
                            solara.Markdown(f"**Total Records**\n{len(df_processed):,}")
                        with solara.Column():
                            solara.Markdown(f"**Columns**\n{len(df_processed.columns)}")
                        with solara.Column():
                            solara.Markdown(f"**Missing Values**\n{df_processed.isnull().sum().sum()}")

                # Show processed DataFrame
                with solara.Card("Processed DataFrame", elevation=1):
                    solara.Markdown(f"**Columns:** {', '.join(df_processed.columns)}")
                    solara.DataFrame(df_processed.head(50), items_per_page=20)
            else:
                solara.Warning("⚠️ Preprocessed data not available!")

    else:
        # RAW DATA VIEW - Show original data + preprocessing button
        with solara.Card("📊 Raw Data Preview", elevation=2):
            ctrl = app_controller.value

            if ctrl and ctrl.data_processor:
                solara.Success("✅ Data loaded successfully!")

                # Get DataFrames from controller
                df_api = ctrl.data_processor.weather_api
                df_stm = ctrl.data_processor.weather_stm

                # Preprocessing button
                solara.Button(
                    "⚡ Run Preprocessing",
                    on_click=lambda: preprocess_data(),
                    color="warning",
                    block=True,
                    style="margin-bottom: 1rem;"
                )

                # Summary Statistics
                with solara.Card("📈 Summary", elevation=1, style="background: #e8f5e9"):
                    with solara.Columns([1, 1, 1]):
                        with solara.Column():
                            solara.Markdown(f"**API Records**\n{len(df_api):,}")
                        with solara.Column():
                            solara.Markdown(f"**STM Records**\n{len(df_stm):,}")
                        with solara.Column():
                            solara.Markdown(f"**Columns**\n{len(df_api.columns)}")

                # Collapsible Weather API DataFrame
                with solara.Card("Weather API Data", elevation=1):
                    solara.Button(
                        f"{'▼' if not show_api_preview.value else '▲'} Show/Hide Weather API Data ({len(df_api)} rows)",
                        on_click=lambda: show_api_preview.set(not show_api_preview.value),
                        block=True
                    )
                    if show_api_preview.value:
                        solara.Markdown(f"**Columns:** {', '.join(df_api.columns)}")
                        solara.DataFrame(df_api.head(20), items_per_page=10)

                # Collapsible Weather Station DataFrame
                with solara.Card("Weather Station Data", elevation=1):
                    solara.Button(
                        f"{'▼' if not show_stm_preview.value else '▲'} Show/Hide Weather Station Data ({len(df_stm)} rows)",
                        on_click=lambda: show_stm_preview.set(not show_stm_preview.value),
                        block=True
                    )
                    if show_stm_preview.value:
                        solara.Markdown(f"**Columns:** {', '.join(df_stm.columns)}")
                        solara.DataFrame(df_stm.head(20), items_per_page=10)
            else:
                solara.Warning("⚠️ Data loaded but controller not initialized!")