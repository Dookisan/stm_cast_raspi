"""
created_by: Elias Schebath
"""
import solara
import pandas as pd
import numpy as np
from ..config.state import app_controller, data_preprocessed


@solara.component
def LinearCorrectorPage():
    """Linear FIR corrector model"""
    # Local state
    day_range = solara.use_reactive((21, 24))  # (start, end) - User selectable!
    is_initializing = solara.use_reactive(False)
    is_loading = solara.use_reactive(False)
    is_building = solara.use_reactive(False)
    fir_initialized = solara.use_reactive(False)
    data_loaded = solara.use_reactive(False)
    matrices_built = solara.use_reactive(False)

    # Fixed parameters
    LAG_VALUE = 9  # Fixed lag for FIR filter

    solara.Markdown("## 🔧 Linear Corrector (FIR Model)")

    ctrl = app_controller.value

    # Handler for initializing FIR filter
    def initialize_fir():
        is_initializing.value = True
        try:
            ctrl.init_linear_model(LAG_VALUE)
            fir_initialized.value = True
            print(f"✅ FIR Filter initialized with lag={LAG_VALUE}")
        except Exception as e:
            print(f"❌ Error initializing FIR: {e}")
        finally:
            is_initializing.value = False

    # Handler for loading training data
    def load_input_data():
        is_loading.value = True
        try:
            day_start, day_end = day_range.value
            ctrl.get_fir_training_data(day_start, day_end)
            data_loaded.value = True
            print(f"✅ Training data loaded (days {day_start}-{day_end})")
        except Exception as e:
            print(f"❌ Error loading day data: {e}")
        finally:
            is_loading.value = False

    # Handler for building training matrices
    def create_fir_matrices():
        is_building.value = True
        try:
            ctrl.data_processor.create_corrector_training_maticies(
                bias_value=1,
                lag=LAG_VALUE,
                prediction_horizon=24
            )
            matrices_built.value = True
            print(f"✅ Matrices built: E{ctrl.data_processor.E.shape}, E_hat{ctrl.data_processor.E_hat.shape}")
        except Exception as e:
            print(f"❌ Training error: {e}")
        finally:
            is_building.value = False

    def calculate_fir_coeff():
        try:
            ctrl.predict_linear_sequence()
        except Exception as e:
            print(f"Error in OLS optimization")

    # Model Initialization Card
    with solara.Card("🔧 FIR Filter Initialization", elevation=2):
        if not fir_initialized.value:
            solara.Markdown(f"**Fixed Lag Value:** {LAG_VALUE}")
            solara.Button(
                "Initialize FIR Filter",
                color="primary",
                block=True,
                on_click=initialize_fir,
                disabled=is_initializing.value
            )
            if is_initializing.value:
                solara.ProgressLinear(True)
                solara.Markdown("*Initializing FIR Filter...*")
        else:
            solara.Success(f"✅ FIR Filter initialized (lag={LAG_VALUE})")

    # Data Loading Card
    with solara.Card("📊 Load Training Data", elevation=2):
        if not fir_initialized.value:
            solara.Info("Initialize FIR Filter first")
        elif not data_preprocessed.value:
            solara.Warning("⚠️ Please preprocess data first (Data Upload page)")
        else:
            solara.Markdown("**Select day range for training:**")
            solara.SliderRangeInt(
                label=f"Days: {day_range.value[0]} to {day_range.value[1]}",
                value=day_range,
                min=1,
                max=30,
                step=1
            )

            solara.Button(
                "🔧 Load Training Data",
                color="primary",
                block=True,
                on_click=load_input_data,
                disabled=data_loaded.value or is_loading.value
            )

            if is_loading.value:
                solara.ProgressLinear(True)
                solara.Markdown("*Loading training data...*")

            if data_loaded.value and not is_loading.value:
                solara.Success(f"✅ Data loaded (days {day_range.value[0]}-{day_range.value[1]})")

    # Build Training Matrices Card
    with solara.Card("🏗️ Build Training Matrices", elevation=2):
        if not data_loaded.value:
            solara.Info("Load training data first")
        else:
            solara.Button(
                "▶️ Build Matrices",
                color="success",
                block=True,
                on_click=create_fir_matrices,
                disabled=matrices_built.value or is_building.value
            )

            if is_building.value:
                solara.ProgressLinear(True)
                solara.Markdown("*Building training matrices...*")

            if matrices_built.value and not is_building.value:
                solara.Success("✅ Training matrices built successfully")

                # Display matrices
                if ctrl.data_processor.E is not None and ctrl.data_processor.E_hat is not None:
                    solara.Markdown("### 📊 Training Matrices")

                    # E Matrix (Input)
                    with solara.Card("Input Matrix (E)", elevation=1):
                        solara.Markdown(f"**Shape:** {ctrl.data_processor.E.shape}")
                        solara.Markdown(f"**Description:** Past error values + bias term")
                        E_df = pd.DataFrame(
                            ctrl.data_processor.E,
                            columns=[f"e[t-{i}]" for i in range(LAG_VALUE)] + ["bias"]
                        )
                        solara.DataFrame(E_df, items_per_page=10)

                    # E_hat Matrix (Output/Target)
                    with solara.Card("Target Matrix (E_hat)", elevation=1):
                        solara.Markdown(f"**Shape:** {ctrl.data_processor.E_hat.shape}")
                        solara.Markdown(f"**Description:** Future 24h prediction horizon")
                        E_hat_df = pd.DataFrame(
                            ctrl.data_processor.E_hat,
                            columns=[f"e[t+{i+1}]" for i in range(ctrl.data_processor.E_hat.shape[1])]
                        )
                        solara.DataFrame(E_hat_df, items_per_page=10)

    with solara.Card("Calculate FIR Coefficients "):
        if not matrices_built.value:
            solara.Info("Build training matrecies first")

        else:
            solara.Button(
                "▶️ Calculate Coefficients",
                color="success",
                block=True,
                on_click=calculate_fir_coeff,
                disabled=is_building.value
            )
