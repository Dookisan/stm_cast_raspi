"""
created_by: Elias Schebath
"""
import solara
import pandas as pd
from ..config.state import app_controller, data_preprocessed


@solara.component
def NeuralNetworkPage():
    """Neural network model configuration"""
    # Local state
    input_layer_choice = solara.use_reactive(5)  # Default: hour 5
    nn_input_loaded = solara.use_reactive(False)
    is_initializing = solara.use_reactive(False)  # Loading state for initialization
    is_loading_input = solara.use_reactive(False)  # Loading state for input layer
    is_training = solara.use_reactive(False)  # Training state
    training_log = solara.use_reactive([])  # Training progress messages
    training_complete = solara.use_reactive(False)  # Training done flag

    solara.Markdown("## 🧠 Neural Network Model")

    ctrl = app_controller.value

    # Handler for initializing NN model
    def initialize_nn():
        is_initializing.value = True
        try:
            ctrl.init_neuronal_network()
            ctrl.generate_nn_config()
            print("✅ Neural Network initialized and configured")
        except Exception as e:
            print(f"❌ Error initializing NN: {e}")
        finally:
            is_initializing.value = False

    # Handler for loading NN input layer
    def load_input_layer():
        is_loading_input.value = True
        try:
            ctrl.get_nn_input_layer(input_layer_choice.value)
            nn_input_loaded.value = True
            print(f"✅ NN Input Layer loaded with choice={input_layer_choice.value}")
        except Exception as e:
            print(f"❌ Error loading NN input layer: {e}")
        finally:
            is_loading_input.value = False

    # Handler for training
    def train_neural_network():
        is_training.value = True
        training_log.value = []
        training_complete.value = False

        try:
            # Step 1: Split matrices
            training_log.value = training_log.value + ["📊 Preparing training/test split..."]
            ctrl.split_matricies()
            training_log.value = training_log.value + ["✅ Data split complete"]

            # Step 2: Run random search
            training_log.value = training_log.value + ["🚀 Starting Random Search..."]
            ctrl.random_param_search()
            training_log.value = training_log.value + ["✅ Random Search complete"]

            # Step 3: Analyze results
            training_log.value = training_log.value + ["📊 Analyzing results..."]
            ctrl.analyze_model_results(top_n=5)
            training_log.value = training_log.value + ["✅ Analysis complete"]

            training_complete.value = True
            print("✅ Neural Network training complete")
        except Exception as e:
            training_log.value = training_log.value + [f"❌ Error: {str(e)}"]
            print(f"❌ Training error: {e}")
        finally:
            is_training.value = False

    # Model Initialization
    with solara.Card("🔧 Model Initialization", elevation=2):
        if not ctrl.neural_network_model:
            solara.Warning("⚠️ Neural Network Model not initialized yet")
            solara.Button(
                "Initialize Neural Network",
                color="primary",
                block=True,
                on_click=initialize_nn,
                disabled=is_initializing.value
            )
            if is_initializing.value:
                solara.ProgressLinear(True)
                solara.Markdown("*Initializing Neural Network...*")
        else:
            solara.Success("✅ Neural Network Model initialized")

    # Input Layer Configuration
    with solara.Card("⚙️ Input Layer Configuration", elevation=2):
        if not ctrl.neural_network_model:
            solara.Info("Initialize Neural Network first")
        elif data_preprocessed.value:
            solara.Markdown("""
            **Select the hour for feature/target split:**
            - Hours 0-8: Used as features
            - Hour 9: Used as target
            """)

            solara.SliderInt(
                label=f"Hour Choice: {input_layer_choice.value}",
                value=input_layer_choice,
                min=0,
                max=9,
                step=1
            )

            solara.Button(
                "🔧 Load Input Layer",
                color="primary",
                block=True,
                on_click=load_input_layer,
                disabled=nn_input_loaded.value or is_loading_input.value
            )

            if is_loading_input.value:
                solara.ProgressLinear(True)
                solara.Markdown("*Loading input layer data...*")

            if nn_input_loaded.value and not is_loading_input.value:
                solara.Success(f"✅ Input layer loaded successfully (hour choice: {input_layer_choice.value})")

                # Display loaded data
                if ctrl.data_processor and ctrl.data_processor.error_matrix is not None:
                    solara.Markdown("### 📊 Loaded Training Data")

                    # Error Matrix (Features) - Convert numpy array to DataFrame
                    with solara.Card("Feature Matrix (Error Matrix)", elevation=1):
                        solara.Markdown(f"**Shape:** {ctrl.data_processor.error_matrix.shape}")
                        error_matrix_df = pd.DataFrame(
                            ctrl.data_processor.error_matrix,
                            columns=[f"Feature_{i}" for i in range(ctrl.data_processor.error_matrix.shape[1])]
                        )
                        solara.DataFrame(error_matrix_df, items_per_page=10)

                    # Error Target - Convert numpy array to DataFrame
                    with solara.Card("Target Vector (Error Target)", elevation=1):
                        solara.Markdown(f"**Shape:** {ctrl.data_processor.error_target.shape}")
                        error_target_df = pd.DataFrame(
                            ctrl.data_processor.error_target,
                            columns=["Target"]
                        )
                        solara.DataFrame(error_target_df, items_per_page=10)
        else:
            solara.Warning("⚠️ Please preprocess data first (Data Upload page)")

    # Parameter Space Info (einfach)
    with solara.Card("🎯 Parameter Search Space", elevation=2):
        if ctrl.neural_network_model:
            # Initialize param_space if not already done
            if ctrl.neural_network_model.param_space is None:
                ctrl.neural_network_model._init_config()

            param_space = ctrl.neural_network_model.param_space

            solara.Markdown("**Random Search will sample from these parameters:**")
            solara.Markdown(f"""
            - **Optimizer:** {', '.join(param_space['optimizer'])}
            - **Learning Rate:** {', '.join(str(lr) for lr in param_space['learning_rate'])}
            - **Hidden Units:** {', '.join(str(u) for u in param_space['hidden_units'])}
            - **Hidden Layers:** {', '.join(str(l) for l in param_space['hidden_layers'])}
            - **Activation:** {', '.join(param_space['activation'])}
            - **Dropout:** {', '.join(str(d) for d in param_space['dropout'])}
            - **Epochs:** {', '.join(str(e) for e in param_space['epochs'])}
            - **Batch Size:** {', '.join(str(b) for b in param_space['batch_size'])}
            - **Patience:** {', '.join(str(p) for p in param_space['patience'])}
            """)
        else:
            solara.Warning("⚠️ Neural Network Model not initialized. Initialize in controller first.")

    # Training (einfach)
    with solara.Card("🏋️ Training", elevation=2):
        if not ctrl.neural_network_model:
            solara.Info("Initialize Neural Network first")
        elif not nn_input_loaded.value:
            solara.Info("Load Input Layer first")
        else:
            # Training controls
            solara.Button(
                "▶️ Train Neural Network",
                color="success",
                block=True,
                on_click=train_neural_network,
                disabled=is_training.value or training_complete.value
            )

            # Training progress
            if is_training.value:
                solara.ProgressLinear(True)
                solara.Markdown("*Training in progress...*")

            # Training log
            if training_log.value:
                with solara.Card("Training Log", elevation=1):
                    for msg in training_log.value:
                        solara.Markdown(f"- {msg}")

            # Show results summary after training
            if training_complete.value and ctrl.neural_network_model.search_results:
                with solara.Card("📊 Training Results", elevation=1):
                    results = ctrl.neural_network_model.search_results
                    valid_results = [r for r in results if r['success']]

                    if valid_results:
                        best = max(valid_results, key=lambda x: x['r2_score'])
                        solara.Success(f"✅ Training Complete! {len(valid_results)} successful tests")
                        solara.Markdown(f"""
                        ### 🏆 Best Result:
                        - **R² Score:** {best['r2_score']:.4f}
                        - **RMSE:** {best['rmse']:.4f}
                        - **Config:** {best['config']}
                        """)
                    else:
                        solara.Warning("⚠️ No successful results")

