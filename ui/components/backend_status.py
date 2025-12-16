"""
Backend Status Indicator Component
Shows if the backend controller is ready and what components are loaded.

created_by: Elias Schebath
"""
import solara
from ..config.state import app_controller, backend_ready, data_loaded


@solara.component
def BackendStatus():
    """Display backend controller status"""
    
    ctrl = app_controller.value
    
    if not backend_ready.value or ctrl is None:
        with solara.Card("⚠️ Backend Status", elevation=2, style="background: #ffeb3b"):
            solara.Markdown("**Backend not ready!**")
        return
    
    # Backend is ready - show component status
    with solara.Card("✅ Backend Status", elevation=1, style="background: #e8f5e9"):
        solara.Markdown(f"""
        **Components:**
        - Data Processor: {'✅' if ctrl.data_processor else '⏳ Not initialized'}
        - Data Plotter: {'✅' if ctrl.data_plotter else '⏳ Not initialized'}
        - FIR Filter: {'✅' if ctrl.fir_filter else '⏳ Not initialized'}
        - Neural Network: {'✅' if ctrl.neural_network_model else '⏳ Not initialized'}
        
        **Data Status:** {'✅ Loaded' if data_loaded.value else '⏳ No data loaded'}
        """)


@solara.component
def RequiresBackend(content_component):
    """
    Wrapper component that only shows content if backend is ready.
    Use this to protect pages that need the controller.
    """
    if not backend_ready.value or app_controller.value is None:
        with solara.Column():
            solara.Warning("⚠️ Backend not ready. Please wait...")
            BackendStatus()
        return
    
    # Backend ready - show the wrapped content
    content_component()
