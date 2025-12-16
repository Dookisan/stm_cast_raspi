"""
Contains all the CSS for the STM_Cast UI
created_by: Elias Schebath
"""

CUSTOM_CSS = """
<style>
    .stm-header {
        background: linear-gradient(135deg, #4FC3F7 0%, #29B6F6 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stm-title {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        text-align: center;
    }
    .stm-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        text-align: center;
    }
    .stm-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #FFB74D;
    }
    .stm-metric {
        background: linear-gradient(135deg, #FFB74D 0%, #FFA726 100%);
        padding: 1.5rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .stm-metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .stm-metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
</style>
"""