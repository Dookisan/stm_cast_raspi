"""
created_by: Elias Schebath
"""
import solara
import plotly.express as px
import pandas as pd 
import numpy as np

from utils.plotter import animate
from ..config.state import app_controller



def generate_temperature_data(days=5):
    rows = days * 24
    columns = [f"{str(h).zfill(2)}:00" for h in range(24)]
    t = np.linspace(0, 24 * days, rows)
    base_temp = 20 + 5 * np.sin(2 * np.pi * (t - 8) / 24)

    data = []

    for i in range(rows):
        current_hour_idx = i % 24
        prediction_horizon = np.arange(i, i + 24)
        row_values = 20 + 5 * np.sin(2 * np.pi * (prediction_horizon - 8) / 24)
        noise = np.random.normal(0, 0.5, size=24) * (np.arange(24) / 12)
        row_values += noise
        row_values[0] = base_temp[i] + np.random.normal(0, 0.1)
        data.append(row_values)

    df = pd.DataFrame(data, columns=columns)
    df.index = np.arange(1, rows + 1)
    df['id'] = df.index
    return df

@solara.component
def ResultsPage():
    """Results visualization and analysis"""
    solara.Markdown("## 📈 Results & Analysis")

    with solara.Card("Performance Metrics Temperature", elevation=2):
        ctrl = app_controller.value
        ctrl.animate_data(pd.to_datetime('24.01.2026 22:05', dayfirst=True), "temperature")

        #start_zeit = pd.to_datetime('24.01.2026 22:05', dayfirst=True)
        #animation = animate(start_zeit,"temperature")

    with solara.Card("Performance Metrics Humidity", elevation=2):
        df_temp = generate_temperature_data(days=1)
        df_melted = df_temp.melt(id_vars=['id'], var_name='hour_string', value_name='temperature')

        df_melted['hour'] = df_melted['hour_string'].apply(lambda x: int(x.split(':')[0]))

        fig = px.line(df_melted,
                        x='id',
                        y='temperature',
                        animation_frame='hour',
                        animation_group='id',
                        hover_name='id',
                        range_y=[df_melted['temperature'].min() - 2, df_melted['temperature'].max() + 2],
                        )

        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 500

        fig.show()