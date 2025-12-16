"""
created_by: Schebath Elias
"""
import solara
from ..components.metric_card import MetricCard
from ..components.backend_status import BackendStatus

@solara.component
def DashboardPage():
    """Main dashboard with overview"""
    solara.Markdown("## 📊 Dashboard Overview")

    # Backend Status Card - zeigt ob Controller ready ist
    BackendStatus()

    solara.Markdown("---")

    # Metrics row
    with solara.Row(gap="20px"):
        MetricCard("Models Trained", "2", "🎯")
        MetricCard("Data Points", "1,234", "📈")
        MetricCard("Accuracy", "94.5%", "✨")
        MetricCard("Last Updated", "Today", "🕐")

    solara.Markdown("---")

    # Quick stats
    with solara.Card("System Status", elevation=2):
        solara.Success("✅ System operational")
        solara.Info("📡 Data connection active")
        solara.Markdown("**Next prediction:** In 30 minutes")