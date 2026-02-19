"""
Smart Industrial Maintenance Dashboard
========================================
Streamlit-based interactive dashboard for monitoring, risk assessment,
and maintenance scheduling.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Smart Maintenance System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Custom CSS
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main { font-family: 'Inter', sans-serif; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
        margin: 8px 0;
    }
    .metric-value {
        font-size: 2.5em;
        font-weight: 700;
        background: linear-gradient(135deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.9em;
        color: #8892b0;
        margin-top: 4px;
    }

    .risk-critical { color: #FF4444; font-weight: 700; }
    .risk-elevated { color: #FFAA00; font-weight: 700; }
    .risk-normal   { color: #44BB44; font-weight: 700; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
    }

    h1 { color: #e0e0ff; }
    h2 { color: #c0c0e0; }
    h3 { color: #a0a0d0; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Data Loading
# ============================================================================
@st.cache_data
def load_data():
    """Load processed data and model outputs."""
    data = {}

    # Processed sequences
    for split in ["train", "test"]:
        path = os.path.join(config.PROCESSED_DATA_DIR, f"{split}_data.npz")
        if os.path.exists(path):
            loaded = np.load(path)
            data[split] = {k: loaded[k] for k in loaded.files}

    # Synthetic data
    logs_path = os.path.join(config.SYNTHETIC_DATA_DIR, "maintenance_logs.csv")
    if os.path.exists(logs_path):
        data["maintenance_logs"] = pd.read_csv(logs_path)

    context_path = os.path.join(config.SYNTHETIC_DATA_DIR, "operational_context.csv")
    if os.path.exists(context_path):
        data["operational_context"] = pd.read_csv(context_path)

    # Recommendations
    rec_path = os.path.join(config.PROCESSED_DATA_DIR, "recommendations.csv")
    if os.path.exists(rec_path):
        data["recommendations"] = pd.read_csv(rec_path)

    return data


# ============================================================================
# Dashboard Pages
# ============================================================================
def render_fleet_overview(data):
    """Fleet Overview page: overall health status."""
    st.header("🏭 Fleet Overview")

    if "test" not in data:
        st.warning("⚠️ No test data found. Run the training pipeline first.")
        st.code("python scripts/train_all.py", language="bash")
        return

    test = data["test"]
    n_units = len(np.unique(test["unit_ids"]))
    n_sequences = len(test["X"])

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_units}</div>
            <div class="metric-label">Total Machines</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        critical = int(np.sum(test["y_binary"] == 1)) if "y_binary" in test else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="background: linear-gradient(135deg, #FF4444, #FF6B6B); -webkit-background-clip: text;">{critical}</div>
            <div class="metric-label">Near-Failure Samples</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        avg_rul = np.mean(test["y_rul"]) if "y_rul" in test else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_rul:.0f}</div>
            <div class="metric-label">Avg RUL (cycles)</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        healthy = int(np.sum(test["y_rul"] > 50)) if "y_rul" in test else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="background: linear-gradient(135deg, #44BB44, #66DD66); -webkit-background-clip: text;">{healthy}</div>
            <div class="metric-label">Healthy Samples</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # RUL Distribution
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            x=test["y_rul"], nbins=50,
            title="RUL Distribution",
            labels={"x": "Remaining Useful Life (cycles)", "y": "Count"},
            color_discrete_sequence=["#3a7bd5"],
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Per-unit health
        unit_health = []
        for uid in np.unique(test["unit_ids"]):
            mask = test["unit_ids"] == uid
            min_rul = test["y_rul"][mask].min()
            status = "Critical" if min_rul < 20 else "Warning" if min_rul < 50 else "Healthy"
            unit_health.append({"Unit": f"Engine-{uid:03d}", "Min RUL": min_rul, "Status": status})

        df_health = pd.DataFrame(unit_health)
        color_map = {"Critical": "#FF4444", "Warning": "#FFAA00", "Healthy": "#44BB44"}

        fig = px.bar(
            df_health.sort_values("Min RUL"),
            x="Unit", y="Min RUL", color="Status",
            color_discrete_map=color_map,
            title="Per-Unit Health Status",
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_risk_assessment(data):
    """Risk Assessment page: failure probability and uncertainty."""
    st.header("⚠️ Risk Assessment")

    if "recommendations" in data:
        rec = data["recommendations"]

        # Risk summary
        col1, col2, col3 = st.columns(3)
        with col1:
            critical = len(rec[rec["risk_level"] == "Service Immediately"])
            st.metric("🔴 Critical Risk", critical)
        with col2:
            elevated = len(rec[rec["risk_level"] == "Schedule Soon"])
            st.metric("🟡 Elevated Risk", elevated)
        with col3:
            normal = len(rec[rec["risk_level"] == "Continue Monitoring"])
            st.metric("🟢 Normal", normal)

        st.divider()

        # Risk table with color coding
        st.subheader("Machine Risk Assessment")

        def color_risk(val):
            if val == "Service Immediately":
                return "background-color: rgba(255,68,68,0.3); color: #FF4444; font-weight: bold"
            elif val == "Schedule Soon":
                return "background-color: rgba(255,170,0,0.3); color: #FFAA00; font-weight: bold"
            else:
                return "background-color: rgba(68,187,68,0.3); color: #44BB44; font-weight: bold"

        styled = rec.style.applymap(color_risk, subset=["risk_level"])
        st.dataframe(styled, use_container_width=True, height=400)

        # Risk distribution pie chart
        fig = px.pie(
            rec, names="risk_level",
            color="risk_level",
            color_discrete_map={
                "Service Immediately": "#FF4444",
                "Schedule Soon": "#FFAA00",
                "Continue Monitoring": "#44BB44",
            },
            title="Risk Level Distribution",
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No recommendations found. Run the inference pipeline first.")
        st.code("python scripts/run_pipeline.py", language="bash")


def render_maintenance_schedule(data):
    """Maintenance Schedule page: optimized plan visualization."""
    st.header("🔧 Maintenance Schedule")

    if "recommendations" not in data:
        st.warning("⚠️ No schedule data. Run the pipeline first.")
        return

    rec = data["recommendations"]
    scheduled = rec[rec["scheduled_slot"] != "N/A"].copy()

    if len(scheduled) == 0:
        st.info("No machines are currently scheduled for maintenance.")
        return

    scheduled["scheduled_slot"] = pd.to_numeric(scheduled["scheduled_slot"])

    # Gantt chart
    fig = px.bar(
        scheduled.sort_values("scheduled_slot"),
        y="machine", x="scheduled_slot",
        color="risk_level",
        color_discrete_map={
            "Service Immediately": "#FF4444",
            "Schedule Soon": "#FFAA00",
            "Continue Monitoring": "#44BB44",
        },
        orientation="h",
        title="Maintenance Schedule (Gantt View)",
        labels={"scheduled_slot": "Time Slot", "machine": "Machine"},
    )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=max(300, len(scheduled) * 35),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Schedule table
    st.subheader("Scheduled Maintenance Details")
    st.dataframe(scheduled[["machine", "risk_score", "risk_level", "action", "scheduled_slot"]],
                 use_container_width=True)


def render_maintenance_logs(data):
    """Maintenance Logs page: historical maintenance data."""
    st.header("📋 Maintenance History")

    if "maintenance_logs" not in data:
        st.warning("⚠️ No maintenance logs found.")
        return

    logs = data["maintenance_logs"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Events", len(logs))
    with col2:
        st.metric("Total Cost", f"${logs['cost_usd'].sum():,.0f}")
    with col3:
        st.metric("Avg Downtime", f"{logs['downtime_hours'].mean():.1f} hrs")

    st.divider()

    # Cost by failure type
    col1, col2 = st.columns(2)
    with col1:
        cost_by_type = logs.groupby("failure_type")["cost_usd"].sum().reset_index()
        fig = px.bar(
            cost_by_type, x="failure_type", y="cost_usd",
            title="Cost by Failure Type",
            color="cost_usd",
            color_continuous_scale="Reds",
        )
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Planned vs unplanned
        plan_counts = logs["was_planned"].value_counts().reset_index()
        plan_counts.columns = ["Type", "Count"]
        plan_counts["Type"] = plan_counts["Type"].map({True: "Planned", False: "Unplanned"})
        fig = px.pie(
            plan_counts, names="Type", values="Count",
            color_discrete_sequence=["#44BB44", "#FF4444"],
            title="Planned vs Unplanned Maintenance",
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    # Full log table
    st.subheader("Maintenance Log Records")
    st.dataframe(logs, use_container_width=True, height=400)


def render_operational_context(data):
    """Operational Context page."""
    st.header("⚙️ Operational Context")

    if "operational_context" not in data:
        st.warning("⚠️ No operational context data found.")
        return

    ctx = data["operational_context"]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            ctx, x="machine_type",
            color="priority_level",
            title="Fleet Composition",
            barmode="group",
        )
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            ctx, x="total_cycles", y="max_operating_temp_c",
            color="priority_level",
            title="Cycles vs Operating Temperature",
            size="rated_speed_rpm",
        )
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Machine Specifications")
    st.dataframe(ctx, use_container_width=True)


# ============================================================================
# Main App
# ============================================================================
def main():
    # Sidebar
    st.sidebar.title("🏭 Smart Maintenance")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["Fleet Overview", "Risk Assessment", "Maintenance Schedule",
         "Maintenance History", "Operational Context"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"📍 Device: {config.DEVICE}")
    st.sidebar.caption("FSE 570 Capstone Project")
    st.sidebar.caption("Arizona State University")

    # Load data
    data = load_data()

    # Render selected page
    if page == "Fleet Overview":
        render_fleet_overview(data)
    elif page == "Risk Assessment":
        render_risk_assessment(data)
    elif page == "Maintenance Schedule":
        render_maintenance_schedule(data)
    elif page == "Maintenance History":
        render_maintenance_logs(data)
    elif page == "Operational Context":
        render_operational_context(data)


if __name__ == "__main__":
    main()
