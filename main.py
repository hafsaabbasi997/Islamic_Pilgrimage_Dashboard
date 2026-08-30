
import pickle
import warnings

import numpy as np
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Islamic Pilgrimage Analytics",
    page_icon="🕋",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = Path("Islamic_Pilgrimage_2000_2026_Dataset.csv")
MODEL_FILE = Path("best_pilgrimage_model.pkl")

NUMERIC_COLS = [
    "Total_Pilgrims",
    "Male_Percentage",
    "Female_Percentage",
    "Average_Age",
    "Average_Travel_Cost_USD",
    "Visa_Approval_Rate",
    "Flights_Arranged",
    "Hotels_Booked",
    "Satisfaction_Rating",
    "Economic_Impact_USD",
]


def format_number(value):
    return f"{value:,.2f}"

# ============================================================
# PROFESSIONAL THEME
# ============================================================
st.markdown(
    """
    <style>
    .stApp { background: #f4f7f6; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#0b302b,#123f38); }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    .hero {
        padding: 30px 34px; border-radius: 22px; color: white;
        background: linear-gradient(135deg,#092e29,#12665a,#2b8c78);
        box-shadow: 0 12px 35px rgba(9,46,41,.16); margin-bottom: 24px;
    }
    .hero h1 { margin: 0; font-size: 34px; font-weight: 800; }
    .hero p { margin: 8px 0 0; opacity: .88; font-size: 15px; }
    .section { font-size: 22px; font-weight: 750; color:#123f38; margin: 22px 0 10px; }
    .insight {
        background:#e8f5f1; border-left:5px solid #197a69;
        padding:13px 16px; border-radius:10px; color:#163b36; margin:7px 0;
    }
    div[data-testid="stMetric"] {
        background:white; border:1px solid #e1e8e5; border-radius:15px;
        padding:15px; box-shadow:0 4px 15px rgba(20,40,35,.045);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD DATA / MODEL
# ============================================================
@st.cache_data

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    return pd.read_csv(DATA_FILE)


@st.cache_resource

def load_model():
    if not MODEL_FILE.exists():
        return None
    with open(MODEL_FILE, "rb") as file:
        return pickle.load(file)


try:
    df = load_data()
except Exception as exc:
    st.error(f"Unable to load the dataset. {exc}")
    st.stop()

model = load_model()

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>🕋 Islamic Pilgrimage Analytics Dashboard</h1>
        <p>Global pilgrimage trends, country comparisons, demographics, economics and predictive analytics • 2000–2026</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.markdown("# Dashboard Controls")
st.sidebar.caption("Filter the analytical view")

country_options = sorted(df["Country"].dropna().unique())
type_options = sorted(df["Pilgrimage_Type"].dropna().unique())

selected_countries = st.sidebar.multiselect(
    "Countries", country_options, default=country_options
)
selected_types = st.sidebar.multiselect(
    "Pilgrimage Type", type_options, default=type_options
)
selected_years = st.sidebar.slider(
    "Year Range",
    int(df["Year"].min()),
    int(df["Year"].max()),
    (int(df["Year"].min()), int(df["Year"].max())),
)

filtered = df[
    df["Country"].isin(selected_countries)
    & df["Pilgrimage_Type"].isin(selected_types)
    & df["Year"].between(*selected_years)
].copy()

if filtered.empty:
    st.warning("No records match the selected filters. Please broaden the filters.")
    st.stop()

# ============================================================
# EXECUTIVE KPIs
# ============================================================
st.markdown('<div class="section">Executive Overview</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Records", format_number(len(filtered)))
k2.metric("Total Pilgrims", format_number(filtered.Total_Pilgrims.sum()))
k3.metric("Countries", format_number(filtered.Country.nunique()))
k4.metric("Avg Satisfaction", f"{format_number(filtered.Satisfaction_Rating.mean())} / 5")
k5.metric("Economic Impact", f"${format_number(filtered.Economic_Impact_USD.sum())}")

# ============================================================
# NAVIGATION
# ============================================================
tabs = st.tabs([
    "Overview",
    "Trends",
    "Demographics",
    "Economics & Operations",
    "Relationships",
    "Prediction",
    "Data Explorer",
])

# ============================================================
# TAB 1 — OVERVIEW
# ============================================================
with tabs[0]:
    st.markdown('<div class="section">Pilgrimage Overview</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    type_summary = (
        filtered.groupby("Pilgrimage_Type", as_index=False)["Total_Pilgrims"]
        .sum().sort_values("Total_Pilgrims", ascending=False)
    )

    with c1:
        fig = px.bar(
            type_summary, x="Pilgrimage_Type", y="Total_Pilgrims",
            title="Total Pilgrims by Pilgrimage Type",
            labels={"Pilgrimage_Type":"Pilgrimage Type", "Total_Pilgrims":"Pilgrims"},
            template="plotly_white", text="Total_Pilgrims",
            color="Pilgrimage_Type",
            color_discrete_map={
                "Hajj": "green",
                "Umrah": "blue",
                "Ziyarat": "orange",
            }
        )
        fig.update_traces(
            texttemplate="%{text:,.2f}",
            textposition="outside"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        country_summary = (
            filtered.groupby("Country", as_index=False)["Total_Pilgrims"]
            .sum().nlargest(10, "Total_Pilgrims")
        )
        fig = px.bar(
            country_summary.sort_values("Total_Pilgrims"),
            x="Total_Pilgrims", y="Country", orientation="h",
            title="Top 10 Countries by Total Pilgrims",
            labels={"Total_Pilgrims":"Pilgrims", "Country":""},
            template="plotly_white", text_auto=".3s"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section">Key Indicators</div>', unsafe_allow_html=True)
    summary = filtered[NUMERIC_COLS].describe().T
    summary["median"] = filtered[NUMERIC_COLS].median()
    summary = summary.applymap(format_number)
    st.dataframe(summary, use_container_width=True)

# ============================================================
# TAB 2 — TRENDS
# ============================================================
with tabs[1]:
    st.markdown('<div class="section">Pilgrimage Trends Over Time</div>', unsafe_allow_html=True)

    yearly = filtered.groupby("Year", as_index=False).agg(
        Total_Pilgrims=("Total_Pilgrims", "sum"),
        Economic_Impact_USD=("Economic_Impact_USD", "sum"),
        Satisfaction_Rating=("Satisfaction_Rating", "mean"),
    )

    fig = px.line(
        yearly, x="Year", y="Total_Pilgrims", markers=True,
        title="Total Pilgrims by Year", template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    t1, t2 = st.columns(2)
    with t1:
        fig = px.line(
            yearly, x="Year", y="Economic_Impact_USD", markers=True,
            title="Economic Impact by Year", template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        fig = px.line(
            yearly, x="Year", y="Satisfaction_Rating", markers=True,
            title="Average Satisfaction by Year", template="plotly_white"
        )
        fig.update_yaxes(range=[0,5])
        st.plotly_chart(fig, use_container_width=True)

    # Type-wise yearly trend
    trend_type = filtered.groupby(["Year", "Pilgrimage_Type"], as_index=False)["Total_Pilgrims"].sum()
    fig = px.line(
        trend_type, x="Year", y="Total_Pilgrims", color="Pilgrimage_Type",
        markers=True, title="Pilgrim Volume Trend by Pilgrimage Type",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 3 — DEMOGRAPHICS
# ============================================================
with tabs[2]:
    st.markdown('<div class="section">Demographic Profile</div>', unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        fig = px.histogram(
            filtered, x="Average_Age", nbins=20,
            title="Distribution of Average Pilgrim Age",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    with d2:
        fig = px.box(
            filtered, x="Pilgrimage_Type", y="Average_Age",
            color="Pilgrimage_Type", title="Age Distribution by Pilgrimage Type",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    gender = pd.DataFrame({
        "Gender": ["Male", "Female"],
        "Percentage": [filtered.Male_Percentage.mean(), filtered.Female_Percentage.mean()],
    })
    fig = px.pie(
        gender, names="Gender", values="Percentage",
        title="Average Gender Composition", hole=.48, template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 4 — ECONOMICS & OPERATIONS
# ============================================================
with tabs[3]:
    st.markdown('<div class="section">Economic & Operational Analysis</div>', unsafe_allow_html=True)

    e1, e2 = st.columns(2)
    with e1:
        fig = px.scatter(
            filtered, x="Average_Travel_Cost_USD", y="Total_Pilgrims",
            color="Pilgrimage_Type", hover_data=["Country", "Year"],
            title="Travel Cost vs Pilgrim Volume", template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    with e2:
        fig = px.scatter(
            filtered, x="Hotels_Booked", y="Economic_Impact_USD",
            color="Pilgrimage_Type", hover_data=["Country", "Year"],
            title="Hotels Booked vs Economic Impact", template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    op = filtered.groupby("Pilgrimage_Type", as_index=False).agg(
        Flights_Arranged=("Flights_Arranged", "sum"),
        Hotels_Booked=("Hotels_Booked", "sum"),
        Economic_Impact_USD=("Economic_Impact_USD", "sum"),
    )
    fig = px.bar(
        op, x="Pilgrimage_Type", y=["Flights_Arranged", "Hotels_Booked"],
        barmode="group", title="Operational Arrangements by Pilgrimage Type",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    econ = filtered.groupby("Country", as_index=False)["Economic_Impact_USD"].sum().nlargest(10, "Economic_Impact_USD")
    fig = px.bar(
        econ.sort_values("Economic_Impact_USD"), x="Economic_Impact_USD", y="Country",
        orientation="h", title="Top Countries by Economic Impact",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 5 — RELATIONSHIPS
# ============================================================
with tabs[4]:
    st.markdown('<div class="section">Relationships & Multivariate Analysis</div>', unsafe_allow_html=True)

    corr = filtered[NUMERIC_COLS].corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, title="Correlation Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True)

    r1, r2 = st.columns(2)
    with r1:
        x_col = st.selectbox("Select X variable", NUMERIC_COLS, index=3)
    with r2:
        y_col = st.selectbox("Select Y variable", NUMERIC_COLS, index=0)

    fig = px.scatter(
        filtered, x=x_col, y=y_col, color="Pilgrimage_Type",
        hover_data=["Country", "Year"],
        title=f"{x_col.replace('_',' ')} vs {y_col.replace('_',' ')}",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 6 — PREDICTION
# ============================================================
with tabs[5]:
    st.markdown('<div class="section">Machine Learning Prediction</div>', unsafe_allow_html=True)
    st.caption("Random Forest model trained for Total_Pilgrims prediction using the project's saved model.")

    if model is None:
        st.warning("best_pilgrimage_model.pkl was not found in the project folder.")
    else:
        features = list(model.feature_names_in_)
        country_features = [c for c in features if c.startswith("Country_")]
        type_features = [c for c in features if c.startswith("Pilgrimage_Type_")]

        p1, p2, p3 = st.columns(3)
        with p1:
            p_country = st.selectbox("Country", country_options, key="pred_country")
            p_type = st.selectbox("Pilgrimage Type", type_options, key="pred_type")
        with p2:
            p_male = st.number_input("Male Percentage", 0.0, 100.0, float(df.Male_Percentage.mean()))
            p_female = 100.0 - p_male
            p_age = st.number_input("Average Age", float(df.Average_Age.min()), float(df.Average_Age.max()), float(df.Average_Age.mean()))
            p_cost = st.number_input("Average Travel Cost (USD)", 0.0, float(df.Average_Travel_Cost_USD.max()*2), float(df.Average_Travel_Cost_USD.mean()))
        with p3:
            p_visa = st.number_input("Visa Approval Rate", 0.0, 100.0, float(df.Visa_Approval_Rate.mean()))
            p_flights = st.number_input("Flights Arranged", 0, int(df.Flights_Arranged.max()*2), int(df.Flights_Arranged.median()))
            p_hotels = st.number_input("Hotels Booked", 0, int(df.Hotels_Booked.max()*2), int(df.Hotels_Booked.median()))

        p4, p5 = st.columns(2)
        with p4:
            p_sat = st.number_input("Satisfaction Rating", 0.0, 5.0, float(df.Satisfaction_Rating.mean()))
        with p5:
            p_econ = st.number_input("Economic Impact (USD)", 0.0, float(df.Economic_Impact_USD.max()*2), float(df.Economic_Impact_USD.mean()))

        if st.button("Predict Total Pilgrims", type="primary", use_container_width=True):
            X_pred = pd.DataFrame(0.0, index=[0], columns=features)
            values = {
                "Male_Percentage": p_male,
                "Female_Percentage": p_female,
                "Average_Age": p_age,
                "Average_Travel_Cost_USD": p_cost,
                "Visa_Approval_Rate": p_visa,
                "Flights_Arranged": p_flights,
                "Hotels_Booked": p_hotels,
                "Satisfaction_Rating": p_sat,
                "Economic_Impact_USD": p_econ,
            }
            for col, value in values.items():
                if col in X_pred.columns:
                    X_pred.loc[0, col] = value

            country_col = f"Country_{p_country}"
            type_col = f"Pilgrimage_Type_{p_type}"
            if country_col in X_pred.columns:
                X_pred.loc[0, country_col] = 1
            if type_col in X_pred.columns:
                X_pred.loc[0, type_col] = 1

            try:
                prediction = float(model.predict(X_pred)[0])
                st.success(f"Estimated Total Pilgrims: {prediction:,.0f}")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

# ============================================================
# TAB 7 — DATA EXPLORER
# ============================================================
with tabs[6]:
    st.markdown('<div class="section">Data Explorer</div>', unsafe_allow_html=True)
    st.dataframe(filtered, use_container_width=True, height=500)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Filtered Dataset",
        csv,
        "filtered_islamic_pilgrimage_data.csv",
        "text/csv",
        use_container_width=True,
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("Islamic Pilgrimage Analytics • Python • Pandas • Plotly • Streamlit")
