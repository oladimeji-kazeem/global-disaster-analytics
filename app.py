import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel("public_emdat_2025-07-28.xlsx")
    df = df.dropna(subset=["Latitude", "Longitude"])
    df["Year"] = df["Start Year"]
    df["Total Damage (USD)"] = df["Total Damage ('000 US$)"] * 1000
    return df

def format_number(num):
    if pd.isna(num):
        return "N/A"
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    return str(num)

df = load_data()

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    years = st.slider("Year Range", int(df["Year"].min()), int(df["Year"].max()), (2000, 2023))
    disaster_group = st.multiselect("Disaster Group", df["Disaster Group"].dropna().unique(), default=list(df["Disaster Group"].dropna().unique()))
    disaster_subgroup = st.multiselect("Disaster Subgroup", df["Disaster Subgroup"].dropna().unique(), default=list(df["Disaster Subgroup"].dropna().unique()))
    disaster_subtype = st.multiselect("Disaster Subtype", df["Disaster Subtype"].dropna().unique(), default=list(df["Disaster Subtype"].dropna().unique()))
    region = st.multiselect("Region", df["Region"].dropna().unique(), default=list(df["Region"].dropna().unique()))
    subregion = st.multiselect("Subregion", df["Subregion"].dropna().unique(), default=list(df["Subregion"].dropna().unique()))
    country = st.multiselect("Country", df["Country"].dropna().unique(), default=list(df["Country"].dropna().unique()))

# Filter data
filtered_df = df[
    (df["Year"] >= years[0]) & (df["Year"] <= years[1]) &
    (df["Disaster Group"].isin(disaster_group)) &
    (df["Disaster Subgroup"].isin(disaster_subgroup)) &
    (df["Disaster Subtype"].isin(disaster_subtype)) &
    (df["Region"].isin(region)) &
    (df["Subregion"].isin(subregion)) &
    (df["Country"].isin(country))
]

# KPIs
total_events = len(filtered_df)
total_deaths = filtered_df["Total Deaths"].sum()
total_affected = filtered_df["Total Affected"].sum()
total_injured = filtered_df["No. Injured"].sum()
total_damage = filtered_df["Total Damage (USD)"].sum()

st.markdown("<h1 style='text-align: center;'>🌍 Global Disaster Dashboard</h1>", unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Events", format_number(total_events))
col2.metric("Deaths", format_number(total_deaths))
col3.metric("Injured", format_number(total_injured))
col4.metric("Affected", format_number(total_affected))
col5.metric("Damage (USD)", format_number(total_damage))

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Charts", "🗺️ Geo Map", "🗺️ Region Overview", "🤖 Predictive Analytics", "📈 Forecast"])

with tab1:
    st.subheader("Total Deaths by Country")
    st.plotly_chart(px.bar(filtered_df.groupby("Country")["Total Deaths"].sum().sort_values(ascending=False).head(10).reset_index(), x="Country", y="Total Deaths"))

    st.subheader("Total Affected by Country")
    st.plotly_chart(px.bar(filtered_df.groupby("Country")["Total Affected"].sum().sort_values(ascending=False).head(10).reset_index(), x="Country", y="Total Affected"))

    st.subheader("Injuries by Country")
    st.plotly_chart(px.bar(filtered_df.groupby("Country")["No. Injured"].sum().sort_values(ascending=False).head(10).reset_index(), x="Country", y="No. Injured"))

with tab2:
    st.subheader("Disaster Events Map")
    map_df = filtered_df.rename(columns={"Latitude": "latitude", "Longitude": "longitude"})
    st.map(map_df[["latitude", "longitude"]])

with tab3:
    region_summary = filtered_df.groupby("Region")[["Total Deaths", "Total Affected", "No. Injured", "Total Damage (USD)"]].sum().reset_index()
    st.dataframe(region_summary)

    st.subheader("Total Affected by Region")
    st.plotly_chart(px.bar(region_summary.sort_values("Total Affected", ascending=False), x="Region", y="Total Affected"))

    st.subheader("Total Damage by Region")
    st.plotly_chart(px.bar(region_summary.sort_values("Total Damage (USD)", ascending=False), x="Region", y="Total Damage (USD)"))

    st.subheader("Total Death by Region")
    st.plotly_chart(px.bar(region_summary.sort_values("Total Deaths", ascending=False), x="Region", y="Total Deaths"))

    st.subheader("Total Injured by Region")
    st.plotly_chart(px.bar(region_summary.sort_values("No. Injured", ascending=False), x="Region", y="No. Injured"))

with tab4:
    st.subheader("Predictive Analytics (ARIMA Model Forecasts)")
    metrics = ["Total Deaths", "Total Affected", "No. Injured", "Total Damage (USD)"]
    for metric in metrics:
        try:
            ts_data = df.dropna(subset=["Year", metric])
            ts_series = ts_data.groupby("Year")[metric].sum()
            ts_series = ts_series.astype(float)
            model = ARIMA(ts_series, order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=10)
            forecast.index = range(ts_series.index.max() + 1, ts_series.index.max() + 1 + len(forecast))

            fig, ax = plt.subplots()
            ts_series.plot(ax=ax, label="Historical")
            forecast.plot(ax=ax, label="Forecast", linestyle="--")
            ax.set_title(f"Forecast of {metric}")
            ax.set_xlabel("Year")
            ax.set_ylabel(metric)
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not generate forecast for {metric}: {e}")

with tab5:
    st.subheader("📈 Forecast: Total Affected (Next 10 Years)")
    ts_df = df.groupby("Year")["Total Affected"].sum().dropna()
    ts_df = ts_df[ts_df > 0].astype("float64")
    model = ARIMA(ts_df, order=(2, 1, 2))
    model_fit = model.fit()
    forecast_years = 10
    forecast = model_fit.forecast(steps=forecast_years)
    forecast_years_range = list(range(ts_df.index.max() + 1, ts_df.index.max() + 1 + forecast_years))

    fig, ax = plt.subplots()
    ts_df.plot(label="Historical", ax=ax)
    forecast.index = forecast_years_range
    forecast.plot(ax=ax, style="--", label="Forecast")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Affected")
    ax.set_title("Forecast: Total Affected (ARIMA)")
    ax.legend()
    st.pyplot(fig)