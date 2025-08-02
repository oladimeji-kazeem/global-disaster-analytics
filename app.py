
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="EM-DAT Disaster Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_emdat_data.csv", parse_dates=["entry_date", "last_update"])

df = load_data()

st.title("🌍 EM-DAT Disaster Dashboard")

# Sidebar filters
st.sidebar.header("Filter Data")
year_range = st.sidebar.slider("Select Year Range", 
                               int(df['entry_date'].dt.year.min()), 
                               int(df['entry_date'].dt.year.max()), 
                               (2000, 2023))

filtered_df = df[(df['entry_date'].dt.year >= year_range[0]) & 
                 (df['entry_date'].dt.year <= year_range[1])]

disaster_type = st.sidebar.multiselect("Disaster Type", options=sorted(df['disaster_type'].dropna().unique()), default=None)
if disaster_type:
    filtered_df = filtered_df[filtered_df['disaster_type'].isin(disaster_type)]

country = st.sidebar.multiselect("Country ISO", options=sorted(df['iso'].dropna().unique()), default=None)
if country:
    filtered_df = filtered_df[filtered_df['iso'].isin(country)]

# Page 1: Overview
st.subheader("📈 Disaster Overview")

col1, col2 = st.columns(2)

with col1:
    yearly_counts = filtered_df['entry_date'].dt.year.value_counts().sort_index()
    st.plotly_chart(px.line(x=yearly_counts.index, y=yearly_counts.values,
                            labels={"x": "Year", "y": "Disaster Count"},
                            title="Disaster Events Over Time"), use_container_width=True)

with col2:
    top_types = filtered_df['disaster_type'].value_counts().nlargest(10)
    st.plotly_chart(px.bar(x=top_types.index, y=top_types.values,
                           labels={"x": "Disaster Type", "y": "Count"},
                           title="Top 10 Disaster Types"), use_container_width=True)

# Page 2: Economic Impact
st.subheader("💰 Economic Impact")

df_damage = filtered_df.dropna(subset=["total_damage_adjusted_000_us"])
df_damage['year'] = df_damage['entry_date'].dt.year

agg_damage = df_damage.groupby('year')['total_damage_adjusted_000_us'].sum()

st.plotly_chart(px.bar(x=agg_damage.index, y=agg_damage.values,
                       labels={"x": "Year", "y": "Total Damage (Adjusted, $'000)"},
                       title="Total Adjusted Damage by Year"), use_container_width=True)

# Page 3: Country-Level Insights
st.subheader("🌐 Country-Level Insights")

country_counts = filtered_df['iso'].value_counts().nlargest(10)
st.plotly_chart(px.bar(x=country_counts.index, y=country_counts.values,
                       labels={"x": "Country ISO", "y": "Disaster Count"},
                       title="Top 10 Countries by Disaster Events"), use_container_width=True)

country_damage = df_damage.groupby('iso')['total_damage_adjusted_000_us'].sum().nlargest(10)
st.plotly_chart(px.bar(x=country_damage.index, y=country_damage.values,
                       labels={"x": "Country ISO", "y": "Total Damage (Adjusted, $'000)"},
                       title="Top 10 Countries by Damage"), use_container_width=True)
