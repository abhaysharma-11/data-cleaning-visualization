import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Titanic Dashboard",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Titanic Data Cleaning & Visualization Dashboard")
st.markdown("### Data Analytics Project using Python, Pandas, Matplotlib, Seaborn & Streamlit")

# -------------------------
# Load Data
# -------------------------
df = pd.read_csv("cleaned_data/cleaned_titanic.csv")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("Filters")

selected_class = st.sidebar.selectbox(
    "Passenger Class",
    ["All", 1, 2, 3]
)

if selected_class != "All":
    df = df[df["Pclass"] == selected_class]

# -------------------------
# Metrics
# -------------------------
st.header("Dataset Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", int(df.isnull().sum().sum()))

# -------------------------
# Dataset Preview
# -------------------------
st.header("Dataset Preview")

st.dataframe(df.head())

# -------------------------
# Statistics
# -------------------------
st.header("Summary Statistics")

st.dataframe(df.describe())

# -------------------------
# Charts
# -------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Survival Count")

    fig, ax = plt.subplots()
    sns.countplot(x="Survived", data=df, ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Gender Distribution")

    fig, ax = plt.subplots()
    sns.countplot(x="Sex", data=df, ax=ax)
    st.pyplot(fig)

# -------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Passenger Class")

    fig, ax = plt.subplots()
    sns.countplot(x="Pclass", data=df, ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Age Distribution")

    fig, ax = plt.subplots()
    ax.hist(df["Age"], bins=20)
    st.pyplot(fig)

# -------------------------

st.subheader("Fare Distribution")

fig, ax = plt.subplots()

ax.hist(df["Fare"], bins=20)

st.pyplot(fig)

# -------------------------

st.subheader("Correlation Heatmap")

numeric_df = df.select_dtypes(include=["number"])

fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

st.success("Dashboard Loaded Successfully!")