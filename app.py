import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymcdm import weights as w
from pymcdm.methods import TOPSIS, MABAC, ARAS, WSM
from pymcdm.helpers import rrankdata
from pymcdm import visuals

# Setup
st.set_page_config(page_title="Dynamic MCDM Dashboard", layout="wide")
st.title("⚡ Dynamic Multi-Criteria Decision Making")

# --- 1. DATA INPUT ---
st.sidebar.header("1. Data Setup")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    # Default data
    data = {
        'Alternative': ['A1', 'A2', 'A3', 'A4'],
        'Cost ($)': [500, 600, 800, 450],
        'Quality (1-10)': [7, 8, 9, 6],
        'Reliability (%)': [0.95, 0.70, 0.85, 0.99],
        'Social Impact': [0.2, 0.8, 0.5, 0.4]
    }
    df = pd.DataFrame(data)

st.subheader("Interactive Decision Matrix")
# data_editor allows real-time changes to the dataframe structure
edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="main_editor")

# Safety check for empty data
if edited_df.empty or len(edited_df.columns) < 2:
    st.error("Please ensure the matrix has at least one alternative and one criterion.")
    st.stop()

# Extract names and data
alts_names = edited_df.iloc[:, 0].astype(str).tolist()
criteria_names = edited_df.columns[1:]
alts_data = edited_df.iloc[:, 1:].to_numpy()

# Check for non-numeric values in criteria
if not np.issubdtype(alts_data.dtype, np.number):
    st.error("Error: All criteria values must be numeric. Please check your input.")
    st.stop()

# --- 2. DYNAMIC CONFIGURATION ---
st.sidebar.divider()
st.sidebar.header("2. Criteria Weighting & Type")

weights_list = []
types_list = []

# Use columns in sidebar for a cleaner look
for col in criteria_names:
    with st.sidebar.expander(f"Settings: {col}", expanded=True):
        c1, c2 = columns = st.columns(2)
        with c1:
            weight = st.number_input(f"Weight", 0.0, 10.0, 1.0, step=0.1, key=f"w_{col}")
            weights_list.append(weight)
        with c2:
            ctype = st.selectbox("Type", ["Benefit", "Cost"], key=f"t_{col}")
            types_list.append(1 if ctype == "Benefit" else -1)

# Normalize weights
weights = np.array(weights_list)
if weights.sum() > 0:
    weights = weights / weights.sum()
else:
    weights = np.ones(len(weights_list)) / len(weights_list)

# --- 3. METHOD SELECTION ---
st.sidebar.divider()
st.sidebar.header("3. Evaluation Methods")
available_methods = {
    'TOPSIS': TOPSIS(),
    'MABAC': MABAC(),
    'ARAS': ARAS(),
    'WSM (SAW)': WSM()
}

selected_method_names = st.sidebar.multiselect(
    "Select methods to compare:",
    list(available_methods.keys()),
    default=['TOPSIS', 'WSM (SAW)']
)

# --- 4. CALCULATION & VISUALIZATION ---
if not selected_method_names:
    st.info("Select at least one method in the sidebar to see results.")
else:
    try:
        prefs = []
        ranks = []
        types_arr = np.array(types_list)
        
        for name in selected_method_names:
            method = available_methods[name]
            # Calculate preference and rank
            pref = method(alts_data, weights, types_arr)
            rank = rrankdata(pref)
            
            prefs.append(pref)
            ranks.append(rank)

        # UI Tabs for better organization
        tab1, tab2, tab3 = st.tabs(["📊 Results Table", "🎯 Rankings", "📈 Visual Comparison"])

        with tab1:
            st.subheader("Preference Scores")
            pref_df = pd.DataFrame(zip(*prefs), columns=selected_method_names, index=alts_names).round(4)
            st.dataframe(pref_df, use_container_width=True)

        with tab2:
            st.subheader("Final Rankings")
            rank_df = pd.DataFrame(zip(*ranks), columns=selected_method_names, index=alts_names).astype(int)
            # Highlight rank 1
            st.dataframe(rank_df.style.highlight_min(axis=0, color='#d4edda'), use_container_width=True)

        with tab3:
            st.subheader("Polar Ranking Visualization")
            if len(ranks) > 0:
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
                visuals.polar_plot(ranks, labels=selected_method_names, ax=ax)
                st.pyplot(fig)
            
            st.divider()
            
            # Additional Bar Chart for comparison
            st.subheader("Rank Distribution")
            st.bar_chart(rank_df)

    except Exception as e:
        st.error(f"An error occurred during calculation: {e}")
        st.info("Tip: Ensure your data matrix doesn't contain zeros if using methods like VIKOR or certain normalizations.")
