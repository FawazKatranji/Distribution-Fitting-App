
#Bt: Fawaz

#This project have been uploaded onto github through the following repository https://github.com/FawazKatranji/Distribution-Fitting-App
#The project have been uploaded onto streamlit on this link https://github.com/FawazKatranji/Distribution-Fitting-App

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (
    norm, expon, gamma, weibull_min, lognorm, beta,
    rayleigh, chi2, uniform, pareto
)
import pandas as pd

#Setting the title 
st.set_page_config(page_title="Distribution Fitting App")
st.title("Distribution Fitting App (NE111 Final Project by Fawaz)")
st.write("Analyze your dataset and visualize fitted probability distributions.\n")

# TABS
tab_data, tab_fit = st.tabs(["Data", "Distribution Fitting"])


# ============================================================
# ----------------------- TAB 1: DATA ------------------------
# ============================================================
with tab_data:

    st.header("1) Input Your Data")
    st.write("Choose how you'd like to load your data:")

    data_option = st.radio("Input method:", ["Upload CSV", "Manual Entry"], horizontal=True)

    data = None
    
    # Data entry options 
    # Option 1
    if data_option == "Upload CSV":
        file = st.file_uploader("Upload CSV (one column of numbers)", type=["csv"])

        if file is not None:
            data = np.genfromtxt(file, delimiter=',')
            data = data[~np.isnan(data)]
    
    # Option 2
    elif data_option == "Manual Entry":
        manual_text = st.text_area("Enter comma-separated values:", "")
        if manual_text.strip():
            try:
                data = np.array([float(x.strip()) for x in manual_text.split(",")])
            except:
                st.error("⚠️ Invalid format — use commas between numbers.")
    st.markdown("---")

    # Show table and the stats
    if data is not None and len(data) > 0:
        st.header("2️) Data Preview")

        df = pd.DataFrame({
            "Index": range(1, len(data) + 1),
            "Value": data
        })

        st.dataframe(df, use_container_width=True)

        st.markdown("### Summary Statistics")

        col1, col2, col3, col4 = st.columns(4)

        #stats
        col1.metric("Count", len(data))
        col2.metric("Min", np.min(data))
        col3.metric("Max", np.max(data))
        col4.metric("Mean", round(np.mean(data), 4))
        st.metric("Std Dev", round(np.std(data), 4))

    else:
        st.info("Upload or enter data to display it here.")



# ============================================================
# ------------------ TAB 2: DISTRIBUTION FITTING -------------
# ============================================================
with tab_fit:

    st.header("3️) Select a Distribution")

    distribution_map = {
        "Normal (Gaussian)": norm,
        "Exponential": expon,
        "Gamma": gamma,
        "Weibull": weibull_min,
        "Log-normal": lognorm,
        "Beta": beta,
        "Rayleigh": rayleigh,
        "Chi-square": chi2,
        "Uniform": uniform,
        "Pareto": pareto
    }

    dist_name = st.selectbox("Choose a distribution to fit:", list(distribution_map.keys()))
    dist = distribution_map[dist_name]

    st.markdown("---")

    if data is None or len(data) == 0:
        st.warning("⚠️ Please load data in the **Data** tab first.")
    else:

        # AUTOMATIC FIT SECTION
        st.subheader("4️) Automatic Fit")

        if st.button("Run Automatic Fit"):
            params = dist.fit(data)

            st.markdown("#### Fitted Parameters")
            st.write(params)

            fig, ax = plt.subplots(figsize=(7, 4))

            # Histogram
            ax.hist(data, bins=20, density=True, alpha=0.5, label="Data")

            # PDF curve
            x = np.linspace(min(data), max(data), 500)
            pdf = dist.pdf(x, *params)
            ax.plot(x, pdf, linewidth=2, label=f"{dist_name} Fit")

            ax.legend()
            st.pyplot(fig)

            # ===============================
            #          ERROR METRICS
            # ===============================

            # Histogram points for comparison
            hist_vals, hist_edges = np.histogram(data, bins=20, density=True)
            hist_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
            pdf_hist = dist.pdf(hist_centers, *params)

            # Errors
            mse = np.mean((hist_vals - pdf_hist) ** 2)
            mae = np.mean(np.abs(hist_vals - pdf_hist))
            max_error = np.max(np.abs(hist_vals - pdf_hist))

            # Display metrics
            st.markdown("### Fit Error Metrics")
            colA, colB, colC = st.columns(3)

            colA.metric("MSE", f"{mse:.6f}")
            colB.metric("MAE", f"{mae:.6f}")
            colC.metric("Max Error", f"{max_error:.6f}")


        st.markdown("---")

        # MANUAL FIT SECTION
        st.subheader("5️) Manual Fit")

        with st.expander("Manual Fit Controls (Optional)"):

            param_count = len(dist.shapes.split(",")) if dist.shapes else 0
            manual_params = []

            # Shape parameters
            if param_count > 0:
                shape_names = dist.shapes.split(",")
                for name in shape_names:
                    val = st.slider(f"{name}:", 0.1, 10.0, 1.0)
                    manual_params.append(val)

            # Location and  scale
            loc = st.slider("loc:", float(min(data)), float(max(data)), float(np.mean(data)))
            scale = st.slider("scale:", 0.1, float(np.std(data)) * 5, float(np.std(data)))

            manual_params.extend([loc, scale])

        # Plot manual result
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.hist(data, bins=20, density=True, alpha=0.5, label="Data")

        x = np.linspace(min(data), max(data), 500)
        pdf2 = dist.pdf(x, *manual_params)
        ax2.plot(x, pdf2, linewidth=2, label="Manual Fit")

        ax2.legend()
        st.pyplot(fig2)
