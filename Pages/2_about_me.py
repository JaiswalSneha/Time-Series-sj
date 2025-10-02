import streamlit as st

st.title("📈 Time Series Forecasting App")

st.markdown("---")

# About Me
st.header("About Me")
st.markdown("""
👤 I’m SJ — the creator of this app.

With my background in data and product, I specialize in building tools that bridge analytics and user experience.

In past roles I’ve worked as:
- **Data Analyst** — Experienced with SQL, Python, Excel, Tableau, Consumer Bureau data, and microfinance analytics  
- **Data Engineer** — Worked with IBM DataStage, Informatica, Ataccama, Collibra  

Now, as a Product Manager, I combine business strategy, data insights, and product thinking to solve real-world problems.
""")

st.markdown("---")

# Purpose of This App
st.header("Purpose of This App")
st.markdown("""
This app is built to help you **analyze, model, and forecast** time series data — such as sales or “products sold” over time.

You can:
- Inspect trends, seasonality, and residuals  
- Configure ARIMA / SARIMA parameters  
- Run forecasts and compare predictions  
- Use Prophet as an alternative forecasting method  
- Evaluate model performance and diagnostics  
""")

st.markdown("---")

# Technology Used
st.header("Technology Used")
st.markdown("""
- **Python** — for data manipulation and modeling  
- **Streamlit** — for interactive UI  
- **statsmodels** — for SARIMAX, seasonal decomposition  
- **Prophet** — for alternative forecasting  
- **scikit-learn** — for computing metrics (MSE, RMSE)  
- **pandas / NumPy** — for data handling  
""")

st.markdown("---")

# What You Can Do With It
st.header("What You Can Do With It")
st.markdown("""
1. Filter data by **store**, **product**, and **year(s)**  
2. Select rolling average window, and apply smoothing  
3. Configure **ARIMA / SARIMA** orders (p, d, q, P, D, Q, m)  
4. Plot ACF / PACF diagnostics  
5. Run stationarity tests (ADF, KPSS)  
6. Decompose time series into **trend**, **seasonal**, and **residual** components  
7. Forecast future values and visualize predictions vs actuals  
8. Compare **SARIMA model** results with **Prophet model**  
9. View error metrics and model fit statistics  
""")

st.markdown("---")

# Note about the Inference
st.header("A Note About Inference")
st.markdown("""
- The model assumes the series is **stationary** or made stationary via differencing.  
- Invalid parameter combinations (e.g., overlap between seasonal and non-seasonal AR lags) will be flagged and execution paused.  
- Forecast results include both historical and predicted data; slice out only the future portion if needed.  
- The seasonal period `m` must represent a meaningful cycle (e.g., weekly, monthly, yearly) — `m = 1` is invalid in most cases.  
- Larger SARIMAX models (with many parameters) can be computationally slow — please allow time for fitting.  
""")

st.markdown("---")

# Ready to Explore
st.header("Ready to Explore?")
st.markdown("""
Use the sidebar to choose your filters and model parameters.  
Click **Proceed** to generate diagnostics, plots, and forecasts.

Enjoy exploring your data 
""")
