import streamlit as st

st.markdown("""
## 🧠 Challenges in Building a SARIMA Model

### 1. 📊 Parameter Selection (p, d, q, P, D, Q, m)
- **Issue**: Determining the appropriate parameters for both non-seasonal and seasonal components was challenging.
- **Solution**: Utilized autocorrelation (ACF) and partial autocorrelation (PACF) plots to identify suitable values for AR and MA terms.

### 2. 🔮 Forecast Size Mismatch
- **Issue**: Expected the forecast output to contain only the forecasted values, but it included both historical and forecasted data.
- **Solution**: Implemented slicing to extract only the future data points.

### 3. 🔁 Seasonality Detection & Multiple Seasonality
- **Issue**: Confusion in selecting the correct seasonal period ("m") when dealing with daily data and exploring various seasonal periods.
- **Solution**: Conducted exploratory data analysis to identify dominant seasonal patterns and adjusted "m" accordingly.

### 4. ✅ Checking Model Assumptions
- **Issue**: Ensuring residuals were white noise, making the data stationary, and avoiding overfitting.
- **Solution**: Applied differencing techniques and validated residuals through statistical tests.

### 5. 📈 Graph Plotting & Visual Diagnostics
- **Issue**: Plotting ACF and PACF for both seasonal and non-seasonal components, understanding cutoffs, and visualizing forecasts alongside historical data.
- **Solution**: Utilized Matplotlib and Seaborn libraries for clear and informative visualizations.

### 6. ⚠️ Handling Invalid Parameter Combinations
- **Issue**: Encountered errors indicating overlap between seasonal and non-seasonal AR terms.
- **Solution**: Learned the rule that "p should not include seasonal lags when P >= 1" and adjusted parameters accordingly.

### 7. 🔄 Understanding Differencing (d vs D)
- **Issue**: Confusion about how many times differencing is applied when both d and D are > 0 and the difference between ordinary and seasonal differencing.
- **Solution**: Grasped that ordinary differencing removes trend, while seasonal differencing removes repeating cycles.

### 8. 🧭 Choosing Seasonal Period m
- **Issue**: Encountered errors like "ValueError: Seasonal periodicity must be greater than 1" and "ValueError: Must include nonzero seasonal periodicity if including seasonal AR, MA, or differencing".
- **Solution**: Understood that m=1 is meaningless and m=0 is only valid when P=D=Q=0.

### 9. ✅ Valid vs Invalid Parameter Combinations
- **Issue**: Difficulty in knowing when (p,d,q,P,D,Q,m) is valid.
- **Solution**: Learned that overlaps between non-seasonal and seasonal lags (like p=2, P=1, m=2) make the model invalid.

### 10. 🔄 Lag Overlap Between Seasonal and Non-seasonal Components
- **Issue**: Asked why models are invalid when both AR parts use the same lag.
- **Solution**: Understood that the model cannot assign two coefficients to the same lag, leading to an identifiability problem.

### 11. 📉 PACF/ACF Plot Limitations
- **Issue**: Hit error: "ValueError: Can only compute partial correlations for lags up to 50% of the sample size."
- **Solution**: Realized that you can’t request more lags than half the dataset size for PACF.

### 12. 🖥️ Computational Cost of SARIMAX
- **Issue**: Asked: “Does SARIMAX take a lot of time to compute?”
- **Solution**: Noticed SARIMAX is heavier than ARIMA because it estimates more parameters (both seasonal + non-seasonal).

### 13. 🤖 Automating Validity Checks
- **Issue**: Wanted an if-else logic to decide if a combination is valid or not.
- **Solution**: Wrote Python code to detect invalid cases (like overlaps, m=1, seasonal terms without cycle).

### 14. 🧠 Explaining Model Theory Simply
- **Issue**: Struggled when explanations were heavy on math.
- **Solution**: Needed very simple, intuitive rules (like “same lag = invalid, different lag = valid”).
""")
