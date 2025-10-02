import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_squared_error
st.set_page_config(page_title="Wide app", layout="wide")
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import time
import backend
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pathlib import Path

st.title('Time Series Analysis')
st.markdown('---')

# ======================== Read Dataframe and Initialization ==============================
# Define the path to the 'data' directory
data_dir = Path(__file__).parent.parent / 'data'

# Define the path to the 'train.csv' file
train_file = data_dir / 'train.csv'
test_file = data_dir / 'test.csv'

org_df = pd.read_csv(train_file)

test_org_df = pd.read_csv(test_file)
org_df = org_df.rename(columns = {'Date':'date','number_sold':'sold'})
org_df['date'] = pd.to_datetime(org_df['date'])
org_df['year'] = org_df['date'].dt.year


# ======================== Sidebar Input ==============================

st.sidebar.header('User Input Data')
store_selected = st.sidebar.selectbox('Select the Store', org_df['store'].unique())
product_selected = st.sidebar.selectbox('Select the Product', org_df['product'].unique())
year_selected = st.sidebar.multiselect('Select the Year', org_df['year'].unique(),default=None)

if not year_selected:
    year_selected = org_df['year'].unique()


roll_avg = st.sidebar.selectbox('Select the Rolling Avg window',range(0,366),index = 30)

# transformation = st.sidebar.selectbox('Select One Transformation',['None', 'Min Max Normalization', 'Standard Scaler', 'Square Root Transform', 'Log Transform'])

st.sidebar.text('ARIMA')
p = st.sidebar.number_input('p - Number of autoregressive (AR) terms', min_value=0, max_value=365, value=0, step=1)
d = st.sidebar.number_input('d - Number of non-seasonal differences', min_value=0, max_value=5, value=0, step=1)
q = st.sidebar.number_input('q - Number of moving average (MA) terms', min_value=0, max_value=365, value=0, step=1)
P = st.sidebar.number_input('P - Number of seasonal AR terms', min_value=0, max_value=10000, value=0, step=1)
D = st.sidebar.number_input('D - Number of seasonal differences', min_value=0, max_value=5, value=0, step=1)
Q = st.sidebar.number_input('Q - Number of seasonal MA terms', min_value=0, max_value=1000, value=0, step=1)
m = st.sidebar.number_input('m - Number of time steps in one seasonal cycle', min_value=0, max_value=1000, value=0,step=1)

forecast_size = int(st.sidebar.number_input('Forecast size / Test data size',min_value=1, max_value=10000, value=1, step=1))

temp = org_df[(org_df['store']==store_selected) & (org_df['product']==product_selected)]
temp = temp[temp['year'].isin(year_selected)]
max_lag = int(temp.shape[0]/2)
del temp

acf_pacf_lag = int(st.sidebar.number_input('ACF-PACF lag', min_value=1, max_value=max_lag, value=30, step=1))


# ------------------------PROCEED ---------------------
select_box = st.sidebar.button('Proceed')
start_time = time.time()

# ======================== Initialize ==============================
df = pd.DataFrame()
skew_status = ''
val = 0
adf_result,adf_stationary,adf_pvalue = 0,0,0
kpss_result,kpss_stat,kpss_pvalue = 0,0,0


# *****************************************************************************************************************************
# ******************************* Starting Calculation *******************************
# *****************************************************************************************************************************





if select_box:

    # check for valid parameter set
    status = backend.check_sarimax_validity(p, d, q, P, D, Q, m)
    if status is not None:
        st.text(status)
        st.stop()


    st.markdown(f"#### Product : {product_selected}")
    st.markdown(f"#### Store : {store_selected}")
    df,test_df = backend.filter_data(org_df, test_org_df, store_selected, product_selected, year_selected)
    st.markdown(f'##### Size of the data :  {df.shape[0]}')

    # ====================== train data =================================

    train_data = df.set_index('date')
    train_data = train_data['sold']
    # train_data = train_data.asfreq('D')

    test_data = test_df.set_index('Date')
    test_data = test_data['number_sold']
    # test_data = test_data.asfreq('D')

    # =========================== ARIMA =================================

    #-----------Updated later to SARIMA--------------------
    # arima = ARIMA(train_data, order=(p, d, q))
    # model = arima.fit()
    # predicted = model.forecast(steps=forecast_size)

    sarima = SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, m))
    model = sarima.fit()
    predicted = model.forecast(steps=forecast_size)
    mse_score = mean_squared_error(test_data[0:forecast_size], predicted)
    rmse_score = np.sqrt(mean_squared_error(test_data[0:forecast_size], predicted))

    st.markdown('---')
    st.markdown('## 📝 Visualization')
    col1, col2, col3 = st.columns(3)

    # ======================== Product vs Time ==============================
    with col1:
        st.markdown("### 📈 Product Sale v/s Time")
        backend.plot_line_graph(df)
    with col2:
        # --------------------Rolling Average -------------------------------
        st.markdown("### 🔄 Rolling Average")
        roll_df = pd.DataFrame()
        roll_df['sold'] = df['sold'].rolling(window=roll_avg).mean()
        roll_df['date'] = df['date']
        roll_df = roll_df.dropna()
        backend.plot_line_graph(roll_df, 'roll')

    with col3:
        st.markdown("### 📊 Nature of the product")
        skew_status = backend.interpret_skewness(df['sold'].skew())
        skewness = np.round(df['sold'].skew(), 2)
        backend.plot_bar_graph(df, store_selected, product_selected)
        st.markdown(f"*Note : Skewness is {skewness} - {skew_status}*")

        # ==================== Trend seasonal REsidual==================================

    st.markdown('---')
    st.markdown('## 📝 Seasonal Decomposition')
    col_tsr1, col_tsr2, col_tsr3 = st.columns(3)
    with col_tsr1:
        decomp_additive = seasonal_decompose(df['sold'], model='additive', period=30)
        st.markdown("### 📈 Trend")
        backend.trend_season_resid(df, decomp_additive.trend, 'trend')
    with col_tsr2:
        st.markdown("### 🌤️ Seasonal")
        backend.trend_season_resid(df, decomp_additive.seasonal, 'season')
    with col_tsr3:
        st.markdown("### 🧩 Residual")
        backend.trend_season_resid(df, decomp_additive.resid, 'residual')


    st.markdown('---')
    st.markdown('## 📝 Forecasting')

    col_final1, col_final2,col_probhet= st.columns(3)
    with col_final1:
        st.markdown("### 🎯 Test v/s Predicted")
        backend.final_prediction(test_data[0:forecast_size], predicted)
    with col_final2:
        st.markdown(f"### ⚙️ p:{p},d:{d},q:{q},P:{P},D:{D},Q:{Q}, m:{m}")
        backend.final_forecast(df, predicted, forecast_size)
    with col_probhet:
        st.markdown("### 🔮 Output from Prophet")
        backend.prophet_model(df, forecast_size)


    # ============================= Printing the scores ===============================================

    st.markdown('---')

    col_11, col_12, col_13, col_14 = st.columns(4)
    with col_11:
        st.markdown("<h4 style='text-align: center;'>📌 MSE Score:</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center;'>{np.round(mse_score, 2)}</h4>", unsafe_allow_html=True)
    with col_12:
        st.markdown("<h4 style='text-align: center;'>📌 RMSE Score:</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center;'>{np.round(rmse_score, 2)}</h4>", unsafe_allow_html=True)
    with col_13:
        st.markdown("<h4 style='text-align: center;'>📌 AIC Score:</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center;'>{np.round(model.aic, 2)}</h4>", unsafe_allow_html=True)
    with col_14:
        st.markdown("<h4 style='text-align: center;'>📌 BIC Score:</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center;'>{np.round(model.bic, 2)}</h4>", unsafe_allow_html=True)


    st.markdown('---')
    st.markdown('## 📝 Stationary Test')

    # ======================== ADF ==============================

    st.markdown('### 🔍 Augmented Dickey-Fuller Test')
    adf_result = adfuller(df['sold'], autolag='AIC')
    adf_pvalue = adf_result[1]

    col11, col12, col13 = st.columns(3)
    with col11:
        st.markdown("<h4 style='text-align: center;'> ADF Statistic</h4>", unsafe_allow_html=True)
        st.markdown(f"<h5 style='text-align: center;'>{adf_result[0]}</h5>", unsafe_allow_html=True)
    with col12:
        st.markdown("<h4 style='text-align: center;'> ADF p-value</h4>", unsafe_allow_html=True)
        st.markdown(f"<h5 style='text-align: center;'>{adf_pvalue}</h5>", unsafe_allow_html=True)
    adf_stationary = adf_pvalue < 0.05
    with col13:
        st.markdown("<h4 style='text-align: center;'> Interpretation</h4>", unsafe_allow_html=True)
        if adf_stationary:
            st.markdown(f"<h5 style='text-align: center;'>Stationary (reject Ho)</h5>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h5 style='text-align: center;'>Non-stationary (fail to reject Ho)</h5>", unsafe_allow_html=True)

        # ======================== KPSS ==============================
    st.markdown('### 🔍 KPSS Test')

    kpss_result = kpss(df['sold'], regression='c', nlags='auto')
    kpss_stat = kpss_result[0]
    kpss_pvalue = kpss_result[1]

    col11, col12, col13 = st.columns(3)
    with col11:
        st.markdown("<h4 style='text-align: center;'> KPSS Statistic </h4>", unsafe_allow_html=True)
        st.markdown(f"<h5 style='text-align: center;'>{kpss_stat}</h5>", unsafe_allow_html=True)
    with col12:
        st.markdown("<h4 style='text-align: center;'> KPSS p-value </h4>", unsafe_allow_html=True)
        st.markdown(f"<h5 style='text-align: center;'>{kpss_pvalue}</h5>", unsafe_allow_html=True)

    kpss_stationary = kpss_pvalue >= 0.05
    with col13:
        st.markdown("<h4 style='text-align: center;'> Interpretation </h4>", unsafe_allow_html=True)
        if kpss_stationary:
            st.markdown(f"<h5 style='text-align: center;'>Stationary (reject Ho)</h5>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h5 style='text-align: center;'>Non-stationary (fail to reject Ho)</h5>",unsafe_allow_html=True)


    st.markdown('---')
    # ======================== KPSS & ADF Combined ==============================
    st.markdown('### 💡 Combined Conclusion')
    if adf_stationary and kpss_stationary:
        st.markdown(f"<h5>Both tests agree: Series is likely STATIONARY.</h5>", unsafe_allow_html=True)
    elif not adf_stationary and not kpss_stationary:
        st.markdown(f"<h5>Both tests agree: Series is likely NON-STATIONARY.</h5>",unsafe_allow_html=True)
    else:
        st.markdown(f"<h5>Tests disagree: Series may be TREND STATIONARY or need further differencing.</h5>",unsafe_allow_html=True)



    # =============After Differentiation Plt ==============


    df_case3 = pd.DataFrame()

    flag = 0
    if d>0 and D>0:
        df_case3 = train_data.diff(periods=m).diff(periods=d).dropna()
        flag = 1
    elif d>0:
        df_case3 = train_data.diff(periods=d).dropna()
        flag = 1
    elif D>0:
        df_case3 = train_data.diff(periods=m).dropna()
        flag = 1
    else:
        flag = 0


    if flag==1:
        st.markdown('---')
        st.markdown('## 📝 After Differentiation')
        col_diff1, col_diff2, col_diff3, col_diff4 = st.columns([2, 1, 1, 1])
        # Setup for ADF and KPSS
        adf_result_diff = adfuller(df_case3, autolag='AIC')
        adf_pvalue_diff = adf_result_diff[1]
        adf_stationary_diff = adf_pvalue_diff < 0.05

        kpss_result_diff = kpss(df_case3, regression='c', nlags='auto')
        kpss_stat_diff = kpss_result_diff[0]
        kpss_pvalue_diff = kpss_result_diff[1]
        kpss_stationary_diff = kpss_pvalue_diff >= 0.05


        with col_diff1:
            backend.differentiation(df_case3)
        with col_diff2:
            st.markdown("<h4 style='text-align: center;'> ADF Statistic</h4>", unsafe_allow_html=True)
            st.markdown(f"<h5 style='text-align: center;'>{adf_result_diff[0]}</h5>", unsafe_allow_html=True)
            st.markdown("<h4 style='text-align: center;'> KPSS Statistic </h4>", unsafe_allow_html=True)
            st.markdown(f"<h5 style='text-align: center;'>{kpss_stat_diff}</h5>", unsafe_allow_html=True)
        with col_diff3:
            st.markdown("<h4 style='text-align: center;'> ADF p-value</h4>", unsafe_allow_html=True)
            st.markdown(f"<h5 style='text-align: center;'>{adf_pvalue_diff}</h5>", unsafe_allow_html=True)
            st.markdown("<h4 style='text-align: center;'> KPSS p-value </h4>", unsafe_allow_html=True)
            st.markdown(f"<h5 style='text-align: center;'>{kpss_pvalue_diff}</h5>", unsafe_allow_html=True)

        with col_diff4:
            st.markdown("<h4 style='text-align: center;'> Interpretation</h4>", unsafe_allow_html=True)
            if adf_stationary_diff:
                st.markdown(f"<h5 style='text-align: center;'>Stationary (reject Ho)</h5>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h5 style='text-align: center;'>Non-stationary (fail to reject Ho)</h5>",unsafe_allow_html=True)
            st.markdown("<h4 style='text-align: center;'> Interpretation </h4>", unsafe_allow_html=True)
            if kpss_stationary_diff:
                st.markdown(f"<h5 style='text-align: center;'>Stationary (reject Ho)</h5>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h5 style='text-align: center;'>Non-stationary (fail to reject Ho)</h5>",unsafe_allow_html=True)

    # ======================== ACF & PACF ==============================

    st.markdown('---')

    col21,col22 = st.columns(2)
    with col21:
        st.markdown("### 📊 ACF : Autocorrelation Function")
        backend.acf_pacf(df, 'acf', acf_pacf_lag)
    with col22:
        st.markdown("### 📊 PACF : Partial Autocorrelation Function")
        backend.acf_pacf(df, 'pacf', acf_pacf_lag)


    end_time = time.time()
    st.markdown(f"*Time taken: {end_time - start_time} seconds*")
