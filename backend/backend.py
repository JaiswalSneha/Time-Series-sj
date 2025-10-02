import matplotlib.pyplot as plt
import streamlit as st
from Tools.scripts.make_ctype import values
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np
from statsmodels.tsa.stattools import pacf
import pandas as pd
from prophet import Prophet

# ======================== Filter Store and product ==============================
def filter_data(org_df,test_org_df,store,product,year_selected):
    df = org_df[(org_df['store']==store) & (org_df['product']==product)]
    df = df[df['year'].isin(year_selected)]
    test_df = test_org_df[(test_org_df['store'] == store) & (test_org_df['product'] == product)]
    return df,test_df


def plot_line_graph(df,graph_type='custom'):
    if df.empty:
        return None
    else:
        fig, ax = plt.subplots()
        if graph_type == 'custom':
            ax.plot(df['date'],df['sold'],label='Products sold',color='purple')
        else:
            ax.plot(df['date'], df['sold'], label='Products sold', color='pink')
        ax.set_xlabel('Date',fontsize=10)
        ax.set_ylabel('Product sold',fontsize=10)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        return st.pyplot(fig)


def plot_bar_graph(df,store,product):
    if df.empty:
        return None
    else:
        fig, ax = plt.subplots()
        ax.hist(df['sold'],color = 'orange')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Product sold', fontsize=10)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        # ax.set_title('Checking the Nature of the product sale')
        # ax.grid(True)
        return st.pyplot(fig)

# ======================== Skewness detection ==============================
def interpret_skewness(skew):
        if -0.5 <= skew <= 0.5:
            return "approximately symmetric (good)"
        elif -1 <= skew < -0.5:
            return "moderate left skew (slight)"
        elif 0.5 < skew <= 1:
            return "moderate right skew (slight)"
        elif skew < -1:
            return "extreme left skew (bad)"
        elif skew > 1:
            return "extreme right skew (bad)"
        else:
            return "undefined skewness"

# ======================== trend_season_resid ==============================
def trend_season_resid(df,val,label):

        fig, ax = plt.subplots()
        if label=='trend':
            ax.plot(df['date'], val,color='blue')
        elif label=='season':
            ax.plot(df['date'], val,color='green')
        elif label=='residual':
            ax.plot(df['date'], val,color='red')
        ax.set_xlabel('Date')
        ax.set_ylabel('Product sold')
        # ax.set_title(f'Checking the {label} of the product sale')
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        return st.pyplot(fig)






# ======================== ACF & PACF ==============================
def acf_pacf(df,val,acf_pacf_lag):
    if df.empty:
        return None
    elif val=='acf':
        fig,ax = plt.subplots()
        plot_acf(df['sold'],lags = acf_pacf_lag,ax = ax,color='green')
        # ax.set_title('Autocorrelation Function (ACF) Plot')
        st.pyplot(fig)
        return None
    elif val=='pacf':
        fig, ax = plt.subplots()
        plot_pacf(df['sold'],lags = acf_pacf_lag, ax=ax,color='green')
        # ax.set_title('Autocorrelation Function (PACF) Plot')
        st.pyplot(fig)

        # pacf_vals = pacf(df['sold'], nlags=acf_pacf_lag)
        # # Calculate standard error and critical bounds
        # n = len(df['sold'])
        # se = 1 / np.sqrt(n)
        # crit = 1.96 * se
        # # Find significant lags
        # significant_lags = [(lag, val,crit) for lag, val in enumerate(pacf_vals) if abs(val) > crit]
        # # st.dataframe(significant_lags)
        return None
    return None


# ====================== Data Transformation =================================

def data_transformation(train_data,test_data ,transformation = 'None'):
    if transformation == 'Min Max Normalization':
        minmax_model = MinMaxScaler()
        train_data['sold'] = minmax_model.fit_transform(train_data['sold'].values.reshape(-1, 1))
        test_data['number_sold'] = minmax_model.fit_transform(test_data['number_sold'].values.reshape(-1, 1))
        return train_data, test_data,minmax_model
    elif transformation == 'Standard Scaler':
        std_scale_model = StandardScaler()
        train_data['sold'] = std_scale_model.fit_transform(train_data['sold'].values.reshape(-1, 1))
        test_data['number_sold'] = std_scale_model.fit_transform(test_data['number_sold'].values.reshape(-1, 1))
        return train_data, test_data, std_scale_model
    elif transformation == 'Square Root Transform':
        train_data['sold'] = np.sqrt(train_data['sold'])
        test_data['number_sold'] = np.sqrt(test_data['number_sold'])
        return train_data, test_data, None
    elif transformation == 'Log Transform':
        train_data['sold'] = np.log(train_data['sold'])
        test_data['number_sold'] = np.log(test_data['number_sold'])
        return train_data, test_data, None
    return None


def final_prediction(test, predict):

    test.index = range(len(test))
    predict.index = range(len(predict))

    fig, ax = plt.subplots()
    ax.plot(test, label='Actual Sales', color='orange')
    ax.plot(predict, label='Predicted Sales', color='blue')
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Products Sold', fontsize=10)
    # Hide x-axis tick labels
    # ax.tick_params(axis='x', labelbottom=False)
    ax.legend()
    # ax.set_title('Actual vs Predicted Sales')
    st.pyplot(fig)

def final_forecast(df,predict,forecast_size):

    forecast_dates = pd.date_range(start=df['date'].values[-1] + pd.Timedelta(days=1), periods=forecast_size, freq='D')
    fig, ax = plt.subplots()
    ax.plot(df['date'].values, df['sold'], label="Historical", color="blue")
    ax.plot(forecast_dates, predict, label="Forecast", color="orange", linestyle="--")
    ax.axis('tight')
    ax.legend()
    st.pyplot(fig)

def prophet_model(df,forecast_size):
    m = Prophet()
    df_pro = df.copy()
    df_pro.rename(columns={'date':'ds','sold':'y'}, inplace=True)
    df_pro = df_pro[['ds', 'y']]
    df_pro.columns = ['ds', 'y']
    model = m.fit(df_pro)
    future = m.make_future_dataframe(periods=forecast_size, freq='D')
    forecast = m.predict(future)
    forecast_tail = forecast.tail(forecast_size)['yhat']
    forecast_upper = forecast.tail(forecast_size)['yhat_upper']
    forecast_lower = forecast.tail(forecast_size)['yhat_lower']
    # st.text(forecast_tail)

    forecast_dates = pd.date_range(start=df['date'].values[-1] + pd.Timedelta(days=1), periods=forecast_size, freq='D')
    fig, ax = plt.subplots()
    ax.plot(df['date'].values, df['sold'], label="Historical", color="green")
    ax.plot(forecast_dates, forecast_tail  , label="Forecast", color="red", linestyle="--")
    ax.plot(forecast_dates, forecast_upper, label="Forecast", color="orange")
    ax.plot(forecast_dates, forecast_lower, label="Forecast", color="orange")
    ax.legend()
    st.pyplot(fig)


def differentiation(df_case3):
    fig, ax = plt.subplots()
    ax.plot(df_case3.index, df_case3.values, label='After Differentiation', color='purple')
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Product sold', fontsize=10)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    return st.pyplot(fig)



def check_sarimax_valid(p, d, q, PS, DS, QS, m):
    # Rule 2: m = 0 case
    if m == 0:
        if PS == 0 and DS == 0 and QS == 0:
            return None
        else:
            return "❌ Invalid: Seasonal terms (P,D,Q) require m > 1."

    # Rule 3: m = 1 case
    if m == 1:
        return "❌ Invalid: Seasonal period m=1 is not allowed."


    # Rule 4: Check overlap in AR lags
    nonseasonal_ar_lags = set(range(1, p+1))
    seasonal_ar_lags = set([k*m for k in range(1, PS+1)])
    if nonseasonal_ar_lags & seasonal_ar_lags:
        return f"❌ Invalid: Overlap in AR lags → {nonseasonal_ar_lags & seasonal_ar_lags}"

    # Rule 5: Check overlap in MA lags
    nonseasonal_ma_lags = set(range(1, q+1))
    seasonal_ma_lags = set([k*m for k in range(1, QS+1)])
    if nonseasonal_ma_lags & seasonal_ma_lags:
        return f"❌ Invalid: Overlap in MA lags → {nonseasonal_ma_lags & seasonal_ma_lags}"

    return None