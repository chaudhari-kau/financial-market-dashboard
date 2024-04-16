import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from alpha_vantage.fundamentaldata import FundamentalData
from stocknews import StockNews
import plotly.graph_objs as go
from datetime import datetime

def get_first_trading_day(ticker):
    # Downloading data from a very early start date
    data = yf.download(ticker, start="1900-01-01")
    # The first index is the first trading day
    first_trading_day = data.index[0]
    return first_trading_day

def calculate_financial_ratios(ticker):
    # Initialize all variables at the start to ensure they are in scope, even if the following code fails
    pb_ratio = np.nan
    pe_ratio = np.nan
    peg_ratio = np.nan
    dividend_yield = np.nan
    current_price = np.nan

    # Fetch data
    stock = yf.Ticker(ticker)

    # Get stock info
    try:
        info = stock.info
        current_price = info.get('currentPrice', np.nan)
        print("Got current price", current_price)
        
        # Calculate P/B Ratio
        book_value_per_share = info.get('bookValue', np.nan)
        print("Got book value", book_value_per_share)
        pb_ratio = current_price / book_value_per_share if book_value_per_share and not np.isnan(book_value_per_share) else np.nan

        # Calculate P/E Ratio
        earnings_per_share = info.get('trailingEps', np.nan)
        print("Got earnings per share", earnings_per_share)
        pe_ratio = current_price / earnings_per_share if earnings_per_share and not np.isnan(earnings_per_share) else np.nan

        # PEG Ratio (This might not be available directly)
        peg_ratio = info.get('pegRatio', np.nan)
        print("Got pegratio", peg_ratio)

        # Calculate Dividend Yield (as a percentage)
        dividend_yield = info.get('dividendYield', 0) * 100
        print("DivYield", dividend_yield)

    except Exception as e:
        print(f"An error occurred: {e}")

    return pb_ratio, pe_ratio, peg_ratio, dividend_yield

def format_timestamp(timestamp_str):
    # Parse the original timestamp string, assuming the format is like 'Wed, 10 Apr 2024 21:17:49 +0000'
    dt = datetime.strptime(timestamp_str, '%a, %d %b %Y %H:%M:%S %z')
    
    # Remove the timezone information and convert to AM/PM format
    formatted_timestamp = dt.strftime('%a, %d %b %Y %I:%M:%S %p')
    
    return formatted_timestamp

# Set page configuration to wide mode
st.set_page_config(layout="wide")

st.title('📈📉 Stock Dashboard')
# Create columns for inputs
col1, col2, col3 = st.columns(3)

df = pd.read_csv('stockname_data.csv')

with col1:
    # Ticker input
    stock_name = st.selectbox('Stock Name', df['stockname'])
    ticker = df[df['stockname'] == stock_name]['shortname'].item()

# Get the first trading day
first_day = get_first_trading_day(ticker)
# Set today's date
today = datetime.today().date()

with col2:
    # Start date input
    start_date = st.date_input('Start Date', first_day)

with col3:
    # End date input
    end_date = st.date_input('End Date', today)

# Download data
nasdaq_data = yf.download('^IXIC', start=start_date, end=end_date)
sp500_data = yf.download('^GSPC', start=start_date, end=end_date)
data = yf.download(ticker, start=start_date, end=end_date)

# Calculate percentage change
data['Pct Change'] = data['Adj Close'].pct_change().fillna(0) + 1
nasdaq_data['Pct Change'] = nasdaq_data['Adj Close'].pct_change().fillna(0) + 1
sp500_data['Pct Change'] = sp500_data['Adj Close'].pct_change().fillna(0) + 1

# Cumulative product to get growth from start date
data['Cumulative'] = data['Pct Change'].cumprod() - 1
nasdaq_data['Cumulative'] = nasdaq_data['Pct Change'].cumprod() - 1
sp500_data['Cumulative'] = sp500_data['Pct Change'].cumprod() - 1

candlestick_container = st.container()

with candlestick_container:
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'])])

    # Updating layout of the chart
    fig.update_layout(title=f'Candlestick Chart for {stock_name}',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False)  # Hide range slider
    
    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

pb_col, pe_col, peg_col, divy_col = st.columns(4)
pb_ratio, pe_ratio, peg_ratio, dividend_yield = calculate_financial_ratios(ticker)

# Display financial ratios in a row below the chart
ratio1, ratio2, ratio3, ratio4 = st.columns(4)
with ratio1:
    st.metric(label="P/B Ratio", value=f"{pb_ratio:.4f}")
with ratio2:
    st.metric(label="P/E Ratio", value=f"{pe_ratio:.4f}")
with ratio3:
    st.metric(label="PEG Ratio", value=f"{peg_ratio:.4f}")
with ratio4:
    st.metric(label="Dividend Yield", value=f"{dividend_yield:.4f}%")

compare_container = st.container()
with compare_container:
    # New comparison plot for percentage change
    comp_fig_pct = go.Figure()
    comp_fig_pct.add_trace(go.Scatter(x=data.index, y=data['Cumulative'], mode='lines', name=stock_name))
    comp_fig_pct.add_trace(go.Scatter(x=nasdaq_data.index, y=nasdaq_data['Cumulative'], mode='lines', name='NASDAQ'))
    comp_fig_pct.add_trace(go.Scatter(x=sp500_data.index, y=sp500_data['Cumulative'], mode='lines', name='S&P 500'))
    comp_fig_pct.update_layout(title=f"{stock_name} vs NASDAQ vs S&P 500: Cumulative Return", xaxis_title='Date', yaxis_title='Cumulative Return')
    st.plotly_chart(comp_fig_pct, use_container_width=True)


pricing_data, fundamental_data, news = st.tabs(["Pricing Data", "Fundamental Data", "Top 5 News"])
with pricing_data:
    st.header('Price Movements')
    data2 = data
    data2['% Change'] = data['Adj Close'] / data['Adj Close'].shift(1) - 1
    data2.dropna (inplace = True)
    st.dataframe(data2)

    value1, value2, value3 = st.columns(3)
    with value1:
        annual_return = data2['% Change'].mean()*252*100
        st.metric(label="Annual Return", value=f"{annual_return:.4f}%")
    with value2:
        stdev = np.std(data2['% Change'])*np.sqrt(252)
        st.metric(label="Standard Deviation", value=f"{stdev:.4f}%")
    with value3:
        st.metric(label="Risk Adj. Return", value=f"{annual_return/(stdev*100):.4f}")


with fundamental_data:
    st.subheader('Balance Sheet')
    try:
        key = 'O882PKFFFI4TKZIN'
        fd = FundamentalData (key, output_format = 'pandas')
        balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
        bs = balance_sheet.T[2:]
        bs.columns = list(balance_sheet.T.iloc[0])
        st.write(bs)
        st.subheader('Income Statement')
        income_statement = fd.get_income_statement_annual (ticker)[0]
        is1 = income_statement.T[2:]
        is1.columns = list(income_statement.T.iloc[0])
        st.write(is1)
        st.subheader('Cash Flow Statement')
        cash_flow = fd.get_cash_flow_annual(ticker)[0]
        cf = cash_flow.T[2:]
        cf.columns = list(cash_flow.T.iloc[0])
        st.write(cf)
    except Exception as e:
        st.markdown("Balance sheet is currently unavailable, please try again later.")
        print(f"Exception in AlphaVantage API Call: {e}")

with news:
    st.header(f'News of {stock_name}')
    sn = StockNews([ticker], save_news=False)
    df_news = sn.read_rss()
    for i in range(5):
        st.subheader(df_news['title'][i])
        publish_time = format_timestamp(df_news['published'][i])
        st.markdown(f"*{publish_time}*")
        st.markdown(df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i]
        st.write('Title Sentiment:', title_sentiment)
        news_sentiment = df_news['sentiment_summary'][i]
        st.write('News Sentiment', news_sentiment)


 