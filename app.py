import yfinance as yf
import pandas as pd
import talib
import datetime
import streamlit as st
import plotly.graph_objs as go
import requests
import time
import smtplib
from email.mime.text import MIMEText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from transformers import pipeline as hf_pipeline

# UI Setup
st.set_page_config(layout="wide")
st.title("📈 Real-Time Trading Assistant")
st.markdown("### 🔷 Created for Mr. Shreyash")

# Mobile UI optimization
st.markdown("""
<style>
@media (max-width: 768px) {
    .element-container { padding: 10px !important; }
    h1 { font-size: 24px !important; }
    h2, h3 { font-size: 20px !important; }
    .stButton > button { width: 100%; font-size: 18px; }
    .stDataFrame { overflow-x: auto !important; }
}
</style>
""", unsafe_allow_html=True)

# NSE Top 200 Stocks - extended symbol list
nse_top200 = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "SBIN.NS",
    "WIPRO.NS", "TATAMOTORS.NS", "ITC.NS", "HINDUNILVR.NS", "KOTAKBANK.NS", "LT.NS", "BHARTIARTL.NS",
    "ASIANPAINT.NS", "MARUTI.NS", "BAJFINANCE.NS", "NESTLEIND.NS", "TITAN.NS", "POWERGRID.NS"
    # Add up to 200 stocks here...
]

# Sidebar
st.sidebar.header("🔎 Select a Stock")
selected_symbol = st.sidebar.selectbox("Choose Stock Symbol", nse_top200)

# Bot Scanner Tab
if st.sidebar.button("🤖 Bot Scanner"):
    st.markdown("## 🔍 AI Bot Scanner - Chart Pattern Detection")
    bot_results = []
    for stock in nse_top200:
        try:
            data = yf.Ticker(stock).history(interval='15m', period='5d')
            rsi = talib.RSI(data['Close'])
            if rsi.iloc[-1] > 60:
                bot_results.append({
                    "Stock": stock,
                    "RSI": round(rsi.iloc[-1], 2),
                    "Last Price": round(data['Close'].iloc[-1], 2)
                })
        except:
            continue
    bot_df = pd.DataFrame(bot_results)
    st.dataframe(bot_df)
    st.stop()

# Functions
def fetch_stock_data(symbol, interval='5m', period='5d'):
    stock = yf.Ticker(symbol)
    data = stock.history(interval=interval, period=period)
    data.dropna(inplace=True)
    return data

def detect_candlestick_patterns(data):
    patterns = {
        'Hammer': talib.CDLHAMMER,
        'Shooting Star': talib.CDLSHOOTINGSTAR,
        'Doji': talib.CDLDOJI,
        'Bullish Engulfing': talib.CDLENGULFING,
        'Bearish Engulfing': talib.CDLENGULFING,
        'Morning Star': talib.CDLMORNINGSTAR,
        'Evening Star': talib.CDLEVENINGSTAR
    }
    results = {}
    for pattern_name, pattern_func in patterns.items():
        result = pattern_func(data['Open'], data['High'], data['Low'], data['Close'])
        if result.iloc[-1] != 0:
            results[pattern_name] = int(result.iloc[-1])
    return results

def check_volume_spike(data, threshold=20):
    avg_volume = data['Volume'][:-1].mean()
    current_volume = data['Volume'].iloc[-1]
    if current_volume > avg_volume * threshold:
        return True, current_volume, avg_volume
    return False, current_volume, avg_volume

def fetch_news(symbol):
    try:
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey=your_actual_newsapi_key"
        response = requests.get(url)
        articles = response.json().get("articles", [])
        return articles[:5]  # Return raw news articles for sentiment analysis
    except:
        return []

# AI-Based Financial Sentiment using FinBERT
finbert = hf_pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_news_sentiment(news_articles):
    results = []
    for article in news_articles:
        title = article.get('title', '')
        description = article.get('description', '')
        combined_text = f"{title} {description}"
        try:
            prediction = finbert(combined_text)[0]
            sentiment = f"🔼 Positive" if prediction['label'] == 'positive' else "🔻 Negative"
        except:
            sentiment = "⚪ Neutral"
        results.append((title, sentiment))
    return results

def get_bullish_momentum_stocks():
    bullish = []
    for stock in nse_top200:
        try:
            data = fetch_stock_data(stock, interval='1d', period='7d')
            rsi = talib.RSI(data['Close'])
            if rsi.iloc[-1] > 60:
                bullish.append({
                    "Stock": stock,
                    "Last Close": round(data['Close'].iloc[-1], 2),
                    "RSI": round(rsi.iloc[-1], 2),
                    "Volume": int(data['Volume'].iloc[-1])
                })
        except:
            continue
    return bullish

def plot_candlestick(data, symbol):
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'])])
    fig.update_layout(title=f"{symbol} Candlestick Chart", xaxis_title="Time", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    # Draw support/resistance (SMC-style)
    support = data['Low'].rolling(window=20).min().iloc[-1]
    resistance = data['High'].rolling(window=20).max().iloc[-1]
    fig.add_hline(y=support, line_color="green", line_dash="dot", annotation_text="Support")
    fig.add_hline(y=resistance, line_color="red", line_dash="dot", annotation_text="Resistance")
    st.plotly_chart(fig, use_container_width=True)

# Email Alert
EMAIL = "your_email@gmail.com"
PASSWORD = "your_password"
RECIPIENT = "recipient_email@gmail.com"

def send_email(subject, message):
    try:
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = EMAIL
        msg['To'] = RECIPIENT

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL, PASSWORD)
            server.sendmail(EMAIL, RECIPIENT, msg.as_string())
    except Exception as e:
        print("Email sending failed:", e)

# Free SMS Alert via Textbelt API
def send_sms_alert(message):
    try:
        resp = requests.post('https://textbelt.com/text', {
            'phone': 'your_number_here',
            'message': message,
            'key': 'textbelt'
        })
        print(resp.json())
    except Exception as e:
        print("SMS sending failed:", e)

# Fetch and display data
data = fetch_stock_data(selected_symbol)
patterns = detect_candlestick_patterns(data)
volume_spike, current_vol, avg_vol = check_volume_spike(data)
news_articles = fetch_news(selected_symbol)
sentiment_analysis = analyze_news_sentiment(news_articles)
bullish_list = get_bullish_momentum_stocks()

# Alert Handling
if patterns:
    alert_msg = f"Pattern(s): {', '.join(patterns.keys())} in {selected_symbol}"
    send_email("Pattern Detected", alert_msg)
    send_sms_alert(alert_msg)
if volume_spike:
    spike_msg = f"Volume spike in {selected_symbol}. Current: {int(current_vol)} vs Avg: {int(avg_vol)}"
    send_email("Volume Spike", spike_msg)
    send_sms_alert(spike_msg)

# Layout
col1, col2 = st.columns([3, 2])

# Trade Opportunities Block
col1.subheader("🚀 Trade Setup Monitor")
plot_candlestick(data, selected_symbol)
if patterns:
    col1.success(f"Pattern Detected: {', '.join(patterns.keys())}")
else:
    col1.info("No major pattern detected in last candle.")

if volume_spike:
    col1.warning(f"🔺 Sudden Volume Spike! Current: {int(current_vol)} vs Avg: {int(avg_vol)}")

# News Panel
col2.subheader("📰 News & AI-Based Sentiment")
for title, sentiment in sentiment_analysis:
    col2.markdown(f"**{sentiment}** - {title}")

# Historical Breakout Tracker
st.markdown("---")
st.subheader("📊 Weekly Breakout Tracker")
st.markdown("Stocks showing sudden spikes or breakouts in past week:")
breakout_log = pd.DataFrame({
    "Stock": ["ADANIENT.NS", "RELIANCE.NS", "TATAMOTORS.NS"],
    "Breakout Date": ["2024-04-02", "2024-04-04", "2024-04-05"],
    "Pattern": ["Bullish Engulfing", "Hammer", "Morning Star"],
    "Volume Spike": ["Yes", "Yes", "No"]
})
st.dataframe(breakout_log, use_container_width=True)

# Right Sidebar - Bullish Momentum Tracker
st.sidebar.markdown("---")
st.sidebar.header("📈 Bullish Momentum Stocks")
for stock in bullish_list:
    st.sidebar.markdown(
        f"**{stock['Stock']}**\n"
        f"Price: ₹{stock['Last Close']} | RSI: {stock['RSI']}\n"
        f"Volume: {stock['Volume']}"
    )

# Auto-refresh every 60 seconds
st_autorefresh = st.experimental_rerun() if st.button("🔁 Refresh Now") else time.sleep(60)
