import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
import urllib.request
import io
import json
import concurrent.futures
from google import genai 
from google.genai import types
from pydantic import BaseModel

# --- ⚙️ การตั้งค่าหน้าเว็บ (Premium Dark Theme) ---
st.set_page_config(
    page_title="PropFirmX - AI Trader Dashboard", 
    layout="wide", 
    page_icon="🟩",
    initial_sidebar_state="expanded"
)

# ดึง API Key จาก Secrets
API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# --- 📊 โครงสร้างข้อมูล JSON สำหรับ AI ---
class StockAnalysisResult(BaseModel):
    sentiment: str
    risk_level: str
    support_zone: str
    resistance_zone: str
    action_suggestion: str
    detailed_reason: str

# --- 🔄 ระบบจำข้อมูลและสถานะเว็บ ---
if 'active_ticker' not in st.session_state:
    st.session_state.active_ticker = "AAPL"
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = []
if 'ai_analysis' not in st.session_state:
    st.session_state.ai_analysis = None
if 'timeframe' not in st.session_state:
    st.session_state.timeframe = "6M (รายวัน)"

tf_mapping = {
    "1D (1 นาที)": {"period": "1d", "interval": "1m", "tv": "1"},
    "1W (15 นาที)": {"period": "7d", "interval": "15m", "tv": "15"},
    "1M (รายวัน)": {"period": "1mo", "interval": "1d", "tv": "D"},
    "6M (รายวัน)": {"period": "6mo", "interval": "1d", "tv": "D"},
    "1Y (รายสัปดาห์)": {"period": "1y", "interval": "1wk", "tv": "W"}
}
current_tf = tf_mapping[st.session_state.timeframe]

# --- 🎨 🗜️ การฉีด CSS เข้าไปเพื่อคุมโทนสีแบบลึก (Prop Firm Style เหมือนในภาพ) ---
st.markdown("""
<style>
    /* ตั้งค่าพื้นหลังหลักให้เป็นสีกรมท่าเข้มลึก */
    .stApp {
        background-color: #0b0e14;
        color: #f1f5f9;
    }
    /* ปรับแต่งสไตล์การ์ดข้อมูล */
    .prop-card {
        background-color: #111622;
        border: 1px solid #1e293b;
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    .prop-card-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .prop-card-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
    }
    .prop-card-delta {
        font-size: 0.85rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 🛠️ ฟังก์ชันดึงข้อมูลดั้งเดิม ---
def fetch_data_with_header(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response: 
        return response.read()

@st.cache_data(ttl=3600)
def load_market_tickers(market):
    try:
        if market == "S&P 500":
            csv_bytes = fetch_data_with_header('https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv')
            return pd.read_csv(io.BytesIO(csv_bytes))['Symbol'].tolist()
        elif "NASDAQ" in market:
            html_bytes = fetch_data_with_header('https://en.wikipedia.org/wiki/Nasdaq-100')
            tables = pd.read_html(io.StringIO(html_bytes.decode('utf-8')))
            for df in tables:
                if 'Ticker' in df.columns or 'Symbol' in df.columns:
                    return df['Ticker' if 'Ticker' in df.columns else 'Symbol'].tolist()
        else:
            html_bytes = fetch_data_with_header('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')
            tables = pd.read_html(io.StringIO(html_bytes.decode('utf-8')))
            for df in tables:
                if 'Symbol' in df.columns: return df['Symbol'].tolist()
    except: pass
    return ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA']

def clean_df_columns(df, ticker):
    if df is None or df.empty: return pd.DataFrame()
    df_cleaned = df.copy()
    if isinstance(df_cleaned.columns, pd.MultiIndex):
        if ticker in df_cleaned.columns.get_level_values(1): df_cleaned = df_cleaned.xs(ticker, axis=1, level=1)
        elif ticker in df_cleaned.columns.get_level_values(0): df_cleaned = df_cleaned.xs(ticker, axis=1, level=0)
        else: df_cleaned.columns = df_cleaned.columns.get_level_values(0)
    df_cleaned.index = pd.to_datetime(df_cleaned.index)
    cleaned_dict = {col: df_cleaned[col].squeeze() for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in df_cleaned.columns}
    return pd.DataFrame(cleaned_dict, index=df_cleaned.index)

def scan_single_stock(ticker):
    try:
        raw_df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=False, progress=False)
        df = clean_df_columns(raw_df, ticker).dropna()
        if not df.empty and len(df) >= 30:
            close = df['Close']
            df['MA20'] = close.rolling(window=20).mean()
            df['STD'] = close.rolling(window=20).std()
            df['BB_Upper'] = df['MA20'] + (df['STD'] * 2)
            df['BB_Lower'] = df['MA20'] - (df['STD'] * 2)
            df['Vol_MA'] = df['Volume'].rolling(window=20).mean()
            delta = close.diff()
            rs = delta.where(delta>0,0).ewm(alpha=1/14).mean() / -delta.where(delta<0,0).ewm(alpha=1/14).mean()
            df['RSI'] = 100 - (100 / (1 + rs))
            df['MACD'] = close.ewm(span=12).mean() - close.ewm(span=26).mean()
            df['Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Hist'] = df['MACD'] - df['Signal']
            
            c, pc = float(close.iloc[-1]), float(close.iloc[-2])
            u, pu = float(df['BB_Upper'].iloc[-1]), float(df['BB_Upper'].iloc[-2])
            v, vm = float(df['Volume'].iloc[-1]), float(df['Vol_MA'].iloc[-1])
            rsi = float(df['RSI'].iloc[-1])
            mh, pmh = float(df['MACD_Hist'].iloc[-1]), float(df['MACD_Hist'].iloc[-2])
            
            if pc <= pu and c > u and v > (vm * 1.5): return [ticker, round(c, 2), "🟢 BUY (BB Breakout)"]
            elif pmh <= 0 and mh > 0: return [ticker, round(c, 2), "⚔️ MACD Golden Cross"]
            elif rsi < 30: return [ticker, round(c, 2), "📉 RSI Oversold"]
            elif rsi > 70: return [ticker, round(c, 2), "🔥 RSI Overbought"]
            else: return [ticker, round(c, 2), "⚪ No Signal"]
    except: pass
    return None

# ==========================================
# 📌 SIDEBAR CONTROLLER (แถบซ้ายดึงข้อมูล)
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='color:#38bdf8; text-align:center;'>📊 PropFirmX AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:0.8rem; color:gray;'>Overview of your trading performance</p>", unsafe_allow_html=True)
    st.divider()
    
    st.write("🔍 **พิมพ์สัญลักษณ์หุ้นเพื่อวิเคราะห์**")
    search_ticker = st.text_input("ชื่อย่อหุ้น (Ticker):", value=st.session_state.active_ticker).upper()
    if st.button("⚡ ดึงข้อมูลราคากราฟ", use_container_width=True):
        st.session_state.active_ticker = search_ticker
        st.session_state.ai_analysis = None
        st.rerun()
        
    st.write("⏱️ **เลือก Timeframe**")
    selected_tf = st.selectbox("ช่วงเวลา:", list(tf_mapping.keys()), index=list(tf_mapping.keys()).index(st.session_state.timeframe))
    if selected_tf != st.session_state.timeframe:
        st.session_state.timeframe = selected_tf
        st.rerun()
        
    st.divider()
    st.write("🏛️ **ระบบสแกนหากลุ่มหุ้นนำตลาด**")
    selected_market = st.selectbox("ตลาดหุ้นเป้าหมาย", ["NASDAQ 100", "S&P 500", "Dow Jones 30"])
    
    if st.button("🚀 สแกนระบบสมองกลควอนตัม", type="primary", use_container_width=True):
        st.session_state.scan_results = []
        tickers_list = load_market_tickers(selected_market)
        
        with st.status(f"กำลังกวาดข้อมูลตลาด {selected_market}...", expanded=False) as status:
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
                futures = {executor.submit(scan_single_stock, t): t for t in tickers_list}
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    if res and "No Signal" not in res[2]: results.append(res)
            st.session_state.scan_results = results
            status.update(label=f"✅ สแกนเสร็จสิ้น! พบสตรีมสัญญาณ {len(results)} ตัว", state="complete")

# ==========================================
# 📌 MAIN WORKSPACE (ถอดแบบจากภาพเลย์เอาต์เป๊ะๆ)
# ==========================================
ticker = st.session_state.active_ticker

# ดึงข้อมูลมาคำนวณเบื้องหลัง
try:
    info = yf.Ticker(ticker).info
    current_price = info.get('currentPrice', 0.0)
    prev_close = info.get('previousClose', 1.0)
    price_change = ((current_price - prev_close) / prev_close) * 100
    sector = info.get('sector', 'N/A')
    mcap = f"${info.get('marketCap', 0)/1e9:.2f}B" if info.get('marketCap') else "N/A"
except:
    current_price, price_change, sector, mcap = 150.00, 1.25, "Technology", "N/A"

# 1️⃣ TOP ROW METRICS GRID (ถอดแบบมาจากแถบด้านบนสุดในภาพ)
st.markdown("### Account Metrics & Targets")
m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)

with m_col1:
    delta_color = "#22c55e" if price_change >= 0 else "#ef4444"
    st.markdown(f"""
    <div class="prop-card">
        <div class="prop-card-title">Account Balance</div>
        <div class="prop-card-value">${current_price*150:,.2f}</div>
        <div class="prop-card-delta" style="color:{delta_color};">+{price_change:.2f}% (Today)</div>
    </div>
    """, unsafe_allow_html=True)
with m_col2:
    st.markdown(f"""
    <div class="prop-card">
        <div class="prop-card-title">Equity</div>
        <div class="prop-card-value">${current_price*152:,.2f}</div>
        <div class="prop-card-delta" style="color:#22c55e;">Floating Gain</div>
    </div>
    """, unsafe_allow_html=True)
with m_col3:
    st.markdown(f"""
    <div class="prop-card">
        <div class="prop-card-title">Profit / Loss</div>
        <div class="prop-card-value" style="color:#22c55e;">+${current_price*2.5:,.2f}</div>
        <div class="prop-card-delta" style="color:#22c55e;">🟢 Target Path</div>
    </div>
    """, unsafe_allow_html=True)
with m_col4:
    st.markdown("""
    <div class="prop-card">
        <div class="prop-card-title">Profit Target</div>
        <div class="prop-card-value">$3,000.00</div>
        <div class="prop-card-delta" style="color:gray;">80.35% Complete</div>
    </div>
    """, unsafe_allow_html=True)
with m_col5:
    st.markdown("""
    <div class="prop-card">
        <div class="prop-card-title">Max Daily Loss</div>
        <div class="prop-card-value" style="color:#ef4444;">$2,500.00</div>
        <div class="prop-card-delta" style="color:gray;">Used: $420.35</div>
    </div>
    """, unsafe_allow_html=True)
with m_col6:
    st.markdown("""
    <div class="prop-card">
        <div class="prop-card-title">Drawdown Limit</div>
        <div class="prop-card-value">$1,250.00</div>
        <div class="prop-card-delta" style="color:gray;">Safe Zone</div>
    </div>
    """, unsafe_allow_html=True)

# 2️⃣ MIDDLE SECTION: กราฟฝั่งซ้าย + แดชบอร์ด Rules ฝั่งขวา (แบ่งสัดส่วน 3:1 ตามภาพ)
col_left_main, col_right_panel = st.columns([3, 1])

with col_left_main:
    st.markdown(f"#### 📈 Live Market Technical Chart: {ticker} ({st.session_state.timeframe})")
    tradingview_html = f"""
    <div class="tradingview-widget-container" style="height:400px;width:100%">
      <div id="tradingview_chart" style="height:100%;width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{ticker}",
        "interval": "{current_tf['tv']}",
        "timezone": "Asia/Bangkok",
        "theme": "dark",
        "style": "1",
        "locale": "th",
        "enable_publishing": false,
        "hide_side_toolbar": false,
        "studies": ["RSI@tv-basicstudies", "MASimple@tv-basicstudies"],
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """
    components.html(tradingview_html, height=410)

with col_right_panel:
    st.markdown("#### 🛡️ Account Status & Rules")
    with st.container(border=True):
        st.write("🔴 **Challenge Type:** `2-Step Challenge`")
        st.write("📆 **Days Remaining:** `18 Days`")
        st.divider()
        st.write("📊 **Rules Progress**")
        
        st.caption("Profit Target (80.35%)")
        st.progress(0.80)
        
        st.caption("Max Daily Loss (16.81%)")
        st.progress(0.16)
        
        st.caption("Max Overall Loss (25.61%)")
        st.progress(0.25)
        
        st.caption("Consistency Rule (92%)")
        st.progress(0.92)

st.divider()

# 3️⃣ BOTTOM SECTION: ตารางเทรดล่าสุด + วงแหวนประมวลผลระบบ AI + ปฏิทินเศรษฐกิจ
col_b1, col_b2, col_b3 = st.columns([1.4, 0.9, 0.9])

with col_b1:
    st.markdown("#### 📋 Recent Market Signals / Scanned Tickers")
    if st.session_state.scan_results:
        df_scan = pd.DataFrame(st.session_state.scan_results, columns=["Ticker", "Price", "Signal"])
        st.dataframe(df_scan, use_container_width=True, hide_index=True, height=220)
    else:
        # แสดง Mock ข้อมูลเสมือนจริงแบบในภาพถ้าไม่มีการกดสแกน
        mock_trades = pd.DataFrame([
            {"Symbol": "EURUSD", "Direction": "BUY", "Size": "1.00", "Price": "$1.0852", "P/L (USD)": "+$405.00"},
            {"Symbol": "XAUUSD", "Direction": "SELL", "Size": "0.50", "Price": "$2,353.45", "P/L (USD)": "+$266.50"},
            {"Symbol": "GBPUSD", "Direction": "BUY", "Size": "1.00", "Price": "$1.2745", "P/L (USD)": "+$333.00"},
            {"Symbol": "AAPL", "Direction": "BUY", "Size": "20.00", "Price": f"${current_price}", "P/L (USD)": "Pending"}
        ])
        st.dataframe(mock_trades, use_container_width=True, hide_index=True, height=220)

with col_b2:
    st.markdown("#### 🎯 Performance Summary")
    with st.container(border=True):
        # สร้างวงแหวน Win Rate เลียนแบบในรูปด้วย HTML / CSS Base
        st.markdown("""
        <div style="text-align: center; padding: 10px;">
            <div style="display: inline-block; width: 100px; height: 100px; border-radius: 50%; background: conic-gradient(#22c55e 73%, #ef4444 0); padding: 10px;">
                <div style="width: 80px; height: 80px; border-radius: 50%; background-color: #111622; margin: 0 auto; line-height: 80px; font-weight: bold; font-size: 1.3rem;">73%</div>
            </div>
            <p style="margin-top: 5px; font-size: 0.9rem; color: #94a3b8;">Win Rate Metrics</p>
        </div>
        """, unsafe_allow_html=True)
        
        # รายละเอียดสถิติประกอบด้านล่างวงแหวน
        st.text("Total Trades: 142")
        st.text("Winning Trades: 104")
        st.text("Losing Trades: 38")

with col_b3:
    st.markdown("#### 🔮 Deep AI Analyst Insights")
    # ดึงค่าเทคนิคอลดิบมาเตรียมส่งให้ AI
    try:
        raw_df = yf.download(ticker, period="1mo", interval="1d", auto_adjust=False, progress=False)
        df_latest = clean_df_columns(raw_df, ticker).dropna()
        c_p = float(df_latest['Close'].iloc[-1])
        delta = df_latest['Close'].diff()
        rs = delta.where(delta>0,0).ewm(alpha=1/14).mean() / -delta.where(delta<0,0).ewm(alpha=1/14).mean()
        rsi_val = float((100 - (100 / (1 + rs))).iloc[-1])
    except:
        c_p, rsi_val = 150.0, 55.0

    if st.button("🧠 รันสมองกลวิเคราะห์กลยุทธ์", type="secondary", use_container_width=True):
        if not API_KEY:
            st.error("กรุณากรอก GEMINI_API_KEY ใน secrets")
        else:
            with st.spinner("AI กำลังแกะสัญญารูปแบบราคา..."):
                prompt = f"วิเคราะห์หุ้น {ticker} ราคาล่าสุด ${c_p:.2f} ค่า RSI อยู่ที่ {rsi_val:.2f} ออกแบบโซนรับต้านทางจิตวิทยาให้ตรงตามสเปกกองทุนเทรด"
                try:
                    client = genai.Client(api_key=API_KEY)
                    response = client.models.generate_content(
                        model='gemini-3.1-flash-lite',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=StockAnalysisResult,
                            temperature=0.2,
                            # จุดที่ 1: ปรับแต่งข้อความคุมพฤติกรรม AI ให้บังคับตอบทุกช่องเป็นภาษาไทยย่อสั้นๆ
                            system_instruction="คุณคือนักบริหารความเสี่ยงมืออาชีพของกองทุนระดับโลก ต้องวิเคราะห์และตอบกลับทุกฟิลด์เป็นภาษาไทยที่กระชับ คมคาย 100% ห้ามใช้ภาษาอังกฤษปนในผลลัพธ์เด็ดขาด เช่น ให้ใช้คำว่า ขาขึ้น, ขาลง, ไซด์เวย์, เสี่ยงสูง, เสี่ยงปานกลาง, ถือ/รอ ดูแนวโน้ม"
                        )
                    )
                    st.session_state.ai_analysis = json.loads(response.text)
                except Exception as e:
                    st.error(f"Error: {e}")

    # --- ส่วนแสดงผลบทวิเคราะห์ (ปรับเปลี่ยนหัวข้อภาษาไทยตามภาพ image_d969a9.png) ---
    if st.session_state.ai_analysis:
        res = st.session_state.ai_analysis
        
        # จุดที่ 2: เปลี่ยนชื่อหัวข้อแสดงผลเป็นภาษาไทย
        st.markdown(f"**มุมมองเทรนด์:** `{res['sentiment']}`")
        st.markdown(f"**ระดับความเสี่ยง:** `{res['risk_level']}` | **คำแนะนำ:** `{res['action_suggestion']}`")
        st.markdown(f"🛡  **แนวรับ:** `{res['support_zone']}` | 🚀 **แนวต้าน:** `{res['resistance_zone']}`")
        
        # กล่องแสดงผลเหตุผลเชิงลึก
        st.markdown(f"""
        <div style="background-color: #161b26; border-left: 4px solid #38bdf8; padding: 12px; border-radius: 6px; margin-top: 12px; font-size: 0.9rem; color: #cbd5e1; line-height: 1.5;">
            <strong style="color: #38bdf8;">📝 ความเห็นและบทวิเคราะห์เชิงลึก:</strong><br>
            <div style="margin-top: 6px;">{res['detailed_reason']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("💡 คลิกปุ่มด้านบนเพื่อให้ AI เจาะลึกโครงสร้างราคาแบบ Prop Firm Strategy")