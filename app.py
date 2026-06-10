import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import json
import urllib.request
import io
import concurrent.futures
from google import genai
from google.genai import types
from pydantic import BaseModel

# 🔑 ตั้งค่า API 
# ลบบรรทัดเดิมที่ใส่ Key ตรงๆ ทิ้งไป แล้วแทนที่ด้วย 2 บรรทัดนี้:
import streamlit as st

# ดึง Key มาจากตู้เซฟของ Streamlit
API_KEY = st.secrets["GEMINI_API_KEY"]

st.set_page_config(page_title="Pro Stock Scanner", layout="wide", page_icon="📈")

# --- โครงสร้าง JSON สำหรับ AI ---
class StockAnalysisResult(BaseModel):
    sentiment: str
    risk_level: str
    support_zone: str
    resistance_zone: str
    action_suggestion: str
    detailed_reason: str

# --- ตัวแปร Session ---
if 'ticker' not in st.session_state:
    st.session_state.ticker = "AAPL"
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = []
if 'is_scanning' not in st.session_state:
    st.session_state.is_scanning = False

# --- ฟังก์ชันสแกนหุ้น (ดึงมาจากเวอร์ชันเดิมของคุณ) ---
def get_tickers(market):
    try:
        req = urllib.request.Request('https://en.wikipedia.org/wiki/Nasdaq-100', headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            html = response.read()
        tables = pd.read_html(io.StringIO(html.decode('utf-8')))
        for df in tables:
            if 'Ticker' in df.columns: return df['Ticker'].tolist()
            if 'Symbol' in df.columns: return df['Symbol'].tolist()
    except:
        pass
    return ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'AMD', 'NFLX'] # ตัวสำรอง

def scan_single_stock(ticker):
    try:
        raw_df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=False, progress=False)
        if raw_df is None or raw_df.empty: return None
        
        # จัดการ MultiIndex ของ yfinance ล่าสุด
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)
            
        df = raw_df.dropna()
        if len(df) < 30: return None
        
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
        df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
        
        c, pc = float(close.iloc[-1]), float(close.iloc[-2])
        u, pu = float(df['BB_Upper'].iloc[-1]), float(df['BB_Upper'].iloc[-2])
        l = float(df['BB_Lower'].iloc[-1])
        v, vm = float(df['Volume'].iloc[-1]), float(df['Vol_MA'].iloc[-1])
        rsi = float(df['RSI'].iloc[-1])
        mh, pmh = float(df['MACD_Hist'].iloc[-1]), float(df['MACD_Hist'].iloc[-2])
        
        signal = "⚪ No Signal"
        if pc <= pu and c > u and v > (vm * 1.5): signal = "🟢 BUY (BB Breakout)"
        elif pmh <= 0 and mh > 0: signal = "⚔️ MACD Golden Cross"
        elif rsi < 30: signal = "📉 RSI Oversold"
        elif rsi > 70: signal = "🔥 RSI Overbought"
        elif c < l: signal = "🔴 BB Breakdown"
        
        if signal != "⚪ No Signal":
            return {"Ticker": ticker, "Price": round(c, 2), "Signal": signal}
    except:
        pass
    return None

# --- ฟังก์ชัน AI ---
def analyze_with_ai(ticker):
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period="1mo")
        if hist.empty: return "❌ ไม่พบข้อมูลราคาหุ้น"
        current_price = hist['Close'].iloc[-1]
        
        prompt = f"""วิเคราะห์เทคนิคอลหุ้น {ticker} ราคาปัจจุบัน ${current_price:.2f}
        วิเคราะห์แนวโน้ม แนวรับ แนวต้าน และให้คำแนะนำ ตอบตาม JSON schema ที่กำหนด"""
        
        client = genai.Client(api_key=API_KEY)
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite', 
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StockAnalysisResult,
                temperature=0.2,
                system_instruction="คุณคือผู้ช่วยวิเคราะห์หุ้น ตอบเป็นภาษาไทยที่สุภาพและเข้าใจง่าย"
            )
        )
        result = json.loads(response.text)
        md_text = f"### 📊 มุมมอง: {result['sentiment']} | ⚠️ ความเสี่ยง: {result['risk_level']}\n"
        md_text += f"**🎯 แนะนำ:** {result['action_suggestion']}\n\n"
        md_text += f"**🛡️ แนวรับ:** {result['support_zone']} | **🚀 แนวต้าน:** {result['resistance_zone']}\n"
        md_text += "---\n**📝 บทวิเคราะห์:**\n" + result['detailed_reason']
        return md_text
    except Exception as e:
        return f"❌ เกิดข้อผิดพลาดกับ AI: {str(e)}"

# ==========================================
# UI: Sidebar (เมนูควบคุม)
# ==========================================
with st.sidebar:
    st.title("👑 Pro Scanner")
    st.markdown("ระบบสแกนหุ้น & AI วิเคราะห์")
    
    # ค้นหาหุ้น
    st.write("🔍 **ค้นหาหุ้นที่ต้องการ**")
    col1, col2 = st.columns([3, 1])
    with col1:
        new_ticker = st.text_input("Ticker", value=st.session_state.ticker, label_visibility="collapsed")
    with col2:
        if st.button("ค้นหา", use_container_width=True):
            st.session_state.ticker = new_ticker.upper()
            st.rerun()

    st.divider()
    
    # ระบบสแกนหุ้น
    st.write("🚀 **ระบบสแกนตลาด (NASDAQ 100)**")
    if st.button("เริ่มสแกนหุ้น", type="primary", use_container_width=True):
        st.session_state.scan_results = []
        tickers = get_tickers("NASDAQ")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        # ใช้ ThreadPool เพื่อให้สแกนเร็วขึ้น
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(scan_single_stock, t): t for t in tickers}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                progress_bar.progress((i + 1) / len(tickers))
                status_text.text(f"กำลังสแกน... {i+1}/{len(tickers)}")
                res = future.result()
                if res:
                    results.append(res)
        
        st.session_state.scan_results = results
        status_text.text(f"✅ สแกนเสร็จสิ้น! พบ {len(results)} ตัวที่เข้าเงื่อนไข")
        progress_bar.empty()
        st.rerun()

    # แสดงผลสแกน
    if st.session_state.scan_results:
        st.write("📋 **หุ้นที่เข้าเงื่อนไข:**")
        for item in st.session_state.scan_results:
            # สร้างปุ่มสำหรับหุ้นแต่ละตัวที่สแกนเจอ
            if st.button(f"{item['Signal']} | {item['Ticker']} (${item['Price']})", key=f"scan_{item['Ticker']}", use_container_width=True):
                st.session_state.ticker = item['Ticker']
                st.rerun()

# ==========================================
# UI: Main Area (พื้นที่หลัก)
# ==========================================
st.header(f"📈 กราฟ Advanced Chart: {st.session_state.ticker}")

# 1. 🌟 กราฟ TradingView (แก้ปัญหาแบนแต๊ดแต๋ โดยการล็อก Height ทั้ง 3 จุด)
tradingview_html = f"""
<div class="tradingview-widget-container" style="height:600px;width:100%">
  <div class="tradingview-widget-container__widget" style="height:100%;width:100%"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
  {{
  "autosize": false,
  "width": "100%",
  "height": "600",
  "allow_symbol_change": false, 
  "calendar": false,
  "details": false,
  "hide_side_toolbar": false,
  "hide_top_toolbar": false,
  "hide_legend": false,
  "hide_volume": false,
  "interval": "D",
  "locale": "th_TH",
  "save_image": true,
  "style": "1",
  "symbol": "{st.session_state.ticker}",
  "theme": "dark",
  "timezone": "Asia/Bangkok",
  "backgroundColor": "#131722",
  "gridColor": "rgba(242, 242, 242, 0.06)",
  "withdateranges": true,
  "studies": [
    "STD;Bollinger_Bands_B",
    "STD;DEMA",
    "STD;MA%1Cross",
    "STD;RSI"
  ]
}}
  </script>
</div>
"""

# ใช้ความสูง 650 เผื่อพื้นที่ขอบ
components.html(tradingview_html, height=650)

st.divider()

# 2. 🤖 พื้นที่สำหรับ AI Analysis
col_ai1, col_ai2 = st.columns([1, 4])
with col_ai1:
    st.subheader("🤖 AI Analyst")
    if st.button("✨ กดวิเคราะห์", type="primary", use_container_width=True):
        st.session_state.analyze_now = True

with col_ai2:
    if st.session_state.get('analyze_now', False):
        with st.spinner("กำลังให้ AI เจาะลึกข้อมูลกราฟ..."):
            analysis_result = analyze_with_ai(st.session_state.ticker)
            st.success("✅ วิเคราะห์เสร็จสิ้น")
            st.markdown(analysis_result)
        st.session_state.analyze_now = False # รีเซ็ตสถานะ