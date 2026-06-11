import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import urllib.request
import io
import json
import concurrent.futures
import mplfinance as mpf
import matplotlib.pyplot as plt
from google import genai 
from google.genai import types
from pydantic import BaseModel

# --- ⚙️ หน้าตั้งค่าระบบเว็บ ---
st.set_page_config(page_title="Pro Stock Scanner - Web AI", layout="wide", page_icon="👑")

# ดึง API Key จากระบบ Secrets ของ Streamlit (หรือกรอกเองกรณีทดสอบในเครื่อง)
API_KEY = st.secrets["GEMINI_API_KEY"]

# --- 📊 Pydantic Schema สำหรับโครงสร้าง JSON ของ AI ---
class StockAnalysisResult(BaseModel):
    sentiment: str
    risk_level: str
    support_zone: str
    resistance_zone: str
    action_suggestion: str
    detailed_reason: str

# --- 🔄 ระบบหน่วยความจำเว็บ (Session State) ---
if 'active_ticker' not in st.session_state:
    st.session_state.active_ticker = "AAPL"
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = []
if 'ai_analysis' not in st.session_state:
    st.session_state.ai_analysis = None
if 'timeframe_period' not in st.session_state:
    st.session_state.timeframe_period = "6mo"
if 'timeframe_interval' not in st.session_state:
    st.session_state.timeframe_interval = "1d"

# --- 🛠️ ฟังก์ชันช่วยเหลือ (Helper Functions) ---
def fetch_data_with_header(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response: 
        return response.read()

def clean_df_columns(df, ticker):
    if df is None or df.empty: return pd.DataFrame()
    df_cleaned = df.copy()
    if isinstance(df_cleaned.columns, pd.MultiIndex):
        if ticker in df_cleaned.columns.get_level_values(1): 
            df_cleaned = df_cleaned.xs(ticker, axis=1, level=1)
        elif ticker in df_cleaned.columns.get_level_values(0): 
            df_cleaned = df_cleaned.xs(ticker, axis=1, level=0)
        else: 
            df_cleaned.columns = df_cleaned.columns.get_level_values(0)
    df_cleaned.index = pd.to_datetime(df_cleaned.index)
    cleaned_dict = {col: df_cleaned[col].squeeze() for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in df_cleaned.columns}
    return pd.DataFrame(cleaned_dict, index=df_cleaned.index)

# --- 🚀 ฟังก์ชันหลักในการสแกนตลาด ---
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
            df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
            
            c, pc = float(close.iloc[-1]), float(close.iloc[-2])
            u, pu = float(df['BB_Upper'].iloc[-1]), float(df['BB_Upper'].iloc[-2])
            l = float(df['BB_Lower'].iloc[-1])
            v, vm = float(df['Volume'].iloc[-1]), float(df['Vol_MA'].iloc[-1])
            rsi = float(df['RSI'].iloc[-1])
            mh, pmh = float(df['MACD_Hist'].iloc[-1]), float(df['MACD_Hist'].iloc[-2])
            
            if pc <= pu and c > u and v > (vm * 1.5): return [ticker, round(c, 2), "🟢 BUY (BB Breakout)"]
            elif pmh <= 0 and mh > 0: return [ticker, round(c, 2), "⚔️ MACD Golden Cross"]
            elif rsi < 30: return [ticker, round(c, 2), "📉 RSI Oversold"]
            elif rsi > 70: return [ticker, round(c, 2), "🔥 RSI Overbought"]
            elif c < l: return [ticker, round(c, 2), "🔴 BB Breakdown"]
            else: return [ticker, round(c, 2), "⚪ No Signal"]
    except: pass
    return None

# ==========================================
# 📌 SIDEBAR: เมนูควบคุมการทำงานทั้งหมด
# ==========================================
with st.sidebar:
    st.title("👑 Pro Stock Scanner")
    st.subheader("ระบบควบคุมผ่านเว็บ")
    
    # ส่วนค้นหาหุ้นแมนนวล
    st.write("🔍 **ค้นหาหุ้นรายตัว**")
    search_ticker = st.text_input("กรอกชื่อหุ้น (Ticker):", value=st.session_state.active_ticker).upper()
    if st.button("ดึงข้อมูลกราฟ", use_container_width=True):
        st.session_state.active_ticker = search_ticker
        st.session_state.ai_analysis = None
        st.rerun()
        
    st.divider()
    
    # ส่วนเลือกตลาดเพื่อสแกน
    st.write("🏛️ **ระบบสแกนกลุ่มตลาด**")
    selected_market = st.selectbox("เลือกตลาด", ["NASDAQ 100", "S&P 500", "Dow Jones 30"])
    
    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        if st.button("🔥 Top Gainers", use_container_width=True):
            st.session_state.scan_results = []
            sample_tickers = ['NVDA', 'TSLA', 'AMD', 'AAPL', 'AMZN', 'META', 'GOOGL', 'PLTR', 'NFLX', 'MSFT']
            with st.spinner("กำลังดึงข้อมูล..."):
                for t in sample_tickers:
                    hist = yf.Ticker(t).history(period="2d")
                    if not hist.empty:
                        c_p, p_p = hist['Close'].iloc[-1], hist['Close'].iloc[-2]
                        chg = ((c_p - p_p) / p_p) * 100
                        if chg > 0:
                            st.session_state.scan_results.append([t, round(c_p, 2), f"🔥 +{chg:.2f}%"])
        
    with col_sc2:
        if st.button("🩸 Top Losers", use_container_width=True):
            st.session_state.scan_results = []
            sample_tickers = ['NVDA', 'TSLA', 'AMD', 'AAPL', 'AMZN', 'META', 'GOOGL', 'PLTR', 'NFLX', 'MSFT']
            with st.spinner("กำลังดึงข้อมูล..."):
                for t in sample_tickers:
                    hist = yf.Ticker(t).history(period="2d")
                    if not hist.empty:
                        c_p, p_p = hist['Close'].iloc[-1], hist['Close'].iloc[-2]
                        chg = ((c_p - p_p) / p_p) * 100
                        if chg < 0:
                            st.session_state.scan_results.append([t, round(c_p, 2), f"🩸 {chg:.2f}%"])

    if st.button("🚀 เริ่มสแกนตลาด", type="primary", use_container_width=True):
        st.session_state.scan_results = []
        tickers_list = []
        try:
            if selected_market == "S&P 500":
                csv_bytes = fetch_data_with_header('https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv')
                tickers_list = pd.read_csv(io.BytesIO(csv_bytes))['Symbol'].tolist()
            elif "NASDAQ" in selected_market:
                html_bytes = fetch_data_with_header('https://en.wikipedia.org/wiki/Nasdaq-100')
                tables = pd.read_html(io.StringIO(html_bytes.decode('utf-8')))
                for df in tables:
                    if 'Ticker' in df.columns or 'Symbol' in df.columns:
                        col = 'Ticker' if 'Ticker' in df.columns else 'Symbol'
                        tickers_list = df[col].tolist()
                        break
            else:
                html_bytes = fetch_data_with_header('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')
                tables = pd.read_html(io.StringIO(html_bytes.decode('utf-8')))
                for df in tables:
                    if 'Symbol' in df.columns:
                        tickers_list = df['Symbol'].tolist()
                        break
        except:
            tickers_list = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA']
            
        tickers_list = [str(t).replace('.', '-') for t in tickers_list]
        
        # คัดกรองเฉพาะตัวที่ติดสัญญาณด้วย Multi-threading (ความเร็วสูงขึ้นบน Cloud)
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = {executor.submit(scan_single_stock, t): t for t in tickers_list}
            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                progress_bar.progress((idx + 1) / len(tickers_list))
                status_text.text(f"สแกนสำเร็จ: {idx+1}/{len(tickers_list)}")
                res = future.result()
                if res:
                    results.append(res)
        
        st.session_state.scan_results = results
        status_text.text(f"✅ สแกนเสร็จสิ้น! พบข้อมูล {len(results)} รายการ")
        progress_bar.empty()

    st.divider()
    
    # กล่องตัวกรองสัญญาณผลลัพธ์
    filter_choice = st.selectbox("🎯 ตัวกรองผลลัพธ์:", ["ทั้งหมด", "🟢 BUY (BB Breakout)", "⚔️ MACD Golden Cross", "📉 RSI Oversold", "🔥 RSI Overbought", "🔴 BB Breakdown", "⚪ No Signal"])

# ==========================================
# 📌 MAIN WORKSPACE: พื้นที่แสดงผลกราฟและ AI
# ==========================================

# ส่วนแสดงตารางผลสแกนหุ้นที่เจอด้านบน
if st.session_state.scan_results:
    st.subheader("📋 ผลลัพธ์การสแกน / หุ้นที่พบสัญญาณ")
    # นำผลลัพธ์มาแปลงเป็น DataFrame เพื่อแสดงตารางแบบ Interactive
    df_scan = pd.DataFrame(st.session_state.scan_results, columns=["Ticker", "Price", "Signal"])
    if filter_choice != "ทั้งหมด":
        df_scan = df_scan[df_scan["Signal"].str.contains(filter_choice, na=False, regex=False)]
    
    st.dataframe(df_scan, use_container_width=True, hide_index=True)
    
    # ทำปุ่มให้คลิกเลือกหุ้นจากผลลัพธ์สแกนได้อย่างรวดเร็ว
    ticker_options = df_scan["Ticker"].tolist()
    if ticker_options:
        selected_from_table = st.selectbox("🖱️ เลือกหุ้นติดสัญญาณเพื่อเปิดกราฟและส่งให้ AI:", ticker_options)
        if st.button("เปิดหุ้นตัวนี้", use_container_width=True):
            st.session_state.active_ticker = selected_from_table
            st.session_state.ai_analysis = None
            st.rerun()

st.divider()

# แถบวิเคราะห์ข้อมูลหุ้นที่เลือกทำงานอยู่
ticker = st.session_state.active_ticker
st.header(f"📈 หุ้นปัจจุบัน: {ticker}")

# แสดงข้อมูลพื้นฐาน (Sector / Market Cap)
try:
    info = yf.Ticker(ticker).info
    sector = info.get('sector', 'N/A')
    mcap = f"${info.get('marketCap', 0)/1e9:.2f}B" if info.get('marketCap') else "N/A"
    st.caption(f"🏢 อุตสาหกรรม: {sector} | 💰 Market Cap: {mcap}")
except:
    st.caption("🏢 อุตสาหกรรม: N/A | 💰 Market Cap: N/A")

# ส่วนเลือก Timeframe
col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)
with col_t1:
    if st.button("📅 1D (Intraday)", use_container_width=True): 
        st.session_state.timeframe_period, st.session_state.timeframe_interval = "1d", "1m"
        st.rerun()
with col_t2:
    if st.button("📅 1W", use_container_width=True): 
        st.session_state.timeframe_period, st.session_state.timeframe_interval = "7d", "15m"
        st.rerun()
with col_t3:
    if st.button("📅 1M", use_container_width=True): 
        st.session_state.timeframe_period, st.session_state.timeframe_interval = "1mo", "1d"
        st.rerun()
with col_t4:
    if st.button("📅 6M (Default)", use_container_width=True): 
        st.session_state.timeframe_period, st.session_state.timeframe_interval = "6mo", "1d"
        st.rerun()
with col_t5:
    if st.button("📅 1Y", use_container_width=True): 
        st.session_state.timeframe_period, st.session_state.timeframe_interval = "1y", "1d"
        st.rerun()

# --- 📊 ดึงข้อมูลเพื่อวาดกราฟเทคนิคอลด้วย Mplfinance ---
try:
    raw_df = yf.download(ticker, period=st.session_state.timeframe_period, interval=st.session_state.timeframe_interval, auto_adjust=False, progress=False)
    df = clean_df_columns(raw_df, ticker).dropna()
    
    if not df.empty:
        close = df['Close']
        df['EMA_50'] = close.ewm(span=50, adjust=False).mean()
        df['MA20'] = close.rolling(window=20).mean()
        df['BB_Upper'] = df['MA20'] + (close.rolling(window=20).std() * 2)
        df['BB_Lower'] = df['MA20'] - (close.rolling(window=20).std() * 2)
        df['MACD'] = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']
        df['MACD_Color'] = np.where(df['MACD_Hist'] > 0, '#22C55E', '#EF4444')
        df['Vol_MA'] = df['Volume'].rolling(window=20).mean()
        delta = close.diff()
        rs = delta.where(delta>0,0).ewm(alpha=1/14).mean() / -delta.where(delta<0,0).ewm(alpha=1/14).mean()
        df['RSI'] = 100 - (100 / (1 + rs))

        # เก็บตัวแปรส่งต่อให้ AI วิเคราะห์
        current_price = float(df['Close'].iloc[-1])
        current_rsi = float(df['RSI'].iloc[-1])
        current_macd = "บวก (ตัดขึ้น)" if float(df['MACD_Hist'].iloc[-1]) > 0 else "ลบ (ตัดลง)"
        current_trend = "ขาขึ้น" if current_price > float(df['EMA_50'].iloc[-1]) else "ขาลง"
        current_vol = "ใช่ (ผิดปกติ)" if float(df['Volume'].iloc[-1]) > (float(df['Vol_MA'].iloc[-1]) * 1.5) else "ปกติ"

        # วาดรูปผ่าน Mplfinance ลง Matplotlib Fig
        plot_df = df.tail(120) if len(df) > 120 else df
        mc = mpf.make_marketcolors(up='#22C55E', down='#EF4444', edge='inherit', wick='inherit', volume='inherit')
        s = mpf.make_mpf_style(marketcolors=mc, base_mpf_style='nightclouds', facecolor='#0F172A', edgecolor='#1E293B', figcolor='#0F172A', gridcolor='#1E293B', gridstyle='--')
        
        ap = [
            mpf.make_addplot(plot_df['EMA_50'], color='#F59E0B', width=1.2, panel=0),
            mpf.make_addplot(plot_df['BB_Upper'], color='#38BDF8', width=0.8, alpha=0.5, panel=0),
            mpf.make_addplot(plot_df['BB_Lower'], color='#38BDF8', width=0.8, alpha=0.5, panel=0),
            mpf.make_addplot(plot_df['MACD_Hist'], type='bar', color=plot_df['MACD_Color'].tolist(), panel=1, ylabel='MACD'),
            mpf.make_addplot(plot_df['MACD'], color='#38BDF8', width=1.2, panel=1),
            mpf.make_addplot(plot_df['Signal'], color='#F59E0B', width=1.2, panel=1),
            mpf.make_addplot(plot_df['RSI'], color='#A855F7', width=1.5, panel=2, ylabel='RSI'),
            mpf.make_addplot([70]*len(plot_df), color='#EF4444', linestyle=':', panel=2),
            mpf.make_addplot([30]*len(plot_df), color='#22C55E', linestyle=':', panel=2)
        ]
        
        fig, axlist = mpf.plot(plot_df, type='candle', volume=True, addplot=ap, style=s, returnfig=True, panel_ratios=(4, 1.5, 1.5), figsize=(12, 7), tight_layout=True)
        axlist[0].set_ylabel("ราคา ($)")
        
        # นำ Matplotlib Render ขึ้นหน้าเว็บ Streamlit นกกระจอกไม่ทันกินน้ำ!
        st.pyplot(fig)
        
    else:
        st.error("❌ ข้อมูลของหุ้นตัวนี้ว่างเปล่า ไม่สามารถสร้างกราฟได้")
except Exception as e:
    st.error(f"❌ ไม่สามารถโหลดกราฟได้เนื่องจาก: {str(e)}")

st.divider()

# --- 🤖 ผู้ช่วย AI วิเคราะห์เชิงลึก (Structured Outputs JSON) ---
st.subheader("🤖 Pro AI Analyst")

if st.button("✨ เริ่มต้นวิเคราะห์ทางเทคนิคด้วย AI", type="primary"):
    if 'current_price' in locals():
        with st.spinner("AI กำลังแกะสัญญาณกราฟและประมวลผลคำตอบแบบโครงสร้าง JSON..."):
            prompt = f"""
            วิเคราะห์เทคนิคอลหุ้น {ticker} ราคาปัจจุบัน ${current_price:.2f}
            - RSI: {current_rsi:.2f}
            - MACD: {current_macd}
            - ภาพรวมเทียบ EMA50: {current_trend}
            - Volume ผิดปกติไหม: {current_vol}
            
            กรุณาวิッセージและตอบตามโครงสร้างที่ระบบกำหนดไว้อย่างเคร่งครัด
            """
            try:
                client = genai.Client(api_key=API_KEY)
                response = client.models.generate_content(
                    model='gemini-3.1-flash-lite',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=StockAnalysisResult,
                        temperature=0.2,
                        system_instruction="คุณคือผู้ช่วยวิเคราะห์หุ้นมืออาชีพ ต้องตอบกลับเป็นภาษาไทยที่สุภาพและเข้าใจง่ายเท่านั้น ห้ามใช้ภาษาอังกฤษปนในจุดที่ไม่จำเป็น"
                    )
                )
                st.session_state.ai_analysis = json.loads(response.text)
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ AI: {str(e)}")
    else:
        st.warning("⚠️ กรุณาค้นหาหุ้นที่ถูกต้องก่อนสั่งวิเคราะห์ด้วย AI")

# แสดงผลบทวิเคราะห์ของ AI เมื่อรันสำเร็จ
if st.session_state.ai_analysis:
    res = st.session_state.ai_analysis
    st.info("💡 บทวิเคราะห์ทางเทคนิครายวันส่งตรงจาก AI")
    
    col_ai1, col_ai2, col_ai3 = st.columns(3)
    with col_ai1:
        st.metric(label="📊 มุมมองเทรนด์", value=res['sentiment'])
    with col_ai2:
        st.metric(label="⚠️ ระดับความเสี่ยง", value=res['risk_level'])
    with col_ai3:
        st.metric(label="🎯 คำแนะนำเบื้องต้น", value=res['action_suggestion'])
        
    st.write(f"🛡️ **แนวรับสำคัญ:** `{res['support_zone']}` | 🚀 **แนวต้านสำคัญ:** `{res['resistance_zone']}`")
    st.markdown(f"📝 **บทวิเคราะห์เชิงลึก:**\n\n{res['detailed_reason']}")