import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 设置 Streamlit 页面配置
st.set_page_config(page_title="缠论自动化分析系统", layout="wide")

class ChanAnalyzer:
    def __init__(self, symbol, period='2y', interval='1d'):
        self.symbol = symbol
        self.period = period
        self.interval = interval
        self.df = None
        self.standardized_df = None
        self.bi_list = []
        self.zhongshus = []
        self.waves = [] 
        self.signals = [] 

    def fetch_data(self):
        """获取真实市场数据并计算MACD"""
        ticker = yf.Ticker(self.symbol)
        self.df = ticker.history(period=self.period, interval=self.interval)
        if self.df.empty:
            return None
        self.df.reset_index(inplace=True)
        
        # 计算 MACD
        exp1 = self.df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD_DIF'] = exp1 - exp2
        self.df['MACD_DEA'] = self.df['MACD_DIF'].ewm(span=9, adjust=False).mean()
        self.df['MACD_HIST'] = (self.df['MACD_DIF'] - self.df['MACD_DEA']) * 2
        return self.df

    def process_inclusion(self):
        """K线包含处理"""
        if self.df is None or self.df.empty: return
        data = self.df.copy()
        new_data = []
        curr = data.iloc[0].to_dict()
        direction = 1 
        for i in range(1, len(data)):
            nxt = data.iloc[i].to_dict()
            is_inclusion = (curr['High'] >= nxt['High'] and curr['Low'] <= nxt['Low']) or \
                           (nxt['High'] >= curr['High'] and nxt['Low'] <= curr['Low'])
            if is_inclusion:
                if direction == 1: 
                    curr['High'] = max(curr['High'], nxt['High'])
                    curr['Low'] = max(curr['Low'], nxt['Low'])
                else: 
                    curr['High'] = min(curr['High'], nxt['High'])
                    curr['Low'] = min(curr['Low'], nxt['Low'])
            else:
                direction = 1 if nxt['High'] > curr['High'] else -1
                new_data.append(curr)
                curr = nxt
        new_data.append(curr)
        self.standardized_df = pd.DataFrame(new_data)
        return self.standardized_df

    def find_bi(self):
        """识别笔 (Bi)"""
        df = self.standardized_df
        if df is None: return []
        points = []
        for i in range(1, len(df) - 1):
            if df.iloc[i]['High'] > df.iloc[i-1]['High'] and df.iloc[i]['High'] > df.iloc[i+1]['High']:
                points.append({'index': i, 'type': 'top', 'price': df.iloc[i]['High'], 
                               'date': df.iloc[i]['Date'], 'macd': df.iloc[i]['MACD_DIF']})
            elif df.iloc[i]['Low'] < df.iloc[i-1]['Low'] and df.iloc[i]['Low'] < df.iloc[i+1]['Low']:
                points.append({'index': i, 'type': 'bottom', 'price': df.iloc[i]['Low'], 
                               'date': df.iloc[i]['Date'], 'macd': df.iloc[i]['MACD_DIF']})
        
        valid_bi = []
        for p in points:
            if not valid_bi:
                valid_bi.append(p)
                continue
            last = valid_bi[-1]
            if p['type'] != last['type'] and (p['index'] - last['index']) >= 4:
                valid_bi.append(p)
            elif p['type'] == last['type']:
                if (p['type'] == 'top' and p['price'] > last['price']) or \
                   (p['type'] == 'bottom' and p['price'] < last['price']):
                    valid_bi[-1] = p
        self.bi_list = valid_bi
        return valid_bi

    def find_zhongshu(self):
        """识别中枢"""
        if len(self.bi_list) < 4: return []
        zs_list = []
        for i in range(len(self.bi_list) - 3):
            b1_s, b1_e = self.bi_list[i]['price'], self.bi_list[i+1]['price']
            b2_s, b2_e = self.bi_list[i+1]['price'], self.bi_list[i+2]['price']
            b3_s, b3_e = self.bi_list[i+2]['price'], self.bi_list[i+3]['price']
            h = min(max(b1_s, b1_e), max(b2_s, b2_e), max(b3_s, b3_e))
            l = max(min(b1_s, b1_e), min(b2_s, b2_e), min(b3_s, b3_e))
            if h > l:
                zs_list.append({'start_date': self.bi_list[i]['date'], 'end_date': self.bi_list[i+3]['date'], 'high': h, 'low': l})
        self.zhongshus = zs_list
        return zs_list

    def identify_signals(self):
        """识别信号"""
        if len(self.bi_list) < 5: return []
        signals = []
        for i in range(4, len(self.bi_list)):
            curr = self.bi_list[i]
            prev_same = self.bi_list[i-2]
            # 一类
            if curr['type'] == 'bottom' and prev_same['type'] == 'bottom':
                if curr['price'] < prev_same['price'] and abs(curr['macd']) < abs(prev_same['macd']):
                    signals.append({'date': curr['date'], 'price': curr['price'], 'type': 'B1', 'text': '一买'})
            elif curr['type'] == 'top' and prev_same['type'] == 'top':
                if curr['price'] > prev_same['price'] and abs(curr['macd']) < abs(prev_same['macd']):
                    signals.append({'date': curr['date'], 'price': curr['price'], 'type': 'S1', 'text': '一卖'})
            # 三类 (简化)
            if self.zhongshus:
                last_zs = self.zhongshus[-1]
                if curr['type'] == 'bottom' and curr['price'] > last_zs['high'] and self.bi_list[i-1]['price'] > last_zs['high']:
                    signals.append({'date': curr['date'], 'price': curr['price'], 'type': 'B3', 'text': '三买'})
                elif curr['type'] == 'top' and curr['price'] < last_zs['low'] and self.bi_list[i-1]['price'] < last_zs['low']:
                    signals.append({'date': curr['date'], 'price': curr['price'], 'type': 'S3', 'text': '三卖'})
        self.signals = signals
        return signals

    def get_figure(self):
        """返回 Plotly Figure 对象供 Streamlit 使用"""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                           row_heights=[0.7, 0.3], subplot_titles=('价格走势 & 缠论标注', 'MACD 动能'))
        
        # K线
        fig.add_trace(go.Candlestick(x=self.df['Date'], open=self.df['Open'], high=self.df['High'],
                                     low=self.df['Low'], close=self.df['Close'], name='K线', opacity=0.4), row=1, col=1)
        # 笔
        bi_x = [p['date'] for p in self.bi_list]
        bi_y = [p['price'] for p in self.bi_list]
        fig.add_trace(go.Scatter(x=bi_x, y=bi_y, mode='lines+markers', name='笔', 
                                 line=dict(color='yellow', width=1.5)), row=1, col=1)
        # 中枢
        for zs in self.zhongshus:
            fig.add_shape(type="rect", x0=zs['start_date'], y0=zs['low'], x1=zs['end_date'], y1=zs['high'],
                          line=dict(color="cyan", width=1), fillcolor="cyan", opacity=0.1, row=1, col=1)
        # 信号
        for sig in self.signals:
            color = "red" if sig['type'].startswith('S') else "lime"
            fig.add_trace(go.Scatter(x=[sig['date']], y=[sig['price']], mode='markers+text', 
                                     name=sig['text'], text=[sig['text']], textposition="top center",
                                     marker=dict(color=color, size=12, symbol="star")), row=1, col=1)
        # MACD
        fig.add_trace(go.Bar(x=self.df['Date'], y=self.df['MACD_HIST'], name='柱状图'), row=2, col=1)
        
        fig.update_layout(height=800, template='plotly_dark', xaxis_rangeslider_visible=False)
        return fig

# --- Streamlit UI 部分 ---
st.sidebar.title("🔍 缠论参数设置")
ticker_input = st.sidebar.text_input("请输入股票代码 (如: 3032.HK, AAPL, 0700.HK)", "3032.HK")
time_period = st.sidebar.selectbox("时间范围", ["1y", "2y", "5y", "max"], index=1)
chart_interval = st.sidebar.selectbox("K线周期", ["1d", "1wk", "1mo"], index=0)

st.title(f"📊 {ticker_input} 自动化缠论分析")

if st.sidebar.button("开始分析"):
    with st.spinner('正在获取数据并计算结构...'):
        analyzer = ChanAnalyzer(ticker_input, period=time_period, interval=chart_interval)
        data = analyzer.fetch_data()
        
        if data is not None:
            analyzer.process_inclusion()
            analyzer.find_bi()
            analyzer.find_zhongshu()
            analyzer.identify_signals()
            
            # 显示指标卡片
            c1, c2, c3 = st.columns(3)
            c1.metric("当前价格", f"{data.iloc[-1]['Close']:.2f}")
            c2.metric("识别笔数", len(analyzer.bi_list))
            c3.metric("识别中枢", len(analyzer.zhongshus))
            
            # 绘制图表
            fig = analyzer.get_figure()
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示信号列表
            if analyzer.signals:
                st.subheader("🔔 最新买卖信号")
                sig_df = pd.DataFrame(analyzer.signals)[['date', 'type', 'price', 'text']]
                st.table(sig_df.tail(5))
        else:
            st.error("未找到数据，请检查代码输入是否正确（港股需加 .HK，美股直接输入代码）。")
else:
    st.info("在左侧输入代码并点击『开始分析』按钮。")
