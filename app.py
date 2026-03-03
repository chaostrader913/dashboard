import streamlit as st

st.set_page_config(page_title="技术分析平台", page_icon="📈", layout="wide")

# 主页内容
st.title("📈 技术分析平台")

# 欢迎信息
st.markdown("""
## 欢迎使用技术分析平台

本平台提供以下功能：

### 📊 图表网格
- 同时监控多只股票的技术形态
- 自定义网格布局和显示指标
- 快速对比不同股票走势

### 🔍 信号扫描仪
- 全市场扫描技术信号
- 自定义扫描条件
- 实时结果展示和导出

### 📈 指标演示
- 单个指标的详细说明
- 参数可调交互式图表
- 实时数据演示
""")

# 快速启动
st.subheader("🚀 快速启动")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("### 📊 图表网格")
    st.markdown("同时监控多只股票")
    if st.button("打开图表网格", key="btn_grid"):
        st.switch_page("pages/02_Chart_Grid.py")

with col2:
    st.info("### 🔍 信号扫描仪")
    st.markdown("发现交易机会")
    if st.button("打开信号扫描仪", key="btn_scanner"):
        st.switch_page("pages/03_Signal_Scanner.py")

with col3:
    st.info("### 📈 指标演示")
    st.markdown("学习技术指标")
    if st.button("查看指标", key="btn_indicators"):
        st.switch_page("pages/01_MACD.py")

# 最近信号预览
st.subheader("🔥 热门信号")
# 这里可以添加缓存的热门信号
