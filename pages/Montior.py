import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="Painel de Ações - S&P 500 e IBOVESPA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título do painel
st.title("📊 Painel de Ações em Tempo Real")
st.markdown("Monitoramento de ações do **S&P 500** e **IBOVESPA**")

# Seleção de ações
sp500_tickers = ['^GSPC', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'PG']
ibovespa_tickers = ['^BVSP', 'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA', 'WEGE3.SA', 'ABEV3.SA', 'RENT3.SA', 'SUZB3.SA', 'B3SA3.SA']

# Inicializar session state
if "previous_sp500_data" not in st.session_state:
    st.session_state.previous_sp500_data = {}
if "previous_ibovespa_data" not in st.session_state:
    st.session_state.previous_ibovespa_data = {}
if "update_interval" not in st.session_state:
    st.session_state.update_interval = 60
if "selected_sp500" not in st.session_state:
    st.session_state.selected_sp500 = "^GSPC"
if "selected_ibovespa" not in st.session_state:
    st.session_state.selected_ibovespa = "^BVSP"

# Função para obter dados históricos (com cache)
@st.cache_data(ttl=60)
def get_historical_data(ticker, period="1d", interval="1m"):
    data = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        progress=False
    )
    
    if isinstance(data.columns, pd.MultiIndex):
        ticker_data = pd.DataFrame()
        ticker_data['Datetime'] = data.index
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if (ticker, col) in data.columns:
                ticker_data[col] = data[(ticker, col)]
            else:
                ticker_data[col] = data[col].values
        return ticker_data
    return data

# Função para obter dados das ações para métricas (corrigida para evitar NaN)
def get_stock_data(tickers):
    # Buscar dados diários para evitar NaN fora do horário de negociação
    data = yf.download(
        tickers=tickers,
        period="1d",
        interval="1m",
        group_by='ticker',
        progress=False
    )
    
    df_list = []
    for ticker in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex) and ticker in data.columns.levels[0]:
                ticker_data = data[ticker]
            elif not isinstance(data.columns, pd.MultiIndex) and len(tickers) == 1:
                ticker_data = data
            else:
                continue

            # Remover linhas com NaN para garantir valores válidos
            ticker_data = ticker_data.dropna()

            if ticker_data.empty:
                continue

            # Primeiro "Open" válido do dia
            first_open = ticker_data['Open'].iloc[0]
            # Último "Close" válido do dia
            last_close = ticker_data['Close'].iloc[-1]
            # Volume total do dia
            total_volume = ticker_data['Volume'].sum()

            # Calcular variação diária
            if first_open != 0 and not pd.isna(first_open) and not pd.isna(last_close):
                variation = (last_close - first_open) / first_open * 100
            else:
                variation = 0

            df_list.append({
                'Ticker': ticker,
                'Preço': round(last_close, 2),
                'Variação (%)': round(variation, 2),
                'Volume': int(total_volume)
            })
        except Exception as e:
            st.error(f"Erro ao processar {ticker}: {e}")
            continue
    
    return pd.DataFrame(df_list)

# Sidebar para controles
with st.sidebar:
    st.subheader("Configurações")
    update_interval = st.slider(
        "Intervalo de atualização (segundos)", 
        min_value=10, 
        max_value=300, 
        value=st.session_state.update_interval,
        key="run_every"
    )
    st.session_state.update_interval = update_interval
    
    chart_period = st.selectbox(
        "Período do gráfico",
        options=["1d", "5d", "1mo", "3mo", "6mo", "1y"],
        index=0,
        key="chart_period"
    )
    
    chart_interval = st.selectbox(
        "Intervalo do gráfico",
        options=["1m", "5m", "15m", "30m", "60m", "1d"],
        index=0,
        key="chart_interval"
    )
    
    if st.button("Atualizar agora"):
        st.rerun()

# Função para criar o gráfico
def create_stock_chart(data, title):
    if data.empty:
        st.error(f"Sem dados disponíveis para {title}")
        return px.line(title=f"{title} - Sem dados disponíveis")
    
    chart_data = data.copy()
    
    if 'Close' in chart_data.columns:
        if isinstance(chart_data['Close'].iloc[0], (list, np.ndarray)):
            chart_data['Close'] = chart_data['Close'].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
    else:
        st.error(f"Coluna 'Close' não encontrada nos dados para {title}")
        return px.line(title=f"{title} - Dados incompletos")
    
    if isinstance(chart_data.index, pd.DatetimeIndex):
        chart_data = chart_data.reset_index()
        chart_data = chart_data.rename(columns={'index': 'Datetime'})
    
    fig = px.line(
        chart_data,
        x='Datetime',
        y='Close',
        title=title,
        labels={'Close': 'Preço', 'Datetime': 'Data/Hora'},
    )
    
    fig.update_xaxes(
        type='date',
        tickformat='%H:%M\n%d/%m/%Y',
        dtick=3600000,
        tickmode='auto',
        nticks=10,
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    
    fig.update_traces(line=dict(width=2))
    return fig

# Layout com duas colunas
col1, col2 = st.columns(2)

# S&P 500
with col1:
    st.subheader("S&P 500")
    selected_sp500 = st.selectbox(
        "Selecione uma ação do S&P 500:",
        options=sp500_tickers,
        index=sp500_tickers.index(st.session_state.selected_sp500),
        key="sp500_select"
    )
    st.session_state.selected_sp500 = selected_sp500
    sp500_chart_container = st.empty()
    sp500_time = st.empty()
    sp500_metrics_header = st.empty()
    sp500_metrics_container = st.empty()

# IBOVESPA
with col2:
    st.subheader("IBOVESPA")
    selected_ibovespa = st.selectbox(
        "Selecione uma ação do IBOVESPA:",
        options=ibovespa_tickers,
        index=ibovespa_tickers.index(st.session_state.selected_ibovespa),
        key="ibovespa_select"
    )
    st.session_state.selected_ibovespa = selected_ibovespa
    ibovespa_chart_container = st.empty()
    ibovespa_time = st.empty()
    ibovespa_metrics_header = st.empty()
    ibovespa_metrics_container = st.empty()

# Função para atualizar os dados e a visualização
@st.fragment(run_every=st.session_state.update_interval)
def update_stock_data():
    try:
        # Obter dados para métricas
        sp500_data = get_stock_data(sp500_tickers)
        ibovespa_data = get_stock_data(ibovespa_tickers)
        
        # Obter dados históricos para os gráficos
        sp500_hist_data = get_historical_data(st.session_state.selected_sp500, period=st.session_state.chart_period, interval=st.session_state.chart_interval)
        ibovespa_hist_data = get_historical_data(st.session_state.selected_ibovespa, period=st.session_state.chart_period, interval=st.session_state.chart_interval)
        
        # Criar e exibir os gráficos
        with col1:
            sp500_fig = create_stock_chart(sp500_hist_data, f"{st.session_state.selected_sp500} - Preço")
            sp500_chart_container.plotly_chart(sp500_fig, use_container_width=True, theme="streamlit")
            sp500_time.caption(f"Última atualização: {datetime.now().strftime('%H:%M:%S')}")
        
        with col2:
            ibovespa_fig = create_stock_chart(ibovespa_hist_data, f"{st.session_state.selected_ibovespa} - Preço")
            ibovespa_chart_container.plotly_chart(ibovespa_fig, use_container_width=True, theme="streamlit")
            ibovespa_time.caption(f"Última atualização: {datetime.now().strftime('%H:%M:%S')}")
        
        # Exibir métricas como tabelas
        with col1:
            sp500_metrics_header.subheader("Métricas principais - S&P 500")
            if not sp500_data.empty:
                sp500_metrics_container.table(sp500_data)
            else:
                sp500_metrics_container.write("Nenhum dado disponível para S&P 500.")
        
        with col2:
            ibovespa_metrics_header.subheader("Métricas principais - IBOVESPA")
            if not ibovespa_data.empty:
                ibovespa_metrics_container.table(ibovespa_data)
            else:
                ibovespa_metrics_container.write("Nenhum dado disponível para IBOVESPA.")
        
        # Armazenar dados atuais para a próxima comparação
        st.session_state.previous_sp500_data = {row['Ticker']: row['Preço'] for _, row in sp500_data.iterrows()}
        st.session_state.previous_ibovespa_data = {row['Ticker']: row['Preço'] for _, row in ibovespa_data.iterrows()}
    
    except Exception as e:
        st.error(f"Erro ao atualizar dados: {e}")

# Iniciar a atualização
update_stock_data()