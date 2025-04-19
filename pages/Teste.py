import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from portfolio_utils_v2 import (
    portfolio_rebalance, calculate_metrics, calculate_technical_indicators,
    create_target, XGBClassifier, RandomForestClassifier, LGBMClassifier,
    MLPClassifier
)
import os
import pickle
from datetime import datetime

st.set_page_config(layout="wide")

st.title("Backtest de Estratégias de Rebalanceamento - Ibovespa")
st.write("Análise detalhada da evolução de capital e desempenho de estratégias de portfólio no Ibovespa.")

os.makedirs("data", exist_ok=True)
os.makedirs("results_ibovespa", exist_ok=True)

def is_new_week(last_date, current_date):
    return current_date.isocalendar()[1] > last_date.isocalendar()[1]

@st.cache_data
def load_data():
    index_data_path = "data/index_data_bvsp.pkl"
    assets_data_path = "data/assets_data_bvsp.pkl"
    current_date = datetime(2025, 4, 4)

    def download_and_save_data():
        index_data_daily = yf.download('^BVSP', start='2000-01-01', end=current_date.strftime('%Y-%m-%d'), interval='1d')
        index_data_daily = index_data_daily[['Adj Close']].rename(columns={'Adj Close': 'Close'}) if 'Adj Close' in index_data_daily.columns else index_data_daily[['Close']]
        index_data = index_data_daily[index_data_daily.index.weekday == 4].dropna()

        assets = {
            'Financeiro': ['BBDC4.SA'],
            'Energia': ['PETR4.SA', 'ELET3.SA'],
            'Consumo': ['ABEV3.SA', 'MGLU3.SA'],
            'Mineração': ['VALE3.SA'],
            'Indústria': ['EMBR3.SA', 'WEGE3.SA'],
            'Varejo': ['LREN3.SA', 'B3SA3.SA']
        }
        assets_list = [item for sublist in assets.values() for item in sublist]

        assets_data_daily = yf.download(assets_list, start='2000-01-01', end=current_date.strftime('%Y-%m-%d'), interval='1d')['Close']
        assets_data = assets_data_daily[assets_data_daily.index.weekday == 4].dropna()

        with open(index_data_path, 'wb') as f:
            pickle.dump(index_data, f)
        with open(assets_data_path, 'wb') as f:
            pickle.dump(assets_data, f)

        return index_data, assets_data

    if os.path.exists(index_data_path) and os.path.exists(assets_data_path):
        with open(index_data_path, 'rb') as f:
            index_data = pickle.load(f)
        with open(assets_data_path, 'rb') as f:
            assets_data = pickle.load(f)

        if is_new_week(index_data.index[-1].to_pydatetime(), current_date):
            index_data, assets_data = download_and_save_data()
    else:
        index_data, assets_data = download_and_save_data()

    index_data_temp = calculate_technical_indicators(index_data.copy())
    index_data_temp = create_target(index_data_temp).dropna()
    assets_returns_temp = assets_data.pct_change().dropna()
    common_index = index_data_temp.index.intersection(assets_returns_temp.index)

    return index_data, assets_data, common_index

# Carregar dados
index_data, assets_data, common_index = load_data()

# Definir modelos e estratégias
models = {
    'XGBoost': XGBClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
    'MLPClassifier': MLPClassifier(random_state=42, max_iter=1000, shuffle=False)
}

param_grids = {
    'XGBoost': {'max_depth': [3, 5], 'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
    'RandomForest': {'n_estimators': [50, 100], 'max_depth': [None, 10]},
    'LightGBM': {'num_leaves': [15, 31], 'learning_rate': [0.01, 0.1], 'n_estimators': [50, 100]},
    'MLPClassifier': {'hidden_layer_sizes': [(50,), (100,)], 'learning_rate_init': [0.001, 0.01]}
}

strategies = {
    'XGBoost': (models['XGBoost'], param_grids['XGBoost'], None),
    'RandomForest': (models['RandomForest'], param_grids['RandomForest'], None),
    'LightGBM': (models['LightGBM'], param_grids['LightGBM'], None),
    'MLPClassifier': (models['MLPClassifier'], param_grids['MLPClassifier'], None),
    'Markowitz': (None, None, None),
    'Fixed Lambda 0.1': (None, None, 0.1),
    'Fixed Lambda 0.5': (None, None, 0.5),
    'Fixed Lambda 1.0': (None, None, 1.0),
    'Buy and Hold': (None, None, None),
    'Ibovespa': (None, None, None)
}

# Executar backtest
results = {}
weights_histories = {}
prob_hists = {}
capital_histories = {}
initial_capital = 100000

for strategy_name, (model, param_grid, lambda_val) in strategies.items():
    with st.spinner(f"Calculando {strategy_name}..."):
        try:
            results_df, weights_history, prob_hist, capital_history = portfolio_rebalance(
                index_data, assets_data, strategy_name, model, param_grid, lambda_val,
                common_index=common_index, test_start_date='2015-01-01', index_name='ibovespa',
                initial_capital=initial_capital
            )
            if not results_df.empty:
                results[strategy_name] = results_df
                weights_histories[strategy_name] = weights_history
                prob_hists[strategy_name] = prob_hist
                capital_histories[strategy_name] = capital_history
            else:
                st.warning(f"Resultados vazios para {strategy_name}. Pulando estratégia.")
        except Exception as e:
            st.error(f"Erro ao processar {strategy_name}: {str(e)}")

# Calcular métricas
strategy_metrics = {}
for strategy_name in results.keys():
    metrics = calculate_metrics(results[strategy_name], initial_capital)
    strategy_metrics[strategy_name] = metrics

sorted_strategies = sorted(
    strategy_metrics.keys(),
    key=lambda x: strategy_metrics[x]['Retorno Cumulativo (%)'],
    reverse=True
)

# Layout principal
col_left, col_right = st.columns([6.15, 5])

with col_right:
    st.subheader("Filtros")
    selected_strategies = st.multiselect(
        "Selecione estratégias para análise",
        sorted_strategies,
        default=sorted_strategies,
        key='selected_strategies'
    )

    st.subheader("Resumo das Estratégias")
    comparison_data = []
    for strategy_name in selected_strategies:
        metrics = strategy_metrics[strategy_name]
        comparison_data.append({
            'Estratégia': strategy_name,
            'Capital Final (R$)': f"{metrics['Capital Final']:,.2f}",
            'Retorno Cumulativo (%)': f"{metrics['Retorno Cumulativo (%)']:.2f}",
            'Sharpe Ratio': f"{metrics['Sharpe Ratio']:.2f}",
            'Max Drawdown (%)': f"{abs(metrics['Max Drawdown (%)']):.2f}"
        })
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

with col_left:
    st.subheader("Evolução do Capital")
    capital_data = pd.DataFrame()
    for strategy in selected_strategies:
        temp_df = pd.DataFrame({
            'Data': results[strategy]['Index'],
            'Capital (R$)': results[strategy]['Capital'],
            'Estratégia': strategy
        })
        capital_data = pd.concat([capital_data, temp_df])

    if not capital_data.empty:
        fig_capital = px.line(
            capital_data,
            x='Data',
            y='Capital (R$)',
            color='Estratégia',
            title='Evolução do Capital (R$)',
            height=800,
            labels={'Capital (R$)': 'Capital'}
        )
        fig_capital.update_layout(yaxis_tickformat=",.0f", yaxis_title="Capital (R$)")
        st.plotly_chart(fig_capital, use_container_width=True)

# Seção de métricas detalhadas e análise
st.subheader("Análise Detalhada")
for strategy_name in sorted_strategies:
    with st.expander(f"{strategy_name}", expanded=False):
        # Métricas detalhadas
        st.write("**Métricas de Desempenho**")
        metrics = strategy_metrics[strategy_name]
        baseline = strategy_metrics.get('Ibovespa', {})

        # Organizar métricas em pares de colunas
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Capital Inicial", f"R$ {metrics['Capital Inicial']:,.2f}")
        with col2:
            st.metric("Capital Final", f"R$ {metrics['Capital Final']:,.2f}",
                     delta=f"R$ {metrics['Capital Final'] - baseline.get('Capital Final', initial_capital):,.2f}")

        col3, col4 = st.columns(2)
        with col3:
            st.metric("Retorno Cumulativo", f"{metrics['Retorno Cumulativo (%)']:.2f}%",
                     delta=f"{metrics['Retorno Cumulativo (%)'] - baseline.get('Retorno Cumulativo (%)', 0):.2f}%")
        with col4:
            st.metric("Retorno Anualizado", f"{metrics['Retorno Anualizado (%)']:.2f}%",
                     delta=f"{metrics['Retorno Anualizado (%)'] - baseline.get('Retorno Anualizado (%)', 0):.2f}%")

        col5, col6 = st.columns(2)
        with col5:
            st.metric("Volatilidade Anualizada", f"{metrics['Volatilidade Anualizada (%)']:.2f}%")
        with col6:
            st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}",
                     delta=f"{metrics['Sharpe Ratio'] - baseline.get('Sharpe Ratio', 0):.2f}")

        col7, col8 = st.columns(2)
        with col7:
            st.metric("Max Drawdown", f"{abs(metrics['Max Drawdown (%)']):.2f}%")
        with col8:
            # Adicionar sugestão semanal do portfólio aqui para aproveitar o espaço
            latest_weights = weights_histories[strategy_name][-1] if weights_histories[strategy_name] else None
            if latest_weights is not None:
                weights_df = pd.DataFrame({
                    'Ativo': assets_data.columns,
                    'Peso': latest_weights
                })
                weights_df = weights_df[weights_df['Peso'] > 0.009]
                weights_df['Peso (%)'] = weights_df['Peso'].apply(lambda w: f"{w*100:.2f}")
                weights_df = weights_df[['Ativo', 'Peso (%)']].sort_values(by='Peso (%)', ascending=False).reset_index(drop=True)
                st.write("**Sugestão Semanal do Portfólio**")
                st.table(weights_df)

        # Gráfico de Drawdown
        capital = results[strategy_name]['Capital']
        peak = capital.cummax()
        drawdown = (capital - peak) / peak * 100
        dd_df = pd.DataFrame({
            'Data': results[strategy_name]['Index'],
            'Drawdown (%)': drawdown
        })
        fig_dd = px.area(
            dd_df,
            x='Data',
            y='Drawdown (%)',
            title=f'Drawdown - {strategy_name}',
            height=400
        )
        fig_dd.update_layout(yaxis_tickformat=".1f")
        st.plotly_chart(fig_dd, use_container_width=True)

        # Retornos semanais
        fig_returns = px.line(
            results[strategy_name],
            x='Index',
            y='Portfolio_Return',
            title=f'Retornos Semanais - {strategy_name}',
            height=400,
            labels={'Portfolio_Return': 'Retorno (%)'}
        )
        fig_returns.update_layout(yaxis_tickformat=".2%")
        st.plotly_chart(fig_returns, use_container_width=True)

        # Histograma das Probabilidades (apenas para estratégias de ML)
        ml_strategies = ['XGBoost', 'RandomForest', 'LightGBM', 'MLPClassifier']
        if strategy_name in ml_strategies and strategy_name in prob_hists and prob_hists[strategy_name] is not None:
            st.subheader("Distribuição das Probabilidades")
            prob_df = pd.DataFrame({
                'Probabilidade Alta': prob_hists[strategy_name]
            })
            fig_hist = px.histogram(
                prob_df,
                x='Probabilidade Alta',
                nbins=20,
                title=f'Distribuição das Probabilidades - {strategy_name}'
            )
            st.plotly_chart(fig_hist, use_container_width=True)
# Evolução dos Pesos
st.subheader("Evolução dos Pesos")
valid_strategies = [s for s in sorted_strategies if weights_histories.get(s)]
selected_strategy_weights = st.selectbox(
    "Selecione uma estratégia para os pesos",
    valid_strategies,
    key='weights_strategy'
)

if selected_strategy_weights:
    weights_df = pd.DataFrame(
        weights_histories[selected_strategy_weights],
        index=results[selected_strategy_weights]['Index'],
        columns=assets_data.columns
    ).melt(ignore_index=False, var_name='Ativo', value_name='Peso').reset_index()
    fig_weights = px.area(
        weights_df,
        x='Index',
        y='Peso',
        color='Ativo',
        title=f'Evolução dos Pesos - {selected_strategy_weights}',
        height=500,
        labels={'Peso': 'Peso (%)'}
    )
    fig_weights.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig_weights, use_container_width=True)
