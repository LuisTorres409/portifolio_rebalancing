import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from portfolio_utils import (
    portfolio_rebalance, calculate_metrics, calculate_technical_indicators,
    create_target, XGBClassifier, RandomForestClassifier, LGBMClassifier,
    MLPClassifier
)
import os
import pickle
from datetime import datetime

st.set_page_config(layout="wide")

st.title("Backtest de Estratégias de Rebalanceamento - Ibovespa")
st.write("Este aplicativo realiza um backtest de várias estratégias de rebalanceamento de portfólio com base em ativos do Ibovespa.")

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

index_data, assets_data, common_index = load_data()

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

results = {}
weights_histories = {}
prob_hists = {}

for strategy_name, (model, param_grid, lambda_val) in strategies.items():
    with st.spinner(f"Calculando {strategy_name}..."):
        results_df, weights_history, prob_hist = portfolio_rebalance(
            index_data, assets_data, strategy_name, model, param_grid, lambda_val, 
            common_index=common_index, test_start_date='2015-01-01', index_name='ibovespa'
        )
        results[strategy_name] = results_df
        weights_histories[strategy_name] = weights_history
        prob_hists[strategy_name] = prob_hist

strategy_returns = {
    name: calculate_metrics(results[name])["Retorno Cumulativo (%)"]
    for name in strategies.keys()
}
sorted_strategies = sorted(strategy_returns.keys(), key=lambda x: strategy_returns[x], reverse=True)

# Pré-calcular os dados de pesos históricos
weights_history_data = {}
for strategy_name in sorted_strategies:
    if weights_histories[strategy_name] is not None:
        try:
            weights_history_df = pd.DataFrame(
                weights_histories[strategy_name],
                index=results[strategy_name]['Index'],
                columns=assets_data.columns
            )
            weights_history_df = weights_history_df.melt(
                ignore_index=False,
                var_name='Ativo',
                value_name='Peso'
            ).reset_index()
            weights_history_data[strategy_name] = weights_history_df
        except Exception as e:
            st.warning(f"Erro ao processar pesos históricos para {strategy_name}: {e}")
            weights_history_data[strategy_name] = None

col_left, col_right = st.columns([6, 4])

with col_right:
    st.subheader("Filtros")
    selected_strategies = st.multiselect(
        "Selecione as estratégias para visualizar no gráfico",
        sorted_strategies,
        default=sorted_strategies,
        key='selected_strategies'
    )

    st.subheader("Comparação de Resultados")
    comparison_data = []
    for strategy_name in sorted_strategies:
        metrics = calculate_metrics(results[strategy_name])
        comparison_data.append({
            'Estratégia': strategy_name,
            'Retorno Cumulativo (%)': f"{metrics['Retorno Cumulativo (%)']:.2f}",
            'Retorno Anualizado (%)': f"{metrics['Retorno Anualizado (%)']:.2f}",
            'Desvio Padrão Anualizado (%)': f"{metrics['Desvio Padrão Anualizado (%)']:.2f}",
            'Sharpe Ratio': f"{metrics['Sharpe Ratio']:.2f}"
        })
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df[comparison_df['Estratégia'].isin(selected_strategies)].reset_index(drop=True)
    st.table(comparison_df)

with col_left:
    st.subheader("Retorno Cumulativo de Todas as Estratégias")
    
    # Create a fixed color mapping for all strategies
    color_discrete_map = {
        'XGBoost': '#1f77b4',
        'RandomForest': '#ff7f0e',
        'LightGBM': '#2ca02c',
        'MLPClassifier': '#d62728',
        'Markowitz': '#9467bd',
        'Fixed Lambda 0.1': '#8c564b',
        'Fixed Lambda 0.5': '#e377c2',
        'Fixed Lambda 1.0': '#7f7f7f',
        'Buy and Hold': '#bcbd22',
        'Ibovespa': '#17becf'
    }
    
    # Prepare the data for plotting
    cumulative_data = pd.DataFrame()
    for strategy in selected_strategies:  # Use the selection from col_right
        cumulative_return = (1 + results[strategy]['Portfolio_Return']).cumprod() - 1
        temp_df = pd.DataFrame({
            'Data': results[strategy]['Index'],
            'Retorno Cumulativo (%)': cumulative_return * 100,
            'Estratégia': strategy
        })
        cumulative_data = pd.concat([cumulative_data, temp_df])

    # Create and display the plot
    if not cumulative_data.empty:
        fig_cum = px.line(
            cumulative_data,
            x='Data',
            y='Retorno Cumulativo (%)',
            color='Estratégia',
            color_discrete_map=color_discrete_map,  # Apply fixed color mapping
            title='Retorno Cumulativo das Estratégias Selecionadas',
            height=800
        )
        st.plotly_chart(fig_cum, use_container_width=True)
    else:
        st.warning("Selecione pelo menos uma estratégia para visualizar")

st.subheader("Resultados Detalhados das Estratégias")
for i in range(0, len(sorted_strategies), 3):
    cols = st.columns(3)
    for j, col in enumerate(cols):
        if i + j < len(sorted_strategies):
            strategy_name = sorted_strategies[i + j]
            with col:
                with st.expander(f"{strategy_name}", expanded=False):
                    col1, col2 = st.columns(2)

                    latest_weights = weights_histories[strategy_name][-1] if weights_histories[strategy_name] else None
                    if latest_weights is not None:
                        # Cria DataFrame com valores numéricos
                        weights_df = pd.DataFrame({
                            'Ativo': assets_data.columns,
                            'Peso': latest_weights  # valores reais de 0 a 1
                        })

                        # Filtra os pesos positivos
                        weights_df = weights_df[weights_df['Peso'] > 0.009]

                        # Converte para percentual com string formatada após o filtro
                        weights_df['Peso (%)'] = weights_df['Peso'].apply(lambda w: f"{w*100:.2f}")

                        # Reorganiza colunas e ordena
                        weights_df = weights_df[['Ativo', 'Peso (%)']].sort_values(by='Peso (%)', ascending=False).reset_index(drop=True)

                        with col1:
                            st.write("Sugestão Semanal do Portfólio")
                            st.table(weights_df)

                    metrics = calculate_metrics(results[strategy_name])
                    baseline_metrics = calculate_metrics(results['Ibovespa'])
                    with col2:
                        st.write("Indicadores de Desempenho")
                        st.metric("Retorno Cumulativo (%)", f"{metrics['Retorno Cumulativo (%)']:.2f}", delta=f"{metrics['Retorno Cumulativo (%)'] - baseline_metrics['Retorno Cumulativo (%)']:.2f}")
                        st.metric("Retorno Anualizado (%)", f"{metrics['Retorno Anualizado (%)']:.2f}", delta=f"{metrics['Retorno Anualizado (%)'] - baseline_metrics['Retorno Anualizado (%)']:.2f}")
                        st.metric("Desvio Padrão Anualizado (%)", f"{metrics['Desvio Padrão Anualizado (%)']:.2f}", delta=f"{metrics['Desvio Padrão Anualizado (%)'] - baseline_metrics['Desvio Padrão Anualizado (%)']:.2f}")
                        st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}", delta=f"{metrics['Sharpe Ratio'] - baseline_metrics['Sharpe Ratio']:.2f}")

                    fig = px.line(
                        results[strategy_name],
                        x='Index',
                        y='Portfolio_Return',
                        title=f'Retorno Semanal - {strategy_name}'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Histograma das Probabilidades (apenas para estratégias de ML)
                    if strategy_name in prob_hists and prob_hists[strategy_name] is not None:
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

# Novo gráfico de pesos históricos no final
st.subheader("Evolução dos Pesos ao Longo do Tempo")
valid_strategies = [s for s in sorted_strategies if weights_history_data.get(s) is not None]
selected_strategy = st.selectbox(
    "Selecione uma estratégia para visualizar os pesos históricos",
    valid_strategies,
    key='weights_history_strategy'
)

if selected_strategy and weights_history_data[selected_strategy] is not None:
    fig_weights = px.line(
        weights_history_data[selected_strategy],
        x='Index',
        y='Peso',
        color='Ativo',
        title=f'Distribuição de Pesos - {selected_strategy}',
        height=600
    )
    st.plotly_chart(fig_weights, use_container_width=True)
else:
    st.write("Nenhum dado de pesos históricos disponível para a estratégia selecionada.")