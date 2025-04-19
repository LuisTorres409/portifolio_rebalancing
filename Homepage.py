import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def main():
    st.set_page_config(
        page_title="QuantLab - Otimização de Portfólio",
        page_icon="📊",
        layout="wide"
    )

    # Sidebar navigation
    show_homepage()

def show_homepage():
    # Header with logo
    _, col2 = st.columns([1, 4])
    with col2:
        st.title("QuantLab - Otimização de Portfólio Inteligente")
        st.markdown("""
        **Plataforma quantitativa avançada para construção e otimização de carteiras de investimentos**
        """)

    st.markdown("---")

    # Introduction
    st.header("📌 Sobre o Projeto")
    st.markdown("""
    O QuantLab implementa estratégias quantitativas sofisticadas para otimização de portfólio nos mercados do 
    **S&P 500** (EUA) e **Ibovespa** (Brasil). Combinamos:
    
    - Algoritmos de machine learning para previsão de retornos
    - Técnicas modernas de otimização de carteiras
    - Análise de risco robusta
    
    Tudo integrado em uma interface intuitiva para auxiliar na tomada de decisões de investimento.
    """)

    # Key Features
    st.header("✨ Diferenciais do QuantLab")
    features = [
        "🔍 **Análise Técnica Avançada**: 15+ indicadores técnicos calculados automaticamente",
        "🤖 **Inteligência Artificial**: Modelos preditivos (XGBoost, LightGBM) com calibração dinâmica",
        "📊 **Otimização Adaptativa**: Algoritmos que se ajustam às condições de mercado",
        "⚖️ **Controle de Risco**: Múltiplas métricas de risco integradas (VaR, CVaR, Drawdown)",
        "📈 **Backtesting Rigoroso**: Avaliação de desempenho em diferentes regimes de mercado",
        "📱 **Visualização Interativa**: Gráficos dinâmicos e relatórios completos"
    ]
    for feature in features:
        st.markdown(feature)

    st.markdown("---")

    # Methodology Section
    st.header("📚 Metodologias Científicas")

    # Mean-Gini Explanation
    with st.expander("🧠 Otimização Mean-Gini (Abordagem Principal)", expanded=True):
        st.markdown("""
        ### Formulação Matemática

        Maximizamos a função objetivo:

        $$
        \\max_w \\Phi(w) = \\lambda\\cdot\\left(\\frac{\\mu_p}{R_{\\max}}\\right) - (1-\\lambda)\\cdot\\left(\\frac{\\sigma_p}{G_{\\max}}\\right)
        $$

        Onde:
        - $w$: Vetor de pesos do portfólio
        - $\\lambda = 1 - P(\\text{retorno} > 0)$: Parâmetro de aversão ao risco derivado do modelo de ML
        - $\\mu_p = w^T \\mu$: Retorno esperado do portfólio
        - $\\sigma_p = \\sqrt{w^T \\Sigma w}$: Risco (desvio padrão) do portfólio
        - $R_{\\max} = \\max(\\mu)$: Máximo retorno entre os ativos
        - $G_{\\max} = \\max(\\text{diag}(\\Sigma))$: Máxima variância entre os ativos

        ### Vantagens Competitivas
        - ✅ **Adaptabilidade**: O parâmetro $\\lambda$ é ajustado dinamicamente com base nas probabilidades do modelo de ML
        - ✅ **Normalização**: Retorno e risco são escalados para comparação equitativa
        - ✅ **Robustez**: Menos sensível a outliers que abordagens tradicionais
        - ✅ **Diversificação**: Limites de concentração (pesos entre 5% e 80% por ativo)
        """)

    # Markowitz Explanation
    with st.expander("📉 Otimização Markowitz (Referência Clássica)"):
        st.markdown("""
        ### Formulação Matemática

        O problema de otimização é formulado como:

        $$
        \\begin{aligned}
        & \\max_w w^T\\mu - \\gamma w^T\\Sigma w \\\\
        & \\text{sujeito a:} \\\\
        & \\sum w_i = 1 \\\\
        & 0 \\leq w_i \\leq 1 \\quad \\forall i
        \\end{aligned}
        $$

        Onde:
        - $\\gamma$: Coeficiente de aversão ao risco
        - $\\Sigma$: Matriz de covariância dos retornos

        ### Características Principais
        - 📌 **Fundação Teórica**: Base da Teoria Moderna de Portfólio (Markowitz, 1952)
        - 📌 **Balanço Retorno-Risco**: Encontra o melhor trade-off entre retorno esperado e variância
        - 📌 **Implementação**: Utilizamos janela móvel de 3 anos para estimação de parâmetros
        - 📌 **Limitações**: Sensível a erros de estimativa dos parâmetros de entrada
        """)

    # ML Models Explanation
    with st.expander("🤖 Modelos Preditivos de Machine Learning"):
        st.markdown("""
        ### Arquitetura Híbrida Quantitativa

        Integração perfeita entre ML e otimização:

        1. **Engenharia de Features**:
           - 15+ indicadores técnicos (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
           - Features macroeconômicas (quando disponíveis)
           - Target: $y_t = \\mathbb{I}(r_{t+1} > 0)$ (indicador de retorno positivo)

        2. **Modelagem Preditiva**:
           ```python
           from xgboost import XGBClassifier
           
           model = XGBClassifier(
               n_estimators=200,
               max_depth=4,
               learning_rate=0.05,
               subsample=0.8,
               colsample_bytree=0.8,
               early_stopping_rounds=20
           )
           model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
           prob_up = model.predict_proba(X_live)[:, 1]  # Probabilidade de retorno positivo
           ```

        3. **Integração com Otimização**:
           - $\\lambda = 1 - \\text{prob\_up}$: Converte probabilidade em aversão ao risco
           - Rebalanceamento semanal com janela móvel de treinamento
           - Mecanismo de meta-learning para adaptar hiperparâmetros
        """)

    st.markdown("---")

    # How to Use
    st.header("🚀 Guia Rápido")
    st.markdown("""
    1. **Selecione o mercado** na barra lateral:
       - 🇺🇸 S&P 500: Para exposição a grandes empresas globais
       - 🇧🇷 Ibovespa: Para foco no mercado acionário brasileiro
    
    2. **Escolha sua estratégia**:
       - 🧠 **Mean-Gini Adaptativo**: Nossa abordagem mais sofisticada (recomendado)
       - 📉 **Markowitz Clássico**: Para comparação com o benchmark teórico
       - 🏦 **Buy-and-Hold**: Benchmark passivo
    
    3. **Personalize os parâmetros** (opcional):
       - Horizonte de investimento
       - Restrições de alocação
       - Nível de aversão ao risco
    
    4. **Analise os resultados**:
       - 📈 Performance absoluta e relativa
       - 📉 Métricas de risco ajustado
       - 🏗 Composição e evolução da carteira
       - 🔄 Estatísticas de turnover e custos
    """)

    # Footer
    st.markdown("""
    ---
    **Desenvolvido por** [Seu Nome] | [GitHub](https://github.com/seuuser) | [LinkedIn](https://linkedin.com/in/seuuser)
    
    *Dados históricos fornecidos pelo Yahoo Finance. Resultados passados não garantem desempenho futuro. 
    Plataforma destinada a fins educacionais e de pesquisa quantitativa.*
    """)

if __name__ == "__main__":
    main()