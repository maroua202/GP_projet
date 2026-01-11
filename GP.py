import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from datetime import datetime, timedelta
import requests
import time

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# ==================== FONCTION DE SCRAP YAHOO ====================
def download_yahoo_data(ticker, start_date, end_date):
    start_ts = int(time.mktime(datetime.strptime(start_date, "%Y-%m-%d").timetuple()))
    end_ts = int(time.mktime(datetime.strptime(end_date, "%Y-%m-%d").timetuple()))

    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": start_ts,
        "period2": end_ts,
        "interval": "1d",
        "events": "history"
    }

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0'}
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    result = data['chart']['result'][0]
    timestamps = result['timestamp']
    ohlcv = result['indicators']['quote'][0]
    adj_close = result['indicators']['adjclose'][0]['adjclose']

    df = pd.DataFrame({
        'Date': pd.to_datetime(timestamps, unit='s'),
        'Open': ohlcv['open'],
        'High': ohlcv['high'],
        'Low': ohlcv['low'],
        'Close': ohlcv['close'],
        'Adj Close': adj_close,
        'Volume': ohlcv['volume']
    })

    df.set_index('Date', inplace=True)
    return df['Adj Close']

# ==================== CLASSE DL POUR OPTIMISATION ====================
class PortfolioOptimizerDL:
    def __init__(self, n_actifs, taux_sans_risque=0.02):
        self.n_actifs = n_actifs
        self.taux_sans_risque = taux_sans_risque

        self.reseau = nn.Sequential(
            nn.Linear(n_actifs * 3, 50),
            nn.ReLU(),
            nn.Linear(50, n_actifs),
            nn.Softmax(dim=-1)
        )
        self.optimizer = torch.optim.Adam(self.reseau.parameters(), lr=0.01)

    def calculer_sharpe(self, poids, rendements_moyens, matrice_cov):
        rendement_portefeuille = torch.sum(poids * rendements_moyens)
        variance = torch.matmul(poids.T, torch.matmul(matrice_cov, poids))
        sharpe = (rendement_portefeuille - self.taux_sans_risque) / torch.sqrt(variance + 1e-8)
        return sharpe

    def optimiser(self, rendements_moyens_annualises, matrice_cov_annualisee, n_iterations=1000):
        rendements_tensor = torch.tensor(rendements_moyens_annualises, dtype=torch.float32)
        cov_tensor = torch.tensor(matrice_cov_annualisee, dtype=torch.float32)

        meilleur_sharpe = -float('inf')
        meilleurs_poids = None

        for i in range(n_iterations):
            features = torch.cat([
                rendements_tensor,
                torch.diag(cov_tensor),
                torch.randn(self.n_actifs) * 0.1
            ])
            poids = self.reseau(features)
            sharpe = self.calculer_sharpe(poids, rendements_tensor, cov_tensor)
            
            loss = -sharpe
            penalty_diversification = -torch.sum(poids * torch.log(poids + 1e-8))
            loss = loss + 0.1 * penalty_diversification

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            sharpe_value = sharpe.item()
            if sharpe_value > meilleur_sharpe:
                meilleur_sharpe = sharpe_value
                meilleurs_poids = poids.detach().numpy()

        return meilleurs_poids, meilleur_sharpe

# ==================== FONCTIONS D'AIDE ====================
def calculate_var_tvar(returns, weights, confidence_level=0.95):
    """Calculate VaR and TVaR (Conditional VaR)"""
    portfolio_returns = returns @ weights
    var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    tvar = portfolio_returns[portfolio_returns <= var].mean()
    return var, tvar

def generate_efficient_frontier(rendements_moyens, matrice_cov, risk_free_rate, n_portfolios=5000):
    n_actifs = len(rendements_moyens)
    rendements_portfolios = []
    volatilites_portfolios = []
    ratios_sharpe = []
    all_weights = []

    for _ in range(n_portfolios):
        poids = np.random.random(n_actifs)
        poids = poids / np.sum(poids)
        all_weights.append(poids)
        
        rendement = np.sum(poids * rendements_moyens)
        volatilite = np.sqrt(poids @ matrice_cov @ poids)
        sharpe = (rendement - risk_free_rate) / (volatilite + 1e-8)
        
        rendements_portfolios.append(rendement)
        volatilites_portfolios.append(volatilite)
        ratios_sharpe.append(sharpe)

    return rendements_portfolios, volatilites_portfolios, ratios_sharpe, all_weights

# ==================== STREAMLIT UI ====================
st.title("🏦 DeepAlloc Portfolio Optimizer")
st.markdown("---")

# Sidebar pour les paramètres
with st.sidebar:
    st.header("Paramètres")
    
    # Entrée des tickers
    st.subheader("Actifs")
    default_tickers = "SPY, BND, GLD, VTI, QQQ"
    tickers_input = st.text_area("Entrez les symboles (séparés par des virgules):", 
                                 value=default_tickers, height=100)
    tickers = [ticker.strip() for ticker in tickers_input.split(',') if ticker.strip()]
    
    # Période
    st.subheader("Période historique")
    years = st.slider("Années de données historiques", 1, 10, 3)
    
    # Taux sans risque
    st.subheader("Paramètres financiers")
    risk_free_rate = st.number_input("Taux sans risque (%)", min_value=0.0, max_value=10.0, value=2.0) / 100
    
    # Niveau de confiance VaR
    confidence_level = st.slider("Niveau de confiance VaR (%)", 90, 99, 95) / 100
    
    # Paramètres DL
    st.subheader("Paramètres Deep Learning")
    n_iterations = st.slider("Nombre d'itérations", 500, 5000, 1000, step=500)
    
    if st.button(" Lancer l'optimisation", type="primary"):
        st.session_state.run_optimization = True
    else:
        st.session_state.run_optimization = False

# Main content
if tickers and st.session_state.get('run_optimization', False):
    try:
        # Section de chargement
        with st.spinner(f"Téléchargement des données pour {len(tickers)} actifs..."):
            end_date = datetime.today()
            start_date = end_date - timedelta(days=years*365)
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # Télécharger chaque actif via la fonction personnalisée
            data_dict = {}
            for ticker in tickers:
                data_dict[ticker] = download_yahoo_data(ticker, start_str, end_str)
            
            data = pd.DataFrame(data_dict)
            
            if data.empty or len(data) < 10:
                st.error("Erreur lors du téléchargement des données. Vérifiez les symboles.")
                st.stop()
            
            daily_returns = np.log(data / data.shift(1)).dropna()
            
            cumulative_returns = daily_returns.cumsum()
            st.subheader("📈 Evolution historique des différents actifs")
            fig, ax = plt.subplots(figsize=(10, 5))
            cumulative_returns.plot(ax=ax)
            st.pyplot(fig)
            
            # Calculs statistiques
            rendements_moyens = daily_returns.mean() * 252
            matrice_cov = daily_returns.cov() * 252
            
            # Affichage des statistiques
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Jours de données", len(daily_returns))
            with col2:
                st.metric("Actifs analysés", len(tickers))
            with col3:
                st.metric("Période", f"{years} ans")
            
            st.markdown("---")
            
            # Optimisation DL
            st.subheader("🎯 Optimisation Deep Learning")
            
            with st.spinner("Optimisation en cours..."):
                optimiseur = PortfolioOptimizerDL(
                    n_actifs=len(tickers),
                    taux_sans_risque=risk_free_rate
                )
                
                poids_opt, sharpe_opt = optimiseur.optimiser(
                    rendements_moyens.values,
                    matrice_cov.values,
                    n_iterations=n_iterations
                )
            
            # Calcul des métriques
            rendement = np.sum(poids_opt * rendements_moyens.values)
            volatilite = np.sqrt(poids_opt @ matrice_cov.values @ poids_opt)
            
            # Calcul VaR et TVaR
            var_1d, tvar_1d = calculate_var_tvar(daily_returns.values, poids_opt, confidence_level)
            var_annual = var_1d * np.sqrt(252)
            tvar_annual = tvar_1d * np.sqrt(252)
            
            # ==================== AFFICHAGE DES RÉSULTATS ====================
            st.success("Optimisation terminée!")
            
            # Métriques principales
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Sharpe Ratio", f"{sharpe_opt:.4f}")
            with col2:
                st.metric("Rendement annuel", f"{rendement*100:.2f}%")
            with col3:
                st.metric("Volatilité annuelle", f"{volatilite*100:.2f}%")
            with col4:
                st.metric(f"VaR ({confidence_level*100:.0f}%) 1 jour", f"{var_1d*100:.2f}%")
            with col5:
                st.metric(f"TVaR ({confidence_level*100:.0f}%) 1 jour", f"{tvar_1d*100:.2f}%")
            
            st.markdown("---")
            
            # Layout en colonnes
            left_col, right_col = st.columns([1, 1])
            
            with left_col:
                # Tableau des poids
                st.subheader(" Distribution du portefeuille")
                
                weights_df = pd.DataFrame({
                    'Actif': tickers,
                    'Poids (%)': poids_opt * 100,
                    'Rendement (%)': rendements_moyens.values * 100
                })
                
                # Trier par poids décroissant
                weights_df = weights_df.sort_values('Poids (%)', ascending=False)
                
                # Afficher le tableau
                st.dataframe(
                    weights_df.style.format({
                        'Poids (%)': '{:.2f}%',
                        'Rendement (%)': '{:.2f}%'
                    }),
                    use_container_width=True
                )
                
                # Pie chart avec Plotly
                st.subheader(" Répartition du portefeuille")
                
                fig_pie = px.pie(
                    weights_df,
                    values='Poids (%)',
                    names='Actif',
                    hole=0.3,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with right_col:
                # Efficient Frontier avec Plotly
                st.subheader("Frontière Efficiente")
                
                with st.spinner("Génération de la frontière efficiente..."):
                    rendements_pf, volatilites_pf, sharpes_pf, all_weights = generate_efficient_frontier(
                        rendements_moyens.values,
                        matrice_cov.values,
                        risk_free_rate,
                        n_portfolios=3000
                    )
                
                # Création du scatter plot
                fig = go.Figure()
                
                # Tous les portefeuilles
                fig.add_trace(go.Scatter(
                    x=volatilites_pf,
                    y=rendements_pf,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=sharpes_pf,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Sharpe Ratio")
                    ),
                    name='Portefeuilles aléatoires',
                    hovertemplate='<b>Volatilité:</b> %{x:.3f}<br>' +
                                 '<b>Rendement:</b> %{y:.3f}<br>' +
                                 '<b>Sharpe:</b> %{marker.color:.3f}<extra></extra>'
                ))
                
                # Portefeuille optimal (DL)
                fig.add_trace(go.Scatter(
                    x=[volatilite],
                    y=[rendement],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color='red',
                        symbol='star',
                        line=dict(width=2, color='black')
                    ),
                    name='Portefeuille Optimal (DL)',
                    hovertemplate='<b>Portefeuille Optimal</b><br>' +
                                 '<b>Volatilité:</b> %{x:.3f}<br>' +
                                 '<b>Rendement:</b> %{y:.3f}<br>' +
                                 '<b>Sharpe:</b> ' + f'{sharpe_opt:.3f}<extra></extra>'
                ))
                
                # Portefeuille à volatilité minimale
                idx_min_vol = np.argmin(volatilites_pf)
                fig.add_trace(go.Scatter(
                    x=[volatilites_pf[idx_min_vol]],
                    y=[rendements_pf[idx_min_vol]],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='green',
                        symbol='triangle-up',
                        line=dict(width=2, color='black')
                    ),
                    name='Volatilité minimale',
                    hovertemplate='<b>Volatilité minimale</b><br>' +
                                 '<b>Volatilité:</b> %{x:.3f}<br>' +
                                 '<b>Rendement:</b> %{y:.3f}<extra></extra>'
                ))
                
                # Portefeuille à Sharpe maximum (Monte Carlo)
                idx_max_sharpe = np.argmax(sharpes_pf)
                fig.add_trace(go.Scatter(
                    x=[volatilites_pf[idx_max_sharpe]],
                    y=[rendements_pf[idx_max_sharpe]],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='blue',
                        symbol='square',
                        line=dict(width=2, color='black')
                    ),
                    name='Sharpe max (MC)',
                    hovertemplate='<b>Sharpe max (MC)</b><br>' +
                                 '<b>Volatilité:</b> %{x:.3f}<br>' +
                                 '<b>Rendement:</b> %{y:.3f}<extra></extra>'
                ))
                
                # Mise en forme
                fig.update_layout(
                    title='Frontière Efficiente',
                    xaxis_title='Volatilité annuelle',
                    yaxis_title='Rendement annuel',
                    hovermode='closest',
                    height=500,
                    showlegend=True,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Section supplémentaire : Distribution des rendements et VaR
            st.markdown("---")
            st.subheader("Analyse des risques")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution des rendements du portefeuille
                portfolio_returns = daily_returns.values @ poids_opt
                
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=portfolio_returns * 100,
                    nbinsx=50,
                    name='Rendements',
                    marker_color='skyblue',
                    opacity=0.7
                ))
                
                # Ligne VaR
                fig_dist.add_vline(
                    x=var_1d * 100,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"VaR {confidence_level*100:.0f}%: {var_1d*100:.2f}%",
                    annotation_position="top right"
                )
                
                fig_dist.update_layout(
                    title='Distribution des rendements quotidiens',
                    xaxis_title='Rendement quotidien (%)',
                    yaxis_title='Fréquence',
                    height=400
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                # Matrice de corrélation
                st.subheader(" Matrice de corrélation")
                
                corr_matrix = daily_returns.corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1
                )
                fig_corr.update_layout(height=400)
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # Téléchargement des résultats
            st.markdown("---")
            st.subheader("Exporter les résultats")
            
            # Créer un DataFrame avec tous les résultats
            results_df = pd.DataFrame({
                'Asset': tickers,
                'Weight': poids_opt,
                'Weight_%': poids_opt * 100,
                'Annual_Return_%': rendements_moyens.values * 100,
                'Risk_Contribution_%': (poids_opt * np.diag(matrice_cov.values) / volatilite**2) * 100
            })
            
            # Métriques globales
            summary_metrics = pd.DataFrame({
                'Metric': ['Sharpe Ratio', 'Annual Return %', 'Annual Volatility %', 
                          f'VaR 1d ({confidence_level*100:.0f}%)', f'TVaR 1d ({confidence_level*100:.0f}%)',
                          'Number of Assets', 'Investment Period (years)'],
                'Value': [sharpe_opt, rendement*100, volatilite*100, 
                         var_1d*100, tvar_1d*100, len(tickers), years]
            })
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label=" Télécharger les poids",
                    data=results_df.to_csv(index=False),
                    file_name="portfolio_weights.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.download_button(
                    label="Télécharger les métriques",
                    data=summary_metrics.to_csv(index=False),
                    file_name="portfolio_metrics.csv",
                    mime="text/csv"
                )
            
            with col3:
                # Générer un rapport
                if st.button(" Générer un rapport complet"):
                    with st.expander(" Rapport d'optimisation"):
                        st.markdown(f"""
                        ## Rapport d'Optimisation de Portefeuille
                        
                        **Date:** {datetime.today().strftime('%Y-%m-%d')}
                        
                        ### Paramètres
                        - Actifs analysés: {', '.join(tickers)}
                        - Période historique: {years} ans
                        - Taux sans risque: {risk_free_rate*100:.2f}%
                        - Niveau de confiance VaR: {confidence_level*100:.0f}%
                        
                        ### Résultats de l'optimisation
                        - **Ratio de Sharpe:** {sharpe_opt:.4f}
                        - **Rendement attendu annuel:** {rendement*100:.2f}%
                        - **Volatilité annuelle:** {volatilite*100:.2f}%
                        - **VaR ({confidence_level*100:.0f}%) 1 jour:** {var_1d*100:.2f}%
                        - **TVaR ({confidence_level*100:.0f}%) 1 jour:** {tvar_1d*100:.2f}%
                        
                        ### Allocation optimale
                        """)
                        
                        st.dataframe(results_df.style.format({
                            'Weight': '{:.4f}',
                            'Weight_%': '{:.2f}%',
                            'Annual_Return_%': '{:.2f}%',
                            'Risk_Contribution_%': '{:.2f}%'
                        }))
            
    except Exception as e:
        st.error(f"Une erreur s'est produite: {str(e)}")
        st.info("Veuillez vérifier les symboles des actifs et réessayer.")
else:
    # Page d'accueil
    if not st.session_state.get('run_optimization', False):
        st.markdown("""
        ## Bienvenue dans l'optimiseur de portefeuille avec Deep Learning!
        
        ###  Fonctionnalités:
        1. **Optimisation par Deep Learning** - Utilise un réseau neuronal pour trouver la meilleure allocation
        2. **Analyse de risque** - Calcul de la VaR (Value at Risk) et TVaR (Conditional VaR)
        3. **Frontière efficiente** - Visualisation interactive des portefeuilles optimaux
        4. **Gestion multi-actifs** - Analysez jusqu'à 20 actifs simultanément
        
        ###  Comment utiliser:
        1. Dans la barre latérale, entrez les symboles des actifs (ex: AAPL, TSLA, GOOGL)
        2. Ajustez les paramètres selon vos besoins
        3. Cliquez sur "🚀 Lancer l'optimisation"
        
        ###  Exemples de symboles:
        - Actions: AAPL, MSFT, GOOGL, AMZN, TSLA
        - ETFs: SPY, QQQ, VTI, BND, GLD
        - Crypto: BTC-USD, ETH-USD (si disponible)
        
        **Note:** Les données sont récupérées via Yahoo Finance.
        """)
        
        # Exemples rapides
        st.subheader(" Exemples rapides")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Portefeuille actions US", use_container_width=True):
                st.session_state.run_optimization = True
                # Cette partie nécessiterait une mise à jour de l'état
        
        with col2:
            if st.button("Portefeuille équilibré", use_container_width=True):
                st.session_state.run_optimization = True
        
        with col3:
            if st.button("Portefeuille diversifié", use_container_width=True):
                st.session_state.run_optimization = True

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Application développée avec Streamlit • Données financières de Yahoo Finance</p>
    <p> <strong>Avertissement:</strong> Cet outil est à des fins éducatives seulement. 
    Les investissements comportent des risques.</p>
</div>
""", unsafe_allow_html=True)