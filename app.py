import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from fpdf import FPDF
import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error

# --- CONFIGURATION ---
st.set_page_config(page_title="Supply Chain AI Predictor", layout="wide", initial_sidebar_state="expanded")

# --- STYLE CSS MODERNE ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    [data-testid="stSidebar"] hr {
        background-color: rgba(255,255,255,0.1) !important;
        margin: 1.5rem 0;
    }

    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stFileUploader > div > div {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px;
    }
    
    /* CARTES MÉTRIQUES */
    [data-testid="stMetric"] {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.05);
        transition: all 0.2s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(59, 130, 246, 0.1);
        border-color: #3b82f6;
    }
    
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 800;
        font-size: 1.8rem;
    }

    /* BOUTON */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.3);
    }

    /* BADGES & LAYOUT */
    .abc-container {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 20px;
        padding: 15px;
        background: #f8fafc;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }

    .class-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.4rem 1.2rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1rem;
    }
    
    .class-a { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
    .class-b { background: #fef9c3; color: #a16207; border: 1px solid #fde047; }
    .class-c { background: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }

</style>
""", unsafe_allow_html=True)

# --- FONCTIONS ---

def smart_column_detection(columns, keywords, exclude_keywords=None):
    if exclude_keywords is None: exclude_keywords = []
    col_lower = [str(c).lower() for c in columns]
    for key in keywords:
        for i, col in enumerate(col_lower):
            if key in col:
                if not any(bad in col for bad in exclude_keywords):
                    return columns[i]
    return None

def calculate_abc_classification(df, col_item, col_sales):
    abc_df = df.copy()
    abc_df[col_sales] = pd.to_numeric(abc_df[col_sales], errors='coerce')
    abc_df = abc_df.dropna(subset=[col_sales])
    abc_summary = abc_df.groupby(col_item)[col_sales].sum().reset_index()
    abc_summary = abc_summary.sort_values(by=col_sales, ascending=False)
    total = abc_summary[col_sales].sum()
    if total == 0: return {}
    abc_summary['cum_pct'] = (abc_summary[col_sales] / total).cumsum()
    abc_summary['class'] = abc_summary['cum_pct'].apply(lambda x: 'A' if x <= 0.8 else ('B' if x <= 0.95 else 'C'))
    return abc_summary.set_index(col_item)['class'].to_dict()

# --- NOUVELLE FONCTION PDF PRO ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.set_text_color(44, 62, 80)
        self.cell(0, 10, 'SUPPLY CHAIN AI - Rapport de Prevision', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()} - Genere par Supply Chain AI', 0, 0, 'C')

def create_pdf(item, forecast, safety, cmd, service, abc, score, mae):
    pdf = PDF()
    pdf.add_page()
    
    # Date du rapport
    today = datetime.datetime.now().strftime("%d/%m/%Y")
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, f"Date du rapport : {today}", 0, 1, 'R')
    pdf.ln(5)

    # Section 1: Info Produit
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, f"  PRODUIT : {item}", 1, 1, 'L', 1)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"  Classe ABC : {abc}", 0, 1)
    pdf.cell(0, 8, f"  Niveau de Service Cible : {service*100}%", 0, 1)
    pdf.ln(5)

    # Section 2: Analyse IA
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "  ANALYSE INTELLIGENTE", 1, 1, 'L', 1)
    pdf.set_font("Arial", '', 11)
    
    confiance = "Elevee" if score > 70 else ("Moyenne" if score > 40 else "Faible (Nouveau Produit)")
    pdf.cell(0, 8, f"  Score de Confiance IA : {int(score)}/100 ({confiance})", 0, 1)
    
    if mae > 0:
        pdf.cell(0, 8, f"  Marge d'erreur moyenne : {int(mae)} unites/jour", 0, 1)
    else:
        pdf.cell(0, 8, f"  Marge d'erreur : N/A (Historique insuffisant)", 0, 1)
    pdf.ln(5)

    # Section 3: Recommandation
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "  PLAN D'APPROVISIONNEMENT (30 Jours)", 1, 1, 'L', 1)
    pdf.ln(5)
    
    pdf.set_font("Arial", '', 11)
    pdf.cell(90, 10, "Prevision de la demande :", 0, 0)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, f"{int(forecast)} unites", 0, 1)
    
    pdf.set_font("Arial", '', 11)
    pdf.cell(90, 10, "Stock de Securite :", 0, 0)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, f"{int(safety)} unites", 0, 1)
    
    pdf.ln(5)
    pdf.set_fill_color(220, 255, 220) # Vert clair
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 15, f"  COMMANDE RECOMMANDEE : {int(cmd)} unites", 1, 1, 'C', 1)

    return pdf.output(dest='S').encode('latin-1')

# --- INTERFACE (TITRE AMÉLIORÉ) ---

# Titre centré avec dégradé CSS
st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 style='background: linear-gradient(to right, #2563eb, #9333ea); 
                   -webkit-background-clip: text; 
                   -webkit-text-fill-color: transparent; 
                   font-size: 3rem; 
                   font-weight: 800; 
                   margin-bottom: 0;'>
            Supply Chain AI Predictor🚀
        </h1>
        <p style='color: #64748b; font-size: 1.2rem; font-weight: 500;'>
            Pilotez vos stocks avec la puissance du Machine Learning
        </p>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    uploaded_file = st.file_uploader("Importer CSV", type=['csv'])
    
    if uploaded_file:
        st.divider()
        st.markdown("### 🛡️ Gestion des risques")
        service_level = st.select_slider("Niveau de service cible", options=[0.80, 0.90, 0.95, 0.99], value=0.95)
        descriptions = {0.80: "🚀 Économique (Stock bas)", 0.90: "📊 Standard (Équilibré)", 0.95: "🛡️ Sécurisé (Recommandé)", 0.99: "💎 Premium (Zéro rupture)"}
        st.info(f"{descriptions[service_level]}")
        st.caption(f"Risque de rupture accepté : **{int((1-service_level)*100)}%**")

# --- TRAITEMENT ---
if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file, sep=None, engine='python')
    except Exception as e:
        st.error(f"Erreur lecture : {e}")
        st.stop()

    all_cols = df_raw.columns.tolist()
    
    # --- MAPPING ---
    guess_date = smart_column_detection(all_cols, ['date', 'time', 'jour', 'day'], []) or all_cols[0]
    idx_date = all_cols.index(guess_date)
    
    nums = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    if not nums: st.error("Pas de colonne numérique."); st.stop()
        
    guess_sales = smart_column_detection(
        nums, 
        keywords=['sold', 'turnover', 'revenue', 'amount', 'total', 'sale', 'vent', 'qty', 'quantity', 'qte'], 
        exclude_keywords=['id', 'transaction', 'price', 'prix', 'code', 'discount']
    )
    idx_sales = nums.index(guess_sales) if guess_sales else 0
    
    guess_item = smart_column_detection(
        all_cols, 
        keywords=['product', 'produit', 'name', 'nom', 'item', 'art', 'sku', 'ref', 'id'], 
        exclude_keywords=['transaction', 'order', 'row', 'date', 'payment', 'ship', 'region']
    )
    idx_item = all_cols.index(guess_item) if guess_item else 0

    with st.sidebar:
        st.divider()
        st.markdown("### 🗺️ Mapping")
        col_date = st.selectbox("Date", all_cols, index=idx_date)
        col_sales = st.selectbox("Ventes", nums, index=idx_sales)
        col_item = st.selectbox("Produit", all_cols, index=idx_item)
    
    # Conversion Date
    try:
        df_raw[col_date] = pd.to_datetime(df_raw[col_date], errors='coerce', dayfirst=True)
        if df_raw[col_date].isna().mean() > 0.30:
            uploaded_file.seek(0)
            temp = pd.read_csv(uploaded_file, sep=None, engine='python')
            df_raw[col_date] = pd.to_datetime(temp[col_date], errors='coerce', format='mixed', dayfirst=True)
        df_raw = df_raw.dropna(subset=[col_date])
        if df_raw.empty: st.error("Dates illisibles."); st.stop()
    except Exception as e:
        st.error(f"Erreur date : {e}"); st.stop()

    # --- MAIN ---
    st.markdown("---")
    items = df_raw[col_item].unique()
    selected_item = st.selectbox("📦 Produit à analyser", items)
    
    df_filtered = df_raw[df_raw[col_item] == selected_item].copy()
    data = df_filtered.set_index(col_date)[col_sales].resample('D').sum().fillna(0).reset_index()
    data = data.sort_values(col_date)

    abc_dict = calculate_abc_classification(df_raw, col_item, col_sales)
    curr_class = abc_dict.get(selected_item, 'C')
    class_css = f"class-{curr_class.lower()}"

    st.markdown(f"""
        <div class="abc-container">
            <div style="flex-grow: 1;">
                <h3 style="margin:0; padding:0; color:#1e293b;">Historique : {selected_item}</h3>
                <p style="margin:0; font-size:0.9rem; color:#64748b;">Analyse des tendances passées</p>
            </div>
            <div class="class-badge {class_css}">
                Classe {curr_class}
            </div>
        </div>
    """, unsafe_allow_html=True)

    data['Moyenne_7j'] = data[col_sales].rolling(window=7).mean()
    fig_hist = px.line(data, x=col_date, y=col_sales, labels={col_sales: "Ventes", col_date: "Date"})
    fig_hist.add_scatter(x=data[col_date], y=data['Moyenne_7j'], name='Moyenne 7j', line=dict(color='#3b82f6', width=3))
    fig_hist.update_layout(template="plotly_white", margin=dict(l=0,r=0,t=10,b=0), height=350, showlegend=True, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("###")
    if st.button("🚀 Lancer l'IA de Prévision", type="primary"):
        progress = st.progress(0)
        
        # Prophet
        df_p = data.rename(columns={col_date: 'ds', col_sales: 'y'})
        progress.progress(20)
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        m.fit(df_p)
        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)
        forecast['yhat'] = forecast['yhat'].clip(lower=0)
        progress.progress(100)
        progress.empty()

        # Fiabilité
        perf = forecast.set_index('ds')[['yhat']].join(df_p.set_index('ds'), how='inner').dropna()
        has_data = len(perf) > 10
        if has_data:
            mae = mean_absolute_error(perf['y'], perf['yhat'])
            avg = perf['y'].mean() if perf['y'].mean() > 0 else 1
            score = max(0, 100 - (mae / avg * 100))
        else:
            mae = 0
            score = 0

        # Audit
        st.markdown("### 🕵️‍♂️ Audit de Confiance de l'IA")
        col_gauge, col_text = st.columns([1, 2])
        with col_gauge:
            if has_data:
                st.metric("Score de Confiance", f"{int(score)}/100")
                st.progress(int(score) / 100)
            else:
                st.info("ℹ️ Mode Nouveau Produit")
        
        with col_text:
            if has_data:
                if score > 80: st.success("✅ **Excellent :** Le modèle est très fiable.")
                elif score > 60: st.warning("⚠️ **Moyen :** Ventes volatiles, prudence.")
                else: st.error("🛑 **Faible :** Ventes très irrégulières.")
                st.caption(f"Marge d'erreur moyenne : **{int(mae)} unités** par jour.")
            else:
                st.info("Pas assez d'historique pour noter l'IA. Marge de sécurité appliquée par défaut.")

        # Calculs
        z = {0.80:0.84, 0.90:1.28, 0.95:1.65, 0.99:2.33}[service_level]
        rmse = np.sqrt(((perf['y']-perf['yhat'])**2).mean()) if has_data else 0
        pred_30 = forecast.iloc[-30:]['yhat'].sum()
        safety = z * rmse if has_data else pred_30 * 0.5
        cmd = pred_30 + safety

        # KPIs
        st.divider()
        st.markdown("### 📦 Plan d'Approvisionnement")
        k1, k2, k3 = st.columns(3)
        k1.metric("Besoin estimé (30j)", f"{int(pred_30)} u")
        k2.metric("Stock Sécurité", f"{int(safety)} u", help="Assurance rupture")
        k3.metric("COMMANDE À PASSER", f"{int(cmd)} u", delta="Prioritaire")

        # Graphique Futur
        st.markdown("#### 📅 Projection : Les 30 prochains jours")
        fig_fut = px.line(forecast.tail(60), x='ds', y='yhat', labels={'ds':'Date', 'yhat':'Demande Prévue'})
        fig_fut.add_hline(y=safety, line_dash="dot", line_color="#ef4444", annotation_text="Niveau de Sécurité")
        fig_fut.update_layout(template="plotly_white", margin=dict(l=0,r=0,t=30,b=0), height=400)
        st.plotly_chart(fig_fut, use_container_width=True)

        # PDF avec arguments
        pdf = create_pdf(selected_item, pred_30, safety, cmd, service_level, curr_class, score, mae)
        st.download_button("📄 Télécharger le Bon de Commande", pdf, f"Rapport_{selected_item}.pdf", use_container_width=True)

else:
    st.info("👋 Veuillez importer un fichier CSV dans la barre latérale.")