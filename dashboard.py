mport os
import io
import base64

import numpy as np
import pandas as pd
import joblib

import plotly.graph_objects as go

from dash import Dash, dcc, html, dash_table, Input, Output, State, callback, no_update

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report

# ============================================================
# PALETTE — Security Operations Console
# ============================================================
INK     = "#0a0e17"
PANEL   = "#10172a"
PANEL_2 = "#16203a"
LINE    = "#232f4d"
TEXT    = "#e9eef9"
MUTED   = "#7e8aa6"
TEAL    = "#43e0c8"
BLUE    = "#5b8def"
HIGH    = "#ff5470"
MED     = "#ffa630"
LOW     = "#2fd98f"
GRID    = "#1b2540"

FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Flow Bytes/s", "Flow Packets/s", "Fwd Packet Length Mean",
    "Bwd Packet Length Mean", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "SYN Flag Count", "FIN Flag Count",
    "ACK Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Active Mean", "Idle Mean"
]

ORDRE_PRIORITE = {"Haute": 1, "Moyenne": 2, "Basse": 3}
COULEUR_PRIORITE = {"Haute": HIGH, "Moyenne": MED, "Basse": LOW}


# ============================================================
# LOGIQUE MÉTIER (identique au modèle d'origine)
# ============================================================
def generer_demo(n_normal=8000, n_att=2000):
    rng = np.random.default_rng(42)
    blocs = []
    for att, n in [(0, n_normal), (1, n_att)]:
        blocs.append(pd.DataFrame({
            "Flow Duration": np.abs(rng.normal(500 if att else 100, 150 if att else 30, n)).round(2),
            "Total Fwd Packets": rng.integers(500 if att else 10, 5000 if att else 100, n),
            "Total Backward Packets": rng.integers(10, 100, n),
            "Total Length of Fwd Packets": rng.integers(50000 if att else 1000, 500000 if att else 10000, n),
            "Total Length of Bwd Packets": rng.integers(1000, 5000 if att else 10000, n),
            "Flow Bytes/s": np.abs(rng.normal(50000 if att else 5000, 15000 if att else 1500, n)).round(2),
            "Flow Packets/s": np.abs(rng.normal(500 if att else 50, 150 if att else 15, n)).round(2),
            "Fwd Packet Length Mean": np.abs(rng.normal(100 if att else 500, 30 if att else 100, n)).round(2),
            "Bwd Packet Length Mean": np.abs(rng.normal(500, 100, n)).round(2),
            "Min Packet Length": rng.integers(20, 40 if att else 60, n),
            "Max Packet Length": rng.integers(40 if att else 1000, 100 if att else 1400, n),
            "Packet Length Mean": np.abs(rng.normal(100 if att else 500, 30 if att else 100, n)).round(2),
            "SYN Flag Count": rng.integers(100 if att else 0, 1000 if att else 3, n),
            "FIN Flag Count": rng.integers(0, 1 if att else 2, n),
            "ACK Flag Count": rng.integers(0, 5 if att else 50, n),
            "Down/Up Ratio": np.abs(rng.normal(0.1 if att else 1.0, 0.05 if att else 0.3, n)).round(2),
            "Average Packet Size": np.abs(rng.normal(100 if att else 500, 30 if att else 100, n)).round(2),
            "Active Mean": np.abs(rng.normal(200 if att else 50, 50 if att else 20, n)).round(2),
            "Idle Mean": np.abs(rng.normal(10 if att else 100, 5 if att else 30, n)).round(2),
            "Label": np.full(n, att),
        }))
    return pd.concat(blocs, ignore_index=True).sample(frac=1, random_state=1).reset_index(drop=True)


# Sous-chaînes de noms de colonnes à NE PAS utiliser comme caractéristiques
MOTS_ID = ["ip", "timestamp", "flow_id", "flowid", "fid", "uid",
           "date", "time", "src_port", "dst_port", "sport", "dport"]
# Noms candidats pour la colonne d'étiquette (vérité terrain)
MOTS_LABEL = ["label", "attack", "class", "category", "target"]
# Valeurs considérées comme « normal / bénin » lors de la binarisation
VAL_NORMAL = {"0", "0.0", "normal", "benign", "background", "none", "no", "false", "legitimate"}


def choisir_features(df):
    """Renvoie la liste des colonnes numériques exploitables.
    1) jeu canonique (démo / CIC-IDS) s'il est présent ;
    2) sinon, auto-détection : colonnes numériques hors identifiants/étiquettes."""
    canon = [c for c in FEATURES if c in df.columns]
    if canon:
        return canon
    feats = []
    for c in df.columns:
        lc = str(c).strip().lower()
        if any(m in lc for m in MOTS_ID):
            continue
        if lc in MOTS_LABEL or any(m in lc for m in MOTS_LABEL):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= 0.5 and s.nunique(dropna=True) > 1:   # numérique et non constante
            feats.append(c)
    return feats


def trouver_label(df):
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in MOTS_LABEL or any(m in lc for m in MOTS_LABEL):
            return c
    return None


def binariser_label(s):
    """0 = trafic normal/bénin, 1 = attaque. Gère le numérique (0/1) et le texte."""
    t = s.astype(str).str.strip().str.lower()
    return (~t.isin(VAL_NORMAL)).astype(int)


def preparer(df):
    df.columns = df.columns.str.strip()
    feats = choisir_features(df)
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)   # inf (ex. Flow Bytes/s) -> NaN
    X = X.fillna(X.median())
    X = X.fillna(0)
    return df, X, feats


def detecter(X, contamination, canonique):
    # Le modèle sauvegardé n'est réutilisé que pour le schéma standard
    if canonique and os.path.exists("modele_isolation_forest_network.joblib"):
        objet = joblib.load("modele_isolation_forest_network.joblib")
        scaler, modele = objet["scaler"], objet["modele"]
        X_scaled = scaler.transform(X)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        modele = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
        modele.fit(X_scaled)
    anomalie = np.where(modele.predict(X_scaled) == -1, 1, 0)
    score = -modele.decision_function(X_scaled)
    return anomalie, score


def traiter(df, contamination):
    df, X, feats = preparer(df)
    if X.shape[1] == 0:
        raise ValueError("Aucune colonne numérique exploitable.")
    canonique = set(feats).issubset(set(FEATURES))
    anomalie, score = detecter(X, contamination, canonique)
    df = df.copy()
    df["anomalie"] = anomalie
    df["score"] = score.round(4)
    sn = (score - score.min()) / (score.max() - score.min() + 1e-12)
    df["priorite"] = np.where(sn >= 0.7, "Haute", np.where(sn >= 0.4, "Moyenne", "Basse"))
    lab = trouver_label(df)
    y_vrai = binariser_label(df[lab]) if lab is not None else None
    return df, X, score, y_vrai, feats


def parse_upload(contents):
    _, b64 = contents.split(",", 1)
    decoded = base64.b64decode(b64)
    texte = decoded.decode("utf-8", errors="replace")
    # sep=None + engine python : détecte automatiquement , ; \t
    return pd.read_csv(io.StringIO(texte), sep=None, engine="python",
                       na_values=["-", "", "NaN"])


# ============================================================
# HABILLAGE DES FIGURES PLOTLY
# ============================================================
def style_fig(fig, h=340, title=None):
    fig.update_layout(
        paper_bgcolor=INK, plot_bgcolor=INK,
        font=dict(color=MUTED, family="Inter", size=12),
        margin=dict(l=12, r=12, t=42 if title else 16, b=12),
        height=h, title=dict(text=title, font=dict(color=TEXT, family="Space Grotesk", size=15), x=0.01),
        legend=dict(font=dict(color=TEXT), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=PANEL_2, font=dict(color=TEXT), bordercolor=LINE),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=LINE, color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=LINE, color=MUTED)
    return fig


def fig_vide(msg):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(color=MUTED, size=13))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return style_fig(fig, h=340)


# ============================================================
# COMPOSANTS UI
# ============================================================
def carte(libelle, valeur, couleur):
    return html.Div(className="carte", style={"--acc": couleur}, children=[
        html.P(libelle, className="libelle"),
        html.P(valeur, className="valeur", style={"color": couleur}),
    ])


def graph(gid):
    return dcc.Graph(id=gid, config={"displayModeBar": False}, className="bloc")


def construire_rapport(y_vrai, pred):
    """Tableau précision / rappel / F1 / support, au style du tableau de bord."""
    if y_vrai is None:
        return html.Div("Colonne « Label » absente — rapport indisponible",
                        style={"padding": "18px", "color": MUTED, "fontSize": "13px"})

    rep = classification_report(y_vrai, pred, labels=[0, 1],
                                target_names=["Normal", "Attaque"],
                                output_dict=True, zero_division=0)

    def td(v, mono=True, color=TEXT, bold=False, align="right"):
        return html.Td(v, style={
            "padding": "9px 16px", "textAlign": align,
            "fontFamily": "JetBrains Mono" if mono else "Inter",
            "color": color, "fontWeight": "700" if bold else "500",
            "fontSize": "12.5px", "borderBottom": f"1px solid {LINE}"})

    def ligne(nom, d, accent=TEXT):
        return html.Tr([
            td(nom, mono=False, color=accent, bold=True, align="left"),
            td(f"{d['precision']:.3f}"),
            td(f"{d['recall']:.3f}"),
            td(f"{d['f1-score']:.3f}", color=accent, bold=True),
            td(f"{int(d['support']):,}", color=MUTED),
        ])

    entete = html.Tr([html.Th(h, style={
        "padding": "10px 16px", "textAlign": "left" if i == 0 else "right",
        "color": MUTED, "fontSize": "11px", "letterSpacing": "1px",
        "textTransform": "uppercase", "fontWeight": "600",
        "borderBottom": f"1px solid {LINE}"})
        for i, h in enumerate(["Classe", "Précision", "Rappel", "F1-score", "Support"])])

    total = int(rep["macro avg"]["support"])
    lignes = [
        ligne("Normal", rep["Normal"], LOW),
        ligne("Attaque", rep["Attaque"], HIGH),
        html.Tr([
            td("Exactitude", mono=False, color=TEAL, bold=True, align="left"),
            td(""), td(""),
            td(f"{rep['accuracy']:.3f}", color=TEAL, bold=True),
            td(f"{total:,}", color=MUTED),
        ]),
        ligne("Moyenne macro", rep["macro avg"], MUTED),
        ligne("Moyenne pondérée", rep["weighted avg"], MUTED),
    ]

    return html.Table([html.Thead(entete), html.Tbody(lignes)],
                      style={"width": "100%", "borderCollapse": "collapse"})


# ============================================================
# APP
# ============================================================
app = Dash(__name__, title="Détection d'Anomalies Réseau")
server = app.server

app.index_string = """
<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap');
  * { box-sizing: border-box; }
  body { margin:0; background:#0a0e17; color:#e9eef9; font-family:'Inter',sans-serif; }
  .layout { display:flex; min-height:100vh; }
  .sidebar { width:300px; flex:0 0 300px; background:#10172a; border-right:1px solid #232f4d;
             padding:22px 20px; }
  .sidebar h3 { font-family:'Space Grotesk'; font-size:14px; letter-spacing:.5px; margin:0 0 16px; }
  .sidebar label { font-size:11px; text-transform:uppercase; letter-spacing:1.2px; color:#7e8aa6;
                   font-weight:600; display:block; margin:18px 0 8px; }
  .main { flex:1; padding:24px 30px; max-width:1500px; }
  .bandeau { position:relative; overflow:hidden; border:1px solid #232f4d; border-radius:16px;
             padding:24px 28px; margin-bottom:22px;
             background: radial-gradient(120% 160% at 0% 0%, rgba(67,224,200,.16), transparent 55%),
                         radial-gradient(120% 160% at 100% 0%, rgba(91,141,239,.16), transparent 55%), #10172a; }
  .bandeau::before { content:""; position:absolute; left:0; top:0; bottom:0; width:4px;
                     background:linear-gradient(180deg,#43e0c8,#5b8def); }
  .eyebrow { font-family:'JetBrains Mono'; font-size:11px; letter-spacing:3px; text-transform:uppercase;
             color:#43e0c8; margin:0 0 8px; }
  .bandeau h1 { font-family:'Space Grotesk'; font-weight:700; font-size:28px; margin:0; }
  .bandeau .sous { color:#7e8aa6; font-size:13.5px; margin:8px 0 0; }
  .point { display:inline-block; width:8px; height:8px; border-radius:50%; background:#2fd98f;
           box-shadow:0 0 0 3px rgba(47,217,143,.18); margin-right:8px; }
  .grille { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:8px; }
  .grille-3 { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }
  .grille-2 { display:grid; grid-template-columns:repeat(2,1fr); gap:16px; }
  .carte { position:relative; overflow:hidden; background:#10172a; border:1px solid #232f4d;
           border-radius:14px; padding:16px 18px 16px 20px; transition:.15s; }
  .carte:hover { transform:translateY(-2px); border-color:#324063; }
  .carte::before { content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background:var(--acc,#43e0c8); }
  .carte .libelle { font-size:11px; letter-spacing:1.4px; text-transform:uppercase; color:#7e8aa6;
                    font-weight:600; margin:0 0 6px; }
  .carte .valeur { font-family:'JetBrains Mono'; font-weight:700; font-size:28px; line-height:1; margin:0; }
  .bloc { background:#10172a; border:1px solid #232f4d; border-radius:14px; padding:6px; margin-top:6px; }
  .titre-bloc { font-family:'Space Grotesk'; font-size:15px; margin:18px 0 2px; }
  .badge { display:inline-block; font-family:'JetBrains Mono'; font-size:12px; font-weight:600;
           padding:5px 12px; border-radius:999px; margin-right:8px; }
  .tele { background:#0a0e17; border:1px dashed #324063; border-radius:12px; padding:18px;
          text-align:center; color:#7e8aa6; font-size:13px; cursor:pointer; }
  .tele:hover { border-color:#43e0c8; color:#e9eef9; }
  .btn-dl { background:linear-gradient(90deg,#43e0c8,#5b8def); color:#0a0e17; font-weight:700;
            border:none; border-radius:10px; padding:11px 18px; cursor:pointer; font-size:13px; margin-top:14px; }
  .btn-dl:hover { filter:brightness(1.08); }
  /* Onglets */
  .tabs-bar .tab { background:transparent !important; border:none !important; color:#7e8aa6 !important;
                   font-weight:600 !important; font-family:'Inter' !important; border-radius:8px !important; }
  .tabs-bar .tab--selected { background:#16203a !important; color:#e9eef9 !important;
                             box-shadow:inset 0 0 0 1px #232f4d; }
  .tabs-bar { background:#10172a; border:1px solid #232f4d !important; border-radius:12px;
              padding:6px; gap:6px; }
  .Select-control, .Select-menu-outer { background:#0a0e17 !important; border-color:#232f4d !important; color:#e9eef9 !important; }
</style></head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>
"""

app.layout = html.Div(className="layout", children=[
    # ----- Barre latérale -----
    html.Div(className="sidebar", children=[
        html.H3("⚙️  Paramètres"),
        html.Label("Source de données"),
        dcc.Upload(id="upload", className="tele", children="📄  Glisser un CSV  ·  ou cliquer"),
        html.Div(id="etat-fichier", style={"fontSize": "12px", "color": MUTED, "marginTop": "10px"}),
        html.Label("Contamination (part d'anomalies)"),
        dcc.Slider(id="contam", min=0.01, max=0.40, step=0.01, value=0.10,
                   marks={0.01: "1%", 0.20: "20%", 0.40: "40%"},
                   tooltip={"placement": "bottom", "always_visible": False}),
    ]),

    # ----- Contenu principal -----
    html.Div(className="main", children=[
        html.Div(className="bandeau", children=[
            html.P("Network Intrusion Monitoring · Isolation Forest", className="eyebrow"),
            html.H1("Détection d'Anomalies Réseau"),
            html.P([html.Span(className="point"),
                    "Projet IA pour la sécurité et la performance des réseaux — ENSA Kénitra"],
                   className="sous"),
        ]),

        html.Div(id="kpis", className="grille"),
        html.Div(style={"height": "10px"}),

        dcc.Tabs(id="tabs", value="t1", parent_className="tabs-bar",
                 className="tabs-bar", children=[
            dcc.Tab(label="📊 Vue d'ensemble", value="t1", className="tab", selected_className="tab--selected",
                    children=[
                        html.Div(style={"height": "14px"}),
                        html.Div(id="cartes-prio", className="grille-3"),
                        html.Div(className="grille-2", children=[
                            html.Div([html.P("Niveaux de priorité", className="titre-bloc"), graph("g-prio")]),
                            html.Div([html.P("Normal vs Anomalie", className="titre-bloc"), graph("g-na")]),
                        ]),
                    ]),
            dcc.Tab(label="📈 Graphiques", value="t2", className="tab", selected_className="tab--selected",
                    children=[
                        html.Div(style={"height": "14px"}),
                        html.Div(className="grille-2", children=[
                            html.Div([html.P("Distribution des scores", className="titre-bloc"), graph("g-hist")]),
                            html.Div([html.P("Nuage de points (rouge = anomalie)", className="titre-bloc"), graph("g-scatter")]),
                            html.Div([html.P("Matrice de confusion", className="titre-bloc"), graph("g-cm")]),
                            html.Div([html.P("Courbe ROC", className="titre-bloc"), graph("g-roc")]),
                        ]),
                        html.P("Rapport de classification", className="titre-bloc"),
                        html.Div(id="rapport", className="bloc",
                                 style={"padding": "8px 12px", "overflowX": "auto"}),
                    ]),
            dcc.Tab(label="🚨 Alertes priorisées", value="t3", className="tab", selected_className="tab--selected",
                    children=[
                        html.Div(style={"height": "14px"}),
                        html.Div(className="grille-2", style={"gridTemplateColumns": "1fr 1fr"}, children=[
                            html.Div([html.Label("Filtrer par priorité",
                                                 style={"fontSize": "11px", "textTransform": "uppercase",
                                                        "letterSpacing": "1.2px", "color": MUTED}),
                                      dcc.Dropdown(id="filtre", multi=True,
                                                   options=[{"label": n, "value": n} for n in ["Haute", "Moyenne", "Basse"]],
                                                   value=["Haute", "Moyenne", "Basse"],
                                                   style={"color": "#111"})]),
                            html.Div([html.Label("Lignes affichées",
                                                 style={"fontSize": "11px", "textTransform": "uppercase",
                                                        "letterSpacing": "1.2px", "color": MUTED}),
                                      dcc.Slider(id="nlignes", min=100, max=5000, step=100, value=1000,
                                                 marks={100: "100", 2500: "2500", 5000: "5000"})]),
                        ]),
                        html.Div(id="badges", style={"margin": "16px 0"}),
                        dash_table.DataTable(
                            id="table",
                            page_size=25,
                            sort_action="native",
                            style_table={"overflowX": "auto", "borderRadius": "12px"},
                            style_header={"backgroundColor": PANEL_2, "color": TEXT, "fontWeight": "600",
                                          "border": f"1px solid {LINE}", "fontFamily": "Inter"},
                            style_cell={"backgroundColor": PANEL, "color": TEXT, "border": f"1px solid {LINE}",
                                        "fontFamily": "JetBrains Mono", "fontSize": "12px", "padding": "8px",
                                        "maxWidth": 160, "textOverflow": "ellipsis", "overflow": "hidden"},
                        ),
                        html.Button("⬇️  Télécharger le CSV priorisé", id="btn-dl", className="btn-dl"),
                        dcc.Download(id="download"),
                    ]),
        ]),
    ]),

    dcc.Store(id="store-alertes"),
])


# ============================================================
# CALLBACK PRINCIPAL : calcul + KPIs + figures
# ============================================================
@callback(
    Output("kpis", "children"), Output("cartes-prio", "children"),
    Output("g-prio", "figure"), Output("g-na", "figure"),
    Output("g-hist", "figure"), Output("g-scatter", "figure"),
    Output("g-cm", "figure"), Output("g-roc", "figure"),
    Output("rapport", "children"),
    Output("store-alertes", "data"), Output("etat-fichier", "children"),
    Input("upload", "contents"), Input("contam", "value"),
    State("upload", "filename"),
)
def calculer(contents, contamination, filename):
    if contents:
        try:
            df = parse_upload(contents)
            df.columns = df.columns.str.strip()
            feats_dispo = choisir_features(df)
            if not feats_dispo:
                apercu = ", ".join(map(str, list(df.columns)[:6]))
                df = generer_demo()
                etat = ("⚠️ Aucune colonne numérique exploitable. "
                        f"Colonnes lues : {apercu}… — démo utilisée.")
            else:
                std = set(feats_dispo).issubset(set(FEATURES))
                etat = (f"✅ {filename} · {len(feats_dispo)} caractéristiques "
                        + ("(schéma standard)" if std else "(auto-détectées)"))
        except Exception as e:
            df, etat = generer_demo(), f"⚠️ Lecture impossible ({e}) — démo utilisée."
    else:
        df, etat = generer_demo(), "📊 Aucun fichier : données de démonstration."

    df, X, score, y_vrai, feats = traiter(df, contamination)

    # ---- KPIs ----
    auc = f"{roc_auc_score(y_vrai, score):.3f}" if y_vrai is not None else "—"
    kpis = [
        carte("Connexions analysées", f"{len(df):,}", TEAL),
        carte("Anomalies détectées", f"{int(df['anomalie'].sum()):,}", HIGH),
        carte("Taux d'anomalies", f"{df['anomalie'].mean():.1%}", MED),
        carte("ROC-AUC", auc, LOW),
    ]
    cartes_prio = [carte(f"Priorité {n}", f"{int((df['priorite'] == n).sum()):,}", COULEUR_PRIORITE[n])
                   for n in ["Haute", "Moyenne", "Basse"]]

    # ---- Bar priorités ----
    vc = df["priorite"].value_counts().reindex(["Basse", "Moyenne", "Haute"]).fillna(0)
    f_prio = go.Figure(go.Bar(x=vc.values, y=vc.index, orientation="h",
                              marker_color=[COULEUR_PRIORITE[i] for i in vc.index],
                              text=[f"{int(v):,}" for v in vc.values], textposition="auto"))
    style_fig(f_prio, h=300)

    # ---- Normal vs anomalie ----
    nv = df["anomalie"].map({0: "Normal", 1: "Anomalie"}).value_counts().reindex(["Normal", "Anomalie"]).fillna(0)
    f_na = go.Figure(go.Bar(x=nv.values, y=nv.index, orientation="h",
                            marker_color=[BLUE, HIGH],
                            text=[f"{int(v):,}" for v in nv.values], textposition="auto"))
    style_fig(f_na, h=300)

    # ---- Histogramme des scores ----
    f_hist = go.Figure(go.Histogram(x=df["score"], nbinsx=50, marker_color=TEAL))
    f_hist.update_xaxes(title="Score d'anomalie"); f_hist.update_yaxes(title="Connexions")
    style_fig(f_hist)

    # ---- Nuage de points (Scattergl pour la performance) ----
    ax_x = feats[0]
    ax_y = feats[1] if len(feats) > 1 else feats[0]
    f_sc = go.Figure()
    for lab, coul, nom in [(0, BLUE, "Normal"), (1, HIGH, "Anomalie")]:
        m = df["anomalie"] == lab
        f_sc.add_trace(go.Scattergl(
            x=X.loc[m, ax_x], y=X.loc[m, ax_y],
            mode="markers", name=nom, marker=dict(color=coul, size=5, opacity=0.45)))
    f_sc.update_xaxes(title=ax_x); f_sc.update_yaxes(title=ax_y)
    style_fig(f_sc)

    # ---- Matrice de confusion / ROC ----
    if y_vrai is not None:
        mc = confusion_matrix(y_vrai, df["anomalie"])
        f_cm = go.Figure(go.Heatmap(
            z=mc, x=["Normal", "Attaque"], y=["Normal", "Attaque"],
            colorscale=[[0, PANEL], [1, TEAL]], showscale=False,
            text=mc, texttemplate="%{text}", textfont=dict(color=TEXT, size=16)))
        f_cm.update_xaxes(title="Prédiction"); f_cm.update_yaxes(title="Réalité", autorange="reversed")
        style_fig(f_cm)

        fpr, tpr, _ = roc_curve(y_vrai, score)
        a = roc_auc_score(y_vrai, score)
        f_roc = go.Figure()
        f_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=TEAL, width=3),
                                   fill="tozeroy", fillcolor="rgba(67,224,200,0.12)",
                                   name=f"AUC = {a:.3f}"))
        f_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                   line=dict(color=MUTED, dash="dash"), showlegend=False))
        f_roc.update_xaxes(title="Faux positifs"); f_roc.update_yaxes(title="Vrais positifs")
        style_fig(f_roc)
    else:
        f_cm = fig_vide("Colonne « Label » absente — matrice indisponible")
        f_roc = fig_vide("Colonne « Label » absente — ROC indisponible")

    # ---- Rapport de classification ----
    rapport = construire_rapport(y_vrai, df["anomalie"])

    # ---- Alertes pour la table (toutes les anomalies, triées) ----
    alertes = df[df["anomalie"] == 1].copy()
    alertes["__rang"] = alertes["priorite"].map(ORDRE_PRIORITE)
    alertes = alertes.sort_values(["__rang", "score"], ascending=[True, False]).drop(columns="__rang")
    store = alertes.to_json(orient="split")

    return kpis, cartes_prio, f_prio, f_na, f_hist, f_sc, f_cm, f_roc, rapport, store, etat


# ============================================================
# TABLE : filtre priorité + nb de lignes
# ============================================================
@callback(
    Output("table", "data"), Output("table", "columns"),
    Output("table", "style_data_conditional"), Output("badges", "children"),
    Input("store-alertes", "data"), Input("filtre", "value"), Input("nlignes", "value"),
)
def maj_table(store, filtre, nlignes):
    if not store:
        return [], [], [], no_update
    alertes = pd.read_json(io.StringIO(store), orient="split")
    filtre = filtre or []

    resume = alertes["priorite"].value_counts().reindex(["Haute", "Moyenne", "Basse"]).fillna(0).astype(int)
    badges = [html.Span(f"{n} · {resume[n]}", className="badge",
                        style={"background": COULEUR_PRIORITE[n] + "22",
                               "color": COULEUR_PRIORITE[n],
                               "border": f"1px solid {COULEUR_PRIORITE[n]}66"})
              for n in ["Haute", "Moyenne", "Basse"]]

    vue = alertes[alertes["priorite"].isin(filtre)].head(nlignes)
    cols = [{"name": c, "id": c} for c in vue.columns]
    style = [{"if": {"filter_query": f'{{priorite}} = "{n}"'},
              "backgroundColor": COULEUR_PRIORITE[n] + "1a"} for n in COULEUR_PRIORITE]
    return vue.to_dict("records"), cols, style, badges


# ============================================================
# TÉLÉCHARGEMENT : toutes les alertes (non plafonnées)
# ============================================================
@callback(
    Output("download", "data"),
    Input("btn-dl", "n_clicks"), State("store-alertes", "data"),
    prevent_initial_call=True,
)
def telecharger(n, store):
    if not store:
        return no_update
    alertes = pd.read_json(io.StringIO(store), orient="split")
    return dcc.send_data_frame(alertes.to_csv, "alertes_priorisees_network.csv", index=False)


if __name__ == "__main__":
    app.run(debug=True)
