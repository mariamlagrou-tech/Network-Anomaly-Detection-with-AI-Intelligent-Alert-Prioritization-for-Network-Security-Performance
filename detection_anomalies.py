import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)

sns.set(style="whitegrid")

# ============================================================
# CONFIGURATION
# ============================================================
FICHIER = "dataset_network.csv"   # chemin de votre fichier CSV
CONTAMINATION = 0.10              # proportion d'anomalies estimée (à ajuster)

# ============================================================
# ATTRIBUTS IMPORTANTS POUR LA DÉTECTION D'ANOMALIES
# ============================================================
FEATURES_IMPORTANTS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "SYN Flag Count",
    "FIN Flag Count",
    "ACK Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Active Mean",
    "Idle Mean"
]

print("=" * 60)
print("DÉTECTION D'ANOMALIES RÉSEAU - Isolation Forest")
print("=" * 60)
print("\nAttributs sélectionnés :", len(FEATURES_IMPORTANTS))
print(FEATURES_IMPORTANTS)

# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================
if os.path.exists(FICHIER):
    df = pd.read_csv(FICHIER, na_values=["-", "", "NaN"])
    print(f"\nDonnées réelles chargées : {FICHIER}")
else:
    print("\nFichier introuvable -> génération de données de démonstration.")
    rng = np.random.default_rng(42)
    blocs = []
    
    # Données normales (8000)
    n_normal = 8000
    blocs.append(pd.DataFrame({
        "Flow Duration": np.abs(rng.normal(100, 30, n_normal)).round(2),
        "Total Fwd Packets": rng.integers(10, 100, n_normal),
        "Total Backward Packets": rng.integers(10, 100, n_normal),
        "Total Length of Fwd Packets": rng.integers(1000, 10000, n_normal),
        "Total Length of Bwd Packets": rng.integers(1000, 10000, n_normal),
        "Flow Bytes/s": np.abs(rng.normal(5000, 1500, n_normal)).round(2),
        "Flow Packets/s": np.abs(rng.normal(50, 15, n_normal)).round(2),
        "Fwd Packet Length Mean": np.abs(rng.normal(500, 100, n_normal)).round(2),
        "Bwd Packet Length Mean": np.abs(rng.normal(500, 100, n_normal)).round(2),
        "Min Packet Length": rng.integers(20, 60, n_normal),
        "Max Packet Length": rng.integers(1000, 1400, n_normal),
        "Packet Length Mean": np.abs(rng.normal(500, 100, n_normal)).round(2),
        "SYN Flag Count": rng.integers(0, 3, n_normal),
        "FIN Flag Count": rng.integers(0, 2, n_normal),
        "ACK Flag Count": rng.integers(5, 50, n_normal),
        "Down/Up Ratio": np.abs(rng.normal(1.0, 0.3, n_normal)).round(2),
        "Average Packet Size": np.abs(rng.normal(500, 100, n_normal)).round(2),
        "Active Mean": np.abs(rng.normal(50, 20, n_normal)).round(2),
        "Idle Mean": np.abs(rng.normal(100, 30, n_normal)).round(2),
        "Label": np.full(n_normal, 0),
    }))
    
    # Données d'attaque (2000) - DDoS/SYN flood
    n_attack = 2000
    blocs.append(pd.DataFrame({
        "Flow Duration": np.abs(rng.normal(500, 150, n_attack)).round(2),
        "Total Fwd Packets": rng.integers(500, 5000, n_attack),
        "Total Backward Packets": rng.integers(10, 100, n_attack),
        "Total Length of Fwd Packets": rng.integers(50000, 500000, n_attack),
        "Total Length of Bwd Packets": rng.integers(1000, 5000, n_attack),
        "Flow Bytes/s": np.abs(rng.normal(50000, 15000, n_attack)).round(2),
        "Flow Packets/s": np.abs(rng.normal(500, 150, n_attack)).round(2),
        "Fwd Packet Length Mean": np.abs(rng.normal(100, 30, n_attack)).round(2),
        "Bwd Packet Length Mean": np.abs(rng.normal(500, 100, n_attack)).round(2),
        "Min Packet Length": rng.integers(20, 40, n_attack),
        "Max Packet Length": rng.integers(40, 100, n_attack),
        "Packet Length Mean": np.abs(rng.normal(100, 30, n_attack)).round(2),
        "SYN Flag Count": rng.integers(100, 1000, n_attack),
        "FIN Flag Count": rng.integers(0, 1, n_attack),
        "ACK Flag Count": rng.integers(0, 5, n_attack),
        "Down/Up Ratio": np.abs(rng.normal(0.1, 0.05, n_attack)).round(2),
        "Average Packet Size": np.abs(rng.normal(100, 30, n_attack)).round(2),
        "Active Mean": np.abs(rng.normal(200, 50, n_attack)).round(2),
        "Idle Mean": np.abs(rng.normal(10, 5, n_attack)).round(2),
        "Label": np.full(n_attack, 1),
    }))
    
    df = pd.concat(blocs, ignore_index=True).sample(frac=1, random_state=1).reset_index(drop=True)

df.columns = df.columns.str.strip()
features = [c for c in FEATURES_IMPORTANTS if c in df.columns]
print(f"\nVariables utilisées : {len(features)}")
print(features)

# ============================================================
# 2. PRÉPARATION DES DONNÉES
# ============================================================
X = df[features].apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())

print(f"\nDimensions : {df.shape}")
print("Données manquantes traitées par médiane")

# ============================================================
# 3. STANDARDISATION
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 4. ISOLATION FOREST
# ============================================================
print("\nEntraînement du modèle Isolation Forest...")
modele = IsolationForest(n_estimators=100, contamination=CONTAMINATION, random_state=42)
modele.fit(X_scaled)

df["anomalie"] = np.where(modele.predict(X_scaled) == -1, 1, 0)
df["score"] = -modele.decision_function(X_scaled)

print(f"Anomalies détectées : {int(df['anomalie'].sum())} / {len(df)}")

# ============================================================
# 5. PRIORISATION DES ALERTES
# ============================================================
score_norm = (df["score"] - df["score"].min()) / (df["score"].max() - df["score"].min())
df["priorite"] = np.where(score_norm >= 0.7, "Haute",
                  np.where(score_norm >= 0.4, "Moyenne", "Basse"))

print("\nRépartition des priorités :")
print(df["priorite"].value_counts())

# ============================================================
# 6. ÉVALUATION (si "Label" existe)
# ============================================================
if "Label" in df.columns:
    y_vrai = pd.to_numeric(df["Label"], errors="coerce").fillna(0).astype(int)
    
    print("\n" + "=" * 60)
    print("ÉVALUATION DU MODÈLE")
    print("=" * 60)
    print(classification_report(y_vrai, df["anomalie"], 
                                target_names=["Normal", "Attaque"]))
    print(f"ROC-AUC : {round(roc_auc_score(y_vrai, df['score']), 3)}")

    mc = confusion_matrix(y_vrai, df["anomalie"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(mc, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Attaque"], 
                yticklabels=["Normal", "Attaque"])
    plt.title("Matrice de confusion")
    plt.xlabel("Prédiction")
    plt.ylabel("Réalité")
    plt.tight_layout()
    plt.savefig("01_matrice_confusion.png", dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(y_vrai, df["score"])
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color="red", 
             label=f"AUC = {roc_auc_score(y_vrai, df['score']):.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Faux positifs")
    plt.ylabel("Vrais positifs")
    plt.title("Courbe ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig("02_courbe_roc.png", dpi=150)
    plt.close()

# ============================================================
# 7. GRAPHIQUES
# ============================================================
print("\nGénération des graphiques...")

plt.figure(figsize=(7, 5))
couleurs = df["anomalie"].map({0: "blue", 1: "red"})
plt.scatter(X["Total Fwd Packets"], X["Flow Bytes/s"], 
            c=couleurs, s=10, alpha=0.3)
plt.xlabel("Total Fwd Packets")
plt.ylabel("Flow Bytes/s")
plt.title("Détection d'anomalies (rouge = anomalie)")
plt.tight_layout()
plt.savefig("03_anomalies.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 4))
plt.hist(df["score"], bins=50, color="#2980b9")
plt.xlabel("Score d'anomalie")
plt.ylabel("Nombre de connexions")
plt.title("Distribution des scores d'anomalie")
plt.tight_layout()
plt.savefig("04_scores.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
df["priorite"].value_counts().reindex(["Basse", "Moyenne", "Haute"]).plot(
    kind="bar", color=["#27ae60", "#f39c12", "#c0392b"])
plt.title("Répartition des niveaux de priorité")
plt.ylabel("Nombre de connexions")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("05_priorites.png", dpi=150)
plt.close()

# ============================================================
# 8. SAUVEGARDE
# ============================================================
print("\nSauvegarde du modèle et des alertes...")

joblib.dump({
    "scaler": scaler, 
    "modele": modele, 
    "features": features
}, "modele_isolation_forest_network.joblib")

alertes = df[df["anomalie"] == 1].sort_values("score", ascending=False)
alertes.to_csv("alertes_priorisees_network.csv", index=False)

print("\n" + "=" * 60)
print("TERMINÉ")
print("=" * 60)
print("\nFichiers générés:")
print("  - alertes_priorisees_network.csv")
print("  - modele_isolation_forest_network.joblib")
print("  - 01_matrice_confusion.png")
print("  - 02_courbe_roc.png")
print("  - 03_anomalies.png")
print("  - 04_scores.png")
print("  - 05_priorites.png")
