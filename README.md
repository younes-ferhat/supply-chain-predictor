# 📦 Supply Chain AI Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![AI Model](https://img.shields.io/badge/Model-Prophet-orange)](https://facebook.github.io/prophet/)

**Supply Chain AI Predictor** est une solution Data Science open-source conçue pour démocratiser la prévision de la demande. Elle permet aux Supply Chain Managers de passer de l'intuition (ou d'Excel) à une approche pilotée par l'Intelligence Artificielle.

👉 **[Tester l'application en ligne (Démo disponible)][(https://supply-chain-predictor-jwdccg982ctiqzi4afhyjp.streamlit.app/)**

---

## 🎯 Objectifs du Projet

La gestion des stocks est un équilibre précaire : trop de stock coûte cher (BFR), pas assez fait perdre des ventes.
Ce projet vise à :
1.  **Automatiser** l'analyse des tendances de ventes.
2.  **Sécuriser** les approvisionnements via un calcul statistique du stock de sécurité.
3.  **Faciliter** la prise de décision avec un rapport PDF prêt à l'emploi.

---

## 🧠 L'Intelligence Artificielle sous le capot

L'application utilise **Facebook Prophet**, un modèle de série temporelle additif.

* **Pourquoi ce choix ?** Contrairement aux moyennes mobiles classiques, Prophet décompose le signal pour identifier :
    * La tendance de fond (croissance/décroissance).
    * La saisonnalité hebdomadaire (pics du week-end).
    * La saisonnalité annuelle (Soldes, Noël, Black Friday).
* **Audit de Confiance :** L'IA ne se contente pas de prédire. Elle compare ses prédictions passées avec la réalité pour s'attribuer un **Score de Fiabilité (0-100%)**. Si le score est bas, l'algorithme recommande automatiquement un stock de sécurité plus élevé.

---

## ✨ Fonctionnalités Clés

* **📂 Importation Universelle & Intelligente :** L'algorithme de mapping détecte seul les colonnes (Date, Quantité/Montant, Produit) peu importe le format de votre CSV (Amazon, ERP interne, etc.).
* **🎮 Mode Démo Intégré :** Pas de données sous la main ? Activez le mode démo pour tester l'outil avec un jeu de données réel inclus.
* **📊 Classification ABC :** Segmentation automatique des produits selon la loi de Pareto (les 20% des produits qui font 80% du CA).
* **🛡️ Gestion des Risques :** Ajustement dynamique du stock de sécurité selon le taux de service cible (de 80% à 99.9%).
* **📑 Reporting Automatisé :** Génération d'un Bon de Commande PDF incluant les métriques clés et la décision de l'IA.

---

## 💾 Données attendues

L'application accepte tout fichier **CSV** (`.csv`).
L'algorithme de détection cherche :
1.  **Une colonne Temporelle :** (Format date détecté automatiquement).
2.  **Une colonne Métrique :** (Unités vendues, Chiffre d'affaires, Quantité...).
3.  **Une colonne Identifiant :** (Nom du produit, SKU, ID...).

*Note : Le séparateur (virgule ou point-virgule) est détecté automatiquement.*

---

## 🚧 Limites actuelles & Roadmap

Ce projet est en constante évolution. Voici les axes d'amélioration identifiés :

* **Scope actuel :** Prévision mono-produit (un produit à la fois).
    * *Amélioration prévue :* Tableau de bord global pour visualiser tout le catalogue d'un coup.
* **Facteurs externes :** Le modèle se base uniquement sur l'historique.
    * *Amélioration prévue :* Intégration de variables exogènes (météo, budget marketing, promotions) via un modèle XGBoost.
* **Données :** Traitement de fichiers CSV locaux.
    * *Amélioration prévue :* Connexion directe à une base de données SQL ou une API (Shopify/WooCommerce).

---

## 💻 Installation Locale

Pour exécuter le projet sur votre machine :

1.  Cloner le dépôt :
    ```bash
    git clone [https://github.com/VOTRE_PSEUDO/supply-chain-predictor.git](https://github.com/VOTRE_PSEUDO/supply-chain-predictor.git)
    cd supply-chain-predictor
    ```

2.  Installer les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

3.  Lancer l'application :
    ```bash
    streamlit run app.py
    ```

---

## 👤 Auteur

**Younes Ferhat**
* [[Mon LinkedIn](VOTRE_LIEN_LINKEDIN)](https://www.linkedin.com/in/younes-ferhat)

---
