# 📦 Supply Chain AI Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![AI Model](https://img.shields.io/badge/Model-Prophet-orange)](https://facebook.github.io/prophet/)

**Supply Chain AI Predictor** is an open-source Data Science solution designed to democratize demand forecasting. It allows Supply Chain Managers to move away from intuition (or Excel) towards an AI-driven approach.

👉 **[Test the App Online (Live Demo)](https://supply-chain-predictor-jwdccg982ctiqzi4afhyjp.streamlit.app/)**

---

## 🎯 Project Objectives

Inventory management is a precarious balance: too much stock is expensive (Working Capital), while too little leads to lost sales.
This project aims to:
1.  **Automate** sales trend analysis.
2.  **Secure** replenishment via statistical safety stock calculation.
3.  **Facilitate** decision-making with a ready-to-use PDF report.

---

## 🧠 Under the Hood: The AI

The application uses **Facebook Prophet**, an additive time-series model.

* **Why this choice?** Unlike classic moving averages, Prophet decomposes the signal to identify:
    * The underlying trend (growth/decline).
    * Weekly seasonality (weekend peaks).
    * Yearly seasonality (Sales, Christmas, Black Friday).
* **Confidence Audit:** The AI doesn't just predict. It compares its past predictions with reality to assign itself a **Reliability Score (0-100%)**. If the score is low, the algorithm automatically recommends a higher safety stock.

---

## ✨ Key Features

* **📂 Universal & Smart Import:** The mapping algorithm automatically detects columns (Date, Quantity/Amount, Product) regardless of your CSV format (Amazon, Internal ERP, etc.).
* **🎮 Integrated Demo Mode:** No data on hand? Activate demo mode to test the tool with a real included dataset.
* **📊 ABC Analysis:** Automatic product segmentation based on the Pareto principle (the 20% of products generating 80% of revenue).
* **🛡️ Risk Management:** Dynamic safety stock adjustment based on the target service level (from 80% to 99.9%).
* **📑 Automated Reporting:** Generation of a PDF Purchase Order including key metrics and the AI's decision.

---

## 💾 Expected Data

The application accepts any **CSV** file (`.csv`).
The detection algorithm looks for:
1.  **A Time Column:** (Date format detected automatically).
2.  **A Metric Column:** (Units sold, Revenue, Quantity...).
3.  **An Identifier Column:** (Product Name, SKU, ID...).

*Note: The separator (comma or semicolon) is detected automatically.*

---

## 🚧 Current Limitations & Roadmap

This project is constantly evolving. Here are the identified areas for improvement:

* **Current Scope:** Single-product forecasting (one product at a time).
    * *Planned Improvement:* Global dashboard to visualize the entire catalog at once.
* **External Factors:** The model is based solely on history.
    * *Planned Improvement:* Integration of exogenous variables (weather, marketing budget, promotions) via an XGBoost model.
* **Data Source:** Local CSV file processing.
    * *Planned Improvement:* Direct connection to a SQL database or API (Shopify/WooCommerce).

---

## 💻 Local Installation

To run the project on your machine:

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/supply-chain-predictor.git](https://github.com/YOUR_USERNAME/supply-chain-predictor.git)
    cd supply-chain-predictor
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Launch the application:
    ```bash
    streamlit run app.py
    ```

---

## 👤 Author

**Younes Ferhat**
* [My LinkedIn](https://www.linkedin.com/in/younes-ferhat)

---
