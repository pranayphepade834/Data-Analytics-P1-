# Nassau Candy Factory Reallocation & Shipping Optimization Dashboard


---

# Project Overview

The **Nassau Candy Factory Reallocation & Shipping Optimization Dashboard** is an interactive data analytics application developed using **Python, Streamlit, Plotly, Pandas, and Scikit-learn**.

The dashboard helps determine the optimal manufacturing factory for candy products by analyzing:

- Shipping Distance
- Predicted Lead Time
- Profitability
- Operational Risk
- Shipping Method
- Delivery Region

The project combines machine learning predictions with interactive business dashboards to support data-driven supply chain decisions.

---

# Business Problem

Manufacturing the same product across multiple factories can result in different:

- Shipping costs
- Delivery times
- Profit margins
- Operational risks

Selecting the wrong factory increases transportation costs and delivery delays.

This dashboard recommends the most suitable factory by evaluating multiple business factors simultaneously.

---

# Features

## Factory Optimization Simulator

- Compare all manufacturing factories
- Predict lead time
- Compare shipping distance
- Visual performance comparison

## What-If Scenario Analysis

Simulate different manufacturing locations to analyze:

- Lead time changes
- Shipping distance variation
- Profit impact

## Recommendation Dashboard

Automatically recommends the best factory based on:

- Machine learning predictions
- Business constraints
- Shipping optimization

## Risk & Impact Panel

Analyze:

- Operational Risk
- Recommendation Confidence
- Estimated Profit Impact

---

# Dashboard Preview

## 1. Factory Optimization Simulator

![Factory Optimization Simulator](images/dashboard.png)

---

## 2. What-If Scenario Analysis

![What-If Scenario Analysis](images/simulator.png)

---

## 3. Recommendation Dashboard

![Recommendation Dashboard](images/recommendation.png)

---

## 4. Risk & Impact Panel

![Risk & Impact Panel](images/Risk.png)

---

# Machine Learning Workflow

```text
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Machine Learning Model
      │
      ▼
Lead Time Prediction
      │
      ▼
Factory Recommendation
      │
      ▼
Interactive Dashboard
```

---

# Technology Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Machine Learning | Scikit-learn |
| IDE | Visual Studio Code |
| Version Control | Git & GitHub |

---

# Project Structure

```text
Data-Analytics-P1-/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── nassau_candy_cleaned.csv
│
├── images/
│   ├── dashboard.png
│   ├── simulator.png
│   ├── recommendation.png
│   └── Risk.png
│
└── LICENSE
```

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/pranayphepade834/Data-Analytics-P1-.git
```

## Navigate to the Project Folder

```bash
cd Data-Analytics-P1-
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

---

# Business Insights

The dashboard helps organizations:

- Reduce transportation costs
- Improve delivery performance
- Optimize factory utilization
- Increase operational efficiency
- Support strategic manufacturing decisions

---

# Future Enhancements

- Interactive factory map
- Live logistics tracking
- AI-generated recommendations
- Inventory optimization
- Demand forecasting
- Cloud deployment
- PDF report generation

---

# Author

**Pranay Phepade**

Electronics & Telecommunication Engineering

Aspiring Data Analyst | Machine Learning Enthusiast

## Connect

**GitHub:** https://github.com/pranayphepade834

**LinkedIn:** *(Add your LinkedIn profile URL)*

---

# License

This project is licensed under the **MIT License**.
