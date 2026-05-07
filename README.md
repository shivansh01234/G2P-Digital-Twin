🧬 Gene-to-Plant (G2P) Digital Twin
**Lead Engineer:** Shivansh Sahu  
**Version:** Production v1.0  
**Live Application:** [Insert your Streamlit Cloud URL here]

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)
![Physics Engine](https://img.shields.io/badge/Physics-SciPy_ODE-green.svg)

## 📌 Executive Summary
The **G2P Digital Twin** is a high-throughput industrial biomanufacturing simulator. It bridges the gap between theoretical bioinformatics and factory-floor realities by integrating **machine learning, cellular thermodynamics, industrial transport physics, and dynamic optimization** into a single cohesive architecture. 

By inputting a target amino acid sequence and a host organism, this digital twin predicts the metabolic burden, simulates the physical bioreactor run, and uses AI to generate an hour-by-hour operational trajectory that prevents bioreactor crashes and maximizes final product yield (kg).

---

## 🏗️ The 4-Pillar Architecture

### 1. Bio-AI Layer (Bioinformatics)
* **Engine:** Triple-Branch Stacking Ensemble (Random Forest, XGBoost, Gradient Boosting).
* **Data:** Trained on a synthesized 15,000-record experimental dataset.
* **Function:** Extracts 14 biophysical features (GRAVY, Instability Index, Aliphatic Index) via `Biopython` and predicts the metabolic cellular burden across 5 different industrial microbes.

### 2. Mechanistic Solver (Cellular Physics)
* **Engine:** SciPy `solve_ivp` (LSODA Method).
* **Function:** Converts AI Burden predictions into physical kinetic parameters. Solves coupled Monod-growth differential equations for Fed-Batch kinetics, mapping metabolic strain to growth rate (μ), substrate consumption, and cell death rates.

### 3. Transport Simulator (Industrial Physics)
* **Function:** Simulates real-world bioreactor constraints. Calculates oxygen mass transfer coefficients (kLa), metabolic heat generation, and oxygen supply vs. demand based on tank volume and impeller dynamics.

### 4. Dynamic Trajectory Optimizer
* **Engine:** Stochastic Global Search (Differential Evolution) & Local Gradient Descent (L-BFGS-B).
* **Function:** Simulates thousands of biological trajectories to find the optimal PID control parameters (Agitation RPM, Aeration L/h, and Sugar Feed Rate L/h) required to maximize yield and avoid total oxygen crashes.

---

## 🎛️ Dashboard Features
* **Mission Control:** Input custom amino acid sequences, select from 5 host microbes (e.g., *E. coli*, *Pichia pastoris*), and set factory constraints.
* **Interactive Telemetry:** View 10 dynamic charts mapping biomass, product yield, substrate depletion, and optimal feeding curves.
* **Trajectory Report:** Exportable hour-by-hour control blueprints designed for factory floor execution.
* **Architecture Blueprints:** A dedicated knowledge hub breaking down the ODE math and Stacking Classifier branches.

---

## 🛠️ Tech Stack
* **Frontend:** Streamlit, Matplotlib, Seaborn
* **Bioinformatics:** Biopython
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Mathematics / Physics:** SciPy (ODE solvers, Differential Evolution)

---

## 🚀 Installation & Quick Start

Want to run the Digital Twin locally on your machine?

**1. Clone the repository**
```bash
git clone [https://github.com/YOUR_USERNAME/G2P-Digital-Twin.git](https://github.com/YOUR_USERNAME/G2P-Digital-Twin.git)
cd G2P-Digital-Twin
