# 🌍 AI Global Economy Simulator

A **production-quality, research-grade economic simulation platform** that models a synthetic world economy where countries interact through trade, economic growth, inflation dynamics, and energy production. Machine learning models forecast future trends; an interactive Streamlit dashboard visualises everything in real time.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Synthetic World** | 30+ countries with realistic GDP, inflation, trade, energy, and stability attributes |
| **Multi-round Simulation** | Configurable economic engine evolving all indicators across multiple time steps |
| **ML Forecasting** | GDP growth, inflation prediction, and recession risk models (Random Forest, Gradient Boosting) |
| **Trade Network** | NetworkX-powered directed trade graph with PageRank-based economic influence scoring |
| **Interactive Dashboard** | Six-tab Streamlit dashboard: Overview · Country Analysis · Forecast · Trade Network · World Map · Raw Data |
| **World Map** | Plotly geo-scatter map coloured by any economic metric |
| **Sidebar Controls** | Real-time parameter tuning (energy price, tech growth, trade openness, inflation pressure, rounds) |

---

## 🏗️ Architecture

```
global-economy-ai/
├── app.py                        # Streamlit entry-point
├── config.py                     # Global constants & defaults
│
├── data/
│   ├── country_generator.py      # Synthetic country generation
│   └── economic_dataset.py       # Dataset builder & ML feature prep
│
├── models/
│   ├── gdp_forecast_model.py     # GDP growth Random Forest regressor
│   ├── inflation_predictor.py    # Inflation Gradient Boosting regressor
│   └── recession_risk_model.py   # Recession probability classifier
│
├── simulation/
│   ├── economy_engine.py         # Multi-round simulation engine
│   ├── trade_network.py          # NetworkX trade graph
│   └── resource_dynamics.py      # Energy resource modelling
│
├── analytics/
│   ├── global_metrics.py         # Round-level world aggregates
│   └── economic_analysis.py      # Country ranking, health scores, forecasts
│
├── ui/
│   ├── dashboard.py              # Section renderers
│   ├── charts.py                 # Plotly chart builders
│   ├── world_map.py              # Geo-scatter world map
│   ├── trade_network_viz.py      # NetworkX → Plotly graph visualisation
│   └── controls.py               # Sidebar parameter controls
│
├── utils/
│   └── helpers.py                # Formatting & utility functions
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** – interactive web dashboard
- **Pandas / NumPy** – data manipulation
- **Scikit-learn** – Random Forest & Gradient Boosting models
- **Plotly** – interactive charts, maps, and network graphs
- **NetworkX** – trade network analysis

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/ravigohel142996/AI-Global-Economy-Simulator.git
cd AI-Global-Economy-Simulator
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the app

```bash
streamlit run app.py
```

The dashboard opens automatically at `http://localhost:8501`.

### Streamlit Cloud Deployment

Push the repository to GitHub and deploy directly from [streamlit.io/cloud](https://streamlit.io/cloud) — no additional configuration required.

---

## 📊 Dashboard Sections

| Tab | Content |
|---|---|
| 🌐 **Overview** | World GDP, inflation, trade volume, stability KPIs + global charts |
| 🔍 **Country Analysis** | Per-country KPIs, time-series panels, radar economic profile |
| 🔮 **Forecast** | GDP growth projections, inflation trends, recession risk, ML feature importances |
| 🕸️ **Trade Network** | Interactive force-layout trade graph + influence rankings |
| 🗺️ **World Map** | Geo-scatter bubble map coloured by any metric |
| 📋 **Raw Data** | Sortable data table with health scores |

---

## ⚙️ Simulation Parameters

| Parameter | Range | Description |
|---|---|---|
| Number of Countries | 10–40 | How many synthetic nations to generate |
| Simulation Rounds | 5–30 | Economic time steps |
| Energy Price (USD) | 20–200 | Global energy cost lever |
| Technology Growth | 0–10 % | Annual tech improvement rate |
| Trade Openness | 0–1 | How strongly trade affects GDP |
| Inflation Pressure | 0–1 | External inflationary shock intensity |

---

## 🔭 Future Improvements

- Central bank monetary policy module (interest rate setting)
- Country-specific fiscal policy (government spending, taxation)
- Real historical data calibration (World Bank / IMF datasets)
- Multi-agent reinforcement learning for autonomous country agents
- PDF/CSV export of simulation results
- Time-based animation of world map evolution
