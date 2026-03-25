# 🚗 Japanese Car Demand Elasticity Analysis

> *"How sensitive are U.S. sales of Japanese automakers to changes in financing rates, loan availability, and GDP growth?"*

---

## Overview

An econometric analysis of **price and income elasticity** for five major Japanese automakers — Toyota, Honda, Mazda, Nissan, and Subaru — in the U.S. market from **2008 to 2025**. The project quantifies how each manufacturer's sales volume responds to macroeconomic shifts including consumer price inflation, new car finance rates, loan amounts, and GDP growth.

---

## Tools & Technologies

| Tool | Purpose |
|---|---|
| **Python** (pandas, NumPy, Matplotlib, seaborn) | Data analysis, elasticity calculations, correlation analysis, visualization |
| **Tableau** | Interactive demand elasticity dashboard |

---

## Data Source

- Semi-annual sales data (2008–2025) for Toyota, Honda, Mazda, Nissan, and Subaru
- Economic indicators: CPI, new car finance rate (weighted), new car loan amount, GDP growth rate
- Data file: `Japanese_Car_Sales_Elasticity.csv` (included in repository)

---

## Methodology

### Analysis Layer (`CarSalesAnalysis`)
1. **Descriptive Statistics** — Mean, std deviation, min/max for sales volumes and all economic indicators per manufacturer
2. **Correlation Analysis** — Pearson correlation between each economic variable and total market sales; per-manufacturer sensitivity scoring (High / Moderate / Low)
3. **Price Elasticity** — Compares average sales during high vs. low finance rate periods (split at median rate); calculates % change in sales volume as proxy for price elasticity
4. **Income Elasticity** — Compares average sales during positive vs. negative GDP growth periods
5. **Market Analysis** — Market share breakdown, period-over-period sales table, overall growth rates 2008–2025

### Export Layer (`TableauExporter`)
Exports 8 analysis-ready CSVs to `tableau_data/` for dashboard use:

| File | Contents |
|---|---|
| `01_raw_data.csv` | Cleaned source data |
| `02_sales_long_format.csv` | Melted to long format for line charts |
| `03_market_share.csv` | Market share % by manufacturer |
| `04_elasticity_metrics.csv` | Price & income elasticity per manufacturer |
| `05_correlation_matrix.csv` | Full correlation matrix (long format) |
| `06_growth_rates.csv` | Growth % from first to last period |
| `07_time_series_analysis.csv` | Per-period, per-manufacturer with all economic variables |
| `08_summary_statistics.csv` | Descriptive stats for all variables |

---

## Key Findings

- **Finance rate sensitivity varies significantly by brand** — some manufacturers show high correlation between finance rate changes and sales volume, others are largely rate-insensitive
- **GDP growth is a stronger demand driver for some brands than others** — reflected in differing income elasticity scores
- **Market share is dominated by Toyota**, with Nissan and Honda as the next two largest players; Mazda and Subaru hold smaller but stable shares
- Full quantified elasticity coefficients and manufacturer rankings are in the Tableau dashboard

---

## Visualizations

| Chart | Description |
|---|---|
| Sales Trends | Line chart of all 5 manufacturers over 34 semi-annual periods |
| Market Share | Horizontal bar chart showing manufacturer share of total U.S. sales |
| Price Sensitivity | Correlation-with-finance-rate bar chart by manufacturer |
| Economic Impact | 2×2 scatter grid — sales vs. CPI, finance rate, GDP growth, loan amount |

---

## Repository Contents

```
├── Japanese_Car_Demand_Elasticity_Analysis.py   # Full Python pipeline
├── Japanese_Car_Sales_Elasticity.csv            # Source data
├── Japanese_Car_Demand_Elasticity_Analysis.twbx # Tableau workbook
├── Japanese_Car_Demand_Elasticity_Report.pdf    # Written analysis report
└── tableau_data/                                # Exported CSVs for Tableau
```

---

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn

# 2. Make sure the CSV is in the same folder as the script, then run:
python Japanese_Car_Demand_Elasticity_Analysis.py
```

The script runs the full analysis, generates 4 charts saved to the script directory, and exports 8 CSVs to `tableau_data/`.

---

*Semi-annual U.S. market data covering 2008–2025.*
