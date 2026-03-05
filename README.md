# Non-stationary-societal-willingness-to-pay-for-safety-LQI-framework-
Python implementation of a non-stationary formulation of the societal willingness-to-pay (SWTP) and the societal value of a statistical life (SVSL) derived from the Life Quality Index (LQI), including demographic projections and survival modelling.

# README

## Overview

This repository contains the code used to compute the **societal willingness-to-pay (SWTP)** and the **societal value of a statistical life (SVSL)** under **non-stationary socioeconomic and demographic conditions** using the **Life Quality Index (LQI)** framework.

The implementation accompanies the paper:

**Cao (2026) – *Non-stationary risk acceptance in light of climate change*** presented at the **4th International Conference on Natural Hazards & Infrastructre** (ICONHIC2026).

The repository provides an implementation of the analytical framework used in the paper and allows reproduction of the numerical results presented in the manuscript.

The code evaluates how **economic growth, demographic change, and mortality improvements** influence risk-acceptance metrics over time and compares:

* stationary SWTP / SVSL,
* non-stationary SWTP / SVSL, and
* time-averaged SWTP / SVSL over a decision horizon.

---

## Background

Risk acceptance criteria for structural safety are often derived from the **Life Quality Index (LQI)** using the principle of **marginal life-saving costs**. In this framework, the **societal willingness-to-pay (SWTP)** represents the marginal societal expenditure that preserves welfare when mortality risk is reduced.

Most applications of the LQI assume **stationary socioeconomic parameters**, including:

* life expectancy,
* age distribution, and
* economic output.

However, long-lived infrastructure systems operate in **non-stationary environments**, where economic conditions, demographics, and mortality evolve over time.

This repository implements a **time-dependent formulation of SWTP and SVSL**, allowing these quantities to be evaluated under evolving socioeconomic conditions.

---

## Features

The repository includes tools to:

* compute **discounted remaining life expectancy** under stationary and non-stationary mortality,
* evaluate the **demographic constant** used in LQI-based SWTP formulations,
* compute **SWTP and SVSL** under time-dependent socioeconomic parameters,
* evaluate **time-averaged SWTP/SVSL over a decision horizon**, and
* reproduce the figures and tables presented in the manuscript.

The implementation uses population and economic projections from the **Shared Socioeconomic Pathways (SSP)** dataset by **IIASA and the Wittgenstein Center** and the Penn World Table 9.1.

---

## Data sources

The analysis uses publicly available data sources:

* **Wittgenstein Centre / IIASA population projections** (DOI: 10.5281/ZENODO.7767425),
* **Shared Socioeconomic Pathways (SSP) scenarios** (DOI: 10.1016/j.gloenvcha.2016.10.009),
* **World Bank labour participation data** (2025, https://databank.worldbank.org/source/world-development-indicators), and
* **Penn World Table (PWT 9.1)** (DOI: 10.1257/aer.20130954).

These datasets are used to obtain:

* population age distributions,
* mortality projections,
* GDP per capita trajectories,
* labour and work-leisure parameters.

---

## Repository structure (example)

```
repository
│
├── src
│   ├── Figures                
│       └── generated figures
│   ├── input.db                # Input database with all relevant data
│   ├── lqi_swtp.py             # SWTP and SVSL functions
│   └── SWTP_SVSL.ipynb         # Jupyter notebook calculating SWTP/SVSL and plotting
│
├── notebooks
│   └── SWTP_SVSL.ipynb         # analysis notebook
│
├── Cao (2026),  Cao (2026), Non-stationary risk acceptance in light of climate change.pdf
│
└── README.md
```

---

## Reproducing the analysis

1. Install Python dependencies
2. Download the required datasets
3. Run the analysis notebook

The notebook reproduces:

* discounted remaining life expectancy,
* demographic parameters,
* stationary vs non-stationary SWTP, and
* time-averaged SWTP and SVSL.

---

## Assumptions

The implementation follows the modelling assumptions described in the paper:

* constant mortality reduction across ages,
* piecewise-constant hazard rates within 5-year bins,
* trapezoidal numerical integration,
* constant work-leisure ratio,
* socioeconomic projections based on **SSP2**.

---

## Citation

If you use this code, please cite:

```
Cao, A. S. (2026).
Non-stationary risk acceptance in light of climate change.
4th International Conference on Natural Hazards & Infrastructure (ICONHIC2026).
```

---

## License

MIT License.

---

## Contact

Alex Sixie Cao (cao@ibk.baug.ethz.ch; alex.cao@empa.ch)  
Empa – Swiss Federal Laboratories for Materials Science and Technology  
ETH Zurich – Institute of Structural Engineering  
NTU - School of Civil and Environmental Engineering  
