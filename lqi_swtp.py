from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================

@dataclass(frozen=True)
class Config:
    db_path: Path = Path("input.db")
    country: str = "China"
    ssp: str = "SSP2"
    t0: int = 2025
    t0_age: int = 2025
    T: float = 50.0
    dt: float = 1.0
    dt_avg: float = 1.0

    # Discounting: gamma_M(y) = epsilon * delta(y) + rho, delta from GDP PPP pc.
    epsilon: float = 2.0
    rho: float = 0.02
    gamma_M_const: float = 0.02  # fallback if GDP growth not computable

    beta_year: int = 2017  # beta, P15+, H are taken at this year


# ============================================================
# SQLite loading
# ============================================================

_REQUIRED_TABLES = (
    "Survival_female", "Survival_male",
    "female_l", "male_l",
    "g_SSPs", "beta", "P15+_all", "H",
    "population_age",
)

def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).replace("\u00a0", " ")).strip()

def load_required_tables(db_path: Path | str) -> dict[str, pd.DataFrame]:
    db_path = Path(db_path)
    with sqlite3.connect(db_path) as conn:
        tabs = set(pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"])
        missing = sorted(set(_REQUIRED_TABLES) - tabs)
        if missing:
            raise RuntimeError(f"Missing required tables in {db_path}: {missing}")
        out = {t: pd.read_sql_query(f'SELECT * FROM "{t}"', conn) for t in _REQUIRED_TABLES}

    # normalize Scenario whitespace where present
    for k in ("Survival_female", "Survival_male", "female_l", "male_l", "g_SSPs"):
        if "Scenario" in out[k].columns:
            out[k] = out[k].assign(Scenario=lambda d: d["Scenario"].map(_norm_space))
    return out


# ============================================================
# Parsing helpers
# ============================================================

def parse_period(label: str) -> tuple[int, int]:
    y0, y1 = str(label).split("-")
    return int(y0), int(y1)

def parse_age_bin(label: str) -> tuple[int, int, int, float]:
    """Return (a_start, a_end, width, midpoint) for SSP 5-year bins and 'newborn'."""
    label = str(label).strip()
    if label.lower() == "newborn":
        return (0, 1, 1, 0.5)

    m = re.match(r"^(\d+)--(\d+)$", label)
    if m:
        a0 = int(m.group(1))
        a1 = int(m.group(2)) + 1
        L = a1 - a0
        return a0, a1, L, a0 + 0.5 * L

    m = re.match(r"^(\d+)\+$", label)
    if m:
        a0 = int(m.group(1))
        L = 5
        return a0, a0 + L, L, a0 + 0.5 * L

    raise ValueError(f"Unrecognized age label: {label}")

def hazard_from_surv_prob(p: float, L: float) -> float:
    p = min(max(float(p), 1e-12), 1.0)
    if p >= 1.0:
        return 0.0
    return -math.log(p) / float(L)


# ============================================================
# Demography: h(a,t) via population_age
# ============================================================

def build_age_sex_weights(population_age: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pop_df, age_bins) at cfg.t0_age for cfg.country/cfg.ssp.

    pop_df: rows per (Sex, age bin) with pop_thousand
    age_bins: aggregated bins with fA_mass and sex weights wF_age/wM_age
    """
    sw = population_age.query("Area==@cfg.country and Scenario==@cfg.ssp and Year==@cfg.t0_age").copy()
    if sw.empty:
        raise ValueError("No population_age rows for selected country/scenario/year.")

    rows = []
    for _, r in sw.iterrows():
        a0, a1, L, mid = parse_age_bin(r["Age"])
        rows.append({
            "Sex": str(r["Sex"]).strip(),
            "a_start": a0,
            "a_end": a1,
            "L": L,
            "a_mid": mid,
            "pop_thousand": float(r["Value"]),
        })
    pop_df = pd.DataFrame(rows)

    pop_tot = (
        pop_df.groupby(["a_start", "a_end", "L", "a_mid"], as_index=False)["pop_thousand"]
        .sum()
        .sort_values("a_start")
        .reset_index(drop=True)
    )
    Z = float(pop_tot["pop_thousand"].sum())
    pop_tot["fA_mass"] = pop_tot["pop_thousand"] / Z

    sex_by_age = (
        pop_df.groupby(["a_start", "a_end"], as_index=False)
        .apply(lambda g: pd.Series({
            "PF": float(g.loc[g["Sex"] == "Female", "pop_thousand"].sum()),
            "PM": float(g.loc[g["Sex"] == "Male", "pop_thousand"].sum()),
        }))
        .reset_index(drop=True)
    )
    sex_by_age["wF_age"] = sex_by_age["PF"] / (sex_by_age["PF"] + sex_by_age["PM"])
    sex_by_age["wM_age"] = 1.0 - sex_by_age["wF_age"]

    age_bins = pop_tot.merge(sex_by_age, on=["a_start", "a_end"], how="left")
    if abs(float(age_bins["fA_mass"].sum()) - 1.0) > 1e-10:
        raise RuntimeError("Age weights do not sum to 1.0.")
    return pop_df, age_bins


# ============================================================
# Survival -> hazard lookup (SSP bins)
# ============================================================

def build_surv_lookup(survival_df: pd.DataFrame, cfg: Config) -> tuple[dict[tuple[str, str], float], list[str], list[str]]:
    d = survival_df[(survival_df["Area"] == cfg.country) & (survival_df["Scenario"] == cfg.ssp)].copy()
    d = d[d["Age"].str.lower() != "newborn"]
    lut = {(r["Period"], r["Age"]): float(r["Value"]) for _, r in d.iterrows()}
    periods = sorted(d["Period"].unique(), key=lambda p: parse_period(p)[0])
    ages = sorted(d["Age"].unique(), key=lambda x: parse_age_bin(x)[0])
    return lut, periods, ages

def year_to_period(y: int, periods: list[str]) -> str:
    for p in periods:
        y0, y1 = parse_period(p)
        if y0 <= y < y1:
            return p
    return periods[-1] if y >= parse_period(periods[-1])[0] else periods[0]

def age_to_bin(a: float, ages: list[str]) -> str:
    for lab in ages:
        a0, a1, _, _ = parse_age_bin(lab)
        if a0 <= a < a1:
            return lab
    return ages[-1] if a >= parse_age_bin(ages[-1])[0] else ages[0]

def mu(sex: str, year: int, age: float, *, lut_f, lut_m, periods, ages) -> float:
    period = year_to_period(int(year), periods)
    abin = age_to_bin(float(age), ages)
    _, _, L, _ = parse_age_bin(abin)
    p = (lut_f if sex == "Female" else lut_m)[(period, abin)]
    return hazard_from_surv_prob(p, L)


# ============================================================
# Cohort integrals
# ============================================================

def time_grid(cfg: Config) -> np.ndarray:
    n = int(cfg.T / cfg.dt)  # floor to match Riemann discretization
    return np.arange(0, n + 1) * cfg.dt

def cohort_survival_curve(
    cfg: Config,
    sex: str,
    a0: float,
    stationary: bool,
    *,
    mu_kwargs: dict,
) -> tuple[np.ndarray, np.ndarray]:
    tgrid = time_grid(cfg)
    S = np.ones_like(tgrid, dtype=float)

    # mu at t=0
    s0 = float(tgrid[0])
    year0 = cfg.t0 if stationary else int(cfg.t0 + s0)
    mu_prev = mu(sex, year0, a0 + s0, **mu_kwargs)

    cumhaz = 0.0
    for i in range(1, len(tgrid)):
        s1 = float(tgrid[i])
        year1 = cfg.t0 if stationary else int(cfg.t0 + s1)
        mu_curr = mu(sex, year1, a0 + s1, **mu_kwargs)

        # trapezoidal increment
        cumhaz += 0.5 * (mu_prev + mu_curr) * cfg.dt
        S[i] = math.exp(-cumhaz)

        mu_prev = mu_curr

    return tgrid, S

def discounted_remaining_life_expectancy(
    cfg: Config,
    sex: str,
    a0: float,
    stationary: bool,
    *,
    mu_kwargs: dict,
    discount_factors: np.ndarray,
) -> float:
    tgrid, S = cohort_survival_curve(cfg, sex, a0, stationary, mu_kwargs=mu_kwargs)
    if len(discount_factors) != len(tgrid):
        raise ValueError("discount_factors length mismatch with time grid")
    integrand = discount_factors * S
    return float(np.trapz(integrand, tgrid))

def d_ed_dDelta_at_0(
    cfg: Config,
    sex: str,
    a0: float,
    stationary: bool,
    *,
    mu_kwargs: dict,
    discount_factors: np.ndarray,
) -> float:
    tgrid, S = cohort_survival_curve(cfg, sex, a0, stationary, mu_kwargs=mu_kwargs)
    if len(discount_factors) != len(tgrid):
        raise ValueError("discount_factors length mismatch with time grid")
    integrand = tgrid * discount_factors * S
    return float(np.trapz(integrand, tgrid))


# ============================================================
# Age-averaging: E_A[e_d] and C_Δ
# ============================================================

def compute_age_averages(cfg: Config, age_bins: pd.DataFrame, *, stationary: bool, mu_kwargs: dict, discount_factors: np.ndarray) -> tuple[pd.DataFrame, float, float]:
    rows = []
    for _, r in age_bins.iterrows():
        a_mid = float(r["a_mid"])
        wF = float(r["wF_age"])
        wM = float(r["wM_age"])

        ed = (
            wF * discounted_remaining_life_expectancy(cfg, "Female", a_mid, stationary, mu_kwargs=mu_kwargs, discount_factors=discount_factors)
            + wM * discounted_remaining_life_expectancy(cfg, "Male", a_mid, stationary, mu_kwargs=mu_kwargs, discount_factors=discount_factors)
        )
        ded = (
            wF * d_ed_dDelta_at_0(cfg, "Female", a_mid, stationary, mu_kwargs=mu_kwargs, discount_factors=discount_factors)
            + wM * d_ed_dDelta_at_0(cfg, "Male", a_mid, stationary, mu_kwargs=mu_kwargs, discount_factors=discount_factors)
        )

        rows.append({
            "a_start": int(r["a_start"]),
            "a_end": int(r["a_end"]),
            "a_mid": a_mid,
            "L": int(r["L"]),
            "fA_mass": float(r["fA_mass"]),
            "e_d": ed,
            "de_d_dDelta": ded,
        })

    out = pd.DataFrame(rows)
    E_ed = float(np.sum(out["e_d"] * out["fA_mass"]))
    out["de_d_dDelta_over_e_d"] = out["de_d_dDelta"] / out["e_d"]
    C_Delta = float(np.sum(out["de_d_dDelta_over_e_d"] * out["fA_mass"]))
    return out, E_ed, C_Delta


# ============================================================
# q(t) and g(t)
# ============================================================

def compute_g_t0(cfg: Config, g_ssp: pd.DataFrame) -> float:
    g_row = g_ssp.query("Region==@cfg.country and Scenario==@cfg.ssp")
    if g_row.empty:
        raise ValueError("g_SSPs: missing row for (country, ssp).")
    return float(g_row[str(cfg.t0)].iloc[0])

def compute_q_t0(
    cfg: Config,
    *,
    pop_age: pd.DataFrame,
    l_f: pd.DataFrame,
    l_m: pd.DataFrame,
    P15_tbl: pd.DataFrame,
    H_tbl: pd.DataFrame,
    beta_tbl: pd.DataFrame,
    periods: list[str],
) -> tuple[float, float, float]:
    pop_df, _ = build_age_sex_weights(pop_age, cfg)

    PF_tot = float(pop_df.loc[pop_df["Sex"] == "Female", "pop_thousand"].sum())
    PM_tot = float(pop_df.loc[pop_df["Sex"] == "Male", "pop_thousand"].sum())
    wF_tot = PF_tot / (PF_tot + PM_tot)
    wM_tot = 1.0 - wF_tot

    t0_period = year_to_period(cfg.t0, periods)
    lF = float(l_f.query("Area==@cfg.country and Scenario==@cfg.ssp and Period==@t0_period")["Value"].iloc[0])
    lM = float(l_m.query("Area==@cfg.country and Scenario==@cfg.ssp and Period==@t0_period")["Value"].iloc[0])
    l_t0 = wF_tot * lF + wM_tot * lM

    col_p15 = f"{cfg.beta_year} [YR{cfg.beta_year}]"
    P15_share = float(P15_tbl.loc[P15_tbl["Country"] == cfg.country, col_p15].iloc[0]) / 100.0
    h_yr = float(H_tbl.loc[(H_tbl["Entity"] == cfg.country) & (H_tbl["Year"] == cfg.beta_year), "Average annual working hours per worker"].iloc[0])

    total_hours = l_t0 * 365.0 * 24.0
    work_hours = P15_share * max(l_t0 - 15.0, 0.0) * h_yr
    w_frac = work_hours / total_hours

    beta_val = float(beta_tbl.loc[beta_tbl["Country"] == cfg.country, str(cfg.beta_year)].iloc[0])
    q_t0 = (1.0 / beta_val) * w_frac / (1.0 - w_frac)
    return q_t0, beta_val, w_frac

def compute_q_year(
    cfg: Config,
    *,
    year: int,
    pop_df: pd.DataFrame,
    l_f: pd.DataFrame,
    l_m: pd.DataFrame,
    P15_tbl: pd.DataFrame,
    H_tbl: pd.DataFrame,
    beta_tbl: pd.DataFrame,
    periods: list[str],
) -> float:
    PF_tot = float(pop_df.loc[pop_df["Sex"] == "Female", "pop_thousand"].sum())
    PM_tot = float(pop_df.loc[pop_df["Sex"] == "Male", "pop_thousand"].sum())
    wF_tot = PF_tot / (PF_tot + PM_tot)
    wM_tot = 1.0 - wF_tot

    period = year_to_period(year, periods)
    lF = float(l_f.query("Area==@cfg.country and Scenario==@cfg.ssp and Period==@period")["Value"].iloc[0])
    lM = float(l_m.query("Area==@cfg.country and Scenario==@cfg.ssp and Period==@period")["Value"].iloc[0])
    l_t = wF_tot * lF + wM_tot * lM

    col_p15 = f"{cfg.beta_year} [YR{cfg.beta_year}]"
    P15_share = float(P15_tbl.loc[P15_tbl["Country"] == cfg.country, col_p15].iloc[0]) / 100.0
    h_yr = float(H_tbl.loc[(H_tbl["Entity"] == cfg.country) & (H_tbl["Year"] == cfg.beta_year), "Average annual working hours per worker"].iloc[0])
    beta_val = float(beta_tbl.loc[beta_tbl["Country"] == cfg.country, str(cfg.beta_year)].iloc[0])

    total_hours = l_t * 365.0 * 24.0
    work_hours = P15_share * max(l_t - 15.0, 0.0) * h_yr
    w_frac = work_hours / total_hours
    return (1.0 / beta_val) * w_frac / (1.0 - w_frac)


# ============================================================
# GDP-driven discounting
# ============================================================

def gdp_series_for_country_ssp(g_ssp: pd.DataFrame, country: str, ssp: str) -> pd.Series:
    row = g_ssp.query("Region==@country and Scenario==@ssp")
    if row.empty:
        raise ValueError("g_SSPs: missing row for (country, ssp).")
    row = row.iloc[0]
    year_cols = [c for c in g_ssp.columns if re.fullmatch(r"\d{4}", str(c))]
    s = row[year_cols].astype(float)
    s.index = s.index.astype(int)
    return s.sort_index()

def gdp_series_annual(g_ssp: pd.DataFrame, *, country: str, ssp: str, year_min: int, year_max: int) -> pd.Series:
    s = gdp_series_for_country_ssp(g_ssp, country, ssp)
    s = s.loc[(s.index >= year_min) & (s.index <= year_max)]
    years = np.arange(year_min, year_max + 1, dtype=int)
    ln_interp = np.interp(years.astype(float), s.index.values.astype(float), np.log(s.values.astype(float)))
    return pd.Series(np.exp(ln_interp), index=years, name="gdp_ppp_pc")

def gamma_by_year_from_gdp(cfg: Config, gdp: pd.Series, year_start: int, year_end: int) -> pd.Series:
    y_req = list(range(int(year_start), int(year_end) + 2))
    avail = gdp.reindex(y_req)
    if avail.isna().any():
        ln = np.log(avail.astype(float)).interpolate(method="linear", limit_direction="both")
        avail = np.exp(ln)
    out = {}
    last = None
    for y in range(int(year_start), int(year_end) + 1):
        g_y = float(avail.loc[y])
        g_n = float(avail.loc[y + 1])
        if g_y > 0.0:
            delta = g_n / g_y - 1.0
            last = float(cfg.epsilon * delta + cfg.rho)
            out[y] = last
        else:
            out[y] = float(last) if last is not None else float(cfg.gamma_M_const)
    return pd.Series(out)

def discount_factors_from_gamma(cfg: Config, gamma_by_year: pd.Series) -> np.ndarray:
    tgrid = time_grid(cfg)
    gam = np.zeros_like(tgrid, dtype=float)

    for i, t in enumerate(tgrid):
        y = int(cfg.t0 + math.floor(float(t)))
        gam[i] = float(gamma_by_year.loc[y]) if y in gamma_by_year.index else float(gamma_by_year.iloc[-1])

    # trapezoidal cumulative integral of gamma
    cum = np.cumsum(0.5 * (gam[:-1] + gam[1:]) * cfg.dt)
    df = np.ones_like(tgrid, dtype=float)
    df[1:] = np.exp(-cum)
    return df

def build_discount_factors(cfg: Config, g_ssp: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    year_end = int(cfg.t0 + math.ceil(cfg.T))
    gdp = gdp_series_for_country_ssp(g_ssp, cfg.country, cfg.ssp)
    gamma = gamma_by_year_from_gdp(cfg, gdp, cfg.t0, year_end)
    return gamma, discount_factors_from_gamma(cfg, gamma)


# ============================================================
# SVSL / SWTP summaries
# ============================================================

def compute_svsl_swtp(cfg: Config, *, g_t0: float, q_t0: float, E_ed_ns: float, E_ed_st: float, C_Delta_ns: float, C_Delta_st: float) -> dict[str, float]:
    return {
        "SVSL_ns": (g_t0 / q_t0) * E_ed_ns,
        "SVSL_st": (g_t0 / q_t0) * E_ed_st,
        "SWTP_unit_ns": (g_t0 / q_t0) * C_Delta_ns,
        "SWTP_unit_st": (g_t0 / q_t0) * C_Delta_st,
    }


# ============================================================
# Time-averaged SVSL_T and SWTP_T (remaining horizon t0+s;T-s)
# ============================================================

def time_average_svsl_swtp(
    cfg: Config,
    *,
    tables: dict[str, pd.DataFrame],
    g_ssp: pd.DataFrame,
    use_stationary_mortality: bool = False,
    nonstationary_demography: bool = False,
) -> dict[str, object]:
    l_f = tables["female_l"]
    l_m = tables["male_l"]
    P15_tbl = tables["P15+_all"]
    H_tbl = tables["H"]
    beta_tbl = tables["beta"]
    pop_age = tables["population_age"]
    surv_f = tables["Survival_female"]
    surv_m = tables["Survival_male"]

    periods = sorted(
        {str(p) for p in pd.concat([l_f["Period"], l_m["Period"]]).unique()},
        key=lambda p: parse_period(p)[0],
    )

    lut_f, periods_mu, ages_mu = build_surv_lookup(surv_f, cfg)
    lut_m, _, _ = build_surv_lookup(surv_m, cfg)
    mu_kwargs = {"lut_f": lut_f, "lut_m": lut_m, "periods": periods_mu, "ages": ages_mu}

    t0 = int(cfg.t0)
    T = float(cfg.T)
    dt_avg = float(cfg.dt_avg)
    if T < 0 or dt_avg <= 0:
        raise ValueError("Invalid cfg.T or cfg.dt_avg.")

    s_grid = np.arange(0.0, T + 1e-12, dt_avg)
    years = (t0 + s_grid).astype(int)

    g_ann = gdp_series_annual(g_ssp, country=cfg.country, ssp=cfg.ssp, year_min=int(years.min()), year_max=int(years.max()))

    # precompute available population years once (SSP population_age is typically 5-year)
    pop_years = (
        pop_age.loc[(pop_age["Area"] == cfg.country) & (pop_age["Scenario"] == cfg.ssp), "Year"]
        .dropna().astype(int).unique()
    )
    if len(pop_years) == 0:
        raise ValueError("population_age has no rows for selected country/scenario.")
    pop_years = np.sort(pop_years)

    if not nonstationary_demography:
        pop_df0, age_bins0 = build_age_sex_weights(pop_age, replace(cfg, t0_age=int(cfg.t0_age)))

    svsl_t, swtp_t, E_ed_t, C_Delta_t, q_t, y_pop_used = [], [], [], [], [], []

    for s, y in zip(s_grid, years):
        rem = max(T - float(s), 0.0)
        cfg_y = replace(cfg, t0=int(y), T=float(rem))

        if nonstationary_demography:
            idx = np.searchsorted(pop_years, int(y), side="right") - 1
            y_pop = int(pop_years[max(idx, 0)])
            pop_df_y, age_bins_y = build_age_sex_weights(pop_age, replace(cfg_y, t0_age=y_pop))
        else:
            y_pop = int(cfg.t0_age)
            pop_df_y, age_bins_y = pop_df0, age_bins0

        _, disc = build_discount_factors(cfg_y, g_ssp)

        _, E_ed, C_Delta = compute_age_averages(
            cfg_y,
            age_bins_y,
            stationary=bool(use_stationary_mortality),
            mu_kwargs=mu_kwargs,
            discount_factors=disc,
        )

        q_y = float(
            compute_q_year(
                cfg,
                year=int(y),
                pop_df=pop_df_y,
                l_f=l_f,
                l_m=l_m,
                P15_tbl=P15_tbl,
                H_tbl=H_tbl,
                beta_tbl=beta_tbl,
                periods=periods,
            )
        )

        factor = float(g_ann.loc[int(y)] / q_y)

        svsl_t.append(factor * float(E_ed))
        swtp_t.append(factor * float(C_Delta))
        E_ed_t.append(float(E_ed))
        C_Delta_t.append(float(C_Delta))
        q_t.append(float(q_y))
        y_pop_used.append(y_pop)

    svsl_t = np.asarray(svsl_t, dtype=float)
    swtp_t = np.asarray(swtp_t, dtype=float)

    if T > 0:
        SVSL_T = float(np.trapz(svsl_t, s_grid) / T)
        SWTP_T = float(np.trapz(swtp_t, s_grid) / T)
    else:
        SVSL_T = float(svsl_t[-1])
        SWTP_T = float(swtp_t[-1])

    return {
        "SVSL_T": SVSL_T,
        "SWTP_T": SWTP_T,
        "series": pd.DataFrame({
            "s": s_grid,
            "year": years,
            "year_pop": y_pop_used,
            "g": [float(g_ann.loc[int(yy)]) for yy in years],
            "q": q_t,
            "E_ed": E_ed_t,
            "C_Delta": C_Delta_t,
            "SVSL": svsl_t,
            "SWTP_unit": swtp_t,
        }),
    }
