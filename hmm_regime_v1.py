"""
HMM REGIME DETECTOR v1 — WDO B3

Detecta automaticamente o regime de mercado atual
e ativa a estrategia correta para cada regime.

REGIMES DETECTADOS:
0 = Tendencia + baixa vol  -> EMA crossover
1 = Lateral + vol normal   -> RSI reversion
2 = Alta volatilidade      -> Volatility short
3 = Caotico/sem padrao     -> NAO OPERA

PIPELINE:
1. HMM treina nos dados historicos
2. Classifica cada periodo em regime
3. Para cada regime -> testa estrategias especificas
4. Walk-Forward Rolling valida robustez
5. Resultado: qual estrategia operar hoje
"""

import pandas as pd
import numpy as np
import json, sys, os, time, warnings, math
from datetime import datetime
from numba import njit
from scipy import stats
warnings.filterwarnings("ignore")

CSV_PATH   = "/workspace/strategy_composer/wdo_clean.csv"
OUTPUT_DIR = "/workspace/param_opt_output/hmm_regime"
CAPITAL    = 50_000.0
MULT       = 10.0
COMM       = 5.0
SLIP       = 2.0
MIN_TRADES = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# SECAO 1: DADOS
# ================================================================

def carregar():
    print("[DATA] Carregando...", flush=True)
    df = pd.read_csv(CSV_PATH, parse_dates=["datetime"], index_col="datetime")
    df.columns = [c.lower() for c in df.columns]
    df = df[df.index.dayofweek < 5]
    df = df[(df.index.hour >= 9) & (df.index.hour < 18)]
    df = df.dropna().sort_index()
    df = df[~df.index.duplicated(keep="last")]
    print(f"[DATA] {len(df):,} candles | {df.index[0].date()} -> {df.index[-1].date()}", flush=True)
    return df


# ================================================================
# SECAO 2: FEATURES PARA HMM (diarias)
# ================================================================

def calcular_features_diarias(df):
    """
    Agrega 1min para diario e calcula features para o HMM.
    HMM funciona melhor em dados diarios — captura regime macro.
    """
    print("[HMM] Calculando features diarias...", flush=True)

    # Agregar para diario
    daily = df.resample("1D").agg(
        open=("open",   "first"),
        high=("high",   "max"),
        low=("low",     "min"),
        close=("close", "last"),
        volume=("volume","sum"),
    ).dropna()

    c = daily["close"].values
    h = daily["high"].values
    l = daily["low"].values
    v = daily["volume"].values

    features = pd.DataFrame(index=daily.index)

    # 1. Retorno diario
    features["ret"] = daily["close"].pct_change()

    # 2. Volatilidade (desvio padrao dos retornos 5d)
    features["vol_5d"] = features["ret"].rolling(5).std()

    # 3. Volatilidade 20d
    features["vol_20d"] = features["ret"].rolling(20).std()

    # 4. Ratio de volatilidade (vol recente vs historica)
    features["vol_ratio"] = features["vol_5d"] / (features["vol_20d"] + 1e-9)

    # 5. Tendencia (EMA 10 vs EMA 30)
    ema10 = daily["close"].ewm(span=10).mean()
    ema30 = daily["close"].ewm(span=30).mean()
    features["trend"] = (ema10 - ema30) / ema30

    # 6. Range diario normalizado
    features["range_norm"] = (h - l) / (c + 1e-9)

    # 7. Momentum 10d
    features["mom_10"] = daily["close"].pct_change(10)

    # 8. Volume zscore
    vol_mean = daily["volume"].rolling(20).mean()
    vol_std  = daily["volume"].rolling(20).std()
    features["vol_z"] = (v - vol_mean) / (vol_std + 1e-9)

    features = features.dropna()
    print(f"[HMM] {len(features)} dias com features", flush=True)
    return features, daily


# ================================================================
# SECAO 3: HMM GAUSSIANO (implementado em numpy puro)
# ================================================================

class GaussianHMM:
    """
    HMM Gaussiano simples implementado em numpy.
    Sem dependencia de hmmlearn.
    Algoritmo Baum-Welch para treinamento.
    """

    def __init__(self, n_states=4, n_iter=100, tol=1e-4, seed=42):
        self.n_states = n_states
        self.n_iter   = n_iter
        self.tol      = tol
        self.seed     = seed
        self.fitted   = False

    def _init_params(self, X):
        np.random.seed(self.seed)
        n, d = X.shape
        K    = self.n_states

        idx = np.random.choice(n, K, replace=False)
        self.means_     = X[idx].copy()
        self.covars_    = np.array([np.eye(d) * np.var(X, axis=0) for _ in range(K)])
        self.transmat_  = np.ones((K, K)) / K
        self.startprob_ = np.ones(K) / K

    def _gaussian_pdf(self, x, mean, cov):
        """Probabilidade de x dado media e covariancia."""
        d    = len(mean)
        diff = x - mean
        try:
            cov_inv = np.linalg.inv(cov + np.eye(d) * 1e-6)
            det     = np.linalg.det(cov + np.eye(d) * 1e-6)
            if det <= 0:
                det = 1e-10
            norm = 1.0 / (np.sqrt((2 * np.pi) ** d * det))
            exp  = np.exp(-0.5 * diff @ cov_inv @ diff)
            return norm * exp + 1e-300
        except Exception:
            return 1e-300

    def _emission_probs(self, X):
        """Calcula probabilidades de emissao B[t,k] para todos t,k."""
        n, d = X.shape
        K    = self.n_states
        B    = np.zeros((n, K))
        for k in range(K):
            for t in range(n):
                B[t, k] = self._gaussian_pdf(X[t], self.means_[k], self.covars_[k])
        return B

    def _forward(self, B):
        """Algoritmo Forward."""
        n, K  = B.shape
        alpha = np.zeros((n, K))
        alpha[0] = self.startprob_ * B[0]
        alpha[0] /= (alpha[0].sum() + 1e-300)

        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ self.transmat_) * B[t]
            s = alpha[t].sum()
            if s > 0:
                alpha[t] /= s

        return alpha

    def _backward(self, B):
        """Algoritmo Backward."""
        n, K = B.shape
        beta = np.zeros((n, K))
        beta[-1] = 1.0

        for t in range(n - 2, -1, -1):
            beta[t] = self.transmat_ @ (B[t + 1] * beta[t + 1])
            s = beta[t].sum()
            if s > 0:
                beta[t] /= s

        return beta

    def fit(self, X):
        """Treina HMM com Baum-Welch."""
        print(f"[HMM] Treinando {self.n_states} estados...", flush=True)
        X = np.array(X, dtype=np.float64)
        self._init_params(X)
        n, d = X.shape
        K    = self.n_states

        prev_ll = -np.inf

        for iteration in range(self.n_iter):
            # E-step
            B     = self._emission_probs(X)
            alpha = self._forward(B)
            beta  = self._backward(B)

            # Gamma: prob de estar no estado k no tempo t
            gamma     = alpha * beta
            gamma_sum = gamma.sum(axis=1, keepdims=True)
            gamma_sum[gamma_sum == 0] = 1e-300
            gamma /= gamma_sum

            # Xi: prob de transitar de k para l no tempo t
            xi = np.zeros((n - 1, K, K))
            for t in range(n - 1):
                for k in range(K):
                    for l in range(K):
                        xi[t, k, l] = (alpha[t, k] * self.transmat_[k, l] *
                                       B[t + 1, l] * beta[t + 1, l])
                s = xi[t].sum()
                if s > 0:
                    xi[t] /= s

            # M-step
            self.startprob_ = gamma[0] + 1e-10
            self.startprob_ /= self.startprob_.sum()

            # Matriz de transicao
            xi_sum  = xi.sum(axis=0)
            gamma_t = gamma[:-1].sum(axis=0)
            for k in range(K):
                if gamma_t[k] > 0:
                    self.transmat_[k] = xi_sum[k] / gamma_t[k]
                else:
                    self.transmat_[k] = 1.0 / K
                self.transmat_[k] /= self.transmat_[k].sum()

            # Medias e covariancias
            for k in range(K):
                gk     = gamma[:, k]
                gk_sum = gk.sum()
                if gk_sum > 0:
                    self.means_[k] = (gk[:, None] * X).sum(axis=0) / gk_sum
                    diff = X - self.means_[k]
                    self.covars_[k] = (gk[:, None, None] *
                                       diff[:, :, None] *
                                       diff[:, None, :]).sum(axis=0) / gk_sum
                    # Regularizacao
                    self.covars_[k] += np.eye(d) * 1e-4

            # Log-likelihood
            ll = np.log(B.sum(axis=1) + 1e-300).sum()
            if abs(ll - prev_ll) < self.tol:
                print(f"[HMM] Convergiu em {iteration + 1} iteracoes", flush=True)
                break
            prev_ll = ll

        self.fitted = True
        return self

    def predict(self, X):
        """Viterbi: sequencia mais provavel de estados."""
        X    = np.array(X, dtype=np.float64)
        B    = self._emission_probs(X)
        n, K = B.shape

        viterbi = np.zeros((n, K))
        backptr = np.zeros((n, K), dtype=int)

        viterbi[0] = np.log(self.startprob_ + 1e-300) + np.log(B[0] + 1e-300)

        for t in range(1, n):
            for k in range(K):
                trans        = viterbi[t - 1] + np.log(self.transmat_[:, k] + 1e-300)
                backptr[t, k] = np.argmax(trans)
                viterbi[t, k] = trans[backptr[t, k]] + np.log(B[t, k] + 1e-300)

        states     = np.zeros(n, dtype=int)
        states[-1] = np.argmax(viterbi[-1])
        for t in range(n - 2, -1, -1):
            states[t] = backptr[t + 1, states[t + 1]]

        return states

    def predict_proba(self, X):
        """Probabilidade de cada estado."""
        X     = np.array(X, dtype=np.float64)
        B     = self._emission_probs(X)
        alpha = self._forward(B)
        beta  = self._backward(B)
        gamma = alpha * beta
        s     = gamma.sum(axis=1, keepdims=True)
        s[s == 0] = 1e-300
        return gamma / s


# ================================================================
# SECAO 4: INTERPRETACAO DOS REGIMES
# ================================================================

def interpretar_regimes(features, states, n_states):
    """
    Interpreta cada regime baseado nas suas caracteristicas medias.
    Rotula como: TENDENCIA, LATERAL, ALTA_VOL, CAOTICO
    """
    print("\n[HMM] Interpretando regimes...", flush=True)
    rotulos      = {}
    stats_regime = {}

    for k in range(n_states):
        mask = states == k
        if mask.sum() == 0:
            continue
        f_k = features[mask]
        stats_regime[k] = {
            "n_dias":    int(mask.sum()),
            "pct":       round(float(mask.mean() * 100), 1),
            "ret_medio": round(float(f_k["ret"].mean() * 100), 3),
            "vol_media": round(float(f_k["vol_5d"].mean() * 100), 3),
            "vol_ratio": round(float(f_k["vol_ratio"].mean()), 2),
            "trend_med": round(float(f_k["trend"].mean() * 100), 3),
            "mom_med":   round(float(f_k["mom_10"].mean() * 100), 3),
        }

    # Classificar regimes por volatilidade e tendencia
    vols   = {k: stats_regime[k]["vol_media"]       for k in stats_regime}
    trends = {k: abs(stats_regime[k]["trend_med"])   for k in stats_regime}

    for k in stats_regime:
        vol   = vols[k]
        trend = trends[k]
        ret   = stats_regime[k]["ret_medio"]
        vr    = stats_regime[k]["vol_ratio"]

        if vr > 1.5:
            rotulo     = "ALTA_VOL"
            estrategia = "volatility_short"
        elif abs(trend) > 0.3:
            rotulo     = "TENDENCIA"
            estrategia = "ema_crossover"
        elif vol < np.percentile(list(vols.values()), 40):
            rotulo     = "LATERAL"
            estrategia = "rsi_reversion"
        else:
            rotulo     = "CAOTICO"
            estrategia = "nao_opera"

        rotulos[k] = {"rotulo": rotulo, "estrategia": estrategia}
        stats_regime[k].update(rotulos[k])

        print(f"  Estado {k}: {rotulo:12} | "
              f"Vol={vol:.3f}% | Trend={trend:.3f}% | "
              f"Ret={ret:.3f}% | {stats_regime[k]['n_dias']} dias -> {estrategia}",
              flush=True)

    return stats_regime, rotulos


# ================================================================
# SECAO 5: SIMULADOR NUMBA
# ================================================================

@njit(cache=True)
def simular_numba(open_next, high, low, entries, exits, sl_pts, tp_pts,
                  capital, mult, comm, slip):
    n        = len(open_next)
    pnls     = np.empty(n, dtype=np.float64)
    n_trades = 0
    em_pos   = False
    entry_p  = sl = tp = 0.0
    direction = 1

    for i in range(n - 1):
        if em_pos:
            hit_sl = (direction == 1 and low[i]  <= sl) or (direction == -1 and high[i] >= sl)
            hit_tp = (direction == 1 and high[i] >= tp) or (direction == -1 and low[i]  <= tp)
            force  = exits[i]
            if hit_sl or hit_tp or force:
                saida = sl if hit_sl else (tp if hit_tp else open_next[i])
                pts   = (saida - entry_p) * direction
                pnl   = pts * mult - comm - slip * mult * 0.1
                pnls[n_trades] = pnl
                n_trades += 1
                em_pos = False
            continue
        if entries[i] and not em_pos:
            ep = open_next[i]
            if np.isnan(ep) or ep <= 0:
                continue
            entry_p   = ep
            direction = 1
            sl        = entry_p - sl_pts
            tp        = entry_p + tp_pts
            em_pos    = True

    return pnls[:n_trades]


def metricas_rapidas(pnls):
    if len(pnls) < MIN_TRADES:
        return None
    wins  = pnls[pnls > 0]
    loses = pnls[pnls <= 0]
    if len(loses) == 0 or len(wins) == 0:
        return None
    pf  = abs(wins.sum() / loses.sum())
    wr  = len(wins) / len(pnls) * 100
    sh  = pnls.mean() / (pnls.std() + 1e-9) * np.sqrt(252 * 390) * 0.01
    exp = pnls.mean()
    eq  = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(pnls)])
    pk  = np.maximum.accumulate(eq)
    mdd = float(((eq - pk) / pk * 100).min())
    return {
        "total_trades":     len(pnls),
        "win_rate":         round(wr, 2),
        "profit_factor":    round(pf, 3),
        "sharpe":           round(sh, 3),
        "expectancy_brl":   round(exp, 2),
        "total_pnl_brl":    round(float(pnls.sum()), 2),
        "max_drawdown_pct": round(mdd, 2),
    }


# ================================================================
# SECAO 6: INDICADORES RAPIDOS
# ================================================================

def indicadores(df):
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)

    ind = {
        "close":     c,
        "high":      h,
        "low":       l,
        "open":      o,
        "open_next": np.concatenate([o[1:], [c[-1]]]),
    }

    # EMAs
    for p in [5, 10, 20, 50]:
        a   = 2 / (p + 1)
        out = np.empty_like(c); out[0] = c[0]
        for i in range(1, len(c)):
            out[i] = a * c[i] + (1 - a) * out[i - 1]
        ind[f"ema_{p}"] = out

    # RSI 14
    d  = np.diff(c, prepend=c[0])
    g  = np.where(d > 0, d, 0.0)
    ls = np.where(d < 0, -d, 0.0)
    ag = np.full(len(c), np.nan)
    al = np.full(len(c), np.nan)
    p  = 14
    if p < len(c):
        ag[p] = g[1:p + 1].mean()
        al[p] = ls[1:p + 1].mean()
        for i in range(p + 1, len(c)):
            ag[i] = (ag[i - 1] * (p - 1) + g[i]) / p
            al[i] = (al[i - 1] * (p - 1) + ls[i]) / p
    ind["rsi_14"] = 100 - (100 / (1 + ag / (al + 1e-9)))

    # ATR 14
    prev = np.roll(c, 1); prev[0] = c[0]
    tr   = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    atr  = np.full(len(c), np.nan)
    if 14 < len(c):
        atr[13] = tr[:14].mean()
        for i in range(14, len(c)):
            atr[i] = (atr[i - 1] * 13 + tr[i]) / 14
    ind["atr_14"] = atr

    # Volatilidade relativa
    ret = np.diff(c, prepend=c[0]) / (c + 1e-9)
    v5  = pd.Series(ret).rolling(5).std().values * 100
    v20 = pd.Series(ret).rolling(20).std().values * 100
    ind["vol_ratio"] = v5 / (v20 + 1e-9)

    # MACD hist
    e12 = np.empty_like(c); e12[0] = c[0]; a12 = 2 / 13
    e26 = np.empty_like(c); e26[0] = c[0]; a26 = 2 / 27
    for i in range(1, len(c)):
        e12[i] = a12 * c[i] + (1 - a12) * e12[i - 1]
        e26[i] = a26 * c[i] + (1 - a26) * e26[i - 1]
    mac = e12 - e26
    sig = np.empty_like(c); sig[0] = mac[0]; a9 = 2 / 10
    for i in range(1, len(c)):
        sig[i] = a9 * mac[i] + (1 - a9) * sig[i - 1]
    ind["macd_hist"] = mac - sig

    # Sessao
    hora = df.index.hour
    ind["session_am"] = ((hora >= 9)  & (hora < 12)).astype(np.int8)
    ind["session_pm"] = ((hora >= 13) & (hora < 17)).astype(np.int8)

    return ind


def sinais_por_estrategia(ind, estrategia, params=None):
    """Gera sinais para cada tipo de estrategia."""
    def h1(x):
        return np.roll(x, 1)

    if estrategia == "ema_crossover":
        ef  = ind["ema_5"]; es = ind["ema_20"]
        ent = (ef > es) & (h1(ef) <= h1(es))
        ext = (ef < es) & (h1(ef) >= h1(es))
        ent[0] = ext[0] = False

    elif estrategia == "rsi_reversion":
        rsi = ind["rsi_14"]
        ent = (rsi < 30) & (h1(rsi) >= 30)
        ext = rsi > 50
        ent[0] = ext[0] = False

    elif estrategia == "volatility_short":
        vr   = ind["vol_ratio"]
        hist = ind["macd_hist"]
        sess = ind["session_pm"].astype(bool)
        exp  = vr > 1.5
        ent  = exp & (hist < 0) & (h1(hist) >= 0) & sess
        ext  = exp & (hist > 0) & (h1(hist) <= 0)

    else:
        return None, None

    return ent.astype(np.bool_), ext.astype(np.bool_)


# ================================================================
# SECAO 7: WALK-FORWARD ROLLING
# ================================================================

def walk_forward_rolling(df, regime_diario, rotulos,
                         window_is_dias=120, window_oos_dias=30):
    """
    Walk-Forward Rolling:
    - Janela IS: 120 dias
    - Janela OOS: 30 dias
    - Desloca 30 dias a cada iteracao
    - Para cada janela: detecta regime -> testa estrategia -> valida OOS
    """
    print(f"\n[WFO] Walk-Forward Rolling", flush=True)
    print(f"  IS={window_is_dias} dias | OOS={window_oos_dias} dias", flush=True)

    datas_unicas = df.index.normalize().unique()
    resultados   = []
    n_janelas    = 0

    i = window_is_dias
    while i + window_oos_dias <= len(datas_unicas):
        data_is_start  = datas_unicas[i - window_is_dias]
        data_is_end    = datas_unicas[i - 1]
        data_oos_start = datas_unicas[i]
        data_oos_end   = datas_unicas[min(i + window_oos_dias - 1, len(datas_unicas) - 1)]

        df_is  = df[(df.index.normalize() >= data_is_start) &
                    (df.index.normalize() <= data_is_end)]
        df_oos = df[(df.index.normalize() >= data_oos_start) &
                    (df.index.normalize() <= data_oos_end)]

        if len(df_is) < 1000 or len(df_oos) < 100:
            i += window_oos_dias
            continue

        # Regime dominante no IS
        reg_is = regime_diario[
            (regime_diario.index.normalize() >= data_is_start) &
            (regime_diario.index.normalize() <= data_is_end)
        ]

        if len(reg_is) == 0:
            i += window_oos_dias
            continue

        regime_dom  = int(reg_is["state"].mode()[0])
        info_regime = rotulos.get(regime_dom, {})
        estrategia  = info_regime.get("estrategia", "nao_opera")
        rotulo      = info_regime.get("rotulo", "?")

        if estrategia == "nao_opera":
            i += window_oos_dias
            continue

        # Testar no OOS
        ind_oos  = indicadores(df_oos)
        ent, ext = sinais_por_estrategia(ind_oos, estrategia)
        if ent is None:
            i += window_oos_dias
            continue

        atr_pts = float(np.nanmean(ind_oos["atr_14"]))
        sl_pts  = atr_pts * 1.0
        tp_pts  = sl_pts  * 2.0

        pnls = simular_numba(
            ind_oos["open_next"].astype(np.float64),
            ind_oos["high"].astype(np.float64),
            ind_oos["low"].astype(np.float64),
            ent, ext, sl_pts, tp_pts, CAPITAL, MULT, COMM, SLIP,
        )
        m = metricas_rapidas(pnls)

        resultado = {
            "janela":     n_janelas + 1,
            "is_start":   str(data_is_start.date()),
            "is_end":     str(data_is_end.date()),
            "oos_start":  str(data_oos_start.date()),
            "oos_end":    str(data_oos_end.date()),
            "regime":     rotulo,
            "estado":     regime_dom,
            "estrategia": estrategia,
            "metricas":   m,
            "lucrativo":  m is not None and m["profit_factor"] > 1.0,
        }
        resultados.append(resultado)

        pf_str = f"{m['profit_factor']:.3f}" if m else "N/A"
        luc    = "OK" if resultado["lucrativo"] else "RUIM"
        print(f"  J{n_janelas + 1:02d} {data_oos_start.date()} "
              f"[{rotulo:10} -> {estrategia:20}] "
              f"PF={pf_str} {luc}", flush=True)

        n_janelas += 1
        i += window_oos_dias

    return resultados


# ================================================================
# SECAO 8: ANALISE DOS RESULTADOS WFO
# ================================================================

def analisar_wfo(resultados):
    """Calcula WFE e outras metricas do Walk-Forward."""
    if not resultados:
        return {}

    lucrativos = sum(1 for r in resultados if r["lucrativo"])
    total      = len(resultados)
    wfe        = lucrativos / total * 100

    pf_list = [r["metricas"]["profit_factor"]
               for r in resultados if r["metricas"]]

    por_estrategia = {}
    for r in resultados:
        e = r["estrategia"]
        if e not in por_estrategia:
            por_estrategia[e] = {"total": 0, "lucrativos": 0, "pf_list": []}
        por_estrategia[e]["total"] += 1
        if r["lucrativo"]:
            por_estrategia[e]["lucrativos"] += 1
        if r["metricas"]:
            por_estrategia[e]["pf_list"].append(r["metricas"]["profit_factor"])

    print(f"\n{'=' * 60}", flush=True)
    print(f"  RESULTADOS WALK-FORWARD", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Total janelas:  {total}", flush=True)
    print(f"  Lucrativas:     {lucrativos}/{total}", flush=True)
    print(f"  WFE:            {wfe:.1f}% ({'ROBUSTO' if wfe >= 60 else 'FRAGIL'})", flush=True)
    if pf_list:
        print(f"  PF medio:       {np.mean(pf_list):.3f}", flush=True)
    print(f"\n  Por estrategia:", flush=True)
    for e, v in por_estrategia.items():
        wfe_e = v["lucrativos"] / v["total"] * 100
        pf_m  = np.mean(v["pf_list"]) if v["pf_list"] else 0
        print(f"  {e:25} WFE={wfe_e:.0f}% PF_med={pf_m:.3f} "
              f"({v['lucrativos']}/{v['total']})", flush=True)

    return {
        "total_janelas":  total,
        "lucrativos":     lucrativos,
        "wfe_pct":        round(wfe, 1),
        "pf_medio":       round(float(np.mean(pf_list)), 3) if pf_list else 0,
        "por_estrategia": {
            e: {
                "wfe":        round(v["lucrativos"] / v["total"] * 100, 1),
                "pf_medio":   round(float(np.mean(v["pf_list"])), 3) if v["pf_list"] else 0,
                "total":      v["total"],
                "lucrativos": v["lucrativos"],
            }
            for e, v in por_estrategia.items()
        },
    }


# ================================================================
# SECAO 9: REGIME ATUAL
# ================================================================

def detectar_regime_atual(df, hmm_model, features, rotulos):
    """
    Detecta o regime dos ultimos 30 dias para decidir
    qual estrategia operar HOJE.
    """
    print(f"\n{'=' * 60}", flush=True)
    print(f"  REGIME ATUAL (ultimos 30 dias)", flush=True)

    f_recentes = features.tail(30)
    if len(f_recentes) < 5:
        print("  Dados insuficientes para detectar regime atual", flush=True)
        return None

    X_rec        = f_recentes[["ret", "vol_5d", "vol_ratio", "trend", "mom_10"]].values
    X_rec        = np.nan_to_num(X_rec, nan=0.0)
    states_rec   = hmm_model.predict(X_rec)
    regime_atual = int(stats.mode(states_rec, keepdims=True)[0][0])
    info         = rotulos.get(regime_atual, {})

    print(f"  Regime:     {info.get('rotulo', '?')}", flush=True)
    print(f"  Estrategia: {info.get('estrategia', '?')}", flush=True)
    print(f"  Estado HMM: {regime_atual}", flush=True)

    ultima_data = df.index[-1]
    print(f"  Ultima data: {ultima_data.date()}", flush=True)

    return {
        "data":       str(ultima_data.date()),
        "regime":     info.get("rotulo", "?"),
        "estrategia": info.get("estrategia", "?"),
        "estado_hmm": regime_atual,
    }


# ================================================================
# SECAO 10: MAIN
# ================================================================

def main():
    print("=" * 60, flush=True)
    print("  HMM REGIME DETECTOR v1 — WDO B3", flush=True)
    print("  Detecta regime -> ativa estrategia certa", flush=True)
    print("=" * 60, flush=True)

    df = carregar()

    # Features diarias para HMM
    features, daily = calcular_features_diarias(df)

    # Selecionar colunas para HMM
    cols = ["ret", "vol_5d", "vol_ratio", "trend", "mom_10"]
    X    = features[cols].values
    X    = np.nan_to_num(X, nan=0.0)

    # Normalizar
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0) + 1e-9
    X_norm = (X - X_mean) / X_std

    # Treinar HMM
    hmm = GaussianHMM(n_states=4, n_iter=50, seed=42)
    hmm.fit(X_norm)

    # Sequencia de estados
    states = hmm.predict(X_norm)

    # Criar serie de regimes diarios
    regime_diario = pd.DataFrame({
        "state": states,
        **{c: features[c].values for c in cols},
    }, index=features.index)

    # Interpretar regimes
    stats_regime, rotulos = interpretar_regimes(features, states, 4)

    # Aquece Numba
    print("\n[JIT] Compilando...", flush=True)
    d = np.ones(100, dtype=np.float64) * 5000
    b = np.zeros(100, dtype=np.bool_); b[10] = True
    _ = simular_numba(d, d, d, b, b, 10.0, 20.0, 50000, 10, 5, 2)
    print("[JIT] Pronto!", flush=True)

    # Walk-Forward Rolling
    resultados_wfo = walk_forward_rolling(df, regime_diario, rotulos)

    # Analise WFO
    analise = analisar_wfo(resultados_wfo)

    # Regime atual
    regime_atual = detectar_regime_atual(df, hmm, features, rotulos)

    # Salvar
    output = {
        "gerado_em":      datetime.now().isoformat(),
        "regimes":        stats_regime,
        "rotulos":        rotulos,
        "wfo_resultados": resultados_wfo,
        "wfo_analise":    analise,
        "regime_atual":   regime_atual,
    }

    path = f"{OUTPUT_DIR}/hmm_resultado.json"
    with open(path, "w") as fp:
        json.dump(output, fp, indent=2, default=str)

    print(f"\n[OK] Salvo em {path}", flush=True)

    # Resumo final
    print(f"\n{'=' * 60}", flush=True)
    print(f"  RESUMO FINAL", flush=True)
    print(f"{'=' * 60}", flush=True)
    if regime_atual:
        print(f"  HOJE operar: {regime_atual['estrategia']}", flush=True)
        print(f"  Regime:      {regime_atual['regime']}", flush=True)
    if analise:
        wfe = analise.get("wfe_pct", 0)
        print(f"  WFE geral:   {wfe}% ({'ROBUSTO' if wfe >= 60 else 'FRAGIL'})", flush=True)


if __name__ == "__main__":
    main()
