import io
import base64
import ccxt 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import t as studentt
from numba import njit
from numpy.typing import NDArray
from typing import Optional
import pywt  # for wavelet shrinkage
from flask import Flask, render_template_string
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

###############################################################################
# HELPER FUNCTIONS
###############################################################################
@njit(cache=True)
def ema(arr_in: NDArray, window: int, alpha: Optional[float] = 0) -> NDArray:
    alpha = 3 / float(window + 1) if alpha == 0 else alpha
    n = arr_in.size
    ewma = np.empty(n, dtype=np.float64)
    ewma[0] = arr_in[0]
    for i in range(1, n):
        ewma[i] = (arr_in[i] * alpha) + (ewma[i-1] * (1 - alpha))
    return ewma

def gradual_normalize(values: np.ndarray, window: int = 50, scale: float = 1e4) -> np.ndarray:
    n = len(values)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start_idx = max(0, i - window + 1)
        window_vals = values[start_idx : i + 1]
        max_abs_val = np.max(np.abs(window_vals))
        if max_abs_val == 0:
            out[i] = 0
        else:
            out[i] = values[i] / max_abs_val * scale
    return out

def compute_investment_performance(data, labels):
    returns = np.diff(data) / data[:-1]
    strat_returns = [returns[i] if labels[i] == 1 else -returns[i] for i in range(len(returns))]
    return np.prod(1 + np.array(strat_returns)) - 1

def wavelet_shrinkage(data, wavelet='db4', level=2):
    coeff = pywt.wavedec(data, wavelet, mode='per')
    sigma = np.median(np.abs(coeff[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeff_thresh = [coeff[0]] + [pywt.threshold(c, uthresh, mode='soft') for c in coeff[1:]]
    data_denoised = pywt.waverec(coeff_thresh, wavelet, mode='per')
    return data_denoised[:len(data)]

###############################################################################
# DATA FETCHING
###############################################################################
def fetch_data(symbol="BTC/USD", timeframe="1m", lookback_minutes=1440):
    exchange = ccxt.kraken()
    now_ms = exchange.milliseconds()
    cutoff_ts = now_ms - lookback_minutes * 60 * 1000
    all_ohlcv = []
    since = cutoff_ts
    max_limit = 1440
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=max_limit)
        if not ohlcv:
            break
        all_ohlcv += ohlcv
        last_timestamp = ohlcv[-1][0]
        if last_timestamp <= cutoff_ts or len(ohlcv) < max_limit:
            break
        since = last_timestamp + 1
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["stamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.sort_values("stamp").reset_index(drop=True)

###############################################################################
# INDICATOR CLASSES (BVC and ACIBVC)
###############################################################################
class HawkesBVC:
    def __init__(self, window=20, kappa=0.1, dof=0.25):
        self.window = window
        self.kappa = kappa
        self.dof = dof

    def _label(self, r, sigma):
        if sigma > 0.0:
            return 2 * studentt.cdf(r / sigma, df=self.dof) - 1.0
        else:
            return 0.0

    def eval(self, df: pd.DataFrame, scale=1e4):
        df = df.copy().sort_values("stamp")
        prices = df["close"]
        cumr = np.log(prices / prices.iloc[0])
        r = cumr.diff().fillna(0.0)
        volume = df["volume"]
        sigma = r.rolling(self.window).std().fillna(0.0)
        alpha_exp = np.exp(-self.kappa)
        labels = np.array([self._label(r.iloc[i], sigma.iloc[i]) for i in range(len(r))])
        bvc = np.zeros(len(volume), dtype=float)
        current_bvc = 0.0
        for i in range(len(volume)):
            current_bvc = current_bvc * alpha_exp + volume.values[i] * labels[i]
            bvc[i] = current_bvc
        max_abs = np.max(np.abs(bvc))
        if max_abs != 0:
            bvc = bvc / max_abs * scale
        return pd.DataFrame({"stamp": df["stamp"], "bvc": bvc})

class ACIBVC:
    def __init__(self, kappa=0.1):
        self.kappa = kappa

    def estimate_intensity(self, times, beta):
        intensities = [0.0]
        for i in range(1, len(times)):
            delta_t = times[i] - times[i-1]
            intensities.append(intensities[-1] * np.exp(-beta * delta_t) + 1)
        return np.array(intensities)

    def eval(self, df: pd.DataFrame, scale=1e5):
        df = df.copy().sort_values("stamp")
        df["time_s"] = df["stamp"].astype(np.int64) // 10**9
        times = df["time_s"].values
        intensities = self.estimate_intensity(times, self.kappa)
        df = df.iloc[:len(intensities)]
        df["intensity"] = intensities
        df["price_change"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)
        df["label"] = df["intensity"] * df["price_change"]
        df["weighted_volume"] = df["volume"] * df["label"]
        alpha_exp = np.exp(-self.kappa)
        bvc_list = []
        current_bvc = 0.0
        for wv in df["weighted_volume"].values:
            current_bvc = current_bvc * alpha_exp + wv
            bvc_list.append(current_bvc)
        bvc = np.array(bvc_list)
        max_abs = np.max(np.abs(bvc))
        if max_abs != 0:
            bvc = bvc / max_abs * scale
        df["bvc"] = bvc
        return df[["stamp", "bvc"]].copy()

###############################################################################
# TUNING FUNCTIONS
###############################################################################
def tune_kappa_classification(df_prices, kappa_grid=None, scale=1e4, indicator_type='hawkes'):
    if kappa_grid is None:
        kappa_grid = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    best_kappa = None
    best_f1 = -np.inf
    best_metrics = None
    df_temp = df_prices.copy().sort_values("stamp")
    close_vals = df_temp["close"].values
    gt_labels = np.zeros(len(close_vals))
    for i in range(len(close_vals) - 1):
        gt_labels[i] = 1 if close_vals[i+1] > close_vals[i] else -1
    gt_labels[-1] = gt_labels[-2]
    
    for k in kappa_grid:
        if indicator_type == 'hawkes':
            model = HawkesBVC(window=20, kappa=k)
        else:
            model = ACIBVC(kappa=k)
        indicator_df = model.eval(df_temp.copy(), scale=scale)
        merged = df_temp.merge(indicator_df, on="stamp", how="inner")
        pred_labels = np.where(merged["bvc"].values >= 0, 1, -1)
        accuracy = accuracy_score(gt_labels[:len(pred_labels)], pred_labels)
        precision = precision_score(gt_labels[:len(pred_labels)], pred_labels, pos_label=1)
        recall = recall_score(gt_labels[:len(pred_labels)], pred_labels, pos_label=1)
        f1 = f1_score(gt_labels[:len(pred_labels)], pred_labels, pos_label=1)
        gt_bin = (gt_labels[:len(pred_labels)] == 1).astype(int)
        pred_bin = (pred_labels == 1).astype(int)
        try:
            auc = roc_auc_score(gt_bin, pred_bin)
        except Exception:
            auc = 0.5
        net_yield = compute_investment_performance(merged["close"].values, pred_labels)
        if f1 > best_f1:
            best_f1 = f1
            best_kappa = k
            best_metrics = (accuracy, precision, recall, f1, auc, net_yield)
    return best_kappa, best_metrics

def tune_window_classification(df_prices, window_grid=None, kappa=1e-4, scale=1e4, indicator_type='hawkes'):
    if window_grid is None:
        window_grid = [10, 15, 20, 25, 30]
    best_window = None
    best_f1 = -np.inf
    best_metrics = None
    df_temp = df_prices.copy().sort_values("stamp")
    close_vals = df_temp["close"].values
    gt_labels = np.zeros(len(close_vals))
    for i in range(len(close_vals) - 1):
        gt_labels[i] = 1 if close_vals[i+1] > close_vals[i] else -1
    gt_labels[-1] = gt_labels[-2]
    
    for w in window_grid:
        if indicator_type == 'hawkes':
            model = HawkesBVC(window=w, kappa=kappa)
        else:
            continue
        indicator_df = model.eval(df_temp.copy(), scale=scale)
        merged = df_temp.merge(indicator_df, on="stamp", how="inner")
        pred_labels = np.where(merged["bvc"].values >= 0, 1, -1)
        accuracy = accuracy_score(gt_labels[:len(pred_labels)], pred_labels)
        precision = precision_score(gt_labels[:len(pred_labels)], pred_labels, pos_label=1)
        recall = recall_score(gt_labels[:len(pred_labels)], pred_labels, pos_label=1)
        f1 = f1_score(gt_labels[:len(pred_labels)], pred_labels, pos_label=1)
        gt_bin = (gt_labels[:len(pred_labels)] == 1).astype(int)
        pred_bin = (pred_labels == 1).astype(int)
        try:
            auc = roc_auc_score(gt_bin, pred_bin)
        except Exception:
            auc = 0.5
        net_yield = compute_investment_performance(merged["close"].values, pred_labels)
        if f1 > best_f1:
            best_f1 = f1
            best_window = w
            best_metrics = (accuracy, precision, recall, f1, auc, net_yield)
    return best_window, best_metrics

###############################################################################
# AUTO LABELING FUNCTION (for momentum signal using wavelet-denoised prices)
###############################################################################
def auto_labeling(data_list, timestamp_list, w):
    labels = np.zeros(len(data_list))
    FP = data_list[0]
    x_H = data_list[0]
    HT = timestamp_list[0]
    x_L = data_list[0]
    LT = timestamp_list[0]
    Cid = 0
    FP_N = 0
    for i in range(len(data_list)):
        if data_list[i] > FP + data_list[0] * w:
            x_H = data_list[i]
            HT = timestamp_list[i]
            FP_N = i
            Cid = 1
            break
        if data_list[i] < FP - data_list[0] * w:
            x_L = data_list[i]
            LT = timestamp_list[i]
            FP_N = i
            Cid = -1
            break
    for i in range(FP_N, len(data_list)):
        if Cid > 0:
            if data_list[i] > x_H:
                x_H = data_list[i]
                HT = timestamp_list[i]
            if data_list[i] < x_H - x_H * w and LT < HT:
                for j in range(len(data_list)):
                    if timestamp_list[j] > LT and timestamp_list[j] <= HT:
                        labels[j] = 1
                x_L = data_list[i]
                LT = timestamp_list[i]
                Cid = -1
        elif Cid < 0:
            if data_list[i] < x_L:
                x_L = data_list[i]
                LT = timestamp_list[i]
            if data_list[i] > x_L + x_L * w and HT <= LT:
                for j in range(len(data_list)):
                    if timestamp_list[j] > HT and timestamp_list[j] <= LT:
                        labels[j] = -1
                x_H = data_list[i]
                HT = timestamp_list[i]
                Cid = 1
    labels[0] = labels[1] if len(labels) > 1 else Cid
    labels = np.where(labels == 0, Cid, labels)
    assert len(labels) == len(timestamp_list)
    timestamp2label_dict = {timestamp_list[i]: labels[i] for i in range(len(timestamp_list))}
    return labels, timestamp2label_dict

###############################################################################
# MAIN SCRIPT (FLASK APP) WITH INDICATOR-NORMALIZED COLORING & MOMENTUM
###############################################################################
app = Flask(__name__)

template = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CNO Dashboard</title>
  <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
  <style>
      body { background-color: #f8f9fa; }
      .container { margin-top: 20px; }
  </style>
</head>
<body>
  <div class="container">
    <h1 class="my-4">CNO Dashboard</h1>
    <img src="data:image/png;base64,{{ plot_url }}" class="img-fluid" alt="Plot">
  </div>
</body>
</html>
"""

@app.route("/")
def index():
    # Fetch main data
    df = fetch_data(symbol=ticker_main, timeframe=timeframe, lookback_minutes=720)
    df = df.sort_values("stamp").reset_index(drop=True)
    # Compute additional price fields
    df["ScaledPrice"] = np.log(df["close"] / df["close"].iloc[0]) * 1e4
    df["ScaledPrice_EMA"] = ema(df["ScaledPrice"].values, window=36)
    # Compute VWAP
    df["cum_vol"] = df["volume"].cumsum()
    df["cum_pv"] = (df["close"] * df["volume"]).cumsum()
    df["vwap"] = df["cum_pv"] / df["cum_vol"]
    if df["vwap"].iloc[0] == 0 or not np.isfinite(df["vwap"].iloc[0]):
        df["vwap_transformed"] = df["ScaledPrice"]
    else:
        df["vwap_transformed"] = np.log(df["vwap"] / df["vwap"].iloc[0]) * 1e4

    # --- Compute Indicator based on Analysis Type using optimal kappa
    if analysis_type == "BVC":
        optimal_kappa, metrics = tune_kappa_classification(df, kappa_grid=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
                                                         scale=1e4, indicator_type='hawkes')
        optimal_window, window_metrics = tune_window_classification(df, window_grid=[10, 15, 20, 25, 30],
                                                                    kappa=optimal_kappa, scale=1e4, indicator_type='hawkes')
        indicator_title = "BVC"
        indicator_df = HawkesBVC(window=optimal_window, kappa=optimal_kappa).eval(df.copy(), scale=1e4)
    elif analysis_type == "ACI":
        optimal_kappa, metrics = tune_kappa_classification(df, kappa_grid=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
                                                         scale=1e5, indicator_type='aci')
        indicator_title = "ACI"
        indicator_df = ACIBVC(kappa=optimal_kappa).eval(df.copy(), scale=1e5)
    
    # Merge the indicator into the main DataFrame
    df_merged = df.merge(indicator_df, on="stamp", how="inner")
    df_merged = df_merged.sort_values("stamp")
    df_merged["bvc"] = df_merged["bvc"].fillna(method="ffill").fillna(0)
    
    # --- Compute Momentum Signal using wavelet-denoised auto labeling on close prices
    denoised_close = wavelet_shrinkage(df["close"].values.astype(np.float64), wavelet='db4', level=2)
    momentum_labels, _ = auto_labeling(denoised_close, df["stamp"].values, w=0.0003)
    df_merged["momentum"] = momentum_labels
    df_merged["indicator_momentum"] = df_merged["bvc"] * df_merged["momentum"]

    # Create the main plot
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    norm_indicator = plt.Normalize(df_merged["indicator_momentum"].min(), df_merged["indicator_momentum"].max())
    for i in range(len(df_merged)-1):
        xvals = df_merged["stamp"].iloc[i:i+2]
        yvals = df_merged["ScaledPrice"].iloc[i:i+2]
        indicator_mom_val = df_merged["indicator_momentum"].iloc[i]
        cmap = plt.cm.Blues if indicator_mom_val >= 0 else plt.cm.Reds
        base_color = cmap(norm_indicator(indicator_mom_val))
        darker_color = (0.8 * base_color[0], 0.8 * base_color[1], 0.8 * base_color[2], base_color[3])
        ax.plot(xvals, yvals, color=darker_color, linewidth=1)
    ax.plot(df_merged["stamp"], df_merged["ScaledPrice_EMA"], color="black", linewidth=1, label="EMA(10)")
    for i in range(len(df_merged)-1):
        xvals = df_merged["stamp"].iloc[i:i+2]
        yvals = df_merged["vwap_transformed"].iloc[i:i+2]
        vwap_color = "blue" if df_merged["ScaledPrice"].iloc[i] > df_merged["vwap_transformed"].iloc[i] else "red"
        ax.plot(xvals, yvals, color=vwap_color, linewidth=1)
    # Add watermark: ticker (smaller) and "CNO" below it
    ax.text(0.5, 0.55, ticker_main, transform=ax.transAxes, fontsize=16, color="lightgray",
            alpha=0.3, ha="center", va="center", zorder=0)
    ax.text(0.5, 0.45, "CNO", transform=ax.transAxes, fontsize=12, color="lightgray",
            alpha=0.3, ha="center", va="center", zorder=0)
    ax.set_xlabel("Time", fontsize=8)
    ax.set_ylabel("Scaled Price", fontsize=8)
    ax.set_title(f"Price with EMA & VWAP (Colored by {indicator_title} Momentum)", fontsize=10)
    ax.legend(fontsize=7)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    ax.set_ylim(df_merged["ScaledPrice"].min()-50, df_merged["ScaledPrice"].max()+50)
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode("utf8")
    plt.close(fig)
    
    return render_template_string(template, plot_url=plot_data)

template = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CNO Dashboard</title>
  <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
  <style>
      body { background-color: #f8f9fa; }
      .container { margin-top: 20px; }
  </style>
</head>
<body>
  <div class="container">
    <h1 class="my-4">CNO Dashboard</h1>
    <img src="data:image/png;base64,{{ plot_url }}" class="img-fluid" alt="Plot">
  </div>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)
