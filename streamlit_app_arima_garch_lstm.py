# streamlit_app_arima_garch_lstm.py
# Dashboard ARIMA · ARIMA-GARCH — BBCA.JK
# Kevin Imtinan Fawwaz — NIM 1206225037

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import os
import time
import io as _io
SEED       = 42
np.random.seed(SEED)
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ARIMA-GARCH | BBCA.JK",
    page_icon="📈",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    .sidebar-title { font-size: 20px; font-weight: bold; margin-bottom: 10px; }
    .sidebar-box {
        background-color: white; padding: 12px 16px;
        border-radius: 10px; box-shadow: 0 0 8px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">📌 Navigation Menu</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
    selected_menu = st.radio("", [
        "📊 Data & EDA",
        "🔍 Uji Stasioneritas & ARCH",
        "📐 Identifikasi Ordo (Auto)",
        "🤖 Train & Forecast Semua Model",
        "📋 Evaluasi & Perbandingan",

    ], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

st.title("📈 Forecasting Return Saham BBCA.JK")
st.caption("ARIMA · ARIMA-GARCH | Kevin Imtinan Fawwaz — 1206225037")


# ── Helper metrik ──────────────────────────────────────────────────────────────
def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae  = np.mean(np.abs(y_true - y_pred))
    mask = np.abs(y_true) > 1e-10
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
    da   = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
    denom = np.sum(y_true**2) + np.sum(y_pred**2)
    theil = np.sqrt(np.sum((y_true - y_pred)**2) / denom) if denom > 0 else np.nan
    return dict(RMSE=rmse, MAE=mae, MAPE=mape, DA=da, TheilU=theil)


# ── Load & preprocess data ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Memuat data BBCA.JK...")
def load_data():
    df = pd.read_csv(
        "BBCA_JK_2015-01-02_2025-12-31.csv",
        parse_dates=["Date"]
    )
    df = df.set_index("Date").sort_index()
    df = df.dropna()

    close_col = "Close"
    data = df.copy()
    data["Return"] = np.log(data[close_col] / data[close_col].shift(1))
    data = data.dropna(subset=["Return"])
    data["Return_raw"] = data["Return"].copy()
    mu = data["Return"].mean()
    sd = data["Return"].std()
    data["Return"] = data["Return"].clip(mu - 3*sd, mu + 3*sd)

    n     = len(data)
    train = data.iloc[:int(n*0.80)].copy()
    val   = data.iloc[int(n*0.80):int(n*0.90)].copy()
    test  = data.iloc[int(n*0.90):].copy()
    return data, train, val, test, close_col

data, train, val, test, close_col = load_data()
test_steps = len(test)
# =============================================================================
# HALAMAN 1 — Data & EDA
# =============================================================================
if selected_menu == "📊 Data & EDA":
    st.header("📄 Preview Data & EDA")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series", "📊 Distribusi", "🔬 Decomposition", "📉 Volatilitas"])

    with tab1:
        st.subheader(f"BBCA.JK — {len(data)} observasi harian ({data.index[0].date()} – {data.index[-1].date()})")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Observasi", len(data))
        col2.metric("Train / Val / Test", f"{len(train)} / {len(val)} / {len(test)}")
        col3.metric("Periode", f"{data.index[0].year}–{data.index[-1].year}")

        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
        axes[0].plot(data.index, data['Close'], color='navy', lw=0.9)
        axes[0].set_title('Harga Penutupan BBCA', fontweight='bold')
        axes[0].set_ylabel('Harga (IDR)')

        axes[1].plot(train.index, train['Return_raw'], color='steelblue', lw=0.6, label=f'Train ({len(train)})')
        axes[1].plot(val.index,   val['Return_raw'],   color='orange',    lw=0.6, label=f'Val ({len(val)})')
        axes[1].plot(test.index,  test['Return_raw'],  color='crimson',   lw=0.6, label=f'Test ({len(test)})')
        axes[1].axhline(0, color='k', lw=0.4, ls='--')
        axes[1].set_title('Log-Return Harian (raw, 80/10/10 split)', fontweight='bold')
        axes[1].set_ylabel('Log Return'); axes[1].legend(fontsize=9)

        axes[2].plot(data.index, data['Return'], color='teal', lw=0.6)
        axes[2].axhline(0, color='k', lw=0.4, ls='--')
        axes[2].set_title('Log-Return Winsorized (±3σ) — dipakai sebagai input model', fontweight='bold')
        axes[2].set_ylabel('Log Return'); axes[2].set_xlabel('Tanggal')
        plt.tight_layout(); st.pyplot(fig)

    with tab2:
        from scipy import stats
        from scipy.stats import jarque_bera
        r = data['Return_raw']
        jb_s, jb_p = jarque_bera(r)
        st.subheader("Statistik Deskriptif & Distribusi")
        dcols = st.columns(5)
        dcols[0].metric("Mean",     f"{r.mean():.5f}")
        dcols[1].metric("Std",      f"{r.std():.5f}")
        dcols[2].metric("Skewness", f"{r.skew():.4f}")
        dcols[3].metric("Kurtosis", f"{r.kurtosis():.4f}")
        dcols[4].metric("JB p-val", f"{jb_p:.2e}")

        if jb_p < 0.05:
            st.error("H₀ Jarque-Bera DITOLAK → distribusi TIDAK normal (fat tails / skewness)")
        else:
            st.success("H₀ Jarque-Bera GAGAL DITOLAK → distribusi normal")

        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
        axes2[0].hist(r, bins=60, color='steelblue', edgecolor='white', alpha=0.8, density=True)
        xn = np.linspace(r.min(), r.max(), 200)
        axes2[0].plot(xn, stats.norm.pdf(xn, r.mean(), r.std()), 'r-', lw=2, label='Normal')
        axes2[0].set_title('Distribusi'); axes2[0].legend()
        stats.probplot(r, dist='norm', plot=axes2[1])
        axes2[1].set_title('Q-Q Plot')
        axes2[2].boxplot(r, vert=True, patch_artist=True,
                         boxprops=dict(facecolor='steelblue', alpha=0.6))
        axes2[2].set_title('Boxplot')
        plt.suptitle('Distribusi Log-Return BBCA', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig2)

    with tab3:
        from statsmodels.tsa.seasonal import seasonal_decompose
        st.subheader("Dekomposisi Time Series (Harga, additive, period=252)")
        decomp = seasonal_decompose(data[close_col], model='additive', period=252, extrapolate_trend='freq')
        fig3, axes3 = plt.subplots(4, 1, figsize=(14, 11))
        for ax, series, title, color in zip(
            axes3,
            [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid],
            ['Data Asli', 'Tren', 'Musiman (period=252)', 'Residual'],
            ['navy', 'darkorange', 'green', 'gray']
        ):
            ax.plot(data.index, series, color=color, lw=0.8)
            ax.set_title(title, fontweight='bold'); ax.set_ylabel('')
        plt.tight_layout(); st.pyplot(fig3)

    with tab4:
        st.subheader("Volatility Clustering — Squared Returns (r²)")
        vol_proxy = data['Return_raw'] ** 2
        roll_vol  = vol_proxy.rolling(20).mean()
        fig4, axes4 = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        axes4[0].scatter(data.index, vol_proxy.values, s=3, alpha=0.4, color='steelblue')
        axes4[0].axhline(vol_proxy.mean(), color='red', ls='--', lw=1,
                          label=f'Mean r² = {vol_proxy.mean():.6f}')
        axes4[0].set_title('Squared Returns (r²)', fontweight='bold'); axes4[0].legend()
        axes4[1].plot(data.index, roll_vol.values, color='darkorange', lw=1, label='Rolling vol (20d)')
        axes4[1].set_title('Rolling Volatility (window=20)', fontweight='bold'); axes4[1].legend()
        axes4[1].set_xlabel('Tanggal')
        plt.tight_layout(); st.pyplot(fig4)


# =============================================================================
# HALAMAN 2 — Uji Stasioneritas & ARCH
# =============================================================================
elif selected_menu == "🔍 Uji Stasioneritas & ARCH":
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import het_arch
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from scipy.stats import jarque_bera

    st.header("🔍 Uji Stasioneritas Mean & Efek ARCH")

    # ── ADF ────────────────────────────────────────────────────────────────────
    st.subheader("Uji ADF — Stasioneritas Mean")
    adf_price  = adfuller(data[close_col].dropna(), autolag='AIC')
    adf_return = adfuller(data['Return'].dropna(),  autolag='AIC')

    adf_df = pd.DataFrame([
        {
            'Seri': f'Harga ({close_col})',
            'ADF Stat': round(adf_price[0], 4),
            'p-value':  round(adf_price[1], 6),
            'CV 5%':    round(adf_price[4]['5%'], 4),
            'Stasioner': '✗ Tidak' if adf_price[1] >= 0.05 else '✓ Ya',
        },
        {
            'Seri': 'Log-Return (winsorized)',
            'ADF Stat': round(adf_return[0], 4),
            'p-value':  round(adf_return[1], 6),
            'CV 5%':    round(adf_return[4]['5%'], 4),
            'Stasioner': '✗ Tidak' if adf_return[1] >= 0.05 else '✓ Ya',
        },
    ])
    st.dataframe(adf_df, use_container_width=True)
    st.caption("H₀ ADF: data memiliki unit root (tidak stasioner). p < 0.05 → tolak H₀ → stasioner.")

    st.divider()

    # ── ACF/PACF return ────────────────────────────────────────────────────────
    st.subheader("ACF & PACF Log-Return (Training Set)")
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    plot_acf( train['Return'].dropna(), lags=40, ax=axes[0], alpha=0.05, zero=False)
    axes[0].set_title('ACF Log-Return BBCA (training set)', fontweight='bold')
    axes[0].set_ylim(-0.2, 0.2)
    plot_pacf(train['Return'].dropna(), lags=40, ax=axes[1], alpha=0.05, zero=False)
    axes[1].set_title('PACF Log-Return BBCA (training set)', fontweight='bold')
    axes[1].set_ylim(-0.2, 0.2)
    plt.tight_layout(); st.pyplot(fig)

    st.divider()

    # ── ARCH-LM pada residual ARIMA (jika ordo sudah tersedia) ─────────────────
    st.subheader("Uji ARCH-LM — Justifikasi GARCH")
    if 'best_arima_order' in st.session_state:
        from statsmodels.tsa.arima.model import ARIMA as ARIMA_model
        order = st.session_state['best_arima_order']
        st.info(f"Menggunakan ARIMA{order} dari hasil Auto ARIMA di halaman sebelumnya.")
        arima_diag = ARIMA_model(train['Return'], order=order).fit()
        resid_diag = arima_diag.resid
        lm_stat, lm_pval, _, _ = het_arch(resid_diag.dropna(), nlags=10)

        col1, col2 = st.columns(2)
        col1.metric("LM Statistic", f"{lm_stat:.4f}")
        col2.metric("p-value", f"{lm_pval:.6f}")
        if lm_pval < 0.05:
            st.success(f"✅ Efek ARCH terdeteksi (p={lm_pval:.4f} < 0.05) → GARCH dijustifikasi")
        else:
            st.warning(f"⚠️ Tidak ada efek ARCH (p={lm_pval:.4f} ≥ 0.05)")

        resid_sq = resid_diag ** 2
        fig2, axes2 = plt.subplots(2, 1, figsize=(13, 8))
        plot_acf( resid_sq.dropna(), lags=30, ax=axes2[0], alpha=0.05, zero=False)
        axes2[0].set_title('ACF ε²_t — Bukti Visual Volatility Clustering', fontweight='bold')
        axes2[0].set_ylim(-0.2, 0.2)
        plot_pacf(resid_sq.dropna(), lags=30, ax=axes2[1], alpha=0.05, zero=False)
        axes2[1].set_title('PACF ε²_t', fontweight='bold')
        axes2[1].set_ylim(-0.2, 0.2)
        plt.tight_layout(); st.pyplot(fig2)
    else:
        st.info("Jalankan terlebih dahulu halaman **📐 Identifikasi Ordo** agar uji ARCH-LM tersedia.")


# =============================================================================
# HALAMAN 3 — Identifikasi Ordo (Auto)
# =============================================================================
elif selected_menu == "📐 Identifikasi Ordo (Auto)":
    st.header("📐 Identifikasi Ordo Otomatis")

    # ── Auto ARIMA ─────────────────────────────────────────────────────────────
    st.subheader("Auto ARIMA — Pemilihan Ordo (p,d,q)")
    st.markdown(
        "Auto ARIMA mencari kombinasi (p,d,q) terbaik berdasarkan **AIC** "
        "menggunakan stepwise search dengan deteksi stasioneritas otomatis (ADF/KPSS)."
    )

    run_auto = st.button("🔍 Jalankan Auto ARIMA")
    if run_auto or 'best_arima_order' in st.session_state:
        if run_auto:
            with st.spinner("Menjalankan Auto ARIMA..."):
                from pmdarima import auto_arima
                auto_model = auto_arima(
                    train['Return'],
                    start_p=0, max_p=3, start_q=0, max_q=3,
                    d=None, information_criterion='aic',
                    stepwise=True, seasonal=False,
                    error_action='ignore', suppress_warnings=True, trace=False
                )
                st.session_state['best_arima_order'] = auto_model.order
                st.session_state['best_arima_aic']   = auto_model.aic()
            st.success(f"✅ Auto ARIMA selesai")

        order_a = st.session_state['best_arima_order']
        aic_a   = st.session_state['best_arima_aic']

        st.success(f"**Model terpilih: ARIMA{order_a}** | AIC = {aic_a:.4f}")
        st.info("Ordo ini akan digunakan otomatis di semua tahap berikutnya.")

        # Ringkasan parameter
        from statsmodels.tsa.arima.model import ARIMA as ARIMA_model
        m_sum = ARIMA_model(train['Return'], order=order_a).fit()
        buf = _io.StringIO()
        buf.write(str(m_sum.summary()))
        st.text(buf.getvalue())
    else:
        st.info("Klik tombol di atas untuk menjalankan Auto ARIMA.")

    st.divider()

    # ── Grid Search GARCH ──────────────────────────────────────────────────────
    st.subheader("Grid Search GARCH — Ordo (p,q)")
    st.markdown(
        "GARCH di-fit pada **residual ARIMA** (bukan return mentah). "
        "Ordo terbaik dipilih berdasarkan **Log-Likelihood tertinggi**."
    )

    if 'best_arima_order' not in st.session_state:
        st.warning("Jalankan Auto ARIMA terlebih dahulu.")
    else:
        run_garch = st.button("🔍 Jalankan Grid Search GARCH")
        if run_garch or 'best_garch_order' in st.session_state:
            if run_garch:
                from arch import arch_model as arch_model_fn
                from statsmodels.tsa.arima.model import ARIMA as ARIMA_model
                order_a = st.session_state['best_arima_order']
                arima_tmp = ARIMA_model(train['Return'], order=order_a).fit()
                resid_gs  = arima_tmp.resid * 100

                with st.spinner("Menjalankan Grid Search GARCH(p,q) — p,q ∈ {1,2,3}..."):
                    rows_gs = []
                    best_ll, best_go = -np.inf, None
                    for p in range(1, 4):
                        for q in range(1, 4):
                            try:
                                mg = arch_model_fn(resid_gs, vol='Garch', p=p, q=q,
                                                   dist='normal', rescale=False).fit(disp='off')
                                rows_gs.append({'p': p, 'q': q,
                                                'LogLik': round(mg.loglikelihood, 4),
                                                'AIC':    round(mg.aic, 4),
                                                'BIC':    round(mg.bic, 4)})
                                if mg.loglikelihood > best_ll:
                                    best_ll = mg.loglikelihood
                                    best_go = (p, q)
                            except:
                                pass

                st.session_state['best_garch_order'] = best_go
                st.session_state['garch_gs_table']   = pd.DataFrame(rows_gs).sort_values('LogLik', ascending=False)
                st.success(f"✅ Grid Search selesai")

            go  = st.session_state['best_garch_order']
            gst = st.session_state['garch_gs_table']

            st.success(f"**Model terpilih: GARCH{go}** | LogLik = {gst.iloc[0]['LogLik']}")
            st.dataframe(gst.reset_index(drop=True), use_container_width=True)

            # Parameter detail
            from arch import arch_model as arch_model_fn
            from statsmodels.tsa.arima.model import ARIMA as ARIMA_model
            order_a  = st.session_state['best_arima_order']
            arima_g  = ARIMA_model(train['Return'], order=order_a).fit()
            resid_g  = arima_g.resid * 100
            mg_final = arch_model_fn(resid_g, vol='Garch', p=go[0], q=go[1],
                                     dist='normal', rescale=False).fit(disp='off')

            st.subheader(f"Parameter GARCH{go}")
            omega = mg_final.params['omega']
            alphas = [mg_final.params[f'alpha[{i+1}]'] for i in range(go[1])]
            betas  = [mg_final.params[f'beta[{j+1}]']  for j in range(go[0])]
            persist = sum(alphas) + sum(betas)

            param_rows = [{'Parameter': 'ω (omega)', 'Nilai': round(omega, 6),
                           'p-value': round(mg_final.pvalues['omega'], 4),
                           'Keterangan': 'Baseline variance'}]
            for i, a in enumerate(alphas, 1):
                param_rows.append({'Parameter': f'α_{i}', 'Nilai': round(a, 6),
                                   'p-value': round(mg_final.pvalues[f'alpha[{i}]'], 4),
                                   'Keterangan': 'Efek shock baru'})
            for j, b in enumerate(betas, 1):
                param_rows.append({'Parameter': f'β_{j}', 'Nilai': round(b, 6),
                                   'p-value': round(mg_final.pvalues[f'beta[{j}]'], 4),
                                   'Keterangan': 'Persistensi volatilitas'})
            st.dataframe(pd.DataFrame(param_rows), use_container_width=True)

            stat_msg = "✅ Stasioner (α+β < 1)" if persist < 1 else "⚠️ Tidak stasioner (α+β ≥ 1)"
            st.metric(f"Persistensi α+β = {persist:.6f}", stat_msg)
        else:
            st.info("Klik tombol di atas untuk menjalankan Grid Search GARCH.")


# =============================================================================
# HALAMAN 4 — Train & Forecast Semua Model
# =============================================================================
elif selected_menu == "🤖 Train & Forecast Semua Model":
    st.header("🤖 Train & Forecast — One-Step Ahead")

    # ── Guard: ordo harus sudah ada ───────────────────────────────────────────
    if 'best_arima_order' not in st.session_state or 'best_garch_order' not in st.session_state:
        st.warning("Jalankan terlebih dahulu halaman **📐 Identifikasi Ordo (Auto)** "
                   "untuk mendapatkan ordo ARIMA & GARCH.")
        st.stop()

    best_arima_order = st.session_state['best_arima_order']
    best_garch_order = st.session_state['best_garch_order']

    st.info(f"Ordo yang digunakan: **ARIMA{best_arima_order}** · **GARCH{best_garch_order}**")

    run_all = st.button("🚀 Jalankan Training & Forecasting Semua Model", type="primary")

    if run_all:
        from statsmodels.tsa.arima.model import ARIMA as ARIMA_model
        from arch import arch_model as arch_model_fn

        test_raw = test['Return_raw'].values
        full_ret_train_val = pd.concat([train['Return'], val['Return']])

        # ─────────────────────────────────────────────────────────────────────
        # MODEL 1: ARIMA — One-Step Ahead
        # ─────────────────────────────────────────────────────────────────────
        st.subheader("1️⃣ ARIMA — One-Step Ahead")
        prog1 = st.progress(0, "Forecasting ARIMA...")
        forecast_arima = []
        arima_history  = list(full_ret_train_val.values)
        for i in range(test_steps):
            m   = ARIMA_model(arima_history, order=best_arima_order).fit()
            fc  = m.forecast(steps=1)
            forecast_arima.append(float(fc.iloc[0]) if hasattr(fc, 'iloc') else float(fc[0]))
            arima_history.append(test['Return'].values[i])
            prog1.progress(int((i+1)/test_steps*100))
        forecast_arima = np.array(forecast_arima)
        m_arima = calc_metrics(test_raw, forecast_arima)
        st.success(f"ARIMA selesai — RMSE={m_arima['RMSE']:.6f}  DA={m_arima['DA']:.1f}%")
        st.dataframe(pd.DataFrame([{
            'RMSE': round(m_arima['RMSE'], 6), 'MAE': round(m_arima['MAE'], 6),
            'DA(%)': round(m_arima['DA'], 2), 'Theil U': round(m_arima['TheilU'], 4),
            'MAPE(%)': round(m_arima['MAPE'], 4),
        }]), use_container_width=True)
        fig_a1, ax_a1 = plt.subplots(figsize=(13, 4))
        ax_a1.plot(test.index, test_raw, color='black', lw=1.5, label='Aktual')
        ax_a1.plot(test.index, forecast_arima, color='steelblue', lw=1, ls='--', alpha=0.85,
                   label=f'ARIMA{best_arima_order}')
        ax_a1.axhline(0, color='k', lw=0.4, ls='--')
        ax_a1.set_title(f'Forecast ARIMA{best_arima_order} vs Aktual (Test Set)', fontweight='bold')
        ax_a1.set_ylabel('Log Return'); ax_a1.set_xlabel('Tanggal')
        ax_a1.legend(fontsize=9)
        plt.tight_layout(); st.pyplot(fig_a1)

        # ─────────────────────────────────────────────────────────────────────
        # MODEL 2: ARIMA-GARCH — One-Step Ahead
        # ─────────────────────────────────────────────────────────────────────
        st.subheader("2️⃣ ARIMA-GARCH — One-Step Ahead")
        prog2 = st.progress(0, "Forecasting ARIMA-GARCH...")
        forecast_ag = []
        ag_history  = list(full_ret_train_val.values)
        for i in range(test_steps):
            a_t    = ARIMA_model(ag_history, order=best_arima_order).fit()
            fc_ag  = a_t.forecast(steps=1)
            a_pred = float(fc_ag.iloc[0]) if hasattr(fc_ag, 'iloc') else float(fc_ag[0])
            r_t    = a_t.resid * 100
            g_t    = arch_model_fn(r_t, vol='Garch',
                                   p=best_garch_order[0], q=best_garch_order[1],
                                   dist='normal', rescale=False).fit(disp='off')
            vol_last  = float(g_t.conditional_volatility[-1]) / 100
            sign_last = np.sign(float(a_t.resid[-1]))
            forecast_ag.append(a_pred + sign_last * vol_last * 0.8)
            ag_history.append(test['Return'].values[i])
            prog2.progress(int((i+1)/test_steps*100))
        forecast_ag = np.array(forecast_ag)
        m_ag = calc_metrics(test_raw, forecast_ag)
        st.success(f"ARIMA-GARCH selesai — RMSE={m_ag['RMSE']:.6f}  DA={m_ag['DA']:.1f}%")
        st.dataframe(pd.DataFrame([{
            'RMSE': round(m_ag['RMSE'], 6), 'MAE': round(m_ag['MAE'], 6),
            'DA(%)': round(m_ag['DA'], 2), 'Theil U': round(m_ag['TheilU'], 4),
            'MAPE(%)': round(m_ag['MAPE'], 4),
        }]), use_container_width=True)
        fig_ag1, ax_ag1 = plt.subplots(figsize=(13, 4))
        ax_ag1.plot(test.index, test_raw, color='black', lw=1.5, label='Aktual')
        ax_ag1.plot(test.index, forecast_ag, color='darkorange', lw=1, ls='--', alpha=0.85,
                    label=f'ARIMA{best_arima_order}-GARCH{best_garch_order}')
        ax_ag1.axhline(0, color='k', lw=0.4, ls='--')
        ax_ag1.set_title(f'Forecast ARIMA{best_arima_order}-GARCH{best_garch_order} vs Aktual (Test Set)', fontweight='bold')
        ax_ag1.set_ylabel('Log Return'); ax_ag1.set_xlabel('Tanggal')
        ax_ag1.legend(fontsize=9)
        plt.tight_layout(); st.pyplot(fig_ag1)

        # ── Simpan semua ke session state ──────────────────────────────────────
        st.session_state.update({
            'forecast_arima': forecast_arima, 'm_arima': m_arima,
            'forecast_ag':    forecast_ag,    'm_ag':    m_ag,
            'test_raw':       test_raw,
        })
        st.balloons()
        st.success("✅ Semua model selesai! Lanjut ke halaman **📋 Evaluasi & Perbandingan**.")

    elif 'forecast_arima' in st.session_state:
        forecast_arima = st.session_state['forecast_arima']
        forecast_ag    = st.session_state['forecast_ag']
        m_arima  = st.session_state['m_arima']
        m_ag     = st.session_state['m_ag']
        test_raw = st.session_state['test_raw']

        st.subheader("1️⃣ ARIMA — Hasil Forecast")
        st.dataframe(pd.DataFrame([{
            'RMSE': round(m_arima['RMSE'], 6), 'MAE': round(m_arima['MAE'], 6),
            'DA(%)': round(m_arima['DA'], 2), 'Theil U': round(m_arima['TheilU'], 4),
            'MAPE(%)': round(m_arima['MAPE'], 4),
        }]), use_container_width=True)
        fig_a2, ax_a2 = plt.subplots(figsize=(13, 4))
        ax_a2.plot(test.index, test_raw, color='black', lw=1.5, label='Aktual')
        ax_a2.plot(test.index, forecast_arima, color='steelblue', lw=1, ls='--', alpha=0.85,
                   label=f'ARIMA{best_arima_order}')
        ax_a2.axhline(0, color='k', lw=0.4, ls='--')
        ax_a2.set_title(f'Forecast ARIMA{best_arima_order} vs Aktual (Test Set)', fontweight='bold')
        ax_a2.set_ylabel('Log Return'); ax_a2.set_xlabel('Tanggal')
        ax_a2.legend(fontsize=9)
        plt.tight_layout(); st.pyplot(fig_a2)

        st.divider()

        st.subheader("2️⃣ ARIMA-GARCH — Hasil Forecast")
        st.dataframe(pd.DataFrame([{
            'RMSE': round(m_ag['RMSE'], 6), 'MAE': round(m_ag['MAE'], 6),
            'DA(%)': round(m_ag['DA'], 2), 'Theil U': round(m_ag['TheilU'], 4),
            'MAPE(%)': round(m_ag['MAPE'], 4),
        }]), use_container_width=True)
        fig_ag2, ax_ag2 = plt.subplots(figsize=(13, 4))
        ax_ag2.plot(test.index, test_raw, color='black', lw=1.5, label='Aktual')
        ax_ag2.plot(test.index, forecast_ag, color='darkorange', lw=1, ls='--', alpha=0.85,
                    label=f'ARIMA{best_arima_order}-GARCH{best_garch_order}')
        ax_ag2.axhline(0, color='k', lw=0.4, ls='--')
        ax_ag2.set_title(f'Forecast ARIMA{best_arima_order}-GARCH{best_garch_order} vs Aktual (Test Set)', fontweight='bold')
        ax_ag2.set_ylabel('Log Return'); ax_ag2.set_xlabel('Tanggal')
        ax_ag2.legend(fontsize=9)
        plt.tight_layout(); st.pyplot(fig_ag2)

        st.info("Buka **📋 Evaluasi & Perbandingan** untuk perbandingan lengkap semua model.")
    else:
        st.info("Klik tombol di atas untuk memulai training semua model secara otomatis.")


# =============================================================================
# HALAMAN 5 — Evaluasi & Perbandingan
# =============================================================================
elif selected_menu == "📋 Evaluasi & Perbandingan":
    st.header("📋 Evaluasi & Perbandingan Semua Model")

    if 'forecast_arima' not in st.session_state:
        st.warning("Belum ada hasil forecast. Jalankan **🤖 Train & Forecast** terlebih dahulu.")
        st.stop()

    best_arima_order = st.session_state['best_arima_order']
    best_garch_order = st.session_state['best_garch_order']
    forecast_arima = st.session_state['forecast_arima']
    forecast_ag    = st.session_state['forecast_ag']
    m_arima = st.session_state['m_arima']
    m_ag    = st.session_state['m_ag']
    test_raw = st.session_state['test_raw']

    # ── Tabel ringkasan ────────────────────────────────────────────────────────
    st.subheader("📊 Tabel Perbandingan Metrik (Return)")
    model_labels = {
        f'ARIMA{best_arima_order}': m_arima,
        f'ARIMA{best_arima_order}-GARCH{best_garch_order}': m_ag,
    }
    rows = [{'Model': k,
             'RMSE':    round(v['RMSE'],   6),
             'MAE':     round(v['MAE'],    6),
             'DA(%)':   round(v['DA'],     2),
             'Theil U': round(v['TheilU'], 4),
             'MAPE(%)': round(v['MAPE'],   4)}
            for k, v in model_labels.items()]
    cmp_df = pd.DataFrame(rows).sort_values('RMSE').reset_index(drop=True)
    cmp_df.index += 1
    st.dataframe(cmp_df, use_container_width=True)
    st.caption("Diurutkan berdasarkan RMSE terkecil. MAPE tidak direkomendasikan untuk log-return (pembagi mendekati 0).")

    # ── Bar chart metrik ───────────────────────────────────────────────────────
    fig_bar, axes_b = plt.subplots(1, 3, figsize=(14, 4))
    pals = ['gold' if i == 0 else 'steelblue' for i in range(len(cmp_df))]
    names_sorted = cmp_df['Model'].values
    for ax_b, met in zip(axes_b, ['RMSE', 'MAE', 'DA(%)']):
        vals = cmp_df[met].values
        bars = ax_b.bar(range(len(names_sorted)), vals, color=pals, edgecolor='white')
        ax_b.set_xticks(range(len(names_sorted)))
        ax_b.set_xticklabels(names_sorted, rotation=25, ha='right', fontsize=8)
        ax_b.set_title(met, fontweight='bold')
        for bar, v in zip(bars, vals):
            ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                      f'{v:.5f}' if met != 'DA(%)' else f'{v:.2f}%',
                      ha='center', fontsize=7)
    plt.suptitle('Perbandingan Metrik (emas = terbaik berdasarkan RMSE)', fontweight='bold')
    plt.tight_layout(); st.pyplot(fig_bar)

    st.divider()

    # ── Plot gabungan forecast ─────────────────────────────────────────────────
    st.subheader("📉 Forecast Semua Model vs Aktual")
    all_forecasts = {
        f'ARIMA{best_arima_order}': (forecast_arima, 'steelblue'),
        f'ARIMA{best_arima_order}-GARCH{best_garch_order}': (forecast_ag, 'darkorange'),
    }
    fig_fc, ax_fc = plt.subplots(figsize=(15, 5))
    ax_fc.plot(test.index, test_raw, color='black', lw=2, label='Aktual', zorder=5)
    ls_list = ['-', '--']
    for idx, (name, (fc, col)) in enumerate(all_forecasts.items()):
        m = model_labels[name]
        ax_fc.plot(test.index, fc, color=col, lw=1.2, ls=ls_list[idx], alpha=0.85,
                   label=f"{name} (RMSE={m['RMSE']:.5f} DA={m['DA']:.1f}%)")
    ax_fc.axhline(0, color='k', lw=0.4, ls='--')
    ax_fc.set_title('Perbandingan Forecast Semua Model', fontweight='bold')
    ax_fc.set_ylabel('Log Return'); ax_fc.set_xlabel('Tanggal')
    ax_fc.legend(fontsize=8, loc='upper left')
    plt.tight_layout(); st.pyplot(fig_fc)

    st.divider()

    # ── Plot individual + scatter ──────────────────────────────────────────────
    st.subheader("📈 Detail per Model")
    for name, (fc, col) in all_forecasts.items():
        m = model_labels[name]
        fig_m, axes_m = plt.subplots(1, 3, figsize=(15, 4))
        axes_m[0].plot(test.index, test_raw, color='black', lw=1.2, label='Aktual')
        axes_m[0].plot(test.index, fc, color=col, lw=1, ls='--', alpha=0.85, label='Prediksi')
        axes_m[0].axhline(0, color='k', lw=0.4, ls='--')
        axes_m[0].set_title('Aktual vs Prediksi', fontweight='bold'); axes_m[0].legend(fontsize=8)

        err = test_raw - fc
        axes_m[1].bar(test.index, err, color=col, alpha=0.5, width=1)
        axes_m[1].axhline(0, color='red', lw=1, ls='--')
        axes_m[1].set_title(f'Residual | MAE={m["MAE"]:.5f}', fontweight='bold')

        lim  = max(abs(test_raw).max(), abs(fc).max()) * 1.15
        corr = np.sign(test_raw) == np.sign(fc)
        axes_m[2].scatter(test_raw[corr],  fc[corr],  alpha=0.4, s=8, color=col,   label=f'Benar ({corr.sum()})')
        axes_m[2].scatter(test_raw[~corr], fc[~corr], alpha=0.4, s=8, color='gray', label=f'Salah ({(~corr).sum()})')
        axes_m[2].plot([-lim, lim], [-lim, lim], 'r--', lw=1.2, label='Perfect')
        axes_m[2].set_xlim(-lim, lim); axes_m[2].set_ylim(-lim, lim)
        axes_m[2].set_xlabel('Aktual'); axes_m[2].set_ylabel('Prediksi')
        axes_m[2].legend(fontsize=8); axes_m[2].set_title('Scatter', fontweight='bold')
        plt.suptitle(f'{name} | RMSE={m["RMSE"]:.5f}  DA={m["DA"]:.1f}%  Theil_U={m["TheilU"]:.4f}',
                     fontweight='bold'); plt.tight_layout()
        st.pyplot(fig_m)

    st.divider()

    # ── Konversi ke harga & tabel harga ───────────────────────────────────────
    st.subheader("💰 Konversi Prediksi Return → Harga")
    st.latex(r"\hat{P}_t = P_{t-1} \times e^{\hat{r}_t}")

    test_start_idx  = data.index.get_loc(test.index[0])
    price_prev      = data['Close'].values[test_start_idx - 1: test_start_idx - 1 + test_steps]
    price_actual    = test['Close'].values

    def price_metrics(p_true, p_pred):
        rmse_p  = np.sqrt(np.mean((p_pred - p_true)**2))
        mae_p   = np.mean(np.abs(p_pred - p_true))
        mape_p  = np.mean(np.abs((p_true - p_pred) / p_true)) * 100
        theil_p = np.sqrt(np.mean((p_pred - p_true)**2)) / np.sqrt(np.mean(p_true**2))
        return dict(RMSE=rmse_p, MAE=mae_p, MAPE=mape_p, TheilU=theil_p)

    price_fc = {}
    for name, (fc, _) in all_forecasts.items():
        price_fc[name] = price_prev * np.exp(fc)
    price_naive = price_prev.copy()

    rows_p = []
    for name, pp in price_fc.items():
        mp = price_metrics(price_actual, pp)
        rows_p.append({'Model': name, 'RMSE(Rp)': round(mp['RMSE'], 2),
                       'MAE(Rp)': round(mp['MAE'], 2), 'MAPE(%)': round(mp['MAPE'], 4),
                       'Theil U': round(mp['TheilU'], 4)})
    mp_naive = price_metrics(price_actual, price_naive)
    rows_p.append({'Model': 'Naive (P_{t-1})', 'RMSE(Rp)': round(mp_naive['RMSE'], 2),
                   'MAE(Rp)': round(mp_naive['MAE'], 2), 'MAPE(%)': round(mp_naive['MAPE'], 4),
                   'Theil U': round(mp_naive['TheilU'], 4)})
    price_cmp_df = pd.DataFrame(rows_p).sort_values('RMSE(Rp)').reset_index(drop=True)
    price_cmp_df.index += 1
    st.dataframe(price_cmp_df, use_container_width=True)

    # Plot harga gabungan
    fig_ph, ax_ph = plt.subplots(figsize=(15, 5))
    ax_ph.plot(test.index, price_actual, color='black', lw=2, label='Harga Aktual', zorder=5)
    _pcols = ['steelblue', 'darkorange']
    for (name, pp), col in zip(price_fc.items(), _pcols):
        ax_ph.plot(test.index, pp, color=col, lw=1.2, ls='--', alpha=0.85, label=name)
    ax_ph.set_title('Prediksi Harga BBCA — Semua Model (Test Set)', fontweight='bold')
    ax_ph.set_ylabel('Harga (IDR)'); ax_ph.set_xlabel('Tanggal')
    ax_ph.legend(fontsize=8)
    ax_ph.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'Rp{x:,.0f}'))
    plt.tight_layout(); st.pyplot(fig_ph)

    # ── Export ─────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("💾 Export Hasil")
    export_df = pd.DataFrame({'Date': test.index.strftime('%Y-%m-%d'),
                               'Actual_Return': test_raw,
                               'ARIMA': forecast_arima,
                               'ARIMA_GARCH': forecast_ag})
    st.download_button("⬇️ Download Forecast Return (CSV)",
                       export_df.to_csv(index=False).encode(),
                       "forecast_return_results.csv", "text/csv")
