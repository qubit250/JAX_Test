"""JAX: ティフトン芝ソッドのスタッド滑り抵抗の微分可能モデル (繊維束+張力要素)。

直径 8mm の円柱スタッドを深さ 10mm まで押し込み、横方向に 66.7 mm/s で
滑らせたときの滑り抵抗 F(t) をモデル化する。実測の特徴量:

    - t = 0.2 s  でピーク 120 N
    - t = 1.5 s  で残留値 30 N まで緩和

ソッドを「砂マトリクス + 匍匐茎(ストロン)ネットワーク」の複合材とみなし、
抵抗力を3成分の和で表す:

    F(delta) = F_sand + F_cohort + F_sweep      (delta = v*t: スタッド変位)

1. F_sand   : 砂の支圧抵抗。弾性→完全塑性を tanh で平滑化した飽和曲線。
2. F_cohort : 試験開始時にスタッド周囲で係合している匍匐茎群 (初期コホート)。
              各匍匐茎は節(発根点)間をアンカーとする張力要素で、クリンプ
              (うねり) delta_c が伸び切ってから線形に張力を発揮し、
              Weibull 分布に従う伸びで破断/引き抜けする (繊維束モデル)。
              ピーク荷重と、その後の緩和 (コホートの逐次破断) を担う。
3. F_sweep  : スタッドが掃引しながら新たに係合する匍匐茎の定常寄与。
              残留抵抗のうち砂で説明できない分を担う。

モデルは JAX で微分可能に実装し、実測特徴量 (ピーク値・ピーク時刻・残留値)
への勾配ベースのフィッティング (Adam) でパラメータを同定する。
最後にピーク荷重の各パラメータに対する感度 (対数微分) も勾配で評価する。

単位系: 力 [N], 長さ [mm], 時間 [s]。
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

# 日本語ラベル用フォント: 環境にあるものを先頭から採用する
from matplotlib import font_manager as _fm
_JP_FONTS = ["IPAPGothic", "IPAGothic", "Noto Sans CJK JP", "Hiragino Sans",
             "Yu Gothic", "Meiryo", "TakaoPGothic"]
_avail = {f.name for f in _fm.fontManager.ttflist}
plt.rcParams["font.family"] = [f for f in _JP_FONTS if f in _avail] + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# 試験条件
# ---------------------------------------------------------------------------
V_SLIDE = 66.7        # 滑り速度 [mm/s]
STUD_D = 8.0          # スタッド径 [mm]
STUD_DEPTH = 10.0     # 貫入深さ [mm]
T_END = 1.5           # 解析時間 [s]
N_T = 1501            # 時間格子点数

# 実測 (フィット目標) の特徴量
T_PEAK_OBS = 0.2      # ピーク時刻 [s]
F_PEAK_OBS = 120.0    # ピーク荷重 [N]
F_RESID_OBS = 30.0    # t=1.5s の残留荷重 [N]
# ピーク後は「1.5s かけて 30N まで緩和」— 指数型テンプレートで形状を与える
TAU_RELAX = (T_END - T_PEAK_OBS) / 3.0   # 緩和時定数 [s] (t=1.5s で約95%緩和)

# ---------------------------------------------------------------------------
# モデルパラメータ (フィット初期値)。正値制約のため log 空間で最適化する。
# ---------------------------------------------------------------------------
PARAM_NAMES = ["k_s", "F_sand_res", "A_coh", "delta_c", "s_b", "m_w", "lam_k"]
THETA0 = {
    "k_s": 3.0,         # 砂の初期剛性 [N/mm]
    "F_sand_res": 20.0,  # 砂の残留 (飽和) 抵抗 [N]
    "A_coh": 13.5,       # 初期コホートの合成剛性 N0*k_f [N/mm]
    "delta_c": 4.0,      # クリンプ緩み (張力立ち上がりまでの遊び) [mm]
    "s_b": 13.0,         # Weibull 破断伸びスケール [mm]
    "m_w": 1.5,          # Weibull 形状係数 (小さいほど破断伸びのばらつき大) [-]
    "lam_k": 0.13,       # 掃引再係合の線密度×剛性 lambda*k_f [N/mm^2]
}


def survival(x, s_b, m_w):
    """クリンプ後伸び x に対する匍匐茎の生存率 (Weibull)。

    x=0 で jnp.power の勾配 (指数・底の両方向) が NaN/inf になるため
    微小値でガードする。
    """
    x_pos = jnp.maximum(x, 0.0) + 1e-9
    return jnp.exp(-jnp.power(x_pos / s_b, m_w))


def force_components(log_theta, t):
    """時刻配列 t に対する (F_sand, F_cohort, F_sweep) を返す。"""
    th = {k: jnp.exp(v) for k, v in zip(PARAM_NAMES, log_theta)}
    delta = V_SLIDE * t

    # 1. 砂: 弾性→完全塑性の平滑近似
    f_sand = th["F_sand_res"] * jnp.tanh(th["k_s"] * delta / th["F_sand_res"])

    # 2. 初期コホート: 張力 (クリンプ後の伸びに比例) × 生存率
    stretch = jnp.maximum(delta - th["delta_c"], 0.0)
    f_cohort = th["A_coh"] * stretch * survival(stretch, th["s_b"], th["m_w"])

    # 3. 掃引再係合: 係合位置 x で積分 (台形則の累積和)
    #    F_sweep(delta) = lam_k * int_0^delta (x - delta_c)_+ S(x) dx
    integrand = jnp.maximum(delta - th["delta_c"], 0.0) * survival(
        delta - th["delta_c"], th["s_b"], th["m_w"]
    )
    d_delta = delta[1] - delta[0]
    cumtrap = jnp.concatenate(
        [jnp.zeros(1), jnp.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * d_delta)]
    )
    f_sweep = th["lam_k"] * cumtrap

    return f_sand, f_cohort, f_sweep


def total_force(log_theta, t):
    return sum(force_components(log_theta, t))


# ---------------------------------------------------------------------------
# フィッティング: 特徴量損失 + 弱い対数事前分布 (物理的初期値への正則化)
# ---------------------------------------------------------------------------
T_GRID = jnp.linspace(0.0, T_END, N_T)
LOG_THETA0 = jnp.log(jnp.array([THETA0[k] for k in PARAM_NAMES]))


def loss(log_theta):
    f = total_force(log_theta, T_GRID)
    f_max = jnp.max(f)
    # ピーク時刻は softmax による soft-argmax で微分可能に評価
    w = jax.nn.softmax(f / 0.5)
    t_pk = jnp.sum(w * T_GRID)
    f_end = f[-1]
    # ピーク後の緩和形状: 指数型テンプレートとの残差 (ピーク前は重み 0)
    f_ref = F_RESID_OBS + (F_PEAK_OBS - F_RESID_OBS) * jnp.exp(
        -(T_GRID - T_PEAK_OBS) / TAU_RELAX
    )
    post = T_GRID > T_PEAK_OBS
    relax_err = jnp.sum(jnp.where(post, ((f - f_ref) / F_PEAK_OBS) ** 2, 0.0)) / jnp.sum(post)
    return (
        ((f_max - F_PEAK_OBS) / F_PEAK_OBS) ** 2
        + ((t_pk - T_PEAK_OBS) / T_PEAK_OBS) ** 2
        + ((f_end - F_RESID_OBS) / F_RESID_OBS) ** 2
        + relax_err
        + 0.003 * jnp.mean((log_theta - LOG_THETA0) ** 2)
    )


@jax.jit
def adam_step(log_theta, m, v, i, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
    g = jax.grad(loss)(log_theta)
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g**2
    m_hat = m / (1 - b1**i)
    v_hat = v / (1 - b2**i)
    return log_theta - lr * m_hat / (jnp.sqrt(v_hat) + eps), m, v


def fit(n_iter=4000):
    log_theta = LOG_THETA0
    m = jnp.zeros_like(log_theta)
    v = jnp.zeros_like(log_theta)
    for i in range(1, n_iter + 1):
        log_theta, m, v = adam_step(log_theta, m, v, i)
        if i % 1000 == 0:
            print(f"  iter {i:5d}  loss = {loss(log_theta):.3e}")
    return log_theta


def main():
    print("フィッティング開始 (Adam, 特徴量: ピーク値/ピーク時刻/残留値)")
    log_theta = fit()
    theta = {k: float(jnp.exp(v)) for k, v in zip(PARAM_NAMES, log_theta)}

    f_sand, f_coh, f_sweep = force_components(log_theta, T_GRID)
    f_tot = f_sand + f_coh + f_sweep
    i_pk = int(jnp.argmax(f_tot))
    t_pk, f_pk = float(T_GRID[i_pk]), float(f_tot[i_pk])
    f_end = float(f_tot[-1])

    print("\n同定パラメータ:")
    for k in PARAM_NAMES:
        print(f"  {k:10s} = {theta[k]:8.3f}")
    print("\nモデル応答:")
    print(f"  ピーク: {f_pk:6.1f} N @ t = {t_pk:.3f} s  (目標 {F_PEAK_OBS} N @ {T_PEAK_OBS} s)")
    print(f"  残留 : {f_end:6.1f} N @ t = {T_END} s    (目標 {F_RESID_OBS} N)")

    # 見かけのヤング率 (ピークまでの割線値): E = sigma / eps, eps = delta_pk / d
    delta_pk = V_SLIDE * t_pk
    sigma_pk = f_pk / (STUD_D * STUD_DEPTH)          # [N/mm^2 = MPa]
    e_app = sigma_pk * STUD_D / delta_pk
    print(f"  見かけのヤング率 (割線): {e_app:.2f} MPa")

    # ピーク荷重の感度: dF_peak / d(log theta) — 微分可能モデルの利点
    grad_pk = jax.grad(lambda lt: jnp.max(total_force(lt, T_GRID)))(log_theta)
    print("\nピーク荷重の対数感度 dF_peak/dlog(theta) [N]:")
    for k, g in zip(PARAM_NAMES, grad_pk):
        print(f"  {k:10s} : {float(g):+7.1f}")

    plot(np.asarray(T_GRID), np.asarray(f_tot), np.asarray(f_sand),
         np.asarray(f_coh), np.asarray(f_sweep))


def plot(t, f_tot, f_sand, f_coh, f_sweep):
    surface, ink, ink2 = "#fcfcfb", "#0b0b0b", "#52514e"
    c_tot, c_sand, c_coh, c_sweep = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    ax.plot(t, f_tot, color=c_tot, lw=2.2, label="合計 F(t)", zorder=4)
    ax.plot(t, f_coh, color=c_coh, lw=1.6, label="匍匐茎 初期コホート", zorder=3)
    ax.plot(t, f_sand, color=c_sand, lw=1.6, label="砂 (支圧)", zorder=3)
    ax.plot(t, f_sweep, color=c_sweep, lw=1.6, label="匍匐茎 掃引再係合", zorder=3)

    # 実測の特徴量 (フィット目標)
    ax.scatter([T_PEAK_OBS, T_END], [120.0, 30.0], s=55, facecolor=surface,
               edgecolor=ink2, lw=1.6, zorder=5)
    ax.annotate("実測ピーク 120 N @ 0.2 s", (T_PEAK_OBS, 120.0),
                xytext=(10, 6), textcoords="offset points", color=ink2, fontsize=9)
    ax.annotate("残留 30 N @ 1.5 s", (T_END, 30.0),
                xytext=(-8, 10), textcoords="offset points", ha="right",
                color=ink2, fontsize=9)

    # 系列の直接ラベル (線の終端付近)
    for y, label, c in [(f_tot[-1], "合計", c_tot), (f_sand[-1], "砂", c_sand),
                        (f_sweep[-1] - 1.5, "再係合", c_sweep)]:
        ax.annotate(label, (t[-1], y), xytext=(4, 0), textcoords="offset points",
                    color=c, fontsize=9, va="center")
    i_c = int(np.argmax(f_coh))
    ax.annotate("初期コホート", (t[i_c], f_coh[i_c]), xytext=(8, 4),
                textcoords="offset points", color=c_coh, fontsize=9)

    ax.set_xlabel("時間 t [s]", color=ink)
    ax.set_ylabel("滑り抵抗 F [N]", color=ink)
    ax.set_title("スタッド滑り抵抗の複合材モデル (砂 + ティフトン匍匐茎ネットワーク)",
                 color=ink, fontsize=11)
    ax.set_xlim(0, T_END * 1.06)
    ax.set_ylim(0, 135)
    ax.grid(color="#e8e7e3", lw=0.7)
    ax.tick_params(colors=ink2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(ink2)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=ink)

    fig.tight_layout()
    fig.savefig("turf_stolon_traction.png", facecolor=surface)
    print("\n図を turf_stolon_traction.png に保存しました")


if __name__ == "__main__":
    main()
