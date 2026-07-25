"""
変分パルス制御 — 2-qubit JT モデル (PennyLane 自動微分)
=========================================================
H(t) = ε/2·ZI + ω/2·IZ + g(t)·XX

時間依存結合 g(t) を区分定数でパラメータ化し、
PennyLane の Adam 最適化器で P(|11>) を最大化する。

物理的意義:
  ε=−ω (CI seam): H_eff = g(t)·σ_x のみ → pi pulse 条件 int g dt = pi/2 で P=1
  ε≠−ω (off-res): 定数 g では P_max=(g/Ω₁)^2 < 1 に制限
                  変分 g(t) は量子制御定理で P=1 を達成可能
                  ([σ_z, σ_x] = −2i σ_y → Lie 代数が完全)

スキャン:
  seam    : ε=−1 (CI seam) — 定数 g* vs 変分 g(t)
  off_res : ε=−0.5, 0.0 — 非共鳴での劇的改善を実証
  compare : ε スキャン — 変分制御の優位性マップ

IBM モード:
  シミュレータで最適化 → 最適パルスを IBM 実機で評価 (1 circuit/点)
  (最適化ループを IBM で回すと CNOT 数 × iter = 数万回路 → 非現実的)

使い方:
  python pennylane_variational_pulse_2q.py [simulate|ibm_real] [seam|off_res|compare] [TOKEN]
"""
import sys
import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── パラメータ ────────────────────────────────────────────────
OMEGA       = 1.0
T_FINAL     = 5.0
N_STEPS_OPT = 20     # 最適化 Trotter 数 (dt=0.25, CNOT=40)
N_STEPS_IBM = 10     # IBM 評価 Trotter 数 (dt=0.5,  CNOT=20)
N_ITER      = 300    # Adam 反復数
STEP_SIZE   = 0.05   # Adam 学習率
G_STAR      = np.pi / (2.0 * T_FINAL)  # ≈ 0.3142 (resonance pi-pulse)
G_INIT      = G_STAR
SHOTS       = 8192

EPS_COMPARE = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
EPS_OFF_RES = np.array([-0.5, 0.0])


# ── 解析関数 ──────────────────────────────────────────────────

def rabi_freq(eps: float, g: float) -> float:
    return float(np.sqrt(((eps + OMEGA) / 2.0) ** 2 + g ** 2))


def p_analytic(eps: float, g: float, T: float = T_FINAL) -> float:
    o1 = rabi_freq(eps, g)
    return float((g / o1) ** 2 * np.sin(o1 * T) ** 2) if o1 > 1e-10 else 0.0


def p_max_const(eps: float, g: float) -> float:
    o1 = rabi_freq(eps, g)
    return float((g / o1) ** 2) if o1 > 1e-10 else 0.0


# ── 変分回路と損失関数 ────────────────────────────────────────

def make_cost_fn(eps: float, n_steps: int):
    """区分定数 g(t)=[g_0,...,g_{N-1}] の損失 1−P(|11>) を返す。"""
    dt  = T_FINAL / n_steps
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="autograd")
    def circuit(g_params):
        for k in range(n_steps):
            qml.RZ(float(eps) * dt,   wires=0)
            qml.RZ(float(OMEGA) * dt, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RX(2.0 * g_params[k] * dt, wires=0)
            qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    def cost(g_params):
        return 1.0 - circuit(g_params)[3]

    return cost


def eval_fixed(eps: float, g_arr: np.ndarray, n_steps: int,
               mode: str, backend=None) -> float:
    """固定 g_arr で回路を評価 (勾配不要 / IBM 実機評価用)。"""
    dt = T_FINAL / n_steps
    if mode == "ibm_real":
        dev = qml.device("qiskit.remote", wires=2, backend=backend)
    else:
        dev = qml.device("default.qubit", wires=2)

    @qml.set_shots(SHOTS if mode != "simulate" else None)
    @qml.qnode(dev)
    def circuit():
        for k in range(n_steps):
            qml.RZ(float(eps) * dt,          wires=0)
            qml.RZ(float(OMEGA) * dt,         wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RX(float(2.0 * g_arr[k]) * dt, wires=0)
            qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    return float(circuit()[3])


# ── Adam 最適化 ───────────────────────────────────────────────

def optimize_pulse(eps: float, n_steps: int = N_STEPS_OPT,
                   n_iter: int = N_ITER, verbose: bool = True):
    """Adam で区分定数 g(t) を最適化。(g_opt, P_opt, history) を返す。"""
    cost   = make_cost_fn(eps, n_steps)
    params = pnp.ones(n_steps, requires_grad=True) * G_INIT
    opt    = qml.AdamOptimizer(stepsize=STEP_SIZE)
    history = []

    for step in range(n_iter):
        params, val = opt.step_and_cost(cost, params)
        P = 1.0 - float(val)
        history.append(P)
        if verbose and (step % 100 == 0 or step == n_iter - 1):
            print(f"    iter {step:3d}: P = {P:.4f}")

    g_opt = np.array(params)
    P_opt = eval_fixed(eps, g_opt, n_steps, "simulate")
    return g_opt, P_opt, history


def downsample(g_opt: np.ndarray, n_target: int) -> np.ndarray:
    """最適パルスを n_target 点に線形補間 (IBM 用)。"""
    return np.interp(
        np.linspace(0, 1, n_target),
        np.linspace(0, 1, len(g_opt)),
        g_opt,
    )


# ── プロット ──────────────────────────────────────────────────

def _t_mids(n_steps):
    return (np.arange(n_steps) + 0.5) * T_FINAL / n_steps


def plot_seam(eps, g_opt, P_const_c, P_opt_sim, history, ibm_data=None):
    t = _t_mids(N_STEPS_OPT)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.step(t, np.ones(N_STEPS_OPT) * G_STAR, where='mid',
            color='steelblue', lw=2, label=f"Const g*={G_STAR:.3f}")
    ax.step(t, g_opt, where='mid', color='firebrick', lw=2, ls='--',
            label="Optimized g(t)")
    ax.set_xlabel("t");  ax.set_ylabel("g(t)")
    ax.set_title(f"Pulse shape (ε={eps})")
    ax.legend(fontsize=9);  ax.set_xlim(0, T_FINAL);  ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(history, color='firebrick', lw=2)
    ax.axhline(P_const_c, color='steelblue', ls='--', lw=2,
               label=f"Const g* circuit P={P_const_c:.4f}")
    ax.axhline(1.0, color='gray', ls=':', lw=1, alpha=0.5)
    ax.set_xlabel("Iteration");  ax.set_ylabel("P(|11>)")
    ax.set_title("Optimization convergence")
    ax.legend(fontsize=9)
    ax.set_ylim(max(0.7, min(history) - 0.05), 1.02)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    labels = ["Analytic\n(const g*)", "Circuit\n(const g*)", "Optimized\n(sim)"]
    values = [p_analytic(eps, G_STAR), P_const_c, P_opt_sim]
    colors = ['navy', 'steelblue', 'firebrick']
    if ibm_data:
        labels += ["Const\n(IBM)", "Optimized\n(IBM)"]
        values += [ibm_data['P_const'], ibm_data['P_opt']]
        colors += ['royalblue', 'orangered']
    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='black')
    ax.axhline(1.0, color='gray', ls='--', lw=1.5)
    ax.set_ylabel("P(|11>)");  ax.set_ylim(0, 1.15)
    ax.set_title(f"Summary (ε={eps})")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(
        f"Variational Pulse — CI Seam (ε={eps}, T={T_FINAL}, N={N_STEPS_OPT})", fontsize=12)
    plt.tight_layout()
    fname = "variational_pulse_seam.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight");  plt.close()
    print(f"\n  -> Plot saved: {fname}")


def plot_off_res(results):
    eps_list = sorted(results.keys())
    n = len(eps_list)
    t = _t_mids(N_STEPS_OPT)
    fig, axes = plt.subplots(2, n, figsize=(7 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, eps in enumerate(eps_list):
        r = results[eps]

        ax = axes[0, col]
        ax.step(t, np.ones(N_STEPS_OPT) * G_STAR, where='mid',
                color='steelblue', lw=2, label=f"Const g*={G_STAR:.3f}")
        ax.step(t, r['g_opt'], where='mid', color='firebrick', lw=2, ls='--',
                label="Optimized g(t)")
        ax.axhline(0, color='gray', ls=':', alpha=0.3)
        ax.set_xlabel("t");  ax.set_ylabel("g(t)")
        ax.set_title(f"Pulse shape (ε={eps:+.1f})")
        ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        ax.plot(r['history'], color='firebrick', lw=2, label=f"P_opt={r['P_opt']:.3f}")
        ax.axhline(r['P_const'], color='steelblue', ls='--', lw=2,
                   label=f"Const g*: {r['P_const']:.3f}")
        ax.axhline(r['P_max'],  color='darkorange', ls=':', lw=2,
                   label=f"P_max(const): {r['P_max']:.3f}")
        ax.set_xlabel("Iteration");  ax.set_ylabel("P(|11>)")
        ax.set_title(f"Convergence (ε={eps:+.1f})")
        ax.legend(fontsize=8);  ax.set_ylim(-0.05, 1.05);  ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Variational Pulse — Off-Resonance (T={T_FINAL}, N={N_STEPS_OPT})", fontsize=12)
    plt.tight_layout()
    fname = "variational_pulse_off_res.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight");  plt.close()
    print(f"\n  -> Plot saved: {fname}")


def plot_compare(results):
    eps_vals = sorted(results.keys())
    P_analy = [results[e]['P_analy'] for e in eps_vals]
    P_const = [results[e]['P_const'] for e in eps_vals]
    P_opt   = [results[e]['P_opt']   for e in eps_vals]
    P_max_v = [results[e]['P_max']   for e in eps_vals]
    delta   = [P_opt[i] - P_const[i] for i in range(len(eps_vals))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(eps_vals, P_analy, 'b--o', lw=2, ms=7, label="Analytic (const g*)")
    ax.plot(eps_vals, P_const, 'bs',   lw=2, ms=7, label="Circuit (const g*)")
    ax.plot(eps_vals, P_opt,   'r^',   lw=2, ms=9, label="Optimized g(t)")
    ax.plot(eps_vals, P_max_v, 'g:',   lw=1.5, label="P_max limit (const g)")
    ax.fill_between(eps_vals, P_const, P_opt, alpha=0.15, color='red')
    ax.axvline(-1.0, color='gray', ls=':', lw=1, alpha=0.5, label="CI seam")
    ax.set_xlabel(r"$\varepsilon$");  ax.set_ylabel("P(|11>)")
    ax.set_title(f"Variational vs constant g (T={T_FINAL})")
    ax.legend(fontsize=9);  ax.set_ylim(-0.05, 1.1);  ax.grid(True, alpha=0.3)

    ax = axes[1]
    clrs = ['firebrick' if d > 0 else 'steelblue' for d in delta]
    bars = ax.bar(eps_vals, delta, color=clrs, alpha=0.8,
                  edgecolor='black', width=0.25)
    ax.axhline(0, color='black', lw=1)
    ax.axvline(-1.0, color='gray', ls=':', lw=1, alpha=0.5, label="CI seam")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$\Delta P = P_{opt} - P_{const}$")
    ax.set_title("Gain from variational control")
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, delta):
        offset = 0.005 if val >= 0 else -0.005
        va = 'bottom' if val >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                f"{val:+.3f}", ha='center', va=va, fontsize=8)

    plt.suptitle(
        f"Variational Pulse — ε Scan (T={T_FINAL}, N={N_STEPS_OPT}, iter={N_ITER})",
        fontsize=12)
    plt.tight_layout()
    fname = "variational_pulse_compare.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight");  plt.close()
    print(f"\n  -> Plot saved: {fname}")


# ── スキャン実行 ──────────────────────────────────────────────

def run_scan(scan_type: str = "seam", mode: str = "simulate", token=None):
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  変分パルス制御  (2-qubit JT)  —  {scan_type} スキャン")
    print(f"  H(t) = ε/2·ZI + ω/2·IZ + g(t)·XX,  ω={OMEGA}, T={T_FINAL}")
    print(f"  g(t): 区分定数 N={N_STEPS_OPT} ステップ (dt={T_FINAL/N_STEPS_OPT:.3f})")
    print(f"  最適化: Adam lr={STEP_SIZE}, iter={N_ITER}, g_init=G*={G_STAR:.4f}")
    print(sep)

    backend = None
    if mode == "ibm_real":
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        backend = service.least_busy(
            operational=True, simulator=False, min_num_qubits=2)
        print(f"\n  バックエンド: {backend.name}")

    # ── seam ─────────────────────────────────────────────────
    if scan_type == "seam":
        eps = -1.0
        print(f"\n  [ε={eps} (CI seam): 変分 g(t) vs 定数 g*={G_STAR:.4f}]")
        print(f"  解析: ε=−1 → P = sin^2(int g dt); 定数 g* で P=1.000 (厳密)")

        g_const    = np.ones(N_STEPS_OPT) * G_STAR
        P_const_a  = p_analytic(eps, G_STAR)
        P_const_c  = eval_fixed(eps, g_const, N_STEPS_OPT, "simulate")
        print(f"\n  定数 g* = {G_STAR:.4f}:")
        print(f"    P_analytic = {P_const_a:.4f}")
        print(f"    P_circuit  = {P_const_c:.4f}  (Trotter 誤差 {(P_const_a-P_const_c)*100:.2f}%)")

        print(f"\n  Adam 最適化中 (N={N_STEPS_OPT}, iter={N_ITER})...")
        g_opt, P_opt_sim, history = optimize_pulse(
            eps, N_STEPS_OPT, N_ITER, verbose=True)

        print(f"\n  最適化結果:")
        print(f"    P_opt_sim = {P_opt_sim:.4f}  (vs P_const_c={P_const_c:.4f})")
        print(f"    ΔP = {P_opt_sim - P_const_c:+.4f}")
        print(f"    Trotter 誤差回復: "
              f"{min((P_opt_sim-P_const_c)/(P_const_a-P_const_c+1e-10)*100,100):.1f}%")

        ibm_data = None
        if mode == "ibm_real":
            print(f"\n  IBM 実機評価 (N={N_STEPS_IBM}, 各 1 回路)...")
            g_ibm_c = np.ones(N_STEPS_IBM) * G_STAR
            g_ibm_o = downsample(g_opt, N_STEPS_IBM)
            P_ci    = eval_fixed(eps, g_ibm_c, N_STEPS_IBM, mode, backend)
            P_oi    = eval_fixed(eps, g_ibm_o, N_STEPS_IBM, mode, backend)
            print(f"    P_const_IBM  = {P_ci:.4f}")
            print(f"    P_opt_IBM    = {P_oi:.4f}")
            print(f"    IBM ΔP       = {P_oi - P_ci:+.4f}")
            ibm_data = {'P_const': P_ci, 'P_opt': P_oi}

        plot_seam(eps, g_opt, P_const_c, P_opt_sim, history, ibm_data)

    # ── off_res ───────────────────────────────────────────────
    elif scan_type == "off_res":
        print(f"\n  [非共鳴: 定数 g の P_max 制限を変分 g(t) で突破]")
        print(f"  量子制御定理: [σ_z, σ_x] = −2iσ_y → Lie 代数完全 → P=1 可能")
        print(f"\n  {'ε':>6}  {'P_max':>6}  {'P_const':>8}  "
              f"{'P_opt':>8}  {'ΔP':>8}  {'gain×':>6}")
        print("  " + "-" * 54)

        results = {}
        for eps in EPS_OFF_RES:
            eps = float(eps)
            P_max_v = p_max_const(eps, G_STAR)
            P_const = eval_fixed(eps, np.ones(N_STEPS_OPT)*G_STAR, N_STEPS_OPT, "simulate")
            print(f"\n  ε={eps:+.1f}: 最適化中 (P_max={P_max_v:.3f}, P_const={P_const:.3f})...")
            g_opt, P_opt, history = optimize_pulse(
                eps, N_STEPS_OPT, N_ITER, verbose=True)

            gain = P_opt / max(P_const, 1e-4)
            print(f"\n  {eps:>+6.1f}  {P_max_v:>6.3f}  {P_const:>8.4f}  "
                  f"{P_opt:>8.4f}  {P_opt-P_const:>+8.4f}  {gain:>6.1f}x")

            ibm = {}
            if mode == "ibm_real":
                print(f"    → IBM 評価中...")
                P_ci = eval_fixed(eps, np.ones(N_STEPS_IBM)*G_STAR,
                                  N_STEPS_IBM, mode, backend)
                P_oi = eval_fixed(eps, downsample(g_opt, N_STEPS_IBM),
                                  N_STEPS_IBM, mode, backend)
                print(f"    P_const_IBM={P_ci:.4f}, P_opt_IBM={P_oi:.4f}")
                ibm = {'P_const_ibm': P_ci, 'P_opt_ibm': P_oi}

            results[eps] = {
                'g_opt': g_opt, 'P_const': P_const, 'P_opt': P_opt,
                'P_max': P_max_v, 'history': history, **ibm,
            }

        print(f"\n  [知見]")
        for eps, r in sorted(results.items()):
            print(f"  ε={eps:+.1f}: P_const={r['P_const']:.3f} → P_opt={r['P_opt']:.3f}  "
                  f"(×{r['P_opt']/max(r['P_const'],1e-4):.1f}改善, "
                  f"P_max 上限={r['P_max']:.3f} {'超越' if r['P_opt']>r['P_max'] else '以下'})")
        plot_off_res(results)

    # ── compare ───────────────────────────────────────────────
    elif scan_type == "compare":
        print(f"\n  [ε スキャン: 変分 g(t) vs 定数 g*={G_STAR:.3f}]")
        print(f"\n  {'ε':>6}  {'P_analytic':>10}  {'P_const_c':>10}  "
              f"{'P_opt':>10}  {'ΔP':>8}  {'gain×':>6}")
        print("  " + "-" * 62)

        results = {}
        for eps in EPS_COMPARE:
            eps = float(eps)
            P_a   = p_analytic(eps, G_STAR)
            P_max = p_max_const(eps, G_STAR)
            P_c   = eval_fixed(eps, np.ones(N_STEPS_OPT)*G_STAR, N_STEPS_OPT, "simulate")
            g_opt, P_opt, history = optimize_pulse(
                eps, N_STEPS_OPT, N_ITER, verbose=False)
            gain = P_opt / max(P_c, 1e-4)
            print(f"  {eps:>+6.1f}  {P_a:>10.4f}  {P_c:>10.4f}  "
                  f"{P_opt:>10.4f}  {P_opt-P_c:>+8.4f}  {gain:>6.1f}x")

            ibm = {}
            if mode == "ibm_real":
                P_ci = eval_fixed(eps, np.ones(N_STEPS_IBM)*G_STAR,
                                  N_STEPS_IBM, mode, backend)
                P_oi = eval_fixed(eps, downsample(g_opt, N_STEPS_IBM),
                                  N_STEPS_IBM, mode, backend)
                print(f"    IBM: P_const={P_ci:.4f}, P_opt={P_oi:.4f}")
                ibm = {'P_const_ibm': P_ci, 'P_opt_ibm': P_oi}

            results[eps] = {
                'g_opt': g_opt, 'P_analy': P_a, 'P_const': P_c,
                'P_opt': P_opt, 'P_max': P_max, 'history': history, **ibm,
            }

    else:
        print(f"  エラー: scan_type は 'seam', 'off_res', 'compare' を指定")
        sys.exit(1)

    if scan_type == "compare":
        plot_compare(results)

    print(f"\n{sep}\n")


if __name__ == "__main__":
    mode      = sys.argv[1] if len(sys.argv) > 1 else "simulate"
    scan_type = sys.argv[2] if len(sys.argv) > 2 else "seam"
    token     = sys.argv[3] if len(sys.argv) > 3 else None

    if mode == "ibm_real" and token is None:
        print("使い方: python pennylane_variational_pulse_2q.py ibm_real"
              " <seam|off_res|compare> <TOKEN>")
        sys.exit(1)

    run_scan(scan_type=scan_type, mode=mode, token=token)
