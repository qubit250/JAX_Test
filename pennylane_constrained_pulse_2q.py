"""
振幅制約付き変分パルス制御 (Task 3b)
======================================
Task 3 (unconstrained) の IBM 実機問題を解決:
  ε=-0.5: |g|≤2.0 の急激な符号反転パルス → IBM で P: 0.494→0.323 劣化

解決策:
  1. tanh 制約  : g_k = G_MAX · tanh(θ_k)    [|g(t)| ≤ G_MAX を保証]
  2. TV 正則化  : Loss += λ · Σ(g_{k+1}−g_k)²  [高周波振動を抑制]

期待効果:
  ε=-0.5: P_sim 若干低下, P_IBM: 0.323 → 0.50+ (IBM 劣化解消)
  ε=0.0:  P_sim ほぼ維持, P_IBM: 0.939 維持 (成功を確認)

スキャン:
  constrained  : ε=-0.5, 0.0 — 無制約 vs 制約付き (シム + IBM)
  lambda_scan  : λ スキャン (ε=-0.5) — 最適正則化強度の探索
  robust       : ε 全スキャン — 制約付きパルスの全体マップ

使い方:
  python pennylane_constrained_pulse_2q.py [simulate|ibm_real] [constrained|lambda_scan|robust] [TOKEN]
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
N_STEPS_OPT = 20
N_STEPS_IBM = 10
N_ITER_UNC  = 300    # 無制約
N_ITER_CON  = 400    # 制約付き (収束やや遅い)
STEP_SIZE   = 0.05
G_STAR      = np.pi / (2.0 * T_FINAL)   # ≈ 0.3142
G_MAX       = 1.0    # 振幅上限 (Task 3 失敗: |g|≤2.0 → 半分に制限)
LAMBDA_TV   = 0.05   # TV 正則化強度
SHOTS       = 8192

EPS_CONSTRAINED = np.array([-0.5, 0.0])
EPS_ROBUST      = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
LAMBDA_VALUES   = np.array([0.0, 0.01, 0.05, 0.1, 0.3, 1.0])


# ── 解析関数 ──────────────────────────────────────────────────

def rabi_freq(eps, g):
    return float(np.sqrt(((eps + OMEGA) / 2.0) ** 2 + g ** 2))

def p_analytic(eps, g, T=T_FINAL):
    o1 = rabi_freq(eps, g)
    return float((g / o1) ** 2 * np.sin(o1 * T) ** 2) if o1 > 1e-10 else 0.0

def p_max_const(eps, g):
    o1 = rabi_freq(eps, g)
    return float((g / o1) ** 2) if o1 > 1e-10 else 0.0


# ── 回路 (共通) ───────────────────────────────────────────────

def _make_circuit_node(eps, n_steps):
    """autograd インターフェースの QNode を返す。"""
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

    return circuit


def eval_fixed(eps, g_arr, n_steps, mode, backend=None):
    """固定パルスを評価 (勾配不要)。"""
    dt = T_FINAL / n_steps
    dev = (qml.device("qiskit.remote", wires=2, backend=backend)
           if mode == "ibm_real" else qml.device("default.qubit", wires=2))

    @qml.set_shots(SHOTS if mode != "simulate" else None)
    @qml.qnode(dev)
    def circuit():
        for k in range(n_steps):
            qml.RZ(float(eps) * dt,              wires=0)
            qml.RZ(float(OMEGA) * dt,             wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RX(float(2.0 * g_arr[k]) * dt,  wires=0)
            qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    return float(circuit()[3])


# ── 損失関数 ──────────────────────────────────────────────────

def make_unconstrained_cost(eps, n_steps):
    circuit = _make_circuit_node(eps, n_steps)
    def cost(g_params):
        return 1.0 - circuit(g_params)[3]
    return cost


def make_constrained_cost(eps, n_steps, g_max=G_MAX, lambda_tv=LAMBDA_TV):
    """
    tanh 制約 + TV 正則化付きコスト。
    raw_params ∈ ℝ^N → g = g_max · tanh(raw) ∈ (−g_max, +g_max)
    Loss = (1 − P) + λ · Σ(g_{k+1}−g_k)²
    """
    circuit = _make_circuit_node(eps, n_steps)

    def cost(raw_params):
        g  = g_max * pnp.tanh(raw_params)
        P  = circuit(g)[3]
        tv = pnp.sum((g[1:] - g[:-1]) ** 2)
        return 1.0 - P + lambda_tv * tv

    return cost


# ── 最適化 ────────────────────────────────────────────────────

def optimize_unconstrained(eps, n_steps=N_STEPS_OPT, n_iter=N_ITER_UNC,
                           verbose=False):
    """Task 3 無制約最適化 (比較用)。"""
    cost   = make_unconstrained_cost(eps, n_steps)
    params = pnp.ones(n_steps, requires_grad=True) * G_STAR
    opt    = qml.AdamOptimizer(stepsize=STEP_SIZE)
    history = []

    for step in range(n_iter):
        params, val = opt.step_and_cost(cost, params)
        history.append(max(0.0, 1.0 - float(val)))
        if verbose and (step % 100 == 0 or step == n_iter - 1):
            print(f"    [unc] iter {step:3d}: P≈{history[-1]:.4f}")

    g_opt = np.array(params)
    P_opt = eval_fixed(eps, g_opt, n_steps, "simulate")
    tv    = float(np.sum(np.diff(g_opt) ** 2))
    return g_opt, P_opt, history, tv


def optimize_constrained(eps, n_steps=N_STEPS_OPT, n_iter=N_ITER_CON,
                         g_max=G_MAX, lambda_tv=LAMBDA_TV, verbose=True):
    """tanh 制約 + TV 正則化付き最適化。"""
    cost = make_constrained_cost(eps, n_steps, g_max, lambda_tv)

    # g_init = G_STAR に対応する raw 初期値
    init = float(np.arctanh(np.clip(G_STAR / g_max, -0.9999, 0.9999)))
    raw  = pnp.ones(n_steps, requires_grad=True) * init
    opt  = qml.AdamOptimizer(stepsize=STEP_SIZE)
    history = []

    for step in range(n_iter):
        raw, val = opt.step_and_cost(cost, raw)
        history.append(max(0.0, 1.0 - float(val)))   # 近似 P (TV ペナルティ含む)
        if verbose and (step % 100 == 0 or step == n_iter - 1):
            g_cur = float(g_max) * np.tanh(np.array(raw))
            tv    = float(np.sum(np.diff(g_cur) ** 2))
            print(f"    [con] iter {step:3d}: loss_P≈{history[-1]:.4f}  "
                  f"TV={tv:.4f}  |g|_max={np.max(np.abs(g_cur)):.3f}")

    g_opt = float(g_max) * np.tanh(np.array(raw))
    P_opt = eval_fixed(eps, g_opt, n_steps, "simulate")
    tv    = float(np.sum(np.diff(g_opt) ** 2))
    return g_opt, P_opt, history, tv


def downsample(g, n):
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(g)), g)


# ── プロット ──────────────────────────────────────────────────

def _t(n):
    return (np.arange(n) + 0.5) * T_FINAL / n


def plot_constrained(results):
    """各 ε について unconstrained / constrained の 3 行比較。"""
    eps_list = sorted(results.keys())
    n = len(eps_list)
    t = _t(N_STEPS_OPT)

    fig, axes = plt.subplots(3, n, figsize=(7 * n, 12))
    if n == 1:
        axes = axes.reshape(3, 1)

    for col, eps in enumerate(eps_list):
        r = results[eps]
        ru, rc = r['unc'], r['con']

        # Row 0: pulse shape
        ax = axes[0, col]
        ax.step(t, np.ones(N_STEPS_OPT) * G_STAR, where='mid',
                color='steelblue', lw=1.5, alpha=0.6, label=f"Const g*={G_STAR:.3f}")
        ax.step(t, ru['g_opt'], where='mid', color='gray', lw=1.5, ls=':',
                label=f"Unconstrained (P={ru['P_opt']:.3f}, TV={ru['tv']:.2f})")
        ax.step(t, rc['g_opt'], where='mid', color='firebrick', lw=2, ls='--',
                label=f"Constrained  (P={rc['P_opt']:.3f}, TV={rc['tv']:.2f})")
        ax.axhline( G_MAX, color='firebrick', ls=':', alpha=0.3, lw=1)
        ax.axhline(-G_MAX, color='firebrick', ls=':', alpha=0.3, lw=1)
        ax.set_xlabel("t");  ax.set_ylabel("g(t)")
        ax.set_title(f"Pulse shape (ε={eps:+.1f})")
        ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)
        ax.set_ylim(-G_MAX * 1.4, G_MAX * 1.4)

        # Row 1: convergence
        ax = axes[1, col]
        ax.plot(ru['history'], color='gray', lw=1.5, ls=':', alpha=0.7,
                label="Unconstrained (Task 3)")
        ax.plot(rc['history'], color='firebrick', lw=2,
                label=f"Constrained (λ={LAMBDA_TV})")
        P_base = eval_fixed(eps, np.ones(N_STEPS_OPT) * G_STAR, N_STEPS_OPT, "simulate")
        ax.axhline(P_base, color='steelblue', ls='--', lw=2,
                   label=f"Const g* P={P_base:.3f}")
        ax.set_xlabel("Iteration");  ax.set_ylabel("P or loss_P")
        ax.set_title(f"Convergence (ε={eps:+.1f})")
        ax.legend(fontsize=8);  ax.set_ylim(-0.05, 1.05);  ax.grid(True, alpha=0.3)

        # Row 2: summary bar
        ax = axes[2, col]
        labels = ["Sim\nconst g*", "Sim\nTask3\nunc.", "Sim\nTask3b\ncon."]
        values = [P_base, ru['P_opt'], rc['P_opt']]
        colors = ['steelblue', 'dimgray', 'firebrick']
        if 'P_ibm_const' in rc:
            labels += ["IBM\nconst g*", "IBM\nTask3\nunc.", "IBM\nTask3b\ncon."]
            values += [rc['P_ibm_const'], ru.get('P_ibm_opt', 0), rc['P_ibm_opt']]
            colors += ['cornflowerblue', 'silver', 'orangered']
        bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor='black')
        ax.axhline(1.0, color='gray', ls='--', lw=1)
        ax.set_ylabel("P(|11>)");  ax.set_ylim(0, 1.15)
        ax.set_title(f"Summary (ε={eps:+.1f})")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.3f}", ha='center', va='bottom',
                    fontsize=8, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(
        f"Task 3b: Constrained Pulse (|g|≤{G_MAX}, λ_TV={LAMBDA_TV}, "
        f"T={T_FINAL}, N={N_STEPS_OPT})", fontsize=12)
    plt.tight_layout()
    fname = "constrained_pulse_compare.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight");  plt.close()
    print(f"\n  -> Plot saved: {fname}")


def plot_lambda_scan(lambda_results, eps):
    lambdas  = sorted(lambda_results.keys())
    P_opts   = [lambda_results[l]['P_opt'] for l in lambdas]
    TV_norms = [lambda_results[l]['tv']    for l in lambdas]
    t = _t(N_STEPS_OPT)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.semilogx([max(l, 1e-3) for l in lambdas], P_opts, 'r^-', lw=2, ms=8)
    ax.axhline(p_analytic(eps, G_STAR), color='steelblue', ls='--', lw=2,
               label=f"Const g* P={p_analytic(eps,G_STAR):.3f}")
    ax.set_xlabel("λ_TV");  ax.set_ylabel("P_opt (sim)")
    ax.set_title(f"P vs regularization (ε={eps:+.1f}, |g|≤{G_MAX})")
    ax.legend(fontsize=9);  ax.set_ylim(0, 1.1);  ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogx([max(l, 1e-3) for l in lambdas], TV_norms, 'b^-', lw=2, ms=8)
    ax.set_xlabel("λ_TV");  ax.set_ylabel("TV norm = Σ(Δg)²")
    ax.set_title("Pulse smoothness vs regularization")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    cmap = plt.cm.viridis(np.linspace(0, 1, len(lambdas)))
    for i, lam in enumerate(lambdas):
        g = lambda_results[lam]['g_opt']
        label = f"λ={lam:.2f} P={P_opts[i]:.3f}" if lam > 0 else f"λ=0 P={P_opts[i]:.3f}"
        ax.step(t, g, where='mid', color=cmap[i], lw=1.5, label=label)
    ax.axhline( G_MAX, color='red', ls=':', alpha=0.3, lw=1)
    ax.axhline(-G_MAX, color='red', ls=':', alpha=0.3, lw=1)
    ax.set_xlabel("t");  ax.set_ylabel("g(t)")
    ax.set_title(f"Pulse shapes (ε={eps:+.1f})")
    ax.legend(fontsize=7, ncol=2);  ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"λ_TV Scan: Regularization Effect (ε={eps:+.1f}, |g|≤{G_MAX}, N={N_STEPS_OPT})",
        fontsize=12)
    plt.tight_layout()
    fname = "constrained_pulse_lambda_scan.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight");  plt.close()
    print(f"\n  -> Plot saved: {fname}")


def plot_robust(eps_vals, P_base_list, P_unc_list, P_con_list):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(eps_vals, P_base_list, 'b--o', lw=2, ms=7, label="Const g*")
    ax.plot(eps_vals, P_unc_list,  'g:^',  lw=2, ms=8, label="Unconstrained (Task 3)")
    ax.plot(eps_vals, P_con_list,  'r-s',  lw=2, ms=9,
            label=f"Constrained (Task 3b, |g|≤{G_MAX}, λ={LAMBDA_TV})")
    ax.fill_between(eps_vals, P_base_list, P_con_list, alpha=0.1, color='red')
    ax.axvline(-1.0, color='gray', ls=':', lw=1, alpha=0.5, label="CI seam")
    ax.set_xlabel(r"$\varepsilon$");  ax.set_ylabel("P(|11>)")
    ax.set_title(f"Robust variational pulse — ε scan (T={T_FINAL})")
    ax.legend(fontsize=9);  ax.set_ylim(-0.05, 1.1);  ax.grid(True, alpha=0.3)

    delta_3b = [P_con_list[i] - P_base_list[i] for i in range(len(eps_vals))]
    delta_unc = [P_unc_list[i] - P_base_list[i] for i in range(len(eps_vals))]
    ax = axes[1]
    x = np.arange(len(eps_vals));  w = 0.35
    bars1 = ax.bar(x - w/2, delta_unc, w, color='forestgreen', alpha=0.7,
                   edgecolor='black', label="Task 3 (unconstrd)")
    bars2 = ax.bar(x + w/2, delta_3b, w, color='firebrick', alpha=0.7,
                   edgecolor='black', label="Task 3b (constrd)")
    ax.axhline(0, color='black', lw=1)
    ax.set_xticks(x);  ax.set_xticklabels([f"{e:+.1f}" for e in eps_vals])
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("ΔP = P_opt − P_const")
    ax.set_title("Gain over constant g* (constrained vs unconstrained)")
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(
        f"Robust Variational Pulse — ε Scan (T={T_FINAL}, N={N_STEPS_OPT})",
        fontsize=12)
    plt.tight_layout()
    fname = "constrained_pulse_robust.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight");  plt.close()
    print(f"\n  -> Plot saved: {fname}")


# ── スキャン実行 ──────────────────────────────────────────────

def run_scan(scan_type="constrained", mode="simulate", token=None):
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  振幅制約付き変分パルス (Task 3b)  —  {scan_type} スキャン")
    print(f"  H(t) = ε/2·ZI + ω/2·IZ + g(t)·XX,  ω={OMEGA}, T={T_FINAL}")
    print(f"  制約: |g(t)| ≤ G_MAX={G_MAX}  [tanh: g=G_MAX·tanh(θ)]")
    print(f"  正則化: λ_TV={LAMBDA_TV}  [TV = Σ(g_{{k+1}}−g_k)²]")
    print(f"  最適化: Adam lr={STEP_SIZE}")
    print(sep)

    backend = None
    if mode == "ibm_real":
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        backend = service.least_busy(
            operational=True, simulator=False, min_num_qubits=2)
        print(f"\n  バックエンド: {backend.name}")

    # ── constrained ───────────────────────────────────────────
    if scan_type == "constrained":
        print(f"\n  [Task 3 (無制約) vs Task 3b (|g|≤{G_MAX}, λ={LAMBDA_TV})]")
        print(f"\n  ε   指標          無制約(Task3)  制約付き(3b)")
        print("  " + "-" * 50)

        results = {}
        for eps in EPS_CONSTRAINED:
            eps = float(eps)
            P_base  = eval_fixed(eps, np.ones(N_STEPS_OPT)*G_STAR, N_STEPS_OPT, "simulate")
            P_max_v = p_max_const(eps, G_STAR)
            print(f"\n  ε={eps:+.1f}  (P_const={P_base:.3f}, P_max={P_max_v:.3f}):")

            print("    無制約最適化 (Task 3)...")
            g_u, P_u, h_u, tv_u = optimize_unconstrained(eps, N_STEPS_OPT, N_ITER_UNC)
            print(f"    P_opt={P_u:.4f}  TV={tv_u:.3f}  |g|_max={np.max(np.abs(g_u)):.3f}")

            print(f"    制約付き最適化 (|g|≤{G_MAX}, λ={LAMBDA_TV})...")
            g_c, P_c, h_c, tv_c = optimize_constrained(
                eps, N_STEPS_OPT, N_ITER_CON, G_MAX, LAMBDA_TV, verbose=True)
            print(f"    P_opt={P_c:.4f}  TV={tv_c:.3f}  |g|_max={np.max(np.abs(g_c)):.3f}")
            print(f"    平滑化: TV {tv_u:.2f} → {tv_c:.2f} "
                  f"({(1-tv_c/max(tv_u,1e-4))*100:.0f}% 削減)")

            ru = {'g_opt': g_u, 'P_opt': P_u, 'history': h_u, 'tv': tv_u}
            rc = {'g_opt': g_c, 'P_opt': P_c, 'history': h_c, 'tv': tv_c}

            if mode == "ibm_real":
                print(f"\n    IBM 評価 (N={N_STEPS_IBM}, CNOT=20)...")
                g_ibm_base = np.ones(N_STEPS_IBM) * G_STAR
                P_bi = eval_fixed(eps, g_ibm_base, N_STEPS_IBM, mode, backend)
                P_ui = eval_fixed(eps, downsample(g_u, N_STEPS_IBM), N_STEPS_IBM, mode, backend)
                P_ci = eval_fixed(eps, downsample(g_c, N_STEPS_IBM), N_STEPS_IBM, mode, backend)
                print(f"    P_const_IBM   = {P_bi:.4f}")
                print(f"    P_Task3_IBM   = {P_ui:.4f}  (無制約)")
                print(f"    P_Task3b_IBM  = {P_ci:.4f}  (制約付き)"
                      f"  {'↑ 改善' if P_ci > P_ui else '↓ 劣化'}")
                ru['P_ibm_opt'] = P_ui
                rc['P_ibm_const'] = P_bi;  rc['P_ibm_opt'] = P_ci

            results[eps] = {'unc': ru, 'con': rc}

        plot_constrained(results)

    # ── lambda_scan ───────────────────────────────────────────
    elif scan_type == "lambda_scan":
        eps = -0.5
        print(f"\n  [λ_TV スキャン: ε={eps:+.1f}, |g|≤{G_MAX}]")
        print(f"  λ ∈ {list(LAMBDA_VALUES)}")
        print(f"\n  {'λ_TV':>8}  {'P_opt':>8}  {'TV_norm':>9}  {'|g|_max':>8}")
        print("  " + "-" * 42)

        lambda_results = {}
        for lam in LAMBDA_VALUES:
            lam = float(lam)
            g_opt, P_opt, history, tv = optimize_constrained(
                eps, N_STEPS_OPT, N_ITER_CON, G_MAX, lam, verbose=False)
            g_max_act = float(np.max(np.abs(g_opt)))
            print(f"  {lam:>8.3f}  {P_opt:>8.4f}  {tv:>9.4f}  {g_max_act:>8.4f}")
            lambda_results[lam] = {'g_opt': g_opt, 'P_opt': P_opt,
                                   'history': history, 'tv': tv}

        lambdas = sorted(lambda_results.keys())
        best    = max(lambda_results, key=lambda l: lambda_results[l]['P_opt'])
        r_min   = lambda_results[min(lambdas)]
        r_max   = lambda_results[max(lambdas)]
        print(f"\n  最適 λ*={best:.3f} (P={lambda_results[best]['P_opt']:.4f})")
        print(f"  λ→0:  P≈{r_min['P_opt']:.3f}  TV≈{r_min['tv']:.2f}  (振動大)")
        print(f"  λ→1:  P≈{r_max['P_opt']:.3f}  TV≈{r_max['tv']:.2f}  (過正則化)")
        plot_lambda_scan(lambda_results, eps)

    # ── robust ────────────────────────────────────────────────
    elif scan_type == "robust":
        print(f"\n  [ε スキャン: 制約付き変分パルスの全体マップ]")
        print(f"\n  {'ε':>6}  {'P_const':>8}  {'P_unc':>8}  {'P_con':>8}  {'ΔP(3b)':>8}")
        print("  " + "-" * 50)

        eps_vals   = [float(e) for e in EPS_ROBUST]
        P_base_lst = []
        P_unc_lst  = []
        P_con_lst  = []

        for eps in eps_vals:
            P_base = eval_fixed(eps, np.ones(N_STEPS_OPT)*G_STAR, N_STEPS_OPT, "simulate")
            _, P_u, _, _ = optimize_unconstrained(eps, N_STEPS_OPT, N_ITER_UNC)
            _, P_c, _, _ = optimize_constrained(
                eps, N_STEPS_OPT, N_ITER_CON, G_MAX, LAMBDA_TV, verbose=False)
            print(f"  {eps:>+6.1f}  {P_base:>8.4f}  {P_u:>8.4f}  {P_c:>8.4f}  {P_c-P_base:>+8.4f}")
            P_base_lst.append(P_base)
            P_unc_lst.append(P_u)
            P_con_lst.append(P_c)

        plot_robust(eps_vals, P_base_lst, P_unc_lst, P_con_lst)

    else:
        print(f"  エラー: scan_type は 'constrained', 'lambda_scan', 'robust'")
        sys.exit(1)

    print(f"\n{sep}\n")


if __name__ == "__main__":
    mode      = sys.argv[1] if len(sys.argv) > 1 else "simulate"
    scan_type = sys.argv[2] if len(sys.argv) > 2 else "constrained"
    token     = sys.argv[3] if len(sys.argv) > 3 else None

    if mode == "ibm_real" and token is None:
        print("使い方: python pennylane_constrained_pulse_2q.py ibm_real"
              " <constrained|lambda_scan|robust> <TOKEN>")
        sys.exit(1)

    run_scan(scan_type=scan_type, mode=mode, token=token)
