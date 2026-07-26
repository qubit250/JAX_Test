"""
3-qubit 光化学反応モデル — Duschinsky 回転 (Task 4)
=======================================================
2-qubit JT モデルを 3-qubit に拡張し、Duschinsky 角 φ を導入。

H(φ) = ω₁/2·ZII + ε/2·IZI + ω₂/2·IIZ
       + g·cos(φ)·X₀X₁I  [XXI: mode1-electronic]
       + g·sin(φ)·IX₁X₂  [IXX: electronic-mode2]

量子ビット配置: q0=mode1, q1=electronic, q2=mode2
  (線形接続 q0-q1-q2 — 隣接 CNOT のみ, [XXI, IXX]=0)

CI 条件:
  CI₁: ε = -ω₁ = -1.0  (XXI チャンネル, |000⟩↔|110⟩ 共鳴)
  CI₂: ε = -ω₂ = -1.5  (IXX チャンネル, |000⟩↔|011⟩ 共鳴)

Duschinsky 角 φ の役割:
  φ=0   → XXI のみ → CI₁ で P_CH1≈1, P_CH2=0
  φ=π/4 → 等配分   → 両 CI が活性, 分岐比 η 中間
  φ=π/2 → IXX のみ → CI₂ で P_CH2≈1, P_CH1=0

Trotter 分解 (1 ステップ):
  RZ(ω₁dt,0)·RZ(εdt,1)·RZ(ω₂dt,2)
  CNOT(0,1)·RX(2g·cos(φ)dt, 0)·CNOT(0,1)  [XXI]
  CNOT(1,2)·RX(2g·sin(φ)dt, 1)·CNOT(1,2)  [IXX]
  CNOT 数: 4 × N_steps  → IBM: 40/回路

スキャン:
  eps_scan : ε スキャン, φ={0,π/4,π/2} → 二重 CI トポロジー
  phi_scan : φ スキャン, ε={-1.0,-1.5} → Duschinsky 分岐比
  map2d    : (ε,φ) 2D マップ (シミュレーションのみ)

使い方:
  python pennylane_3q_photochem.py [simulate|ibm_real] [eps_scan|phi_scan|map2d] [TOKEN]
"""

import sys
import numpy as np
import scipy.linalg
import pennylane as qml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Parameters ──────────────────────────────────────────────────────────────
OMEGA1      = 1.0    # mode 1 frequency  → CI₁ at ε = -ω₁ = -1.0
OMEGA2      = 1.5    # mode 2 frequency  → CI₂ at ε = -ω₂ = -1.5
G_CONST     = 0.30   # coupling strength (≈ g* from 2q model)
T_FINAL     = 5.0
N_STEPS_IBM = 10     # 40 CNOT per circuit
N_STEPS_SIM = 50     # high-accuracy simulation
SHOTS       = 8192

EPS_IBM   = np.array([-2.5, -2.0, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0])
EPS_SIM   = np.linspace(-2.5, 0.5, 25)
PHI_IBM   = np.array([0.0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2])
PHI_SIM   = np.linspace(0.0, np.pi/2, 13)
PHI_SHOW  = [0.0, np.pi/4, np.pi/2]   # for eps_scan: 3 reference angles
EPS_CI    = [-OMEGA1, -OMEGA2]          # CI₁ and CI₂ positions


# ── Exact solution — 4-dim block Hamiltonian ─────────────────────────────────
def block_H_4d(eps, g, phi):
    """
    4-dim block Hamiltonian in basis {|000⟩, |110⟩, |011⟩, |101⟩}.

    Diagonal: H₀ = ω₁/2·Z₀ + ε/2·Z₁ + ω₂/2·Z₂
      |abc⟩ energy = ω₁/2·(1-2a) + ε/2·(1-2b) + ω₂/2·(1-2c)

    Off-diagonal:
      XXI (X₀X₁):  |000⟩↔|110⟩  and  |011⟩↔|101⟩  coupling g·cos(φ)
      IXX (X₁X₂):  |000⟩↔|011⟩  and  |110⟩↔|101⟩  coupling g·sin(φ)
    """
    c = float(np.cos(phi));  s = float(np.sin(phi))
    diag = np.array([
        ( OMEGA1 + eps  + OMEGA2) / 2,   # |000⟩
        (-OMEGA1 - eps  + OMEGA2) / 2,   # |110⟩
        ( OMEGA1 - eps  - OMEGA2) / 2,   # |011⟩
        (-OMEGA1 + eps  - OMEGA2) / 2,   # |101⟩
    ])
    H = np.diag(diag).astype(complex)
    g1, g2 = g * c, g * s
    H[0, 1] = H[1, 0] = g1   # XXI: |000⟩↔|110⟩
    H[2, 3] = H[3, 2] = g1   # XXI: |011⟩↔|101⟩
    H[0, 2] = H[2, 0] = g2   # IXX: |000⟩↔|011⟩
    H[1, 3] = H[3, 1] = g2   # IXX: |110⟩↔|101⟩
    return H


def p_exact_3q(eps, g, phi, T=T_FINAL):
    """Exact probabilities from |000⟩ via matrix exponentiation."""
    H   = block_H_4d(float(eps), float(g), float(phi))
    psi = scipy.linalg.expm(-1j * H * T)[:, 0]  # initial state = |000⟩
    return {
        'P000': float(abs(psi[0]) ** 2),
        'P110': float(abs(psi[1]) ** 2),   # CH1: mode1+elec (CI₁)
        'P011': float(abs(psi[2]) ** 2),   # CH2: elec+mode2 (CI₂)
        'P101': float(abs(psi[3]) ** 2),   # INT: mode1+mode2
    }


def p_analytic_ch1(eps, g=G_CONST, T=T_FINAL):
    """Analytic P_CH1 at phi=0: 2q CI model with detuning ε+ω₁."""
    delta = (float(eps) + OMEGA1) / 2.0
    o1    = float(np.sqrt(delta ** 2 + g ** 2))
    return float((g / o1) ** 2 * np.sin(o1 * T) ** 2) if o1 > 1e-10 else 0.0


def p_analytic_ch2(eps, g=G_CONST, T=T_FINAL):
    """Analytic P_CH2 at phi=pi/2: 2q CI model with detuning ε+ω₂."""
    delta = (float(eps) + OMEGA2) / 2.0
    o1    = float(np.sqrt(delta ** 2 + g ** 2))
    return float((g / o1) ** 2 * np.sin(o1 * T) ** 2) if o1 > 1e-10 else 0.0


# ── Trotter circuit ──────────────────────────────────────────────────────────
def make_dev_3q(mode, backend=None):
    if mode == "ibm_real":
        return qml.device("qiskit.remote", wires=3, backend=backend)
    return qml.device("default.qubit", wires=3)


def eval_circuit_3q(dev, eps, g, phi, n_steps, mode="simulate"):
    """
    Run 3-qubit Trotter circuit and return state probabilities.

    Circuit per step:
      RZ(ω₁·dt, q0)  RZ(ε·dt, q1)  RZ(ω₂·dt, q2)
      CNOT(0,1) · RX(2g·cos(φ)·dt, q0) · CNOT(0,1)   [XXI]
      CNOT(1,2) · RX(2g·sin(φ)·dt, q1) · CNOT(1,2)   [IXX]

    probs index = q0*4 + q1*2 + q2:
      |110⟩ → index 6   |011⟩ → index 3   |101⟩ → index 5
    """
    dt = T_FINAL / n_steps
    g1 = float(g) * float(np.cos(phi))
    g2 = float(g) * float(np.sin(phi))

    @qml.set_shots(SHOTS if mode != "simulate" else None)
    @qml.qnode(dev)
    def circuit():
        for _ in range(n_steps):
            qml.RZ(float(OMEGA1) * dt, wires=0)
            qml.RZ(float(eps)    * dt, wires=1)
            qml.RZ(float(OMEGA2) * dt, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.RX(2.0 * g1 * dt, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.RX(2.0 * g2 * dt, wires=1)
            qml.CNOT(wires=[1, 2])
        return qml.probs(wires=[0, 1, 2])

    p = circuit()
    return {
        'P000':   float(p[0]),
        'P110':   float(p[6]),   # CH1
        'P011':   float(p[3]),   # CH2
        'P101':   float(p[5]),   # INT
        'P_leak': float(p[1] + p[2] + p[4] + p[7]),
    }


# ── IBM backend setup ────────────────────────────────────────────────────────
def get_backend(token, min_qubits=3):
    from qiskit_ibm_runtime import QiskitRuntimeService
    try:
        svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    except Exception:
        svc = QiskitRuntimeService(
            channel="ibm_quantum_platform", token=token,
            instance="open-instance")
    backend = svc.least_busy(operational=True, simulator=False,
                              min_num_qubits=min_qubits)
    print(f"  Backend: {backend.name}")
    return backend


# ── ε scan ──────────────────────────────────────────────────────────────────
def run_eps_scan(mode="simulate", token=None):
    """
    ε scan showing dual CI topology.
    Simulation: phi={0, pi/4, pi/2}  IBM: phi=pi/4 only (shows both peaks).
    """
    n_steps  = N_STEPS_IBM if mode == "ibm_real" else N_STEPS_SIM
    eps_vals = EPS_IBM     if mode == "ibm_real" else EPS_SIM
    backend  = get_backend(token) if mode == "ibm_real" else None

    print(f"\n{'='*72}")
    print(f"  3-qubit Photochem — ε scan (Dual CI Topology)")
    print(f"  H(φ) = ω₁/2·ZII + ε/2·IZI + ω₂/2·IIZ + g·cos(φ)·XXI + g·sin(φ)·IXX")
    print(f"  ω₁={OMEGA1}, ω₂={OMEGA2}, g={G_CONST}, T={T_FINAL}")
    print(f"  CI₁: ε=-ω₁={-OMEGA1:.1f}  CI₂: ε=-ω₂={-OMEGA2:.1f}")
    print(f"  N_steps={n_steps}, CNOT={4*n_steps}/circuit, mode={mode}")
    print(f"{'='*72}")

    phi_set = [np.pi/4] if mode == "ibm_real" else PHI_SHOW
    results = {}

    for phi in phi_set:
        lbl = f"φ={phi/np.pi:.3f}π (cos={np.cos(phi):.3f}, sin={np.sin(phi):.3f})"
        print(f"\n  [{lbl}]")
        print(f"  {'ε':>6}  {'P_CH1':>8}  {'P_CH2':>8}  "
              f"{'P_ex_CH1':>10}  {'P_ex_CH2':>10}  {'err_CH1%':>9}  {'err_CH2%':>9}")
        print(f"  {'-'*74}")

        phi_results = {}
        for eps in eps_vals:
            dev  = make_dev_3q(mode, backend)
            r    = eval_circuit_3q(dev, float(eps), G_CONST, phi, n_steps, mode)
            ex   = p_exact_3q(float(eps), G_CONST, phi)
            err1 = abs(r['P110'] - ex['P110']) / max(ex['P110'], 1e-4) * 100
            err2 = abs(r['P011'] - ex['P011']) / max(ex['P011'], 1e-4) * 100
            phi_results[float(eps)] = {**r, 'exact': ex}

            ci_mark = ""
            if abs(eps + OMEGA1) < 0.01:
                ci_mark = " ← CI₁"
            elif abs(eps + OMEGA2) < 0.01:
                ci_mark = " ← CI₂"

            print(f"  {eps:>+6.2f}  {r['P110']:>8.4f}  {r['P011']:>8.4f}  "
                  f"{ex['P110']:>10.4f}  {ex['P011']:>10.4f}  "
                  f"{err1:>8.2f}%  {err2:>8.2f}%{ci_mark}")

        results[float(phi)] = phi_results

    print(f"\n  [物理的観察]")
    print(f"  φ=0:   CI₁ (ε=-{OMEGA1}) で P_CH1≈1  (mode1 経由 XXI チャンネル)")
    print(f"  φ=π/2: CI₂ (ε=-{OMEGA2}) で P_CH2≈1  (mode2 経由 IXX チャンネル)")
    print(f"  φ=π/4: 両 CI 共鳴 → 二重ピーク (Duschinsky 混合の特徴)")
    return results


# ── φ scan ──────────────────────────────────────────────────────────────────
def run_phi_scan(mode="simulate", token=None):
    """Duschinsky angle scan — branching ratio at CI₁ and CI₂."""
    n_steps  = N_STEPS_IBM if mode == "ibm_real" else N_STEPS_SIM
    phi_vals = PHI_IBM     if mode == "ibm_real" else PHI_SIM
    backend  = get_backend(token) if mode == "ibm_real" else None

    print(f"\n{'='*72}")
    print(f"  3-qubit Photochem — Duschinsky φ scan")
    print(f"  Branching ratio: η = P_CH1 / (P_CH1 + P_CH2)")
    print(f"{'='*72}")

    results = {}
    for eps in EPS_CI:
        ci_name = "CI₁" if abs(eps + OMEGA1) < 0.01 else "CI₂"
        print(f"\n  [ε={eps:.1f} — {ci_name} point]")
        print(f"  {'φ/π':>6}  {'P_CH1':>8}  {'P_CH2':>8}  {'P_INT':>8}  "
              f"{'η':>8}  {'ex_CH1':>8}  {'ex_CH2':>8}")
        print(f"  {'-'*66}")

        eps_results = {}
        for phi in phi_vals:
            dev   = make_dev_3q(mode, backend)
            r     = eval_circuit_3q(dev, float(eps), G_CONST, phi, n_steps, mode)
            ex    = p_exact_3q(float(eps), G_CONST, phi)
            total = r['P110'] + r['P011']
            eta   = r['P110'] / total if total > 1e-6 else float('nan')
            eps_results[float(phi)] = {**r, 'exact': ex, 'eta': eta}

            print(f"  {phi/np.pi:>6.3f}  {r['P110']:>8.4f}  {r['P011']:>8.4f}  "
                  f"{r['P101']:>8.4f}  {eta:>8.4f}  {ex['P110']:>8.4f}  {ex['P011']:>8.4f}")

        results[float(eps)] = eps_results

    print(f"\n  [Duschinsky 物理]")
    print(f"  φ=0   (XXI のみ): CI₁ (ε=-{OMEGA1}) で P_CH1→1, ε=-{OMEGA2} で P_CH1 中程度")
    print(f"  φ=π/2 (IXX のみ): CI₂ (ε=-{OMEGA2}) で P_CH2→1, ε=-{OMEGA1} で P_CH2 中程度")
    print(f"  → φ が反応収率の分岐比 η を連続的に制御 (Duschinsky 回転効果)")
    return results


# ── 2D map ───────────────────────────────────────────────────────────────────
def run_map2d():
    """2D (ε, φ) grid using exact solution (simulation only)."""
    n_eps, n_phi = 40, 20
    eps_grid = np.linspace(-2.5, 0.5, n_eps)
    phi_grid = np.linspace(0.0, np.pi/2, n_phi)

    print(f"\n  Computing 2D map ({n_eps}×{n_phi} grid, exact expm)...")
    P110 = np.zeros((n_phi, n_eps))
    P011 = np.zeros((n_phi, n_eps))
    eta  = np.zeros((n_phi, n_eps))

    for i, phi in enumerate(phi_grid):
        for j, eps in enumerate(eps_grid):
            ex = p_exact_3q(float(eps), G_CONST, phi)
            P110[i, j] = ex['P110']
            P011[i, j] = ex['P011']
            s = ex['P110'] + ex['P011']
            eta[i, j] = ex['P110'] / s if s > 1e-4 else 0.5

    return eps_grid, phi_grid, P110, P011, P110 + P011, eta


# ── Plotting ─────────────────────────────────────────────────────────────────
_COLORS = {0.0: 'steelblue', np.pi/4: 'firebrick', np.pi/2: 'seagreen'}
_LS     = {0.0: '-',         np.pi/4: '--',         np.pi/2: '-.'}
_MARKS  = {0.0: 'o',         np.pi/4: 's',          np.pi/2: '^'}


def plot_eps_scan(results, mode="simulate"):
    """3-panel eps scan: P_CH1, P_CH2, total yield vs ε."""
    phi_list = sorted(results.keys())
    eps_fine = np.linspace(-2.5, 0.5, 300)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titles = [
        "P_CH1 = P(|110⟩)\nChannel 1: mode1+elec  (CI₁ at ε=-1.0)",
        "P_CH2 = P(|011⟩)\nChannel 2: elec+mode2  (CI₂ at ε=-1.5)",
        "P_total = P_CH1 + P_CH2\nTotal reaction yield",
    ]
    keys = ['P110', 'P011', 'total']

    for col, (title, key) in enumerate(zip(titles, keys)):
        ax = axes[col]
        for phi in phi_list:
            phi_data = results[phi]
            c   = _COLORS.get(phi, 'purple')
            ls  = _LS.get(phi, ':')
            mk  = _MARKS.get(phi, 'D')
            lbl = f"φ={phi/np.pi:.2f}π"

            # Exact fine-grid curves
            if key == 'P110':
                y_fine  = [p_exact_3q(e, G_CONST, phi)['P110'] for e in eps_fine]
                y_an    = [p_analytic_ch1(e) if abs(phi) < 1e-6 else None for e in eps_fine]
            elif key == 'P011':
                y_fine  = [p_exact_3q(e, G_CONST, phi)['P011'] for e in eps_fine]
                y_an    = [p_analytic_ch2(e) if abs(phi - np.pi/2) < 1e-6 else None
                           for e in eps_fine]
            else:
                ex_list = [p_exact_3q(e, G_CONST, phi) for e in eps_fine]
                y_fine  = [ex['P110'] + ex['P011'] for ex in ex_list]
                y_an    = [None] * len(eps_fine)

            ax.plot(eps_fine, y_fine, color=c, ls=ls, lw=2, alpha=0.75,
                    label=f"{lbl} exact")

            # Circuit evaluation points
            eps_pts = sorted(phi_data.keys())
            if key == 'P110':
                y_pts = [phi_data[e]['P110'] for e in eps_pts]
            elif key == 'P011':
                y_pts = [phi_data[e]['P011'] for e in eps_pts]
            else:
                y_pts = [phi_data[e]['P110'] + phi_data[e]['P011'] for e in eps_pts]
            ax.scatter(eps_pts, y_pts, color=c, s=60, zorder=5, marker=mk,
                       label=f"{lbl} circuit" if mode == "ibm_real" else None)

        ax.axvline(-OMEGA1, color='gray',  ls=':', lw=1.5, alpha=0.8,
                   label=f"CI₁ ε={-OMEGA1:.1f}")
        ax.axvline(-OMEGA2, color='black', ls=':', lw=1.5, alpha=0.8,
                   label=f"CI₂ ε={-OMEGA2:.1f}")
        ax.set_xlabel("ε (detuning)")
        ax.set_ylabel("Probability")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "3-qubit Photochemical Model — Dual CI Topology + Duschinsky Rotation\n"
        r"$H(\varphi)=\frac{\omega_1}{2}ZII+\frac{\varepsilon}{2}IZI+\frac{\omega_2}{2}IIZ"
        r"+g\cos\varphi\cdot XXI+g\sin\varphi\cdot IXX$, "
        f"ω₁={OMEGA1}, ω₂={OMEGA2}, g={G_CONST}, T={T_FINAL}",
        fontsize=9
    )
    plt.tight_layout()
    fname = "photochem_3q_eps_scan.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  -> Plot saved: {fname}")


def plot_phi_scan(results, mode="simulate"):
    """2-panel φ scan: P_CH1 & P_CH2 at CI₁ and CI₂."""
    eps_list = sorted(results.keys())
    phi_fine = np.linspace(0.0, np.pi/2, 200)

    fig, axes = plt.subplots(1, len(eps_list), figsize=(7 * len(eps_list), 5))
    if len(eps_list) == 1:
        axes = [axes]

    for col, eps in enumerate(eps_list):
        ax  = axes[col]
        ci  = "CI₁" if abs(eps + OMEGA1) < 0.01 else "CI₂"
        data = results[eps]

        # Exact curves
        ex_list = [p_exact_3q(eps, G_CONST, p) for p in phi_fine]
        ax.plot(phi_fine / np.pi, [e['P110'] for e in ex_list],
                'b-', lw=2, label="P_CH1=P(|110⟩) exact")
        ax.plot(phi_fine / np.pi, [e['P011'] for e in ex_list],
                'r-', lw=2, label="P_CH2=P(|011⟩) exact")
        ax.plot(phi_fine / np.pi, [e['P101'] for e in ex_list],
                'g--', lw=1.5, alpha=0.6, label="P_INT=P(|101⟩) exact")

        # Circuit points
        phis   = sorted(data.keys())
        ax.scatter(np.array(phis) / np.pi,
                   [data[p]['P110'] for p in phis],
                   c='blue', s=60, zorder=5, marker='o',
                   label="P_CH1 circuit")
        ax.scatter(np.array(phis) / np.pi,
                   [data[p]['P011'] for p in phis],
                   c='red',  s=60, zorder=5, marker='s',
                   label="P_CH2 circuit")

        # Branching ratio on twin axis
        ax2 = ax.twinx()
        eta_fine = [e['P110'] / max(e['P110'] + e['P011'], 1e-6) for e in ex_list]
        ax2.plot(phi_fine / np.pi, eta_fine, 'k-.', lw=1.5, alpha=0.5,
                 label="η=P₁/(P₁+P₂)")
        ax2.set_ylabel("Branching ratio η", color='black')
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(fontsize=8, loc='upper center')

        ax.set_xlabel("φ/π (Duschinsky angle)")
        ax.set_ylabel("Transition probability")
        ax.set_title(f"ε={eps:.1f} ({ci} point)\nDuschinsky branching ratio", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"3-qubit Photochem — Duschinsky Branching Ratio\n"
        f"ω₁={OMEGA1}, ω₂={OMEGA2}, g={G_CONST}, T={T_FINAL}, "
        f"N={N_STEPS_IBM if mode == 'ibm_real' else N_STEPS_SIM} steps",
        fontsize=10
    )
    plt.tight_layout()
    fname = "photochem_3q_phi_scan.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  -> Plot saved: {fname}")


def plot_map2d(eps_grid, phi_grid, P110, P011, P_sum, eta):
    """4-panel (ε, φ) 2D map."""
    phi_deg = phi_grid * 180.0 / np.pi
    ext = [eps_grid[0], eps_grid[-1], phi_deg[0], phi_deg[-1]]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    data_list = [
        (P110, "P_CH1 = P(|110⟩)\nChannel 1 (CI₁ at ε=-1.0)", "viridis"),
        (P011, "P_CH2 = P(|011⟩)\nChannel 2 (CI₂ at ε=-1.5)", "plasma"),
        (P_sum, "P_total = P_CH1 + P_CH2\nTotal yield", "inferno"),
        (eta,  "Branching ratio η\n= P_CH1 / P_total",       "RdBu"),
    ]

    for ax, (data, title, cmap) in zip(axes, data_list):
        im = ax.imshow(data, aspect='auto', origin='lower', cmap=cmap,
                       extent=ext, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.axvline(-OMEGA1, color='white', ls='--', lw=1.5, alpha=0.9,
                   label=f"CI₁ ε={-OMEGA1:.1f}")
        ax.axvline(-OMEGA2, color='cyan',  ls='--', lw=1.5, alpha=0.9,
                   label=f"CI₂ ε={-OMEGA2:.1f}")
        ax.set_xlabel("ε (detuning)")
        ax.set_ylabel("φ (degrees)")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, loc='upper right')

    plt.suptitle(
        "3-qubit PES Map — Dual CI Topology + Duschinsky Branching\n"
        f"ω₁={OMEGA1}, ω₂={OMEGA2}, g={G_CONST}, T={T_FINAL}  (exact expm, 40×20 grid)",
        fontsize=9
    )
    plt.tight_layout()
    fname = "photochem_3q_map2d.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Plot saved: {fname}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mode     = sys.argv[1] if len(sys.argv) > 1 else "simulate"
    scan_typ = sys.argv[2] if len(sys.argv) > 2 else "eps_scan"
    token    = sys.argv[3] if len(sys.argv) > 3 else None

    if mode == "ibm_real" and token is None:
        print("ERROR: TOKEN required for ibm_real mode", file=sys.stderr)
        sys.exit(1)

    if scan_typ == "eps_scan":
        r = run_eps_scan(mode, token)
        plot_eps_scan(r, mode)

    elif scan_typ == "phi_scan":
        r = run_phi_scan(mode, token)
        plot_phi_scan(r, mode)

    elif scan_typ == "map2d":
        if mode == "ibm_real":
            print("  map2d is simulation-only; switching to simulate")
            mode = "simulate"
        data = run_map2d()
        plot_map2d(*data)
        r1 = run_eps_scan("simulate")
        plot_eps_scan(r1, "simulate")
        r2 = run_phi_scan("simulate")
        plot_phi_scan(r2, "simulate")

    else:
        print(f"Unknown scan type: {scan_typ}. "
              f"Choose: eps_scan | phi_scan | map2d", file=sys.stderr)
        sys.exit(1)
