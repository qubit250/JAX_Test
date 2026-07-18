"""
Spin-Boson Model (スピン-ボゾン モデル)
=======================================
H = (ε/2)·ZI + (Δ/2)·XI + (ω/2)·IZ + g·ZX

Wire 0: 2準位系 (TLS) — 官能基 / CNT π 電子
Wire 1: フォノンモード (2準位近似) — CNT 呼吸モード

α = 2g²/ω²: Kondo パラメータ
  α < 0.5: コヒーレント (コヒーレント振動)
  α = 0.5: Toulouse 点 (指数緩和の閾値)
  α > 0.5: 局在化 (過減衰)

CNT(10,10) 典型値: ω ≈ 35 meV, g ≈ 30 meV → α ≈ 0.65 (局在化近傍)

厳密解: 4×4 ハミルトニアン対角化
  初期状態 |00⟩ (TLS↑, 0フォノン) → ⟨σ_z(t)⟩ を厳密計算

Trotter 回路 (1 ステップ):
  RZ(ε·dt, 0) · RX(Δ·dt, 0) · RZ(ω·dt, 1)
  · [H₁ · CNOT(0→1) · RZ(2g·dt, 1) · CNOT(0→1) · H₁]
  → 2 CNOT/ステップ, N=20 → 40 CNOT/回路

使い方:
  python pennylane_spin_boson.py simulate
  python pennylane_spin_boson.py ibm_sim
  python pennylane_spin_boson.py ibm_real <TOKEN>
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── パラメータ ──────────────────────────────────────────────
EPS0    = 0.0    # TLS バイアス (対称: ε=0)
DELTA   = 0.5    # TLS トンネル振幅
OMEGA   = 1.0    # フォノン周波数
T_FINAL = 4.0    # 最終時刻 (~0.6 Rabi 周期)
N_STEPS = 20     # Trotter ステップ数 (2 CNOT/step → 40 CNOT/回路)
SHOTS   = 8192   # shots / 回路

G_VALUES = np.array([0.20, 0.35, 0.50, 0.60, 0.80])
# α = 2g²/ω²:    [0.08, 0.245, 0.50, 0.72, 1.28]


# ── 補助関数 ────────────────────────────────────────────────
def kondo_alpha(g: float, omega: float = OMEGA) -> float:
    return 2.0 * g**2 / omega**2


def exact_sz(g: float, t: float,
             eps0: float = EPS0, delta: float = DELTA,
             omega: float = OMEGA) -> float:
    """⟨ZI⟩(t) を 4×4 厳密対角化で計算"""
    H = np.zeros((4, 4), dtype=complex)
    # (ε/2)·ZI: +ε/2 for |0?>, -ε/2 for |1?>
    H[0, 0] += eps0 / 2;  H[1, 1] += eps0 / 2
    H[2, 2] -= eps0 / 2;  H[3, 3] -= eps0 / 2
    # (ω/2)·IZ: +ω/2 for |?0>, -ω/2 for |?1>
    H[0, 0] += omega / 2; H[1, 1] -= omega / 2
    H[2, 2] += omega / 2; H[3, 3] -= omega / 2
    # (Δ/2)·XI: |0?><1?| + h.c.
    H[0, 2] += delta / 2; H[2, 0] += delta / 2
    H[1, 3] += delta / 2; H[3, 1] += delta / 2
    # g·ZX (Z₀⊗X₁): Z=+1 sector swaps with +g, Z=-1 sector swaps with -g
    H[0, 1] += g; H[1, 0] += g
    H[2, 3] -= g; H[3, 2] -= g

    E, V = np.linalg.eigh(H)
    psi0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    c = V.conj().T @ psi0
    psi_t = V @ (c * np.exp(-1j * E * t))
    prob = np.abs(psi_t) ** 2
    return float((prob[0] + prob[1]) - (prob[2] + prob[3]))


# ── 回路構築 ────────────────────────────────────────────────
def _build_circuit(dev, g: float,
                   n_steps: int = N_STEPS, t_final: float = T_FINAL,
                   eps0: float = EPS0, delta: float = DELTA,
                   omega: float = OMEGA):
    """Spin-Boson Trotter 回路 (2-qubit, 2 CNOT/step)"""
    import pennylane as qml

    dt = t_final / n_steps

    @qml.qnode(dev)
    def circuit():
        for _ in range(n_steps):
            # exp(-i·(ε/2)·ZI·dt) = RZ(ε·dt, wire=0)
            if eps0 != 0.0:
                qml.RZ(float(eps0 * dt), wires=0)
            # exp(-i·(Δ/2)·XI·dt) = RX(Δ·dt, wire=0)
            qml.RX(float(delta * dt), wires=0)
            # exp(-i·(ω/2)·IZ·dt) = RZ(ω·dt, wire=1)
            qml.RZ(float(omega * dt), wires=1)
            # exp(-i·g·ZX·dt) = H₁ · CNOT(0,1) · RZ(2g·dt, 1) · CNOT(0,1) · H₁
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RZ(float(2.0 * g * dt), wires=1)
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=1)
        return qml.probs(wires=[0, 1])

    return circuit


# ── simulate モード ─────────────────────────────────────────
def run_simulate(g_vals=G_VALUES, n_time: int = 30) -> dict:
    """状態ベクトル: 時系列 ⟨ZI⟩(t) を計算"""
    import pennylane as qml

    t_arr = np.linspace(0.0, T_FINAL, n_time + 1)
    results = {}

    for g in g_vals:
        sz_sim = []
        for t in t_arr:
            if t == 0.0:
                sz_sim.append(1.0)
                continue
            dev  = qml.device("default.qubit", wires=2)
            circ = _build_circuit(dev, g, n_steps=N_STEPS, t_final=t)
            probs = circ()
            sz = float((probs[0] + probs[1]) - (probs[2] + probs[3]))
            sz_sim.append(sz)
        results[float(g)] = (t_arr, np.array(sz_sim))
    return results


# ── ibm_sim モード ──────────────────────────────────────────
def run_ibm_sim(g_vals=G_VALUES, shots: int = SHOTS) -> dict:
    """Qiskit AerSimulator (ノイズモデル付き) で ⟨ZI⟩(T_FINAL)"""
    import pennylane as qml
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (NoiseModel, depolarizing_error,
                                   ReadoutError)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(0.001, 1), ["rx", "rz", "h"]
    )
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(0.01, 2), ["cx"]
    )
    ro_err = ReadoutError([[0.99, 0.01], [0.01, 0.99]])
    noise_model.add_all_qubit_readout_error(ro_err)

    backend = AerSimulator(noise_model=noise_model)
    results = {}

    for g in g_vals:
        dev   = qml.device("qiskit.aer", wires=2, shots=shots, backend=backend)
        circ  = _build_circuit(dev, g)
        probs = circ()
        sz    = float((probs[0] + probs[1]) - (probs[2] + probs[3]))
        results[float(g)] = sz
    return results


# ── ibm_real モード ─────────────────────────────────────────
def run_ibm_real(token: str, g_vals=G_VALUES, shots: int = SHOTS) -> dict:
    """IBM 実機で ⟨ZI⟩(T_FINAL)"""
    import pennylane as qml
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
    )
    backend = service.least_busy(
        operational=True, simulator=False, min_num_qubits=2
    )
    print(f"  -> 使用バックエンド: {backend.name}")

    results = {}
    for g in g_vals:
        dev   = qml.device("qiskit.remote", wires=2, backend=backend, shots=shots)
        circ  = _build_circuit(dev, g)
        probs = circ()
        sz    = float((probs[0] + probs[1]) - (probs[2] + probs[3]))
        sz_ex = exact_sz(g, T_FINAL)
        alpha = kondo_alpha(g)
        print(f"  g={g:.2f} (α={alpha:.3f}): "
              f"⟨ZI⟩={sz:+.4f}  (exact={sz_ex:+.4f}  |Δ|={abs(sz-sz_ex):.4f})")
        results[float(g)] = sz
    return results


# ── 結果出力 ────────────────────────────────────────────────
def print_results(mode: str, results: dict, g_vals=G_VALUES):
    print(f"\n{'='*70}")
    print(f"  Spin-Boson ⟨ZI⟩(T={T_FINAL})  [mode={mode}]")
    print(f"  H = (ε/2)ZI + (Δ/2)XI + (ω/2)IZ + g·ZX")
    print(f"  ε={EPS0}, Δ={DELTA}, ω={OMEGA}, T={T_FINAL}, N={N_STEPS}")
    print(f"{'='*70}")

    errs = []
    print(f"\n  {'g':>5}  {'α':>6}  {'⟨ZI⟩(T)':>10}  {'exact':>10}  "
          f"{'|誤差|':>8}  {'誤差(%)':>8}  {'状態':>10}")
    print("  " + "-" * 66)

    for g in g_vals:
        alpha = kondo_alpha(g)
        sz_ex = exact_sz(g, T_FINAL)
        if mode == "simulate":
            t_arr, sz_arr = results[float(g)]
            sz = float(sz_arr[-1])
        else:
            sz = results[float(g)]
        err  = abs(sz - sz_ex)
        errs.append(err)
        phase = "コヒーレント" if alpha < 0.5 else ("Toulouse" if abs(alpha - 0.5) < 0.01 else "局在化")
        print(f"  {g:>5.2f}  {alpha:>6.3f}  {sz:>+10.4f}  {sz_ex:>+10.4f}  "
              f"{err:>8.4f}  {err*100:>8.2f}%  {phase:>10}")

    print(f"\n  平均絶対誤差: {np.mean(errs):.4f}")
    print(f"  最大絶対誤差: {np.max(errs):.4f}")
    print(f"\n  Trotter 仕様: CNOT={2*N_STEPS}/回路, dt={T_FINAL/N_STEPS:.3f}, "
          f"shots={SHOTS:,}, 回路数={len(g_vals)}")


# ── プロット ────────────────────────────────────────────────
def plot_results(mode: str, results: dict, g_vals=G_VALUES):
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(g_vals)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Spin-Boson Model: Coherent vs Localized  [mode={mode}]\n"
        r"$H = \frac{\varepsilon}{2}ZI + \frac{\Delta}{2}XI + \frac{\omega}{2}IZ + g\cdot ZX$"
        f",  ε={EPS0}, Δ={DELTA}, ω={OMEGA}, T={T_FINAL}",
        fontsize=10,
    )

    ax = axes[0]
    alpha_vals = [kondo_alpha(g) for g in g_vals]

    if mode == "simulate":
        # 時系列プロット
        t_fine = np.linspace(0, T_FINAL, 200)
        for i, g in enumerate(g_vals):
            alpha = kondo_alpha(g)
            t_arr, sz_arr = results[float(g)]
            sz_ex_fine = [exact_sz(g, t) for t in t_fine]
            ax.plot(t_fine, sz_ex_fine, color=colors[i], lw=1.5, alpha=0.5)
            ax.plot(t_arr, sz_arr, color=colors[i], lw=2, marker="o", ms=4,
                    label=f"g={g:.2f} α={alpha:.2f}")
        ax.set_xlabel("Time t", fontsize=12)
        ax.set_ylabel(r"$\langle\sigma_z(t)\rangle$", fontsize=12)
        ax.set_title("TLS coherence (solid=Trotter, faded=exact)", fontsize=10)
        ax.axhline(0, color="gray", ls="--", lw=1, alpha=0.5)
        ax.legend(fontsize=8)
        ax.set_ylim(-1.1, 1.15)
    else:
        # g 依存性プロット
        g_fine = np.linspace(0.1, 0.9, 200)
        sz_ex_fine = [exact_sz(g, T_FINAL) for g in g_fine]
        ax.plot(g_fine, sz_ex_fine, "b-", lw=2, label=f"Exact (T={T_FINAL})")
        sz_meas = [results[float(g)] for g in g_vals]
        sz_ex_pts = [exact_sz(g, T_FINAL) for g in g_vals]
        ax.errorbar(g_vals, sz_meas, yerr=2.0 / np.sqrt(SHOTS),
                    fmt="o", color="orange", ms=8, capsize=4,
                    elinewidth=1.5, markeredgecolor="black",
                    label=f"IBM {mode}")
        ax.plot(g_vals, sz_ex_pts, "bs", ms=6, label="Exact at g_i")
        ax.set_xlabel(r"$g$ (coupling strength)", fontsize=12)
        ax.set_ylabel(r"$\langle\sigma_z(T)\rangle$", fontsize=12)
        ax.set_title(f"TLS coherence at T={T_FINAL}", fontsize=10)
        ax.legend(fontsize=9)
        ax.axhline(0, color="gray", ls="--", lw=1, alpha=0.5)
        ax.set_ylim(-1.1, 1.15)

    ax.grid(True, alpha=0.3)

    # 右パネル: Kondo α 依存性
    ax2 = axes[1]
    alpha_fine = np.array([kondo_alpha(g) for g in np.linspace(0.1, 0.9, 200)])
    sz_alpha_fine = [exact_sz(g, T_FINAL) for g in np.linspace(0.1, 0.9, 200)]
    ax2.plot(alpha_fine, sz_alpha_fine, "b-", lw=2, label=f"Exact (T={T_FINAL})")

    if mode == "simulate":
        sz_pts = [results[float(g)][1][-1] for g in g_vals]
    else:
        sz_pts = [results[float(g)] for g in g_vals]
    sz_ex_pts = [exact_sz(g, T_FINAL) for g in g_vals]
    ax2.scatter(alpha_vals, sz_ex_pts, c=colors, s=80, zorder=5,
                edgecolor="black", marker="s", label="Exact at α_i")
    ax2.scatter(alpha_vals, sz_pts, c=colors, s=100, zorder=6,
                edgecolor="black", marker="o",
                label=f"{'Trotter' if mode=='simulate' else f'IBM {mode}'}")

    ax2.axvline(0.5, color="orange", ls=":", lw=2, label="Toulouse α=0.5")
    ax2.axhline(0, color="gray", ls="--", lw=1, alpha=0.5)
    ax2.fill_betweenx([-1.1, 1.15], 0, 0.5, alpha=0.05, color="blue")
    ax2.fill_betweenx([-1.1, 1.15], 0.5, 1.5, alpha=0.05, color="red")
    ax2.text(0.25, 1.05, "コヒーレント", ha="center", fontsize=9, color="blue")
    ax2.text(0.75, 1.05, "局在化", ha="center", fontsize=9, color="red")
    ax2.set_xlabel("α = 2g²/ω² (Kondo parameter)", fontsize=11)
    ax2.set_ylabel(r"$\langle\sigma_z(T)\rangle$", fontsize=12)
    ax2.set_title("Coherent-to-Localized Crossover", fontsize=10)
    ax2.set_xlim(0, 1.4)
    ax2.set_ylim(-1.1, 1.15)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname     = f"spin_boson_{mode}.png"
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  プロット保存: {save_path}")


# ── メイン ──────────────────────────────────────────────────
def main():
    alpha_str = [f"{kondo_alpha(g):.3f}" for g in G_VALUES]
    print("=" * 70)
    print("  Spin-Boson モデル")
    print("  H = (ε/2)·ZI + (Δ/2)·XI + (ω/2)·IZ + g·ZX")
    print(f"  ε={EPS0}, Δ={DELTA}, ω={OMEGA}, T={T_FINAL}, N={N_STEPS} (40 CNOT/回路)")
    print(f"  g    = {list(G_VALUES)}")
    print(f"  α    = {alpha_str}")
    print("=" * 70)

    if len(sys.argv) < 2:
        print("\n使い方:")
        print("  python pennylane_spin_boson.py simulate")
        print("  python pennylane_spin_boson.py ibm_sim")
        print("  python pennylane_spin_boson.py ibm_real <TOKEN>")
        sys.exit(1)

    mode = sys.argv[1]
    print(f"\n[Mode: {mode}]")

    if mode == "simulate":
        results = run_simulate()
    elif mode == "ibm_sim":
        results = run_ibm_sim()
    elif mode == "ibm_real":
        if len(sys.argv) < 3:
            print("ERROR: ibm_real モードには TOKEN が必要です")
            sys.exit(1)
        results = run_ibm_real(sys.argv[2])
    else:
        print(f"ERROR: 不明なモード '{mode}'")
        sys.exit(1)

    print_results(mode, results)
    plot_results(mode, results)


if __name__ == "__main__":
    main()
