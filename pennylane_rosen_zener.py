"""
Rosen-Zener (RZ) 非断熱遷移モデル
====================================
H(t) = (ε₀/2)·Z + (Δ₀·sech(t/τ)/2)·X

厳密遷移確率 (|0⟩ → |1⟩):
  P_RZ = sin²(π·Δ₀·τ/2) / cosh²(π·ε₀·τ/2)

特徴:
  - Landau-Zener と異なり P は Δ₀τ について周期振動
  - ε₀=0 (共鳴): P = sin²(π·Δ₀·τ/2)  [完全反転が周期的に起きる]
  - 1-qubit のみ → CNOT = 0

Trotter 回路 (1 ステップ):
  RZ(ε₀·dt, 0) → RX(Δ₀·sech(tₖ/τ)·dt, 0)
  → 1-qubit ゲートのみ!

使い方:
  python pennylane_rosen_zener.py simulate
  python pennylane_rosen_zener.py ibm_sim
  python pennylane_rosen_zener.py ibm_real <TOKEN>
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── パラメータ ──────────────────────────────────────────────
TAU      = 1.0   # パルス幅
T_TOTAL  = 10.0  # 総時間 (±5τ をカバー)
N_STEPS  = 50    # Trotter ステップ数 (誤差 < 0.02%)
SHOTS    = 8192  # shots / 回路

D0_VALUES  = np.array([0.5, 1.0, 1.5, 2.0, 2.5])   # パルス振幅
EPS0_VALUES = np.array([0.0, 0.5, 1.0])              # 静的離調


# ── 厳密解 ──────────────────────────────────────────────────
def exact_p_rz(D0: float, eps0: float, tau: float = TAU) -> float:
    """P = sin²(π·Δ₀·τ/2) / cosh²(π·ε₀·τ/2)"""
    return float(np.sin(np.pi * D0 * tau / 2) ** 2
                 / np.cosh(np.pi * eps0 * tau / 2) ** 2)


# ── 回路構築ヘルパー ────────────────────────────────────────
def _build_circuit(dev, D0: float, eps0: float,
                   n_steps: int = N_STEPS,
                   t_total: float = T_TOTAL,
                   tau: float = TAU):
    """Trotter 回路を返す (1-qubit, CNOT=0)"""
    import pennylane as qml

    dt  = t_total / n_steps
    # ステップ中心時刻
    ts  = np.linspace(-t_total / 2 + dt / 2, t_total / 2 - dt / 2, n_steps)
    # sech(tₖ/τ) を事前計算
    sech_vals = 1.0 / np.cosh(ts / tau)

    @qml.qnode(dev)
    def circuit():
        for k in range(n_steps):
            qml.RZ(float(eps0 * dt), wires=0)
            qml.RX(float(D0 * sech_vals[k] * dt), wires=0)
        return qml.probs(wires=[0])

    return circuit


# ── simulate モード ─────────────────────────────────────────
def run_simulate(D0_vals=D0_VALUES, eps0_vals=EPS0_VALUES) -> dict:
    import pennylane as qml

    results = {}
    for eps0 in eps0_vals:
        probs_list = []
        for D0 in D0_vals:
            dev    = qml.device("default.qubit", wires=1)
            circ   = _build_circuit(dev, D0, eps0)
            probs  = circ()
            probs_list.append(float(probs[1]))
        results[float(eps0)] = np.array(probs_list)
    return results


# ── ibm_sim モード ──────────────────────────────────────────
def run_ibm_sim(D0_vals=D0_VALUES, eps0_vals=EPS0_VALUES,
                shots: int = SHOTS) -> dict:
    import pennylane as qml
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (NoiseModel, depolarizing_error,
                                   ReadoutError)

    # ノイズモデル
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(0.001, 1), ["rx", "ry", "rz"]
    )
    ro_err = ReadoutError([[0.99, 0.01], [0.01, 0.99]])
    noise_model.add_all_qubit_readout_error(ro_err)

    backend = AerSimulator(noise_model=noise_model)

    results = {}
    for eps0 in eps0_vals:
        probs_list = []
        for D0 in D0_vals:
            dev   = qml.device("qiskit.aer", wires=1, shots=shots,
                               backend=backend)
            circ  = _build_circuit(dev, D0, eps0)
            probs = circ()
            probs_list.append(float(probs[1]))
        results[float(eps0)] = np.array(probs_list)
    return results


# ── ibm_real モード ─────────────────────────────────────────
def run_ibm_real(token: str, D0_vals=D0_VALUES, eps0_vals=EPS0_VALUES,
                 shots: int = SHOTS) -> dict:
    import pennylane as qml
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
    )
    backend = service.least_busy(
        operational=True, simulator=False, min_num_qubits=1
    )
    print(f"  -> 使用バックエンド: {backend.name}")

    results = {}
    for eps0 in eps0_vals:
        probs_list = []
        for D0 in D0_vals:
            dev   = qml.device("qiskit.remote", wires=1,
                               backend=backend, shots=shots)
            circ  = _build_circuit(dev, D0, eps0)
            probs = circ()
            p1    = float(probs[1])
            p_ex  = exact_p_rz(D0, eps0)
            print(f"  ε₀={eps0:.1f}, Δ₀={D0:.1f}: "
                  f"P={p1:.4f}  (exact={p_ex:.4f}  "
                  f"|Δ|={abs(p1-p_ex):.4f})")
            probs_list.append(p1)
        results[float(eps0)] = np.array(probs_list)
    return results


# ── 結果出力 ────────────────────────────────────────────────
def print_results(mode: str, results: dict,
                  D0_vals=D0_VALUES, eps0_vals=EPS0_VALUES):
    print(f"\n{'='*65}")
    print(f"  Rosen-Zener 遷移確率 P(|0⟩→|1⟩)  [mode={mode}]")
    print(f"  厳密解: P = sin²(π·Δ₀·τ/2) / cosh²(π·ε₀·τ/2)")
    print(f"{'='*65}")

    for eps0 in eps0_vals:
        print(f"\n  ε₀ = {eps0:.1f}")
        print(f"  {'Δ₀':>6}  {'P_sim':>8}  {'P_exact':>8}  {'|誤差|':>8}  {'誤差(%)':>8}")
        print("  " + "-"*46)
        p_arr = results[float(eps0)]
        for i, D0 in enumerate(D0_vals):
            p_ex  = exact_p_rz(D0, eps0)
            err   = abs(p_arr[i] - p_ex)
            print(f"  {D0:>6.2f}  {p_arr[i]:>8.4f}  {p_ex:>8.4f}  "
                  f"{err:>8.4f}  {err*100:>8.2f}%")

    # 全体統計
    all_p   = np.concatenate([results[e] for e in eps0_vals])
    all_ex  = np.array([exact_p_rz(D0, eps0)
                        for eps0 in eps0_vals for D0 in D0_vals])
    abs_err = np.abs(all_p - all_ex)
    print(f"\n  平均絶対誤差: {abs_err.mean():.4f}")
    print(f"  最大絶対誤差: {abs_err.max():.4f}  "
          f"(ε₀={eps0_vals[abs_err.argmax()//len(D0_vals)]:.1f}, "
          f"Δ₀={D0_vals[abs_err.argmax()%len(D0_vals)]:.1f})")
    print(f"  平均相対誤差: {(abs_err/np.maximum(all_ex,0.01)).mean()*100:.2f}%")

    print(f"\n  回路仕様: CNOT=0, gates={2*N_STEPS}/回路, "
          f"shots={SHOTS:,}, 回路数={len(D0_vals)*len(eps0_vals)}")


# ── プロット ────────────────────────────────────────────────
def plot_results(mode: str, results: dict,
                 D0_vals=D0_VALUES, eps0_vals=EPS0_VALUES):
    D0_fine  = np.linspace(0.01, 3.0, 300)
    colors   = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    markers  = ["o", "s", "^"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Rosen-Zener Model: IBM {mode} vs Exact Theory\n"
        r"$H(t)=\frac{\varepsilon_0}{2}Z+\frac{\Delta_0\,\mathrm{sech}(t/\tau)}{2}X$, "
        f"$\\tau={TAU}$, N={N_STEPS} steps, shots={SHOTS:,}",
        fontsize=11,
    )

    # (1) P vs Δ₀ for each ε₀
    ax = axes[0]
    for i, eps0 in enumerate(eps0_vals):
        p_ex_fine = [exact_p_rz(D0, eps0) for D0 in D0_fine]
        ax.plot(D0_fine, p_ex_fine, color=colors[i], lw=2,
                label=f"Exact ε₀={eps0:.1f}")
        ax.errorbar(D0_vals, results[float(eps0)],
                    yerr=1 / np.sqrt(SHOTS),
                    fmt=markers[i], color=colors[i], ms=8, capsize=4,
                    elinewidth=1.5, markeredgecolor="black",
                    label=f"IBM {mode} ε₀={eps0:.1f}")

    ax.set_xlabel(r"Pulse amplitude $\Delta_0$", fontsize=12)
    ax.set_ylabel(r"$P(|0\rangle \to |1\rangle)$", fontsize=12)
    ax.set_title("Transition probability vs coupling", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(0, 3.1)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    ax.axvline(1.0, color="gray", ls=":", alpha=0.5)
    ax.axvline(2.0, color="gray", ls=":", alpha=0.5)
    ax.text(1.0, 1.05, "Δ₀=1\nP=1", ha="center", fontsize=8, color="gray")
    ax.text(2.0, 1.05, "Δ₀=2\nP=0", ha="center", fontsize=8, color="gray")

    # (2) Absolute error
    ax2 = axes[1]
    x = np.arange(len(D0_vals))
    width = 0.25
    for i, eps0 in enumerate(eps0_vals):
        p_arr = results[float(eps0)]
        err   = np.abs(p_arr - np.array([exact_p_rz(D0, eps0)
                                         for D0 in D0_vals]))
        ax2.bar(x + i * width, err * 100, width, color=colors[i],
                alpha=0.8, label=f"ε₀={eps0:.1f}", edgecolor="black", lw=0.5)

    ax2.axhline(100 / np.sqrt(SHOTS), color="blue", ls="--", lw=1.5,
                label=f"Shot 1σ = {100/np.sqrt(SHOTS):.1f}%")
    ax2.axhline(1.0, color="orange", ls=":", lw=1.5,
                label="Readout ~1%")
    ax2.set_xlabel(r"Pulse amplitude $\Delta_0$", fontsize=12)
    ax2.set_ylabel("Absolute error (%)", fontsize=11)
    ax2.set_title("Error by parameter", fontsize=11)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f"{D0:.1f}" for D0 in D0_vals])
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, max(5.0, ax2.get_ylim()[1]))
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fname     = f"rosen_zener_{mode}.png"
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  プロット保存: {save_path}")


# ── メイン ──────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Rosen-Zener 非断熱遷移モデル")
    print(f"  H(t) = (ε₀/2)·Z + (Δ₀·sech(t/τ)/2)·X  (τ={TAU})")
    print(f"  厳密解: P = sin²(π·Δ₀·τ/2) / cosh²(π·ε₀·τ/2)")
    print(f"  Trotter: {N_STEPS} ステップ, CNOT=0, T={T_TOTAL}")
    print("=" * 65)

    if len(sys.argv) < 2:
        print("\n使い方:")
        print("  python pennylane_rosen_zener.py simulate")
        print("  python pennylane_rosen_zener.py ibm_sim")
        print("  python pennylane_rosen_zener.py ibm_real <TOKEN>")
        sys.exit(1)

    mode = sys.argv[1]

    print(f"\n[Mode: {mode}]")
    print(f"  Δ₀ = {D0_VALUES}")
    print(f"  ε₀ = {EPS0_VALUES}")

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
