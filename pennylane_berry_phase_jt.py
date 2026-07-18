"""
Jahn-Teller Berry Phase via Wilson Loop (IBM Quantum)
======================================================
H(θ) = R·cos(θ)·ZI + R·sin(θ)·XX

Block 1 の基底状態: |ψ_-(θ)⟩ = -sin(θ/2)|00⟩ + cos(θ/2)|11⟩

有効 1-qubit 表現:
  |ψ_eff(θ)⟩ = -sin(θ/2)|0⟩ + cos(θ/2)|1⟩
  準備回路: Ry(π+θ, wire=1)

Wilson Loop:
  W = Π_{k=0}^{M-1} ⟨ψ_eff(θ_k)|ψ_eff(θ_{k+1})⟩
    = -(cos(π/M))^M  [解析値]
  γ = arg(W) = π  (E×e JT の Berry 位相)

Hadamard テスト回路 (2 qubit: ancilla=0, system=1):
  Ry(π+θ_k, 1) → H(0) → CRY(Δθ, 0→1) → H(0) → measure(0)
  P(|0⟩) = (1 + ⟨ψ_k|ψ_{k+1}⟩) / 2
  CNOTs: 2 per circuit  (vs 40 in time-evolution mode)

使い方:
  python pennylane_berry_phase_jt.py simulate
  python pennylane_berry_phase_jt.py ibm_sim
  python pennylane_berry_phase_jt.py ibm_real <TOKEN>
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── パラメータ ──────────────────────────────────────────────
M      = 8       # Wilson ループ分割数
R      = 0.5     # パラメータ空間のループ半径
SHOTS  = 8192    # shots / 回路


# ── 解析関数 ────────────────────────────────────────────────
def exact_overlap(theta_k: float, theta_k1: float) -> float:
    """⟨ψ_eff(θ_k)|ψ_eff(θ_k+1)⟩ の解析値"""
    return float(np.cos((theta_k1 - theta_k) / 2.0))


def exact_berry_phase(M: int) -> float:
    """Wilson Loop の解析 Berry 位相 = π (全 M で厳密)"""
    W = -(np.cos(np.pi / M)) ** M
    return float(np.angle(W))


# ── simulate モード ─────────────────────────────────────────
def run_simulate(M: int = M) -> tuple:
    """状態ベクトルで Wilson Loop を直接計算"""
    import pennylane as qml

    dev    = qml.device("default.qubit", wires=2)
    thetas = np.linspace(0, 2 * np.pi, M, endpoint=False)

    overlaps = []
    p0_list  = []

    for k in range(M):
        theta_k  = float(thetas[k])
        theta_k1 = float(thetas[(k + 1) % M])
        alpha_k  = np.pi + theta_k
        delta_a  = theta_k1 - theta_k   # 正規: +2π/M、wrap: -(2π-2π/M)

        @qml.qnode(dev)
        def circuit():
            qml.RY(float(alpha_k), wires=1)
            qml.Hadamard(wires=0)
            qml.CRY(float(delta_a), wires=[0, 1])
            qml.Hadamard(wires=0)
            return qml.probs(wires=[0])

        probs   = circuit()
        p0      = float(probs[0])
        overlap = 2.0 * p0 - 1.0
        p0_list.append(p0)
        overlaps.append(overlap)

    W     = complex(np.prod(overlaps))
    gamma = float(np.angle(W))
    return gamma, overlaps, p0_list


# ── ibm_sim モード ──────────────────────────────────────────
def run_ibm_sim(M: int = M, shots: int = SHOTS) -> tuple:
    """Qiskit AerSimulator (ノイズなし) で Wilson Loop"""
    import pennylane as qml

    thetas   = np.linspace(0, 2 * np.pi, M, endpoint=False)
    overlaps = []
    p0_list  = []

    for k in range(M):
        theta_k  = float(thetas[k])
        theta_k1 = float(thetas[(k + 1) % M])
        alpha_k  = np.pi + theta_k
        delta_a  = theta_k1 - theta_k

        dev = qml.device("qiskit.aer", wires=2, shots=shots)

        @qml.qnode(dev)
        def circuit():
            qml.RY(float(alpha_k), wires=1)
            qml.Hadamard(wires=0)
            qml.CRY(float(delta_a), wires=[0, 1])
            qml.Hadamard(wires=0)
            return qml.probs(wires=[0])

        probs   = circuit()
        p0      = float(probs[0])
        overlap = 2.0 * p0 - 1.0
        p0_list.append(p0)
        overlaps.append(overlap)

    W     = complex(np.prod(overlaps))
    gamma = float(np.angle(W))
    return gamma, overlaps, p0_list


# ── ibm_real モード ─────────────────────────────────────────
def get_ibm_backend(token: str):
    """least_busy IBM バックエンド (≥2 qubit) を取得"""
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
    )
    backend = service.least_busy(
        operational=True, simulator=False, min_num_qubits=2
    )
    return backend


def run_ibm_real(token: str, M: int = M, shots: int = SHOTS) -> tuple:
    """IBM 実機で Wilson Loop"""
    import pennylane as qml

    backend  = get_ibm_backend(token)
    print(f"  -> 使用バックエンド: {backend.name}")

    thetas   = np.linspace(0, 2 * np.pi, M, endpoint=False)
    overlaps = []
    p0_list  = []

    for k in range(M):
        theta_k  = float(thetas[k])
        theta_k1 = float(thetas[(k + 1) % M])
        alpha_k  = np.pi + theta_k
        delta_a  = theta_k1 - theta_k

        dev = qml.device("qiskit.remote", wires=2, backend=backend, shots=shots)

        @qml.qnode(dev)
        def circuit():
            qml.RY(float(alpha_k), wires=1)
            qml.Hadamard(wires=0)
            qml.CRY(float(delta_a), wires=[0, 1])
            qml.Hadamard(wires=0)
            return qml.probs(wires=[0])

        probs   = circuit()
        p0      = float(probs[0])
        overlap = 2.0 * p0 - 1.0
        p0_list.append(p0)
        overlaps.append(overlap)
        print(f"  k={k}: θ={np.degrees(theta_k):.1f}°, P(|0⟩)={p0:.4f}, "
              f"overlap={overlap:+.4f}  (exact={exact_overlap(theta_k, theta_k1):+.4f})")

    W     = complex(np.prod(overlaps))
    gamma = float(np.angle(W))
    return gamma, overlaps, p0_list


# ── 結果出力 ────────────────────────────────────────────────
def print_results(mode: str, gamma: float, overlaps: list, p0_list: list, M: int = M):
    thetas = np.linspace(0, 2 * np.pi, M, endpoint=False)

    print(f"\n{'='*60}")
    print(f"  Berry 位相 Wilson Loop 結果  (M={M})")
    print(f"{'='*60}")
    print(f"\n  {'k':>2}  {'θ_k (°)':>8}  {'P(|0⟩)':>8}  "
          f"{'overlap':>8}  {'exact':>8}  {'差':>8}")
    print("  " + "-" * 54)

    for k in range(M):
        theta_k  = float(thetas[k])
        theta_k1 = float(thetas[(k + 1) % M])
        ov_ex    = exact_overlap(theta_k, theta_k1)
        sign_str = "← wrap" if k == M - 1 else ""
        print(f"  {k:>2}  {np.degrees(theta_k):>8.1f}  {p0_list[k]:>8.4f}  "
              f"{overlaps[k]:>+8.4f}  {ov_ex:>+8.4f}  "
              f"{overlaps[k]-ov_ex:>+8.4f}  {sign_str}")

    W_meas    = complex(np.prod(overlaps))
    gamma_ex  = exact_berry_phase(M)
    err       = abs(abs(gamma) - np.pi)

    print(f"\n  Wilson loop W = {W_meas.real:+.4f}{W_meas.imag:+.4f}i")
    print(f"  Berry 位相 γ  = {gamma:.4f} rad  ({np.degrees(gamma):.2f}°)")
    print(f"  理論値  γ     = {gamma_ex:.4f} rad  ({np.degrees(gamma_ex):.2f}°)")
    print(f"  誤差 |γ| - π  = {err:.4f} rad  ({err/np.pi*100:.2f}%)")
    print(f"\n  判定: {'✓ Berry 位相 π 確認 (非自明な位相)' if err < 0.3 else '△ 誤差が大きい (ノイズ影響)'}")


# ── プロット ────────────────────────────────────────────────
def plot_results(mode: str, gamma: float, overlaps: list, p0_list: list, M: int = M):
    thetas   = np.linspace(0, 2 * np.pi, M, endpoint=False)
    theta_deg = np.degrees(thetas)
    exact_p0 = [(1 + exact_overlap(float(thetas[k]), float(thetas[(k+1) % M]))) / 2
                for k in range(M)]
    exact_ov = [exact_overlap(float(thetas[k]), float(thetas[(k+1) % M]))
                for k in range(M)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"JT Berry Phase — Wilson Loop (M={M}, mode={mode})\n"
        r"$H(\theta) = R\cos\theta\cdot ZI + R\sin\theta\cdot XX$, "
        f"$R={R}$",
        fontsize=12,
    )

    # (1) P(|0⟩) vs θ
    ax = axes[0]
    ax.plot(theta_deg, exact_p0, "b-", lw=2, label="Exact P(|0⟩)")
    ax.scatter(theta_deg, p0_list, c=["red" if k == M-1 else "orange" for k in range(M)],
               s=80, zorder=5)
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6, label="P=0.5 (orthogonal)")
    # annotate wrap-around
    ax.annotate("wrap-around\n(P<0.5 → overlap<0)",
                xy=(theta_deg[-1], p0_list[-1]),
                xytext=(180, 0.3), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="red"), color="red")
    ax.set_xlabel("θ (°)")
    ax.set_ylabel("P(ancilla=|0⟩)")
    ax.set_title("Hadamard test: P(|0⟩)")
    ax.legend(fontsize=9)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # (2) overlap vs θ
    ax2 = axes[1]
    ax2.plot(theta_deg, exact_ov, "b-", lw=2, label="Exact overlap")
    colors = ["red" if k == M-1 else "orange" for k in range(M)]
    ax2.scatter(theta_deg, overlaps, c=colors, s=80, zorder=5, label="Measured")
    ax2.axhline(0, color="gray", ls="--", lw=1, alpha=0.6)
    ax2.set_xlabel("θ_k (°)")
    ax2.set_ylabel(r"$\langle\psi_k|\psi_{k+1}\rangle$")
    ax2.set_title("Wilson loop overlaps")
    ax2.legend(fontsize=9)
    ax2.set_xticks([0, 90, 180, 270, 360])
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, alpha=0.3)

    # (3) Berry phase result
    ax3 = axes[2]
    gamma_ex = exact_berry_phase(M)
    ax3.bar(["Exact", "Measured"], [abs(gamma_ex), abs(gamma)],
            color=["blue", "orange"], alpha=0.8, edgecolor="black")
    ax3.axhline(np.pi, color="red", ls="--", lw=2, label=f"π = {np.pi:.4f}")
    ax3.set_ylabel("Berry phase |γ| (rad)")
    ax3.set_title("Berry Phase γ = π")
    ax3.set_ylim(0, np.pi + 0.5)
    ax3.legend(fontsize=9)
    ax3.text(1, abs(gamma) + 0.05, f"{abs(gamma):.3f} rad\n({np.degrees(abs(gamma)):.1f}°)",
             ha="center", fontsize=10, color="darkred")
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fname = f"berry_phase_jt_{mode}.png"
    plt.savefig(f"/home/user/JAX_Test/{fname}", dpi=150, bbox_inches="tight")
    print(f"\n  プロット保存: {fname}")


# ── メイン ──────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Jahn-Teller Berry 位相 — Wilson Loop 測定")
    print(f"  H(θ) = R·cos(θ)·ZI + R·sin(θ)·XX  (R={R})")
    print(f"  M={M} 点, SHOTS={SHOTS}")
    print(f"  理論値: γ = π = {np.pi:.6f} rad")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("\n使い方:")
        print("  python pennylane_berry_phase_jt.py simulate")
        print("  python pennylane_berry_phase_jt.py ibm_sim")
        print("  python pennylane_berry_phase_jt.py ibm_real <TOKEN>")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "simulate":
        print("\n[Mode: シミュレーション (state vector)]")
        print("  回路: Ry(π+θ) → H → CRY(Δθ) → H  (2 CNOT/回路)")
        gamma, overlaps, p0_list = run_simulate(M)

    elif mode == "ibm_sim":
        print(f"\n[Mode: IBM AerSimulator (shots={SHOTS})]")
        gamma, overlaps, p0_list = run_ibm_sim(M, SHOTS)

    elif mode == "ibm_real":
        if len(sys.argv) < 3:
            print("ERROR: ibm_real モードには TOKEN が必要です")
            sys.exit(1)
        token = sys.argv[2]
        print(f"\n[Mode: IBM 実機 (shots={SHOTS})]")
        gamma, overlaps, p0_list = run_ibm_real(token, M, SHOTS)

    else:
        print(f"ERROR: 不明なモード '{mode}'")
        sys.exit(1)

    print_results(mode, gamma, overlaps, p0_list, M)
    plot_results(mode, gamma, overlaps, p0_list, M)


if __name__ == "__main__":
    main()
