"""最小 Jahn-Teller モデル: 2量子ビット vibronic 結合ダイナミクス

系のハミルトニアン:
    H = ε/2·ZI + ω/2·IZ + g·XX

量子ビット対応:
    q0: 電子状態  |0>=e_θ (基底),  |1>=e_ε (励起)
    q1: 振動モード (2 準位切断)  |0>=n=0,  |1>=n=1

初期状態: |00⟩ = 電子基底状態 + 振動 0 量子

4×4 行列は 2 ブロックに分解する:
    Block 1  {|00⟩, |11⟩}  →  E = ±√((ε+ω)²/4 + g²) = ±Ω₁
    Block 2  {|01⟩, |10⟩}  →  E = ±√((ε-ω)²/4 + g²) = ±Ω₂

初期状態 |00⟩ は Block 1 のみに属するため、厳密解 (解析式) が得られる:
    Ω₁ = √((ε+ω)²/4 + g²)           [一般化 Rabi 周波数]
    P(|11⟩, t) = (g/Ω₁)² sin²(Ω₁ t) [電子励起 + 1 フォノン遷移]

共鳴条件 ε + ω = 0 では Ω₁ = g となり完全 Rabi 振動:
    P(|11⟩, t) = sin²(g t)

円錐交差 (CoIn) の位置: (ε, g) = (0, 0) の 2 次元分岐空間

Trotter 回路 (1 ステップ, CX ゲート 2 個):
    q0: ─[RZ(ε·dt)]──●──[RX(2g·dt)]──●─
    q1: ─[RZ(ω·dt)]─[⊕]──────────────[⊕]─

ゲート数 / ステップ: 2 RZ + 2 CNOT + 1 RX = 5 ゲート

実行モード (コマンドライン引数で指定):
    python pennylane_minimal_jt.py simulate          # default.qubit + Snapshot
    python pennylane_minimal_jt.py catalyst          # @qjit + for_loop + vmap
    python pennylane_minimal_jt.py ibm_sim           # AerSimulator (ノイズあり)
    python pennylane_minimal_jt.py ibm_real <TOKEN>  # IBM 実機
"""

import sys
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane import qjit

warnings.filterwarnings("ignore")

# ============================================================
# 共通パラメータ
# ============================================================
EPSILON     = -1.0   # 電子エネルギー差  (ε = -ω = -1 で Block 1 共鳴)
OMEGA       = 1.0    # 振動周波数
T_TOTAL_SIM = 10.0   # 軌跡シミュレーション時間
T_TOTAL_HW  = 5.0    # IBM ハードウェア用の固定測定時間
N_STEPS_SIM = 150    # simulate モード (qml.exp + Snapshot)
N_STEPS_CAT = 200    # catalyst モード (Trotter)
N_STEPS_HW  = 20     # IBM ハードウェア用 → CNOT × 40
SHOTS_HW    = 8192

G_VALUES     = np.array([0.1, 0.3, 0.5, 0.7, 1.0])
G_VALUES_JAX = jnp.array(G_VALUES)


# ============================================================
# 厳密解 (解析式)
# ============================================================

def omega1(g, eps=EPSILON, omega=OMEGA):
    """Rabi 周波数: Ω₁ = √((ε+ω)²/4 + g²)"""
    return float(np.sqrt(((float(eps) + float(omega)) / 2.0) ** 2 + float(g) ** 2))


def exact_p11(t, g, eps=EPSILON, omega=OMEGA):
    """P(|11⟩, t) = (g/Ω₁)² sin²(Ω₁ t)"""
    O1 = omega1(g, eps, omega)
    if O1 == 0:
        return 0.0
    return (float(g) / O1) ** 2 * np.sin(O1 * float(t)) ** 2


def exact_eigenvalues(eps=EPSILON, omega=OMEGA, g=0.5):
    """4 つの固有エネルギーを返す"""
    O1 = np.sqrt(((eps + omega) / 2) ** 2 + g ** 2)
    O2 = np.sqrt(((eps - omega) / 2) ** 2 + g ** 2)
    return sorted([-O1, -O2, O2, O1])


# ============================================================
# Mode 1: default.qubit — qml.exp + Snapshot で時間発展軌跡
# ============================================================

def run_simulate(g_val, n_steps=N_STEPS_SIM, t_total=T_TOTAL_SIM):
    """qml.exp による厳密時間発展。各ステップ後に Snapshot を取得する。"""
    g   = float(g_val)
    dt  = t_total / n_steps
    dev = qml.device("default.qubit", wires=2)

    H = qml.Hamiltonian(
        [EPSILON / 2, OMEGA / 2, g],
        [qml.PauliZ(0), qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)],
    )

    @qml.qnode(dev)
    def circuit():
        # 初期状態 |00⟩ はデフォルト状態なのでゲート不要
        for _ in range(n_steps):
            qml.exp(H, -1j * dt)
            qml.Snapshot()
        return qml.probs(wires=[0, 1])

    snaps       = qml.snapshots(circuit)()
    final_probs = snaps.pop("execution_results")
    keys        = sorted(snaps.keys(), key=int)

    times = np.array([(i + 1) * dt for i in range(len(keys))])
    # スナップショットは statevector; 各基底状態の確率を計算
    p00 = np.array([np.abs(snaps[k][0]) ** 2 for k in keys])
    p01 = np.array([np.abs(snaps[k][1]) ** 2 for k in keys])
    p10 = np.array([np.abs(snaps[k][2]) ** 2 for k in keys])
    p11 = np.array([np.abs(snaps[k][3]) ** 2 for k in keys])

    return times, p00, p01, p10, p11, final_probs


def plot_trajectories(g_list=None):
    """複数の g 値の時間発展を 1 枚のグラフにまとめて保存する。"""
    if g_list is None:
        g_list = [0.3, 0.5, 1.0]

    fig, axes = plt.subplots(1, len(g_list), figsize=(5 * len(g_list), 4), sharey=True)
    if len(g_list) == 1:
        axes = [axes]

    for ax, g_val in zip(axes, g_list):
        times, _, _, _, p11, _ = run_simulate(g_val)
        p11_ex = np.array([exact_p11(t, g_val) for t in times])
        O1 = omega1(g_val)

        ax.plot(times, p11,    label="simulation (qml.exp)", lw=2)
        ax.plot(times, p11_ex, "--", label="exact: (g/Ω₁)²sin²(Ω₁t)", lw=1.5)
        ax.set_title(f"g = {g_val},  Ω₁ = {O1:.2f},  T_Rabi = {np.pi/O1:.2f}")
        ax.set_xlabel("time  t")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)

    axes[0].set_ylabel("P(|11⟩)  [電子励起 + 1 フォノン]")
    plt.suptitle(
        f"最小 JT モデル: H = ε/2·ZI + ω/2·IZ + g·XX\n"
        f"ε={EPSILON}, ω={OMEGA}  (共鳴条件 ε+ω={EPSILON+OMEGA:.1f})",
        fontsize=11,
    )
    plt.tight_layout()
    fname = "minimal_jt_trajectory.png"
    plt.savefig(fname, dpi=150)
    print(f"  → 軌跡プロット保存: {fname}")


# ============================================================
# Mode 2: Catalyst — @qjit + for_loop + jax.vmap
# ============================================================

def run_catalyst(g_values_jax=None, n_steps=N_STEPS_CAT, t_total=T_TOTAL_HW):
    if g_values_jax is None:
        g_values_jax = G_VALUES_JAX
    dev = qml.device("lightning.qubit", wires=2)
    dt  = t_total / n_steps

    @qjit
    @qml.qnode(dev)
    def circuit(g):
        def step(_, carry):
            # 1 次 Trotter 分解:
            # exp(-iε/2·Z₀·dt) = RZ(ε·dt) on q0
            # exp(-iω/2·Z₁·dt) = RZ(ω·dt) on q1
            # exp(-ig·X₀X₁·dt) = CNOT(c=0,t=1)·RX(2g·dt,q0)·CNOT(c=0,t=1)
            qml.RZ(EPSILON * dt, wires=0)
            qml.RZ(OMEGA   * dt, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RX(2.0 * g * dt, wires=0)
            qml.CNOT(wires=[0, 1])
            return carry

        qml.for_loop(0, n_steps, 1)(step)(None)
        return qml.probs(wires=[0, 1])

    t0    = time.time()
    probs = jax.vmap(circuit)(g_values_jax)
    t1    = time.time()
    print(f"  JIT コンパイル + 初回実行: {t1 - t0:.2f} s")
    t0    = time.time()
    probs = jax.vmap(circuit)(g_values_jax)
    t1    = time.time()
    print(f"  2回目 (JIT キャッシュ):    {t1 - t0:.3f} s")
    return probs


# ============================================================
# IBM ハードウェア / AerSimulator 共通: 静的展開 Trotter 回路
# ============================================================

def build_hw_qnode(device, g_val, n_steps=N_STEPS_HW, t_total=T_TOTAL_HW):
    """for ループを静的展開した Trotter 回路 (IBM トランスパイル対応)。"""
    dt = t_total / n_steps

    @qml.qnode(device)
    def circuit():
        for _ in range(n_steps):
            qml.RZ(float(EPSILON * dt), wires=0)
            qml.RZ(float(OMEGA   * dt), wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RX(float(2.0 * g_val * dt), wires=0)
            qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    return circuit


def circuit_specs_hw():
    dev  = qml.device("default.qubit", wires=2)
    circ = build_hw_qnode(dev, 0.5)
    return qml.specs(circ)()["resources"]


def get_aer_backend(gate_err_1q=0.001, gate_err_2q=0.01, readout_err=0.01):
    """IBM 実機相当のノイズモデル付き AerSimulator。"""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(
        depolarizing_error(gate_err_1q, 1), ["ry", "rz", "rx"]
    )
    nm.add_all_qubit_quantum_error(
        depolarizing_error(gate_err_2q, 2), ["cx"]
    )
    nm.add_all_qubit_readout_error(
        ReadoutError([[1 - readout_err, readout_err],
                      [readout_err, 1 - readout_err]])
    )
    return AerSimulator(noise_model=nm)


def get_ibm_backend(token, instance=None):
    from qiskit_ibm_runtime import QiskitRuntimeService

    kw = {"channel": "ibm_quantum_platform", "token": token}
    if instance:
        kw["instance"] = instance
    service = QiskitRuntimeService(**kw)
    backend = service.least_busy(
        operational=True, simulator=False, min_num_qubits=2
    )
    print(f"  -> 使用バックエンド: {backend.name}")
    return backend


# ============================================================
# 結果の整形表示
# ============================================================

def print_g_sweep(g_values, probs_list, mode, t_total, n_steps):
    """g スイープ結果を表形式で表示し、厳密解と比較する。"""
    print(
        f"\n{'g':>5}  {'Ω₁':>6}  {'Ω₁·T':>7}  "
        f"{'P(|11>)_sim':>12}  {'P(|11>)_exact':>14}  {'|誤差|':>8}"
    )
    print("-" * 68)
    for g, p in zip(g_values, probs_list):
        g   = float(g)
        p11 = float(p[3])
        pex = exact_p11(t_total, g)
        O1  = omega1(g)
        err = abs(p11 - pex)
        print(
            f"{g:>5.2f}  {O1:>6.3f}  {O1*t_total:>7.3f}  "
            f"{p11:>12.5f}  {pex:>14.5f}  {err:>8.5f}"
        )

    print()
    print(f"  モード  : {mode}")
    print(
        f"  パラメータ: ε={EPSILON}, ω={OMEGA}, "
        f"T={t_total}, N={n_steps}, dt={t_total/n_steps:.4f}"
    )
    print(
        f"  厳密解  : P(|11⟩,T) = (g/Ω₁)²·sin²(Ω₁T), "
        f"Ω₁ = √((ε+ω)²/4 + g²)"
    )
    print(f"  共鳴条件: ε+ω = {EPSILON + OMEGA:.1f}  →  Ω₁ = g  (完全 Rabi 振動)")


def print_header():
    print(f"\n{'='*60}")
    print(f"  最小 Jahn-Teller モデル")
    print(f"  H = ε/2·ZI + ω/2·IZ + g·XX")
    print(f"  ε={EPSILON},  ω={OMEGA}  →  共鳴条件 ε+ω = {EPSILON+OMEGA:.1f}")
    print(f"  厳密固有値 E = ±√((ε±ω)²/4 + g²)  (解析式)")
    print(f"{'='*60}")


# ============================================================
# メイン
# ============================================================

def main(mode="simulate", ibm_token=None):
    print_header()

    # ── simulate ─────────────────────────────────────────────
    if mode == "simulate":
        print(f"\n[Mode: default.qubit  (qml.exp + Snapshot)]")
        print(f"  N_STEPS={N_STEPS_SIM},  T_total={T_TOTAL_SIM}")

        print("\n  厳密固有値の確認:")
        for g_val in [0.5, 1.0]:
            eigs = exact_eigenvalues(EPSILON, OMEGA, g_val)
            print(f"    g={g_val}: E = {[f'{e:.4f}' for e in eigs]}")

        print("\n  時間発展軌跡 (g ごとに qml.exp を適用):")
        for g_val in [0.3, 0.5, 1.0]:
            times, p00, p01, p10, p11, final = run_simulate(g_val)
            p11_ex  = np.array([exact_p11(t, g_val) for t in times])
            max_err = np.max(np.abs(p11 - p11_ex))
            O1      = omega1(g_val)

            print(f"\n    g={g_val:.1f}: Ω₁={O1:.3f},  T_Rabi=π/Ω₁={np.pi/O1:.2f}")
            print(f"      最大偏差 |P_sim - P_exact| = {max_err:.2e}")
            print(
                f"      t=T: P(|00>)={final[0]:.4f}  P(|01>)={final[1]:.4f}  "
                f"P(|10>)={final[2]:.4f}  P(|11>)={final[3]:.4f}"
            )
            print(f"      解析式:  P(|11>)={exact_p11(T_TOTAL_SIM, g_val):.4f}")
            # Block 2 は |00> スタートでは常に 0 を確認
            print(
                f"      Block 2 確認: P(|01>)+P(|10>)="
                f"{float(final[1])+float(final[2]):.2e}  (理論値=0)"
            )

        print("\n  軌跡グラフを生成中...")
        plot_trajectories([0.3, 0.5, 1.0])

    # ── catalyst ─────────────────────────────────────────────
    elif mode == "catalyst":
        print(f"\n[Mode: Catalyst (@qjit + for_loop + jax.vmap)]")
        print(f"  N_STEPS={N_STEPS_CAT},  T_total={T_TOTAL_HW}")
        print(f"  回路/ステップ: RZ(q0)+RZ(q1)+CNOT+RX(q0)+CNOT")
        probs = run_catalyst()
        print_g_sweep(G_VALUES, probs, mode, T_TOTAL_HW, N_STEPS_CAT)

    # ── ibm_sim ──────────────────────────────────────────────
    elif mode == "ibm_sim":
        res = circuit_specs_hw()
        print(f"\n[Mode: IBM AerSimulator  (shots={SHOTS_HW})]")
        print(f"  回路仕様: ゲート数={res.num_gates}  深度={res.depth}")
        print(f"  CNOT 数={N_STEPS_HW * 2},  T_circuit≈{N_STEPS_HW*2*0.4:.1f}μs")
        print(f"  N_STEPS={N_STEPS_HW},  T_total={T_TOTAL_HW}")

        backend = get_aer_backend()
        dev     = qml.device(
            "qiskit.remote", wires=2, backend=backend, shots=SHOTS_HW
        )
        probs = [build_hw_qnode(dev, float(g))() for g in G_VALUES]
        print_g_sweep(G_VALUES, probs, mode, T_TOTAL_HW, N_STEPS_HW)

    # ── ibm_real ─────────────────────────────────────────────
    elif mode == "ibm_real":
        if ibm_token is None:
            print("  ERROR: IBM_TOKEN を引数に指定してください")
            print("  使い方: python pennylane_minimal_jt.py ibm_real <TOKEN>")
            sys.exit(1)

        res = circuit_specs_hw()
        print(f"\n[Mode: IBM 実機  (shots={SHOTS_HW})]")
        print(f"  回路仕様: ゲート数={res.num_gates}  深度={res.depth}")
        print(f"  CNOT 数={N_STEPS_HW * 2},  T_circuit≈{N_STEPS_HW*2*0.4:.1f}μs  (T1≈200μs の {N_STEPS_HW*2*0.4/200*100:.0f}%)")
        print(f"  N_STEPS={N_STEPS_HW},  T_total={T_TOTAL_HW}")

        backend = get_ibm_backend(ibm_token)
        dev     = qml.device(
            "qiskit.remote", wires=2, backend=backend, shots=SHOTS_HW
        )
        probs = [build_hw_qnode(dev, float(g))() for g in G_VALUES]
        print_g_sweep(G_VALUES, probs, mode, T_TOTAL_HW, N_STEPS_HW)

    else:
        print(f"  Unknown mode: '{mode}'")
        print("  simulate | catalyst | ibm_sim | ibm_real")
        sys.exit(1)


if __name__ == "__main__":
    run_mode  = sys.argv[1] if len(sys.argv) > 1 else "simulate"
    token_arg = sys.argv[2] if len(sys.argv) > 2 else None
    main(mode=run_mode, ibm_token=token_arg)
