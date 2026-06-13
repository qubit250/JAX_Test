"""PennyLane: 小規模な動的非断熱遷移のシミュレーション (Landau-Zener モデル)。

1量子ビット系に対する時間依存ハミルトニアン

    H(t) = alpha * t * Z + Delta * X    (t: -T/2 -> +T/2)

を考える。ZとXの基底をエネルギー固有基底とみなすと、t=0付近で
2つのエネルギー準位が接近・反発する「avoided crossing」が生じる。

- alpha が小さい(=ゆっくり掃引)場合: 断熱定理が良く満たされ、状態は
  常にハミルトニアンの瞬間的な基底状態に追従する(断熱遷移)。
- alpha が大きい(=速く掃引)場合: 断熱定理が破れ、状態は基底状態に
  追従できず、初期状態(|0>)付近に留まる(非断熱遷移)。

このスクリプトでは、各時間ステップで H(t) を区分定数近似し、
qml.exp(H(t), -i*dt) によって時間発展演算子 exp(-i H(t) dt) を厳密に
適用することで、量子ビットの時間発展をシミュレートする。
シミュレーション結果と、各時刻における瞬間的な基底状態の占有率
(厳密対角化による参照値)を比較することで、非断熱遷移の大きさを可視化する。
"""

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

DELTA = 1.0      # avoided crossing のギャップ幅
T_TOTAL = 20.0   # 掃引の全時間
N_STEPS = 400    # 時間ステップ数


def ground_state_vector(a, delta=DELTA):
    """H = a*Z + delta*X の瞬間的な基底状態(固有ベクトル)を返す。"""
    h_matrix = np.array([[a, delta], [delta, -a]])
    _, eigvecs = np.linalg.eigh(h_matrix)
    return eigvecs[:, 0]


def instantaneous_ground_state_population(a, delta=DELTA):
    """H = a*Z + delta*X の瞬間的な基底状態における |0> の占有率を返す。"""
    return np.abs(ground_state_vector(a, delta)[0]) ** 2


def simulate(alpha, delta=DELTA, t_total=T_TOTAL, n_steps=N_STEPS):
    """H(t) = alpha*t*Z + delta*X による時間発展をシミュレートする。

    Returns:
        times: 各ステップの時刻
        p0_sim: シミュレーションによる各時刻の |0> 占有率
        p0_ground: 瞬間的な基底状態における |0> 占有率(参照値)
        final_probs: 最終時刻における |0>, |1> の占有率
    """
    dt = t_total / n_steps
    times = np.linspace(-t_total / 2, t_total / 2, n_steps)
    initial_state = ground_state_vector(alpha * times[0], delta)

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit():
        # 初期状態は H(t=-T/2) の基底状態(=断熱的に保たれていた状態)とする
        qml.StatePrep(initial_state, wires=0)
        for t in times:
            h_t = qml.Hamiltonian([alpha * t, delta], [qml.PauliZ(0), qml.PauliX(0)])
            qml.exp(h_t, -1j * dt)
            qml.Snapshot()
        return qml.probs(wires=0)

    snapshots = qml.snapshots(circuit)()
    final_probs = snapshots.pop("execution_results")

    ordered_keys = sorted(snapshots.keys(), key=int)
    p0_sim = np.array([np.abs(snapshots[key][0]) ** 2 for key in ordered_keys])
    p0_ground = np.array(
        [instantaneous_ground_state_population(alpha * t, delta) for t in times]
    )

    return times, p0_sim, p0_ground, final_probs


def main():
    cases = [
        (0.1, "slow sweep (adiabatic)"),
        (5.0, "fast sweep (non-adiabatic)"),
    ]

    fig, axes = plt.subplots(1, len(cases), figsize=(10, 4), sharey=True)
    for ax, (alpha, label) in zip(axes, cases):
        times, p0_sim, p0_ground, final_probs = simulate(alpha)
        ax.plot(times, p0_sim, label="simulated P(|0>)")
        ax.plot(times, p0_ground, "--", label="instantaneous ground state P(|0>)")
        ax.set_title(f"alpha = {alpha}\n({label})")
        ax.set_xlabel("time")
        ax.legend()

        print(f"alpha = {alpha} ({label})")
        print(f"  final P(|0>) = {final_probs[0]:.4f}")
        print(f"  final P(|1>) = {final_probs[1]:.4f}")
        print(f"  instantaneous ground state P(|0>) at t=T/2: {p0_ground[-1]:.4f}")

    axes[0].set_ylabel("population of |0>")
    plt.tight_layout()
    plt.savefig("landau_zener_nonadiabatic.png", dpi=150)
    print("\nSaved figure to landau_zener_nonadiabatic.png")


if __name__ == "__main__":
    main()
