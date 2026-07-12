"""PennyLane + Catalyst: IBM実機向け小規模動的非断熱計算。

Landau-Zener 2準位モデル  H(t) = alpha * t * Z + Delta * X

時間発展演算子の厳密分解:
    exp(-i*(a*Z + delta*X)*dt) = RY(-phi) @ RZ(2*norm*dt) @ RY(+phi)
    ここで norm = sqrt(a^2 + delta^2),  phi = arctan2(delta, a)

実行モード (コマンドライン引数で指定):
    python pennylane_catalyst_ibm_nonadiabatic.py catalyst     # デフォルト
    python pennylane_catalyst_ibm_nonadiabatic.py ibm_sim
    python pennylane_catalyst_ibm_nonadiabatic.py ibm_real <TOKEN>
"""

import sys
import warnings
import numpy as np
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane import qjit

warnings.filterwarnings("ignore")

# ============================================================
# 共通パラメータ
# ============================================================
DELTA   = 1.0    # avoided crossing のギャップ
T_TOTAL = 6.0    # 掃引の全時間

N_STEPS_SIM = 100   # Catalyst シミュレーション用ステップ数
N_STEPS_HW  = 15    # IBM ハードウェア用 (回路深度 ≈ 46 に抑える)
SHOTS_HW    = 8192  # IBM 実機での測定ショット数

ALPHA_VALUES = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0])


# ============================================================
# 初期状態の準備
# ============================================================

def _initial_ry_angle_jnp(alpha: jnp.ndarray) -> jnp.ndarray:
    """H(t=-T/2) の基底状態を RY ゲートで準備するための回転角 (JAX)。"""
    a    = alpha * (-T_TOTAL / 2.0)
    norm = jnp.sqrt(a**2 + DELTA**2)
    v1   = jnp.sqrt((norm + a) / (2.0 * norm))         # sin(theta/2)
    v0   = -DELTA / jnp.sqrt(2.0 * norm * (norm + a))  # cos(theta/2)
    return 2.0 * jnp.arctan2(v1, v0)

def _initial_ry_angle_np(alpha: float) -> float:
    """H(t=-T/2) の基底状態を RY ゲートで準備するための回転角 (NumPy)。"""
    a    = alpha * (-T_TOTAL / 2.0)
    norm = np.sqrt(a**2 + DELTA**2)
    v1   = np.sqrt((norm + a) / (2.0 * norm))
    v0   = -DELTA / np.sqrt(2.0 * norm * (norm + a))
    return float(2.0 * np.arctan2(v1, v0))


# ============================================================
# Mode 1: Catalyst / lightning.qubit
#         @qjit + for_loop + jax.vmap で全 alpha を一括 JIT 実行
# ============================================================

def run_catalyst(alpha_values: jnp.ndarray, n_steps: int = N_STEPS_SIM):
    """Catalyst (@qjit) で JIT コンパイルした時間発展を vmap で並列実行。"""
    dev = qml.device("lightning.qubit", wires=1)
    dt  = T_TOTAL / n_steps

    @qjit
    @qml.qnode(dev)
    def circuit(alpha):
        qml.RY(_initial_ry_angle_jnp(alpha), wires=0)

        def step(i, _):
            t    = -T_TOTAL / 2.0 + i * dt
            a    = alpha * t
            norm = jnp.sqrt(a**2 + DELTA**2)
            phi  = jnp.arctan2(DELTA, a)
            # exp(-i*(a*Z + delta*X)*dt) = RY(-phi) RZ(2*norm*dt) RY(+phi)
            qml.RY(-phi,          wires=0)
            qml.RZ(2.0 * norm * dt, wires=0)
            qml.RY(+phi,          wires=0)
            return None

        qml.for_loop(0, n_steps, 1)(step)(None)
        return qml.probs(wires=0)

    # vmap で全 alpha 値を並列計算 (JIT コンパイル済み関数の再利用)
    probs = jax.vmap(circuit)(alpha_values)
    return probs


# ============================================================
# Mode 2/3: IBM 実機 / AerSimulator 用ハードウェア回路
#            for ループを静的に展開 → Qiskit トランスパイル対応
# ============================================================

def build_hw_qnode(device, alpha: float, n_steps: int = N_STEPS_HW):
    """IBM ネイティブゲート (RY, RZ) で展開した時間発展回路を返す。"""
    dt    = T_TOTAL / n_steps
    times = np.linspace(-T_TOTAL / 2, T_TOTAL / 2, n_steps)
    init_angle = _initial_ry_angle_np(alpha)

    @qml.qnode(device)
    def circuit():
        qml.RY(init_angle, wires=0)
        for t in times:
            a    = float(alpha * t)
            norm = float(np.sqrt(a**2 + DELTA**2))
            phi  = float(np.arctan2(DELTA, a))
            qml.RY(-phi,          wires=0)
            qml.RZ(2.0 * norm * dt, wires=0)
            qml.RY(+phi,          wires=0)
        return qml.probs(wires=0)

    return circuit


def circuit_specs(n_steps: int = N_STEPS_HW) -> dict:
    """ハードウェア回路のゲート数・深度を返す。"""
    dev = qml.device("default.qubit", wires=1)
    circ = build_hw_qnode(dev, alpha=1.0, n_steps=n_steps)
    return qml.specs(circ)()["resources"]


# ============================================================
# IBM バックエンド取得
# ============================================================

def get_ibm_backend(token: str, instance: str = "ibm-q/open/main"):
    """最小ビット数の稼働中 IBM 実機バックエンドを返す。"""
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(
        channel="ibm_quantum", token=token, instance=instance
    )
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=1)
    print(f"  -> 使用バックエンド: {backend.name}")
    return backend


def get_aer_backend(gate_error: float = 0.001, readout_error: float = 0.01):
    """IBM 実機相当のノイズモデル付き AerSimulator を返す。"""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(gate_error, 1), ["ry", "rz", "rx"])
    nm.add_all_qubit_readout_error(
        ReadoutError([[1 - readout_error, readout_error],
                      [readout_error, 1 - readout_error]])
    )
    return AerSimulator(noise_model=nm)


# ============================================================
# 結果の表示
# ============================================================

def print_results(alpha_values, probs, mode: str, n_steps: int):
    print(f"\n{'alpha':>6}  {'P(|0>)':>8}  {'P(|1>)':>8}  解釈")
    print("-" * 44)
    for a, p in zip(alpha_values, probs):
        p0, p1 = float(p[0]), float(p[1])
        interp = "断熱 ← |0> に追従" if p0 < 0.3 else (
                 "非断熱 → |0> に留まる" if p0 > 0.45 else "遷移域")
        print(f"{float(a):>6.1f}  {p0:>8.4f}  {p1:>8.4f}  {interp}")
    print()
    print(f"  モード : {mode}")
    print(f"  ステップ数 N_STEPS = {n_steps}  "
          f"(dt = {T_TOTAL/n_steps:.3f},  T = {T_TOTAL})")


# ============================================================
# メイン
# ============================================================

def main(mode: str = "catalyst", ibm_token: str | None = None):
    print(f"\n{'='*54}")
    print(f"  IBM実機検証: 小規模動的非断熱計算 (Landau-Zener)")
    print(f"  H(t) = alpha*t*Z + {DELTA}*X,  T = {T_TOTAL}")
    print(f"{'='*54}")

    # -------- Mode 1: Catalyst --------
    if mode == "catalyst":
        print(f"\n[Mode: Catalyst + lightning.qubit]")
        print(f"  @qjit + for_loop + jax.vmap,  N_STEPS = {N_STEPS_SIM}")

        # 初回: JIT コンパイル + 実行
        import time
        t0 = time.time()
        probs_first = run_catalyst(ALPHA_VALUES, n_steps=N_STEPS_SIM)
        t1 = time.time()
        print(f"  JIT コンパイル + 初回実行: {t1-t0:.2f} s")

        # 2回目: キャッシュ済み JIT
        t0 = time.time()
        probs = run_catalyst(ALPHA_VALUES, n_steps=N_STEPS_SIM)
        t1 = time.time()
        print(f"  2回目 (JIT キャッシュ):    {t1-t0:.3f} s")

        print_results(ALPHA_VALUES, probs, mode, N_STEPS_SIM)

    # -------- Mode 2: IBM AerSimulator --------
    elif mode == "ibm_sim":
        print(f"\n[Mode: IBM AerSimulator (ノイズあり, shots={SHOTS_HW})]")
        print(f"  gate_error=0.1%,  readout_error=1%")

        # 回路仕様
        res = circuit_specs(N_STEPS_HW)
        print(f"  ゲート数: {res.num_gates},  回路深度: {res.depth}")

        backend = get_aer_backend()
        dev = qml.device("qiskit.remote", wires=1, backend=backend, shots=SHOTS_HW)

        probs_hw = []
        for a in ALPHA_VALUES:
            circ = build_hw_qnode(dev, float(a), n_steps=N_STEPS_HW)
            probs_hw.append(circ())

        print_results(ALPHA_VALUES, probs_hw, mode, N_STEPS_HW)

    # -------- Mode 3: IBM 実機 --------
    elif mode == "ibm_real":
        if ibm_token is None:
            print("  ERROR: IBM_TOKEN を引数に指定してください")
            print("  使い方: python script.py ibm_real <YOUR_IBM_TOKEN>")
            sys.exit(1)

        print(f"\n[Mode: IBM 実機 (shots={SHOTS_HW})]")
        res = circuit_specs(N_STEPS_HW)
        print(f"  ゲート数: {res.num_gates},  回路深度: {res.depth}")

        backend = get_ibm_backend(ibm_token)
        dev = qml.device("qiskit.remote", wires=1, backend=backend, shots=SHOTS_HW)

        probs_real = []
        for a in ALPHA_VALUES:
            circ = build_hw_qnode(dev, float(a), n_steps=N_STEPS_HW)
            probs_real.append(circ())

        print_results(ALPHA_VALUES, probs_real, mode, N_STEPS_HW)

    else:
        print(f"  Unknown mode: {mode}")
        print("  catalyst | ibm_sim | ibm_real")
        sys.exit(1)


if __name__ == "__main__":
    run_mode  = sys.argv[1] if len(sys.argv) > 1 else "catalyst"
    token_arg = sys.argv[2] if len(sys.argv) > 2 else None
    main(mode=run_mode, ibm_token=token_arg)
