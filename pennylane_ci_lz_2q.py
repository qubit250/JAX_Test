"""
円錐交差における Landau-Zener 遷移シミュレーション (2-qubit JT モデル)
===================================================================
H(t) = [ε(t)/2]·ZI + [ω/2]·IZ + g·XX

Block 1 {|00⟩, |11⟩} の有効ハミルトニアン:
  H_eff(t) = [(ε(t)+ω)/2]·σ_z + g·σ_x     (基底: {|00⟩, |11⟩})
  CI (円錐交差): (ε+ω)/2 = 0 かつ g = 0

Landau-Zener スイープ:
  ε(t) = -ω + α·t,   t ∈ [-T/2, +T/2],   T = 10.0 (固定)
  t=0 で CI 通過  (α·T/4 = 2.5α >> g=0.3: 有効範囲 α ≥ 0.2)

参照値 (精密数値解):
  Block 1 の 2×2 Schrödinger 方程式を行列指数関数で厳密積分:
    |ψ⟩(t+dt) = exp(-i·H_eff(t)·dt)|ψ⟩(t)
  (LZ 漸近公式 P=1-exp(-2πg²/α) は T→∞ 極限; 有限 T では ~12% 小さい)

Trotter 量子回路 (2 qubit, CNOT × 2 per step):
  各ステップ k で ε_k = -ω + α·t_k (時変):
    RZ(ε_k·dt, q0) · RZ(ω·dt, q1) · CNOT · RX(2g·dt, q0) · CNOT

IBM 互換: N_STEPS=20 → CNOT = 40, 有効 α 範囲 [0.2, 1.0]

スキャン:
  alpha : スイープ速度スキャン → P(|11⟩)(α) の検証 (反応収率 vs 速度)
  g     : 結合強度スキャン     → P(|11⟩)(g)  の検証 (CI ギャップ効果)

円錐交差最適化への接続:
  Berry 位相 γ=π → CI の位相的検出     (既実装: pennylane_berry_phase_jt.py)
  LZ 分岐比 P_LZ → CI 通過時の反応収率  (本スクリプト)
  変分パルス最適化 → α*(t) で目標収率を達成 (次ステップ)

使い方:
  python pennylane_ci_lz_2q.py [simulate|ibm_real] [alpha|g] [TOKEN]
"""
import sys
import numpy as np

# ── パラメータ ────────────────────────────────────────────────
OMEGA        =  1.0    # 振動周波数 ω (固定)
G_DEF        =  0.3    # 結合強度 g (alpha スキャン時の固定値)
ALPHA_DEF    =  1.0    # スイープ速度 α (g スキャン時の固定値)
T_FINAL      = 10.0    # スイープ全時間 (固定; α≥0.2 で α·T/4=2.5α>>g 確保)
N_STEPS_SIM  = 200     # simulate モード Trotter 数 (dt=0.050)
N_STEPS_IBM  = 20      # IBM 実機 Trotter 数 (dt=0.500, CNOT=40)
SHOTS        = 8192    # IBM shots

# ── スキャン値 ────────────────────────────────────────────────
# IBM 実機 (N=20): α≤1.0 で Trotter 誤差 ≤7%; α=2.0 では ~83% → simulate のみ
ALPHA_VALUES_IBM = np.array([0.20, 0.50, 1.00])          # IBM 有効範囲
ALPHA_VALUES_SIM = np.array([0.20, 0.50, 1.00, 2.00])    # simulate 拡張 (α=2 は参考)
G_VALUES         = np.array([0.10, 0.20, 0.30, 0.50, 0.70])


# ── 精密数値参照解 ────────────────────────────────────────────

def exact_block1(g: float, alpha: float, T: float = T_FINAL, n_pts: int = 5000) -> float:
    """Block 1 Schrödinger 方程式の精密数値積分 (行列指数関数)。

    H_eff(t) = [[α·t/2, g],[g, -α·t/2]], 初期状態 |00⟩ = (1, 0)
    各ステップで exp(-i·H_eff·dt)|ψ⟩ を厳密計算する (Trotter 誤差なし)。
    """
    dt   = T / n_pts
    psi  = np.array([1.0 + 0j, 0.0 + 0j])
    for k in range(n_pts):
        t_k   = -T / 2.0 + (k + 0.5) * dt
        delta = alpha * t_k / 2.0
        H_eff = np.array([[delta, g], [g, -delta]])
        E, V  = np.linalg.eigh(H_eff)
        psi   = V @ (np.exp(-1j * E * dt) * (V.conj().T @ psi))
    return float(abs(psi[1]) ** 2)


def lz_asymptotic(g: float, alpha: float) -> float:
    """漸近 LZ 公式 (T→∞ 極限): P = 1 - exp(-2π·g²/α)"""
    return float(1.0 - np.exp(-2.0 * np.pi * g ** 2 / max(alpha, 1e-12)))


def optimal_alpha(g: float, target_yield: float) -> float:
    """目標収率 Y* に対する漸近最適スイープ速度: α* = -2πg²/ln(1-Y*)"""
    if target_yield >= 1.0:
        return 0.0
    return float(-2.0 * np.pi * g ** 2 / np.log(max(1.0 - target_yield, 1e-10)))


# ── 量子回路 ──────────────────────────────────────────────────

def build_lz_circuit(dev, g: float, alpha: float, n_steps: int, mode: str) -> float:
    """LZ スイープ 2-qubit Trotter 回路を実行し P(|11⟩) を返す。

    RZ(ε_k·dt)·RZ(ω·dt)·CNOT·RX(2g·dt)·CNOT の繰り返し (n_steps 回)
    CNOT 総数 = 2·n_steps。IBM では n_steps=20 → CNOT=40。
    """
    import pennylane as qml

    dt       = T_FINAL / n_steps
    t_mids   = np.array([-T_FINAL / 2.0 + (k + 0.5) * dt for k in range(n_steps)])
    eps_vals = -OMEGA + alpha * t_mids   # ε(t_k) = -ω + α·t_k

    @qml.set_shots(SHOTS if mode != "simulate" else None)
    @qml.qnode(dev)
    def circuit():
        # |00⟩: Block 1 ground state when ε+ω = α·(-T/2) = -5α << 0 (for α≥0.2)
        for k in range(n_steps):
            eps_k = float(eps_vals[k])
            qml.RZ(float(eps_k * dt), wires=0)           # exp(-i·ε_k/2·Z₀·dt)
            qml.RZ(float(OMEGA * dt), wires=1)            # exp(-i·ω/2·Z₁·dt)
            qml.CNOT(wires=[0, 1])
            qml.RX(float(2.0 * g * dt), wires=0)         # exp(-i·g·X₀X₁·dt) via CNOT sandwich
            qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    probs = circuit()
    return float(probs[3])   # P(|11⟩) = index 3 in 2-qubit probability vector


# ── スキャン実行 ──────────────────────────────────────────────

def run_scan(scan_type: str = "alpha", mode: str = "simulate", token=None):
    import pennylane as qml

    n_steps = N_STEPS_IBM if mode == "ibm_real" else N_STEPS_SIM
    dt      = T_FINAL / n_steps

    if mode == "ibm_real":
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=2)
        print(f"  バックエンド: {backend.name}")

    def make_dev():
        if mode == "ibm_real":
            return qml.device("qiskit.remote", wires=2, backend=backend)
        return qml.device("default.qubit", wires=2)

    sep = "=" * 74
    print(f"\n{sep}")
    print(f"  円錐交差 Landau-Zener 遷移  (2-qubit JT モデル)  —  {scan_type} スキャン")
    print(f"  H(t) = [ε(t)/2]·ZI + [ω/2]·IZ + g·XX,  ω={OMEGA}")
    print(f"  ε(t) = -ω + α·t,  t ∈ [-{T_FINAL/2:.0f}, +{T_FINAL/2:.0f}]  (t=0 で CI 通過)")
    print(f"  T={T_FINAL}, N_steps={n_steps}, dt={dt:.4f}, CNOT数={n_steps*2}, mode={mode}")
    print(sep)
    print(f"\n  参照値: Block 1 厳密数値積分 (精密 5000 点)")
    print(f"  LZ 漸近式: P_LZ = 1-exp(-2π·g²/α)  [T→∞ 極限, 有限 T で約 12% 過大評価]\n")

    if scan_type == "alpha":
        g = G_DEF
        alpha_values = ALPHA_VALUES_IBM if mode == "ibm_real" else ALPHA_VALUES_SIM
        print(f"  g={g:.2f} 固定, ω={OMEGA}")
        print(f"  注意: α<0.2 は T=10 で終端離調 α·T/4 < g → LZ 非適用")
        if mode == "ibm_real":
            print(f"  IBM モード: α≤1.0 で Trotter 誤差 ≤7%  (α=2.0 は ~83% → 除外)")
        else:
            print(f"  simulate: α=2.0 は参考値 (N=20 IBM では Trotter 誤差 ~83%)")
        print(f"\n  {'α':>6}  {'P_circuit':>10}  {'P_exact':>9}  {'P_LZ':>7}  "
              f"{'err%(ex)':>9}  状態")
        print("  " + "-" * 64)
        errs_ex = []
        for alpha in alpha_values:
            dev   = make_dev()
            p11   = build_lz_circuit(dev, g, float(alpha), n_steps, mode)
            p_ex  = exact_block1(g, float(alpha))
            p_lz  = lz_asymptotic(g, float(alpha))
            err   = abs(p11 - p_ex) / max(abs(p_ex), 1e-6) * 100
            label = ("断熱↓  (|11⟩ 優勢)" if p_ex > 0.80 else
                     "非断熱↑ (|00⟩ 優勢)" if p_ex < 0.20 else "遷移域")
            errs_ex.append(abs(p11 - p_ex))
            print(f"  {alpha:>6.2f}  {p11:>10.4f}  {p_ex:>9.4f}  {p_lz:>7.4f}  "
                  f"{err:>8.2f}%  {label}")

        print(f"\n  MAE (vs 精密解) = {np.mean(errs_ex)*100:.2f}%")
        print(f"\n  [反応プロセス最適化 (LZ 漸近近似による目標収率設計)]")
        for y_target in [0.30, 0.50, 0.70, 0.90]:
            a_opt = optimal_alpha(g, y_target)
            print(f"  Y* = {y_target*100:.0f}%  →  α* ≈ {a_opt:.4f}")

    elif scan_type == "g":
        alpha = ALPHA_DEF
        print(f"  α={alpha:.2f} 固定, ω={OMEGA}")
        print(f"\n  {'g':>6}  {'P_circuit':>10}  {'P_exact':>9}  {'P_LZ':>7}  {'err%(ex)':>9}")
        print("  " + "-" * 52)
        errs_ex = []
        for g in G_VALUES:
            dev   = make_dev()
            p11   = build_lz_circuit(dev, float(g), alpha, n_steps, mode)
            p_ex  = exact_block1(float(g), alpha)
            p_lz  = lz_asymptotic(float(g), alpha)
            err   = abs(p11 - p_ex) / max(abs(p_ex), 1e-6) * 100
            errs_ex.append(abs(p11 - p_ex))
            print(f"  {g:>6.2f}  {p11:>10.4f}  {p_ex:>9.4f}  {p_lz:>7.4f}  {err:>8.2f}%")

        print(f"\n  MAE (vs 精密解) = {np.mean(errs_ex)*100:.2f}%")
        print(f"\n  [物理的観察]")
        print(f"  g 増大 → CI ギャップ拡大 → 断熱性向上 → P(|11⟩) 増大")

    else:
        print(f"  エラー: scan_type は 'alpha' または 'g' を指定してください")
        sys.exit(1)

    print(f"\n{sep}\n")


if __name__ == "__main__":
    mode      = sys.argv[1] if len(sys.argv) > 1 else "simulate"
    scan_type = sys.argv[2] if len(sys.argv) > 2 else "alpha"
    token     = sys.argv[3] if len(sys.argv) > 3 else None

    if mode == "ibm_real" and token is None:
        print("使い方: python pennylane_ci_lz_2q.py ibm_real <alpha|g> <TOKEN>")
        sys.exit(1)

    run_scan(scan_type=scan_type, mode=mode, token=token)
