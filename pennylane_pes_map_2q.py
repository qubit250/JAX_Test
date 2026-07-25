"""
PES 2D マッピング — Jahn-Teller 2-qubit モデル
=================================================
H = ε/2·ZI + ω/2·IZ + g·XX,   ω = 1.0

ポテンシャルエネルギー面 (Block 1 {|00⟩, |11⟩}):
  Ω₁(ε,g) = √((ε+ω)²/4 + g²)          [一般化 Rabi 周波数]
  E±(ε,g)  = ±Ω₁                        [断熱固有エネルギー]
  CI 点:    (ε, g) = (−ω, 0) = (−1, 0)  [縮退点, ギャップ=0]
  CI セアム: ε = −ω = −1 (全 g で P_max=1)

量子ダイナミクスマップ (固定 T = 5.0):
  P_exact(ε,g) = (g/Ω₁)²·sin²(Ω₁·T)   [解析式]
  P_max(ε,g)   = (g/Ω₁)²               [最大遷移確率の上限]

特徴的な構造:
  ε=−1 (CI セアム): P_max = 1, P = sin²(g·T) [Rabi 振動]
  ε≠−1 (非共鳴):   P_max < 1, P ≪ 1         [反応抑制]
  T=5 での共鳴ピーク: g* = π/(2T) ≈ 0.314    [最大 P=1 の g 値]

スキャン種別:
  eps   : g=0.30 固定, ε スキャン  → Lorentz 型共鳴ピーク (IBM 主測定)
  seam  : ε=−1.0 固定, g スキャン → Rabi 振動パターン (CI セアム)
  map2d : (ε, g) 2D グリッド     → PES トポロジーマップ (シミュレーション)

回路:
  初期状態: |00⟩ (Block 1 ground state で ε+ω << 0)
  各ステップ: RZ(ε·dt) · RZ(ω·dt) · CNOT · RX(2g·dt) · CNOT
  IBM: N_STEPS=10, CNOT=20 per circuit, T=5.0

使い方:
  python pennylane_pes_map_2q.py [simulate|ibm_real] [eps|seam|map2d] [TOKEN]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── パラメータ ────────────────────────────────────────────────
OMEGA       =  1.0    # 振動周波数 ω (固定)
EPS_RES     = -1.0    # CI セアム位置 (ε = −ω)
G_DEF       =  0.3    # 結合強度 g (eps スキャン時の固定値)
T_FINAL     =  5.0    # 観測時間 (g*=π/(2T)≈0.314 で P_max=1)
N_STEPS_SIM = 50      # simulate Trotter 数 (dt=0.100, CNOT=100)
N_STEPS_IBM = 10      # IBM 実機 Trotter 数 (dt=0.500, CNOT=20)
SHOTS       = 8192    # IBM shots

# ── 1D スキャン値 ─────────────────────────────────────────────
EPS_VALUES = np.array([-3.0, -2.0, -1.5, -1.0, -0.5,  0.0,  1.0])
G_VALUES   = np.array([ 0.1,  0.2,  0.3,  0.5,  0.7,  1.0])

# ── 2D グリッド ───────────────────────────────────────────────
N_EPS_SIM  = 50    # simulate: 50×40 grid
N_G_SIM    = 40
N_EPS_IBM  = 7     # IBM: 7×6 grid (42 circuits, CNOT=20 each)
N_G_IBM    = 6
EPS_MIN, EPS_MAX = -3.0, 1.5
G_MIN,   G_MAX   = 0.05, 1.0


# ── 解析式 ────────────────────────────────────────────────────

def rabi_freq(eps: float, g: float, omega: float = OMEGA) -> float:
    """Ω₁ = √((ε+ω)²/4 + g²)"""
    return float(np.sqrt(((eps + omega) / 2.0) ** 2 + g ** 2))


def p_exact(eps: float, g: float, T: float = T_FINAL) -> float:
    """Block 1 厳密解: P(|11⟩,T) = (g/Ω₁)²·sin²(Ω₁·T)"""
    o1 = rabi_freq(eps, g)
    if o1 < 1e-10:
        return 0.0
    return float((g / o1) ** 2 * np.sin(o1 * T) ** 2)


def p_max(eps: float, g: float) -> float:
    """最大遷移確率 (時間平均上限): P_max = (g/Ω₁)²"""
    o1 = rabi_freq(eps, g)
    if o1 < 1e-10:
        return 0.0
    return float((g / o1) ** 2)


def gap(eps: float, g: float) -> float:
    """断熱エネルギーギャップ: ΔE = 2Ω₁"""
    return float(2.0 * rabi_freq(eps, g))


# ── 量子回路 ──────────────────────────────────────────────────

def build_static_circuit(dev, eps: float, g: float, n_steps: int, mode: str) -> float:
    """固定 (ε, g) での JT 量子回路を実行して P(|11⟩) を返す。

    初期状態: |00⟩ (Block 1 ground state when ε+ω << 0)
    回路: [RZ(ε·dt)·RZ(ω·dt)·CNOT·RX(2g·dt)·CNOT] × n_steps
    CNOT 総数 = 2·n_steps
    """
    import pennylane as qml

    dt = T_FINAL / n_steps

    @qml.set_shots(SHOTS if mode != "simulate" else None)
    @qml.qnode(dev)
    def circuit():
        for _ in range(n_steps):
            qml.RZ(float(eps * dt),         wires=0)  # exp(-i·ε/2·Z₀·dt)
            qml.RZ(float(OMEGA * dt),        wires=1)  # exp(-i·ω/2·Z₁·dt)
            qml.CNOT(wires=[0, 1])
            qml.RX(float(2.0 * g * dt),     wires=0)  # exp(-i·g·X₀X₁·dt)
            qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    probs = circuit()
    return float(probs[3])   # P(|11⟩)


def make_dev(mode, backend=None):
    import pennylane as qml
    if mode == "ibm_real":
        return qml.device("qiskit.remote", wires=2, backend=backend)
    return qml.device("default.qubit", wires=2)


# ── スキャン実行 ──────────────────────────────────────────────

def run_scan(scan_type: str = "eps", mode: str = "simulate", token=None):
    import pennylane as qml

    n_steps = N_STEPS_IBM if mode == "ibm_real" else N_STEPS_SIM
    dt      = T_FINAL / n_steps
    backend = None

    if mode == "ibm_real":
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=2)
        print(f"  バックエンド: {backend.name}")

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  PES 2D マッピング  (2-qubit JT)  —  {scan_type} スキャン")
    print(f"  H = ε/2·ZI + ω/2·IZ + g·XX,  ω={OMEGA},  T={T_FINAL}")
    print(f"  CI 点: (ε, g) = ({EPS_RES:.1f}, 0)  [ε+ω=0]")
    print(f"  N_steps={n_steps}, dt={dt:.4f}, CNOT数={n_steps*2}/回路, mode={mode}")
    print(sep)
    print(f"\n  解析式: P = (g/Ω₁)²·sin²(Ω₁·T),  Ω₁=√((ε+ω)²/4 + g²)")
    print(f"  P_max  = (g/Ω₁)²  [最大遷移確率の上限]")
    print(f"  最適 g : g* = π/(2T) = {np.pi/(2*T_FINAL):.4f}  (ε=−1 で P=1 となる g)\n")

    # ── ε スキャン ────────────────────────────────────────────
    if scan_type == "eps":
        g = G_DEF
        print(f"  g={g:.2f} 固定, ε をスキャン → CI セアム (ε=−1) での共鳴ピーク")
        print(f"\n  {'ε':>6}  {'Ω₁':>6}  {'P_max':>7}  "
              f"{'P_circuit':>10}  {'P_exact':>9}  {'err%':>8}")
        print("  " + "-" * 58)
        errs = []
        for eps in EPS_VALUES:
            dev = make_dev(mode, backend)
            p11 = build_static_circuit(dev, float(eps), g, n_steps, mode)
            pe  = p_exact(float(eps), g)
            pm  = p_max(float(eps), g)
            o1  = rabi_freq(float(eps), g)
            err = abs(p11 - pe) / max(abs(pe), 1e-4) * 100
            errs.append(abs(p11 - pe))
            mark = " ← CI セアム" if abs(eps + OMEGA) < 0.01 else ""
            print(f"  {eps:>+6.1f}  {o1:>6.3f}  {pm:>7.3f}  "
                  f"{p11:>10.4f}  {pe:>9.4f}  {err:>7.2f}%{mark}")

        print(f"\n  MAE = {np.mean(errs)*100:.2f}%")
        print(f"\n  [物理的観察]")
        print(f"  ε=−1 (CI セアム): P≈1 → 電子−核結合最大 (断熱性最大)")
        print(f"  ε≠−1 (非共鳴):  P≪1 → 反応抑制")
        print(f"  左右対称性: ε=−1±Δε で P が等しい (CI 点の対称性)")

    # ── セアムスキャン ────────────────────────────────────────
    elif scan_type == "seam":
        eps = EPS_RES   # = −1 (CI セアム)
        print(f"  ε={eps:.1f} 固定 (CI セアム), g をスキャン → Rabi 振動パターン")
        print(f"  解析式: P = sin²(g·T) = sin²({T_FINAL}·g)  [共鳴 Ω₁=g]")
        print(f"  最大 P=1 は g* = π/(2T) = {np.pi/(2*T_FINAL):.4f} で達成")
        print(f"\n  {'g':>6}  {'g·T':>6}  "
              f"{'P_circuit':>10}  {'P_exact':>9}  {'err%':>8}")
        print("  " + "-" * 50)
        errs = []
        for gv in G_VALUES:
            dev = make_dev(mode, backend)
            p11 = build_static_circuit(dev, float(eps), float(gv), n_steps, mode)
            pe  = p_exact(float(eps), float(gv))
            err = abs(p11 - pe) / max(abs(pe), 1e-4) * 100
            errs.append(abs(p11 - pe))
            print(f"  {gv:>6.2f}  {gv*T_FINAL:>6.3f}  "
                  f"{p11:>10.4f}  {pe:>9.4f}  {err:>7.2f}%")

        print(f"\n  MAE = {np.mean(errs)*100:.2f}%")
        print(f"\n  [物理的観察]")
        print(f"  g={np.pi/(2*T_FINAL):.3f} で P=1 (最初のπパルス)  [最適 g]")
        print(f"  g 増大で Rabi 振動 → P が 1 と 0 の間を振動")

    # ── 2D マップ ─────────────────────────────────────────────
    elif scan_type == "map2d":
        n_eps = N_EPS_IBM if mode == "ibm_real" else N_EPS_SIM
        n_g   = N_G_IBM  if mode == "ibm_real" else N_G_SIM
        eps_arr = np.linspace(EPS_MIN, EPS_MAX, n_eps)
        g_arr   = np.linspace(G_MIN,   G_MAX,   n_g)

        print(f"  グリッド: {n_eps}×{n_g} = {n_eps*n_g} 点")
        print(f"  ε ∈ [{EPS_MIN}, {EPS_MAX}], g ∈ [{G_MIN}, {G_MAX}]")
        print(f"  総 CNOT = {n_eps * n_g * n_steps * 2}")

        # ── 解析マップ (常に計算)
        EPS_GRID, G_GRID = np.meshgrid(eps_arr, g_arr)
        P_MAX_MAP = np.vectorize(p_max)(EPS_GRID, G_GRID)
        P_T_MAP   = np.vectorize(p_exact)(EPS_GRID, G_GRID)

        # ── 回路マップ
        print(f"\n  回路を実行中 ({n_eps}×{n_g} 点)...")
        P_CIRC_MAP = np.zeros_like(P_T_MAP)
        for i, gv in enumerate(g_arr):
            for j, ev in enumerate(eps_arr):
                dev = make_dev(mode, backend)
                P_CIRC_MAP[i, j] = build_static_circuit(dev, float(ev), float(gv), n_steps, mode)

        mae = np.mean(np.abs(P_CIRC_MAP - P_T_MAP))
        print(f"  完了。MAE (vs 解析) = {mae*100:.2f}%")

        # ── IBM モード: テキスト表示
        if mode == "ibm_real":
            print(f"\n  P_circuit マップ (行=g, 列=ε):")
            print(f"  g\\ε  " + "  ".join(f"{e:+.2f}" for e in eps_arr))
            print("  " + "-" * (len(eps_arr) * 7 + 5))
            for i, gv in enumerate(g_arr):
                row = "  ".join(f"{P_CIRC_MAP[i,j]:.3f}" for j in range(n_eps))
                print(f"  {gv:.2f}  {row}")

        # ── 可視化 (4 パネル)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        cmap = "hot"

        # Panel 1: P_max マップ (CI トポロジー)
        ax = axes[0, 0]
        cf = ax.contourf(EPS_GRID, G_GRID, P_MAX_MAP, levels=50, cmap=cmap)
        plt.colorbar(cf, ax=ax, label="P_max = (g/Ω₁)²")
        ax.axvline(EPS_RES, color="cyan", ls="--", lw=2, label=r"CI seam $\varepsilon=-\omega$")
        ax.scatter([EPS_RES], [0.0], c="white", s=300, zorder=6, marker="*", label="CI point")
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel("g")
        ax.set_title(r"$P_{max}(\varepsilon,g)=(g/\Omega_1)^2$" + "\nMax transition prob — CI topology")
        ax.legend(fontsize=8)

        # Panel 2: エネルギーギャップ ΔE
        ax = axes[0, 1]
        GAP_MAP = np.vectorize(gap)(EPS_GRID, G_GRID)
        cf2 = ax.contourf(EPS_GRID, G_GRID, GAP_MAP, levels=50, cmap="viridis")
        plt.colorbar(cf2, ax=ax, label="ΔE = 2Ω₁")
        ax.contour(EPS_GRID, G_GRID, GAP_MAP, levels=[0.1, 0.3, 0.6, 1.0],
                   colors="white", linewidths=0.8, linestyles="--")
        ax.axvline(EPS_RES, color="red", ls="--", lw=2, label="CI seam")
        ax.scatter([EPS_RES], [0.0], c="red", s=300, zorder=6, marker="*", label="CI point")
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel("g")
        ax.set_title(r"$\Delta E(\varepsilon,g)=2\Omega_1$" + "\nAdiabatic energy gap (zero at CI)")
        ax.legend(fontsize=8)

        # Panel 3: P(T=5) 解析マップ
        ax = axes[1, 0]
        cf3 = ax.contourf(EPS_GRID, G_GRID, P_T_MAP, levels=50, cmap=cmap)
        plt.colorbar(cf3, ax=ax, label=f"P(T={T_FINAL})")
        ax.contour(EPS_GRID, G_GRID, P_T_MAP,
                   levels=[0.3, 0.6, 0.9], colors="white", linewidths=0.8)
        ax.axvline(EPS_RES, color="cyan", ls="--", lw=2, label="CI seam")
        ax.axhline(np.pi / (2 * T_FINAL), color="lime", ls=":", lw=2,
                   label=f"g*=π/2T={np.pi/(2*T_FINAL):.3f}")
        ax.scatter([EPS_RES], [0.0], c="white", s=300, zorder=6, marker="*")
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel("g")
        ax.set_title(r"$P_{exact}(\varepsilon,g,T)=(g/\Omega_1)^2\sin^2(\Omega_1 T)$" + f"\nQuantum dynamics map (T={T_FINAL})")
        ax.legend(fontsize=8)

        # Panel 4: 回路 vs 解析の誤差
        ax = axes[1, 1]
        err_map = np.abs(P_CIRC_MAP - P_T_MAP) * 100
        cf4 = ax.contourf(EPS_GRID, G_GRID, err_map, levels=50, cmap="RdYlGn_r")
        plt.colorbar(cf4, ax=ax, label="|P_circuit − P_exact| (%)")
        ax.axvline(EPS_RES, color="cyan", ls="--", lw=2)
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel("g")
        ax.set_title(f"|P_circuit - P_exact| (%)\nTrotter error map (MAE={mae*100:.2f}%)")

        plt.suptitle(
            f"PES 2D Mapping — 2-qubit JT Model\n"
            r"$H=(\varepsilon/2)ZI+(\omega/2)IZ+g\cdot XX$, "
            f"$\\omega={OMEGA}$, $T={T_FINAL}$, N_steps={n_steps}",
            fontsize=12
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fname = f"pes_map_2q_{mode}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"\n  → マップ画像保存: {fname}")
        plt.close()

    else:
        print(f"  エラー: scan_type は 'eps', 'seam', 'map2d' を指定してください")
        sys.exit(1)

    print(f"\n{sep}\n")


if __name__ == "__main__":
    mode      = sys.argv[1] if len(sys.argv) > 1 else "simulate"
    scan_type = sys.argv[2] if len(sys.argv) > 2 else "eps"
    token     = sys.argv[3] if len(sys.argv) > 3 else None

    if mode == "ibm_real" and token is None:
        print("使い方: python pennylane_pes_map_2q.py ibm_real <eps|seam|map2d> <TOKEN>")
        sys.exit(1)

    run_scan(scan_type=scan_type, mode=mode, token=token)
