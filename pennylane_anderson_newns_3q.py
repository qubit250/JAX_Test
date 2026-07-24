"""
拡張 Anderson-Newns モデル (N_k=2, 3 qubit)
H = ε_d n_d + ε_{k1} n_{k1} + ε_{k2} n_{k2}
    + V_1(c†_d c_{k1} + h.c.) + V_2(c†_d c_{k2} + h.c.)

Jordan-Wigner 変換 (qubit 配置: wire0=k1, wire1=d, wire2=k2):
  n_{k1} → (I - Z_0)/2
  n_d    → (I - Z_1)/2
  n_{k2} → (I - Z_2)/2
  c†_{k1} c_d + h.c. → (X_0 X_1 + Y_0 Y_1)/2  → IsingXY([0,1])
  c†_d c_{k2} + h.c. → (X_1 X_2 + Y_1 Y_2)/2  → IsingXY([1,2])
  ※ d を中央に置くことで Z 文字列なし・最近接ゲートのみ

Trotter 1ステップ:
  RZ(ε_{k1} dt, 0), RZ(ε_d dt, 1), RZ(ε_{k2} dt, 2)
  IsingXY(2 V_1 dt, [0,1]), IsingXY(2 V_2 dt, [1,2])

初期状態: |010⟩  (d 占有, k1・k2 空)
観測量:   n_d(t) = P(wire1=1) = 分子軌道電荷占有数

スキャン種別 (コマンドライン第2引数):
  V   : 対称結合スキャン V_1=V_2=V, ε_k1=ε_k2=0  (N_k=1 比較付き)
  de  : バンド幅スキャン  ε_k1=-Δε, ε_k2=+Δε
  ed  : LUMO エネルギースキャン ε_d を変化

使い方:
  python pennylane_anderson_newns_3q.py [simulate|ibm_real] [V|de|ed] [TOKEN]
"""
import sys
import numpy as np

# ── デフォルトパラメータ ──────────────────────────────
EPS_D   = -0.5    # 分子軌道 LUMO エネルギー (フェルミ準位基準)
EPS_K1  = -0.3    # CNT k 点 1 (バンド下側)
EPS_K2  = +0.3    # CNT k 点 2 (バンド上側)
V1_DEF  = 0.3
V2_DEF  = 0.3
T_FINAL = 4.0
N_STEPS = 20
SHOTS   = 8192

# ── スキャン値 ────────────────────────────────────────
V_VALUES  = np.array([0.10, 0.20, 0.30, 0.50, 0.80])
DE_VALUES = np.array([0.00, 0.10, 0.30, 0.50, 1.00])  # Δε = バンド幅の半分
ED_VALUES = np.array([-1.00, -0.50, 0.00, 0.50, 1.00])


# ── 厳密解 ───────────────────────────────────────────

def exact_nd_3q(v1, v2, eps_k1=EPS_K1, eps_k2=EPS_K2, eps_d=EPS_D, t=T_FINAL):
    """N_k=2 厳密解: 3×3 単一粒子ハミルトニアンの対角化
    基底 {|k1⟩, |d⟩, |k2⟩}, 初期状態 |d⟩ = [0, 1, 0]
    """
    H = np.array([[eps_k1, v1,    0.0   ],
                  [v1,     eps_d, v2    ],
                  [0.0,    v2,    eps_k2]], dtype=float)
    E, U = np.linalg.eigh(H)
    psi0 = np.array([0.0, 1.0, 0.0], dtype=complex)
    c = U.conj().T @ psi0
    psi_t = U @ (c * np.exp(-1j * E * t))
    return float(abs(psi_t[1])**2)


def exact_nd_2q(v, eps_k=0.0, eps_d=EPS_D, t=T_FINAL):
    """N_k=1 厳密解: 2×2 単一粒子ハミルトニアン (比較用)"""
    H = np.array([[eps_k, v    ],
                  [v,     eps_d]], dtype=float)
    E, U = np.linalg.eigh(H)
    psi0 = np.array([0.0, 1.0], dtype=complex)
    c = U.conj().T @ psi0
    psi_t = U @ (c * np.exp(-1j * E * t))
    return float(abs(psi_t[1])**2)


# ── 量子回路 ──────────────────────────────────────────

def build_nd(dev, eps_k1, eps_k2, eps_d, v1, v2, dt, n_steps, mode):
    """3 qubit AN 回路を実行して n_d を返す"""
    import pennylane as qml

    @qml.set_shots(SHOTS if mode != "simulate" else None)
    @qml.qnode(dev)
    def circuit():
        qml.PauliX(wires=1)                               # |010⟩: d 占有
        for _ in range(n_steps):
            qml.RZ(float(eps_k1 * dt), wires=0)
            qml.RZ(float(eps_d  * dt), wires=1)
            qml.RZ(float(eps_k2 * dt), wires=2)
            qml.IsingXY(float(2.0 * v1 * dt), wires=[0, 1])
            qml.IsingXY(float(2.0 * v2 * dt), wires=[1, 2])
        return qml.probs(wires=[0, 1, 2])

    probs = circuit()
    # P(wire1=1): |010⟩=idx2, |011⟩=idx3, |110⟩=idx6, |111⟩=idx7
    return float(probs[2] + probs[3] + probs[6] + probs[7])


# ── スキャン実行 ──────────────────────────────────────

def run_scan(scan_type="V", mode="simulate", token=None):
    import pennylane as qml

    dt = T_FINAL / N_STEPS

    if mode == "ibm_real":
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=3)
        print(f"  バックエンド: {backend.name}")

    def make_dev():
        if mode == "ibm_real":
            return qml.device("qiskit.remote", wires=3, backend=backend)
        return qml.device("default.qubit", wires=3)

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  拡張 Anderson-Newns  N_k=2 (3 qubit)  —  {scan_type} スキャン")
    print(f"  T={T_FINAL}, N_steps={N_STEPS}, dt={dt:.3f}, mode={mode}")
    print(sep)

    def print_header(param_name):
        print(f"\n  {param_name:>6}  {'n_d_3q':>10}  {'exact_3q':>10}  "
              f"{'|Δ|':>8}  {'err%':>7}")
        print("  " + "-" * 50)

    def print_row(param, nd, nd_ex):
        err = abs(nd - nd_ex)
        pct = err / max(abs(nd_ex), 1e-6) * 100
        print(f"  {param:>6.3f}  {nd:>10.4f}  {nd_ex:>10.4f}  {err:>8.4f}  {pct:>6.2f}%")
        return err

    # ── V スキャン ──────────────────────────────────
    if scan_type == "V":
        print(f"\n  ε_d={EPS_D}, ε_k1=ε_k2=0 (対称結合), V_1=V_2=V")
        print(f"\n  [N_k=2 vs N_k=1 の比較]")
        print(f"\n  {'V':>6}  {'nd_3q':>10}  {'ex_3q':>10}  "
              f"{'nd_2q(比較)':>12}  {'ex_2q':>10}  {'err_3q%':>8}")
        print("  " + "-" * 66)

        errs = []
        for v in V_VALUES:
            dev = make_dev()
            nd_3q = build_nd(dev, 0.0, 0.0, EPS_D, v, v, dt, N_STEPS, mode)
            ex_3q = exact_nd_3q(v, v, eps_k1=0.0, eps_k2=0.0)
            ex_2q = exact_nd_2q(v, eps_k=0.0)
            err = abs(nd_3q - ex_3q)
            pct = err / max(abs(ex_3q), 1e-6) * 100
            # N_k=1 は simulate のみ (参考値)
            nd_2q = exact_nd_2q(v, eps_k=0.0)
            print(f"  {v:>6.2f}  {nd_3q:>10.4f}  {ex_3q:>10.4f}  "
                  f"{nd_2q:>12.4f}  {ex_2q:>10.4f}  {pct:>7.2f}%")
            errs.append(err)
        print(f"\n  MAE (3 qubit) = {np.mean(errs)*100:.2f}%")
        print(f"\n  [物理的観察] 対称結合では有効結合 V_eff = √2·V に増強")
        print(f"  → N_k=2 の電荷移動は N_k=1 より速い (broad band 極限への収束)")

    # ── Δε スキャン ────────────────────────────────
    elif scan_type == "de":
        v = 0.3
        print(f"\n  ε_d={EPS_D}, V_1=V_2={v}, ε_k1=-Δε, ε_k2=+Δε (バンド幅効果)")
        print_header("Δε")
        errs = []
        for de in DE_VALUES:
            dev = make_dev()
            nd = build_nd(dev, -de, +de, EPS_D, v, v, dt, N_STEPS, mode)
            nd_ex = exact_nd_3q(v, v, eps_k1=-de, eps_k2=+de)
            errs.append(print_row(de, nd, nd_ex))
        print(f"\n  MAE = {np.mean(errs)*100:.2f}%")
        print(f"\n  [物理的観察] Δε 増大 = k 点エネルギー分散の拡大")
        print(f"  → ε_d との共鳴条件 (ε_k = ε_d) が崩れ n_d の振動が抑制される")

    # ── ε_d スキャン ────────────────────────────────
    elif scan_type == "ed":
        v = 0.3
        print(f"\n  ε_k1={EPS_K1}, ε_k2={EPS_K2}, V_1=V_2={v} (LUMO エネルギー依存性)")
        print_header("ε_d")
        errs = []
        for ed in ED_VALUES:
            dev = make_dev()
            nd = build_nd(dev, EPS_K1, EPS_K2, ed, v, v, dt, N_STEPS, mode)
            nd_ex = exact_nd_3q(v, v, eps_k1=EPS_K1, eps_k2=EPS_K2, eps_d=ed)
            errs.append(print_row(ed, nd, nd_ex))
        print(f"\n  MAE = {np.mean(errs)*100:.2f}%")
        print(f"\n  [物理的観察] ε_d → 0 (共鳴条件) で電荷移動が最大化")
        print(f"  → 官能基 LUMO が CNT バンド中心に近いほど電荷移動効率が高い")

    else:
        print(f"  エラー: scan_type は 'V', 'de', 'ed' を指定してください")
        sys.exit(1)

    print(f"\n{sep}\n")


if __name__ == "__main__":
    mode      = sys.argv[1] if len(sys.argv) > 1 else "simulate"
    scan_type = sys.argv[2] if len(sys.argv) > 2 else "V"
    token     = sys.argv[3] if len(sys.argv) > 3 else None

    if mode == "ibm_real" and token is None:
        print("使い方: python pennylane_anderson_newns_3q.py ibm_real <V|de|ed> <TOKEN>")
        sys.exit(1)

    run_scan(scan_type=scan_type, mode=mode, token=token)
