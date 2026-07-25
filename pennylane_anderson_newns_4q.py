"""
拡張 Anderson-Newns モデル (分子ダイマー, 4 qubit)
H = ε_k1 n_k1 + ε_d1 n_d1 + ε_d2 n_d2 + ε_k2 n_k2
    + V_1(c†_{k1} c_{d1} + h.c.)
    + t  (c†_{d1} c_{d2} + h.c.)
    + V_2(c†_{d2} c_{k2} + h.c.)

Jordan-Wigner (qubit 配置: wire0=k1, wire1=d1, wire2=d2, wire3=k2):
  全ホッピングが最近接 → Z 文字列なし
  IsingXY([0,1]): k1-d1  (V1)
  IsingXY([1,2]): d1-d2  (t)
  IsingXY([2,3]): d2-k2  (V2)

Trotter 1ステップ:
  RZ(ε_k1 dt,0), RZ(ε_d1 dt,1), RZ(ε_d2 dt,2), RZ(ε_k2 dt,3)
  IsingXY(2V1 dt,[0,1]), IsingXY(2t dt,[1,2]), IsingXY(2V2 dt,[2,3])

初期状態: |0100⟩ (d1 占有, 他は空)
観測量:   n_d1(t) = P(wire1=1)  分子軌道 1 電荷占有数
          n_d2(t) = P(wire2=1)  分子軌道 2 電荷占有数

スキャン種別 (コマンドライン第2引数):
  t  : d1-d2 結合スキャン → 2 経路干渉の確認  [主要スキャン]
  V  : 対称結合スキャン V_1=V_2=V            [3-qubit との比較]
  ed1: ε_d1 スキャン → 非対称 LUMO 依存性

使い方:
  python pennylane_anderson_newns_4q.py [simulate|ibm_real] [t|V|ed1] [TOKEN]
"""
import sys
import numpy as np

# ── デフォルトパラメータ ─────────────────────────────
EPS_D1  = -0.5
EPS_D2  = -0.5
EPS_K1  =  0.0
EPS_K2  =  0.0
V1_DEF  =  0.3
V2_DEF  =  0.3
T_DEF   =  0.3   # d1-d2 結合
T_FINAL =  2.0
N_STEPS = 10
SHOTS   = 8192

# ── スキャン値 ────────────────────────────────────────
T_VALUES   = np.array([0.0, 0.1, 0.3, 0.5, 0.8])   # d1-d2 結合スキャン
V_VALUES   = np.array([0.10, 0.20, 0.30, 0.50, 0.80])
ED1_VALUES = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])


# ── 厳密解 ────────────────────────────────────────────

def exact_4q(v1, v2, t_hop, eps_k1=EPS_K1, eps_k2=EPS_K2,
             eps_d1=EPS_D1, eps_d2=EPS_D2, T=T_FINAL):
    """4×4 単一粒子ハミルトニアンの対角化
    基底: {|k1⟩, |d1⟩, |d2⟩, |k2⟩}, 初期状態 |d1⟩=[0,1,0,0]
    戻り値: (n_d1, n_d2)
    """
    H = np.array([[eps_k1, v1,    0,     0     ],
                  [v1,     eps_d1, t_hop, 0     ],
                  [0,      t_hop, eps_d2, v2    ],
                  [0,      0,     v2,    eps_k2 ]], dtype=float)
    E, U = np.linalg.eigh(H)
    psi0 = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
    c = U.conj().T @ psi0
    psi_t = U @ (c * np.exp(-1j * E * T))
    return float(abs(psi_t[1])**2), float(abs(psi_t[2])**2)


def exact_2q(v, eps_k=0.0, eps_d=EPS_D1, T=T_FINAL):
    """2×2 厳密解 (t=0 の検証用)"""
    H = np.array([[eps_k, v   ],
                  [v,     eps_d]], dtype=float)
    E, U = np.linalg.eigh(H)
    psi0 = np.array([0.0, 1.0], dtype=complex)
    c = U.conj().T @ psi0
    psi_t = U @ (c * np.exp(-1j * E * T))
    return float(abs(psi_t[1])**2)


# ── 量子回路 ──────────────────────────────────────────

def build_nd(dev, eps_k1, eps_k2, eps_d1, eps_d2,
             v1, v2, t_hop, dt, n_steps, mode):
    """4 qubit ダイマー AN 回路を実行して (n_d1, n_d2) を返す"""
    import pennylane as qml

    @qml.set_shots(SHOTS if mode != "simulate" else None)
    @qml.qnode(dev)
    def circuit():
        qml.PauliX(wires=1)                           # |0100⟩: d1 占有
        for _ in range(n_steps):
            qml.RZ(float(eps_k1 * dt), wires=0)
            qml.RZ(float(eps_d1 * dt), wires=1)
            qml.RZ(float(eps_d2 * dt), wires=2)
            qml.RZ(float(eps_k2 * dt), wires=3)
            qml.IsingXY(float(2.0 * v1    * dt), wires=[0, 1])  # k1-d1
            qml.IsingXY(float(2.0 * t_hop * dt), wires=[1, 2])  # d1-d2
            qml.IsingXY(float(2.0 * v2    * dt), wires=[2, 3])  # d2-k2
        return qml.probs(wires=[0, 1, 2, 3])

    probs = circuit()
    # n_d1 = P(wire1=1): indices 4-7, 12-15
    nd1 = float(probs[4:8].sum() + probs[12:16].sum())
    # n_d2 = P(wire2=1): indices 2,3,6,7,10,11,14,15
    nd2 = float(probs[[2,3,6,7,10,11,14,15]].sum())
    return nd1, nd2


# ── スキャン実行 ──────────────────────────────────────

def run_scan(scan_type="t", mode="simulate", token=None):
    import pennylane as qml

    dt = T_FINAL / N_STEPS

    if mode == "ibm_real":
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=4)
        print(f"  バックエンド: {backend.name}")

    def make_dev():
        if mode == "ibm_real":
            return qml.device("qiskit.remote", wires=4, backend=backend)
        return qml.device("default.qubit", wires=4)

    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  AN 分子ダイマー  N_k=1+1 (4 qubit)  —  {scan_type} スキャン")
    print(f"  T={T_FINAL}, N_steps={N_STEPS}, dt={dt:.3f}, mode={mode}")
    print(sep)

    def header(p):
        print(f"\n  {p:>6}  {'nd1_IBM':>9}  {'nd1_ex':>9}  "
              f"{'nd2_IBM':>9}  {'nd2_ex':>9}  {'err_d1%':>8}")
        print("  " + "-" * 60)

    def row(p, nd1, ex1, nd2, ex2):
        e1 = abs(nd1 - ex1) / max(abs(ex1), 1e-6) * 100
        print(f"  {p:>6.3f}  {nd1:>9.4f}  {ex1:>9.4f}  "
              f"{nd2:>9.4f}  {ex2:>9.4f}  {e1:>7.2f}%")
        return abs(nd1 - ex1)

    # ── t スキャン ────────────────────────────────────
    if scan_type == "t":
        print(f"\n  V_1=V_2={V1_DEF}, ε_d1=ε_d2={EPS_D1}, "
              f"ε_k1=ε_k2={EPS_K1}")
        print(f"  t=0 の検証: 2-qubit AN と一致するはず "
              f"(n_d1={exact_2q(V1_DEF):.4f}, n_d2=0.0000)")
        header("t")
        errs = []
        for t_val in T_VALUES:
            dev = make_dev()
            nd1, nd2 = build_nd(dev, EPS_K1, EPS_K2, EPS_D1, EPS_D2,
                                 V1_DEF, V2_DEF, t_val, dt, N_STEPS, mode)
            ex1, ex2 = exact_4q(V1_DEF, V2_DEF, t_val)
            errs.append(row(t_val, nd1, ex1, nd2, ex2))
        print(f"\n  MAE (n_d1) = {np.mean(errs)*100:.2f}%")
        print(f"\n  [物理的観察]")
        print(f"  t=0: 2 経路が切断 → n_d1 は 2-qubit AN と同一")
        print(f"  t>0: d1→d2→k2 の第 2 経路が開通 → n_d1・n_d2 が変化")
        print(f"  t≈V: 2 経路が競合 → 構成的/破壊的干渉で n_d1 が極値をとる可能性")

    # ── V スキャン ────────────────────────────────────
    elif scan_type == "V":
        print(f"\n  t={T_DEF}, ε_d1=ε_d2={EPS_D1}, ε_k1=ε_k2={EPS_K1}, "
              f"V_1=V_2=V")
        header("V")
        errs = []
        for v in V_VALUES:
            dev = make_dev()
            nd1, nd2 = build_nd(dev, EPS_K1, EPS_K2, EPS_D1, EPS_D2,
                                 v, v, T_DEF, dt, N_STEPS, mode)
            ex1, ex2 = exact_4q(v, v, T_DEF)
            errs.append(row(v, nd1, ex1, nd2, ex2))
        print(f"\n  MAE (n_d1) = {np.mean(errs)*100:.2f}%")

    # ── ε_d1 スキャン ─────────────────────────────────
    elif scan_type == "ed1":
        eps_d2_fixed = EPS_D2
        print(f"\n  V_1=V_2={V1_DEF}, t={T_DEF}, "
              f"ε_d2={eps_d2_fixed}(固定), ε_k1=ε_k2={EPS_K1}")
        print(f"  ← 2 つの官能基の LUMO エネルギーが異なる場合の効果")
        header("ε_d1")
        errs = []
        for ed1 in ED1_VALUES:
            dev = make_dev()
            nd1, nd2 = build_nd(dev, EPS_K1, EPS_K2, ed1, eps_d2_fixed,
                                 V1_DEF, V2_DEF, T_DEF, dt, N_STEPS, mode)
            ex1, ex2 = exact_4q(V1_DEF, V2_DEF, T_DEF,
                                 eps_d1=ed1, eps_d2=eps_d2_fixed)
            errs.append(row(ed1, nd1, ex1, nd2, ex2))
        print(f"\n  MAE (n_d1) = {np.mean(errs)*100:.2f}%")
        print(f"\n  [物理的観察]")
        print(f"  ε_d1 = ε_d2 = {eps_d2_fixed}: 対称 → n_d1 = n_d2 (等分配)")
        print(f"  ε_d1 ≠ ε_d2: 非対称 → エネルギーの低い軌道に電荷が偏る")

    else:
        print(f"  エラー: scan_type は 't', 'V', 'ed1' を指定してください")
        sys.exit(1)

    print(f"\n{sep}\n")


if __name__ == "__main__":
    mode      = sys.argv[1] if len(sys.argv) > 1 else "simulate"
    scan_type = sys.argv[2] if len(sys.argv) > 2 else "t"
    token     = sys.argv[3] if len(sys.argv) > 3 else None

    if mode == "ibm_real" and token is None:
        print("使い方: python pennylane_anderson_newns_4q.py ibm_real <t|V|ed1> <TOKEN>")
        sys.exit(1)

    run_scan(scan_type=scan_type, mode=mode, token=token)
