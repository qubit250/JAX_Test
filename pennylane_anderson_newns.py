"""
Resonance Level Model (最小 Anderson-Newns サンプル)
H = ε_d n_d + ε_k n_k + V (c†_d c_k + h.c.)

Jordan-Wigner (spinless, 2 qubit):
  n_d → (I - Z_0)/2,  n_k → (I - Z_1)/2
  c†_d c_k + h.c. → (X_0 X_1 + Y_0 Y_1)/2

Trotter 1ステップ:
  RZ(ε_d dt, 0), RZ(ε_k dt, 1), IsingXY(2V dt, [0,1])

初期状態: |10⟩ (分子占有, 金属空)
観測量: n_d(t) = P(qubit0=1) = 分子軌道電荷占有数
"""
import sys
import numpy as np

EPS_D    = -0.5   # 分子軌道エネルギー (フェルミ準位基準)
EPS_K    = 0.0    # 金属軌道エネルギー (フェルミ準位)
T_FINAL  = 4.0    # 時間
N_STEPS  = 20
SHOTS    = 8192

V_VALUES = np.array([0.10, 0.20, 0.30, 0.50, 0.80])


def exact_nd(v, eps_d=EPS_D, eps_k=EPS_K, t=T_FINAL):
    """4x4 行列対角化による厳密解 n_d(T)"""
    H = np.zeros((4, 4))
    H[1, 1] = eps_k
    H[2, 2] = eps_d
    H[3, 3] = eps_d + eps_k
    H[1, 2] = H[2, 1] = v    # hopping
    E, U = np.linalg.eigh(H)
    psi0 = np.array([0, 0, 1, 0], dtype=complex)  # |10⟩
    c = U.conj().T @ psi0
    psi_t = U @ (c * np.exp(-1j * E * t))
    return float(abs(psi_t[2])**2 + abs(psi_t[3])**2)


def run(mode="simulate", token=None):
    import pennylane as qml

    if mode == "ibm_real":
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=2)
        print(f"  -> {backend.name}")

    results = {}
    dt = T_FINAL / N_STEPS

    for v in V_VALUES:
        if mode == "simulate":
            dev = qml.device("default.qubit", wires=2)
        elif mode == "ibm_sim":
            dev = qml.device("default.qubit", wires=2, shots=SHOTS)
        else:
            dev = qml.device("qiskit.remote", wires=2, backend=backend, shots=SHOTS)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=0)          # 初期状態 |10⟩ (mol occupied)
            for _ in range(N_STEPS):
                qml.RZ(float(EPS_D * dt), wires=0)
                qml.RZ(float(EPS_K * dt), wires=1)
                qml.IsingXY(float(2.0 * v * dt), wires=[0, 1])
            return qml.probs(wires=[0, 1])

        probs = circuit()
        nd = float(probs[2] + probs[3])   # P(qubit0=1)
        nd_ex = exact_nd(v)
        results[float(v)] = nd
        print(f"  V={v:.2f}: n_d={nd:.4f}  (exact={nd_ex:.4f}  |Δ|={abs(nd-nd_ex):.4f})")

    print(f"\n  {'V':>6}  {'n_d_IBM':>10}  {'n_d_exact':>10}  {'|誤差|':>8}  {'誤差(%)':>8}")
    print("  " + "-"*52)
    for v in V_VALUES:
        nd = results[float(v)]
        nd_ex = exact_nd(v)
        print(f"  {v:>6.2f}  {nd:>10.4f}  {nd_ex:>10.4f}  {abs(nd-nd_ex):>8.4f}  {abs(nd-nd_ex)/max(abs(nd_ex),1e-6)*100:>7.2f}%")

    errors = [abs(results[float(v)] - exact_nd(v)) for v in V_VALUES]
    print(f"\n  MAE = {np.mean(errors)*100:.2f}%")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "simulate"
    token = sys.argv[2] if len(sys.argv) > 2 else None
    if mode == "ibm_real" and token is None:
        print("使い方: python pennylane_anderson_newns.py ibm_real <TOKEN>")
        sys.exit(1)
    run(mode=mode, token=token)
