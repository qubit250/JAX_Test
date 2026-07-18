"""
Stark-shifted Spin-Boson: ε_Stark スキャン (最小構成 4 回路)
g=0.5 (Toulouse 点, α=0.5) 固定
ε=0 結果は SB 実機結果 (+0.2302) を流用
"""
import sys
import numpy as np

G_FIXED   = 0.5
DELTA     = 0.5
OMEGA     = 1.0
T_FINAL   = 4.0
N_STEPS   = 20
SHOTS     = 8192

EPS_VALUES = np.array([0.25, 0.50, 1.00, 2.00])


def exact_sz(eps0, g=G_FIXED, delta=DELTA, omega=OMEGA, t=T_FINAL):
    H = np.zeros((4, 4), dtype=complex)
    H[0,0] += eps0/2; H[1,1] += eps0/2
    H[2,2] -= eps0/2; H[3,3] -= eps0/2
    H[0,0] += omega/2; H[1,1] -= omega/2
    H[2,2] += omega/2; H[3,3] -= omega/2
    H[0,2] += delta/2; H[2,0] += delta/2
    H[1,3] += delta/2; H[3,1] += delta/2
    H[0,1] += g; H[1,0] += g
    H[2,3] -= g; H[3,2] -= g
    E, V = np.linalg.eigh(H)
    psi0 = np.array([1,0,0,0], dtype=complex)
    c = V.conj().T @ psi0
    psi_t = V @ (c * np.exp(-1j * E * t))
    prob = np.abs(psi_t)**2
    return float((prob[0]+prob[1]) - (prob[2]+prob[3]))


def run(token):
    import pennylane as qml
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=2)
    print(f"  -> {backend.name}")

    results = {0.0: +0.2302}  # SB 実機結果を流用
    dt = T_FINAL / N_STEPS

    for eps0 in EPS_VALUES:
        dev = qml.device("qiskit.remote", wires=2, backend=backend, shots=SHOTS)

        @qml.qnode(dev)
        def circuit():
            for _ in range(N_STEPS):
                qml.RZ(float(eps0 * dt), wires=0)
                qml.RX(float(DELTA * dt), wires=0)
                qml.RZ(float(OMEGA * dt), wires=1)
                qml.Hadamard(wires=1)
                qml.CNOT(wires=[0, 1])
                qml.RZ(float(2.0 * G_FIXED * dt), wires=1)
                qml.CNOT(wires=[0, 1])
                qml.Hadamard(wires=1)
            return qml.probs(wires=[0, 1])

        probs = circuit()
        sz = float((probs[0]+probs[1]) - (probs[2]+probs[3]))
        sz_ex = exact_sz(eps0)
        print(f"  ε={eps0:.2f}: ⟨ZI⟩={sz:+.4f}  (exact={sz_ex:+.4f}  |Δ|={abs(sz-sz_ex):.4f})")
        results[float(eps0)] = sz

    print(f"\n  {'ε_Stark':>8}  {'IBM':>8}  {'exact':>8}  {'|誤差|':>8}")
    print("  " + "-"*38)
    for eps in [0.0, 0.25, 0.50, 1.00, 2.00]:
        sz = results[eps]
        sz_ex = exact_sz(eps)
        note = " (流用)" if eps == 0.0 else ""
        print(f"  {eps:>8.2f}  {sz:>+8.4f}  {sz_ex:>+8.4f}  {abs(sz-sz_ex):>8.4f}{note}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python pennylane_stark_sb_quick.py <TOKEN>")
        sys.exit(1)
    run(sys.argv[1])
