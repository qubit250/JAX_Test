"""IBM 量子バックエンドの負荷状況確認"""
import sys

def check(token):
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)

    backends = service.backends(operational=True, simulator=False, min_num_qubits=2)
    print(f"\n{'バックエンド':<20} {'Qubit':>6} {'キュー':>6} {'稼働':>6}")
    print("-" * 44)
    for b in sorted(backends, key=lambda x: x.status().pending_jobs):
        s = b.status()
        print(f"{b.name:<20} {b.num_qubits:>6} {s.pending_jobs:>6} {'○' if s.operational else '×':>6}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python check_ibm_backends.py <TOKEN>")
        sys.exit(1)
    check(sys.argv[1])
