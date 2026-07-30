"""
Azure Quantum 接続テスト
========================
使い方:
  pip install azure-quantum[qiskit] pennylane-qiskit
  python check_azure.py <RESOURCE_ID> <LOCATION> [BACKEND]

引数:
  RESOURCE_ID: /subscriptions/.../resourceGroups/.../providers/Microsoft.Quantum/Workspaces/...
               Azure Portal → Quantum workspace → Overview → Resource ID
  LOCATION:    eastus, westeurope など (Azure Portal で確認)
  BACKEND:     ionq.simulator (default, 無料) | ionq.qpu.aria-1 | quantinuum.sim.h1-1sc

テスト内容:
  1. azure-quantum インストール確認
  2. Azure Quantum ワークスペース接続
  3. 利用可能バックエンド一覧
  4. Bell 状態テスト (2-qubit, 1 CNOT)
  5. 3-qubit GHZ テスト (2 CNOT)
  6. 3-qubit Trotter 1 ステップ (4 CNOT) — 光化学モデル回路確認
"""

import sys
import numpy as np

RESOURCE_ID = sys.argv[1] if len(sys.argv) > 1 else None
LOCATION    = sys.argv[2] if len(sys.argv) > 2 else None
BACKEND     = sys.argv[3] if len(sys.argv) > 3 else "ionq.simulator"
SHOTS       = 1024

if RESOURCE_ID is None or LOCATION is None:
    print("Usage: python check_azure.py <RESOURCE_ID> <LOCATION> [BACKEND]")
    print("")
    print("  RESOURCE_ID: Azure Portal → Quantum workspace → Overview → Resource ID")
    print("  LOCATION:    eastus, westeurope, etc.")
    print("  BACKEND:     ionq.simulator (default) | ionq.qpu.aria-1 | quantinuum.sim.h1-1sc")
    sys.exit(1)


# ── [1] インストール確認 ───────────────────────────────────────────────────────
print("\n[1] azure-quantum / pennylane-qiskit インストール確認")
try:
    from azure.quantum import Workspace
    from azure.quantum.qiskit import AzureQuantumProvider
    import pennylane as qml
    print("  ✓ azure-quantum, pennylane-qiskit インポート成功")
except ImportError as e:
    print(f"  ✗ {e}")
    print("  → pip install azure-quantum[qiskit] pennylane-qiskit を実行してください")
    sys.exit(1)


# ── [2] ワークスペース接続 ─────────────────────────────────────────────────────
print(f"\n[2] Azure Quantum ワークスペース接続")
print(f"  Resource ID: {RESOURCE_ID[:60]}...")
print(f"  Location:    {LOCATION}")
try:
    workspace = Workspace(resource_id=RESOURCE_ID, location=LOCATION)
    provider  = AzureQuantumProvider(workspace)
    print("  ✓ ワークスペース接続成功")
except Exception as e:
    print(f"  ✗ {e}")
    sys.exit(1)


# ── [3] バックエンド一覧 ───────────────────────────────────────────────────────
print(f"\n[3] 利用可能バックエンド一覧")
try:
    backends = provider.backends()
    for b in backends:
        print(f"  - {b.name()}")
    print(f"  使用バックエンド: {BACKEND}")
except Exception as e:
    print(f"  ✗ {e}")


# ── [4] デバイス作成 ──────────────────────────────────────────────────────────
print(f"\n[4] デバイス作成 (2-qubit, backend={BACKEND})")
try:
    backend2 = provider.get_backend(BACKEND)
    dev2 = qml.device("qiskit.remote", wires=2, backend=backend2, shots=SHOTS)
    print(f"  ✓ 2-qubit デバイス作成成功")
except Exception as e:
    print(f"  ✗ {e}")
    sys.exit(1)


# ── [5] Bell 状態テスト ───────────────────────────────────────────────────────
print(f"\n[5] Bell 状態テスト (H+CNOT, 1 CNOT, shots={SHOTS})")
try:
    @qml.qnode(dev2)
    def bell():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    p = bell()
    print(f"  P(|00⟩) = {p[0]:.3f}  (理論: 0.500)")
    print(f"  P(|11⟩) = {p[3]:.3f}  (理論: 0.500)")
    print(f"  誤差: |P00-0.5|={abs(p[0]-0.5)*100:.1f}%, |P11-0.5|={abs(p[3]-0.5)*100:.1f}%")
    if p[0] > 0.40 and p[3] > 0.40:
        print("  ✓ Bell 状態 OK")
    else:
        print("  △ 誤差大 (ノイズ or 問題あり)")
except Exception as e:
    print(f"  ✗ {e}")
    sys.exit(1)


# ── [6] 3-qubit GHZ テスト ────────────────────────────────────────────────────
print(f"\n[6] 3-qubit GHZ テスト (H+2×CNOT, shots={SHOTS})")
try:
    backend3 = provider.get_backend(BACKEND)
    dev3 = qml.device("qiskit.remote", wires=3, backend=backend3, shots=SHOTS)

    @qml.qnode(dev3)
    def ghz3():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        return qml.probs(wires=[0, 1, 2])

    p = ghz3()
    print(f"  P(|000⟩) = {p[0]:.3f}  (理論: 0.500)")
    print(f"  P(|111⟩) = {p[7]:.3f}  (理論: 0.500)")
    print(f"  漏洩 P_rest = {1-p[0]-p[7]:.3f}  (理論: 0.000)")
    if p[0] > 0.40 and p[7] > 0.40:
        print("  ✓ 3-qubit GHZ OK — 光化学モデルの前提条件クリア")
    else:
        print("  △ 誤差大")
except Exception as e:
    print(f"  ✗ {e}")
    sys.exit(1)


# ── [7] Trotter 1 ステップ ────────────────────────────────────────────────────
print(f"\n[7] 3-qubit Trotter 1 ステップ (4 CNOT, ε=-1.0, φ=π/4)")
EPS    = -1.0
G      = 0.30
T      = 5.0
DT     = T
PHI    = np.pi / 4
G1     = G * np.cos(PHI)
G2     = G * np.sin(PHI)
OMEGA1 = 1.0
OMEGA2 = 1.5

try:
    backend_t = provider.get_backend(BACKEND)
    dev_t = qml.device("qiskit.remote", wires=3, backend=backend_t, shots=SHOTS)

    @qml.qnode(dev_t)
    def trotter_1step():
        qml.RZ(OMEGA1 * DT, wires=0)
        qml.RZ(EPS    * DT, wires=1)
        qml.RZ(OMEGA2 * DT, wires=2)
        qml.CNOT(wires=[0, 1])
        qml.RX(2.0 * G1 * DT, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.RX(2.0 * G2 * DT, wires=1)
        qml.CNOT(wires=[1, 2])
        return qml.probs(wires=[0, 1, 2])

    p = trotter_1step()
    print(f"  P(|000⟩)={p[0]:.4f}  P(|110⟩)={p[6]:.4f}  "
          f"P(|011⟩)={p[3]:.4f}  P(|101⟩)={p[5]:.4f}")
    print("  ✓ Trotter 回路実行成功 — 光化学モデル回路の動作確認 OK")
except Exception as e:
    print(f"  ✗ {e}")

print(f"\n{'='*60}")
print(f"  Azure Quantum 接続テスト完了 (backend={BACKEND})")
print(f"  光化学モデル実行コマンド:")
RID_SHORT = RESOURCE_ID[:40] + "..."
print(f"  python pennylane_3q_photochem.py azure eps_scan \\")
print(f"    '{RID_SHORT}' {LOCATION} {BACKEND}")
print(f"{'='*60}")
