"""
IonQ 接続テスト
================
使い方:
  pip install pennylane-ionq
  python check_ionq.py <IONQ_API_KEY> [target]

target: simulator (default, 無料) | aria-1 | forte-1

テスト内容:
  1. pennylane-ionq インストール確認
  2. IonQ デバイス一覧取得
  3. Bell 状態回路 (2-qubit, 1 CNOT)
  4. 3-qubit GHZ 回路 (2 CNOT) — 光化学モデルと同回路数/qubit数
"""

import sys
import numpy as np

API_KEY = sys.argv[1] if len(sys.argv) > 1 else None
TARGET  = sys.argv[2] if len(sys.argv) > 2 else "simulator"
SHOTS   = 1024

if API_KEY is None:
    print("Usage: python check_ionq.py <IONQ_API_KEY> [simulator|aria-1|forte-1]")
    sys.exit(1)


# ── [1] pennylane-ionq インストール確認 ──────────────────────────────────────
print("\n[1] pennylane-ionq インストール確認")
try:
    import pennylane as qml
    # IonQ デバイスが認識されるか確認
    devs = [d for d in qml.plugin_devices if "ionq" in d]
    if devs:
        print(f"  ✓ pennylane-ionq: {devs}")
    else:
        print("  pennylane-ionq が未検出 — デバイス作成で確認します")
except Exception as e:
    print(f"  ✗ {e}")
    sys.exit(1)


# ── [2] デバイス作成 ─────────────────────────────────────────────────────────
print(f"\n[2] IonQ デバイス作成 (target={TARGET})")
try:
    if TARGET == "simulator":
        dev2 = qml.device("ionq.simulator", wires=2, shots=SHOTS, api_key=API_KEY)
    else:
        dev2 = qml.device("ionq.qpu", wires=2, shots=SHOTS,
                           api_key=API_KEY, target=TARGET)
    print(f"  ✓ 2-qubit デバイス作成成功: {dev2}")
except Exception as e:
    print(f"  ✗ {e}")
    print("  → pip install pennylane-ionq を実行してください")
    sys.exit(1)


# ── [3] Bell 状態テスト ───────────────────────────────────────────────────────
print(f"\n[3] Bell 状態テスト (H+CNOT, 1 CNOT, shots={SHOTS})")
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


# ── [4] 3-qubit GHZ テスト ────────────────────────────────────────────────────
print(f"\n[4] 3-qubit GHZ テスト (H+2×CNOT, shots={SHOTS})")
try:
    if TARGET == "simulator":
        dev3 = qml.device("ionq.simulator", wires=3, shots=SHOTS, api_key=API_KEY)
    else:
        dev3 = qml.device("ionq.qpu", wires=3, shots=SHOTS,
                           api_key=API_KEY, target=TARGET)

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


# ── [5] Trotter 1 ステップテスト ─────────────────────────────────────────────
print(f"\n[5] 3-qubit Trotter 1 ステップ (4 CNOT, ε=-1.0, φ=π/4)")
EPS   = -1.0
G     = 0.30
T     = 5.0
N     = 1
DT    = T / N
PHI   = np.pi / 4
G1    = G * np.cos(PHI)
G2    = G * np.sin(PHI)
OMEGA1 = 1.0
OMEGA2 = 1.5

try:
    if TARGET == "simulator":
        dev_t = qml.device("ionq.simulator", wires=3, shots=SHOTS, api_key=API_KEY)
    else:
        dev_t = qml.device("ionq.qpu", wires=3, shots=SHOTS,
                            api_key=API_KEY, target=TARGET)

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
print(f"  IonQ 接続テスト完了 (target={TARGET})")
print(f"  光化学モデル実行コマンド:")
print(f"  python pennylane_3q_photochem.py ionq eps_scan {API_KEY[:8]}... [aria-1]")
print(f"  python pennylane_3q_photochem.py ionq phi_scan {API_KEY[:8]}... [aria-1]")
print(f"{'='*60}")
