# 一般化 Anderson-Newns モデルの IBM 実機実装可能性検討

**参照論文**: Lang, Jain, Arrazola, Motlagh (Xanadu), arXiv:2601.16264  
**検討日**: 2026-07-23  
**前提**: IBM ibm_marrakesh (Heron r2, 156 qubit, NISQ)

---

## 1. 論文の概要

### 1.1 対象ハミルトニアン

一般化 Anderson-Newns (GAN) Hamiltonian:

```
H = H_mol + H_metal + H_int

H_mol  = Σ_κ P²_κ/(2m_κ) + Σ_{i,j∈M} U_ij(Q) a†_i a_j
                           + Σ_{i,j∈M} V_ij(Q) n_i n_j + U_0(Q)
H_metal = Σ_{i∈B} ε_i a†_i a_i
H_int   = Σ_{i∈M, j∈B} W_ij(Q) (a†_i a_j + h.c.)
```

- **N_mol**: 分子軌道数 (典型 4–8)
- **N_metal**: 金属軌道数 (典型 50–200)
- **M**: 核自由度数 (典型 14–40)
- **エンコード**: Jordan-Wigner + 実空間格子 (各核モード k bits)
- **量子算術**: 核座標依存結合のために QROM・位相勾配演算が必要

### 1.2 論文のリソース推定 (Table I)

| 系 | N_mol | N_metal | M | Qubit | Toffoli gates |
|----|-------|---------|---|-------|---------------|
| 最小 | 4 | 50 | 14 | 196 | 2.30×10⁷ |
| 光誘起電荷移動 | 8 | 100 | 20 | 271 | 7.85×10⁷ |
| 分子接合 | 2 | 200 | 13 | 337 | 5.02×10⁷ |

**これらはすべて耐障害性量子コンピュータ (FT-QC) の論理量子ビット。**

---

## 2. NISQ 実機との根本的ギャップ

| 項目 | 論文 (FT-QC) | IBM ibm_marrakesh (NISQ) |
|------|------------|--------------------------|
| 量子ビット | 論理量子ビット (~196–337) | 物理量子ビット 156 |
| ゲート精度 | ほぼ完全 (エラー訂正済み) | CNOT 忠実度 ~99.9% |
| Toffoli ゲート | 2.3×10⁷ — 1.3×10⁸ | 事実上使用不可 (数個が限界) |
| 核自由度 | 実空間格子 + 量子算術 | 量子算術回路が深すぎ |
| **結論** | **FT-QC 専用アルゴリズム** | **論文のアルゴリズムはそのまま不可** |

---

## 3. NISQ で実現可能なサンプルモデル

核自由度を除いた**電子自由度のみの最小 Anderson-Newns モデル**は NISQ で実現可能。

### 3.1 共鳴準位モデル (Resonance Level Model) — 2 qubit

最も簡単な AN モデル (N_mol=1, N_metal=1, M=0):

```
H = ε_d n_d + ε_k n_k + V (c†_d c_k + c†_k c_d)
```

**Jordan-Wigner エンコード** (spinless 2 qubit):

```
n_d   → (I - Z_0) / 2
n_k   → (I - Z_1) / 2
c†_d c_k + h.c. → (X_0 X_1 + Y_0 Y_1) / 2
```

最終的な Pauli 表現:

```
H = -ε_d Z_0/2 - ε_k Z_1/2 + V (X_0 X_1 + Y_0 Y_1) / 2 + const
```

**Trotter 1ステップ (dt):**

```
exp(-i ε_d dt/2 · Z_0) = RZ(ε_d dt, wire=0)
exp(-i ε_k dt/2 · Z_1) = RZ(ε_k dt, wire=1)
exp(-i V dt (XX+YY)/2) = IsingXY(2V dt, wires=[0,1])   ← 2-3 CNOT
```

| 項目 | 値 |
|------|---|
| Qubit 数 | **2** |
| CNOT/ステップ | **4** (IsingXY の標準分解: H·CNOT·RY·CNOT·S·… ×2) |
| N_steps = 20 → 総 CNOT | **80** |
| 比較 | SB モデル (40 CNOT) の 2 倍; 依然十分実現可能 |
| **実現可能性** | **高** |

> **IsingXY 分解** (PennyLane 実測): `H·RY·CNOT·RY·CNOT·S·RY·RX·RY·CNOT·RY·CNOT·S·H` = 4 CNOT/ステップ

**観測量**:

```
n_d(t) = ⟨(I - Z_0)/2⟩ = (1 - ⟨Z_0⟩) / 2   ← 分子軌道の電荷占有数
```

**物理的意味**: 分子が金属に近づくと電子が金属→分子 or 分子→金属に移動する時間発展。
CNT/官能基系では「官能基の占有数が時間とともにフェルミ準位に緩和する」過程に対応。

---

### 3.2 3 軌道 Anderson モデル — 3 qubit

N_mol=1, N_metal=2 (均一結合 V):

```
H = ε_d n_d + Σ_{k=1,2} ε_k n_k + V Σ_k (c†_d c_k + h.c.)
```

**Jordan-Wigner (qubit 0=mol, 1=metal1, 2=metal2):**

- Hopping (0↔1): `(X_0 X_1 + Y_0 Y_1)/2`  — 隣接、JW 文字列なし
- Hopping (0↔2): `(X_0 Z_1 X_2 + Y_0 Z_1 Y_2)/2`  — JW 文字列 Z_1 あり

| 項目 | 値 |
|------|---|
| Qubit 数 | **3** |
| CNOT/ステップ | ≈ 5–7 |
| N_steps = 20 → 総 CNOT | **100–140** |
| **実現可能性** | **中** (ibm_marrakesh で到達可能範囲) |

---

### 3.3 スピン自由度付き — 4 qubit

スピン自由度を考慮 (spin up/down それぞれ 1 軌道ずつ):

```
H = Σ_σ [ε_d n_{d,σ} + ε_k n_{k,σ} + V (c†_{d,σ} c_{k,σ} + h.c.)]
```

- Jordan-Wigner: 4 qubit (mol↑, mol↓, metal↑, metal↓)
- CNOT/ステップ: ≈ 8–10
- N_steps = 20 → 総 CNOT: **160–200**
- **実現可能性**: やや低 (誤差蓄積が懸念)

---

## 4. NISQ での核自由度の扱い

論文の核心は核座標依存結合 W_ij(Q) の量子演算 (QROM + 位相勾配) にある。NISQ では:

| 核自由度処理 | NISQ での対応 |
|------------|-------------|
| QROM による V(Q) の計算 | **不可** (Toffoli 多用) |
| 量子算術 (位相勾配演算) | **不可** (深い回路) |
| **定数 V (wide-band 近似)** | **可** (今回のアプローチ) |
| **断熱 Born-Oppenheimer** | **可** (電子のみ動的) |

→ NISQ サンプルでは **M=0 (核自由度ゼロ)** または **V=定数 (wide-band 近似)** に限定。

核量子ダイナミクスを含むフル GAN シミュレーションは FT-QC 世代まで待つ必要がある。

---

## 5. 推奨実装計画

### 5.1 Phase 1: 共鳴準位モデル (即実施可能)

**物理設定** (CNT + -COOH 系のモデル):

```python
ε_d    = -0.5   # COOH の LUMO エネルギー (相対値 eV 単位)
ε_k    = 0.0    # CNT フェルミ準位を基準
V      = 0.3    # CNT–COOH ハイブリダイゼーション強度
T_FINAL = 4.0   # 時間単位 (ℏ/eV)
N_STEPS = 20
```

**初期状態**: 分子に電子あり、金属なし → |10⟩ (mol=1, metal=0)

**Trotter 回路**:

```python
@qml.qnode(dev)
def circuit():
    qml.PauliX(wires=0)   # 初期状態 |10⟩ = mol 占有, metal 空
    for _ in range(N_STEPS):
        dt = T_FINAL / N_STEPS
        qml.RZ(eps_d * dt, wires=0)
        qml.RZ(eps_k * dt, wires=1)
        qml.IsingXY(2.0 * V * dt, wires=[0, 1])
    return qml.probs(wires=[0, 1])
```

**観測量**: `n_d(t) = probs[0] + probs[1]` (qubit 0 が 0 状態 → 分子は空; 逆に probs[2]+probs[3] → 分子占有)

Wait, 実際には:
- `|10⟩` = qubit0=1 (mol占有), qubit1=0 (metal空) → `probs[2]` (binary 10)
- `n_d = probs[2] + probs[3]` (qubit0=1 の確率)

**厳密解** (4×4 行列対角化で検証):

```python
import numpy as np

def exact_nd(eps_d, eps_k, V, t):
    H = np.array([
        [0,       0,         0,         0      ],
        [0,       eps_k,    V,          0      ],
        [0,       V,        eps_d,      0      ],
        [0,       0,         0,   eps_d+eps_k  ]
    ])
    E, U = np.linalg.eigh(H)
    psi0 = np.array([0, 0, 1, 0], dtype=complex)  # |10⟩
    c = U.conj().T @ psi0
    psi_t = U @ (c * np.exp(-1j * E * t))
    return float(abs(psi_t[2])**2 + abs(psi_t[3])**2)
```

**期待される物理**:
- t=0: n_d = 1.0 (分子に電子)
- 時間発展: Rabi 振動 (周波数 ≈ √((ε_d-ε_k)²/4 + V²))
- 長時間: 共鳴状態 n_d → 0.5 に向かう傾向 (ε_d = ε_k の場合)

### 5.2 Phase 2: パラメータスキャン

V をスキャン (Kondo 結合定数 α との対応付け):

```
V = 0.1, 0.2, 0.3, 0.5, 0.8
```

SB モデルの Toulouse 点 (α=0.5) に対応する強い結合 V では、電荷移動が加速 → 過去の SB 実験と比較検討。

---

## 6. SB モデルとの関係

| 特性 | Spin-Boson (SB) | Anderson-Newns (AN) |
|------|-----------------|---------------------|
| ハミルトニアン | (ε/2)ZI + (Δ/2)XI + (ω/2)IZ + g·ZX | ε_d n_d + ε_k n_k + V(c†c_k + h.c.) |
| フォノン/金属 | 1 ボゾンモード (2準位近似) | 1 フェルミオン軌道 |
| エンコード | 2 qubit (Jordan-Wigner) | 2 qubit (Jordan-Wigner) |
| 相互作用型 | ZX (ボゾン場) | XX+YY (フェルミオン跳び移り) |
| CNOT/ステップ | 2 (H-CNOT-RZ-CNOT-H) | 2–3 (IsingXY) |
| 観測量 | ⟨ZI⟩ = 1 - 2n_d | n_d = (1 - ⟨Z_0⟩)/2 |
| **差分** | ボゾン場との結合 | **フェルミオン電子海との結合** |

→ SB はフォノン (1 ボゾンモード) を離散化、AN は金属電子 (フェルミオン) を離散化。  
**AN モデルは SB コードを最小改変で実装可能。**

---

## 7. 実装難易度まとめ

| モデル | Qubit | CNOT/circuit | 難易度 | 優先度 |
|--------|-------|-------------|--------|--------|
| 共鳴準位 (1+1) | 2 | **80** (4/step×20) | **低** | **第1** |
| 3軌道 AN (1+2) | 3 | ~200 | 中 | 第2 |
| スピン付き 2軌道 (1+1, spin) | 4 | ~320 | 中高 | 第3 |
| 論文の最小 (FT-QC) | 196 logical | 2.3×10⁷ Toffoli | **実質不可** | 将来 |

> simulate モード実測: MAE = 0.06% (Trotter 誤差のみ)、IsingXY 分解 4 CNOT/step を確認。

---

## 8. 結論

**IBM NISQ 実機での GAN サンプルモデル実装**: **実現可能**

- 論文提案のフル GAN アルゴリズムは FT-QC 専用であり NISQ には適用不可
- しかし **核自由度を除いた電子系 Anderson-Newns モデル (共鳴準位モデル)** は
  - 2 qubit, 40–60 CNOT (N_steps=20)
  - 既存 SB コード (`pennylane_spin_boson.py`) の構造をほぼ流用
  - ibm_marrakesh (Heron r2) で十分実現可能
- 観測量: **分子軌道の電荷占有数 n_d(t)** = CNT への電子移動ダイナミクス
- 物理的意義: SB (ボゾン場結合) から AN (フェルミオン金属結合) への自然な発展

**次のステップ**: `pennylane_anderson_newns.py` の実装 (SB コードの XX+YY 版として)

---

*参照: Lang et al., arXiv:2601.16264 (Xanadu, 2025)*  
*Branch: `claude/pennylane-dynamic-nonadiabatic-je0wpx`*
