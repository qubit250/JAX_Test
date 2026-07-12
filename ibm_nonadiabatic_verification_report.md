# IBM 実機検証レポート: 小規模動的非断熱計算 (Landau-Zener モデル)

**作成日**: 2026-07-12  
**リポジトリ**: `qubit250/JAX_Test`  
**ブランチ**: `claude/pennylane-dynamic-nonadiabatic-je0wpx`

---

## 1. 概要

PennyLane + Catalyst を用いた小規模動的非断熱計算を IBM 量子実機で検証した。
2準位 Landau-Zener モデルの時間発展を1量子ビット回路で実装し、
断熱/非断熱遷移を IBM ibm_kingston 実機上で観測することに成功した。

### 計算系

$$H(t) = \alpha t \cdot Z + \Delta \cdot X \quad (t \in [-T/2, +T/2])$$

| パラメータ | 値 |
|---|---|
| ギャップ $\Delta$ | 1.0 (H.u.) |
| 掃引時間 $T$ | 6.0 |
| 掃引速度 $\alpha$ | 0.1, 0.5, 1.0, 2.0, 5.0 |

---

## 2. 実装

### 2.1 ゲート分解

時間発展演算子を IBM ネイティブゲート (RY, RZ) に厳密分解:

$$\exp(-i(aZ + \Delta X)\,dt) = RY(-\phi) \cdot RZ(2\|H\|dt) \cdot RY(+\phi)$$

$$\phi = \arctan2(\Delta,\, a), \quad \|H\| = \sqrt{a^2 + \Delta^2}$$

> **注意**: ゲート適用順序は `RY(-φ) → RZ → RY(+φ)` (逆順で符号エラーが発生)

### 2.2 初期状態

$H(t=-T/2)$ の基底状態を RY ゲートで準備:

$$\theta_{\rm init} = 2\arctan2\!\left(\sqrt{\frac{\|H_0\|+a_0}{2\|H_0\|}},\; \frac{-\Delta}{\sqrt{2\|H_0\|(\|H_0\|+a_0)}}\right)$$

### 2.3 実行モード

| モード | コマンド | 特徴 |
|---|---|---|
| `catalyst` | `python script.py catalyst` | `@qjit` + `for_loop` + `jax.vmap` / N_STEPS=100 |
| `ibm_sim` | `python script.py ibm_sim` | AerSimulator (gate_err=0.1%, readout_err=1%) / N_STEPS=15 |
| `ibm_real` | `python script.py ibm_real <TOKEN>` | IBM 実機 / N_STEPS=15 |

### 2.4 回路仕様 (IBM 実機向け)

| 項目 | 値 |
|---|---|
| 量子ビット数 | 1 |
| タイムステップ数 (N_STEPS) | 15 |
| 総ゲート数 | 46 |
| 回路深度 | 46 |
| RY ゲート数 | 31 (各 70 ns = 2×SX) |
| RZ ゲート数 | 15 (仮想ゲート, 0 ns) |
| 読み出し時間 | ~500 ns |
| 回路実行時間 (1回) | ~2.67 μs |

---

## 3. QPU 使用時間見積もり

| モード | 1 shot | 総 QPU 時間 | 無料枠消費 |
|---|---|---|---|
| デフォルト (rep_delay=250 μs) | 252.7 μs | **10.35 s** | 1.72% / 月 |
| Active Reset (rep_delay=1 μs) | 3.67 μs | 0.15 s | 0.025% / 月 |

- **shots=8192**, alpha 値 5 種, 統計誤差 ±0.55% (1σ)
- IBM Quantum 無料枠 (10分/月) で **月 57 回**実行可能
- キュー待ち時間は含まず (実測: 約 5 分)

---

## 4. 実機実行結果

**使用バックエンド**: `ibm_kingston`  
**実行日時**: 2026-07-12 17:33 UTC  
**shots**: 8,192

```
======================================================
  IBM実機検証: 小規模動的非断熱計算 (Landau-Zener)
  H(t) = alpha*t*Z + 1.0*X,  T = 6.0
======================================================

[Mode: IBM 実機 (shots=8192)]
  ゲート数: 46,  回路深度: 46
  -> 使用バックエンド: ibm_kingston

 alpha    P(|0>)    P(|1>)  解釈
--------------------------------------------
   0.1    0.3365    0.6635  遷移域
   0.5    0.0886    0.9114  断熱 ← |0> に追従
   1.0    0.1373    0.8627  断熱 ← |0> に追従
   2.0    0.1991    0.8009  断熱 ← |0> に追従
   5.0    0.6592    0.3408  非断熱 → |0> に留まる

  ステップ数 N_STEPS = 15  (dt = 0.400,  T = 6.0)
```

---

## 5. 3モード比較

| alpha | Catalyst (理論) | AerSimulator | **IBM Real** | \|Real−Aer\| | 偏差 |
|---|---|---|---|---|---|
| 0.1 | 0.3471 | 0.3239 | **0.3365** | 0.0126 | 2.3σ ✓ |
| 0.5 | 0.0936 | 0.0854 | **0.0886** | 0.0032 | 0.6σ ✓ |
| 1.0 | 0.0790 | 0.1350 | **0.1373** | 0.0023 | 0.4σ ✓ |
| 2.0 | 0.2698 | 0.1962 | **0.1991** | 0.0029 | 0.5σ ✓ |
| 5.0 | 0.5393 | 0.6536 | **0.6592** | 0.0056 | 1.0σ ✓ |

- AerSim との **平均偏差: 0.53%**、最大偏差: 1.26% (alpha=0.1)
- 全点で 3σ 以内 → AerSim ノイズモデルが実機を精度よく近似

---

## 6. 物理的考察

### 6.1 断熱/非断熱遷移の観測

| | 理論的予測 | IBM 実機 |
|---|---|---|
| 掃引が遅い (alpha=0.5) | 基底状態追従 → P(\|0⟩) 小 | 0.089 ✓ |
| 掃引が速い (alpha=5.0) | diabatic 残留 → P(\|0⟩) 大 | 0.659 ✓ |
| コントラスト (差) | — | **0.570** |

断熱/非断熱の区別が 1 量子ビット 46 ゲートの浅い回路で明確に観測された。

### 6.2 Catalyst vs IBM Real の差異の原因

Catalyst (N_STEPS=100) と IBM Real (N_STEPS=15) の差は主に**トロッター誤差**（時間離散化の粗さ）によるものであり、ハードウェアノイズとは区別できる。特に alpha=2.0 での 7.1% の差は、N_STEPS=15 の粗い離散化で高速掃引の動力学が不正確に表現されているためである。

---

## 7. トラブルシューティング記録

| エラー | 原因 | 修正 |
|---|---|---|
| `ValueError: 'channel' can only be 'ibm_cloud', or 'ibm_quantum_platform'` | qiskit-ibm-runtime 0.20 以降でチャンネル名変更 | `channel="ibm_quantum"` → `"ibm_quantum_platform"` |
| RY*RZ*RY 分解の符号エラー | ゲート適用順序の誤り | `RY(φ) RZ RY(-φ)` → `RY(-φ) RZ RY(+φ)` に修正 |
| 初期状態 fidelity=0 | RY 角度の公式誤り | `arctan2(v₁, v₀)` の正しい定式化を使用 |

---

## 8. 結論

- PennyLane + Catalyst による動的非断熱計算の **IBM 実機での動作検証に成功**
- 回路深度 46、1 量子ビットで Landau-Zener 遷移を観測
- AerSimulator (ノイズモデル) と実機の平均偏差 **0.53%** — 事前シミュレーションが有効
- QPU 消費時間 **約 10 秒**（無料枠の 1.7%）で実行可能

---

## 9. ファイル一覧

| ファイル | 役割 |
|---|---|
| `pennylane_nonadiabatic_dynamics.py` | 基本版 (default.qubit + qml.exp, qml.Snapshot) |
| `pennylane_catalyst_ibm_nonadiabatic.py` | IBM 実機検証版 (Catalyst + RY/RZ 分解) |
| `requirements.txt` | 依存パッケージ一覧 |
| `ibm_nonadiabatic_verification_report.md` | 本レポート |

---

*計算環境: IBM ibm_kingston / PennyLane 0.45.0 / pennylane-catalyst 0.15.0 / qiskit-ibm-runtime 0.45.x*
