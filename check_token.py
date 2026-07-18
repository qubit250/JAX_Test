"""IBM Quantum トークン検証スクリプト (回路実行なし)

使い方:
    python check_token.py <TOKEN>

確認内容:
    1. QiskitRuntimeService への接続
    2. 利用可能バックエンド一覧
    3. JT モデル (2 量子ビット) 向け推奨バックエンド
    4. キュー待ち状況
"""

import sys


def check_token(token: str):
    print("=" * 52)
    print("  IBM Quantum トークン検証")
    print("=" * 52)
    print(f"  トークン長:    {len(token)} 文字")
    print(f"  先頭 20 文字: {token[:20]}...")

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        print("\n  ERROR: qiskit-ibm-runtime が未インストール")
        print("  実行: pip install qiskit-ibm-runtime")
        return

    # ── 1. 接続テスト ──────────────────────────────
    print("\n[1] サービス接続テスト")
    try:
        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=token,
        )
        print("  ✓ 接続成功")
    except Exception as e:
        print(f"  ✗ 接続失敗: {e}")
        return

    # ── 2. インスタンス情報 ────────────────────────
    print("\n[2] インスタンス")
    try:
        instances = service.instances()
        for inst in instances:
            print(f"  - {inst}")
    except Exception as e:
        print(f"  △ 取得失敗: {e}")

    # ── 3. 利用可能バックエンド一覧 ────────────────
    print("\n[3] 利用可能バックエンド (実機のみ)")
    try:
        backends = service.backends(operational=True, simulator=False)
        print(f"  合計: {len(backends)} 件\n")
        print(f"  {'バックエンド名':<22}  {'Qubits':>6}  {'キュー':>6}  {'状態'}")
        print("  " + "-" * 48)
        for b in sorted(backends, key=lambda x: x.num_qubits):
            try:
                st = b.status()
                pending = st.pending_jobs
                status  = "稼働中" if st.operational else "停止中"
            except Exception:
                pending, status = "?", "不明"
            print(f"  {b.name:<22}  {b.num_qubits:>6}  {pending:>6}  {status}")
    except Exception as e:
        print(f"  △ 取得失敗: {e}")

    # ── 4. JT モデル推奨バックエンド ──────────────
    print("\n[4] JT モデル (min 2 qubits) 推奨バックエンド")
    try:
        best = service.least_busy(
            operational=True, simulator=False, min_num_qubits=2
        )
        st = best.status()
        print(f"  ✓ 推奨: {best.name}")
        print(f"    量子ビット数: {best.num_qubits}")
        print(f"    キュー待ち:   {st.pending_jobs} ジョブ")
        print(f"\n  実行コマンド:")
        print(f"  python pennylane_minimal_jt.py ibm_real {token[:12]}...")
    except Exception as e:
        print(f"  △ 推奨バックエンド取得失敗: {e}")

    print("\n  → トークン有効。ibm_real モードで実行可能です。")
    print("=" * 52)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python check_token.py <TOKEN>")
        print("例:     python check_token.py eyJhbGciOi...")
        sys.exit(1)
    check_token(sys.argv[1])
