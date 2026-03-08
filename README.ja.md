# EMG ジェスチャ認識 実験プロジェクト

本プロジェクトは、表面筋電（EMG）をシリアル経由で取得し、4 クラスのジェスチャ分類を行う実験用プロトタイプです。  
Arduino からリアルタイムにデータを読み取り、時系列・周波数・小波特徴を抽出して、ランダムフォレストで学習とオンライン推論を行います。

## 主な機能

- EMG データのリアルタイムシリアル取得（既定: `COM6`, `115200`）
- 4 クラス（前 / 後 / 左 / 右）のサンプル収集とラベル生成
- 固定ウィンドウ分割（200 サンプル/窓）
- 複合特徴量抽出：
  - 時間領域: `RMS`, `MAV`, `WL`, `ZC`, `SSC`, `VAR`, `WA`, `MCV`
  - 周波数領域: `MPF`, `MF`, `SM2`, `ME`
  - パラメトリック: `AR(7)`
  - 時間周波数: `WPT`（Wavelet Packet）
- モデル学習と評価: `RandomForestClassifier`
- 無限ループでのオンライン推論出力

## ディレクトリ構成

- `main.py`: 取得・特徴量生成・学習・推論のメイン処理
- `feature_utils.py`: 特徴量計算ユーティリティ

## 実行環境

Python 3.9+ 推奨。主な依存関係：

- `pyserial`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `pywavelets`
- `statsmodels`
- `tqdm`

インストール例：

```bash
pip install pyserial numpy pandas scipy scikit-learn pywavelets statsmodels tqdm
```

## 使い方

1. EMG デバイスを接続し、シリアル設定を確認する（既定は `COM6`）。
2. 必要に応じて `main.py` のポート名を変更する（macOS/Linux では `/dev/tty.*` や `/dev/ttyUSB*` が一般的）。
3. 実行する：

```bash
python main.py
```

4. 端末表示に従って 4 種類の動作データを収集すると、学習後に推論結果が継続表示されます。

## 注意点

- 現状は研究・検証向けのプロトタイプで、処理はブロッキング中心です。
- `RandomForestClassifier(max_features='auto')` は scikit-learn のバージョンによって警告が出る場合があります。
- 今後は設定ファイル化、データ保存、例外処理、可視化、モデル永続化の追加を推奨します。
