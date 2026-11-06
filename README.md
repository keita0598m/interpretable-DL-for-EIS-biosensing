# EIS バイオセンシングデータ解析のための解釈可能な深層学習

電気化学インピーダンス法（EIS）のデータを対象に、深層学習で次の2つを自動化する研究プロジェクトです。

1. 等価回路の分類：データがどの等価回路モデルに従うかを判定する
2. 回路パラメータの推定：抵抗・キャパシタなどの回路定数を推定する

どちらも、変分オートエンコーダ（VAE）の潜在空間を使い、判断の根拠を人が確認できる形にすることを目指しています。

キーワード：電気化学インピーダンス法（EIS）、変分オートエンコーダ（VAE）、CNN、Physics-Informed Machine Learning

> 研究の詳しい内容と結果は、学会発表のあとに公開する予定です。

## リポジトリ構成

```
.
├── README.md
├── circuit_classification/    # 1. 等価回路分類（VAE + 分類器）
└── parameter_estimation/      # 2. 回路パラメータ推定（1D CNN VAE + PIML）
```

## 実行方法

それぞれ独立した Python 環境として実行できます。

```bash
# 等価回路分類
cd circuit_classification
pip install -r requirements.txt
python main.py

# 回路パラメータ推定
cd ../parameter_estimation
pip install -r requirements.txt
python main.py
```

オプションは各フォルダの README を参照してください。
