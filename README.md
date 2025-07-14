# MNIST-Reverser
csc.t272 プロジェクト課題

## プログラムの使用方法
### 環境構築
> 実行環境\
> WSL Ubuntu 24.04.1\
> python 3.12.3\
> pip 24.0
``` bash
git clone https://github.com/dev-lethe/MNIST-Reverser.git
cd MNIST-Reverser
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### プログラムの実行
``` bash
python3 main.py {options}
```
> [!TIP]
> オプションについて\
> `--load_model` : すでに作成したモデルを使用する\
> `--use_cnn` : CNN モデルを使用する\
> `--data_dir {./data}` : データを保存するディレクトリを指定\
> `--lr {0.002}` : 学習時の学習率\
> `--epochs {7}` : 学習時のエポック数\ 
> `--bs {64}` : バッチサイズ\
> `--layer_dim {1024}` : NN モデルのレイヤーの次元\
> `--channel {32}` : CNN モデルのチャンネルサイズ\
> `--dim {512}` : CNN モデルの隠れ層の次元\
> `--target {0}` : 生成画像の目標の数字\
> `--gen_lr {0.0005}` : 生成時の学習率\
> `--gen_lw {1}` : 生成時のCross Entropy Loss の重み\
> `--gen_tvw {3e-4}` : 生成時のTotal Varidation Loss の重み\
> `--gen_biw {3e-4}` : 生成時のBinarization Loss の重み\
> `--gen_epochs {10000}` : 生成時のエポック数\
> `--use_conf` : 生成画像を識別モデルに通す
