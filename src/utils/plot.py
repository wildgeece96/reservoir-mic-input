import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List

from src.net.reservoir import ESN_2D


# プロットの準備
def generate_realtime_plot(net: ESN_2D, n_mels: int, num_frame: int,
                           mel_freqs: List[float], state_dimensions: int,
                           classes: List[str]):
    """音声の入力とネットワークの出力をリアルタイムで可視化するための前準備を行う関数"""
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(nrows=3, ncols=2, figure=fig)

    ## 音声の可視化部分の初期設定
    audio_ax = fig.add_subplot(gs[0, 0])
    zero_picture = np.zeros([n_mels, num_frame])
    zero_picture[:, 0] = 0.0
    zero_picture[:, 1] = -3.0
    picture = audio_ax.imshow(zero_picture)
    fig.colorbar(picture, ax=audio_ax, shrink=0.8)
    audio_ax.set_yticks(np.arange(0, n_mels, 20))
    audio_ax.set_yticklabels([f"{int(f)}" for f in mel_freqs[::-20]])
    audio_ax.set_ylabel("Frequency (Hz)")
    audio_ax.set_aspect(num_frame / n_mels * 0.50)

    ## ネットワークの予測結果を可視化するための初期設定
    preds_ax = fig.add_subplot(gs[1, 0])
    zero_preds = np.zeros([len(classes), num_frame])
    zero_preds[:, 1] = 3.  # 0 ~ 1 の範囲で値がプロットされる想定なので
    zero_preds[:, 2] = -3.
    preds_picture = preds_ax.imshow(zero_preds)
    preds_ax.set_yticks(list(range(len(classes))), classes)
    preds_ax.set_title("Predicted Type of audio")
    preds_ax.set_aspect(num_frame / len(classes) * 0.50)
    # https://qiita.com/sbajsbf/items/b3ce138de83362bc45b0 を参照した
    # スペクトログラムと推論シーケンスの画像サイズを揃える
    fig.canvas.draw()
    axpos1 = audio_ax.get_position()  # 上の図の描画領域
    axpos2 = preds_ax.get_position()  # 下の図の描画領域
    #幅をax1と同じにする
    preds_ax.set_position([axpos2.x0, axpos2.y0, axpos1.width, axpos2.height])

    ## ネットワーク活性化状況可視化部分の初期設定
    net_ax = fig.add_subplot(gs[:2, 1])
    zero_net = np.zeros([net.height, net.width])
    zero_net[:, 0] = 1.0
    zero_net[:, 1] = -1.0
    net_picture = net_ax.imshow(zero_net)
    fig.colorbar(net_picture, ax=net_ax)
    net_ax.set_aspect(1.0)

    ## ネットワークのニューロンの状態遷移可視化部分の初期設定
    state_ax = fig.add_subplot(gs[2, :2])
    zero_state = np.zeros([net.height * net.width, num_frame])
    state_graphs = []
    for node_idx in range(state_dimensions):
        state_graph, = state_ax.plot(zero_state[node_idx, :])
        state_graphs.append(state_graph)
    net_state_record = zero_state
    state_ax.set_ylim([-1.0, 1.0])
    fig.tight_layout()
    return (audio_ax, picture, net_picture, preds_picture, state_graphs)