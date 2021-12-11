import pyaudio
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from src.get_audio import get_stream
from src.net.reservoir import ESN_2D
from src.audio_process.process import AudioConverter

# TODO: separate these values as environemtal ones.
FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK = 2**12
RATE = 8000  # サンプリングレート
FRAME_NUM = 32
N_MELS = 80
NET_HEIGHT = 30
NET_WIDTH = 30
NET_ALPHA = 0.98
NET_INPUT_SCALE = 0.2

if __name__ == '__main__':
    net = ESN_2D(height=NET_HEIGHT,
                 width=NET_WIDTH,
                 input_dim=N_MELS,
                 alpha=NET_ALPHA,
                 input_scale=NET_INPUT_SCALE)
    audio_converter = AudioConverter(CHUNK, N_MELS, RATE)
    stream = get_stream()
    mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmax=RATE // 2)
    datas_mel = []
    cnt = 0

    # プロットの準備
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

    ## 音声の可視化部分の初期設定
    audio_ax = fig.add_subplot(gs[0, 0])
    zero_picture = np.zeros([N_MELS, FRAME_NUM])
    zero_picture[:, 0] = 1.0
    zero_picture[:, 1] = -3.0
    picture = audio_ax.imshow(zero_picture)
    fig.colorbar(picture, ax=audio_ax)
    audio_ax.set_yticks(np.arange(0, N_MELS, 20))
    audio_ax.set_yticklabels([f"{int(f)}" for f in mel_freqs[::-20]])
    audio_ax.set_ylabel("Frequency (Hz)")
    audio_ax.set_aspect(0.25)

    ## ネットワーク活性化状況可視化部分の初期設定
    net_ax = fig.add_subplot(gs[0, 1])
    zero_net = np.zeros([NET_HEIGHT, NET_WIDTH])
    zero_net[:, 0] = 1.0
    zero_net[:, 1] = -1.0
    net_picture = net_ax.imshow(zero_net)
    fig.colorbar(net_picture, ax=net_ax)
    net_ax.set_aspect(1.0)

    ## ネットワークのニューロンの状態遷移可視化部分の初期設定
    state_ax = fig.add_subplot(gs[1, :2])
    zero_state = np.zeros([NET_HEIGHT, FRAME_NUM])
    state_graphs = []
    for height_idx in range(NET_HEIGHT):
        state_graph, = state_ax.plot(zero_state[height_idx, :])
        state_graphs.append(state_graph)
    net_state_record = zero_state
    state_ax.set_ylim([-1.0, 1.0])

    # 音声の取得 + プロットの開始
    while True:
        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        data_mel = audio_converter.convert_to_mel(data)
        datas_mel.append(data_mel.reshape(1, -1))

        net.step(data_mel.flatten())
        net_state_record = np.concatenate(
            [net_state_record[:, 1:], net.x[:, 0].reshape(-1, 1)], axis=1)
        # 規定ステップごとに描画画像の更新を行う
        if cnt > FRAME_NUM and cnt % 3 == 0:
            datas_mel = datas_mel[-FRAME_NUM:]
            audio_ax.set_title(f"{cnt/RATE*CHUNK:.3f} (sec)")
            picture.set_data(np.concatenate(datas_mel, axis=0).T[::-1])
            net_picture.set_data(net.x)
            for height_idx, state_graph in enumerate(state_graphs):
                state_graph.set_data(np.arange(FRAME_NUM),
                                     net_state_record[height_idx, :])
            plt.pause(0.001)
        cnt += 1
        print("cnt = ", cnt, end='\r')
