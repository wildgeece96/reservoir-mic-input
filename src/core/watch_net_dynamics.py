import os
import argparse
import json
import pyaudio
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from src.get_audio import get_stream
from src.net.reservoir import ESN_2D
from src.audio_process.process import AudioConverter
from src import utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-file-dir",
    type=str,
    default="./out/1211-best-model",
    help=
    "The path to directory where saved model weights and audio configuration file are saved."
)
# TODO: separate these values as environemtal ones.
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000  # サンプリングレート
NUM_FRAME = 32
SHOW_STATE_DIMENSIONS = 10  # 内部状態をプロットするノードの数

if __name__ == '__main__':
    args = parser.parse_args()
    net = ESN_2D().load(args.config_file_dir)
    config_file_path = os.path.join(args.config_file_dir, "args.json")
    with open(config_file_path, "r") as f:
        config = json.load(f)
    chunk_size = config["chunk"]
    n_mels = config["n_mels"]
    audio_converter = AudioConverter(chunk_size, n_mels, RATE)
    stream = get_stream(format=FORMAT, rate=RATE, chunk_size=chunk_size)
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmax=RATE // 2)
    datas_mel = []
    cnt = 0

    audio_ax, picture, net_picture, preds_picture, state_graphs = utils.generate_realtime_plot(
        net,
        n_mels,
        num_frame=NUM_FRAME,
        mel_freqs=mel_freqs,
        state_dimensions=SHOW_STATE_DIMENSIONS,
        classes=["bass", "hi-hat", "snare"])
    net_state_record = np.zeros([net.height * net.width,
                                 NUM_FRAME])  # 最初にゼロ埋めされているネットワークのレコードを作成

    # 音声の取得 + プロットの開始
    while True:
        data = np.frombuffer(stream.read(chunk_size,
                                         exception_on_overflow=False),
                             dtype=np.int16)
        data_mel = audio_converter.convert_to_mel(data)
        datas_mel.append(data_mel.reshape(1, -1))

        net(data_mel.flatten(), return_preds=False)
        net_state_record = np.concatenate(
            [net_state_record[:, 1:],
             net.x_flatten.reshape(-1, 1)], axis=1)
        # 規定ステップごとに描画画像の更新を行う
        if cnt > NUM_FRAME and cnt % 2 == 0:
            datas_mel = datas_mel[-NUM_FRAME:]
            audio_ax.set_title(f"{cnt/RATE*chunk_size:.3f} (sec)")
            picture.set_data(np.concatenate(datas_mel, axis=0).T[::-1])
            net_picture.set_data(net.x)
            for node_idx, state_graph in enumerate(state_graphs):
                state_graph.set_data(np.arange(NUM_FRAME),
                                     net_state_record[node_idx, :])
            pred_probas = net.decoder.predict(net_state_record.T)
            preds_picture.set_data(pred_probas.T)
            plt.pause(0.001)
        cnt += 1
        print("cnt = ", cnt, end='\r')
