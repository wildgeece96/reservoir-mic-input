import argparse
import json
import os

import librosa
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from src import utils
from src.audio_process.process import AudioConverter
from src.get_audio import get_stream
from src.net import ESN

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-file-dir",
    type=str,
    default="../out/1211-best-model",
    help=
    "The path to directory where saved model weights and audio configuration file are saved."
)
parser.add_argument(
    "--chunk-size",
    type=int,
    default=1024,
    help="The size of the chunk per iteration to process audio signal.")
# TODO: separate these values as environemtal ones.
FORMAT = pyaudio.paInt16
CHANNELS = 1
DRAW_STEP = 64
SHOW_STATE_DIMENSIONS = 12  # 内部状態をプロットするノードの数
CLASSES = {"bass": 0, "hi-hat": 1, "snare": 2, "silent": 3}


def softmax(x: np.ndarray):
    """ソフトマックス関数の実装
    x は (n_frame, n_classes) の形状を想定"""
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


if __name__ == '__main__':
    args = parser.parse_args()
    net = ESN.load(args.config_file_dir)
    config_file_path = os.path.join(args.config_file_dir, "args.json")
    with open(config_file_path, "r") as f:
        config = json.load(f)
    chunk_size = args.chunk_size
    n_mels = config["n_mels"]
    n_fft = config["n_fft"] if "n_fft" in config else config["chunk"]
    # サンプリングレート
    rate = config["sample_rate"] if "sample_rate" in config else 8000
    audio_converter = AudioConverter(chunk_size,
                                     n_fft,
                                     n_mels,
                                     sample_rate=rate)
    stream = get_stream(format=FORMAT, rate=rate, chunk_size=chunk_size)
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmax=rate // 2)
    datas_mel = []
    cnt = 0
    num_frame = DRAW_STEP * audio_converter.n_frame
    audio_ax, picture, net_picture, preds_picture, state_graphs = utils.generate_realtime_plot(
        net,
        n_mels,
        num_frame=num_frame,
        mel_freqs=mel_freqs,
        state_dimensions=SHOW_STATE_DIMENSIONS,
        classes=list(CLASSES.keys()))
    net_state_record = np.zeros([net.reservoir_dims[0],
                                 DRAW_STEP])  # 最初にゼロ埋めされているネットワークのレコードを作成

    # 音声の取得 + プロットの開始
    step_state = (np.random.rand(net.reservoir_dims[0]) - 0.5) / 10.0
    while True:
        data = np.frombuffer(stream.read(chunk_size,
                                         exception_on_overflow=False),
                             dtype=np.int16).astype(np.float32)
        data_mel = audio_converter.convert_to_mel(data)  # (n_mels, n_frame)
        datas_mel.append(data_mel)
        minibatch_state_record = []
        for i in range(audio_converter.n_frame):
            step_state, _ = net(
                step_state,
                data_mel[:, i],
                return_preds=False,
            )
            minibatch_state_record.append(step_state.reshape(-1, 1))
        concat_length = DRAW_STEP - len(minibatch_state_record)
        net_state_record = np.concatenate([net_state_record] +
                                          minibatch_state_record,
                                          axis=1)
        # 規定ステップごとに描画画像の更新を行う
        if cnt > DRAW_STEP and cnt % 2 == 0:
            datas_mel = datas_mel[-DRAW_STEP:]
            net_state_record = net_state_record[:, -num_frame:]
            audio_ax.set_title(f"{cnt/rate*chunk_size:.3f} (sec)")
            picture.set_data(np.concatenate(datas_mel,
                                            axis=1)[::-1])  # (n_frame, n_mels)
            net_picture.set_data(net.states[-1].reshape(5, -1))
            for node_idx, state_graph in enumerate(state_graphs):
                state_graph.set_data(np.arange(num_frame),
                                     net_state_record[node_idx, :])
            preds = net.decoder.predict(net_state_record.T).astype(int)
            preds_one_hot = np.identity(len(CLASSES.keys()))[preds]
            preds_picture.set_data(preds_one_hot.T)
            print("preds :", preds_one_hot[:5, :])
            print("spectrogram :", data_mel[:5, :5])
            plt.pause(0.001)
        cnt += 1
        print("cnt = ", cnt, end='\r')
