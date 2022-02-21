"""SAL(Sensitivity adjustment learning)を実装する場所. 
リザバー部分にカオス性を持たせる目的で事前学習的に行う"""
import argparse
import glob
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from src.audio_process import AudioConverter
from src.net import SALReservoir, export_sal_trained_reservoir
from src.utils import (ESNDataGenerator, SALConfigManager, load_audio_data,
                       make_audio_dataset)
from torch.autograd import Variable
from torch.optim import SGD


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims",
                        type=str,
                        default="20,20,20",
                        help="The list of dimensions for Reservoir")
    parser.add_argument(
        "--beta",
        type=float,
        default=0.99,
        help=
        "Rate of previous time step to calculate moving average sensitivity.")
    parser.add_argument("--leaky-rate",
                        type=float,
                        default=0.90,
                        help="The leaky rate of Reservoir.")
    parser.add_argument("--save-path",
                        type=str,
                        default="../out/sal_models",
                        help="The directory path to save trained SAL model")
    parser.add_argument("--sample-rate",
                        type=int,
                        default=8000,
                        help="The sample rate of audio")
    parser.add_argument("--n-mels",
                        type=int,
                        default=32,
                        help="The number of dimensions of mel-spectrogram.")
    parser.add_argument(
        "--chunk",
        type=int,
        default=2**6,
        help=
        "The size of chunk which equals to the window size at spectrograms.")
    parser.add_argument("--n-fft",
                        type=int,
                        default=128,
                        help="The sample size to pass FFT function.")
    parser.add_argument("--input-scale",
                        type=float,
                        default=0.05,
                        help="The scale ratio of the input to Reservoir")
    parser.add_argument("--input-offset",
                        type=float,
                        default=8.0,
                        help="The offset of the input")
    parser.add_argument(
        "--num-concat",
        type=int,
        default=5,
        help="The number of samples concatenated at one sample at training.")
    parser.add_argument("--configs-path",
                        type=str,
                        default="../out/sal/configs.csv")
    return parser.parse_args()


def check_chaoticity(res: SALReservoir,
                     dataloader: ESNDataGenerator,
                     w_in: np.array,
                     fig_dir: str = "../out/sal_models/fig/",
                     fig_prefix: str = "before",
                     mel_freqs: List[float] = [0.0, 1.0, 2.0],
                     classes: Dict = {"bass": 0},
                     input_offset: float = 4.0):
    """モデルのカオス性を摂動を加えた軌道と加えない軌道とを見比べることで確かめる

    Parameters
    ----------
    res : SALReservoir
        [description]
    dataloader : ESNDataGenerator
        [description]
    fig_dir : str, optional
        [description], by default "../out/sal_models/fig/"
    fig_prefix : str, optional
        [description], by default "before"
    mel_freqs : List[float], optional
        [description], by default [0.0, 1.0, 2.0]
    """
    os.makedirs(fig_dir, exist_ok=True)
    cnt = 0
    aspect_ratio = 0.20
    fontsize = 20
    for (spectrogram, label_seq) in dataloader:
        cnt += 1
        if cnt > 5:
            break
        x = torch.Tensor(np.random.rand(res.dims[0]).astype(np.float32) - 0.5)
        x02 = x.clone()
        s_bar = 0
        state_list = []
        state = res.generate_init_state()
        for i in range(spectrogram.shape[1]):
            x = x + torch.Tensor(
                np.dot(spectrogram[:, i] - input_offset, w_in).flatten())
            x, s_bar, state = res(x, state, prev_s=s_bar)
            state_list.append(state)

        s_bar = 0
        state_list02 = []
        state = res.generate_init_state()
        x = x02 + np.random.randn(res.dims[0]).astype(np.float32) * 1e-8
        for i in range(spectrogram.shape[1]):
            x = x + torch.Tensor(
                np.dot(spectrogram[:, i] - input_offset, w_in).flatten())
            x, s_bar, state = res(x, state, prev_s=s_bar)
            state_list02.append(state)
        state01 = np.concatenate(
            [state[0].detach().numpy().reshape(-1, 1) for state in state_list],
            axis=1)
        state02 = np.concatenate([
            state[0].detach().numpy().reshape(-1, 1) for state in state_list02
        ],
                                 axis=1)

        diff = np.sqrt(np.mean(np.abs(state01 - state02)**2, axis=0))
        fig = plt.figure(figsize=(20, 20))

        log_diff = np.log10(diff + 1e-15)
        diff_ax = fig.add_subplot(4, 1, 1)
        diff_ax.set_title("Difference between two states (baseline = -15.0)",
                          fontsize=fontsize)
        diff_ax.plot(log_diff)
        diff_ax.grid()
        diff_ax.set_aspect(spectrogram.shape[1] /
                           (log_diff.max() - log_diff.min()) * aspect_ratio)

        state_seq_ax = fig.add_subplot(4, 1, 2)
        state_seq_ax.set_title("The same element of state sequences",
                               fontsize=fontsize)
        state_seq_ax.plot(state01[0], alpha=0.5, label="not perturbated")
        state_seq_ax.plot(state02[0], alpha=0.5, label="perturbated")
        state_seq_ax.set_ylim([-1.0, 1.0])
        state_seq_ax.legend(fontsize=fontsize)
        state_seq_ax.grid()
        state_seq_ax.set_aspect(spectrogram.shape[1] / 2.0 * aspect_ratio)

        spectrogram_ax = fig.add_subplot(4, 1, 3)
        spectrogram_ax.imshow(spectrogram[::-1])
        spectrogram_ax.set_yticks(np.arange(0, len(mel_freqs), 20))
        spectrogram_ax.set_yticklabels([f"{int(f)}" for f in mel_freqs[::-20]])
        spectrogram_ax.set_ylabel("Frequency (Hz)", fontsize=fontsize)
        spectrogram_ax.set_aspect(spectrogram.shape[1] / spectrogram.shape[0] *
                                  aspect_ratio)

        label_seq_ax = fig.add_subplot(4, 1, 4)
        label_one_hot = np.identity(len(classes))[label_seq.astype(
            np.int)]  # (n_frame, n_class)
        label_seq_ax.imshow(label_one_hot.T)
        label_seq_ax.set_yticks(list(range(len(classes))),
                                list(classes.keys()),
                                fontsize=fontsize)
        label_seq_ax.set_title("Label sequence", fontsize=fontsize)
        label_seq_ax.set_aspect(label_one_hot.shape[0] / len(classes) *
                                aspect_ratio)

        # スペクトログラムと他のグラフの横幅を合わせる
        fig.canvas.draw()
        diff_axpos = diff_ax.get_position()
        state_seq_axpos = state_seq_ax.get_position()
        spectrogram_axpos = spectrogram_ax.get_position()
        label_axpos = label_seq_ax.get_position()
        label_seq_ax.set_position([
            state_seq_axpos.x0,
            label_axpos.y0,
            spectrogram_axpos.width,
            spectrogram_axpos.height,
        ])

        spectrogram_ax.set_position([
            state_seq_axpos.x0, spectrogram_axpos.y0, spectrogram_axpos.width,
            spectrogram_axpos.height
        ])

        diff_ax.set_position([
            state_seq_axpos.x0, diff_axpos.y0, spectrogram_axpos.width,
            spectrogram_axpos.height
        ])
        state_seq_ax.set_position([
            state_seq_axpos.x0, state_seq_axpos.y0, spectrogram_axpos.width,
            spectrogram_axpos.height
        ])

        plt.savefig(os.path.join(fig_dir, fig_prefix + f"{cnt:02d}.png"))
        plt.close()

        cnt += 1


RAW_CLASSES = {"bass": 0, "hi-hat": 1, "snare": 2, "k-snare": 2, "silent": 3}
CLASSES = {"bass": 0, "hi-hat": 1, "snare": 2, "silent": 3}
N_CLASSES = len(CLASSES)
WEIGHTS = [
    [0.2, 1.0, 0.2, 0.05],  # bass から続く音の割合
    [1.0, 0.2, 1.0, 0.05],  # hi-hat から続く音の割合
    [0.2, 1.0, 0.1, 0.05],  # snare から続く音の割合
    [1.0, 0.2, 0.2, 0.0]  # silent から続く音の割合
]


def main(args):
    os.makedirs(args.save_path, exist_ok=True)
    dims = list(map(int, args.dims.split(",")))
    config_manager = SALConfigManager(args.configs_path)
    # 学習データの準備(音声の読み込み)
    converter = AudioConverter(chunk_size=args.chunk,
                               n_fft=args.n_fft,
                               n_mels=args.n_mels,
                               sample_rate=args.sample_rate)
    np.random.seed(31)
    datas = load_audio_data(data_dir="./data/cripped_wav",
                            valid_ratio=0.2,
                            save_file_dir=args.save_path,
                            classes=RAW_CLASSES)
    train_dataset = make_audio_dataset(datas["train"],
                                       converter,
                                       N_CLASSES,
                                       data_mapping=RAW_CLASSES)
    valid_dataset = make_audio_dataset(datas["valid"],
                                       converter,
                                       N_CLASSES,
                                       data_mapping=RAW_CLASSES)

    train_dataloader = ESNDataGenerator(train_dataset,
                                        epochs=5,
                                        num_concat=args.num_concat,
                                        class_weights=WEIGHTS)
    valid_dataloader = ESNDataGenerator(valid_dataset, epochs=1, num_concat=1)

    # リザバーネットワークの用意
    network_config = {
        "dims": dims,
        "beta": args.beta,
        "leaky_rate": args.leaky_rate
    }
    res = SALReservoir(**network_config)

    w_in = (np.random.rand(args.n_mels, dims[0]) -
            0.5) * args.input_scale * 2.0
    # 学習前の状態での内部状態のプロット
    print("Visualising chaoticity before training....")
    check_chaoticity(res,
                     train_dataloader,
                     w_in,
                     fig_dir=os.path.join(args.save_path, "fig"),
                     fig_prefix="before",
                     mel_freqs=converter.mel_freqs,
                     classes=CLASSES,
                     input_offset=args.input_offset)

    # SAL の学習 (1.0 行く直前まで)
    ## optimizer の準備
    print("Start SAL training....")
    parameters = []
    for layer in res.layers:
        parameters += layer.parameters()
    optimizer = SGD(parameters, lr=0.1, weight_decay=0.001)
    num_epochs = 1000
    state_list = []
    for epoch in range(num_epochs):
        epoch_s_list = []
        for (spectrogram, label_seq) in train_dataloader:
            optimizer.zero_grad()
            x = Variable(torch.Tensor(np.random.rand(res.dims[0]) - 0.5))
            s_bar = 0
            state = res.generate_init_state()
            for i in range(spectrogram.shape[1]):
                step_input = Variable(
                    torch.Tensor(spectrogram[:, i] - args.input_offset))
                x = x + torch.Tensor(np.dot(step_input, w_in).flatten())
                x, s_bar, state = res(x, state, prev_s=s_bar)
                state_list.append(state)
            epoch_s_list.append(s_bar.to('cpu').detach().numpy().copy())
            loss = -s_bar
            loss.backward(retain_graph=True)
            optimizer.step()
        print(np.mean(epoch_s_list))
        if np.mean(epoch_s_list) > 0.99:
            break
    print("Training done!!")

    # 学習後の状態での内部状態のプロット
    check_chaoticity(res,
                     train_dataloader,
                     w_in,
                     fig_dir=os.path.join(args.save_path, "fig"),
                     fig_prefix="after",
                     mel_freqs=converter.mel_freqs,
                     classes=CLASSES,
                     input_offset=args.input_offset)

    # 学習させたリザバーの export
    print("Saving trained reservoir...")
    config = {
        "output_dim": len(CLASSES),
        "input_scale": args.input_scale,
        "input_offset": args.input_offset,
        "dtype": np.float32
    }
    export_sal_trained_reservoir(res,
                                 w_in,
                                 config=config,
                                 dir_path=args.save_path)
    with open(os.path.join(args.save_path, "sal_meta.json"), "w") as f:
        json_string = json.dumps(vars(args), indent=4)
        f.write(json_string)
    config_manager.add_config(vars(args))


if __name__ == "__main__":
    args = get_args()
    main(args)
