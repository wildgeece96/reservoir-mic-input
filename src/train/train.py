import os
import argparse
from typing import Tuple, List
from collections import defaultdict
import json
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

import librosa

from src.audio_process.process import AudioConverter
from src.net.reservoir import ESN_2D
from src.utils import make_train_dataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--net-height",
    type=int,
    default=20,
    help="The number of node in height dimension of the network.")
parser.add_argument(
    "--net-width",
    type=int,
    default=20,
    help="The number of node in width dimension of the network.")
parser.add_argument("--net-alpha",
                    type=float,
                    default=0.70,
                    help="The ratio of internal state in update step.")
parser.add_argument("--n-mels",
                    type=int,
                    default=32,
                    help="Number of dimension of mel spectrum.")
parser.add_argument(
    "--chunk",
    type=int,
    default=2**6,
    help="The size of chunk which equals the window size in a spectrogram.")
parser.add_argument("--ridge-alpha",
                    type=float,
                    default=1.0,
                    help="The reguralization strength of Ridge regression.")
parser.add_argument("--save-model",
                    action="store_true",
                    help="Whether to save trained model")
parser.add_argument("--save-model-path",
                    type=str,
                    default="./out/trained_model",
                    help="The directory path to save trained model")
parser.add_argument(
    "--score-path",
    type=str,
    default="./out/scores.json",
    help=
    "The json path which has maximum score during hyperparameter optimization."
)
args = parser.parse_args()

CHANNELS = 1
CHUNK = 2**6
RATE = 8000  # サンプリングレート
N_MELS = args.n_mels
OVERLAP_RATE = 0.0

CLASSES = {"bass": 0, "hi-hat": 1, "snare": 2, "k-snare": 2}
N_CLASSES = 3

network_config = {
    "height": args.net_height,
    "width": args.net_width,
    "input_dim": N_MELS,
    "output_dim": N_CLASSES,
    "alpha": args.net_alpha,  # 直前の state をどれだけ残すか
}


def validate_model(model, input_state, label_seq):
    proba = model.predict(input_state)  # (n_samples, N_CLASSES)
    preds = np.argmax(proba, axis=1)
    label_seq = np.argmax(label_seq, axis=1)
    return np.where(preds == label_seq, 1, 0).mean()


if __name__ == "__main__":
    audio_paths = glob.glob("./data/cripped_wav/*.wav")
    audio_data = []
    for path in audio_paths:
        audio, sr = librosa.load(path, sr=RATE)
        audio_type = path.split("/")[-1].split("_")[0]
        audio_data.append((audio, audio_type))

    converter = AudioConverter(CHUNK, N_MELS, RATE)

    network = ESN_2D(**network_config)

    # 訓練データの準備
    valid_ratio = 0.2
    num_data = len(audio_data)
    np.random.seed(31)
    shuffled_idx = np.random.permutation(num_data)

    num_valid = int(num_data * valid_ratio)
    valid_idxes = shuffled_idx[:num_valid]
    train_idxes = shuffled_idx[num_valid:]

    train_input_spectrogram, train_label_seq = make_train_dataset(
        [audio_data[idx] for idx in train_idxes],
        converter,
        N_CLASSES,
        data_mapping=CLASSES)
    valid_input_spectrogram, valid_label_seq = make_train_dataset(
        [audio_data[idx] for idx in valid_idxes],
        converter,
        N_CLASSES,
        data_mapping=CLASSES)

    train_state = np.zeros(
        [train_input_spectrogram.shape[1], network.height * network.width])
    for idx in range(train_input_spectrogram.shape[1]):
        network(train_input_spectrogram[:, idx])
        train_state[idx, :] = network.x_flatten
        for i in range(10):
            network(np.zeros(N_MELS) - 10.)

    valid_state = np.zeros(
        [valid_input_spectrogram.shape[1], network.height * network.width])
    for idx in range(valid_input_spectrogram.shape[1]):
        network(valid_input_spectrogram[:, idx])
        valid_state[idx, :] = network.x_flatten
        for i in range(10):
            network(np.zeros(N_MELS) - 10.)

    regressor = Ridge(alpha=args.ridge_alpha, normalize=True)
    regressor.fit(train_state, train_label_seq.T)
    # train_score = regressor.score(train_state, train_label_seq.T)
    # valid_score = regressor.score(valid_state, valid_label_seq.T)
    train_score = validate_model(regressor, train_state, train_label_seq.T)
    valid_score = validate_model(regressor, valid_state, valid_label_seq.T)

    print("Score")
    print(f"\t train: {train_score:.2f}")
    print(f"\t valid: {valid_score:.2f}")
    if os.path.exists(args.score_path):
        with open(args.score_path, "r") as f:
            scores = json.load(f)
        prev_score = scores["valid_score"]
    else:
        prev_score = -0.1
    if args.save_model and prev_score < valid_score:
        network.save(args.save_model_path)
        with open(args.score_path, "w") as f:
            json_strings = json.dumps(
                {
                    "valid_score": valid_score,
                    "train_score": train_score
                },
                indent=4)
            f.write(json_strings)
        with open(os.path.join(args.save_model_path, "args.json"), "w") as f:
            json_strings = json.dumps(vars(args), indent=4)
            f.write(json_strings)
