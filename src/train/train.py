import os
import argparse
from typing import Tuple, List
from collections import defaultdict
import json
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression

import librosa

from src.audio_process import AudioConverter
from src.net.reservoir import ESN_2D
from src.utils import make_audio_dataset
from src.utils import ESNDataGenerator

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
parser.add_argument("--net-input-offset",
                    type=float,
                    default=4.5,
                    help="The offset of the input to the network")
parser.add_argument("--net-sparse-rate",
                    type=float,
                    default=0.7,
                    help="The extent of sparsity of the network.")
parser.add_argument(
    "--chunk",
    type=int,
    default=2**6,
    help="The size of chunk which equals the window size in a spectrogram.")
parser.add_argument("--n-fft",
                    type=int,
                    default=128,
                    help="The sample size to pass FFT function.")
parser.add_argument("--train-epochs",
                    type=int,
                    default=10,
                    help="The epoch size for training network.")
parser.add_argument(
    "--train-num-concat",
    type=int,
    default=5,
    help="The number of samples per input spectrogram during training.")
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
CHUNK = args.chunk
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
    "input_offset": args.net_input_offset,
    "sparse_rate": args.net_sparse_rate
}


def validate_model(model, input_state, label_seq):
    preds = model.predict(input_state)  # (n_samples, N_CLASSES)
    return np.where(preds == label_seq, 1, 0).mean()


def generate_states(network, dataloader):
    state = []
    label_seq = []
    for (spectrogram, _label_seq) in dataloader:
        assert spectrogram.shape[1] == _label_seq.shape[
            0], "Size of spectrogram and label_seq are different %s vs %s" % (
                spectrogram.shape, _label_seq.shape)
        for i in range(spectrogram.shape[1]):
            network(spectrogram[:, i])
            state.append(network.x_flatten.reshape(1, -1))
        # wash out
        for i in range(100):
            network(np.zeros(N_MELS) - network.input_offset)
        label_seq.append(_label_seq)
    state = np.concatenate(state, axis=0)
    label_seq = np.concatenate(label_seq, axis=0)
    return (state, label_seq)


if __name__ == "__main__":
    audio_paths = glob.glob("./data/cripped_wav/*.wav")
    audio_data = []
    for path in audio_paths:
        audio, sr = librosa.load(path, sr=RATE)
        audio_type = path.split("/")[-1].split("_")[0]
        audio_data.append((audio, audio_type))

    converter = AudioConverter(chunk_size=CHUNK,
                               n_fft=args.n_fft,
                               n_mels=N_MELS,
                               sample_rate=RATE)

    network = ESN_2D(**network_config)

    # 訓練データの準備
    valid_ratio = 0.2
    num_data = len(audio_data)
    np.random.seed(31)
    shuffled_idx = np.random.permutation(num_data)

    num_valid = int(num_data * valid_ratio)
    valid_idxes = shuffled_idx[:num_valid]
    train_idxes = shuffled_idx[num_valid:]

    train_dataset = make_audio_dataset(
        [audio_data[idx] for idx in train_idxes],
        converter,
        N_CLASSES,
        data_mapping=CLASSES)
    valid_dataset = make_audio_dataset(
        [audio_data[idx] for idx in valid_idxes],
        converter,
        N_CLASSES,
        data_mapping=CLASSES)

    train_dataloader = ESNDataGenerator(train_dataset,
                                        epochs=args.train_epochs,
                                        num_concat=args.train_num_concat)
    valid_dataloader = ESNDataGenerator(valid_dataset, epochs=1, num_concat=1)

    train_state, train_label_seq = generate_states(network, train_dataloader)
    valid_state, valid_label_seq = generate_states(network, valid_dataloader)

    decoder = LogisticRegression(penalty="l2", max_iter=1000)
    decoder.fit(train_state, train_label_seq.T)
    train_score = validate_model(decoder, train_state,
                                 train_label_seq.reshape(-1, 1))
    valid_score = validate_model(decoder, valid_state,
                                 valid_label_seq.reshape(-1, 1))
    network.set_decoder(decoder)

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
