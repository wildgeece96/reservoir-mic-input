import warnings

warnings.simplefilter('ignore')

import os
import argparse
import logging
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
from src.net.reservoir import ESN
from src.utils import make_audio_dataset
from src.utils import load_audio_data
from src.utils import ESNDataGenerator

parser = argparse.ArgumentParser()
parser.add_argument("--dims",
                    type=str,
                    default="20,10,20",
                    help="The list of dimensions for Reservoir")
parser.add_argument("--leaky-rate",
                    type=float,
                    default=0.70,
                    help="The ratio of internal state in update step.")
parser.add_argument("--n-mels",
                    type=int,
                    default=32,
                    help="Number of dimension of mel spectrum.")
parser.add_argument("--input-offset",
                    type=float,
                    default=4.5,
                    help="The offset of the input to the network")
parser.add_argument("--input-scale",
                    type=float,
                    default=0.05,
                    help="The scale ratio of the input to Reservoir")
# parser.add_argument("--net-sparse-rate",
#                     type=float,
#                     default=0.7,
#                     help="The extent of sparsity of the network.")
parser.add_argument("--sample-rate",
                    type=int,
                    default=8000,
                    help="The sample rate of audio")
parser.add_argument(
    "--chunk",
    type=int,
    default=2**6,
    help="The size of chunk which equals to the window size at spectrograms.")
parser.add_argument("--n-fft",
                    type=int,
                    default=128,
                    help="The sample size to pass FFT function.")
parser.add_argument("--train-epochs",
                    type=int,
                    default=3,
                    help="The epoch size for training network.")
parser.add_argument(
    "--train-num-concat",
    type=int,
    default=5,
    help="The number of samples per input spectrogram during training.")
parser.add_argument("--pretrained-model",
                    type=str,
                    default=None,
                    help="The path to pretrained model and configs.")
parser.add_argument("--save-model",
                    action="store_true",
                    help="Whether to save trained model")
parser.add_argument("--save-model-path",
                    type=str,
                    default="../out/trained_model",
                    help="The directory path to save trained model")
parser.add_argument(
    "--score-path",
    type=str,
    default="../out/scores.json",
    help=
    "The json path which has maximum score during hyperparameter optimization."
)


def validate_model(model, input_state, label_seq):
    preds = model.predict(input_state)  # (n_samples, N_CLASSES)
    return np.where(preds == label_seq, 1, 0).mean()


def generate_states(network, dataloader):
    state = []
    label_seq = []
    x = np.random.rand(network.reservoir_dims[0]) - 0.5
    for (spectrogram, _label_seq) in dataloader:
        assert spectrogram.shape[1] == _label_seq.shape[
            0], "Size of spectrogram and label_seq are different %s vs %s" % (
                spectrogram.shape, _label_seq.shape)
        for i in range(spectrogram.shape[1]):
            x, _ = network(x, spectrogram[:, i])
            state.append(x.reshape(1, -1))
        # wash out
        for i in range(100):
            x, _ = network(x, np.zeros(N_MELS) - network.input_offset)
        label_seq.append(_label_seq)
    state = np.concatenate(state, axis=0)
    label_seq = np.concatenate(label_seq, axis=0)
    return (state, label_seq)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.pretrained_model:
        meta_json_path = os.path.join(args.pretrained_model, "sal_meta.json")
        with open(meta_json_path, "r") as f:
            prev_args = json.load(f)
        args.sample_rate = prev_args["sample_rate"]
        args.n_mels = prev_args["n_mels"]
        args.chunk = prev_args["chunk"]
        args.n_fft = prev_args["n_fft"]
        args.input_scale = prev_args["input_scale"]
        args.input_offset = prev_args["input_offset"]
        args.leaky_rate = prev_args["leaky_rate"]
        args.dims = prev_args["dims"]
        print(
            f"Loaded pretraining configuration\n {json.dumps(prev_args, indent=4)}"
        )
    if args.save_model_path:
        os.makedirs(args.save_model_path, exist_ok=True)
    CHANNELS = 1
    CHUNK = args.chunk
    RATE = args.sample_rate  # サンプリングレート
    N_MELS = args.n_mels
    OVERLAP_RATE = 0.0

    RAW_CLASSES = {
        "bass": 0,
        "hi-hat": 1,
        "snare": 2,
        "k-snare": 2,
        "silent": 3
    }
    CLASSES = {"bass": 0, "hi-hat": 1, "snare": 2, "silent": 3}

    N_CLASSES = len(CLASSES)

    network_config = {
        "reservoir_dims": list(map(int, args.dims.split(","))),
        "input_dim": args.n_mels,
        "output_dim": N_CLASSES,
        "leaky_rate": args.leaky_rate,  # 直前の state をどれだけ残すか
        "input_offset": args.input_offset,
        "input_scale": args.input_scale
    }

    converter = AudioConverter(chunk_size=CHUNK,
                               n_fft=args.n_fft,
                               n_mels=N_MELS,
                               sample_rate=RATE)
    if args.pretrained_model:
        network = ESN.load(args.pretrained_model)
    else:
        network = ESN(**network_config)

    # 訓練データの準備
    np.random.seed(31)
    datas = load_audio_data(
        data_dir="./data/cripped_wav",
        valid_ratio=0.2,
        classes=RAW_CLASSES,
        save_file_dir=args.save_model_path,
        sample_rate=args.sample_rate,
        file_list_path=os.path.join(args.pretrained_model, "file_list.json")
        if args.pretrained_model else None)

    train_dataset = make_audio_dataset(datas["train"],
                                       converter,
                                       N_CLASSES,
                                       data_mapping=RAW_CLASSES)
    valid_dataset = make_audio_dataset(datas["valid"],
                                       converter,
                                       N_CLASSES,
                                       data_mapping=RAW_CLASSES)

    train_dataloader = ESNDataGenerator(train_dataset,
                                        epochs=args.train_epochs,
                                        num_concat=args.train_num_concat,
                                        class_weights=[5.0, 5.0, 5.0, 1.0])
    valid_dataloader = ESNDataGenerator(valid_dataset, epochs=3, num_concat=5)

    train_state, train_label_seq = generate_states(network, train_dataloader)
    valid_state, valid_label_seq = generate_states(network, valid_dataloader)
    print("Size of training data: %d" % train_state.shape[0])

    # decoder の学習
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
    print("args:\t %s" % json.dumps(vars(args), indent=4))

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
        args_dict = vars(args)
        args_dict["classes"] = CLASSES
        with open(os.path.join(args.save_model_path, "args.json"), "w") as f:
            json_strings = json.dumps(vars(args), indent=4)
            f.write(json_strings)
