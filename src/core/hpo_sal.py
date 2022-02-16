import itertools
import os
import subprocess
from datetime import datetime

if __name__ == '__main__':
    dimss = ["100,100", "30,30,30", "100,100,100", "30,30"]
    n_mels = ["32", "64", "128"]
    chunks = ["1024"]
    n_ffts = ["64", "128", "512"]
    leaky_rates = ["0.3", "0.5", "0.9"]
    input_offsets = ["5.0", "9.0"]
    input_scales = ["0.01", "0.1", "1.0"]
    sample_rates = ["22050", "8000"]

    date = datetime.now().strftime("%Y%m%d-%H%M")
    os.makedirs(f"../out/trained_model/{date}", exist_ok=True)
    cnt = 0
    for (dims, n_mel, chunk, n_fft, leaky_rate, input_offset, input_scale,
         sample_rate) in itertools.product(dimss, n_mels, chunks, n_ffts,
                                           leaky_rates, input_offsets,
                                           input_scales, sample_rates):
        subprocess.call([
            "python", "src/train/train_integ_sal.py", "--dims", dims,
            "--leaky-rate", leaky_rate, "--n-mels", n_mel, "--input-offset",
            input_offset, "--input-scale", input_scale, "--sample-rate",
            sample_rate, "--chunk", chunk, "--n-fft", n_fft, "--train-sal",
            "--num-concat", "5", "--save-model", "--save-model-path",
            f"../out/trained_model/{date}", "--score-path",
            f"../out/trained_model/{date}/scores.json", "--scores-path",
            f"../out/trained_model/{date}/score_records.csv"
        ])
        cnt += 1
        print(f"Optimization step = {cnt}")
