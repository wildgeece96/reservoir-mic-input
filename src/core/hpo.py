import os
from datetime import datetime
import subprocess
import itertools

if __name__ == '__main__':
    net_heights = [3]
    net_widths = [3]
    net_alphas = [0.01, 0.1, 0.3]
    net_sparse_rates = [0.3, 0.7, 0.9]
    n_mels = [16]
    chunks = [1024]
    ridge_alphas = [0.01, 0.1, 1.0]
    input_offsets = [3.0]
    date = datetime.now().strftime("%Y%m%d-%H%M")
    os.makedirs(f"out/opt/{date}", exist_ok=True)
    cnt = 0
    for (net_height, net_width, net_alpha, n_mel, chunk, ridge_alpha,
         input_offset, net_sparse_rate) in itertools.product(
             net_heights, net_widths, net_alphas, n_mels, chunks, ridge_alphas,
             input_offsets, net_sparse_rates):
        subprocess.call([
            "python", "src/train/train.py", "--net-height",
            str(net_height), "--net-width",
            str(net_width), "--net-alpha",
            str(net_alpha), "--net-sparse-rate",
            str(net_sparse_rate), "--n-mel",
            str(n_mel), "--net-input-offset",
            str(input_offset), "--chunk",
            str(chunk), "--ridge-alpha",
            str(ridge_alpha), "--save-model", "--save-model-path",
            f"out/opt/{date}", "--score-path", f"out/opt/{date}/scores.json"
        ])
        cnt += 1
        print(f"Optimization step = {cnt}")
