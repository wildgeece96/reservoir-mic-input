import os
from datetime import datetime
import subprocess
import itertools

if __name__ == '__main__':
    net_heights = [5, 10, 20, 30]
    net_widths = [5, 10, 20, 30]
    net_alphas = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.95]
    n_mels = [16, 32, 64, 80]
    chunks = [32, 64, 128, 256]
    ridge_alphas = [0.01, 0.1, 1.0, 10.0]
    date = datetime.now().strftime("%Y%m%d-%H%M")
    os.makedirs(f"out/opt/{date}", exist_ok=True)
    for (net_height, net_width, net_alpha, n_mel, chunk,
         ridge_alpha) in itertools.product(net_heights, net_widths, net_alphas,
                                           n_mels, chunks, ridge_alphas):
        subprocess.call([
            "python", "src/train/train.py", "--net-height",
            str(net_height), "--net-width",
            str(net_width), "--net-alpha",
            str(net_alpha), "--n-mel",
            str(n_mel), "--chunk",
            str(chunk), "--ridge-alpha",
            str(ridge_alpha), "--save-model", "--save-model-path",
            f"out/opt/{date}", "--score-path", f"out/opt/{date}/scores.json"
        ])
