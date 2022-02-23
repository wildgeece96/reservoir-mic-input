# Reservoir Mic Input

Realtime analysis of mic input audio with reservoir network.

## Setup

```bash
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

## Train

You can specify other parameters. Like, `--n-fft`, `--n-mels`.

```bash
python src/train/train_integ_sal.py --dims 100,100 --train-sal --save-model --save-model-path ../out/models/sample
```

Watch dynamics with mic input.

```bash
python src/core/watch_net_dynamics.py --config-file-dir ../out/models/sample --chunk-size 1024
```

## Data

The name of data separated according to that format.

```zsh
<recording place>_<sound type>_<number>.wav
```
