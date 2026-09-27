"""Plot the MFCC matrix and log filter-bank energies of one recording.

Usage: python plot_mfcc.py path/to/recording.flac [--seconds 1.0]
"""
import argparse

import matplotlib.pyplot as plt
import soundfile as sf
from python_speech_features import logfbank, mfcc

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("path")
p.add_argument("--seconds", type=float, default=1.0, help="length of the excerpt to plot")
p.add_argument("--save", default="", help="save the figure to this file instead of showing it")
a = p.parse_args()

signal, sr = sf.read(a.path, dtype="float32")
if signal.ndim > 1:
    signal = signal.mean(axis=1)
signal = signal[: int(a.seconds * sr)]
nfft = 512 if sr <= 16000 else 1024

features_mfcc = mfcc(signal, sr, nfft=nfft)
filterbank = logfbank(signal, sr, nfft=nfft)
print(f"MFCC: {features_mfcc.shape[0]} windows x {features_mfcc.shape[1]} coefficients")
print(f"Filter bank: {filterbank.shape[0]} windows x {filterbank.shape[1]} bands")

fig, axes = plt.subplots(2, 1, figsize=(9, 6))
axes[0].imshow(features_mfcc.T, aspect="auto", origin="lower")
axes[0].set_title("MFCC (13 coefficients per 10 ms step)")
axes[0].set_ylabel("coefficient")
axes[1].imshow(filterbank.T, aspect="auto", origin="lower")
axes[1].set_title("Log Mel filter-bank energies (26 bands)")
axes[1].set_ylabel("band")
axes[1].set_xlabel("frame (10 ms)")
fig.tight_layout()
plt.savefig(a.save, dpi=120) if a.save else plt.show()
