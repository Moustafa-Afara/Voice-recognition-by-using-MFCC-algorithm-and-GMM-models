"""
Closed-set speaker identification with MFCC features and one Gaussian Mixture Model per speaker.

Enrolment: for every speaker, MFCC frames from a few training recordings are pooled and a GMM is
fitted to them. Identification: a test recording's MFCC frames are scored under every speaker's
GMM and the speaker with the highest total log-likelihood is chosen.

Data layout expected (any depth below the speaker folder, .wav or .flac):
    <data-root>/<speaker_id>/.../*.flac
LibriSpeech (e.g. dev-clean) already has this layout.

Usage:
    python speaker_id.py --data-root LibriSpeech/dev-clean
    python speaker_id.py --data-root LibriSpeech/dev-clean --method legacy     # original 2020 method
"""
import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from python_speech_features import mfcc
from scipy.signal import resample_poly
from sklearn.mixture import GaussianMixture

AUDIO_EXT = {".wav", ".flac"}


def list_speakers(root: Path) -> dict:
    speakers = {}
    for spk_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(f for f in spk_dir.rglob("*") if f.suffix.lower() in AUDIO_EXT)
        if files:
            speakers[spk_dir.name] = files
    return speakers


def split(speakers: dict, n_train: int, n_test: int, seed: int):
    """Per speaker, sample n_train + n_test distinct recordings at random (seeded)."""
    rng = random.Random(seed)
    train, test = {}, {}
    for spk, files in speakers.items():
        if len(files) < n_train + n_test:
            raise ValueError(f"speaker {spk} has only {len(files)} recordings")
        chosen = rng.sample(files, n_train + n_test)
        train[spk], test[spk] = chosen[:n_train], chosen[n_train:]
    return train, test


def split_cross_session(speakers: dict, n_train: int, n_test: int, seed: int):
    """Train on one recording session and test on a different one (the parent folder of a file is
    taken as its session, e.g. a LibriSpeech chapter). Speakers with fewer than two sessions that
    have enough recordings are left out."""
    rng = random.Random(seed)
    train, test = {}, {}
    for spk, files in speakers.items():
        sessions = {}
        for f in files:
            sessions.setdefault(f.parent, []).append(f)
        tr_ok = [s for s in sessions if len(sessions[s]) >= n_train]
        pairs = [(a, b) for a in tr_ok for b in sessions if b != a and len(sessions[b]) >= n_test]
        if not pairs:
            continue
        a, b = rng.choice(sorted(pairs))
        train[spk], test[spk] = rng.sample(sessions[a], n_train), rng.sample(sessions[b], n_test)
    return train, test


def mfcc_frames(path: Path, sample_rate: int, cmn: bool) -> np.ndarray:
    """13 MFCCs per 25 ms window, 10 ms step (python_speech_features defaults, as in the original)."""
    signal, sr = sf.read(str(path), dtype="float32")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if sample_rate and sr != sample_rate:
        signal, sr = resample_poly(signal, sample_rate, sr), sample_rate
    feats = mfcc(signal, sr, winlen=0.025, winstep=0.010, numcep=13, nfft=int(2 ** np.ceil(np.log2(0.025 * sr))))
    if cmn:  # cepstral mean normalisation: removes the fixed channel colouring of each recording
        feats = feats - feats.mean(axis=0, keepdims=True)
    return feats


def fit_models(train: dict, method: str, sample_rate: int, cmn: bool, seed: int) -> dict:
    models = {}
    for spk, files in train.items():
        frames = np.vstack([mfcc_frames(f, sample_rate, cmn) for f in files])
        if method == "legacy":
            # Original 2020 code: every MFCC coefficient treated as an independent scalar sample.
            gmm = GaussianMixture(n_components=10, covariance_type="full", max_iter=20, random_state=seed)
            gmm.fit(frames.reshape(-1, 1))
        else:
            gmm = GaussianMixture(n_components=16, covariance_type="diag", max_iter=200, reg_covar=1e-3,
                                  random_state=seed)
            gmm.fit(frames)
        models[spk] = gmm
    return models


def identify(models: dict, path: Path, method: str, sample_rate: int, cmn: bool) -> str:
    frames = mfcc_frames(path, sample_rate, cmn)
    names = list(models)
    if method == "legacy":
        # Original decision rule: each scalar votes for its most likely speaker; majority wins.
        scores = np.stack([models[n].score_samples(frames.reshape(-1, 1)) for n in names])
        votes = np.bincount(scores.argmax(axis=0), minlength=len(names))
        return names[int(votes.argmax())]
    total = [models[n].score_samples(frames).sum() for n in names]  # sum of frame log-likelihoods
    return names[int(np.argmax(total))]


def evaluate(root: Path, method: str, n_train: int, n_test: int, seed: int, sample_rate: int, cmn: bool,
             cross_session: bool = False):
    speakers = list_speakers(root)
    if cross_session:
        train, test = split_cross_session(speakers, n_train, n_test, seed)
    else:
        train, test = split(speakers, n_train, n_test, seed)
    models = fit_models(train, method, sample_rate, cmn, seed)
    correct = total = 0
    for spk, files in test.items():
        for f in files:
            correct += identify(models, f, method, sample_rate, cmn) == spk
            total += 1
    return correct, total, len(train)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--method", choices=["gmm", "legacy"], default="gmm",
                   help="gmm: 13-dim MFCC frames, diagonal GMM-16, summed log-likelihood (default); "
                        "legacy: the original 2020 method, kept for comparison")
    p.add_argument("--n-train", type=int, default=7, help="training recordings per speaker")
    p.add_argument("--n-test", type=int, default=3, help="test recordings per speaker")
    p.add_argument("--sample-rate", type=int, default=8000, help="resample to this rate (0 = keep original)")
    p.add_argument("--cmn", action="store_true", help="apply cepstral mean normalisation")
    p.add_argument("--cross-session", action="store_true",
                   help="train and test recordings come from different sessions (harder, more realistic)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="one random train/test split per seed; mean and std are reported")
    a = p.parse_args()

    accs = []
    for seed in a.seeds:
        c, t, n_spk = evaluate(a.data_root, a.method, a.n_train, a.n_test, seed, a.sample_rate, a.cmn, a.cross_session)
        accs.append(c / t)
        print(f"seed {seed}: {c}/{t} correct = {c / t:.3f}  ({n_spk} speakers)")
    accs = np.array(accs)
    print(f"method={a.method} cmn={a.cmn} cross_session={a.cross_session} sr={a.sample_rate or 'native'} "
          f"train/test per speaker={a.n_train}/{a.n_test}: accuracy {accs.mean():.3f} ± {accs.std(ddof=1) if len(accs) > 1 else 0:.3f}")


if __name__ == "__main__":
    main()
