import numpy as np
import librosa
import os
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

src_dir = os.path.join(BASE_DIR, "save_1d")
dst_dir = os.path.join(BASE_DIR, "save_mel_npy")
os.makedirs(dst_dir, exist_ok=True)

files = sorted([f for f in os.listdir(src_dir) if f.endswith(".npy")])
print(f"Found {len(files)} files to process")

for fname in tqdm(files, desc="Generating mel spectrograms"):
    dst_path = os.path.join(dst_dir, fname)
    if os.path.exists(dst_path):
        continue
    y = np.load(os.path.join(src_dir, fname))
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    np.save(dst_path, mel_db)

print(f"Done! Mel spectrograms saved to '{dst_dir}/' ({len(os.listdir(dst_dir))} files)")
