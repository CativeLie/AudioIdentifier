import librosa
import numpy as np
import pandas as pd
import os

csv_path = "UrbanSound8K/metadata/UrbanSound8K.csv"
audio_root = "UrbanSound8K/audio"
save_path = "save_1d"
os.makedirs(save_path, exist_ok=True)

df = pd.read_csv(csv_path)
target_sample = 22050 * 3

for _,row in df.iterrows():
    wav_name = row["slice_file_name"]
    fold_num = row["fold"]
    wav_full = os.path.join(audio_root,f"fold{fold_num}",wav_name)
    y,sr = librosa.load(wav_full,sr=22050)
    if len(y) > target_sample:
        y = y[:target_sample]
    else:
        y = np.pad(y,(0,target_sample-len(y)),mode="constant")
    save_name = wav_name.replace(".wav",".npy")
    np.save(os.path.join(save_path,save_name),y)
print("一维数组全部生成完毕！")