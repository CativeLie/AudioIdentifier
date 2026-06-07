import librosa
import librosa.display
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

csv_path = "UrbanSound8K/metadata/UrbanSound8K.csv"
audio_root = "UrbanSound8K/audio"
pic_save = "save_mel_pic"
os.makedirs(pic_save, exist_ok=True)

df = pd.read_csv(csv_path)
df = df.head(50) # 和前面一致，只处理50个

for _,row in df.iterrows():
    wav_name = row["slice_file_name"]
    fold_num = row["fold"]
    wav_path = os.path.join(audio_root,f"fold{fold_num}",wav_name)
    y,sr = librosa.load(wav_path,sr=22050)
    mel = librosa.feature.melspectrogram(y=y,sr=sr)
    mel_db = librosa.power_to_db(mel,ref=np.max)

    plt.figure(figsize=(3,3))
    librosa.display.specshow(mel_db,sr=sr)
    plt.axis("off")
    save_name = wav_name.replace(".wav",".png")
    plt.savefig(os.path.join(pic_save,save_name),bbox_inches="tight",pad_inches=0)
    plt.close()

print("梅尔频谱图片全部生成完毕！")