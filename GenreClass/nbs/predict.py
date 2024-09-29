from tensorflow.python.keras.models import load_model
import os
import librosa
from librosa import feature
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras import utils
from keras.utils import to_categorical
import tensorflow as tf
import datetime
import pydub
from pydub import AudioSegment
import sys
from pydub.silence import split_on_silence

label_dict = {
    0: 'jazz',
    1: 'raggae',
    2: 'rock',
    3: 'blues',
    4: 'hiphop',
    5: 'country',
    6: 'metal',
    7: 'classical',
    8: 'disco',
    9: 'pop'
}
input_directory = "../data/input/"
# 将音频文件分成30s小片段
model = tf.keras.models.load_model('../model/2024-07-15_16-14-51.txt.h5')
src = "../nbs/test.mp3"
dst = "../nbs/test.wav"
audSeg = AudioSegment.from_mp3(src)
audSeg.export(dst, format="wav")
audio_segment = AudioSegment.from_file(dst, format='wav')

total = int(audio_segment.duration_seconds / 30)
for i in range(total):
    audio_segment[i*30000:(i+1)*30000].export("../data/input/chunk{0}.wav".format(i), format="wav")
audio_segment[total*30000:].export("../data/input/chunk{0}.wav".format(total), format="wav")
mel_specs = []
# 将所有片段输入模型进行结果预测，少数服从多数原则
for file in os.scandir(input_directory):
    y, sr = librosa.core.load(file)
    spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
    librosa.display.specshow(spect,y_axis='frequency', x_axis='time')
    spect = librosa.power_to_db(spect, ref=np.max)

    if spect.shape[1] != 660:
        spect.resize(128, 660, refcheck=False)
    mel_specs.append(spect)
mel_specs = np.array(mel_specs)
mel_specs = mel_specs.reshape((mel_specs.shape[0], 128, 660, 1))
pred_result = model.predict(mel_specs, verbose=1)
print(pred_result)

flag = np.zeros(10)
for x in pred_result:
    index = np.argmax(x)
    flag[int(index)] += 1
m = np.argmax(flag)
print(m)
print('预测流派为：', label_dict[m])





