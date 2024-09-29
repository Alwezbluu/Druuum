import json
from glob import glob

import numpy as np
from main_drumGen import generate
from GenreClass.transformer.model import transformer_classifier
from GenreClass.transformer.prepare_data import random_crop
from GenreClass.transformer.audio_processing import load_audio_file
from utils import create_wav_file_from_mp3_file
basic_genre = [ 'punk', 'pop', 'hip-hop', 'jazz', 'funk']
import os
print(f"当前工作目录: {os.getcwd()}")


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def map_genre(pre_genre, basic_genre):
    lower_basic_genre = [str(s).lower() for s in basic_genre]
    for g1 in pre_genre:
        lower_pre_genre = str(g1).lower()
        for g2 in lower_basic_genre:
            if g2 in lower_pre_genre:
                return g2
    return None



def finalPredict(filename):
    transformer_v2_h5 = "../GenreClass/transformer/transformer.h5"

    CLASS_MAPPING = json.load(open("D:/Downloads/fma_metadata/mapping.json"))

    # base_path = "../audio"
    # files = sorted(list(glob(base_path + "/*.mp3")))
    files = [filename]
    data = [load_audio_file(x, input_length=16000 * 120) for x in files]

    transformer_v2_model = transformer_classifier(n_classes=len(CLASS_MAPPING))

    transformer_v2_model.load_weights(transformer_v2_h5)

    crop_size = np.random.randint(128, 512)
    repeats = 8

    transformer_v2_Y = 0

    for _ in range(repeats):
        X = np.array([random_crop(x, crop_size=crop_size) for x in data])

        transformer_v2_Y += transformer_v2_model.predict(X) / repeats

    transformer_v2_Y = transformer_v2_Y.tolist()

    # for path, pred in zip(files, transformer_v2_Y):
    #
    #     print(path)
    #     pred_tup = [(k, pred[v]) for k, v in CLASS_MAPPING.items()]
    #     pred_tup.sort(key=lambda x: x[1], reverse=True)
    #
    #     for a in pred_tup[:5]:
    #         print(a)
    for path, pred in zip(files, transformer_v2_Y):
        print(path)

        # Create a list of (class_name, prediction_score) tuples
        pred_tup = [(k, pred[v]) for k, v in CLASS_MAPPING.items()]

        # Sort tuples by prediction score in descending order
        pred_tup.sort(key=lambda x: x[1], reverse=True)

        # Print top 5 predicted class names
        top_five_classes = [pred_tup[i][0] for i in range(min(5, len(pred_tup)))]
        print("Top 5 predicted classes:", top_five_classes)

        genre = map_genre(top_five_classes, basic_genre)
        print(genre)
        print(files[0])
        # create_wav_file_from_mp3_file(files[0], '..\\audio\\test.wav')
        create_wav_file_from_mp3_file(files[0], '../GenreClass/audio/test.wav')
        if genre:
            output_file_dir, tempo = generate('../GenreClass/audio/test', genre, True, True, 4)
        else:
            output_file_dir, tempo = generate('../GenreClass/audio/test', 'punk', True, True, 4)
        return top_five_classes, tempo, output_file_dir


if __name__ == "__main__":
    finalPredict("E:\\Downloads\\许巍 - 蓝莲花.mp3")