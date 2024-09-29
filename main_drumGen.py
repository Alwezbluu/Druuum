import os
import click
import librosa
import numpy as np
import utils
from drumGen.drums_generator import predict
from drumGen.methods import beat_tracking
from drumGen.methods.bass_generator import generate_bass
from drumGen.methods.chord_recognition import rearrange_by_chord_recognition, chord_rec
from drumGen.drums_generator.midi2wav import convert_midi_to_wav
from drumGen.methods.novelty_detection import NoveltyDetection
import soundfile as sf
from pydub import AudioSegment
from pydub import effects



def beat_maker(track_name, genre, drums, bass, k):
    # audio processing
    audio_path = "{}.wav".format(track_name)
    audio_data, samp_rate = utils.get_wav_data(audio_path)
    audio_data, index = librosa.effects.trim(audio_data)

    # apply clustering and splitting the track into k-clusters
    # intervals, bound_segs = NoveltyDetection(audio_data, samp_rate, k).novelty_detection()

    # beat tracking and rearranging each interval by chord detection
    # remix = list()
    # for interval in intervals:
    #     try:
    #         time_1, time_2 = np.array([(interval[0] * samp_rate).astype(np.int32), (interval[1] * samp_rate).astype(np.int32)])
    #         tempo, bt_slices = beat_tracking.slice_by_beat_tracking(audio_data[time_1:time_2],
    #                                                                 samp_rate)
    #         inner_arrangement = rearrange_by_chord_recognition(bt_slices, samp_rate)
    #     except:
    #         continue
    #     remix.append(inner_arrangement)

    # rearranging the clusters
    # remix = rearrange_by_chord_recognition(remix, samp_rate)
    y, sr = librosa.load(audio_path)

    onset_env = librosa.onset.onset_strength(y, sr=sr, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, y=y, sr=sr, units='time')
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, bpm=tempo, units='time')
    print("tempo:", tempo)
    # bass generation into midi file and converting the midi to wav file
    if not bass:
        chord_times = np.ediff1d(librosa.frames_to_time(beats, sr=samp_rate)[::4] * 1000)
        chords = chord_rec(audio_data, samp_rate)
        midi_bass = generate_bass(chords, tempo, chord_times)
        midi_bass.save('latest_bass.mid')
        convert_midi_to_wav('latest_bass.mid', 'latest_bass.wav')

        bass_wav_data, _ = utils.get_wav_data('latest_bass.wav')
        bass_wav_data = librosa.effects.trim(librosa.util.normalize(bass_wav_data))[0]

    else:
        bass_wav_data = np.zeros(audio_data.shape[0])

    # drums generation into midi file and converting the midi to wav file
    if drums:
        midi_drums = predict.generate(tempo=round(tempo), genre=genre)
        print(round(tempo))
        midi_drums.save('latest_drums.mid')
        convert_midi_to_wav('latest_drums.mid', 'latest_drums.wav')

        drums_wav_data, _ = utils.get_wav_data('latest_drums.wav')
        drums_wav_data = librosa.effects.trim(librosa.util.normalize(drums_wav_data))[0] * 2

    else:
        drums_wav_data = np.zeros(audio_data.shape[0])

    # sync
    remix_wav_data = librosa.util.normalize(audio_data)

    remix_wav_len = remix_wav_data.shape[0]
    drums_wav_len = drums_wav_data.shape[0]
    bass_wav_len = bass_wav_data.shape[0]
    #
    min_length = min(remix_wav_len, drums_wav_len, bass_wav_len)
    #
    # wav_data = remix_wav_len[:min_length-1] + drums_wav_data[:min_length-1] + bass_wav_data[:min_length-1]
    # wav_data = np.array([
    #    remix_wav_data[:min_length],
    #     drums_wav_data[:min_length],
    #     bass_wav_data[:min_length]]
    # )


    # onsets_frames = librosa.onset.onset_detect(y)
    # print(onsets_frames[1])
    # if len(onsets_frames)>1:
    #     onset_position = onsets_frames[1]
    # else:
    #     onset_position = 0
    # onset_position_ms = librosa.frames_to_time(onset_position, sr=sr) * 1000
    # beats_time = librosa.frames_to_time(beats, sr=sr) * 1000
    # 假设为44拍
    meter = 4
    measures = (len(beats) // meter)
    # beat_strengths = onset_env[beats]
    # measure_beat_strengths = beat_strengths[:measures * meter].reshape(-1, meter)
    # beat_pos_strength = np.sum(measure_beat_strengths, axis=0)
    # downbeat_pos = np.argmax(beat_pos_strength)
    # full_measure_beats = beats[:measures * meter].reshape(-1, meter)
    # downbeat_frames = full_measure_beats[:, downbeat_pos]
    # downbeat_times = librosa.frames_to_time(downbeat_frames, sr=sr)
    # print(downbeat_times[1])
    beat_time = librosa.frames_to_time(beats, sr=sr)

    drum_wav = AudioSegment.from_wav('latest_drums.wav')
    origin_wav = AudioSegment.from_wav(audio_path)

    output_path_list = []
    for i in range(0,3):
        drum_wav = drum_wav.apply_gain(10 * i)
        drum_wav = effects.normalize(drum_wav, headroom=3)
        origin_wav = effects.normalize(origin_wav, headroom=3)
        print(beat_time[1])
        sound_with_drum = origin_wav.overlay(drum_wav, position=beats[1] * 1000)
        output_filename = os.path.basename(audio_path).replace('.wav', f'(remix)_+{i}0db.wav')
        output_path = os.path.join(os.path.dirname(audio_path), output_filename)
        sound_with_drum.export(output_path, format="wav")
        print("file saved at:", output_path)
        output_path_list.append(output_path)

    return output_path_list, tempo

    # return wav_data, samp_rate


# @click.command()
# @click.option('--track', help='track name', type=str, required=True)
# @click.option('--drums', help='apply drums', type=bool, default=True)
# @click.option('--bass', help='apply bass', type=bool, default=False)
# @click.option('--k', help='k clusters', type=int, default=2)
def generate(track, genre, drums, bass, k):
    # final_res, samp_rate = beat_maker(track, genre, drums, bass, k)
    # sf.write(f'{track}(remix).wav', final_res, samp_rate)
    fileList, tempo = beat_maker(track, genre, drums, bass, k)
    if fileList:
        print("done!:)")
    # print("file saved at:", output_path)
    return fileList, tempo


if __name__ == "__main__":
    generate('test', 'True',bass=True, k=4)

