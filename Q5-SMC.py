import librosa
import mir_eval
import numpy as np
from os import listdir

print('SMC_MIREX')

f_score = 0
anno_folder_path = 'SMC_MIREX/SMC_MIREX_Annotations_05_08_2014'
anno_files = listdir(anno_folder_path)
reference_beats_list = []
for annofile in anno_files:
    reference_beats = []
    with open ('{}/{}'.format(anno_folder_path, annofile), 'r') as anno:
        lines = anno.readlines()
        reference_str_beats = [line[:-1] for line in lines]
        for bt in reference_str_beats:
            num = 0
            after_point = 0
            for i in bt:
                if(i == '.'):
                    after_point = 1
                    continue
                if(after_point == 0):
                    num = num * 10 + int(i)
                else:
                    num = num + int(i) * 10**(-1 * after_point)
                    after_point += 1
            reference_beats.append(num)
    reference_beats_list.append(reference_beats)

folder_path = 'SMC_MIREX/SMC_MIREX_Audio'
wavfiles = listdir(folder_path)

for f_id, wavfile in enumerate(wavfiles):
    y, sr = librosa.load('{folder}/{file}'.format(folder = folder_path, file = wavfile))
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    estimated_beats = librosa.frames_to_time(beats, sr=sr)
    f_score += mir_eval.beat.f_measure(np.array(reference_beats_list[f_id]), estimated_beats)

print('f_score: {:.6f}'.format(f_score/len(wavfiles)))