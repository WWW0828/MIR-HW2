import librosa
import mir_eval
import numpy as np
from os import listdir

ASAP = ['Debussy', 'Glinka', 'Liszt']
print('ASAP')
print('|Genre|f_score|')
print('|-----|-------|')
for g_id, genre in enumerate(ASAP):
    f_score = 0
    folder_path = 'ASAP/' + genre + '/wav'
    files = listdir(folder_path)
    for f_id, wavfile in enumerate(files):
        anno_beats_file = 'ASAP/' + genre + '/annotation/' + wavfile[:-4] + '_annotations.txt'
        reference_beats = []
        with open (anno_beats_file, 'r') as anno:
            lines = anno.readlines()
            reference_str_beats = [line.split()[0] for line in lines]
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
        y, sr = librosa.load('{folder}/{file}'.format(folder = folder_path, file = wavfile))
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        estimated_beats = librosa.frames_to_time(beats, sr=sr)
        f_score += mir_eval.beat.f_measure(np.array(reference_beats), estimated_beats)
    print('|{}|{:.6f}|'.format(genre, f_score/len(files)))