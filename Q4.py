import librosa
import mir_eval
import numpy as np
from os import listdir
import utils

Ballroom = ['ChaCha', 'Jive', 'Quickstep', 'Rumba', 'Samba', 'Tango', 'Viennese waltz', 'Waltz']
print('|Genre|f_score|')
print('|-----|-------|')
for g_id, genre in enumerate(Ballroom):
    f_score = 0
    folder_path = 'Ballroom/BallroomData/' + genre
    files = listdir(folder_path)
    for f_id, wavfile in enumerate(files):
        anno_beats_file = 'Ballroom/BallroomAnnotations-master/' + wavfile[:-3] + 'beats'
        reference_beats = []
        with open (anno_beats_file, 'r') as anno:
            lines = anno.readlines()
            reference_str_beats = [line.split()[0] for line in lines]
            for bt in reference_str_beats:
                reference_beats.append(utils.str2float(bt))
        y, sr = librosa.load('{folder}/{file}'.format(folder = folder_path, file = wavfile))
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        estimated_beats = librosa.frames_to_time(beats, sr=sr)
        f_score += mir_eval.beat.f_measure(np.array(reference_beats), estimated_beats)
    print('|{}|{:.6f}|'.format(genre, f_score/len(files)))
        
    