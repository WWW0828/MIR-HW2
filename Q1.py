from pickle import NONE
import librosa
import mir_eval
import numpy as np
from os import listdir

Ballroom = ['ChaCha', 'Jive', 'Quickstep', 'Rumba', 'Samba', 'Tango', 'Viennese waltz', 'Waltz']
ISMIR2004 = ['Abba', 'Alan_Parsons_Project', 'Alirio Diaz', 'aphex_twin', 'Asian_Dub_Foundation', 
           'Autechre', 'Aviador_Dro', 'Bach', 'Bach - Walcha', 'Bebel_Giberto', 'Bela_Bartok',
           'Bernstein_conducts_Stravinsky', 'Billie Holiday CD1', 'Bjork', 'Cabaret_Voltaire',
           'Carlinhos_Brown', 'charles_mingus', 'Classic', 'Elton Medeiros, Nelson Sargento & Galo Preto',
           'Fado', 'Femi_kuti', 'Genesis', 'greek', 'GUITARE+', 'John Frusciante', 'john_coltrane',
           'Jose\' Merce\'', 'Kocani Orkester', 'Manu_Chao', 'more greek', 'Nina Pastori', 'Oliver Chassain',
           'Papakonstantinou', 'Paulinho da Viola & Elton Medeiros', 'Santana', 'Songs', 'Teresa Cristina',
           'Tomatito', 'Vangelis', 'Xatzidakis']

for ds_id, dataset in enumerate([Ballroom, ISMIR2004]):
    for g_id, genre in enumerate(dataset):
        if(ds_id == 0):
            folder_path = 'Ballroom/BallroomData/' + genre
        else:
            folder_path = 'ISMIR2004/' + genre + '/wav'
        files = listdir(folder_path)
        
        for f_id, wavfile in enumerate(files):
            if(ds_id == 0):
                anno_bpm_file = 'Ballroom/BallroomAnnotations/ballroomGroundTruth/' + wavfile[:-3] + 'bpm'
            else:
                anno_bpm_file = 'ISMIR2004/' + genre + '/annotation/' + wavfile[:-4] + ' beat.bpm'

            y, sr = librosa.load('{folder}/{file}'.format(folder = folder_path, file = wavfile))
            hop_length = 2205
            win_length = [0, 40, 60, 80, 100, 120]
            oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
            tempo_auto = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length, norm=None)
            tempo_fouier = librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
            # estimate tempo get T1, T2
            print(wavfile)
            print('ac: ', tempo_auto.shape)
            print(tempo_auto)
            print('fo: ', tempo_fouier.shape)
            print(np.abs(tempo_fouier))

            break
        break
    