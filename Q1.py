from pickle import NONE
import librosa
import mir_eval
import numpy as np
import scipy.signal as sps
import utils
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

for ds_id, dataset in enumerate([ISMIR2004, Ballroom]):
    for g_id, genre in enumerate(dataset):
        if(ds_id == 1):
            folder_path = 'Ballroom/BallroomData/' + genre
        else:
            folder_path = 'ISMIR2004/' + genre + '/wav'
        files = listdir(folder_path)
        
        for f_id, wavfile in enumerate(files):
            if(ds_id == 1):
                anno_bpm_file = 'Ballroom/BallroomAnnotations/ballroomGroundTruth/' + wavfile[:-3] + 'bpm'
            else:
                anno_bpm_file = 'ISMIR2004/' + genre + '/annotation/' + wavfile[:-4] + ' beat.bpm'

            y, sr = librosa.load('{folder}/{file}'.format(folder = folder_path, file = wavfile))
            hop_length = 512
            oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
            tempogram_auto = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length, norm=None)
            tempogram_fourier = librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
            
            # estimate tempo get T1, T2
            tempo_vector_auto = np.sum(tempogram_auto, axis=1)
            tempo_vector_fourier = np.sum(np.abs(tempogram_fourier), axis=1)
            
            tempo_vector_auto = [x/tempogram_auto.shape[1] for x in tempo_vector_auto]
            tempo_vector_fourier = [x/tempogram_fourier.shape[1] for x in tempo_vector_fourier]
            
            for i in range(len(tempo_vector_auto)):
                tempo_vector_auto[i] = [tempo_vector_auto[i], i]
            for i in range(len(tempo_vector_fourier)):
                tempo_vector_fourier[i] = [tempo_vector_fourier[i], i]
            
            tempo_vector_auto = sorted(tempo_vector_auto[2:], key=lambda x: x[0], reverse=True)
            tempo_vector_fourier = sorted(tempo_vector_fourier[2:], key=lambda x:x[0], reverse=True)

            auto_frequency = librosa.tempo_frequencies(len(tempo_vector_auto))
            fourier_frequency = librosa.fourier_tempo_frequencies(hop_length = hop_length)
            
            reference_bpm = ''
            with open(anno_bpm_file, 'r') as anno:
                reference_str_bpm = anno.readline()
                reference_bpm = utils.str2float(reference_str_bpm)

            print('=========================================================================================')
            print(wavfile)
            print('----------------------------------------------------------------------------------------')
            print('estimate_auto:', [auto_frequency[tempo_vector_auto[i][1]] for i in range(2)] )
            print('estimate_fourier:', [fourier_frequency[tempo_vector_fourier[i][1]] for i in range(2)], '\treference:', reference_bpm )
            print('=========================================================================================')
            print()

            # p_score
            # ALOTC_score
        break
    break
    