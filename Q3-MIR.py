import librosa
import numpy as np
import utils
import scipy.signal as sps
from os import listdir

Ballroom = ['ChaCha', 'Jive', 'Quickstep', 'Rumba', 'Samba', 'Tango', 'Viennese waltz', 'Waltz']
ISMIR2004 = ['Abba', 'Alan_Parsons_Project', 'Alirio Diaz', 'aphex_twin', 'Asian_Dub_Foundation', 
           'Autechre', 'Aviador_Dro', 'Bach', 'Bach - Walcha', 'Bebel_Gilberto', 'Bela_Bartok',
           'Bernstein_conducts_Stravinsky', 'Billie Holiday CD1', 'Bjork', 'Cabaret_Voltaire',
           'Carlinhos_Brown', 'charles_mingus', 'Classic', 'Elton Medeiros, Nelson Sargento & Galo Preto',
           'Fado', 'Femi_kuti', 'Genesis', 'greek', 'GUITARE+', 'John Frusciante', 'john_coltrane',
           'Jose\' Merce\'', 'Kocani Orkester', 'Manu_Chao', 'more greek', 'Nina Pastori', 'Olivier Chassain',
           'Papakonstantinou', 'Paulinho da Viola & Elton Medeiros', 'Santana', 'Songs', 'Teresa Cristina',
           'Tomatito', 'Vangelis', 'Xatzidakis']


for ds_id, dataset in enumerate([Ballroom, ISMIR2004]):
    
    if ds_id == 0:
        continue
        print('#### Ballroom Dataset')
    else:
        print('#### ISMIR2004')

    total_files = 0
    AVG_ALOTC_SCORE = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    
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
            hop_length = 512
            win_length = [172, 258, 345, 431, 517]
            oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
            tempogram_auto, tempogram_fourier = [], []
            T12_AUTO, T12_FOURIER = [], []
            s_auto, s_fourier = [], []
            auto_frequency, fourier_frequency = [], []
            for w_id,wl in enumerate(win_length):
                
                tempogram_auto = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length, win_length=wl, norm=None)
                tempogram_fourier = librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length, win_length=wl)
                
                # estimate tempo get T1, T2
                tempo_vector_auto = np.sum(tempogram_auto, axis=1)
                tempo_vector_fourier = np.sum(np.abs(tempogram_fourier), axis=1)
                tempo_vector_auto = [x/tempogram_auto.shape[1] for x in  tempo_vector_auto]
                tempo_vector_fourier = [x/tempogram_fourier.shape[1] for x in tempo_vector_fourier]
                peak_id_auto = sps.argrelmax(np.array( tempo_vector_auto))[0]
                peak_id_fourier = sps.argrelmax(np.array(tempo_vector_fourier))[0]
                
                no_peak_auto, no_peak_fourier = False, False
                if len(peak_id_auto) < 2:
                    no_peak_auto = True
                if len(peak_id_fourier) < 2:
                    no_peak_fourier = True
                
                auto_frequency = librosa.tempo_frequencies(len( tempo_vector_auto))
                fourier_frequency = librosa.fourier_tempo_frequencies(win_length=wl)

                if w_id == 0:
                    reference_bpm = 0
                    with open(anno_bpm_file, 'r') as anno:
                        reference_str_bpm = anno.readline()
                        reference_bpm = utils.str2float(reference_str_bpm)
                
                temp = []
                if not no_peak_auto:
                    for id in peak_id_auto:
                        temp.append([ tempo_vector_auto[id], id])
                else:
                    for i in range(len( tempo_vector_auto)):
                        temp.append([ tempo_vector_auto[i], i])
                tempo_vector_auto = temp

                temp = []
                if not no_peak_fourier:
                    for id in peak_id_fourier:
                        temp.append([tempo_vector_fourier[id], id])
                else:
                    for i in range(len(tempo_vector_fourier)):
                        temp.append([tempo_vector_fourier[i], i])
                tempo_vector_fourier = temp
                
                tempo_vector_auto = sorted( tempo_vector_auto, key=lambda x: x[0], reverse=True)
                tempo_vector_fourier = sorted(tempo_vector_fourier, key=lambda x:x[0], reverse=True)
                
                if no_peak_auto:
                     tempo_vector_auto =  tempo_vector_auto[2:]
                if no_peak_fourier:
                    tempo_vector_fourier = tempo_vector_fourier[2:]

                # tempo_vector = [[max_avg_tempogram1, max_id1], [max_avg_tempogram2, max_id2], ...]
                T12_AUTO.append([auto_frequency[ tempo_vector_auto[0][1]], auto_frequency[ tempo_vector_auto[1][1]]])
                T12_FOURIER.append([fourier_frequency[tempo_vector_fourier[0][1]], fourier_frequency[tempo_vector_fourier[1][1]]])
                
                # ALOTC_score
                AVG_ALOTC_SCORE[0][w_id] += utils.ALOTC_SCORE(T12_AUTO[w_id], reference_bpm)
                AVG_ALOTC_SCORE[1][w_id] += utils.ALOTC_SCORE(T12_FOURIER[w_id], reference_bpm)
        
        total_files += len(files)   
    
    AVG_ALOTC_SCORE = [[score/total_files for score in AVG_ALOTC_SCORE[0]], [score/total_files for score in AVG_ALOTC_SCORE[1]]]
    print('|{}|4s|6s|8s|10s|12s|'.format(genre))
    print('|-----|----------|--------|------------|------------|----------|')
    print('|{}|{:6f}|{:6f}|{:6f}|{:6f}|{:6f}|'.format("AC", AVG_ALOTC_SCORE[0][0], AVG_ALOTC_SCORE[0][1], AVG_ALOTC_SCORE[0][2], AVG_ALOTC_SCORE[0][3], AVG_ALOTC_SCORE[0][4]))
    print('|{}|{:6f}|{:6f}|{:6f}|{:6f}|{:6f}|'.format("FOURIER", AVG_ALOTC_SCORE[1][0], AVG_ALOTC_SCORE[1][1], AVG_ALOTC_SCORE[1][2], AVG_ALOTC_SCORE[1][3], AVG_ALOTC_SCORE[1][4]))
    print()
    
    
    