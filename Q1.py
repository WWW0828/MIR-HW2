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
        print('#### Ballroom Dataset')
    else:
        print('#### ISMIR2004')
        
    print('|Genre|AC/Fourier|P SCORE|ALOTC SCORE|')
    print('|-----|----------|-------|-----------|')

    for g_id, genre in enumerate(dataset):
        
        AVG_P_SCORE = [0, 0]
        AVG_ALOTC_SCORE = [0, 0]

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
            oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
            tempogram_auto = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length, norm=None)
            tempogram_fourier = librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
            
            # estimate tempo get T1, T2
            tempo_vector_auto = np.sum(tempogram_auto, axis=1)
            tempo_vector_fourier = np.sum(np.abs(tempogram_fourier), axis=1)
            tempo_vector_auto = [x/tempogram_auto.shape[1] for x in tempo_vector_auto]
            tempo_vector_fourier = [x/tempogram_fourier.shape[1] for x in tempo_vector_fourier]
            
            peak_id_auto = sps.argrelmax(np.array(tempo_vector_auto))
            peak_id_fourier = sps.argrelmax(np.array(tempo_vector_fourier))

            auto_frequency = librosa.tempo_frequencies(len(tempo_vector_auto))
            fourier_frequency = librosa.fourier_tempo_frequencies(hop_length = hop_length)

            temp = []
            for id in peak_id_auto[0]:
                temp.append([tempo_vector_auto[id], id])
            tempo_vector_auto = temp

            temp = []
            for id in peak_id_fourier[0]:
                temp.append([tempo_vector_fourier[id], id])
            tempo_vector_fourier = temp
            
            tempo_vector_auto = sorted(tempo_vector_auto, key=lambda x: x[0], reverse=True)
            tempo_vector_fourier = sorted(tempo_vector_fourier, key=lambda x:x[0], reverse=True)

            reference_bpm = ''
            with open(anno_bpm_file, 'r') as anno:
                reference_str_bpm = anno.readline()
                reference_bpm = utils.str2float(reference_str_bpm)

            # tempo_vector = [[max_avg_tempogram1, max_id1], [max_avg_tempogram2, max_id2], ...]
            T12_AUTO = [auto_frequency[tempo_vector_auto[0][1]], auto_frequency[tempo_vector_auto[1][1]]]
            T12_FOURIER = [fourier_frequency[tempo_vector_fourier[0][1]], fourier_frequency[tempo_vector_fourier[1][1]]]

            # p_score
            s_auto = (tempo_vector_auto[0][0])/(tempo_vector_auto[0][0] + tempo_vector_auto[1][0])
            s_fourier = (tempo_vector_fourier[0][0])/(tempo_vector_fourier[0][0] + tempo_vector_fourier[1][0])
            AVG_P_SCORE[0] += utils.P_SCORE(T12_AUTO, s_auto, reference_bpm)
            AVG_P_SCORE[1] += utils.P_SCORE(T12_FOURIER, s_fourier, reference_bpm)

            # ALOTC_score
            AVG_ALOTC_SCORE[0] += utils.ALOTC_SCORE(T12_AUTO, reference_bpm)
            AVG_ALOTC_SCORE[1] += utils.ALOTC_SCORE(T12_FOURIER, reference_bpm)
            
            # print('=========================================================================================')
            #print(wavfile)
            # print('--AUTO-----------------------------------------------------------------------------------')
            # print('estimate:', T12_AUTO[0], T12_AUTO[1])
            # print('P_SCORE:', P_SCORE_AC)
            # print('ALOTC_SCORE:', ALOTC_SCORE_AC)
            # print('--FOURIER--------------------------------------------------------------------------------')
            # print('estimate:', T12_FOURIER[0], T12_FOURIER[1])
            # print('P_SCORE:', P_SCORE_FOURIER)
            # print('ALOTC_SCORE:', ALOTC_SCORE_FOURIER)
            # print('--REFERENCE------------------------------------------------------------------------------')
            # print(reference_bpm)
            # print('=========================================================================================')
            # print()

        AVG_P_SCORE = [score/len(files) for score in AVG_P_SCORE]
        AVG_ALOTC_SCORE = [score/len(files) for score in AVG_ALOTC_SCORE]
        print('|{}|{}|{:6f}|{:6f}|'.format(genre,"AC", AVG_P_SCORE[0], AVG_ALOTC_SCORE[0]))
        print('|{}|{}|{:6f}|{:6f}|'.format('',"FOURIER", AVG_P_SCORE[1], AVG_ALOTC_SCORE[1]))
    
    
    