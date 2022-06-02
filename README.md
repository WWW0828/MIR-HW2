# MIR-HW2
- 11020 Music Information Retrieval, Su Li (Academia Sinica) 
- Homework 2 Tempo estimation, beat/downbeat tracking, and meter recognition of audio and symbolic data

## Dataset
- ISMIR2004 (tempo)
- Ballroom (tempo, beat, downbeat)
- SMC (beat)
- JCS (beat, downbeat, meter)
- ASAP (beat, downbeat, meter)

## functions in librosa
- librosa.feature.fourier_tempogram
- librosa.feature.tempogram
- librosa.beat.tempo
- librosa.beat.beat_track
- librosa.tempo_frequencies
- librosa.fourier_tempo_frequencies

## Task1: tempo estimation
### Q1 (20%)
Design an algorithm that estimate the tempo for the ISMIR2004 and the Ballroom dataset. Assume that the tempo of every clip is constant. Note that your algorithm should output two predominant tempi for each clip: 𝑇1 (the slower one) and 𝑇2 (the faster one). For example, you may simply try the two largest peak values in the tempogram over the whole clip. Please compare and discuss the results computed from the Fourier tempogram and the autocorrelation tempogram.

### Q2 (20%)
Instead of using your estimated [𝑇1, 𝑇2] in evaluation, try to use [𝑇1/2, 𝑇2/2], [𝑇1/3, 𝑇2/3], [2𝑇1, 2𝑇2], and [3𝑇1, 3𝑇2] for estimation. What are the resulting P-score values? Also, compare and discuss the results using the Fourier tempogram and the autocorrelation tempogram.


### Q3 (20%)
The window length is also an important factor in tempo estimation. Try to use 4s, 6s, 8s, 10s, 12s for both Fourier tempogram and the autocorrelation tempogram and compare the ALOTC of the eight genres in the Ballroom dataset and ISMIR2004 dataset.

## Task2: using dynamic programming for beat tracking
### Q4 (20%)
Using `librosa.beat.beat_track` to find the beat positions of a song. Evaluate this beat tracking algorithm on the Ballroom dataset. The F-score of beat tracking is defined as 𝐹 ≔ 2𝑃𝑅/(𝑃 + 𝑅), with Precision, P, and Recall, R, being computed from the number of correctly detected onsets TP, the number of false alarms FP, and the number of missed onsets FN, where 𝑃 ≔ 𝑇𝑃/(𝑇𝑃 + 𝐹𝑃) and 𝑅 ≔ 𝑇𝑃/(𝑇𝑃 + 𝐹𝑁). Here, a detected beat is considered a true positive when it is located within a tolerance of ±70 ms around the ground truth annotation. If there are more than one detected beat in this tolerance window, only one is counted as true positive, the others are counted as false alarms. If a detected onset is within the tolerance window of two annotations, then one true positive and one false negative will be counted. This process can be done with `mir_eval.beat`. Similarly, please compute the average F-scores of the eight genres in the Ballroom dataset and discuss the results.

### Q5 (20%)
Also use this algorithm on the SMC, JCS, and ASAP datasets. Compare and discuss the results together with the results of the Ballroom dataset. Could you explain the difference in performance?

## Task3: meter recognition (bonus)
### Q6 (20%)
The meter of a song can be 2-beats, 3-beats, 4-beats, 5-beats, 6-beats, 7-beats, or others. There might be multiple meters existing in a song (e.g., a 4-beats section followed by a 3-beats section). As a task combining both beat tracking and downbeat tracking, meter recognition is still a challenging task. Could you design an algorithm to detect the instantaneous meter of a song? Test the algorithm on the clips in the JCS dataset, and report frame-wise accuracy. The 1, 2, 3, 4, 5 after every line in the annotation file is the meter annotation. You can simply use `madmom.features.beats` (the state-of-the-art beat tracker) or combine other functions mentioned above.
