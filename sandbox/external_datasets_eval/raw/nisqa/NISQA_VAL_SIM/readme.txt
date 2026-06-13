-- NISQA_TRAIN_SIM and NISQA_VAL_SIM Datasets
These datasets contain a large variety of different simulated distortions:
- Additive white Gaussian noise
- Signal correlated MNRU noise.
- Randomly sampled noise clips taken from the DNS-Challenge dataset 
- Lowpass / highpass / bandpass / arbitrary filter with random cutoff frequencies
- Amplitdude clipping
- Speech level changes\item
- Codecs in all available bitrade modes: AMR-NB, AMR-NB, G.711, G.722, EVS, Opus
- Codec tandem and triple tandem
- Packet-loss conditions with random and bursty patterns.
- Combinations of the different distortions
The number of file equals the number of conditions in the datasets because each file was processed with a different condition. The original distortion parameters used to create the dataset are stored in the per-file csv-file. The resulting speech files were then split into a training and a validation set. 

If you use this dataset please cite following publication:
G. Mittag, B. Naderi, A. Chehadi, and S. Möller “NISQA -- A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with Crowdsourced Datasets,” 2021.
www.github.com/gabrielmittag/NISQA

-- Information:

NISQA_TRAIN_SIM
Files: 10,000
Individual speakers: 2,322

NISQA_VAL_SIM
Files: 2,500
Individual speakers: 938

Files per condition: 1
Votes per file: ~5
Language: English

NISQA_TRAIN_SIM_file.csv/NISQA_VAL_SIM_file.csv contains the ratings averaged per file and distortion parameters 
NISQA_TRAIN_SIM_con.csv/NISQA_VAL_SIM_con.csv contains the ratings averaged per condition 

-- Source speech samples:
The source speech samples are taken from four different datasets. The source of each speech file is listed in the 'source' column of the per-file csv-file. The samples were segmented into 6-12 seconds clips.

1) The Librivox audiobook clips of the "DNS-Challenge" [1] dataset. The Librivox audiobooks are part of the public domain (https://librivox.org/; License: https://librivox.org/pages/public-domain/). 

2) TSP speech database. The files are covered by a permissive Simplified BSD licence (see tsp_license.txt)

3) Crowdsourced high-quality UK and Ireland English Dialect speech data set [3]. The dataset is covered by a "Attribution-ShareAlike 4.0 International" license (see ukire_license.txt).

4) AusTalk [4], from which 6-12 clips of the interview task were extracted. The AusTalk license terms can be found in AusTalk_Content_Licence_Terms.pdf. The owners of the AusTalk corpus gave us permission for making this dataset publicly available.

-- Noise files:
Noise files are taken from the DNS-Challenge [1] dataset (https://github.com/microsoft/DNS-Challenge), which in turn are taken from these three datasets:
Audioset: https://research.google.com/audioset/index.html; License: https://creativecommons.org/licenses/by/4.0/
Freesound: https://freesound.org/ Only files with CC0 licenses were selected; License: https://creativecommons.org/publicdomain/zero/1.0/
Demand: https://zenodo.org/record/1227121#.XRKKxYhKiUk; License: https://creativecommons.org/licenses/by-sa/3.0/deed.en_CA

-- License:
The dataset is provided under the original terms of the used source speech and noise samples.

[1] C. K. A. Reddy, E. Beyrami, H. Dubey, V. Gopal, R. Cheng, R. Cutler, S. Matusevych, R. Aichner, A. Aazami, S. Braun,P. Rana, S. Srinivasan, and J. Gehrke, “The INTERSPEECH 2020deep noise suppression challenge: Datasets, subjective speechquality and testing framework,” 2020.
[2] P. Kabal, “TSP speech database,” McGill University, Quebec, Canada, Tech. Rep. Database Version 1.0, 2002.
[3] I. Demirsahin, O. Kjartansson, A. Gutkin, and C. Rivera, “Open-source multi-speaker corpora of the english accents in the britishisles,” inProc. 12th Language Resources and Evaluation Confer-ence (LREC), 2020.
[4] D. Burnham, D. Estival, S. Fazio, J. Viethen, F. Cox, R. Dale,S. Cassidy, J. Epps, R. Togneri, M. Wagner, Y. Kinoshita, R. G̈ocke, J. Arciuli, M. Onslow, T. W. Lewis, A. Butcher, andJ. Hajek, “Building an audio-visual corpus of Australian English:Large corpus collection with an economical portable and replica-ble black box,” inProc. Interspeech 2011, 2011

Contact:
Gabriel Mittag, Quality and Usability Lab, TU-Berlin, 2021
gabriel.mittag@gmail.com


