-- NISQA_TEST_FOR Dataset
This dataset contains simulated distortions with different codecs, background noises, packet-loss, clipping. It also contains live conditions with WhatsApp, Zoom, and Discord (see for the condition list). The dataset was annotated with overall quality and speech quality dimension ratings in the crowd according to ITU-T P.808. If you use this dataset please cite following publication:

G. Mittag, B. Naderi, A. Chehadi, and S. Möller “NISQA -- A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with Crowdsourced Datasets,” 2021.
www.github.com/gabrielmittag/NISQA

-- Information:
Files: 240
Individual speakers: 80 (40 male/40 female)
Conditions: 60
Files per condition: 4
Votes per file: ~30 (50 before filtering crowd ratings)
Votes per condition: ~117 
Language: Australian English

NISQA_TEST_FOR_file.csv contains the ratings averaged per file 
NISQA_TEST_FOR_con.csv contains the ratings averaged per condition 

-- Source speech samples:
The source speech samples are taken from the "Forensic Voice Comparison Databases - Australian English: 500+ speakers" dataset [1] [2]. The database is available for non-commercial research and forensic casework. The conversation samples from this dataset were segmented into 6-12 seconds clips.

-- Noise files:
Noise files are taken from the DNS-Challenge [3] dataset (https://github.com/microsoft/DNS-Challenge), which in turn are taken from these three datasets:
Audioset: https://research.google.com/audioset/index.html; License: https://creativecommons.org/licenses/by/4.0/
Freesound: https://freesound.org/ Only files with CC0 licenses were selected; License: https://creativecommons.org/publicdomain/zero/1.0/
Demand: https://zenodo.org/record/1227121#.XRKKxYhKiUk; License: https://creativecommons.org/licenses/by-sa/3.0/deed.en_CA

-- License:
The dataset is provided under the original terms of the used source speech and noise samples. Therefore, the files may only be used for non-commerical research and forensic casework. The owner of the original forensic speech sample dataset gave us permission for making this dataset publicly available.

[1] G. Morrison, P. Rose, and C. Zhang, “Protocol for the collectionof databases of recordings for forensic-voice-comparison researchand practice,”Australian Journal of Forensic Sciences, vol. 44, pp.155 – 167, 2012.
[2] G.  Morrison,  C.  Zhang,  E.  Enzinger,  F.  Ochoa,  D.  Bleach,M. Johnson, B. Folkes, S. De Souza, N. Cummins, and D. Chow.(2015) Forensic database of voice recordings of 500+ australianenglish  speakers.  [Online].  Available:   http://databases.forensic-voice-comparison.net/
[3] C. K. A. Reddy, E. Beyrami, H. Dubey, V. Gopal, R. Cheng, R. Cutler, S. Matusevych, R. Aichner, A. Aazami, S. Braun,P. Rana, S. Srinivasan, and J. Gehrke, “The INTERSPEECH 2020deep noise suppression challenge: Datasets, subjective speechquality and testing framework,” 2020.

Contact:
Gabriel Mittag, Quality and Usability Lab, TU-Berlin, 2021
gabriel.mittag@gmail.com


