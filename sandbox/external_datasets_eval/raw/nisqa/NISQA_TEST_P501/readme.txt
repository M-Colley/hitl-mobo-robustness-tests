-- NISQA_TEST_P501 Dataset
This dataset contains simulated distortions with different codecs, background noises, packet-loss, clipping. It also contains live conditions with Skype, Zoom, WhatsApp, and mobile network recordings (see condition list). The dataset was annotated with overall quality and speech quality dimension ratings in the crowd according to ITU-T P.808. If you use this dataset please cite following publication:

G. Mittag, B. Naderi, A. Chehadi, and S. Möller “NISQA -- A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with Crowdsourced Datasets,” 2021.
www.github.com/gabrielmittag/NISQA

-- Information:
Files: 240
Individual speakers: 4 (2 male/2 female)
Conditions: 60
Files per condition: 4
Votes per file: ~28 (50 before filtering crowd ratings)
Votes per condition: ~113
Language: British English

NISQA_TEST_P501_file.csv contains the ratings averaged per file 
NISQA_TEST_P501_con.csv contains the ratings averaged per condition 

-- Source speech samples:
The source speech samples are taken from Annex C of the "ITU-T P.501" dataset [1]. See "itut_p501_license.txt" for the dataset license. The samples from this dataset were segmented into 6-12 seconds clips. 

-- Noise files:
Noise files are taken from the DNS-Challenge [2] dataset (https://github.com/microsoft/DNS-Challenge), which in turn are taken from these three datasets:
Audioset: https://research.google.com/audioset/index.html; License: https://creativecommons.org/licenses/by/4.0/
Freesound: https://freesound.org/ Only files with CC0 licenses were selected; License: https://creativecommons.org/publicdomain/zero/1.0/
Demand: https://zenodo.org/record/1227121#.XRKKxYhKiUk; License: https://creativecommons.org/licenses/by-sa/3.0/deed.en_CA

-- License:
The dataset is provided under the original terms of the used source speech and noise samples. The ITU is the copyright owner of the original test signals. The use of the P.501 speech samples for this dataset is made under permission by ITU. The speech quality dataset is made available to the public for free and shall not be included in any commercial product/service. 

[1] ITU-T Rec. P.501: Test signals for use in telephony and other speech-based applications, 2020.
[2] C. K. A. Reddy, E. Beyrami, H. Dubey, V. Gopal, R. Cheng, R. Cutler, S. Matusevych, R. Aichner, A. Aazami, S. Braun,P. Rana, S. Srinivasan, and J. Gehrke, “The INTERSPEECH 2020deep noise suppression challenge: Datasets, subjective speechquality and testing framework,” 2020.

Contact:
Gabriel Mittag, Quality and Usability Lab, TU-Berlin, 2021
gabriel.mittag@gmail.com


