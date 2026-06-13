-- NISQA_TRAIN_LIVE and NISQA_VAL_LIVE Datasets
In this dataset live telephone and Skype calls were conducted, where clean speech files from the DNS-Challenge dataset were played back via a loudspeaker. The speech signal was played back from a laptop on a ``Fostex PM0.4n studio monitor''. Two types of calls were conducted: a fixed-line to mobile phone call, and a Skype call (laptop to laptop). For the first type, a call from a fixed-line VoIP phone (Cisco IP Phone 9790) within the Q&U Lab to a state-of-the-art smartphone (Google Pixel 3) was conducted. The VoIP handset was placed in front of the monitor to capture the speech signal acoustically. The received signal was then stored directly on the Google Pixel 3. The Skype call was conducted between two laptops, where the sending laptop was placed next to the monitor to capture the played back speech signal. The transmitted speech signal was then stored on the receiving laptop. During the call, several real distortions were created in the recording room, such as open window, changing volume and angle of monitor, typing on keyboard. The resulting speech files were then split into a training and a validation set. The same speakers that were used in the simulated training dataset were again used for the live training dataset and vice versa for the validation set. However, new sentences of these speakers that are not contained in the training dataset were used.

If you use this dataset please cite following publication:
G. Mittag, B. Naderi, A. Chehadi, and S. Möller “NISQA -- A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with Crowdsourced Datasets,” 2021.
www.github.com/gabrielmittag/NISQA

-- Information:

NISQA_TRAIN_LIVE
Files: 1020
Individual speakers: 486

NISQA_VAL_LIVE
Files: 200
Individual speakers: 102

Files per condition: 1
Votes per file: ~5
Votes per condition: ~5
Language: English

NISQA_TRAIN_LIVE_file.csv/NISQA_VAL_LIVE_file.csv contains the ratings averaged per file 
NISQA_TRAIN_LIVE_file.csv/NISQA_VAL_LIVE_con.csv contains the ratings averaged per condition 

-- Source speech samples:
The source speech samples are taken from the Librivox audiobook clips of the "DNS-Challenge" [1] dataset. The Librivox audiobooks are part of the public domain (https://librivox.org/; License: https://librivox.org/pages/public-domain/). The samples from this dataset were segmented into 6-12 seconds clips. 

-- License:
The dataset is provided under the original terms of the used source speech samples. Therefore, they may be used for commercial and/or non-commerical research.

[1] C. K. A. Reddy, E. Beyrami, H. Dubey, V. Gopal, R. Cheng, R. Cutler, S. Matusevych, R. Aichner, A. Aazami, S. Braun,P. Rana, S. Srinivasan, and J. Gehrke, “The INTERSPEECH 2020deep noise suppression challenge: Datasets, subjective speechquality and testing framework,” 2020.

Contact:
Gabriel Mittag, Quality and Usability Lab, TU-Berlin, 2021
gabriel.mittag@gmail.com


