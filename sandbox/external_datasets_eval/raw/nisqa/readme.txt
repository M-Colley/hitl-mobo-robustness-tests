-- NISQA Speech Quality Corpus --
This folder contains 2 training, 2 validation and 4 test datasets:

- NISQA_TRAIN_SIM and NISQA_VAL_SIM:
Contains simulated distortions with speech samples from four different datasets. Divided into a training and a validation set.

- NISQA_TRAIN_LIVE and NISQA_VAL_LIVE:
Contains live phone and Skype recordings with Librivox audiobook samples. Divided into training and validation set.

- NISQA_TEST_LIVETALK:
Contains recordings of real phone and VoIP calls.

- NISQA_TEST_FOR:
Contains live and simulated conditions with speech samples from the forensic speech dataset. 

- NISQA_TEST_NSC *removed* (available for non-commercial research institutions):
Contains live and simulated conditions with speech samples from the NSC dataset. This datasets has been removed for license reasons.

- NISQA_TEST_P501:
Contains live and simulated conditions with speech samples from ITU-T Rec. P.501.

Please see the individual readme files in each of the dataset folders for more details about the datasets and the licenses. Generally, all of the files in this corpus can be used for non-commercial research purposes and some of the datasets can be also be used for commercial purposes, please see the individual license files for more details.

If you use any of these datasets please cite following publication:
G. Mittag, B. Naderi, A. Chehadi, and S. Möller “NISQA -- A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with Crowdsourced Datasets,” 2021.
www.github.com/gabrielmittag/NISQA

NISQA_corpus_file.csv contains the ratings averaged per file of all datasets, where the column 'db' indicates the dataset
NISQA_corpus_con.csv contains the ratings averaged per condition of all datasets, where the column 'db' indicates the dataset

If you work for a non-commercial research institution and are interested in the NISQA_TEST_NSC dataset, send an email to gabriel.mittag@gmail.com.

Contact:
Gabriel Mittag, Quality and Usability Lab, TU-Berlin, 2021
gabriel.mittag@gmail.com
