THEANO_FLAGS=device=gpu1 python sst_fine.py --logFname baseModel_sst_word_mincut1_v15938_d5.pkl --numFltrRC 300 --numStep 1 --dropRate 0.5 --L2 1e-5
THEANO_FLAGS=device=gpu1 python sst_fine.py --logFname baseModel_sst_word_mincut1_v15938_d6.pkl --numFltrRC 300 --numStep 1 --dropRate 0.6 --L2 1e-5
THEANO_FLAGS=device=gpu1 python sst_fine.py --logFname baseModel_sst_word_mincut1_v15938_d7.pkl --numFltrRC 300 --numStep 1 --dropRate 0.7 --L2 1e-5
THEANO_FLAGS=device=gpu1 python sst_fine.py --logFname baseModel_sst_word_mincut1_v15938_d8.pkl --numFltrRC 300 --numStep 1 --dropRate 0.8 --L2 1e-5