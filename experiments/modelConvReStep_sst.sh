THEANO_FLAGS=device=cuda0 python model1L.py --model modelReConv --data ssa_v18242.pkl --w2v glove_ssa_v18242.pkl --embd 1 --embdUpdate 0 --numFltrRC 300 --fltrR 3 --numStep 1 --LRN 1 --trainN 3 --dropRate 0.25 --L2 1e-5 --log step1_sst_v18242.pkl
THEANO_FLAGS=device=cuda0 python model1L.py --model modelReConv --data ssa_v18242.pkl --w2v glove_ssa_v18242.pkl --embd 1 --embdUpdate 0 --numFltrRC 300 --fltrR 3 --numStep 2 --LRN 1 --trainN 3 --dropRate 0.25 --L2 1e-5 --log step2_sst_v18242.pkl
THEANO_FLAGS=device=cuda0 python model1L.py --model modelReConv --data ssa_v18242.pkl --w2v glove_ssa_v18242.pkl --embd 1 --embdUpdate 0 --numFltrRC 300 --fltrR 3 --numStep 3 --LRN 1 --trainN 3 --dropRate 0.25 --L2 1e-5 --log step3_sst_v18242.pkl