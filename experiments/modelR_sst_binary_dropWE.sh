THEANO_FLAGS=device=cuda1 python model.py --model modelR --data ssa_binary_v18242.pkl --w2v glove_ssa_binary_v18242.pkl --embd 1 --embdUpdate 0 --numFltrRC 300 --fltrR 5 --numStep 3 --LRN 1 --outUnit 20 --dropWE 1 --dropRate 0.25 --L2 1e-5 --log ssa_binary_v18242_T3_hidd20_glove_dropwe.pkl
THEANO_FLAGS=device=cuda1 python model.py --model modelR --data ssa_binary_v18242.pkl --w2v glove_ssa_binary_v18242.pkl --embd 1 --embdUpdate 0 --numFltrRC 300 --fltrR 5 --numStep 2 --LRN 1 --outUnit 20 --dropWE 1 --dropRate 0.25 --L2 1e-5 --log ssa_binary_v18242_T2_hidd20_glove_dropwe.pkl
THEANO_FLAGS=device=cuda1 python model.py --model modelR --data ssa_binary_v18242.pkl --w2v glove_ssa_binary_v18242.pkl --embd 1 --embdUpdate 0 --numFltrRC 300 --fltrR 5 --numStep 1 --LRN 1 --outUnit 20 --dropWE 1 --dropRate 0.25 --L2 1e-5 --log ssa_binary_v18242_T3_hidd20_glove_dropwe.pkl
#