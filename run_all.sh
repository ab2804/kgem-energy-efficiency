#!/bin/bash

for d in "kinships" "WN18-RR" "FB15k-237"
do
    for m in "ComplEx" "ConvE" "DistMult" "ERMLP" "HolE" "KG2E" "MuRE" "NTN" "ProjE" "QuatE" "RESCAL" "RotatE" "SE" "SimplE" "TransD" "TransE" "TransH" "TransR" "TuckER" "UM"
    do
        for s in "42" "135" "468" "129" "124"
        do
        python3 run.py --dataset $d --model $m --seed $s --output-dir $1
        done
    done
done