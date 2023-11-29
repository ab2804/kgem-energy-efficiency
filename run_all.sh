#!/bin/bash

for d in "kinships"
do
    for m in "ComplEx" "ConvE" "ConvKB" "DistMult" "ERMLP" "HolE" "KG2E" "MuRE" "NTN" "ProjE" "QuatE" "RESCAL" "RotatE" "SE" "SimplE" "TransD" "TransE" "TransH" "TransR" "TuckER" "UM"
    do
        for s in "42" "135" "468"
        do
        python run2.py --dataset $d --model $m --seed $s --output-dir $1
        done
    done
done