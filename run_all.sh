#!/bin/bash

for d in "countries"
do
    for m in "TransE"
    do
        for s in "42" "135" "468"
        do
        python run.py --dataset $d --model $m --seed $s --output-dir $1
        done
    done
done