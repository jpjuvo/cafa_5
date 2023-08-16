#!/bin/bash

declare -a arr=("mf" "bp" "cc")

for mode in 0 1 2 
do
    for ont in "${arr[@]}"
    do
        echo Inference mode "$mode" ont "$ont"
        python src/dgz_inference.py \
            -m $mode \
            -o "$ont"
    done
done