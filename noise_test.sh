#!/bin/bash

for noise_rate in $(seq 0 0.05 1); do
    python train.py --noise_rate $noise_rate
done