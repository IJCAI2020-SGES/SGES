#!/bin/bash

# Run HalfCheetah-v2
for ((i=2016; i<=2020; i++))
do
	python sges.py \
	--env "HalfCheetah-v2" \
	--lr 0.003 \
	--noise_stddev 0.05 \
	--pop_size 16 \
	--elite_size 12 \
	--k 12 \
	--warmup 24 \
	--epochs 600 \
	--num_workers 16 \
	--seed $i
done

# Run Ant-v2
for ((i=2016; i<=2020; i++))
do
	python sges.py \
	--env "Ant-v2"
	--lr 0.001 \
	--noise_stddev 0.02 \
	--pop_size 60 \
	--elite_size 20 \
	--k 20 \
	--warmup 40 \
	--epochs 600 \
	--num_workers 16 \
	--seed $i
done

# Run Humanoid-v2
for ((i=2016; i<=2020; i++))
do
	python sges.py \
	--env "Humanoid-v2" \
	--lr 0.075 \
	--noise_stddev 0.0075 \
	--pop_size 230 \
	--elite_size 230 \
	--k 100 \
	--warmup 100 \
	--epochs 800 \
	--num_workers 23 \
	--seed $i
done

# Run Swimmer-v2
for ((i=2016; i<=2020; i++))
do
	python sges_swimmer-v2.py \
	--env "Swimmer-v2" \
	--lr 0.005 \
	--noise_stddev 0.15 \
	--pop_size 1 \
	--elite_size 1 \
	--k 1 \
	--warmup 4 \
	--epochs 400 \
	--num_workers 16 \
	--seed $i
done



