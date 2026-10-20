#!/bin/sh
cd "$(dirname "$0")/.."
uv run run_rounds.py \
	--rounds 100 \
	--duplicates 10 \
	--project_path outputs/N8_L8_E12_DNA_thompson_bump_sigma10-2 \
	num_samples_prior=100_000 \
    num_samples_train=1_000 \
	posterior_approximation=mcmc \
	sequence_length=8 \
	num_experiments=12 \
	num_syntheses=12 \
	vocabulary=[A,T,C,G] \
	sigma=0.01 \
    acquisition_function=thompson \
	model_type=categorical \
	num_components=8 \
	weights.mode=fixed \
	oracle.type=bump \
	do_visualization=false \