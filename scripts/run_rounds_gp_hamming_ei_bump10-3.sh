#!/bin/sh
cd "$(dirname "$0")/.."
uv run run_rounds.py \
	--rounds 100 \
	--duplicates 10 \
	--project_path outputs/N8_L8_E12_DNA_gp_hamming_ei_bump_sigma10-3 \
	sequence_length=8 \
	num_experiments=12 \
	vocabulary=[A,T,C,G] \
	sigma=0.001 \
	acquisition_function=gp_acqf \
	gp.kernel=hamming \
	gp.acquisition=log_expected_improvement \
	gp.num_restarts=20 \
	gp.raw_samples=4096 \
	oracle.type=bump \
	do_visualization=false
