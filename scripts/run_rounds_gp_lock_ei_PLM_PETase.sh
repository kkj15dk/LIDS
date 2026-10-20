#!/bin/sh
cd "$(dirname "$0")/.."
uv run run_rounds.py \
	--rounds 100 \
	--duplicates 5 \
	--project_path outputs/L5_E96_AA_lock_ei_OraclePLM_sigma10-4_PETase \
	--continue-after-perfect-map \
	sequence_length=5 \
	num_experiments=96 \
	vocabulary=[A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y] \
	sigma=0.0001 \
	acquisition_function=gp_acqf \
	gp.kernel=lock \
	gp.acquisition=log_expected_improvement \
	gp.num_restarts=20 \
	gp.raw_samples=4096 \
	oracle.type=plm \
	oracle.wt_str=1MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPR
