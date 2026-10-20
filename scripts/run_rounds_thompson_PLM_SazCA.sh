#!/bin/sh
cd "$(dirname "$0")/.."
uv run run_rounds.py \
	--rounds 100 \
	--duplicates 5 \
	--project_path outputs/N20_L5_E96_AA_thompson_OraclePLM_sigma10-4_SazCA \
	--continue-after-perfect-map \
	num_samples_prior=100_000 \
    num_samples_train=1_000 \
	posterior_approximation=mcmc \
	num_warmup_mcmc=2000 \
	mcmc_target_accept_prob=0.8 \
	mcmc_max_tree_depth=10 \
	mcmc_write_diagnostics=true \
	mcmc_diagnostics_include_ess=false \
	sequence_length=5 \
	num_experiments=96 \
	num_syntheses=96 \
	vocabulary=[A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y] \
	sigma=0.0001 \
    acquisition_function=thompson \
	model_type=categorical \
	num_components=20 \
	weights.mode=shifted_gamma \
	oracle.type=plm \
	oracle.wt_str=1MKKFILSILSLSIVSIAGEHAILQKNAEVHHWSYEGENGPENWAKLNPEYFWCNLKNQSPVDISDNYKVHAKLEKLHINYNKAVNPEIVNNGHTIQVNVL \
	do_visualization=false \
