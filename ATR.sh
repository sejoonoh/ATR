python src/ATR.py \
	--pregenerated_data ./data/amazon_processed \
  --bert_model ckpt/amazon_fine_tuned_10 \
  --mf_model_path ckpt/amazon_HMF \
  --output rewrite4rec_result \
	--epochs 2 \
	--train_batch_size 32 \
	--learning_rate 1e-5\
  --target_ratio 0.01\
  --coeff 1.0 \
  --mode ours \
	--gpu 0