python src/training.py \
	--pregenerated_data ./data/amazon_processed \
	--bert_model ckpt/all-MiniLM-L6-v2 \
	--output_dir ckpt/amazon_fine_tuned_10 \
	--epochs 10 \
	--train_batch_size 16 \
	--output_step 100000 \
	--learning_rate 3e-5\
	--gpu 0
