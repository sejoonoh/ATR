python src/ATR-2FT.py \
  --bert_model opt-350m \
  --mf_model_path  src/saved/UniSRec_Fine_Tuned_on_Amazon_Book.pth\
  --output result/amazon/2FT/ranking \
	--epochs 2 \
	--train_batch_size 16\
	--learning_rate 1e-5\
  --target_ratio 0.01\
  --coeff 1.0 \
  --gpu 0 \
