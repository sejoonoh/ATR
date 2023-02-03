# Rewrite4Rec: Rewriting Product Descriptions to Promote Ranking in Multimodal Recommender Systems

This repository contains anonymized code and datasets for Rewrite4Rec paper for KDD 2023.  
[Link to Dataset](https://github.com/sejoonoh/Rewrite4Rec/blob/main/dataset.zip)  
The example command of Rewrite4Rec on the Amazon Book dataset and [HybridMF model](https://ieeexplore.ieee.org/document/8852443) is given as follows.

 1. Create `data`, `ckpt`, `result` directories.  
 2. Download the dataset with the above link and unzip them under `data` directory. Put the pre-trained BERT model (e.g., `all-MiniLM-L6-v2`) into `ckpt` directory.
 3. Execute `data_generate.sh` and `keyword_extraction.sh` to generate training data and keywords for **POINTER** model.  
 4. Execute `fine_tune.sh` for the Phase 1 training of **Rewrite4Rec**.  
 5. Run `python src/hybridmf.py` to train the recommendation model on the Amazon Book dataset.  
 6. Execute `rewrite4rec.sh` for the Phase 2 training of **Rewrite4Rec**.  
 7. Check the ranking performance and rewritten texts created by **Rewrite4Rec**.  

# Abstract 
Multimodal recommender systems address data sparsity and coldstart problems by using additional modalities such as text descriptions of items. While product descriptions have been employed as key input features to multimodal recommenders, such descriptions are not optimized with respect to the recommender system to improve the product’s visibility. In this paper, we propose Rewrite4Rec which rewrites text descriptions of target products to promote their ranking across all users to increase the product’s visibility. Rewrite4Rec is trained in a 2-phase manner, where in the first phase, we learn the dataset-specific context, and in the second phase, we perform multi-objective fine-tuning for promoting target products while preserving the semantic meaning of their descriptions. Experiments show that Rewrite4Rec generates rankingoptimized descriptions which boost target products’ ranks across all users. Human evaluation aimed to discern the quality of rewritten descriptions demonstrates superior performance of Rewrite4Rec over a competitive baseline. Optimized product descriptions can be
used by online and e-commerce platforms as a service to product sellers to increase the visibility and sales of their products.
