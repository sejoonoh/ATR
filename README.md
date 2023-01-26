# Rewrite4Rec: Text Rewriting for Target Item Promotions in Multi-modal Recommender Systems

This repository contains anonymized code and datasets for Rewrite4Rec paper for KDD 2023.  
[Link to Dataset](https://drive.google.com/file/d/1AvpAqBQvr0BduHDPVqssM5KIyIO_hHa4/view?usp=sharing)  
The example command of Rewrite4Rec on the Amazon Book dataset and [HybridMF model](https://ieeexplore.ieee.org/document/8852443) is given as follows.

 1. Download the dataset with the above link and unzip them under `data` directory (create `data`, `ckpt`, `result` directories as well).  
 2. Execute `data_generate.sh` and `keyword_extraction.sh` to generate training data and keywords for **POINTER** model.  
 3. Execute `fine_tune.sh` for the Phase 1 training of **Rewrite4Rec**.  
 4. Run `python src/hybridmf.py` to train the recommendation model on the Amazon Book dataset.  
 5. Execute `rewrite4rec.sh` for the Phase 2 training of **Rewrite4Rec**.  
 6. Check the ranking performance and rewritten texts created by **Rewrite4Rec**.  

# Abstract 
Multi-modal recommender systems address data sparsity and coldstart problems by using additional modalities such as text descriptions of items. While item descriptions have been employed as key input features to multi-modal recommenders, there is no existing method for finding optimal descriptions that maximize the ranking of target items. Such a technique can be used as a practical tool for content creators to promote their items on online platforms. In this paper, we propose Rewrite4Rec which rewrites text descriptions of target items to promote their ranking across all users. Rewrite4Rec is trained in a 2-phase manner, where we learn dataset-specific context in the first phase, and we perform multi-objective learning for target item promotions while preserving the semantic meaning of the text in the second phase. Our comprehensive experiments show that Rewrite4Rec generates ranking-optimized descriptions which boost target items’ ranks for all users. We also test whether such descriptions are fluent and meaningful via human evaluations.
