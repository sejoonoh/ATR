# Text Rewriting Attacks to Assess the Robustness of Recommender Systems

This repository contains anonymized code and datasets for ATR paper for RecSys 2023.  
[Link to Dataset](https://github.com/sejoonoh/Rewrite4Rec/blob/main/dataset.zip)  
The example command of ATR on the Amazon Book dataset and [HybridMF model](https://ieeexplore.ieee.org/document/8852443) is given as follows.

 1. Create `data`, `ckpt`, `result` directories.  
 2. Download the dataset with the above link and unzip them under `data` directory. Put the pre-trained BERT model (e.g., `all-MiniLM-L6-v2`) into `ckpt` directory.
 3. Execute `data_generate.sh` and `keyword_extraction.sh` to generate training data and keywords for **POINTER** model.  
 4. Execute `fine_tune.sh` for the Phase 1 training of **ATR**.  
 5. Run `python src/hybridmf.py` to train the recommendation model on the Amazon Book dataset.  
 6. Execute `ATR.sh` for the Phase 2 training of **ATR**.  
 7. Check the ranking performance and rewritten texts created by **ATR**.  

# Abstract 
Large language models have demonstrated powerful performance in various language tasks such as text generation or vectorizing given
text. Using such language models, modern recommender systems have incorporated rich textual features such as titles and descriptions
into their model training and predictions. While product descriptions have been employed as key input features to text-aware
recommenders, the robustness of existing recommenders against manipulating product descriptions has not been investigated yet. If
attackers can easily promote the visibility of their target items via description rewriting, the credibility of an online platform (e.g.,
e-commerce) can be critically damaged. Even worse, the rewriting attack is hard to detect since item descriptions can be frequently
updated by the item owner. In this paper, we propose an adversarial text rewriting framework: ATR which creates ranking-optimized
descriptions of target items to promote their ranking across all users. ATR is trained in a 2-phase manner, where in the first phase, we
learn the dataset-specific context, and in the second phase, we perform adversarial fine-tuning for promoting target items’ ranks while
preserving the semantic meaning of their descriptions. Extensive experiments show that existing recommenders show vulnerability
against the proposed text rewriting attack. Notably, ATR can generate ranking-optimized descriptions of target items even under the
black-box setup (i.e., no access to the recommender and training data). Human evaluation aimed to discern the quality of rewritten
descriptions demonstrates superior performance of ATR over a competitive text generation baseline.

