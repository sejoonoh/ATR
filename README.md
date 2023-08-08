# Product Description Rewriting Attack on Text-Aware Recommender Systems

This repository contains anonymized code and datasets for ATR paper.  
Datasets used in the paper are available here [Link to Dataset](https://drive.google.com/file/d/1mp8NEOHVYC1q-CESFin4cUuO-0-N2Qkz/view?usp=sharing)  
The example command of ATR on the Amazon Book dataset and [HybridMF model](https://ieeexplore.ieee.org/document/8852443) is given as follows.

 1. Create `data`, `ckpt`, `result` directories.  
 2. Download the dataset with the above link and unzip them under `data` directory. Put the pre-trained BERT model (e.g., `all-MiniLM-L6-v2`) into `ckpt` directory.
 3. Execute `data_generate.sh` and `keyword_extraction.sh` to generate training data and keywords for **POINTER** model.  
 4. Execute `fine_tune.sh` for the Phase 1 training of **ATR**.  
 5. Run `python src/hybridmf.py` to train the recommendation model on the Amazon Book dataset.  
 6. Execute `ATR.sh` for the Phase 2 training of **ATR**.  
 7. Check the ranking performance and rewritten texts created by **ATR**.  

The code for ATR with the OPT text generation model will be added later.

# Abstract 
Text-aware recommender systems incorporate rich textual features, such as titles and descriptions, to generate item recommendations
for users. The use of textual features helps mitigate cold-start problems and thus, such recommender systems have attracted increased
attention. However, we argue that the dependency on item descriptions makes the recommender system vulnerable to manipulation
by adversarial sellers. In this paper, we explore the possibility of such manipulation by proposing the first text rewriting framework to
attack text-aware recommender systems. Specifically, ATR creates ranking-optimized descriptions of target items to promote their
ranking across all users. ATR works in two phases – in the first phase, the text generation model is optimized to learn the textual
properties across the dataset, inducing domain adaptation to the target descriptions. In the second phase, the model adversarially
rewrites the target items’ descriptions to increase the items’ predicted ratings across all users while preserving fluency and semantic
meaning. Experiments demonstrate that three existing recommender systems exhibit vulnerability against the proposed text rewriting
attack in MovieLens and two Amazon datasets. Human evaluation aimed to discern the quality of descriptions demonstrates superior
quality of ATR-written descriptions over a GPT-2 baseline. Our work highlights the need to create robust recommender systems.
