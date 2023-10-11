# Product Description Rewriting Attack on Text-Aware Recommender Systems

This repository contains anonymized code and datasets for ATR paper.  
Datasets used in the paper are available here [Link to Dataset](https://drive.google.com/file/d/1mp8NEOHVYC1q-CESFin4cUuO-0-N2Qkz/view?usp=sharing)  
The example command of ATR-2FT+OPT-350M on the Amazon Book dataset and [UniSRec model](https://github.com/RUCAIBox/UniSRec) is given as follows.

 1. Download the dataset with the above link and unzip them under `src/dataset/downstream/amazon_book` directory.
 2. Execute `python src/opt_fine_tune.py` to perform Phase-1 fine-tuning on Amazon book dataset. It will create the fine-tuned model `opt-350m`.
 3. Run `python src/pretrain.py -d amazon_book` to train the recommendation model on the Amazon Book dataset. It will save a trained recommender under `src/saved` directory.
 4. In `ATR-2FT.sh', find -p argument. Replace the path with the exact path of a trained recommender obtained in the above step. 
 5. Execute `ATR-2FT.sh` for the Phase-2 fine-tuning.  
 6. Check the ranking performance and rewritten texts created by **ATR-2FT** in the `result/amazon/2FT` directory.  


The example command of ATR-ICL+LLama-2-Chat-7B on the Amazon Book dataset and [UniSRec model](https://github.com/RUCAIBox/UniSRec) is given as follows.

 1. Download the dataset with the above link and unzip them under `src/dataset/downstream/amazon_book` directory.
 2. Download the [LLama-2 model](https://github.com/facebookresearch/llama) and put them under `src` directory.
 3. Run `python src/pretrain.py -d amazon_book` to train the recommendation model on the Amazon Book dataset. It will save a trained recommender under `src/saved` directory.
 4. In `ATR-ICL.sh', find -p argument. Replace the path with the exact path of a trained recommender obtained in the above step. 
 5. Execute `ATR-ICL.sh` for the in-context learning.   
 6. Check the ranking performance and rewritten texts created by **ATR-ICL** in the `result/amazon/ICL/` directory.  


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
