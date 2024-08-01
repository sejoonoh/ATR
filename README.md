# Adversarial Text Rewriting for Text-aware Recommender Systems (ACM CIKM 2024)

Overview
---------------
**Adversarial Text Rewriting for Text-aware Recommender Systems**  
[Sejoon Oh](https://sejoonoh.github.io/), [Gaurav Verma](https://gaurav22verma.github.io/), and [Srijan Kumar](https://www.cc.gatech.edu/~srijan/)  
*[ACM International Conference on Information and Knowledge Management (CIKM)](https://www.cikm2024.org/), 2024*  

This repository contains code and datasets for ATR paper.  
Datasets used in the paper are available here [Link to Dataset](https://drive.google.com/file/d/1mp8NEOHVYC1q-CESFin4cUuO-0-N2Qkz/view?usp=sharing)  

Usage
---------------
The example command of ATR-2FT+OPT-350M on the Amazon Book dataset and [UniSRec model](https://github.com/RUCAIBox/UniSRec) is given as follows.

 1. Download [processed Amazon Book dataset](https://drive.google.com/file/d/1_k10CKv0VsRON3L_PcTBhoQ09EOer70q/view?usp=drive_link) and unzip them under `src/dataset/downstream/amazon_book` directory.
 2. Execute `python src/opt_fine_tune.py` to perform Phase-1 fine-tuning on Amazon book dataset. It will download the opt-350m model and create the fine-tuned model `opt-350m`.
 3. Download a pre-trained text-aware recommender model [UniSRec-Fined-Tuned-on-Amazon-Book.pth](https://drive.google.com/file/d/1d7fLYgs0ZTAdCwK6wpszWTIm7QciuB3U/view?usp=drive_link) and save the model under `src/saved` directory.
 4. In `ATR-2FT.sh', find -p argument. Replace the path with the exact path of a trained recommender obtained in the above step. 
 5. Execute `ATR-2FT.sh` for the Phase-2 fine-tuning.  
 6. Check the ranking performance and rewritten texts created by **ATR-2FT** in the `result/amazon/2FT` directory.  


The example command of ATR-ICL+LLama-2-Chat-7B on the Amazon Book dataset and [UniSRec model](https://github.com/RUCAIBox/UniSRec) is given as follows.

 1. Download [processed Amazon Book dataset](https://drive.google.com/file/d/1_k10CKv0VsRON3L_PcTBhoQ09EOer70q/view?usp=drive_link) and unzip them under `src/dataset/downstream/amazon_book` directory.
 2. Download the [LLama-2 model](https://github.com/facebookresearch/llama) and put them under current directory. **Don't forget to build the Llama model from source.**
 3. Download a pre-trained text-aware recommender model [UniSRec-Fined-Tuned-on-Amazon-Book.pth](https://drive.google.com/file/d/1d7fLYgs0ZTAdCwK6wpszWTIm7QciuB3U/view?usp=drive_link) and save the model under `src/saved` directory.
 4. In `ATR-ICL.sh', find -p argument. Replace the path with the exact path of a trained recommender obtained in the above step. 
 5. Execute `ATR-ICL.sh` for the in-context learning.   
 6. Check the ranking performance and rewritten texts created by **ATR-ICL** in the `result/amazon/ICL/` directory.  

**For other recommenders and datasets, you will need to preprocess dataset, pre-train the recommender with the dataset, and adjust existing ATR code to the recommender and dataset.**

# Abstract 
Text-aware recommender systems incorporate rich textual features, such as titles and descriptions, to generate item recommendations for users. The use of textual features helps mitigate cold-start problems, and thus, such recommender systems have attracted increased attention. However, we argue that the dependency on item descriptions makes the recommender system vulnerable to manipulation by adversarial sellers on e-commerce platforms. In this paper, we explore the possibility of such manipulation by proposing a new
text rewriting framework to attack text-aware recommender systems. We show that the rewriting attack can be exploited by sellers to unfairly uprank their products, even though the adversarially rewritten descriptions are perceived as realistic by human evaluators. Methodologically, we investigate two different variations to carry out text rewriting attacks: (1) two-phase fine-tuning for greater attack performance, and (2) in-context learning for higher text rewriting quality. Experiments spanning 3 different datasets
and 4 existing approaches demonstrate that recommender systems exhibit vulnerability against the proposed text rewriting attack. Our work adds to the existing literature around the robustness
of recommender systems, while highlighting a new dimension of vulnerability in the age of large-scale automated text generation.
