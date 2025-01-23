# SMIA
source code of the paper "Social Relation-Level Privacy Risks and Preservation in Social Recommender Systems"

## Abstract
The integration of social information into recommender systems (RSs) has gained significant popularity for enhancing recommendation performance and user experience. However, this practice introduces substantial privacy risks, particularly concerning the leakage of sensitive social relationships. While prior research has primarily focused on user-level and interaction-level privacy risks, the vulnerabilities associated with social relation-level privacy leakage remain largely unexplored. To address this gap, we investigate social privacy risks through the lens of membership inference attacks (MIA). Nevertheless, two key challenges arise: (1) the adversary can only access sparse recommendation results, which contain indirect and limited information about social relationships, and (2) extracting socially relevant preferences from these results is inherently difficult. To tackle these challenges, we propose a dual-branch learning framework that disentangles user preferences and extracts social influence and homophily patterns to uncover social relationship signals. Extensive experiments on real-world datasets demonstrate that both social and general RSs are highly vulnerable to such attacks, highlighting the urgent need for robust privacy protection mechanisms. To mitigate these risks, we introduce a socially adversarial learning defense mechanism (\OURDF{}) that selectively obscures sensitive social information in user representations during training, effectively reducing privacy leakage. We further evaluate the effectiveness of our defense and discuss future directions for developing privacy-preserving mechanisms in social recommender systems.


## Start
### 1. data preparing
- create a folder for datasets at `../raw dataset/`.
- download ciao or flickr at the dataset path, and use `data_preprocessing.py` to preprocessing the raw data.

### 2. train target model
- use `main.py --model_name [TARGET_NAME]` to train a target social recommender, [TARGET_NAME]={'DESIGN, DiffNet}.
- use `pp_main.py --xxx` to train a target model with our SAL defense mechanism.
- use `pp_main_baseline.py` to train a target model with ER or DP-SGD defense mechanism.

### 3. perform MIA attacks
- directly run `./mia_from_trained_model.sh` to perform our attacks.

## Citation 
If you use this code, please consider to cite the following paper:

```

```
