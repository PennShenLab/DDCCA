# DDCCA

This repository holds the official code for the paper below:
> Shu Yang #, Jae Young Baik #, Zhuoping Zhou, Jingxuan Bao, Austin Wang, Shizhuo Mu, Yanbo Feng, Zixuan Wen, Bojian Hou, Joseph Lee, Tianqi Shang, Rongguang Wang, Junhao Wen, Heng Huang, Andrew J. Saykin, Paul M. Thompson, Christos Davatzikos, Li Shen. ***DDCCA: 'Diet' deep canonical correlation analysis for high-dimensional genetics study of brain imaging phenotypes in Alzheimer's disease.*** ([link to arxiv preprint](https://arxiv.org/submit/6033116)) *#: Equal contribution*

## 🦸‍ Abstract
Understanding the relationship between genetic variations and brain imaging phenotypes is an important issue in Alzheimer's disease (AD) research. As an alternative to GWAS univariate analyses, canonical correlation analysis (CCA) and its deep learning extension (DCCA) are widely used to identify associations between multiple genetic variants such as SNPs and multiple imaging traits such as brain regions of interest (ROIs) from PET or MRI. However, with the recent availability of numerous genetic variants from genotyping and sequencing data for AD, these approaches often suffer from severe overfitting when dealing with 'fat' genetics data, i.e. large numbers of variants with much smaller numbers of samples. Here, we propose to tackle the challenge by integrating a simple and efficient model parameterization approach into DCCA to handle high dimensional SNP data in AD imaging-genetics study. The proposed method, 'Diet' deep canonical correlation analysis (DDCCA), was applied to nine datasets derived from 955 subjects in the ADNI data. Each dataset contains 68 FreeSurfer cortical ROIs from the florbetapir (AV45) PET imaging and varied numbers of variants from 810 to 11,938 based on different significance thresholds derived from previous studies. As a result, DDCCA outperformed a set of existing CCA methods representing different regularization strategies, when evaluated on test correlations. Moreover, the detected correlations from DDCCA were better attributed by meaningful genetic variants than previous CCA methods. The study supplies a novel and effective tool to study the genetic basis of AD imaging phenotypes for future analyses.

## 📝 Install
DDCCA is implemented with PyTorch v2.0.1 and Python3.9 
To install the related packages, use
```bash
pip install -r requirements.txt
```

## 🔨 Usage
The main program entrance is the `run_DDCCA.py` script, which will perform train-test split based on the random-number-generator seed the user passed through the --rng_seed flag:
```cmd
python run_DDCCA.py --in_imgfile 'DATA/image.csv' --in_snpfile 'DATA/genetic.csv' --rng_seed 0 --which_model DDCCA --optuna_num_trials 50 --batch_size 32 --nn_out_size 25
```
        
### 🤝 Acknowledgements
This study was funded by NIH U01 AG068057 and NIH U01 AG066833. The complete [ADNI](http://adni.loni.usc.edu/) Acknowledgement is available at http://adni.loni.usc.edu/wpcontent/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf.

### 📭 Maintainers
- Shu Yang (shu.yang@pennmedicine.upenn.edu)
- Jae Young Baik (jaybaik@sas.upenn.edu)
