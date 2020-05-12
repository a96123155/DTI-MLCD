# DTI-MLCD
## Predicting drug-target interaction using multi-label learning with community detection method (DTI-MLCD)

### Introduction
Identifying drug-target interactions (DTIs) is an important step for drug discovery and drug repositioning. To reduce heavily experiment cost, booming machine learning has been applied to this field and developed many computational methods, especially binary classification methods. However, there is still much room for improvement in the performance of current methods. Multi-label learning can reduce difficulties faced by binary classification learning with high predictive performance, and has not been explored extensively. The key challenge it faces is the exponential-sized output space, and considering label correlations can help it. Thus, we facilitate the multi-label classification by introducing community detection methods for DTIs prediction, named DTI-MLCD. On the other hand, we updated the gold standard data set proposed in 2008 and still in use today. The proposed DTI-MLCD is performed on the gold standard data set before and after the update, and shows the superiority than other classical machine learning methods and other benchmark proposed methods, which confirms the efficiency of it.

### Requirements
This method developed with Python 3.6, please make sure all the dependencies are installed, which is specified in DTI_MLCD_requirements.txt.


### Reference
Predicting drug-target interaction using multi-label learning with community detection method (DTI-MLCD)


### Run
1. The updated dataset is in the fold **update_dataset**, and the code that how to update the dataset is publicate in the fold **.\code\1_data_update**

2. The drug feature is generate from ChemDes platform and RDKit python package. They need some post-hoc process. The code is in the **.\code\2_drug_feature**

3. The target feature is generate from PROFEAT web server and PFAM database. They need some post-hoc process. The code is in the **.\code\2_target_feature**

4. The train and test codes of DTI-MLCD are **GPCR_TD_model.py** for predicting new drugs, and **GPCR_TT_model.py** for predicting new targets, respectively.

5. In addition, I have provide the analysis code, including dataset analysis, labe correlations analysis, SCV data analysis, community detection results and drawing, and Friedman test.

### Package dependencies

The package is developed in python 3.6, lower version of python is not suggested for the current version.  
Run the following command to install dependencies before running the code: pip install -r DTI_MLCD_requirements.txt.  

### Others
Please read reference and py file for a detailed walk-through.

### Thanks
