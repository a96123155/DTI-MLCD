# DTI-MLCD
## Predicting drug-target interaction using multi-label learning with community detection method (DTI-MLCD)

### Introduction
Identifying drug-target interactions (DTIs) is an important step for drug discovery and drug repositioning. To reduce heavily experiment cost, booming machine learning has been applied to this field and developed many computational methods, especially binary classification methods. However, there is still much room for improvement in the performance of current methods. Multi-label learning can reduce difficulties faced by binary classification learning with high predictive performance, and has not been explored extensively. The key challenge it faces is the exponential-sized output space, and considering label correlations can help it. Thus, we facilitate the multi-label classification by introducing community detection methods for DTIs prediction, named DTI-MLCD. On the other hand, we updated the gold standard data set proposed in 2008 and still in use today. The proposed DTI-MLCD is performed on the gold standard data set before and after the update, and shows the superiority than other classical machine learning methods and other benchmark proposed methods, which confirms the efficiency of it.

### Requirements
This method developed with Python 3.6, please make sure all the dependencies are installed, which is specified in DTI_MLCD_requirements.txt.


### Reference
Predicting drug-target interaction using multi-label learning with community detection method (DTI-MLCD)


### Run NR data set (as a demo)
1. Download fold “.\DTI-CDF\2_Example_NR”.

2. In the “.\DTI-CDF\2_Example_NR” path, run the Example_NR.py file, as follows:  
   Open CMD and input:  
          `cd .\DTI-CDF\2_Example_NR`  
          `python -u Example_NR.py > Example_NR.out`


Please see “Example_NR.out” file for the results/outputs which contains the results of performance metrics, time required for the program to run and the new DTIs predicted by this method.  
If you want to try other data sets, just follow this demo, and the codes and data have been supported in fold “1_all_code” and “1_original_data”, respectively.

### Package dependencies

The package is developed in python 3.6, lower version of python is not suggested for the current version.  
Run the following command to install dependencies before running the code: pip install -r DTI_MLCD_requirements.txt.  

### Others
Please read reference and py file for a detailed walk-through.

### Thanks
