# Biologically-informed Interpretable Deep Learning Framework for BMI Prediction and Gene Interaction Detection

This repo presents a framework for detecting gene-gene interactions from trained DL models and contains the code to reproduce the experiments of the paper (insert paper link once published). 

## Before Running
- Clone this repository to your local machine
- The file requirements.txt contains correct versions of all necessary packages to run this code. Run `pip install -r requirements.txt` from within the repo directory to install all required dependencies
- It is expected that you have already cleaned and preprocessed genetic data to your requirements and trained a pytorch model on this data. The data is expected to be in numpy array format.
- Create an ordered dictionary of genes or regions which will be used as interpretable features. The keys of the dictionary should be names or identifiers for the region and the values should be boolean masks where all loci not associated with this region have value zero and associated loci have value 1. This is expected to be saved as a pickle

## Running instructions
To execute the code, run 
```python feature_ablation.py  path/to/feature_masks path/to/trained_model path/to/test_data results_directory```

