# Installation and Usage
Simply `git clone git@github.com:dreadlesss/Panax_age_predictor.git` the repository to your home computer. 

For detailed information, please refer to the article: [Identification of intrinsic hepatotoxic compounds in Polygonum multiflorum Thunb. using machine-learning methods. Chin Med. 2021, 16:100](https://cmjournal.biomedcentral.com/track/pdf/10.1186/s13020-021-00511-5.pdf).

# Requirements 

The code is written in [Python 3.8.10] and mainly uses the following packages:
* [Sklearn] for model building
* [matplotlib] for plotting figures

# Setup

Install [Anaconda](https://www.anaconda.com/distribution/#download-section) on your computer and create a Conda environment:

```
conda env create -n age_predict -f environment.yml 
```

Activate the new Conda environment:

```
conda activate age_predict
```

Launch Jupyter Notebook:

```
jupyter notebook
```

# DATASET
train data set: train_data.xlsx

test data set: test_data(merged).xlsx

test set 1:test set 1.xlsx

test set 2:test set 2.xlsx

