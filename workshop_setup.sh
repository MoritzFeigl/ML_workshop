#!/bin/bash
conda create --name workshop python=3.9
conda activate workshop
conda install -c anaconda jupyter
conda install -c anaconda pandas
conda install -c conda-forge matplotlib
conda install -c anaconda scikit-learn
conda install -c anaconda seaborn
jupyter-notebook
