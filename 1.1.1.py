# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IuUXRyTXlUtvIaZ5E_IUioZjJWcpFM5P
"""

import pandas as pd
train = pd.read_csv ('analytics/train.csv')
train['Class'].hist();