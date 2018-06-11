import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

data11 = pd.read_csv('winequality-red.csv')

