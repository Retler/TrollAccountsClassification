import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
plt.style.use('seaborn')
plt.rcParams['font.size'] = '20'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

data = pd.read_csv("combined_data.csv")

data["label"] = data["label"].map({"troll": 1, "basline": 0})
cor_sentiment = pointbiserialr(data["label"], data["sentiment"])
cor_lifespan = pointbiserialr(data["label"], data["lifespan"])
cor_f_ratio = pointbiserialr(data["label"], data["f_ratio"])
cor_h_hitrate = pointbiserialr(data["label"], data["h_hitrate"])
