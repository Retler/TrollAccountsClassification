import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelsize'] = '15'
plt.rcParams['xtick.labelsize'] = '15'
plt.rcParams['ytick.labelsize'] = '15'


# data to plot
n_groups = 4
svm = (0.97, 0.24, 0.78, 0.38)
rf = (0.97, 0.85, 0.95, 0.90)
knn = (0.95, 0.48, 0.84, 0.64)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.25
opacity = 0.8

rects1 = plt.bar(index, svm, bar_width,alpha=opacity,label='svm')
rects2 = plt.bar(index + bar_width, rf, bar_width, alpha=opacity, label='rf')
rects3 = plt.bar(index + 2*bar_width, knn, bar_width, alpha=opacity, label='knn')

ax.set_xlabel('Score')
ax.set_ylabel('Value')
ax.set_title('Scores by model')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(('Precision', 'Recall', 'Accuracy', 'F1'))
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)

fig.tight_layout()
plt.show()
