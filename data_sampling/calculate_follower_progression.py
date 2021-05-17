"""
This script read the information of random sampled users, their age and followers/following count
This information is aggregated to calculate the average follower/following progression over time
"""
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn')
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelsize'] = '15'
plt.rcParams['xtick.labelsize'] = '15'
plt.rcParams['ytick.labelsize'] = '15'

DATESET = "./data_sampling/user_data.csv"

data = pd.read_csv(DATESET, header=None, names=["author", "created_at", "followers", "following"])
data.drop_duplicates(subset=['author'], inplace=True)

data["age_weeks"] = data["created_at"].apply(lambda x: int((datetime.today() - datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.000Z")).days / 7) )

ten_year_accounts = data[data["age_weeks"] <= 520]
plt.plot(ten_year_accounts.groupby("age_weeks").count()["author"])
plt.xlabel("Account age in weeks")
plt.ylabel("Number of accounts")
plt.title("Number of accounts per age bin")
plt.show() # The high number of young accounts can be explained by bots or inauthentic accounts which are closed over time

# Limit accounts to last year
ten_year_accounts["age_weeks"] = ten_year_accounts["age_weeks"].apply(lambda x: x - 1)
five_year_accounts = ten_year_accounts[ten_year_accounts["age_weeks"] <= 260]

# Plot moving avg. of followers
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(five_year_accounts.groupby("age_weeks")["followers"].mean().rolling(5, 1).mean())
fig.suptitle('Account age (weeks)', x=0.5, y=0.02, verticalalignment='bottom', horizontalalignment='center', fontsize=15, fontweight='bold')
ax1.set_ylabel("Followers")
ax1.set_title("5 week MA of 'followers' development", fontsize='15', fontweight='bold')

# Export
# ten_year_accounts.groupby("age_weeks")["followers"].mean().rolling(10, 1).mean().to_csv("followers_10wk_avg.csv")

# Plot moving avg. of following

ax2.plot(five_year_accounts.groupby("age_weeks")["following"].mean().rolling(5, 1).mean())
ax2.set_ylabel("Following")
ax2.set_title("5 week MA of 'following' development", fontsize='15', fontweight='bold')

plt.show()
# Export
# ten_year_accounts.groupby("age_weeks")["following"].mean().rolling(10, 1).mean().to_csv("following_10wk_avg.csv")

# # Set which metrics we want to train the model over
# metrics = ["followers", "following"]

# # Train-test split
# train, test = train_test_split(ten_year_accounts, test_size=0.33, random_state=1234)

# # Save models in map
# models = {}

# for metric in metrics:
#     print(f"Training for metric {metric}")
    
#     train_metric = train.groupby("age_weeks")[metric].mean()
#     test_metric =  test.groupby("age_weeks")[metric].mean()
#     X_train, y_train = train_metric.keys().to_numpy().reshape(-1, 1), train_metric.values
#     X_test, y_test = test_metric.keys().to_numpy().reshape(-1, 1), test_metric.values
#     degrees = [1,2,3,4,5,6,7,8]
    
#     # Parameter tuning using 5-fold cross validation (followers)
#     kfold = model_selection.KFold(n_splits=5, random_state=1234, shuffle=True)
#     means = []
#     for d in degrees:
#         polyreg = make_pipeline(PolynomialFeatures(d), LinearRegression())
#         results = model_selection.cross_val_score(polyreg, X_train, y_train, cv=kfold, scoring='r2')
#         means.append(results.mean())
#         print(f"d={d}, mean={results.mean()}, std={results.std()}")
#     # d=4 has been chosen as the best result

#     d = np.argmax(means) + 1
#     print(f"plotting regression with d={d}")
#     polyreg = make_pipeline(PolynomialFeatures(d), LinearRegression())
#     polyreg.fit(X_test, y_test)
#     plt.plot(X_test, y_test)
#     y_pred = polyreg.predict(X_test)
#     r2 = r2_score(y_test, y_pred)
#     print(f"R2 score of final model: {r2}")
#     plt.plot(X_test, y_pred, color="black")
#     plt.title(f"Polynomial regression with degree {d}")
#     plt.show()

#     models[metric] = polyreg
    

# Calculate R2 of followers predictions of accounts that are at most 12 weeks old
# under_12 = test[test["age_weeks"] <= 12].groupby("age_weeks")["followers"].mean()
# model = models["followers"]
# y_pred = model.predict(under_12.keys().to_numpy().reshape(-1,1))
# r2 = r2_score(under_12.values, y_pred)
# r2 is almost 0 here...
