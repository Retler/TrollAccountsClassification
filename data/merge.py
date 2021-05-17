import os
import pandas as pd
import glob

# TODO: Put all russian troll tweet data files in this dir
path = r'./'
all_files = glob.glob(os.path.join(path, "*.csv") )

df = pd.concat((pd.read_csv(f) for f in all_files), axis=0, ignore_index=True)
df.to_csv("tweets_full.csv", index=False)
