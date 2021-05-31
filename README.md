# TrollAccountsClassification
This repository contains work for my Masters Thesis on identifying spam campaign collaborators on social media. In this context, the spam campaign is the IRA (Internet Research Agency) attempt of meddling with the US elections, by spreading misinformation and polarized political views on Twitter.

# Dataset
## Inauthentic accounts
The ["Russian troll tweet" dataset](https://github.com/fivethirtyeight/russian-troll-tweets) is publsihed by "fivethirtyeight". The data set is the work of two professors at Clemson University: Darren Linvill and Patrick Warren and has been released for free use for the academic community. It contains over 3 million tweets of troll accounts associated with IRA.

## Authentic accounts
The [baseline dataset](https://www.kaggle.com/sergejbogachov/tweets-of-random-sampled-accounts-from-2016) contains over 5 million tweets from 1198 authentic accounts. The tweets are dated from year 2016 (US election year). The baseline data is sampled from the full-archive Twitter API. More specific information on how this dataset has been sampled is in the report.

# Structure
The folders in this repository mostly contain scripts for data processing, analysis and plotting.

* data_analysis: contains all the plots from the "Analysis" section together with other calculations used during the analysis.
* data_sampling: contains all the scripts used for sampling baseline data.
* models: contains all the scripts used for model training and evaluation.
* attacks: containst demonstrations of evasion and poisoning attacks agains the detection model.

Note that in order to replicate all experiments, you will need to apply all the preprocessing scripts to the correct order. This is easier to do when following the processing pipeline in the report.
