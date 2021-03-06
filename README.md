# TrollAccountsClassification
This repository contains work for my Masters Thesis on identifying spam campaign collaborators on social media. In this context, the spam campaign is the IRA (Internet Research Agency) attempt of meddling with the US elections, by spreading misinformation and polarized political views on Twitter.

# Dataset
## Inauthentic accounts
The "Russian troll tweet" dataset is publsihed by "fivethirtyeight". The data set is the work of two professors at Clemson University: Darren Linvill and Patrick Warren and has been released for free use for the academic community. The dataset contains over 3 million tweets of troll accounts associated with IRA.

## Authentic accounts
The baseline dataset contains x number of tweets from authentic accounts. The tweets are dated from year 2016 (US election year). The baseline data is scraped from the full-archive Twitter API. The historical data on "follower" and "following" counts is not available through the API and has therefore been inferred from a regression model over the "follower" and "following" count development of random sampled accounts.
