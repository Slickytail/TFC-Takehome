# Prompt
Here is a dataset of different sources of time-series: https://huggingface.co/datasets/theforecastingcompany/GiftEvalPretrain. The idea is to train some reasonable forecasting model on this data, that includes a small training pipeline for this data, and some model architecture.

Some notes:
 • Here are some related repos with papers to models in the time-series space. You might find inspiration there:
   • Salesforce's Moirai: https://github.com/SalesforceAIResearch/uni2ts
   • Google's TimesFM: https://github.com/google-research/timesfm
   • Amazon's Chronos: https://github.com/amazon-science/chronos-forecasting
   • Timer / Sundial: https://github.com/thuml/OpenLTM
 • We are more interested in the way you handle data, the training pipeline, and architecture, than the performance of any model you can train in this short time. We don't expect a giant model or anything competitive.
 • The goal is to build a very first prototype, something simple, that demonstrates that it's possible to learn from this data. We don't expect revolutionary ideas and architectures in this short time - it can be a very simple model.
 • How much time you spend on this is up to you - you are of course very welcome to timebox it as you see fit. Just document your time spent so that we can calibrate accordingly. If you run out of time - just document what you would have done if you had more time.
 • The data linked above is quite large. We don't expect you to train on all the data. It is fine to choose some subset so that you can train it on a laptop, etc.
 • The data we linked is the same as the GiftEvalPretrain from Salesforce: https://huggingface.co/datasets/Salesforce/GiftEvalPretrain. Except that we added a column with dataset name. This dataset is used for the GiftEval leaderboard: https://huggingface.co/spaces/Salesforce/GIFT-Eval
 • Feel free to use any AI coding tools as you wish (ChatGPT, Claude, etc.)
 • Make any decision decision you deem necessary, but document your assumptions.


# My Solution

## AI Use
I mostly code through AI these days.
For this project I used GLM 5.3 Flash, through `pi`. 

My process is generally to read relevant papers/code on my own, and explore the data.
Then, for each module I want, I write a detailed spec of how it should be structured and what it should do.
I usually find that while writing the spec, I run into a bunch of things that I didn't realize I was unclear about,
so it forces me to think through them in order to explain them to the agent.

## Modeling
Model architecture is basically Chronos-2, but with P=1.
I also didn't implement the explicit grouping, since in our training loop we just bucket by number of variables.

Patch encoder obviously helps with speed, since you reduce the number of tokens in the attention calls. This is quite relevant since Chronos-like models do full attention (ie, as opposed to causal) so there's no capacity for KV-caching.
But in this small case, I don't expect patching to substantially improve modeling performance.
Instead, I replace the the patch encoder with a per-value MLP.

## Dataset
I downloaded about 3% of the data, totaling 30GB. Although we might see better generalization from this quick train if we undersampled the large subsets, I decided to use a representative sample of the data to simulate the same data mix we would get if we trained on the whole dataset.

Because this is too much data to hold in memory (on my laptop -- on a dedicated training server it wouldn't really be a big deal), I go for a two-layer shuffle approach. We maintain a shuffled list of all the row groups across all the parquet files, and only load a few of them into memory at a time. This is not a completely random data order, but it should be enough for stable training. Within the current buffer, we bucket by the total number of variables (since we don't support padding) and try to fill batches with time series from different files, for intra-batch diversity.

## Training
Exactly matching Chronos-2 in loss design.
Had issues with extreme outliers causing training instability. Handled this by adding gradient clipping, and marking extreme-but-finite values as missing (in particular, any entries with absolute value after normalization and asinh of > 7, meaning ~500 std deviations above the mean becomes infinite.)
Trains pretty fast on my laptop, with a very small model size and low sequence length.

## Evaluation
We evaluate MASE vs a seasonal baseline, as in the gift-eval repo. On a randomly chosen eval set, reaches <value> MASE, as compared to 1.0 for the naive.

## Time Spent
I spent about six hours on this across a few days, including time spent for training.
