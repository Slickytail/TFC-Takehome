# My Solution

## AI Use
I mostly code through AI these days.
For this project I used GLM 5.3 Flash, through `pi`. 

My process is generally to read relevant papers/code on my own, and explore the data.
Then, for each module I want, I write a detailed spec of how it should be structured and what it should do.
I usually find that while writing the spec, I run into a bunch of things that I didn't realize I was unclear about,
so it forces me to think through them in order to explain them to the agent.

## Modeling
Model architecture is basically Chronos-2, but with the following changes:
  - P=1 (ie, no patch encoding).
    Patch encoder obviously helps with speed, since you reduce the number of tokens in the attention calls. This is quite relevant since Chronos-like models do full attention (ie, as opposed to causal) so there's no capacity for KV-caching.
    But in this small case, I don't expect patching to substantially improve modeling performance.
    Instead, I replace the the patch encoder with a per-value MLP.
  - No explicit grouping. Explicit grouping allows for more efficient training, since it allows for multiple batches of different sizes to be run together. In this minimal example, we simply consider each element of the batch as a group, and only train on batches where all series have the same number of variables.
  - No separator token between the past and future. Again, I expect that a separator token helps with model stability on very long or highly variable sequence lengths, but is relatively unimportant in this small test.

## Dataset
I downloaded about 3% of the data, totaling 30GB. Although we might see better generalization from this quick train if we undersampled the large subsets, I decided to use a representative sample of the data to simulate the same data mix we would get if we trained on the whole dataset.

Because this is too much data to hold in memory (on my laptop -- on a dedicated training server it wouldn't really be a big deal), I go for a two-layer shuffle approach. We maintain a shuffled list of all the row groups across all the parquet files, and only load a few of them into memory at a time. This is not a completely random data order, but it should be enough for stable training. Within the current buffer, we bucket by the total number of variables (since we don't support padding) and try to fill batches with time series from different files, for intra-batch diversity.

I didn't ultimately end up using all of this data -- I was able to train a better-than-naive forecaster with only 80k samples (compared to the 650k total), so I stopped there.

## Training
Exactly matching Chronos-2 in loss design.
Had issues with extreme outliers causing training instability. Handled this by adding gradient clipping, and marking extreme-but-finite values as missing (in particular, any entries with absolute value after normalization and asinh of > 5, meaning ~75 std deviations away from the mean becomes infinite.)
Trains pretty fast on my laptop, with a very small model size and low sequence length.
Ultimately, I didn't use all of the data that I downloaded. My final training run only saw 80k sequences, or about 12% of the total data I fetched.
I periodically ran small validations (200 samples) during training to see that MASE was decreasing.

Since chronos-style training requires explicitly choosing the point to cut off between past and future, I decided to cut each training sequence down to a random short length between 64 and 512 values, and then give the model between 80% and 95% of the history, to predict the rest of the sequence.

## Evaluation
We evaluate MASE (averaged over the entire future) vs a seasonal baseline, as in the gift-eval repo.
On a final evaluation over the full test set (18k sequences):
```
checkpoint: checkpoints/checkpoint-00020000/ | split: test                                                                                                 
  model: MASE = 1.2864
  naive: MASE = 1.8416
  seasonal_naive: MASE = 1.3795
```

Hooray, we beat the seasonal naive prediction by a statistically significant amount!

## Time Spent
I spent about eight hours on this across a few days, including time spent for training.

## Results 
The MASE we got to only slightly beats the naive forecaster, so the model is not really a "strong forecaster". However, based on the GIFT-eval leaderboards, it seems that beating the naive forecast by even a little bit over a diverse test set shows really learning progress. I therefore conclude that our model has learned weak but real forecasting abilities, rather than simply getting lucky with the test set.

The main steps that I expect to improve the model would be:
   - train longer! my MASE (over a tiny validation set) and loss continued to decrease stably at the end of training.
   - larger batch size. transformer models benefit enormously from large batch sizes during pretraining. 
   - a more reasoned balancing of the dataset. The dataset contains sequences of wildly varying lengths, subsets of different sizes, and different numbers of variables. It's quite likely that the loss landscape of this minimal example is dominated by a particular subset or type of data.
