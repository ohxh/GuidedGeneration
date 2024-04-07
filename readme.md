# Guided Generation

This repository contains some code + latex source from work on Guided Generation. 

![Greedy next-token prediction balancing coherence and truth](./latex/fig1.png)


Building off the work in [Discovering Latent Knowledge](https://arxiv.org/abs/2212.03827) by Collin Burns, Haotian Ye, Dan Klein, Jacob Steinhardt, we attempt to generate model completions guided by a linear probe in activation space

See [this thread](https://discord.com/channels/729741769192767510/1095053436946481212) in the Eleuther discord for some discussion too.


### Abstract

In this work, we show that linear classifiers for truth learned from consistent-contrast search in a language model's activation space transfer well to open-ended question answering.  On TruthfulQA, a benchmark designed adversarially to uncover instances where models mimic human falsehoods, this method outperforms raw model probabilities at every model size tested. We then introduce Guided Generation, a novel method for generating text from a language model in line with latent knowledge uncovered from a classifier on model activations. We examine three approaches: simple rejection sampling, greedy search on merged objectives, and beam search on the same.
  
### Code

