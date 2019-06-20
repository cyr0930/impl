# Jigsaw Unintended Bias in Toxicity Classification
* Competition information
    * https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification
* Experiments
    * Baseline
        * Simple deep neural network
            * Embedding(10000 x 16) - GlobalAveragePooling - Dense(16) - Dense(1)
            * Score: 0.88063, Epoch: 30
        * GRU
            * Embedding(30000 x 16) - GRU(16) - Dense(1)
            * Score: 0.90769, Epoch: 4
        * Stacked GRU
            * Embedding(30000 x 16) - GRU(16) - GRU(4) - Dense(1)
            * Score: 0.90857, Epoch: 3
        * GRU + Glove
            * Embedding(Glove.6B.300d) - GRU(128) - Dense(32) - Dense(1)
            * Score: 0.91684, Epoch: 5
        * GRU + Glove + Attention
            * Embedding(Glove.6B.300d) - GRU(128) - Attention - Dense(32) - Dense(1)
            * Score: 0.91460, Epoch: 5
        * **GRU + Glove + Dropout(0.5)**
            * Embedding(Glove.6B.300d) - GRU(512) - Dense(128) - Dense(1)
            * Score: 0.92088, Epoch: 10
    * Debiasing
        * Extend background positive
            * Score: 0.92372
