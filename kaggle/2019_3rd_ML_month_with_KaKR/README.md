# 2019 3rd ML month with KaKR
* Competition URL
    * https://www.kaggle.com/c/2019-3rd-ml-month-with-kakr
* Experiments
    * Baseline
        * Model 1
            * Xception
            * Using full data
            * Epoch 20
            * Score: 0.89699
        * Model 2
            * Xception
            * Swap optimizer Nadam to SGD in the middle of training
            * ReduceLROnPlateau, EarlyStopping
            * Score: 0.89842 
        * Model 3
            * Xception
            * Swap optimizer Adam to SGD in the middle of training
            * ReduceLROnPlateau, EarlyStopping
            * Snapshot ensemble after SGD (weight decay exponentially)
            * Ensemble first 2 models of 5-fold cross validation
            * Score: 0.91368
    * Ensemble
        * Model 1
            * Xception 
            * Swap optimizer Nadam to SGD in the middle of training
            * ReduceLROnPlateau, EarlyStopping
            * Snapshot ensemble after SGD (weight decay exponentially)
            * Ensemble 5-fold cross validation
            * Score: 0.94452
        * Model 2
            * Xception + DenseNet201
            * Swap optimizer Adam to SGD in the middle of training
            * ReduceLROnPlateau, EarlyStopping
            * Snapshot ensemble after SGD (weight decay exponentially)
            * Ensemble 5-fold cross validation
            * Score: 0.95504
    * Final score
        * Public: 0.95504 (24th)
        * Private: 0.94579 (35th)
            