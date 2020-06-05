# Hyperparameter Tuning 

## Train/Validation/Test Sets
Depending on the size of data set the percentage of test/train/validation may vary. For a small data set 70/30 (train/test) split may be good.But for large data set you may need to split it in 90/10 or 95/5 <br>
Not having a test set might be okay if you have a dev/validation set <br>
Validation and Test set should be from the same distribution of data <br>

## Bias/Variance
### Simple Explaination
If the training accuracy is high but testing accuracy is low (over-fitting) => High Vaiance<br>
If the training as well as testing accuracy is low and similar => High bias<br>
If the training accuracy is low and testing is even worse => High Bias and High Variance<br>
If both the training as well as the testing accuracy is high => Low Bias and Low Variance<br>

### Possible Solution for High Bias / High Variance
High Bias : Bigger Network, Train Longer , Play with Network Architecture<br>
High Variance : More Data, Regularization , Play with Network Architecture<br>
