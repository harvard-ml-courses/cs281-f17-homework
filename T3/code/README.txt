#README

In order to reduce memory footprint, we need to do the following two things when load data:

* For problem 2, use flag `load_text=False` in load_jester
* For problem 3, use flag `subsample_rate=0.2` to only use 20% of the training data. We also only keep the most frequent 150 words when loading text.

# Implementation Notes for Ordinal Regression

## If torchtext doesn't work

* `sudo pip uninstall torchtext`
* `sudo pip install git+https://github.com/pytorch/text.git@b57bab91dce024fbb9ef6ba297c695b007aedbcf`

## If you see numerical issues such as nans or infs during optimization

* first check if you used the provided `log_difference` function in `utils.py`.
* if the problem remains, try initializing parameters with smaller values. For example, in `nn.Embedding`, you can do `torch.nn.init.uniform(self.w.weight.data, -0.01,0.01)`.
* if the problem still remains, locate where the issue is, and try `torch.clamp`, but you should be really careful about this, since it's changing our objective. 
