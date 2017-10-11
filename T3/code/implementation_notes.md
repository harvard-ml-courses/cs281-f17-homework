# Implementation Notes for Ordinal Regression

## If torchtext doesn't work

* `sudo pip uninstall torchtext`
* `sudo pip install git+https://github.com/pytorch/text.git@b57bab91dce024fbb9ef6ba297c695b007aedbcf`

## If you see numerical issues such as nans or infs during optimization

* first check if you used the trick mentioned in writeup and used NormLogCDF instead of NormCDF.
* if the problem remains, try initializing parameters with smaller values. For example, in `nn.Embedding`, you can do `torch.nn.init.uniform(self.w.weight.data, -0.01,0.01)`.
* if the problem still remains, locate where the issue is, and try `torch.clamp`, but you should be really careful about this, since it's changing our objective. 
