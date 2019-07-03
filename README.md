# urop2019
Summer UROP 2019 project repository.

## Workflow

Never push directly to the `master` branch

When working on this repository, create a new branch using the convention *yourname_feature* (with `git checkout -b branchname`). For example, the branch that updated this README.md file is called `kevin_update_readme`.

Whilst working on your branch, keep it up to date with the `master` branch by regularly merging the `master` branch into your branch.

When your work is ready, create a pull request. At least one other team member needs to approve your pull request before it can be merged into `master`.

## Boden

The Imperial College Mathematics Department server Boden will be available for the duration of the project. You can access Boden from within the College via ssh: `ssh username@boden.ma.ic.ac.uk`. It is also possible to access from outside either using a VPN or by tunneling (see instructions [here](http://sysnews.ma.ic.ac.uk/ssh-gateways.html)). 

Tensorflow is available use for python3. As an example, try running:

`python3 -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"`

### tmux

When running longer training scripts etc., you should launch them inside a screen using `tmux`. You can find a summary of `tmux` commands [here](https://gist.github.com/henrik/1967800).
testing
testom
