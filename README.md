# alexNet(with Tensorflow, without Keras)

## Required packages
- os
- time
- tensorflow(==2.4.1)
- tensorflow_addons
- numpy
- random

## How to run
1. Run the alex_net.py for training
  - When you finish the training, you get trained parameters in "./output/models/" directory.
2. Run valid.py for validation step
  - When you finish the validation step, you get best parameters(have minimum loss) for AlexNet.(Ofcourse one of your trained parameters. Not world best.)
  - The best_model.npz file create at "./output/" directory.
