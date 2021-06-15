# alexNet(with Tensorflow, without Keras)

## Required packages
- os
- time
- tensorflow(==2.4.1)
- tensorflow_addons
- numpy
- random

## How to run
### Not using docker 
1. Run the alex_net.py for training
  - When you finish the training, you get trained parameters in "./output/models/" directory.
2. Run valid.py for validation step
  - When you finish the validation step, you get best parameters(have minimum loss) for AlexNet.(Ofcourse one of your trained parameters. Not world best.)
  - The best_model.npz file create at "./output/" directory.

### Using docker
1. Build Dockerfile
```sudo docker build -t tf:0.1 .```
3. Run docker image(with volume option)
```docker run -it -v your_clone_directory_path:/alexNet tf:0.1 /bin/bash```
3. Run alex_net.py for training
```
(Docker bash): cd alexNet/
(Docker bash): python3.6 alex_net.py
(Training...)
```
4. Run valid.py for validation
```
(Docker bash): python3.6 valid.py
(Validation...)
```
5. Run test.py for testing
```
(Docker bash): python3.6 test.py
(You got test result)
```
