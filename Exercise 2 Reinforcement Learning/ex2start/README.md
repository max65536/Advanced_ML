DQN code framework for Question 2 of Exercise 2 of the Advanced Machine Learning Course at RWTH Aachen, summer term 2019, Jonathon Luiten

This code has been adapted from https://github.com/devsisters/DQN-tensorflow at commit 3fbfc59 (March 10th, 2017)
Original code copyright Devsisters corp., see LICENSE

## Requirements

- Python 2.7 or Python 3.3+
- [gym](https://github.com/openai/gym)
- [tqdm](https://github.com/tqdm/tqdm)
- [SciPy](http://www.scipy.org/install.html) or [OpenCV2](http://opencv.org/)
- [TensorFlow 0.12.0](https://github.com/tensorflow/tensorflow/tree/r0.12)


## Usage

First, install prerequisites with:

    $ pip install tqdm gym[all]

Make sure, gym version 0.7.0 is installed:

    $ pip install gym==0.7.0

To train a model for Breakout:

    $ python main.py --env_name=Breakout-v0 --is_train=True
    $ python main.py --env_name=Breakout-v0 --is_train=True --display=True

To test and record the screen with gym:

    $ python main.py --is_train=False
    $ python main.py --is_train=False --display=True


## License

MIT License.
