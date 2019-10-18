# talos

[![travis][travis-image]][travis-url]
[![pypi][pypi-image]][pypi-url]
![release][release-image]

[travis-image]: https://img.shields.io/travis/Yoctol/talos.svg?style=flat
[travis-url]: https://travis-ci.org/Yoctol/talos
[pypi-image]: https://img.shields.io/pypi/v/talos.svg?style=flat
[pypi-url]: https://pypi.python.org/pypi/talos
[release-image]: https://img.shields.io/github/release/Yoctol/talos.svg


About this project...

## Installation

``` shell
$ git clone https://github.com/Yoctol/talos.git
```

``` bash
$ pipenv install
```

## Usage

多個 Layers 組合串接
```python
def pre_activate_bn_dense(units):
    return Sequential([
        Dense(
            units=units,
            kernel_initialzer='lecun_normal',
            activation=None,
        ),
        BatchNormalization(),
        Activation('relu'),
    ])

model = Sequential([
    pre_activate_bn_dense(32),
    pre_activate_bn_dense(16),
    pre_activate_bn_dense(8),
])

outputs = model(inputs)
```

方便的 variables/updates scope collection
```python
with tf.control_dependencies(model.updates):
    train_op = SGDOptimizer(loss, var_list=model.trainable_variables)
```

## Test

``` shell
$ pipenv install --dev
```

``` shell
$ pytest
```
