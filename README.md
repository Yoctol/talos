# talos

[![CircleCI][circleci-image]][circleci-url]

[circleci-image]: https://circleci.com/gh/Yoctol/talos.svg?style=shield&circle-token=20e0cbfda638b10e16b0f911708886e8112f4783
[circleci-url]: https://circleci.com/gh/Yoctol/talos


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
