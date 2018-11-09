# talos

[![CircleCI](https://circleci.com/gh/Yoctol/talos/tree/master.svg?style=svg&circle-token=20e0cbfda638b10e16b0f911708886e8112f4783)](https://circleci.com/gh/Yoctol/talos/tree/master)

About this project...

## Installation

``` shell
git clone https://github.com/Yoctol/talos.git
```

``` shell
pipenv install
```

## Usage

```python=
pre_activate_bn_block = lambda: Sequential([
    Dense(
        units=10,
        kernel_initialzer='lecun_normal',
        activation=None,
    ),
    BatchNormalization(),
    Activation('relu'),
])
model = Sequential([
    pre_activate_bn_block(),
    pre_activate_bn_block(),
    pre_activate_bn_block(),
])

outputs = model(inputs)
```

## Test

``` shell
pipenv install --dev
```

``` shell
pytest
```