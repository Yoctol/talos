def test_absolute_import():
    from talos.layers import Dense # noqa
    from talos.layers import Conv1D  # noqa
    from talos.layers import Lambda  # noqa
    from talos.layers import LSTM  # noqa
    from talos.layers import Conv1DTranspose  # noqa


def test_relative_import():
    from .. import Dense # noqa
    from .. import Conv1D  # noqa
    from .. import Lambda  # noqa
    from .. import LSTM  # noqa
    from .. import Conv1DTranspose  # noqa
