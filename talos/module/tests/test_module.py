from ..module import Sequential


def test_sequential(mocker):
    mock_layer = mocker.Mock(
        side_effect=lambda x: x + 1,
        trainable_variables=[1, 2, 3],
        updates=[],
    )
    seq = Sequential([mock_layer for _ in range(5)])
    assert seq(2) == 7  # 2 + 1 * 5

    assert mock_layer.call_count == 5
    assert len(seq.trainable_variables) == 15
    assert len(seq.updates) == 0
