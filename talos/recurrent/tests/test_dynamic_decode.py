from unittest.mock import call

from ..dynamic_decode import dynamic_decode


def test_dynamic_decode(mocker):

    def mock_cell_call(inputs, state):
        return inputs + len(state), state + 's'

    mock_cell = mocker.Mock(side_effect=mock_cell_call)
    first_input = ord('A')
    mocker.patch('tensorflow.stack', lambda lst, axis: ",".join(lst))
    output = dynamic_decode(
        cell=mock_cell,
        first_input=first_input,
        maxlen=5,
        output_layer=lambda x: chr(x),
        next_input_producer=lambda c: ord(c),
        init_state='s',
    )
    expected_cell_calls = [
        call(first_input, 's'),
        call(first_input + 1, 'ss'),
        call(first_input + 3, 'sss'),
        call(first_input + 6, 'ssss'),
        call(first_input + 10, 'sssss'),
    ]
    expected_output = ",".join([
        chr(first_input + 1),
        chr(first_input + 3),
        chr(first_input + 6),
        chr(first_input + 10),
        chr(first_input + 15),
    ])
    mock_cell.assert_has_calls(expected_cell_calls)
    assert output == expected_output
