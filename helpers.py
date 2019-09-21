from collections import defaultdict

def assert_shape(x, shape:list):
    """ assert_shape(conv_input_array, [8, 3, None, None]) """
    assert len(x.shape) == len(shape), (x.shape, shape)
    for _a, _b in zip(x.shape, shape):
        if isinstance(_b, int):
            assert _a == _b, (x.shape, shape)


def assert_shapes(x, x_shape, y, y_shape):
    assert_shape(x, x_shape)
    assert_shape(y, y_shape)

    shapes = defaultdict(set)
    for arr, shape in [(x, x_shape), (y, y_shape)]:
        for i, char in enumerate(shape):
            if isinstance(char, str):
                shapes[char].add(arr.shape[i])


    for _, _set in shapes.items():
        assert len(_set) == 1, (x, x_shape, y, y_shape)

