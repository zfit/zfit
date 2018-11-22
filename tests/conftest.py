"""Used to make pytest functions available globally"""

test_length = 0
test_length = 10


def test_choice(*args):
    if test_length == 0:
        return args[0]
    elif test_length == 10:
        return args[1]
