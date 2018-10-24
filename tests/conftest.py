"""Used to make pytest functions available globally"""

from __future__ import print_function, division, absolute_import

test_length = 0
test_length = 10

def test_choice(*args):
    if test_length == 0:
        return args[0]
    elif test_length == 10:
        return args[1]
