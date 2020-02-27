"""Used to make pytest functions available globally"""

#  Copyright (c) 2020 zfit
#
#
# def pytest_generate_tests(metafunc):
#     if metafunc.config.option.all_jit_levels:
#
#         # We're going to duplicate these tests by parametrizing them,
#         # which requires that each test has a fixture to accept the parameter.
#         # We can add a new fixture like so:
#         metafunc.fixturenames.append('tmp_ct')
#
#         # Now we parametrize. This is what happens when we do e.g.,
#         # @pytest.mark.parametrize('tmp_ct', range(count))
#         # def test_foo(): pass
#         metafunc.parametrize('tmp_ct', range(2))
