#  Copyright (c) 2025 zfit

import copy
import os
import subprocess
import sys

import pytest

import zfit


@pytest.mark.first  # needs to run before initialization
@pytest.mark.parametrize(
    ["n_cpu", "taken", "left"],
    [[3, 5, 0], [0, -1, 0], [10, 3, 7], [5, -1, 0], [8, -3, 2], [5, 0, 5]],
)
def test_cpu_management(n_cpu, taken, left):
    zfit.run.set_n_cpu(n_cpu=n_cpu)
    _cpu = copy.deepcopy(zfit.run._cpu)
    assert zfit.run.n_cpu == n_cpu
    with zfit.run.aquire_cpu(max_cpu=taken) as cpus:
        assert zfit.run.n_cpu == left
        assert isinstance(cpus, list)
        assert len(cpus) == n_cpu - left
    assert zfit.run.n_cpu == n_cpu
    assert _cpu == zfit.run._cpu


def test_set_cpus_explicit_after_import():
    """Test that CPU parallelism can be set after importing zfit (v0.25 behavior restored)."""
    code = """
import os
os.environ['ZFIT_DISABLE_TF_WARNINGS'] = '1'

import zfit
zfit.run.set_cpus_explicit(intra=2, inter=3)

import tensorflow as tf

# Verify that the settings were applied
intra = tf.config.threading.get_intra_op_parallelism_threads()
inter = tf.config.threading.get_inter_op_parallelism_threads()

print(f"intra={intra},inter={inter}")
assert intra == 2, f"Expected intra=2, got {intra}"
assert inter == 3, f"Expected inter=3, got {inter}"
"""
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        timeout=60
    )

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        pytest.fail(f"Test failed with return code {result.returncode}")

    # Parse output
    output = result.stdout.strip()
    assert "intra=2,inter=3" in output, f"Expected 'intra=2,inter=3' in output, got: {output}"


def test_set_cpus_explicit_after_tf_init_raises():
    """Test that calling set_cpus_explicit after TF is initialized raises a helpful error."""
    code = """
import os
os.environ['ZFIT_DISABLE_TF_WARNINGS'] = '1'

import zfit
import tensorflow as tf

# Trigger TF initialization
_ = tf.constant(1)

try:
    zfit.run.set_cpus_explicit(intra=2, inter=2)
    print("FAIL: Expected RuntimeError")
    exit(1)
except RuntimeError as e:
    error_msg = str(e)
    print(f"SUCCESS: Got expected RuntimeError")
    # Check that the error message is helpful
    assert "after" in error_msg.lower() and "initialized" in error_msg.lower(), f"Error message should mention initialization: {error_msg}"
    assert "immediately after" in error_msg.lower(), f"Error message should mention calling immediately after import: {error_msg}"
    exit(0)
"""
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        timeout=60
    )

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        pytest.fail(f"Test failed with return code {result.returncode}")
