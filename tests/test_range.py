from __future__ import print_function, division, absolute_import

import copy
from unittest import TestCase

from zfit.core.limits import Range, convert_to_range, iter_limits

limit1 = ((1, 4), (2, 3.5), (-1, 5))
limit1_true = copy.deepcopy(limit1)
limit1_area = 27.
limit1_dims = (0, 4, 5)
limit1_dims_true = limit1_dims

limit2 = ((0, 2), (-2, 2, 3, 5))
limit2_true = copy.deepcopy(limit2)
limit2_area = 12.
limit2_dims = (1, 3)
limit2_dims_true = limit2_dims

limit3_1dim_1pair = (-1.2, 2.0)
limit3_1dim_1pair_true = ((-1.2, 2.0),)
limit3_1dim_1pair_area = 3.2
limit3_1dim_1pair_dims = 1
limit3_1dim_1pair_dims_true = (1,)

limit3_1dim_1pair_0axis = (-1.2, 2.0)
limit3_1dim_1pair_true_0axis = ((-1.2, 2.0),)
limit3_1dim_1pair_area_0axis = 3.2
limit3_1dim_1pair_dims_0axis = 0
limit3_1dim_1pair_dims_true_0axis = (0,)

limit3_1dim_3pair = (-1.3, 2.1, 4.5, 5.6, -4.3, -1.4)
limit3_1dim_3pair_true = ((-4.3, -1.4, -1.3, 2.1, 4.5, 5.6),)
limit3_1dim_3pair_area = 7.4
limit3_1dim_3pair_dims = 1
limit3_1dim_3pair_dims_true = (1,)

limit4 = ((1, 4, 5, 7), (1.5, 3.5), (-1, 5, -4, -2))
limit4_true = ((1, 4, 5, 7), (1.5, 3.5), (-4, -2, -1, 5))
limit4_area = 80.
limit4_dims = (0, 4, 5)
limit4_dims_true = limit4_dims

limit4_subrange = ((1, 3, 5, 7), (2, 3.5), (-1, 5, -4, -2.3))
limit4_subrange_true = ((1, 3, 5, 7), (2, 3.5), (-4, -2.3, -1, 5))

limit5 = ((1, 4, 5, 6), (1, 3))
limit5_subarea = (6, 2)
limit5_subarea_rel = (0.75, 0.25)

def limits_equal(limit1, limit2):
    for axis1, axis2 in zip(limit1, limit2):
        for lower1, upper1 in iter_limits(axis1):
            for lower2, upper2 in iter_limits(axis2):
                if lower1 == lower2 and upper1 == upper2:
                    break
            else:  # for each lower1, upper2 there has to be the same pair in limit2
                return False
    return True


class TestRange(TestCase):
    def setUp(self):
        self.limit1_range = Range(limits=limit1, dims=limit1_dims)
        self.limit2_range = Range(limits=limit2, dims=limit2_dims)
        self.limit3_1pair_range = Range(limits=limit3_1dim_1pair, dims=limit3_1dim_1pair_dims)
        self.limit3_1pair_0axis_range = Range(limits=limit3_1dim_1pair_0axis, dims=limit3_1dim_1pair_dims_0axis)
        self.limit3_3pair_range = Range(limits=limit3_1dim_3pair, dims=limit3_1dim_3pair_dims)
        self.limit4_range = Range(limits=limit4, dims=limit4_dims)

    def test_convert_to_range(self):
        limit1_range = convert_to_range(limits=limit1, dims=limit1_dims)
        limit1_range_same = convert_to_range(limit1_range)
        self.assertIs(limit1_range, limit1_range_same)
        limit1_range2 = convert_to_range(limits=limit1, dims=limit1_dims)
        self.assertIsNot(limit1_range, limit1_range2)
        self.assertEqual(limit1_range, limit1_range2)

    def test_area(self):
        self.assertEqual(self.limit1_range.area, limit1_area)
        self.assertEqual(self.limit2_range.area, limit2_area)
        self.assertEqual(self.limit3_1pair_range.area, limit3_1dim_1pair_area)
        self.assertEqual(self.limit3_3pair_range.area, limit3_1dim_3pair_area)
        self.assertEqual(self.limit4_range.area, limit4_area)
        range5 = Range(limits=limit5, dims=Range.FULL)
        self.assertEqual(range5.area_by_boundaries(), limit5_subarea)
        self.assertEqual(range5.area_by_boundaries(rel=True), limit5_subarea_rel)
        self.assertEqual(range5.area, sum(limit5_subarea))

    def test_dims(self):
        self.assertEqual(self.limit1_range.dims, limit1_dims_true)
        self.assertEqual(self.limit2_range.dims, limit2_dims_true)
        self.assertEqual(self.limit3_1pair_range.dims, limit3_1dim_1pair_dims_true)
        self.assertEqual(self.limit3_3pair_range.dims, limit3_1dim_3pair_dims_true)
        self.assertEqual(self.limit3_1pair_0axis_range.dims, limit3_1dim_1pair_dims_true_0axis)
        self.assertEqual(self.limit4_range.dims, limit4_dims_true)

    def test_as_tuple(self):
        self.assertTrue(limits_equal(self.limit1_range.get_limits(), limit1_true))
        self.assertTrue(limits_equal(self.limit2_range.get_limits(), limit2_true))
        self.assertTrue(limits_equal(self.limit3_1pair_range.get_limits(), limit3_1dim_1pair_true))
        self.assertTrue(limits_equal(self.limit3_1pair_0axis_range.get_limits(), limit3_1dim_1pair_true_0axis))
        self.assertTrue(limits_equal(self.limit3_3pair_range.get_limits(), limit3_1dim_3pair_true))
        self.assertTrue(limits_equal(self.limit4_range.get_limits(), limit4_true))

    def test_comparison(self):
        self.assertFalse(self.limit1_range <= self.limit2_range)
        self.assertFalse(self.limit1_range >= self.limit2_range)
        self.assertFalse(self.limit1_range == self.limit2_range)
        self.assertFalse(self.limit3_1pair_range <= self.limit3_1pair_0axis_range)
        self.assertFalse(self.limit3_1pair_range >= self.limit3_1pair_0axis_range)
        self.assertFalse(self.limit3_1pair_range == self.limit3_1pair_0axis_range)
        # self.assertTrue(self.limit3_1pair_range <= self.limit3_3pair_range)  # TODO add test with Nones
        self.assertFalse(self.limit3_1pair_range >= self.limit3_3pair_range)
        self.assertFalse(self.limit3_1pair_range == self.limit3_3pair_range)
        # self.assertTrue(self.limit3_3pair_range == Range(dims=limit3_1dim_3pair_dims))
        # limit4_subrange_range = Range(limit4_subrange, dims=limit4_dims)  # TODO add test with Nones in limits
        # self.assertTrue(self.limit4_range > limit4_subrange_range)
        # self.assertTrue(limit4_subrange_range < self.limit4_range)
        # self.assertFalse(self.limit4_range == limit4_subrange_range)

    def test_exception(self):
        # TODO: make tests valid again

        invalid_dim = (0, 4, 6)
        invalid_limits = (1, 3, 4)
        invalid_limits2 = ((1, 2), (4, 5, 6, 7))
        valid_dims2 = (0, 2)
        valid_limits = ((1, 2), (4, 5, 6, 7))
        with self.assertRaises(ValueError) as context:
            Range(limits=valid_limits, dims=invalid_dim)
        with self.assertRaises(ValueError) as context:
            Range(limits=invalid_limits, dims=valid_dims2)
        with self.assertRaises(ValueError) as context:
            Range(limits=invalid_limits2, dims=invalid_dim)
        # valid_range_with_none = Range()
        # valid_range = Range(dims=valid_dims2)
        # self.assertTrue(valid_range_with_none == valid_range)

    def test_conversion(self):
        simple_limits = ((1, 2), (4, 5, 6, 7))
        simple_dims = (1, 3)
        simple_lower, simple_upper = [((1, 4), (1, 6)), ((2, 5), (2, 7))]
        simple_lower2, simple_upper2 = [((1, 4), (3, 6)), ((2, 5), (4, 7))]
        simple_range = Range(limits=simple_limits, dims=simple_dims)
        simple_range2 = Range.from_boundaries(lower=simple_lower2, upper=simple_upper2, dims=simple_dims)
        with self.assertRaises(ValueError) as context:
            simple_range2.get_limits()
        lower, upper = simple_range.get_boundaries()
        self.assertEqual(simple_lower, lower)
        self.assertEqual(simple_upper, upper)
        limits4_lower, limits4_upper = self.limit4_range.get_boundaries()
        limits4_reconversed = Range.limits_from_boundaries(limits4_lower, limits4_upper)
        self.assertEqual(limits4_reconversed, self.limit4_range.get_limits())
        # self.assertEqual(self.limit4_range, )

    def test_complex_example(self):
        complex_lower = ((-3,-999, -11), (3, -999, 10), (-3, -999, 15), (7, -999, -11))
        complex_upper = ((1, 999, 1), (5, 999, 13), (1, 999, 20), (10, 999, 1))
        dims = (0, 4, 8)
        true_complex_sub_areas_by_boundaries = (48., 6., 20., 36.)
        complex_range = Range.from_boundaries(lower=complex_lower, upper=complex_upper, dims=dims)
        complex_subrange = complex_range.subspace(dims=(0, 8))

        self.assertEqual(set(complex_subrange.area_by_boundaries()), set(true_complex_sub_areas_by_boundaries))
        with self.assertRaises(ValueError) as context:
            complex_subrange.get_limits()

    def test_subspace(self):  # TODO
        limits = ((1, 2), (4, 5, 6, 7), (-1, 5, 6, 9))
        sub_limits = ((1, 2), (-1, 5, 6, 9))
        dims = (1, 3, 6)
        sub_dims = (1, 6)
        range_ = Range(limits=limits, dims=dims)
        # print("DEBUG": dims =", dims)
        sub_range = range_.subspace(dims=sub_dims)
        sub_range_true = Range(limits=sub_limits, dims=sub_dims)
        self.assertEqual(sub_range_true, sub_range)
