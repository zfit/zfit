from __future__ import print_function, division, absolute_import

import copy
from unittest import TestCase

from zfit.core.limits import Range, convert_to_range

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


class TestRange(TestCase):
    def setUp(self):
        self.limit1_range = Range(limits=limit1, dims=limit1_dims)
        self.limit2_range = Range(limits=limit2, dims=limit2_dims)
        self.limit3_1pair_range = Range(limits=limit3_1dim_1pair, dims=limit3_1dim_1pair_dims)
        self.limit3_1pair_0axis_range = Range(limits=limit3_1dim_1pair_0axis,
                                              dims=limit3_1dim_1pair_dims_0axis)
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

    def test_dims(self):
        self.assertEqual(self.limit1_range.dims, limit1_dims_true)
        self.assertEqual(self.limit2_range.dims, limit2_dims_true)
        self.assertEqual(self.limit3_1pair_range.dims, limit3_1dim_1pair_dims_true)
        self.assertEqual(self.limit3_3pair_range.dims, limit3_1dim_3pair_dims_true)
        self.assertEqual(self.limit3_1pair_0axis_range.dims, limit3_1dim_1pair_dims_true_0axis)
        self.assertEqual(self.limit4_range.dims, limit4_dims_true)

    def test_as_tuple(self):
        self.assertEqual(self.limit1_range.as_tuple(), limit1_true)
        self.assertEqual(self.limit2_range.as_tuple(), limit2_true)
        self.assertEqual(self.limit3_1pair_range.as_tuple(), limit3_1dim_1pair_true)
        self.assertEqual(self.limit3_1pair_0axis_range.as_tuple(), limit3_1dim_1pair_true_0axis)
        self.assertEqual(self.limit3_3pair_range.as_tuple(), limit3_1dim_3pair_true)
        self.assertEqual(self.limit4_range.as_tuple(), limit4_true)

    def test_comparison(self):
        self.assertFalse(self.limit1_range < self.limit2_range)
        self.assertFalse(self.limit1_range > self.limit2_range)
        self.assertFalse(self.limit1_range == self.limit2_range)
        self.assertFalse(self.limit3_1pair_range < self.limit3_1pair_0axis_range)
        self.assertFalse(self.limit3_1pair_range > self.limit3_1pair_0axis_range)
        self.assertFalse(self.limit3_1pair_range == self.limit3_1pair_0axis_range)
        self.assertTrue(self.limit3_1pair_range < self.limit3_3pair_range)
        self.assertFalse(self.limit3_1pair_range > self.limit3_3pair_range)
        self.assertFalse(self.limit3_1pair_range == self.limit3_3pair_range)
        self.assertTrue(self.limit3_3pair_range == Range(limit3_1dim_3pair,
                                                         dims=limit3_1dim_3pair_dims))
        limit4_subrange_range = Range(limit4_subrange, dims=limit4_dims)
        self.assertTrue(self.limit4_range > limit4_subrange_range)
        self.assertTrue(limit4_subrange_range < self.limit4_range)
        self.assertFalse(self.limit4_range == limit4_subrange_range)

    def test_exception(self):
        invalid_dim = (0, 4, 6)
        invalid_limits = (1, 3, 4)
        invalid_limits2 = ((1, 2), (4, 5, 6, 7))
        valid_dims2 = (0, 2)
        valid_limits = ((1, 2), None, (4, 5, 6, 7))
        with self.assertRaises(ValueError) as context:
            Range(invalid_limits, invalid_dim)
        with self.assertRaises(ValueError) as context:
            Range(invalid_limits2, invalid_dim)
        with self.assertRaises(ValueError) as context:
            Range(invalid_limits2)
        valid_range_with_none = Range(valid_limits)
        valid_range = Range(invalid_limits2, dims=valid_dims2)
        self.assertTrue(valid_range_with_none == valid_range)

    def test_conversion(self):
        simple_limits = ((1, 2), (4, 5, 6, 7))
        simple_dims = (1, 3)
        simple_lower, simple_upper = [((1, 4), (1, 6)), ((2, 5), (2, 7))]
        simple_range = Range(limits=simple_limits, dims=simple_dims)
        lower, upper = simple_range.get_boundaries()
        self.assertEqual(simple_lower, lower)
        self.assertEqual(simple_upper, upper)
        limits4_lower, limits4_upper = self.limit4_range.get_boundaries()
        limits4_reconversed = Range.extract_boundaries(limits4_lower, limits4_upper)
        self.assertEqual(limits4_reconversed, self.limit4_range.as_tuple())
        # self.assertEqual(self.limit4_range, )

    def test_subspace(self):
        limits = ((1, 2), (4, 5, 6, 7), (-1, 5, 6, 9))
        sub_limits = ((1, 2), (-1, 5, 6, 9))
        dims = (1, 3, 6)
        sub_dims = (1, 6)
        range_ = Range(limits=limits, dims=dims)
        # print("DEBUG": dims =", dims)
        sub_range = range_.subspace(dims=sub_dims)
        sub_range_true = Range(limits=sub_limits, dims=sub_dims)
        self.assertEqual(sub_range_true, sub_range)

