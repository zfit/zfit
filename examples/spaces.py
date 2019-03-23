import zfit

simple_limit1 = zfit.Space(obs='obs1', limits=(-5, 1))
simple_limit2 = zfit.Space(obs='obs1', limits=(3, 7.5))

added_limits = simple_limit1 + simple_limit2
# OR equivalently
added_limits = simple_limit1.add(simple_limit2)

first_limit_lower = (-5, 6)
first_limit_upper = (5, 10)

second_limit_lower = (7, 12)
second_limit_upper = (9, 15)

third_limit_lower = (13, 20)
third_limit_upper = (14, 25)

lower = (first_limit_lower, second_limit_lower, third_limit_lower)
upper = (first_limit_upper, second_limit_upper, third_limit_upper)

limits = (lower, upper)

space1 = zfit.Space(obs=['obs1', 'obs2'], limits=limits)

assert space1.obs == ('obs1', 'obs2')
assert space1.n_obs == 2
assert space1.n_limits == 3
