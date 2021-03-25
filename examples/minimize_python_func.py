#  Copyright (c) 2021 zfit

import zfit

zfit.run.set_autograd_mode(False)
zfit.run.set_graph_mode(False)
minimizer_class, minimizer_kwargs, _ = minimizer_class_and_kwargs
minimizer = minimizer_class(**minimizer_kwargs)
func = scipy.optimize.rosen
func.errordef = 0.5
params = np.random.normal(size=5)

result = minimizer.minimize(func, params)
