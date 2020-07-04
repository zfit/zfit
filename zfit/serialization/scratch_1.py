#  Copyright (c) 2020 zfit

from dataclasses import asdict

from zfit import Parameter
from zfit.serialization.base_parameter_serializer import BaseParameterSerializer

p = Parameter('mu', 3)
serializer = BaseParameterSerializer()
rep = p.to_repr()
print(asdict(rep))
print(serializer.dump(rep))
