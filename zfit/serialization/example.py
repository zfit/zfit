#  Copyright (c) 2020 zfit


import zfit
from zfit.serialization.arranger import IndependentParamArranger
from zfit.serialization.repr import IndependentParamRepr, StrRepr, FloatRepr

param = zfit.Parameter('asdf', 10)
param_repr = IndependentParamRepr(uid="IndepParam",
                                  fields={'name'     : StrRepr(uid='str', value=param.name),
                                          'value'    : FloatRepr(uid='float', value=float(param.value())),
                                          'step_size': FloatRepr(uid='float', value=float(param.step_size))
                                          })

dump = IndependentParamArranger().dump(param_repr)
print(dump)
load_dump = IndependentParamArranger().load(dump)
print(load_dump)
# load_dump.fields['name'].value += '_loaded'
# param_loaded = zfit.Parameter(**{k: rep.value for k, rep in load_dump.fields.items()})
# print(param_loaded)
