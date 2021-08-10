#  Copyright (c) 2021 zfit
from zfit_interface.pdf import ZfitPDF

from zfit import convert_to_parameter, z
from zfit.core.func import Func
from zfit.util.exception import AlreadyExtendedPDFError, SpecificFunctionNotImplemented
import zfit_interface.typing as ztyping


class PDF(Func, ZfitPDF):

    def __init__(self, var, extended, norm, label=None):
        super().__init__(var=var, label=label)
        if norm is None:
            norm = self.space
        self.norm = norm
        if extended is not None:
            self._set_yield = extended

    def _set_yield(self, value):
        if self.is_extended:
            raise AlreadyExtendedPDFError(f"Cannot extend {self}, is already extended.")
        value = convert_to_parameter(value)
        self.add_cache_deps(value)
        self._yield = value

    def _pdf(self, var, norm):
        raise SpecificFunctionNotImplemented

    def pdf(self, var: ztyping.VarInputType, norm: ztyping.NormInputType = None, *,
            options=None) -> ztyping.PDFReturnType:
        """Probability density function, normalized over `norm`.

        Args:
          var: `float` or `double` `Tensor`.
          norm: :py:class:`~zfit.Space` to normalize over

        Returns:
          :py:class:`tf.Tensor` of type `self.dtype`.
        """
        var = self._convert_check_input_var(var)
        norm = self._convert_check_input_norm(norm, var=var)
        if var.space is not None:
            return self.integrate(limits=var, norm=norm, options=options)
        value = self._auto_pdf(var=var, norm=norm, options=options)
        return value
        # with self._convert_sort_x(var) as var:
        #     value = self._single_hook_pdf(x=var, norm_range=norm)
        #     if run.numeric_checks:
        #         z.check_numerics(value, message="Check if pdf output contains any NaNs of Infs")
        #     return z.to_real(value)

    @z.function(wraps='model')
    def _auto_pdf(self, var, norm, *, options=None):
        pass

    def integrate(self, limits, norm=None, *, var=None, options=None):
        var = self._convert_check_input_var(limits, var)
        if var.space is None:
            raise ValueError(f"No space is given to integrate of {self}, needs at least one.")
        norm = self._convert_check_input_norm(norm, var=var)
        value = self._auto_integrate(var=var, norm=norm, options=options)
        return value

    def _auto_integrate(self):
        pass
