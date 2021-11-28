#  Copyright (c) 2021 zfit

class HistogramPDF(BaseBinnedPDFV1):

    def __init__(self, data, sysshape=None, extended=None, norm=None, name="BinnedTemplatePDF"):
        obs = data.space
        if extended is None:
            extended = True
        if sysshape is None:
            sysshape = {}
        if sysshape is True:
            import zfit
            sysshape = {f'sysshape_{i}': zfit.Parameter(f'auto_sysshape_{self}_{i}', 1.) for i in
                        range(data.values().shape.num_elements())}
        params = {}
        params.update(sysshape)
        self._template_sysshape = sysshape
        if extended is True:
            self._automatically_extended = True
            if sysshape:
                import zfit

                def sumfunc(params):
                    values = self._data.values()
                    sysshape = list(params.values())
                    if sysshape:
                        sysshape_flat = tf.stack(sysshape)
                        sysshape = tf.reshape(sysshape_flat, values.shape)
                        values = values * sysshape
                    return znp.sum(values)

                from zfit.core.parameter import get_auto_number
                extended = zfit.ComposedParameter(f'TODO_name_selfmade_{get_auto_number()}', sumfunc, params=sysshape)

            else:
                extended = znp.sum(data.values())
        elif extended is not False:
            self._automatically_extended = False
        super().__init__(obs=obs, name=name, params=params, extended=extended, norm=norm)

        self._data = data

    def _ext_pdf(self, x, norm):
        counts = self._counts(x, norm)
        areas = np.prod(self._data.axes.widths, axis=0)
        density = counts / areas
        return density

    def _pdf(self, x, norm):
        counts = self._counts(x, norm)
        areas = np.prod(self._data.axes.widths, axis=0)
        density = counts / areas
        return density

    @supports(norm='norm')
    # @supports(norm=False)
    def _counts(self, x, norm=None):
        if not self._automatically_extended:
            raise SpecificFunctionNotImplemented
        values = self._data.values()
        sysshape = list(self._template_sysshape.values())
        if sysshape:
            sysshape_flat = tf.stack(sysshape)
            sysshape = tf.reshape(sysshape_flat, values.shape)
            values = values * sysshape
        return values

    @supports(norm='norm')
    def _rel_counts(self, x, norm=None):
        values = self._data.values()
        sysshape = list(self._template_sysshape.values())
        if sysshape:
            sysshape_flat = tf.stack(sysshape)
            sysshape = tf.reshape(sysshape_flat, values.shape)
            values = values * sysshape
        return values / znp.sum(values)
