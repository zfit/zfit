import zfit
from zfit import ztf


class CustomPDF(zfit.pdf.ZPDF):
    _PARAMS = ['alpha']  # specify which parameters to take

    def _unnormalized_pdf(self, x):  # overwrite the method
        data = x.unstack_x()
        alpha = self.params['alpha']

        return ztf.exp(alpha * data)


obs = zfit.Space("obs1", limits=(-4, 4))

custom_pdf = CustomPDF(obs=obs, alpha=0.2)

integral = custom_pdf.integrate(limits=(-1, 2))
sample = custom_pdf.sample(n=1000)
probs = custom_pdf.pdf(sample)
