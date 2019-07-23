#  Copyright (c) 2019 zfit

import zfit


class BkgPdf_cosines(zfit.pdf.BaseFunctor):
    """PDF for DSL background

    Angular distribution obtained in the total PDF (using LHCb convention JHEP 02 (2016) 104)
        i.e. the valid of the angles is given for
            - phi: [-pi, pi]
            - theta_K: [0, pi]
            - theta_l: [0, pi]

        The function is normalized over a finite range and therefore a PDF.

    """

    def __init__(self, costheta_l, costheta_k, phi, name="3dAngularPolynomial"):
        pdfs = [costheta_l, costheta_k, phi]
        super().__init__(pdfs=pdfs, name=name)

    _N_OBS = 3

    def _unnormalized_pdf(self, x):
        costheta_l_data, costheta_k_data, phi_data = x.unstack_x()
        costheta_l, costheta_k, phi = self.pdfs

        pdf = costheta_l.pdf(costheta_l_data) + costheta_k.pdf(costheta_k_data), phi_data.pdf(phi_data)
        return pdf


def create_angular():
    # create parameters
    # c00 = zfit.Parameter(...)
    # ... and so on

    costheta_l_space = zfit.Space('costheta_l', limits=(-1, 1))
    costheta_K_space = zfit.Space('costheta_k', limits=(-1, 1))
    phi_space = zfit.Space('phi', limits=(-pi, pi))

    # this part could also be moved inside the __init__, but then the init would need to take all
    # the coefficients and the spaces as arguments
    costheta_l = zfit.pdf.Chebyshev(obs=costheta_l_space, coeffs=[c00, c01, c02])
    costheta_k = zfit.pdf.Chebyshev(obs=costheta_K_space, coeffs=[c10, c11, c12])
    phi = zfit.pdf.Chebyshev(obs=phi_space, coeffs=[c20, c21, c22])

    return BkgPdf_cosines(costheta_l=costheta_l, costheta_k=costheta_k, phi=phi)
