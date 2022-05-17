#  Copyright (c) 2022 zfit

# Functors are PDfs and Functions that depend on other PDFs or Functions. They can be used to define
# in a custom way combinations of PDFs or wrapping a single PDF.
# An example would be to create the sum of two PDFs. Of course this is already implemented in zfit
# as the functor SumPDF([pdf1, pdf2], fracs=...). For advanced uses, you
# can define your own functor as demonstrated below.

import numpy as np

import zfit


class CombinePolynomials(zfit.pdf.BaseFunctor):
    """Example of a functor pdf that adds three polynomials.

    DEMONSTRATION PURPOSE ONLY, DO **NOT** USE IN REAL CASE.
    """

    def __init__(self, angle1, angle2, angle3, name="3dPolynomial"):
        pdfs = [angle1, angle2, angle3]
        super().__init__(pdfs=pdfs, name=name)

    _N_OBS = 3

    def _unnormalized_pdf(self, x):
        cosangle1_data, cosangle2_data, angle3_data = x.unstack_x()
        cosangle1, cosangle2, angle3 = self.pdfs

        pdf = (
            cosangle1.pdf(cosangle1_data)
            + cosangle2.pdf(cosangle2_data)
            + angle3.pdf(angle3_data)
        )
        return pdf


def create_angular():
    # create parameters
    # c00 = zfit.Parameter(...)
    # ... and so on

    cosangle1_space = zfit.Space("cos angle 1", limits=(-1, 1))
    cosangle2_space = zfit.Space("cos angle 2", limits=(-1, 1))
    angle3_space = zfit.Space("angle 3", limits=(-np.pi, np.pi))

    # this part could also be moved inside the __init__, but then the init would need to take all
    # the coefficients and the spaces as arguments
    cosangle1_pdf = zfit.pdf.Chebyshev(obs=cosangle1_space, coeffs=[c00, c01, c02])
    cosangle2_pdf = zfit.pdf.Chebyshev(obs=cosangle2_space, coeffs=[c10, c11, c12])
    angle3_pdf = zfit.pdf.Chebyshev(obs=angle3_space, coeffs=[c20, c21, c22])

    return CombinePolynomials(
        angle1=cosangle1_pdf, angle2=cosangle2_pdf, angle3=angle3_pdf
    )
