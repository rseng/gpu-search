# https://github.com/LoganAMorrison/Hazma

```console
hazma/experimental/pseudo_scalar_mediator/_widths.py:            gpuu = self.gpuu
hazma/experimental/pseudo_scalar_mediator/_widths.py:                    * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh)
hazma/experimental/pseudo_scalar_mediator/_widths.py:                    * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh) ** 2
hazma/experimental/pseudo_scalar_mediator/_widths.py:            gpuu = self.gpuu
hazma/experimental/pseudo_scalar_mediator/_widths.py:                        + fpi * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh)
hazma/experimental/pseudo_scalar_mediator/_widths.py:                        * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh)
hazma/experimental/pseudo_scalar_mediator/_widths.py:                        * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh) ** 2
hazma/experimental/pseudo_scalar_mediator/_widths.py:                        + fpi * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh)
hazma/experimental/pseudo_scalar_mediator/_widths.py:                        * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh)
hazma/experimental/pseudo_scalar_mediator/_widths.py:                        * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh) ** 2
hazma/experimental/pseudo_scalar_mediator/_msqrd.py:    gpuu = model.gpuu
hazma/experimental/pseudo_scalar_mediator/_msqrd.py:            + b0 * fpi * gpuu * vh
hazma/experimental/pseudo_scalar_mediator/_msqrd.py:            + b0 * fpi * gpuu * vh
hazma/experimental/pseudo_scalar_mediator/_msqrd.py:                - 2 * b0 * fpi * gpuu * vh
hazma/experimental/pseudo_scalar_mediator/_msqrd.py:                + b0 * fpi * gpuu * vh
hazma/experimental/pseudo_scalar_mediator/_msqrd.py:                + b0 * fpi * gpuu * vh
hazma/experimental/pseudo_scalar_mediator/_msqrd.py:    gpuu = model.gpuu
hazma/experimental/pseudo_scalar_mediator/_msqrd.py:            + b0 * fpi * gpuu * vh
hazma/experimental/pseudo_scalar_mediator/_msqrd.py:            + b0 * fpi * gpuu * vh
hazma/experimental/pseudo_scalar_mediator/_msqrd.py:                - 2 * b0 * fpi * gpuu * vh
hazma/experimental/pseudo_scalar_mediator/_msqrd.py:                + b0 * fpi * gpuu * vh
hazma/experimental/pseudo_scalar_mediator/_msqrd.py:                + b0 * fpi * gpuu * vh
hazma/experimental/pseudo_scalar_mediator/_msqrd.py:    gpuu = model.gpuu
hazma/experimental/pseudo_scalar_mediator/_msqrd.py:        * (fpi * gpGG * mdq - fpi * gpGG * muq + fpi * gpdd * vh - fpi * gpuu * vh) ** 2
hazma/experimental/pseudo_scalar_mediator/_msqrd.py:        * (fpi * gpGG * mdq - fpi * gpGG * muq + fpi * gpdd * vh - fpi * gpuu * vh)
hazma/experimental/pseudo_scalar_mediator/_msqrd.py:            * (fpi * gpGG * mdq - fpi * gpGG * muq + fpi * gpdd * vh - fpi * gpuu * vh)
hazma/experimental/pseudo_scalar_mediator/_msqrd.py:            * (fpi * gpGG * mdq - fpi * gpGG * muq + fpi * gpdd * vh - fpi * gpuu * vh)
hazma/experimental/pseudo_scalar_mediator/_msqrd.py:                + 4 * fpi * gpuu * vh
hazma/experimental/pseudo_scalar_mediator/_proto.py:    gpuu: float
hazma/experimental/pseudo_scalar_mediator/_cross_sections.py:        gpuu = model.gpuu
hazma/experimental/pseudo_scalar_mediator/_cross_sections.py:                * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh)
hazma/experimental/pseudo_scalar_mediator/_cross_sections.py:                * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh) ** 2
hazma/experimental/pseudo_scalar_mediator/_cross_sections.py:        gpuu = model.gpuu
hazma/experimental/pseudo_scalar_mediator/_cross_sections.py:                    + fpi * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh)
hazma/experimental/pseudo_scalar_mediator/_cross_sections.py:                    * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh)
hazma/experimental/pseudo_scalar_mediator/_cross_sections.py:                    * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh) ** 2
hazma/experimental/pseudo_scalar_mediator/__init__.py:    def __init__(self, mx, mp, gpxx, gpuu, gpdd, gpss, gpee, gpmumu, gpGG, gpFF):
hazma/experimental/pseudo_scalar_mediator/__init__.py:        self._gpuu = gpuu
hazma/experimental/pseudo_scalar_mediator/__init__.py:    def gpuu(self):
hazma/experimental/pseudo_scalar_mediator/__init__.py:        return self._gpuu
hazma/experimental/pseudo_scalar_mediator/__init__.py:    @gpuu.setter
hazma/experimental/pseudo_scalar_mediator/__init__.py:    def gpuu(self, gpuu):
hazma/experimental/pseudo_scalar_mediator/__init__.py:        self._gpuu = gpuu
hazma/experimental/pseudo_scalar_mediator/__init__.py:        * Depends on gpuu, gpdd, gpGG and mp.
hazma/experimental/pseudo_scalar_mediator/__init__.py:        eps = b0 * fpi * (self.gpuu - self.gpdd + (muq - mdq) / vh * self.gpGG)
hazma/experimental/pseudo_scalar_mediator/__init__.py:    def __init__(self, mx, mp, gpxx, gpup, gpdown, gpll):
hazma/experimental/pseudo_scalar_mediator/__init__.py:        self._gpup = gpup
hazma/experimental/pseudo_scalar_mediator/__init__.py:        gpGG = 2.0 * gpup + gpdown
hazma/experimental/pseudo_scalar_mediator/__init__.py:        gpFF = gpll + (8.0 * gpup + gpdown) / 9.0
hazma/experimental/pseudo_scalar_mediator/__init__.py:            gpup * yu,
hazma/experimental/pseudo_scalar_mediator/__init__.py:        self.gpFF = gpll + (8.0 * self.gpup + self.gpdown) / 9.0
hazma/experimental/pseudo_scalar_mediator/__init__.py:    def gpup(self):
hazma/experimental/pseudo_scalar_mediator/__init__.py:        return self._gpup
hazma/experimental/pseudo_scalar_mediator/__init__.py:    @gpup.setter
hazma/experimental/pseudo_scalar_mediator/__init__.py:    def gpup(self, gpup):
hazma/experimental/pseudo_scalar_mediator/__init__.py:        self._gpup = gpup
hazma/experimental/pseudo_scalar_mediator/__init__.py:        self.gpuu = gpup * yu
hazma/experimental/pseudo_scalar_mediator/__init__.py:        self.gpFF = self.gpll + (8.0 * gpup + self.gpdown) / 9.0
hazma/experimental/pseudo_scalar_mediator/__init__.py:        self.gpGG = 2.0 * gpup + self.gpdown
hazma/experimental/pseudo_scalar_mediator/__init__.py:        self.gpFF = self.gpll + (8.0 * self.gpup + gpdown) / 9.0
hazma/experimental/pseudo_scalar_mediator/__init__.py:        self.gpGG = 2.0 * self.gpup + gpdown

```
