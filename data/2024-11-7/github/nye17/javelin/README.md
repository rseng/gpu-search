# https://github.com/nye17/javelin

```console
MANIFEST:javelin/gp/GPutils.py
javelin/gp/NearlyFullRankCovariance.py:from .GPutils import regularize_array, trisolve
javelin/gp/BasisCovariance.py:from .GPutils import regularize_array, trisolve
javelin/gp/FullRankCovariance.py:from .GPutils import regularize_array, trisolve
javelin/gp/Realization.py:from .GPutils import observe, trisolve, regularize_array, caching_call
javelin/gp/Covariance.py:from .GPutils import regularize_array, trisolve, square_and_sum
javelin/gp/Mean.py:from .GPutils import regularize_array, trisolve
javelin/gp/__init__.py:__modules__ = [ 'GPutils',
javelin/gp/__init__.py:from .GPutils import *
javelin/predict.py:from .gp import Mean, Covariance, observe, Realization, GPutils
javelin/predict.py:        m, v = GPutils.point_eval(self.M, self.C, jwant)
javelin/spear.py:from javelin.gp.GPutils import regularize_array

```
