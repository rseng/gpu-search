# https://github.com/austinpeel/herculens

```console
herculens/MassModel/Profiles/dpie.py:    from herculens.MassModel.Profiles.glee.piemd_jax import Piemd_GPU
herculens/MassModel/Profiles/dpie.py:            # NOTE: first 4 arguments of Piemd_GPU do not matter for our use, so we give zeros
herculens/MassModel/Profiles/dpie.py:            self._piemd = Piemd_GPU(0., 0., 0., 0., xx=x, yy=y)
herculens/MassModel/Profiles/dpie.py:        self._piemd = Piemd_GPU(0., 0., 0., 0., xx=x, yy=y)
README.md:- [Fast GPU-boosted lensed quasar simulations](https://github.com/aymgal/herculens_workspace/blob/main/notebooks/herculens__Fast_lensed_quasar_simulations.ipynb)

```
