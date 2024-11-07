# https://github.com/Cantera/cantera

```console
interfaces/dotnet/.gitignore:[Dd]ebugPublic/
include/cantera/thermo/LatticePhase.h:    void getPureGibbs(double* gpure) const override;
include/cantera/thermo/VPStandardStateTP.h:    void getPureGibbs(double* gpure) const override;
include/cantera/thermo/ThermoPhase.h:     * @param gpure  Output vector of standard state Gibbs free energies.
include/cantera/thermo/ThermoPhase.h:    virtual void getPureGibbs(double* gpure) const {
include/cantera/thermo/IdealGasPhase.h:    void getPureGibbs(double* gpure) const override;
include/cantera/thermo/MixtureFugacityTP.h:     * @param[out] gpure   Array of standard state Gibbs free energies. length =
include/cantera/thermo/MixtureFugacityTP.h:    void getPureGibbs(double* gpure) const override;
include/cantera/thermo/IdealSolidSolnPhase.h:     * @param gpure  Output vector of Gibbs functions for species. Length: m_kk.
include/cantera/thermo/IdealSolidSolnPhase.h:    void getPureGibbs(double* gpure) const override;
include/cantera/thermo/SingleSpeciesTP.h:    void getPureGibbs(double* gpure) const override;
src/thermo/IdealGasPhase.cpp:void IdealGasPhase::getPureGibbs(double* gpure) const
src/thermo/IdealGasPhase.cpp:    scale(gibbsrt.begin(), gibbsrt.end(), gpure, RT());
src/thermo/IdealGasPhase.cpp:        gpure[k] += tmp;
src/thermo/LatticePhase.cpp:void LatticePhase::getPureGibbs(double* gpure) const
src/thermo/LatticePhase.cpp:        gpure[k] = RT() * gibbsrt[k] + delta_p * m_speciesMolarVolume[k];
src/thermo/IdealSolidSolnPhase.cpp:void IdealSolidSolnPhase::getPureGibbs(double* gpure) const
src/thermo/IdealSolidSolnPhase.cpp:        gpure[k] = RT() * gibbsrt[k] + delta_p * m_speciesMolarVolume[k];
src/thermo/SingleSpeciesTP.cpp:void SingleSpeciesTP::getPureGibbs(double* gpure) const
src/thermo/SingleSpeciesTP.cpp:    getGibbs_RT(gpure);
src/thermo/SingleSpeciesTP.cpp:    gpure[0] *= RT();

```
