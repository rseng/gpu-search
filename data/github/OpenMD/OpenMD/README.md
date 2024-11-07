# https://github.com/OpenMD/OpenMD

```console
CMakeLists.txt:                -i applications/hydrodynamics/HydroCmd.hpp
CMakeLists.txt:                -i applications/hydrodynamics/HydroCmd.cpp
CMakeLists.txt:                src/applications/hydrodynamics/HydroCmd.cpp)
src/brains/SimInfo.cpp:    molToProcMap_.resize(nGlobalMols_);
src/brains/SimCreator.cpp:    std::vector<int> molToProcMap(nGlobalMols, -1);  // default to an error
src/brains/SimCreator.cpp:              molToProcMap[i] = which_proc;
src/brains/SimCreator.cpp:              molToProcMap[i] = which_proc;
src/brains/SimCreator.cpp:              molToProcMap[i] = which_proc;
src/brains/SimCreator.cpp:          molToProcMap[i] = which_proc;
src/brains/SimCreator.cpp:      MPI_Bcast(&molToProcMap[0], nGlobalMols, MPI_INT, 0, MPI_COMM_WORLD);
src/brains/SimCreator.cpp:      MPI_Bcast(&molToProcMap[0], nGlobalMols, MPI_INT, 0, MPI_COMM_WORLD);
src/brains/SimCreator.cpp:    info->setMolToProcMap(molToProcMap);
src/brains/SimInfo.hpp:      assert(globalIndex >= 0 && globalIndex < molToProcMap_.size());
src/brains/SimInfo.hpp:      return molToProcMap_[globalIndex];
src/brains/SimInfo.hpp:     * Set MolToProcMap array
src/brains/SimInfo.hpp:    void setMolToProcMap(const std::vector<int>& molToProcMap) {
src/brains/SimInfo.hpp:      molToProcMap_ = molToProcMap;
src/brains/SimInfo.hpp:     * The size of molToProcMap_ is equal to total number of molecules
src/brains/SimInfo.hpp:    std::vector<int> molToProcMap_;
src/applications/hydrodynamics/Hydro.cpp:#include "HydroCmd.hpp"
src/applications/hydrodynamics/Hydro.ggo:# Input file for gengetopt. This file generates HydroCmd.cpp and
src/applications/hydrodynamics/Hydro.ggo:# HydroCmd.hpp for parsing command line arguments using getopt and
src/applications/hydrodynamics/Hydro.ggo:args "--no-handle-error --include-getopt --show-required --unamed-opts --file-name=HydroCmd --c-extension=cpp --header-extension=hpp"
src/applications/hydrodynamics/HydroCmd.hpp:/** @file HydroCmd.hpp
src/applications/hydrodynamics/HydroCmd.hpp:#ifndef HYDROCMD_H
src/applications/hydrodynamics/HydroCmd.hpp:#define HYDROCMD_H
src/applications/hydrodynamics/HydroCmd.hpp:#endif /* HYDROCMD_H */
src/applications/hydrodynamics/HydroCmd.cpp:  gengetopt --no-handle-error --include-getopt --show-required --unamed-opts --file-name=HydroCmd --c-extension=cpp --header-extension=hpp
src/applications/hydrodynamics/HydroCmd.cpp:#include "HydroCmd.hpp"

```
