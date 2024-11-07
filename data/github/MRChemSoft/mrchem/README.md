# https://github.com/MRChemSoft/mrchem

```console
external/catch/catch.hpp:#if defined(__GNUC__) && !defined(__clang__) && !defined(__ICC) && !defined(__CUDACC__) && !defined(__LCC__)
external/catch/catch.hpp:#  if !defined(__ibmxl__) && !defined(__CUDACC__)
external/catch/catch.hpp:        // Code accompanying the article "Approximating the erfinv function" in GPU Computing Gems, Volume 2
recipes/recipe_openmpi4.0.py:Stage0 += ucx(cuda=False, ofed=True)
recipes/recipe_openmpi4.0.py:Stage0 += openmpi(cuda=False,
recipes/Singularity.openmpi4.0:    cd /var/tmp/ucx-1.9.0 &&   ./configure --prefix=/usr/local/ucx --disable-assertions --disable-debug --disable-doxygen-doc --disable-logging --disable-params-check --enable-optimizations --with-rdmacm --with-verbs --without-cuda
recipes/Singularity.openmpi4.0:    cd /var/tmp/openmpi-4.0.5 &&  CC=gcc CXX=g++ F77=gfortran F90=gfortran FC=gfortran ./configure --prefix=/usr/local/openmpi --disable-getpwuid --enable-orterun-prefix-by-default --with-pmi=/usr/local/slurm-pmi2 --with-ucx=/usr/local/ucx --without-cuda --without-verbs
src/utils/Bank.cpp:    return openAccount(iclient, comm);
src/utils/Bank.cpp:int Bank::openAccount(int iclient, MPI_Comm comm) {
src/utils/Bank.cpp:    this->account_id = dataBank.openAccount(iclient, comm);
src/utils/Bank.h:    int openAccount(int iclient, MPI_Comm comm);

```
