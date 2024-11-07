# https://github.com/christophmschaefer/miluphcuda

```console
damage.h: * This file is part of miluphcuda.
damage.h: * miluphcuda is free software: you can redistribute it and/or modify
damage.h: * miluphcuda is distributed in the hope that it will be useful,
damage.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
rhs.h: * This file is part of miluphcuda.
rhs.h: * miluphcuda is free software: you can redistribute it and/or modify
rhs.h: * miluphcuda is distributed in the hope that it will be useful,
rhs.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
gravity.h: * This file is part of miluphcuda.
gravity.h: * miluphcuda is free software: you can redistribute it and/or modify
gravity.h: * miluphcuda is distributed in the hope that it will be useful,
gravity.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
io.cu: * This file is part of miluphcuda.
io.cu: * miluphcuda is free software: you can redistribute it and/or modify
io.cu: * miluphcuda is distributed in the hope that it will be useful,
io.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
io.cu:        fprintf(stdout, "\nReading/initialising material constants and copy them to the GPU...\n");
io.cu:    transferMaterialsToGPU();
io.cu:    cudaMemcpyFromSymbol(&maxNodeIndex_host, maxNodeIndex, sizeof(int));
io.cu:    cudaVerify(cudaDeviceSynchronize());
io.cu:    cudaVerify(cudaDeviceSynchronize());
io.cu:	cudaVerifyKernel((calculatePressure<<<numberOfMultiprocessors * 4, NUM_THREADS_PRESSURE>>>()));
io.cu:    cudaVerify(cudaMemcpy(pointmass_host.x, pointmass_device.x, memorySizeForPointmasses, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(pointmass_host.y, pointmass_device.y, memorySizeForPointmasses, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(pointmass_host.vx, pointmass_device.vx, memorySizeForPointmasses, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(pointmass_host.vy, pointmass_device.vy, memorySizeForPointmasses, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(pointmass_host.ax, pointmass_device.ax, memorySizeForPointmasses, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(pointmass_host.ay, pointmass_device.ay, memorySizeForPointmasses, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(pointmass_host.z, pointmass_device.z, memorySizeForPointmasses, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(pointmass_host.vz, pointmass_device.vz, memorySizeForPointmasses, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(pointmass_host.az, pointmass_device.az, memorySizeForPointmasses, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(pointmass_host.m, pointmass_device.m, memorySizeForPointmasses, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(pointmass_host.rmin, pointmass_device.rmin, memorySizeForPointmasses, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(pointmass_host.rmax, pointmass_device.rmax, memorySizeForPointmasses, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.x, p_device.x, memorySizeForTree, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.y, p_device.y, memorySizeForTree, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.vx, p_device.vx, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.vx0, p_device.vx0, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.vy, p_device.vy, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.vy0, p_device.vy0, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.ax, p_device.ax, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.ay, p_device.ay, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.g_ax, p_device.g_ax, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.g_ay, p_device.g_ay, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.z, p_device.z, memorySizeForTree, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.vz, p_device.vz, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.vz0, p_device.vz0, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.az, p_device.az, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.g_az, p_device.g_az, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.m, p_device.m, memorySizeForTree, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.depth, p_device.depth, memorySizeForInteractions, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.rho, p_device.rho, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.drhodt, p_device.drhodt, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.h, p_device.h, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.materialId, p_device.materialId, memorySizeForInteractions, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.p, p_device.p, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.cs, p_device.cs, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.noi, p_device.noi, memorySizeForInteractions, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(interactions_host, interactions, memorySizeForInteractions*MAX_NUM_INTERACTIONS, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(childList_host, (void * )childListd, memorySizeForChildren, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.p_min, p_device.p_min, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.p_max, p_device.p_max, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.rho_min, p_device.rho_min, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.rho_max, p_device.rho_max, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.e_min, p_device.e_min, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.e_max, p_device.e_max, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.cs_min, p_device.cs_min, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.cs_max, p_device.cs_max, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.pold, p_device.pold, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.alpha_jutzi, p_device.alpha_jutzi, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.dalphadt, p_device.dalphadt, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.alpha_jutzi_old, p_device.alpha_jutzi_old, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.compressive_strength, p_device.compressive_strength, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.tensile_strength, p_device.tensile_strength, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.K, p_device.K, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.rho_0prime, p_device.rho_0prime, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.rho_c_plus, p_device.rho_c_plus, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.rho_c_minus, p_device.rho_c_minus, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.shear_strength, p_device.shear_strength, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.flag_rho_0prime, p_device.flag_rho_0prime, memorySizeForInteractions, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.flag_plastic, p_device.flag_plastic, memorySizeForInteractions, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.alpha_epspor, p_device.alpha_epspor, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.epsilon_v, p_device.epsilon_v, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.e, p_device.e, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.dedt, p_device.dedt, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.T, p_device.T, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.Tshear, p_device.Tshear, memorySizeForStress, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.S, p_device.S, memorySizeForStress, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.dSdt, p_device.dSdt, memorySizeForStress, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.local_strain, p_device.local_strain, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.ep, p_device.ep, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.d, p_device.d, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.dddt, p_device.dddt, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.numActiveFlaws, p_device.numActiveFlaws, memorySizeForInteractions, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.flaws, p_device.flaws, memorySizeForActivationThreshold, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.damage_porjutzi, p_device.damage_porjutzi, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaMemcpy(p_host.ddamage_porjutzidt, p_device.ddamage_porjutzidt, memorySizeForParticles, cudaMemcpyDeviceToHost));
io.cu:    cudaVerify(cudaDeviceSynchronize());
device_tools.h: * This file is part of miluphcuda.
device_tools.h: * miluphcuda is free software: you can redistribute it and/or modify
device_tools.h: * miluphcuda is distributed in the hope that it will be useful,
device_tools.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
xsph.h: * This file is part of miluphcuda.
xsph.h: * miluphcuda is free software: you can redistribute it and/or modify
xsph.h: * miluphcuda is distributed in the hope that it will be useful,
xsph.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
little_helpers.cu: * This file is part of miluphcuda.
little_helpers.cu: * miluphcuda is free software: you can redistribute it and/or modify
little_helpers.cu: * miluphcuda is distributed in the hope that it will be useful,
little_helpers.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
sinking.cu: * This file is part of miluphcuda.
sinking.cu: * miluphcuda is free software: you can redistribute it and/or modify
sinking.cu: * miluphcuda is distributed in the hope that it will be useful,
sinking.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
test_cases/gravity_merging/results/plot_plastic_yielding.Granite.py:Plots particles' shear stresses from miluphcuda HDF5 output files + the theoretical
test_cases/gravity_merging/results/plot_plastic_yielding.Granite.py:parser = argparse.ArgumentParser(description="Plots particles' shear stresses from miluphcuda HDF5 output files + the theoretical yield stress limit (parameters hardcoded in the script!).")
test_cases/gravity_merging/run.sh:# If necessary, adapt the paths to the CUDA libs and the miluphcuda executable below, before running it.
test_cases/gravity_merging/run.sh:# set path to CUDA libs [change if necessary]
test_cases/gravity_merging/run.sh:export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
test_cases/gravity_merging/run.sh:# set path to miluphcuda executable [change if necessary]
test_cases/gravity_merging/run.sh:MC=../../miluphcuda
test_cases/gravity_merging/run.sh:# miluphcuda command line
test_cases/gravity_merging/run.sh:$MC -v -A -f impact.0000 -g -H -I rk2_adaptive -Q 1e-4 -m material.cfg -n 75 -t 100.0 -s 1>miluphcuda.output 2>miluphcuda.error &
test_cases/gravity_merging/USAGE.md:Gravity merging test case for miluphcuda
test_cases/gravity_merging/USAGE.md:1. Compile miluphcuda with the `parameter.h` file from this directory.  
test_cases/gravity_merging/USAGE.md:3. Adapt the start script `run.sh` to your system (path to CUDA libs and to miluphcuda executable) and execute it.
test_cases/gravity_merging/USAGE.md:The setup consists of ~50k SPH particles, with a runtime around 1h on most current GPUs (benchmarked on a GTX 970).
test_cases/gravity_merging/parameter.h: * This file is part of miluphcuda.
test_cases/gravity_merging/parameter.h: * miluphcuda is free software: you can redistribute it and/or modify
test_cases/gravity_merging/parameter.h: * miluphcuda is distributed in the hope that it will be useful,
test_cases/gravity_merging/parameter.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
test_cases/shocktube/run.sh:export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64
test_cases/shocktube/run.sh:nice -19 ./miluphcuda  -d 3 -Q 1e-8 -v -n 100 -H -t 0.00228  -f shocktube.0000 -m material.cfg  > output.txt
test_cases/shocktube/parameter.h: * This file is part of miluphcuda.
test_cases/shocktube/parameter.h: * miluphcuda is free software: you can redistribute it and/or modify
test_cases/shocktube/parameter.h: * miluphcuda is distributed in the hope that it will be useful,
test_cases/shocktube/parameter.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
test_cases/colliding_rings/run.sh:export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64
test_cases/colliding_rings/run.sh:nice -19 ./miluphcuda -v -n 500 -H -t 1 -f rings.0000 -m material.cfg > output.txt 
test_cases/colliding_rings/parameter.h: * This file is part of miluphcuda.
test_cases/colliding_rings/parameter.h: * miluphcuda is free software: you can redistribute it and/or modify
test_cases/colliding_rings/parameter.h: * miluphcuda is distributed in the hope that it will be useful,
test_cases/colliding_rings/parameter.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
test_cases/colliding_rings/ReadMe:  ii compile miluphcuda with the parameter.h file from the test_cases/colliding_rings directory
test_cases/colliding_rings/ReadMe:The runtime of the simulation is 2 min on a Nvidia Geforce GTX 1080 Ti.
test_cases/tensile_rod/run.sh:# set path to CUDA libs [change if necessary]
test_cases/tensile_rod/run.sh:export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
test_cases/tensile_rod/run.sh:# set path to miluphcuda executable [change if necessary]
test_cases/tensile_rod/run.sh:MC=../../miluphcuda
test_cases/tensile_rod/run.sh:# miluphcuda cmd line
test_cases/tensile_rod/run.sh:$MC -f rod.0000 -m material.cfg -I rk2_adaptive -Q 1e-5 -v -n 250 -t 3e-2 -H -A  1>miluphcuda.output 2>miluphcuda.error &
test_cases/tensile_rod/run.sh:#./miluphcuda -d 0 -v -I euler_pc -n 100 -H  -t 1e-6 -f rod.0000 -m material.cfg  > output.txt 
test_cases/tensile_rod/run.sh:#./miluphcuda -Q 1e-4 -d 0 -v -n 1000 -H  -t 1e-2 -f rod.0368 -X  -m material.cfg  > output.txt 
test_cases/tensile_rod/ReadMe:(ii)  compile miluphcuda with src/boundary.cu and src/parameter.h
test_cases/tensile_rod/src/parameter.h: * This file is part of miluphcuda.
test_cases/tensile_rod/src/parameter.h: * miluphcuda is free software: you can redistribute it and/or modify
test_cases/tensile_rod/src/parameter.h: * miluphcuda is distributed in the hope that it will be useful,
test_cases/tensile_rod/src/parameter.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
test_cases/tensile_rod/src/boundary.cu: * This file is part of miluphcuda.
test_cases/tensile_rod/src/boundary.cu: * miluphcuda is free software: you can redistribute it and/or modify
test_cases/tensile_rod/src/boundary.cu: * miluphcuda is distributed in the hope that it will be useful,
test_cases/tensile_rod/src/boundary.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
test_cases/sedov/run.sh:nice -19 ./miluphcuda -f sedov.0000 -n 100 -t 1e-4 -H -m material.cfg --verbose > output.txt 2> error.txt
test_cases/sedov/parameter.h: * This file is part of miluphcuda.
test_cases/sedov/parameter.h: * miluphcuda is free software: you can redistribute it and/or modify
test_cases/sedov/parameter.h: * miluphcuda is distributed in the hope that it will be useful,
test_cases/sedov/parameter.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
test_cases/rotating_sphere/run.sh:export LD_LIBRARY_PATH=/usr/local/cuda
test_cases/rotating_sphere/run.sh:nice -19 ./miluphcuda -v -n 100 -H -t 6.283185307179586 -f sphere.0000 -m material.cfg > output.txt 
test_cases/rotating_sphere/ReadMe.md:1. copy parameter.h to the root source directory of miluphcuda (usually cp parameter.h ../../)
test_cases/rotating_sphere/parameter.h: * This file is part of miluphcuda.
test_cases/rotating_sphere/parameter.h: * miluphcuda is free software: you can redistribute it and/or modify
test_cases/rotating_sphere/parameter.h: * miluphcuda is distributed in the hope that it will be useful,
test_cases/rotating_sphere/parameter.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
test_cases/viscously_spreading_ring/run.sh:export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64
test_cases/viscously_spreading_ring/run.sh:nice -19 ./miluphcuda -v -n 600 -H -t 1000 -f viscous_ring.0000 -m material_viscously_spreading_ring.cfg  > output.txt 
test_cases/viscously_spreading_ring/parameter.h: * This file is part of miluphcuda.
test_cases/viscously_spreading_ring/parameter.h: * miluphcuda is free software: you can redistribute it and/or modify
test_cases/viscously_spreading_ring/parameter.h: * miluphcuda is distributed in the hope that it will be useful,
test_cases/viscously_spreading_ring/parameter.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
test_cases/viscously_spreading_ring/boundary.cu: * This file is part of miluphcuda.
test_cases/viscously_spreading_ring/boundary.cu: * miluphcuda is free software: you can redistribute it and/or modify
test_cases/viscously_spreading_ring/boundary.cu: * miluphcuda is distributed in the hope that it will be useful,
test_cases/viscously_spreading_ring/boundary.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
test_cases/dambreak/run.sh:nice -19 ./miluphcuda  -b 1.5 -k wendlandc2 -f dam.0000 -t 0.1 -n 100 -v -H > output.txt 2> error.txt
test_cases/dambreak/parameter.h: * This file is part of miluphcuda.
test_cases/dambreak/parameter.h: * miluphcuda is free software: you can redistribute it and/or modify
test_cases/dambreak/parameter.h: * miluphcuda is distributed in the hope that it will be useful,
test_cases/dambreak/parameter.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
test_cases/dambreak/boundary.cu: * This file is part of miluphcuda.
test_cases/dambreak/boundary.cu: * miluphcuda is free software: you can redistribute it and/or modify
test_cases/dambreak/boundary.cu: * miluphcuda is distributed in the hope that it will be useful,
test_cases/dambreak/boundary.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
test_cases/nakamura/run.sh:# set path to CUDA libs [change if necessary]
test_cases/nakamura/run.sh:export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
test_cases/nakamura/run.sh:# set path to miluphcuda executable [change if necessary]
test_cases/nakamura/run.sh:MC=../../miluphcuda
test_cases/nakamura/run.sh:# miluphcuda cmd line
test_cases/nakamura/run.sh:$MC -f impact.0000 -m material.cfg -n 250 -t 2e-7 -I rk2_adaptive -Q 1e-5 -v -H -A 1>miluphcuda.output 2>miluphcuda.error & disown -h
test_cases/nakamura/parameter.h: * This file is part of miluphcuda.
test_cases/nakamura/parameter.h: * miluphcuda is free software: you can redistribute it and/or modify
test_cases/nakamura/parameter.h: * miluphcuda is distributed in the hope that it will be useful,
test_cases/nakamura/parameter.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
test_cases/nakamura/ReadMe:(ii)  compile miluphcuda with the parameter.h file from the test_cases/nakamura directory
plasticity.cu: * This file is part of miluphcuda.
plasticity.cu: * miluphcuda is free software: you can redistribute it and/or modify
plasticity.cu: * miluphcuda is distributed in the hope that it will be useful,
plasticity.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
timeintegration.cu: * This file is part of miluphcuda.
timeintegration.cu: * miluphcuda is free software: you can redistribute it and/or modify
timeintegration.cu: * miluphcuda is distributed in the hope that it will be useful,
timeintegration.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
timeintegration.cu:    cudaVerify(cudaMemcpyToSymbol(dt, &dt_host, sizeof(double)));
timeintegration.cu:    cudaVerify(cudaMemcpyToSymbol(dtmax, &param.maxtimestep, sizeof(double)));
timeintegration.cu:    cudaVerify(cudaMemcpyToSymbol(theta, &treeTheta, sizeof(double)));
timeintegration.cu:    cudaVerify(cudaMemcpyToSymbol(numParticles, &numberOfParticles, sizeof(int)));
timeintegration.cu:    cudaVerify(cudaMemcpyToSymbol(numPointmasses, &numberOfPointmasses, sizeof(int)));
timeintegration.cu:    cudaVerify(cudaMemcpyToSymbol(maxNumParticles, &maxNumberOfParticles, sizeof(int)));
timeintegration.cu:    cudaVerify(cudaMemcpyToSymbol(numRealParticles, &numberOfRealParticles, sizeof(int)));
timeintegration.cu:    cudaVerify(cudaMemcpyToSymbol(numChildren, &numberOfChildren, sizeof(int)));
timeintegration.cu:    cudaVerify(cudaMemcpyToSymbol(numNodes, &numberOfNodes, sizeof(int)));
timeintegration.cu:    cudaVerify(cudaMemcpyToSymbol(maxNumFlaws, &maxNumFlaws_host, sizeof(int)));
timeintegration.cu:    cudaVerify(cudaMalloc((void**)&minxPerBlock, sizeof(double)*numberOfMultiprocessors));
timeintegration.cu:    cudaVerify(cudaMalloc((void**)&maxxPerBlock, sizeof(double)*numberOfMultiprocessors));
timeintegration.cu:    cudaVerify(cudaMalloc((void**)&minyPerBlock, sizeof(double)*numberOfMultiprocessors));
timeintegration.cu:    cudaVerify(cudaMalloc((void**)&maxyPerBlock, sizeof(double)*numberOfMultiprocessors));
timeintegration.cu:    cudaVerify(cudaMalloc((void**)&minzPerBlock, sizeof(double)*numberOfMultiprocessors));
timeintegration.cu:    cudaVerify(cudaMalloc((void**)&maxzPerBlock, sizeof(double)*numberOfMultiprocessors));
timeintegration.cu:    // set the pointer on the gpu to p_device
timeintegration.cu:    cudaVerify(cudaMemcpyToSymbol(p, &p_device, sizeof(struct Particle)));
timeintegration.cu:    cudaVerify(cudaMemcpyToSymbol(p_rhs, &p_device, sizeof(struct Particle)));
timeintegration.cu:    cudaVerify(cudaMemcpyToSymbol(pointmass, &pointmass_device, sizeof(struct Pointmass)));
timeintegration.cu:    cudaVerify(cudaMemcpyToSymbol(pointmass_rhs, &pointmass_device, sizeof(struct Pointmass)));
timeintegration.cu:    cudaVerifyKernel((initializeSoundspeed<<<numberOfMultiprocessors*4, NUM_THREADS_512>>>()));
timeintegration.cu:    cudaVerifyKernel((ParticleSinking<<<numberOfMultiprocessors*4, NUM_THREADS_PRESSURE>>>()));
timeintegration.cu:	cudaVerifyKernel((get_extrema<<<numberOfMultiprocessors*4, NUM_THREADS_PRESSURE>>>()));
timeintegration.cu:    cudaVerify(cudaFree(minxPerBlock));
timeintegration.cu:    cudaVerify(cudaFree(maxxPerBlock));
timeintegration.cu:    cudaVerify(cudaFree(minyPerBlock));
timeintegration.cu:    cudaVerify(cudaFree(maxyPerBlock));
timeintegration.cu:    cudaVerify(cudaFree(minzPerBlock));
timeintegration.cu:    cudaVerify(cudaFree(maxzPerBlock));
internal_forces.h: * This file is part of miluphcuda.
internal_forces.h: * miluphcuda is free software: you can redistribute it and/or modify
internal_forces.h: * miluphcuda is distributed in the hope that it will be useful,
internal_forces.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
config_parameter.h: * This file is part of miluphcuda.
config_parameter.h: * miluphcuda is free software: you can redistribute it and/or modify
config_parameter.h: * miluphcuda is distributed in the hope that it will be useful,
config_parameter.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
config_parameter.h:#include "cuda_utils.h"
config_parameter.h:void transferMaterialsToGPU();
config_parameter.cu: * This file is part of miluphcuda.
config_parameter.cu: * miluphcuda is free software: you can redistribute it and/or modify
config_parameter.cu: * miluphcuda is distributed in the hope that it will be useful,
config_parameter.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
config_parameter.cu:void transferMaterialsToGPU()
config_parameter.cu:        if materials no. 0 and 2 use ANEOS), necessary for resolving linearizations of multi-dim arrays on GPU */
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matporjutzi_p_elastic_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matporjutzi_p_transition_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matporjutzi_p_compacted_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matporjutzi_alpha_0_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matporjutzi_alpha_e_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matporjutzi_alpha_t_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matporjutzi_n1_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matporjutzi_n2_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matcs_porous_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matcs_solid_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matcrushcurve_style_d, numberOfElements*sizeof(int)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&mat_f_sml_max_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&mat_f_sml_min_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matporsirono_K_0_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matporsirono_rho_0_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matporsirono_rho_s_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matporsirono_gamma_K_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matporsirono_alpha_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matporsirono_pm_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matporsirono_phimax_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matporsirono_phi0_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matporsirono_delta_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matporepsilon_kappa_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matporepsilon_alpha_0_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matporepsilon_epsilon_e_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matporepsilon_epsilon_x_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matporepsilon_epsilon_c_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matnu_d, numberOfElements*sizeof(double)));
config_parameter.cu:	    cudaVerify(cudaMalloc((void **)&matalpha_shakura_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matzeta_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matexponent_tensor_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matepsilon_stress_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matmean_particle_distance_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matLdwEtaLimit_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matLdwAlpha_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matLdwBeta_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matLdwGamma_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&aneos_n_rho_d, numberOfElements*sizeof(int)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&aneos_n_e_d, numberOfElements*sizeof(int)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&aneos_bulk_cs_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&aneos_rho_d, run_aneos_rho_id*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&aneos_e_d, run_aneos_e_id*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&aneos_p_d, run_aneos_matrix_id*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&aneos_cs_d, run_aneos_matrix_id*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&aneos_rho_id_d, numberOfElements*sizeof(int)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&aneos_e_id_d, numberOfElements*sizeof(int)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&aneos_matrix_id_d, numberOfElements*sizeof(int)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matSml_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matnoi_d, numberOfElements*sizeof(int)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matEOS_d, numberOfElements*sizeof(int)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matPolytropicK_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matPolytropicGamma_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matAlpha_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matBeta_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matBulkmodulus_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matShearmodulus_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matYieldStress_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matInternalFriction_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matInternalFrictionDamaged_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matRho0_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matIsothermalSoundSpeed_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matTillRho0_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matTillE0_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matTillEiv_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matTillEcv_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matTilla_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matTillb_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matTillA_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matTillB_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matTillAlpha_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matTillBeta_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matcsLimit_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matRhoLimit_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matN_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matCohesion_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matCohesionDamaged_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matFrictionAngle_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matFrictionAngleDamaged_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matAlphaPhi_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matCohesionCoefficient_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matMeltEnergy_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matDensityFloor_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matEnergyFloor_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matjc_y0_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matjc_B_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matjc_n_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matjc_m_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matjc_edot0_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matjc_C_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matjc_Tref_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matjc_Tmelt_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matCp_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matCV_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matdensity_via_kernel_sum_d, numberOfElements*sizeof(int)));
config_parameter.cu:        cudaVerify(cudaMemcpy(matdensity_via_kernel_sum_d, density_via_kernel_sum, numberOfElements*sizeof(int), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMalloc((void **)&matYoungModulus_d, numberOfElements*sizeof(double)));
config_parameter.cu:        cudaVerify(cudaMemcpy(matYoungModulus_d, young_modulus, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matYoungModulus, &matYoungModulus_d, sizeof(void*)));
config_parameter.cu:        cudaGetSymbolAddress((void **)&pc_pointer, gravConst);
config_parameter.cu:        cudaMemcpy(pc_pointer, &grav_const, sizeof(double), cudaMemcpyHostToDevice);
config_parameter.cu:        cudaGetSymbolAddress((void **)&pc_pointer, scale_height);
config_parameter.cu:        cudaMemcpy(pc_pointer, &scale_height_host, sizeof(double), cudaMemcpyHostToDevice);
config_parameter.cu:        cudaGetSymbolAddress((void **)&pc_pointer, Smin_d);
config_parameter.cu:        cudaMemcpy(pc_pointer, &Smin, sizeof(double), cudaMemcpyHostToDevice);
config_parameter.cu:        cudaGetSymbolAddress((void **)&pc_pointer, emin_d);
config_parameter.cu:        cudaMemcpy(pc_pointer, &emin, sizeof(double), cudaMemcpyHostToDevice);
config_parameter.cu:        cudaGetSymbolAddress((void **)&pc_pointer, rhomin_d);
config_parameter.cu:        cudaMemcpy(pc_pointer, &rhomin, sizeof(double), cudaMemcpyHostToDevice);
config_parameter.cu:        cudaGetSymbolAddress((void **)&pc_pointer, damagemin_d);
config_parameter.cu:        cudaMemcpy(pc_pointer, &damagemin, sizeof(double), cudaMemcpyHostToDevice);
config_parameter.cu:        cudaGetSymbolAddress((void **)&pc_pointer, alphamin_d);
config_parameter.cu:        cudaMemcpy(pc_pointer, &alphamin, sizeof(double), cudaMemcpyHostToDevice);
config_parameter.cu:        cudaGetSymbolAddress((void **)&pc_pointer, betamin_d);
config_parameter.cu:        cudaMemcpy(pc_pointer, &betamin, sizeof(double), cudaMemcpyHostToDevice);
config_parameter.cu:        cudaGetSymbolAddress((void **)&pc_pointer, alpha_epspormin_d);
config_parameter.cu:        cudaMemcpy(pc_pointer, &alpha_epspormin, sizeof(double), cudaMemcpyHostToDevice);
config_parameter.cu:        cudaGetSymbolAddress((void **)&pc_pointer, epsilon_vmin_d);
config_parameter.cu:        cudaMemcpy(pc_pointer, &epsilon_vmin, sizeof(double), cudaMemcpyHostToDevice);
config_parameter.cu:        cudaGetSymbolAddress((void **)&pc_pointer, max_abs_pressure_change);
config_parameter.cu:        cudaMemcpy(pc_pointer, &max_abs_pressure_change_host, sizeof(double), cudaMemcpyHostToDevice);
config_parameter.cu:        cudaVerify(cudaMemcpy(matSml_d, sml, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(mat_f_sml_max_d, f_sml_max , numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(mat_f_sml_min_d, f_sml_min , numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matnoi_d, noi, numberOfElements*sizeof(int), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matEOS_d, eos, numberOfElements*sizeof(int), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matPolytropicK_d, polytropic_K, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matPolytropicGamma_d, polytropic_gamma, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:  	    cudaVerify(cudaMemcpy(matAlpha_d, alpha, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matBeta_d, beta, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matBulkmodulus_d, bulk_modulus, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matnu_d, nu, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:  	    cudaVerify(cudaMemcpy(matalpha_shakura_d, alpha_shakura, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matzeta_d, zeta, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matShearmodulus_d, shear_modulus, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matYieldStress_d, yield_stress, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matInternalFriction_d, internal_friction, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matInternalFrictionDamaged_d, internal_friction_damaged, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matRho0_d, rho_0, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matIsothermalSoundSpeed_d, isothermal_cs, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matTillRho0_d, till_rho_0, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matTillE0_d, till_E_0, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matTillEcv_d, till_E_cv, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matTillEiv_d, till_E_iv, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matTilla_d, till_a, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matTillb_d, till_b, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matTillA_d, till_A, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matTillB_d, till_B, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matTillAlpha_d, till_alpha, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matTillBeta_d, till_beta, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matcsLimit_d, csLimit, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matRhoLimit_d, rho_limit, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matN_d, n, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matCohesion_d, cohesion, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matCohesionDamaged_d, cohesion_damaged, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matFrictionAngle_d, friction_angle, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matFrictionAngleDamaged_d, friction_angle_damaged, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matAlphaPhi_d, alpha_phi, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matCohesionCoefficient_d, cohesion_coefficient, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matMeltEnergy_d, melt_energy, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matDensityFloor_d, density_floor, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matEnergyFloor_d, energy_floor, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(aneos_n_rho_d, g_aneos_n_rho, numberOfElements*sizeof(int), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(aneos_n_e_d, g_aneos_n_e, numberOfElements*sizeof(int), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(aneos_bulk_cs_d, g_aneos_bulk_cs, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:                cudaVerify(cudaMemcpy(aneos_rho_d+aneos_rho_id[i], g_aneos_rho[i], g_aneos_n_rho[i]*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:                cudaVerify(cudaMemcpy(aneos_e_d+aneos_e_id[i], g_aneos_e[i], g_aneos_n_e[i]*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:                    cudaVerify(cudaMemcpy(aneos_p_d+aneos_matrix_id[i]+j*g_aneos_n_e[i], g_aneos_p[i][j], g_aneos_n_e[i]*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:                    cudaVerify(cudaMemcpy(aneos_cs_d+aneos_matrix_id[i]+j*g_aneos_n_e[i], g_aneos_cs[i][j], g_aneos_n_e[i]*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(aneos_rho_id_d, aneos_rho_id, numberOfElements*sizeof(int), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(aneos_e_id_d, aneos_e_id, numberOfElements*sizeof(int), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(aneos_matrix_id_d, aneos_matrix_id, numberOfElements*sizeof(int), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(aneos_n_rho_c, &aneos_n_rho_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(aneos_n_e_c, &aneos_n_e_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(aneos_bulk_cs_c, &aneos_bulk_cs_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(aneos_rho_c, &aneos_rho_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(aneos_e_c, &aneos_e_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(aneos_p_c, &aneos_p_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(aneos_cs_c, &aneos_cs_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(aneos_rho_id_c, &aneos_rho_id_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(aneos_e_id_c, &aneos_e_id_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(aneos_matrix_id_c, &aneos_matrix_id_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpy(matjc_y0_d, jc_y0, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matjc_B_d, jc_B, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matjc_n_d, jc_n, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matjc_m_d, jc_m, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matjc_edot0_d, jc_edot0, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matjc_C_d, jc_C, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matjc_Tref_d, jc_Tref, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matjc_Tmelt_d, jc_Tmelt, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matCp_d, Cp, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matCV_d, CV, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matnu, &matnu_d, sizeof(void*)));
config_parameter.cu:  	    cudaVerify(cudaMemcpyToSymbol(matalpha_shakura, &matalpha_shakura_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matzeta, &matzeta_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpy(matexponent_tensor_d, exponent_tensor, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matepsilon_stress_d, epsilon_stress, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matmean_particle_distance_d, mean_particle_distance, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matexponent_tensor, &matexponent_tensor_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matepsilon_stress, &matepsilon_stress_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matmean_particle_distance, &matmean_particle_distance_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpy(matLdwEtaLimit_d, ldw_eta_limit, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matLdwAlpha_d, ldw_alpha, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matLdwBeta_d, ldw_beta, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matLdwGamma_d, ldw_gamma, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matLdwEtaLimit, &matLdwEtaLimit_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matLdwAlpha, &matLdwAlpha_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matLdwBeta, &matLdwBeta_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matLdwGamma, &matLdwGamma_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpy(matporjutzi_p_elastic_d, porjutzi_p_elastic, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matporjutzi_p_transition_d, porjutzi_p_transition, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matporjutzi_p_compacted_d, porjutzi_p_compacted, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matporjutzi_alpha_0_d, porjutzi_alpha_0, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matporjutzi_alpha_e_d, porjutzi_alpha_e, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matporjutzi_alpha_t_d, porjutzi_alpha_t, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matporjutzi_n1_d, porjutzi_n1, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matporjutzi_n2_d, porjutzi_n2, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matcs_porous_d, cs_porous, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matcs_solid_d, cs_solid, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matcrushcurve_style_d, crushcurve_style, numberOfElements*sizeof(int), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matporjutzi_p_elastic, &matporjutzi_p_elastic_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matporjutzi_p_transition, &matporjutzi_p_transition_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matporjutzi_p_compacted, &matporjutzi_p_compacted_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matporjutzi_alpha_0, &matporjutzi_alpha_0_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matporjutzi_alpha_e, &matporjutzi_alpha_e_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matporjutzi_alpha_t, &matporjutzi_alpha_t_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matporjutzi_n1, &matporjutzi_n1_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matporjutzi_n2, &matporjutzi_n2_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matcs_porous, &matcs_porous_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matcs_solid, &matcs_solid_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matcrushcurve_style, &matcrushcurve_style_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpy(matporsirono_K_0_d, porsirono_K_0, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matporsirono_rho_0_d, porsirono_rho_0, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matporsirono_rho_s_d, porsirono_rho_s, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matporsirono_gamma_K_d, porsirono_gamma_K, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matporsirono_alpha_d, porsirono_alpha, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matporsirono_pm_d, porsirono_pm, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matporsirono_phimax_d, porsirono_phimax, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matporsirono_phi0_d, porsirono_phi0, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matporsirono_delta_d, porsirono_delta, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matporsirono_K_0, &matporsirono_K_0_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matporsirono_rho_0, &matporsirono_rho_0_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matporsirono_rho_s, &matporsirono_rho_s_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matporsirono_gamma_K, &matporsirono_gamma_K_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matporsirono_alpha, &matporsirono_alpha_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matporsirono_pm, &matporsirono_pm_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matporsirono_phimax, &matporsirono_phimax_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matporsirono_phi0, &matporsirono_phi0_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matporsirono_delta, &matporsirono_delta_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpy(matporepsilon_kappa_d, porepsilon_kappa, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matporepsilon_alpha_0_d, porepsilon_alpha_0, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matporepsilon_epsilon_e_d, porepsilon_epsilon_e, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matporepsilon_epsilon_x_d, porepsilon_epsilon_x, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpy(matporepsilon_epsilon_c_d, porepsilon_epsilon_c, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matporepsilon_kappa, &matporepsilon_kappa_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matporepsilon_alpha_0, &matporepsilon_alpha_0_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matporepsilon_epsilon_e, &matporepsilon_epsilon_e_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matporepsilon_epsilon_x, &matporepsilon_epsilon_x_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matporepsilon_epsilon_c, &matporepsilon_epsilon_c_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matSml, &matSml_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(mat_f_sml_max, &mat_f_sml_max_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(mat_f_sml_min, &mat_f_sml_min_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matnoi, &matnoi_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matdensity_via_kernel_sum, &matdensity_via_kernel_sum_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matEOS, &matEOS_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matPolytropicK, &matPolytropicK_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matPolytropicGamma, &matPolytropicGamma_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matAlpha, &matAlpha_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matBeta, &matBeta_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matBulkmodulus, &matBulkmodulus_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matShearmodulus, &matShearmodulus_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matYieldStress, &matYieldStress_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matInternalFriction, &matInternalFriction_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matInternalFrictionDamaged, &matInternalFrictionDamaged_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matRho0, &matRho0_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matIsothermalSoundSpeed, &matIsothermalSoundSpeed_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matTillRho0, &matTillRho0_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matTillE0, &matTillE0_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matTillEiv, &matTillEiv_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matTillEcv, &matTillEcv_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matTilla, &matTilla_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matTillb, &matTillb_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matTillA, &matTillA_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matTillB, &matTillB_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matTillAlpha, &matTillAlpha_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matTillBeta, &matTillBeta_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matcsLimit, &matcsLimit_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matRhoLimit, &matRhoLimit_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matN, &matN_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matCohesion, &matCohesion_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matCohesionDamaged, &matCohesionDamaged_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matFrictionAngle, &matFrictionAngle_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matFrictionAngleDamaged, &matFrictionAngleDamaged_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matAlphaPhi, &matAlphaPhi_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matCohesionCoefficient, &matCohesionCoefficient_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matMeltEnergy, &matMeltEnergy_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matDensityFloor, &matDensityFloor_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matEnergyFloor, &matEnergyFloor_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matjc_y0, &matjc_y0_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matjc_B, &matjc_B_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matjc_n, &matjc_n_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matjc_m, &matjc_m_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matjc_edot0, &matjc_edot0_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matjc_C, &matjc_C_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matjc_Tref, &matjc_Tref_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matjc_Tmelt, &matjc_Tmelt_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matCp, &matCp_d, sizeof(void*)));
config_parameter.cu:        cudaVerify(cudaMemcpyToSymbol(matCV, &matCV_d, sizeof(void*)));
config_parameter.cu:    cudaVerify(cudaFree(aneos_n_rho_d));
config_parameter.cu:    cudaVerify(cudaFree(aneos_n_e_d));
config_parameter.cu:    cudaVerify(cudaFree(aneos_bulk_cs_d));
config_parameter.cu:    cudaVerify(cudaFree(aneos_rho_d));
config_parameter.cu:    cudaVerify(cudaFree(aneos_e_d));
config_parameter.cu:    cudaVerify(cudaFree(aneos_p_d));
config_parameter.cu:    cudaVerify(cudaFree(aneos_cs_d));
config_parameter.cu:    cudaVerify(cudaFree(aneos_rho_id_d));
config_parameter.cu:    cudaVerify(cudaFree(aneos_e_id_d));
config_parameter.cu:    cudaVerify(cudaFree(aneos_matrix_id_d));
config_parameter.cu:    cudaVerify(cudaFree(matSml_d));
config_parameter.cu:    cudaVerify(cudaFree(matInternalFriction_d));
config_parameter.cu:    cudaVerify(cudaFree(matInternalFrictionDamaged_d));
config_parameter.cu:    cudaVerify(cudaFree(matYieldStress_d));
config_parameter.cu:    cudaVerify(cudaFree(matnoi_d));
config_parameter.cu:    cudaVerify(cudaFree(matEOS_d));
config_parameter.cu:    cudaVerify(cudaFree(matPolytropicGamma_d));
config_parameter.cu:    cudaVerify(cudaFree(matPolytropicK_d));
config_parameter.cu:    cudaVerify(cudaFree(matAlpha_d));
config_parameter.cu:    cudaVerify(cudaFree(matAlphaPhi_d));
config_parameter.cu:    cudaVerify(cudaFree(matBeta_d));
config_parameter.cu:    cudaVerify(cudaFree(matBulkmodulus_d));
config_parameter.cu:    cudaVerify(cudaFree(matYoungModulus_d));
config_parameter.cu:    cudaVerify(cudaFree(mat_f_sml_max_d));
config_parameter.cu:    cudaVerify(cudaFree(mat_f_sml_min_d));
config_parameter.cu:    cudaVerify(cudaFree(matRho0_d));
config_parameter.cu:    cudaVerify(cudaFree(matTillRho0_d));
config_parameter.cu:    cudaVerify(cudaFree(matTilla_d));
config_parameter.cu:    cudaVerify(cudaFree(matTillA_d));
config_parameter.cu:    cudaVerify(cudaFree(matTillb_d));
config_parameter.cu:    cudaVerify(cudaFree(matTillB_d));
config_parameter.cu:    cudaVerify(cudaFree(matTillAlpha_d));
config_parameter.cu:    cudaVerify(cudaFree(matTillBeta_d));
config_parameter.cu:    cudaVerify(cudaFree(matcsLimit_d));
config_parameter.cu:    cudaVerify(cudaFree(matTillE0_d));
config_parameter.cu:    cudaVerify(cudaFree(matTillEcv_d));
config_parameter.cu:    cudaVerify(cudaFree(matTillEiv_d));
config_parameter.cu:    cudaVerify(cudaFree(matRhoLimit_d));
config_parameter.cu:    cudaVerify(cudaFree(matShearmodulus_d));
config_parameter.cu:    cudaVerify(cudaFree(matN_d));
config_parameter.cu:    cudaVerify(cudaFree(matCohesion_d));
config_parameter.cu:    cudaVerify(cudaFree(matCohesionDamaged_d));
config_parameter.cu:    cudaVerify(cudaFree(matCohesionCoefficient_d));
config_parameter.cu:    cudaVerify(cudaFree(matMeltEnergy_d));
config_parameter.cu:    cudaVerify(cudaFree(matFrictionAngle_d));
config_parameter.cu:    cudaVerify(cudaFree(matFrictionAngleDamaged_d));
config_parameter.cu:    cudaVerify(cudaFree(matDensityFloor_d));
config_parameter.cu:    cudaVerify(cudaFree(matEnergyFloor_d));
config_parameter.cu:    cudaVerify(cudaFree(matLdwEtaLimit_d));
config_parameter.cu:    cudaVerify(cudaFree(matLdwAlpha_d));
config_parameter.cu:    cudaVerify(cudaFree(matLdwBeta_d));
config_parameter.cu:    cudaVerify(cudaFree(matLdwGamma_d));
config_parameter.cu:    cudaVerify(cudaFree(matporjutzi_p_elastic_d));
config_parameter.cu:    cudaVerify(cudaFree(matporjutzi_p_transition_d));
config_parameter.cu:    cudaVerify(cudaFree(matporjutzi_p_compacted_d));
config_parameter.cu:    cudaVerify(cudaFree(matporjutzi_alpha_0_d));
config_parameter.cu:    cudaVerify(cudaFree(matporjutzi_alpha_e_d));
config_parameter.cu:    cudaVerify(cudaFree(matporjutzi_alpha_t_d));
config_parameter.cu:    cudaVerify(cudaFree(matporjutzi_n1_d));
config_parameter.cu:    cudaVerify(cudaFree(matporjutzi_n2_d));
config_parameter.cu:    cudaVerify(cudaFree(matcs_porous_d));
config_parameter.cu:    cudaVerify(cudaFree(matcs_solid_d));
config_parameter.cu:    cudaVerify(cudaFree(matcrushcurve_style_d));
config_parameter.cu:    cudaVerify(cudaFree(matporsirono_K_0_d));
config_parameter.cu:    cudaVerify(cudaFree(matporsirono_rho_0_d));
config_parameter.cu:    cudaVerify(cudaFree(matporsirono_rho_s_d));
config_parameter.cu:    cudaVerify(cudaFree(matporsirono_gamma_K_d));
config_parameter.cu:    cudaVerify(cudaFree(matporsirono_alpha_d));
config_parameter.cu:    cudaVerify(cudaFree(matporsirono_pm_d));
config_parameter.cu:    cudaVerify(cudaFree(matporsirono_phimax_d));
config_parameter.cu:    cudaVerify(cudaFree(matporsirono_phi0_d));
config_parameter.cu:    cudaVerify(cudaFree(matporsirono_delta_d));
config_parameter.cu:    cudaVerify(cudaFree(matporepsilon_kappa_d));
config_parameter.cu:    cudaVerify(cudaFree(matporepsilon_alpha_0_d));
config_parameter.cu:    cudaVerify(cudaFree(matporepsilon_epsilon_e_d));
config_parameter.cu:    cudaVerify(cudaFree(matporepsilon_epsilon_x_d));
config_parameter.cu:    cudaVerify(cudaFree(matporepsilon_epsilon_c_d));
config_parameter.cu:    cudaVerify(cudaFree(matjc_y0_d));
config_parameter.cu:    cudaVerify(cudaFree(matjc_B_d));
config_parameter.cu:    cudaVerify(cudaFree(matjc_n_d));
config_parameter.cu:    cudaVerify(cudaFree(matjc_m_d));
config_parameter.cu:    cudaVerify(cudaFree(matjc_edot0_d));
config_parameter.cu:    cudaVerify(cudaFree(matjc_C_d));
config_parameter.cu:    cudaVerify(cudaFree(matjc_Tref_d));
config_parameter.cu:    cudaVerify(cudaFree(matjc_Tmelt_d));
config_parameter.cu:    cudaVerify(cudaFree(matCp_d));
config_parameter.cu:    cudaVerify(cudaFree(matCV_d));
linalg.h: * This file is part of miluphcuda.
linalg.h: * miluphcuda is free software: you can redistribute it and/or modify
linalg.h: * miluphcuda is distributed in the hope that it will be useful,
linalg.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
stress.h: * This file is part of miluphcuda.
stress.h: * miluphcuda is free software: you can redistribute it and/or modify
stress.h: * miluphcuda is distributed in the hope that it will be useful,
stress.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
aneos.cu: * This file is part of miluphcuda.
aneos.cu: * miluphcuda is free software: you can redistribute it and/or modify
aneos.cu: * miluphcuda is distributed in the hope that it will be useful,
aneos.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
aneos.cu://        if ( (f = fopen("miluphcuda.warnings", "a")) == NULL )
aneos.cu://            ERRORTEXT("FILE ERROR! Cannot open 'miluphcuda.warnings' for appending!\n")
aneos.cu://        if ( (f = fopen("miluphcuda.warnings", "a")) == NULL )
aneos.cu://            ERRORTEXT("FILE ERROR! Cannot open 'miluphcuda.warnings' for appending!\n")
damage.cu: * This file is part of miluphcuda.
damage.cu: * miluphcuda is free software: you can redistribute it and/or modify
damage.cu: * miluphcuda is distributed in the hope that it will be useful,
damage.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
viscosity.h: * This file is part of miluphcuda.
viscosity.h: * miluphcuda is free software: you can redistribute it and/or modify
viscosity.h: * miluphcuda is distributed in the hope that it will be useful,
viscosity.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
boundary.h: * This file is part of miluphcuda.
boundary.h: * miluphcuda is free software: you can redistribute it and/or modify
boundary.h: * miluphcuda is distributed in the hope that it will be useful,
boundary.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
boundary.h:#include "cuda_utils.h"
miluph.h: * This file is part of miluphcuda.
miluph.h: * miluphcuda is free software: you can redistribute it and/or modify
miluph.h: * miluphcuda is distributed in the hope that it will be useful,
miluph.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
miluph.h:#define MILUPHCUDA_VERSION "devel"
miluph.h:#include "cuda_utils.h"
miluph.h:#include "cuda_profiler_api.h"
codemeta.json:    "name": "miluphcuda: Smooth particle hydrodynamics code",
codemeta.json:    "description": "miluphcuda is the CUDA port of the original miluph code; it runs on single Nvidia GPUs with compute capability 5.0 and higher and provides fast and efficient computation. The code can be used for hydrodynamical simulations and collision and impact physics, and features self-gravity via Barnes-Hut trees and porosity models such as P-alpha and epsilon-alpha. It can model solid bodies, including ductile and brittle materials, as well as non-viscous fluids, granular media, and porous continua.",
codemeta.json:        "https://github.com/christophmschaefer/miluphcuda"
timeintegration.h: * This file is part of miluphcuda.
timeintegration.h: * miluphcuda is free software: you can redistribute it and/or modify
timeintegration.h: * miluphcuda is distributed in the hope that it will be useful,
timeintegration.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
timeintegration.h:#include "cuda_utils.h"
little_helpers.h: * This file is part of miluphcuda.
little_helpers.h: * miluphcuda is free software: you can redistribute it and/or modify
little_helpers.h: * miluphcuda is distributed in the hope that it will be useful,
little_helpers.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
euler.cu: * This file is part of miluphcuda.
euler.cu: * miluphcuda is free software: you can redistribute it and/or modify
euler.cu: * miluphcuda is distributed in the hope that it will be useful,
euler.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
euler.cu:        cudaVerify(cudaMemcpyToSymbol(currentTimeD, &currentTime, sizeof(double)));
euler.cu:        cudaVerify(cudaMemcpyToSymbol(dt, &tmptimestep, sizeof(double)));
euler.cu:                cudaVerify(cudaMemcpyToSymbol(endTimeD, &endTime, sizeof(double)));
euler.cu:                            cudaVerify(cudaMemcpyToSymbol(dt, &tmptimestep, sizeof(double)));
euler.cu:                            cudaVerify(cudaMemcpyToSymbol(dt, &param.maxtimestep, sizeof(double)));
euler.cu:                        cudaVerifyKernel((integrateEuler<<<numberOfMultiprocessors, NUM_THREADS_EULER_INTEGRATOR>>>()));
checks.h: * This file is part of miluphcuda.
checks.h: * miluphcuda is free software: you can redistribute it and/or modify
checks.h: * miluphcuda is distributed in the hope that it will be useful,
checks.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
plasticity.h: * This file is part of miluphcuda.
plasticity.h: * miluphcuda is free software: you can redistribute it and/or modify
plasticity.h: * miluphcuda is distributed in the hope that it will be useful,
plasticity.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
Makefile:# This is the miluphcuda Makefile.
Makefile:CUDA_DIR    = /usr/local/cuda
Makefile:NVCC   = ${CUDA_DIR}/bin/nvcc
Makefile:NVFLAGS  = -ccbin ${CC} -x cu -c -dc -O3  -Xcompiler "-O3 -pthread" -Wno-deprecated-gpu-targets -DVERSION=\"$(GIT_VERSION)\"  --ptxas-options=-v
Makefile:CUDA_LINK_FLAGS = -dlink
Makefile:CUDA_LINK_OBJ = cuLink.o
Makefile:# important: compute capability, corresponding to GPU model (e.g., -arch=sm_52 for 5.2)
Makefile:GPU_ARCH = -arch=sm_52
Makefile:# compute capability    GPU models
Makefile:CUDA_LIB      = ${CUDA_DIR}
Makefile:INCLUDE_DIRS += -I$(CUDA_LIB)/include -I/usr/include/hdf5/serial -I/usr/lib/openmpi/include/
Makefile:LDFLAGS      += -ccbin ${CC} -L$(CUDA_LIB)/lib64 -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lcudart -lpthread -lconfig -lhdf5
Makefile:#LDFLAGS      += -ccbin ${CC} -L$(CUDA_LIB)/lib64 -lcudart -lpthread -lconfig
Makefile:all: miluphcuda
Makefile:CUDA_HEADERS =  cuda_utils.h  checks.h io.h  miluph.h  parameter.h  timeintegration.h  tree.h  euler.h rk2adaptive.h pressure.h soundspeed.h device_tools.h boundary.h predictor_corrector.h predictor_corrector_euler.h memory_handling.h plasticity.h porosity.h aneos.h kernel.h linalg.h xsph.h density.h rhs.h internal_forces.h velocity.h damage.h little_helpers.h gravity.h viscosity.h artificial_stress.h stress.h extrema.h sinking.h coupled_heun_rk4_sph_nbody.h rk4_pointmass.h config_parameter.h
Makefile:CUDA_OBJ = io.o  miluph.o  boundary.o timeintegration.o tree.o memory_handling.o euler.o rk2adaptive.o pressure.o soundspeed.o device_tools.o predictor_corrector.o predictor_corrector_euler.o plasticity.o porosity.o aneos.o kernel.o linalg.o xsph.o density.o rhs.o internal_forces.o velocity.o damage.o little_helpers.o gravity.o viscosity.o artificial_stress.o stress.o extrema.o sinking.o coupled_heun_rk4_sph_nbody.o rk4_pointmass.o config_parameter.o
Makefile:miluphcuda: $(OBJ) $(CUDA_OBJ)
Makefile:#	$(NVCC) $(GPU_ARCH) $(CUDA_LINK_FLAGS) -o $(CUDA_LINK_OBJ) $(CUDA_OBJ)
Makefile:	$(NVCC) $(GPU_ARCH) $(CUDA_OBJ) $(LDFLAGS) -Wno-deprecated-gpu-targets -o $@
Makefile:#	$(CC) $(OBJ) $(CUDA_OBJ) $(CUDA_LINK_OBJ) $(LDFLAGS) -o $@
Makefile:	$(NVCC) $(GPU_ARCH) $(NVFLAGS) $(INCLUDE_DIRS) $<
Makefile:	@rm -f	*.o miluphcuda
Makefile:$(CUDA_OBJ): $(HEADERS) $(CUDA_HEADERS) Makefile
memory_handling.cu: * This file is part of miluphcuda.
memory_handling.cu: * miluphcuda is free software: you can redistribute it and/or modify
memory_handling.cu: * miluphcuda is distributed in the hope that it will be useful,
memory_handling.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->x, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->vx, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->ax, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->feedback_ax, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->y, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->vy, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->ay, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->feedback_ay, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->z, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->vz, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->az, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->feedback_az, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->m, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->rmin, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->rmax, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->feels_particles, integermemorySizeForPointmasses));
memory_handling.cu://	cudaVerify(cudaMalloc((void**)&a->tensorialCorrectionMatrix, memorySizeForStress));
memory_handling.cu:        cudaVerify(cudaMalloc((void**)&a->tensorialCorrectiondWdrr, MAX_NUM_INTERACTIONS * maxNumberOfParticles * sizeof(double)));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->dedt, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->muijmax, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->drhodt, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->S, memorySizeForStress));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->dSdt, memorySizeForStress));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->local_strain, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&a->ep, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&a->edotp, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->Tshear, memorySizeForStress));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->beta, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->beta_old, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->divv_old, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->dbetadt, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->d, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->damage_total, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->dddt, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->numFlaws, memorySizeForInteractions));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->numActiveFlaws, memorySizeForInteractions));
memory_handling.cu:	    cudaVerify(cudaMalloc((void**)&a->flaws, memorySizeForActivationThreshold));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->damage_porjutzi, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->ddamage_porjutzidt, memorySizeForParticles));
memory_handling.cu:        cudaVerify(cudaMalloc((void**)&a->h0, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->real_partner, memorySizeForInteractions));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->pold, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->alpha_jutzi, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->alpha_jutzi_old, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->dalphadt, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->dp, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->dalphadp, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->dalphadrho, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->delpdelrho, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->delpdele, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->f, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&a->compressive_strength, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&a->tensile_strength, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&a->shear_strength, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&a->K, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&a->rho_0prime, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&a->rho_c_plus, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&a->rho_c_minus, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&a->flag_rho_0prime, memorySizeForInteractions));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&a->flag_plastic, memorySizeForInteractions));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&a->alpha_epspor, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&a->dalpha_epspordt, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&a->epsilon_v, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&a->depsilon_vdt, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&a->x0, memorySizeForTree));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&a->y0, memorySizeForTree));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&a->z0, memorySizeForTree));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->x, memorySizeForTree));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->y, memorySizeForTree));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->vx, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->vy, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->dxdt, memorySizeForParticles));
memory_handling.cu: 	cudaVerify(cudaMalloc((void**)&a->dydt, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->xsphvx, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->xsphvy, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->ax, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->g_ax, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->ay, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->g_ay, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->m, memorySizeForTree));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->h, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->dhdt, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->sml_omega, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->rho, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->p, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->e, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->cs, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->noi, memorySizeForInteractions));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->depth, memorySizeForInteractions));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->p_min, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&a->p_max, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&a->rho_min, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&a->rho_max, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->e_min, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&a->e_max, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&a->cs_min, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&a->cs_max, memorySizeForParticles));
memory_handling.cu://	cudaVerify(cudaMalloc((void**)&a->materialId, memorySizeForInteractions));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->T, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->dTdt, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->jc_f, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->z, memorySizeForTree));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->dzdt, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->vz, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->az, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->g_az, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&a->xsphvz, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMemset(a->ax, 0, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMemset(a->g_ax, 0, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMemset(a->ay, 0, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMemset(a->g_ay, 0, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMemset(a->az, 0, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMemset(a->g_az, 0, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->g_ax, src->g_ax, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->g_ay, src->g_ay, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->g_az, src->g_az, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->ax, src->ax, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->vx, src->vx, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->feedback_ax, src->feedback_ax, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->ay, src->ay, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->vy, src->vy, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->feedback_ay, src->feedback_ay, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->az, src->az, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->vz, src->vz, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->feedback_az, src->feedback_az, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->ax, src->ax, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->g_ax, src->g_ax, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->dxdt, src->dxdt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->ay, src->ay, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->g_ay, src->g_ay, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->dydt, src->dydt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->az, src->az, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->g_az, src->g_az, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->dzdt, src->dzdt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->drhodt, src->drhodt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->dhdt, src->dhdt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->sml_omega, src->sml_omega, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->dalphadt, src->dalphadt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->ddamage_porjutzidt, src->ddamage_porjutzidt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->dalpha_epspordt, src->dalpha_epspordt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->depsilon_vdt, src->depsilon_vdt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->dedt, src->dedt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->dSdt, src->dSdt, memorySizeForStress, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->edotp, src->edotp, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(dst->dbetadt, src->dbetadt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->dTdt, src->dTdt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->dddt, src->dddt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->numActiveFlaws, src->numActiveFlaws, memorySizeForInteractions, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy((*dst).m, (*src).m, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy((*dst).feels_particles, (*src).feels_particles, integermemorySizeForPointmasses, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy((*dst).rmin, (*src).rmin, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy((*dst).rmax, (*src).rmax, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy((*dst).x0, (*src).x0, memorySizeForTree, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy((*dst).y0, (*src).y0, memorySizeForTree, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy((*dst).z0, (*src).z0, memorySizeForTree, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy((*dst).m, (*src).m, memorySizeForTree, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy((*dst).h, (*src).h, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy((*dst).cs, (*src).cs, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    //cudaVerify(cudaMemcpy((*dst).materialId, (*src).materialId, memorySizeForInteractions, cudaMemcpyDeviceToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(dst->numFlaws, src->numFlaws, memorySizeForInteractions, cudaMemcpyDeviceToDevice));
memory_handling.cu:    //cudaVerify(cudaMemcpy(dst->flaws, src->flaws, memorySizeForActivationThreshold, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->x, src->x, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->m, src->m, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->vx, src->vx, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->y, src->y, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->vy, src->vy, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->z, src->z, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->vz, src->vz, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->x, src->x, memorySizeForTree, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->x0, src->x0, memorySizeForTree, cudaMemcpyDeviceToDevice));
memory_handling.cu:    //cudaVerify(cudaMemcpy((*dst).materialId, (*src).materialId, memorySizeForInteractions, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->y, src->y, memorySizeForTree, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->y0, src->y0, memorySizeForTree, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->vy, src->vy, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->z0, src->z0, memorySizeForTree, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->vx, src->vx, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->rho, src->rho, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->h, src->h, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->e, src->e, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->alpha_jutzi, src->alpha_jutzi, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->alpha_jutzi_old, src->alpha_jutzi, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->dalphadp, src->dalphadp, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->dalphadrho, src->dalphadrho, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->dp, src->dp, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->delpdelrho, src->delpdelrho, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->delpdele, src->delpdele, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->f, src->f, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->p, src->p, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->pold, src->pold, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->damage_porjutzi, src->damage_porjutzi, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->p_min, src->p_min, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->p_max, src->p_max, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->rho_min, src->rho_min, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->rho_max, src->rho_max, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->e_min, src->e_min, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->e_max, src->e_max, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->cs_min, src->cs_min, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->cs_max, src->cs_max, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->compressive_strength, src->compressive_strength, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->tensile_strength, src->tensile_strength, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->shear_strength, src->shear_strength, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->K, src->K, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->rho_0prime, src->rho_0prime, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->rho_c_plus, src->rho_c_plus, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->rho_c_minus, src->rho_c_minus, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->flag_rho_0prime, src->flag_rho_0prime, memorySizeForInteractions, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->flag_plastic, src->flag_plastic, memorySizeForInteractions, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->alpha_epspor, src->alpha_epspor, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->dalpha_epspordt, src->dalpha_epspordt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->epsilon_v, src->epsilon_v, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->depsilon_vdt, src->depsilon_vdt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->z, src->z, memorySizeForTree, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->vz, src->vz, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->S, src->S, memorySizeForStress, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->ep, src->ep, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->Tshear, src->Tshear, memorySizeForStress, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->beta, src->beta, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->beta_old, src->beta_old, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->divv_old, src->divv_old, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->T, src->T, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->jc_f, src->jc_f, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->d, src->d, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->damage_total, src->damage_total, memorySizeForParticles, cudaMemcpyDeviceToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(dst->numActiveFlaws, src->numActiveFlaws, memorySizeForInteractions, cudaMemcpyDeviceToDevice));
memory_handling.cu:	cudaVerify(cudaFree(a->x));
memory_handling.cu:	cudaVerify(cudaFree(a->vx));
memory_handling.cu:	cudaVerify(cudaFree(a->ax));
memory_handling.cu:	cudaVerify(cudaFree(a->feedback_ax));
memory_handling.cu:	cudaVerify(cudaFree(a->m));
memory_handling.cu:	cudaVerify(cudaFree(a->feels_particles));
memory_handling.cu:	cudaVerify(cudaFree(a->rmin));
memory_handling.cu:	cudaVerify(cudaFree(a->rmax));
memory_handling.cu:	cudaVerify(cudaFree(a->y));
memory_handling.cu:	cudaVerify(cudaFree(a->vy));
memory_handling.cu:	cudaVerify(cudaFree(a->ay));
memory_handling.cu:	cudaVerify(cudaFree(a->feedback_ay));
memory_handling.cu:	cudaVerify(cudaFree(a->z));
memory_handling.cu:	cudaVerify(cudaFree(a->vz));
memory_handling.cu:	cudaVerify(cudaFree(a->az));
memory_handling.cu:	cudaVerify(cudaFree(a->feedback_az));
memory_handling.cu:	cudaVerify(cudaFree(a->x));
memory_handling.cu:	cudaVerify(cudaFree(a->x0));
memory_handling.cu:	cudaVerify(cudaFree(a->dxdt));
memory_handling.cu:	cudaVerify(cudaFree(a->vx));
memory_handling.cu:	cudaVerify(cudaFree(a->ax));
memory_handling.cu:	cudaVerify(cudaFree(a->g_ax));
memory_handling.cu:	cudaVerify(cudaFree(a->m));
memory_handling.cu:	cudaVerify(cudaFree(a->dydt));
memory_handling.cu:	cudaVerify(cudaFree(a->y));
memory_handling.cu:	cudaVerify(cudaFree(a->y0));
memory_handling.cu:	cudaVerify(cudaFree(a->vy0));
memory_handling.cu:	cudaVerify(cudaFree(a->vy));
memory_handling.cu:	cudaVerify(cudaFree(a->ay));
memory_handling.cu:	cudaVerify(cudaFree(a->g_ay));
memory_handling.cu:	cudaVerify(cudaFree(a->xsphvx));
memory_handling.cu:	cudaVerify(cudaFree(a->xsphvy));
memory_handling.cu:	cudaVerify(cudaFree(a->h));
memory_handling.cu:	cudaVerify(cudaFree(a->rho));
memory_handling.cu:	cudaVerify(cudaFree(a->p));
memory_handling.cu:	cudaVerify(cudaFree(a->e));
memory_handling.cu:	cudaVerify(cudaFree(a->cs));
memory_handling.cu:	cudaVerify(cudaFree(a->noi));
memory_handling.cu:	cudaVerify(cudaFree(a->depth));
memory_handling.cu:	cudaVerify(cudaFree(a->p_min));
memory_handling.cu:	cudaVerify(cudaFree(a->p_max));
memory_handling.cu:	cudaVerify(cudaFree(a->rho_min));
memory_handling.cu:	cudaVerify(cudaFree(a->rho_max));
memory_handling.cu:	cudaVerify(cudaFree(a->e_min));
memory_handling.cu:	cudaVerify(cudaFree(a->e_max));
memory_handling.cu:	cudaVerify(cudaFree(a->cs_min));
memory_handling.cu:	cudaVerify(cudaFree(a->cs_max));
memory_handling.cu:	//cudaVerify(cudaFree(a->materialId));
memory_handling.cu:	cudaVerify(cudaFree(a->z));
memory_handling.cu:	cudaVerify(cudaFree(a->z0));
memory_handling.cu:	cudaVerify(cudaFree(a->dzdt));
memory_handling.cu:	cudaVerify(cudaFree(a->vz));
memory_handling.cu:	cudaVerify(cudaFree(a->xsphvz));
memory_handling.cu:	cudaVerify(cudaFree(a->az));
memory_handling.cu:	cudaVerify(cudaFree(a->g_az));
memory_handling.cu:	cudaVerify(cudaFree(a->muijmax));
memory_handling.cu:	cudaVerify(cudaFree(a->divv));
memory_handling.cu:	cudaVerify(cudaFree(a->curlv));
memory_handling.cu:	cudaVerify(cudaFree(a->beta));
memory_handling.cu:	cudaVerify(cudaFree(a->beta_old));
memory_handling.cu:	cudaVerify(cudaFree(a->divv_old));
memory_handling.cu:	cudaVerify(cudaFree(a->dbetadt));
memory_handling.cu:	//cudaVerify(cudaFree(a->tensorialCorrectionMatrix));
memory_handling.cu:	    cudaVerify(cudaFree(a->tensorialCorrectiondWdrr));
memory_handling.cu:	cudaVerify(cudaFree(a->dedt));
memory_handling.cu:	cudaVerify(cudaFree(a->real_partner));
memory_handling.cu:	cudaVerify(cudaFree(a->drhodt));
memory_handling.cu:	cudaVerify(cudaFree(a->dhdt));
memory_handling.cu:    cudaVerify(cudaFree(a->sml_omega));
memory_handling.cu:	cudaVerify(cudaFree(a->S));
memory_handling.cu:	cudaVerify(cudaFree(a->dSdt));
memory_handling.cu:	cudaVerify(cudaFree(a->local_strain));
memory_handling.cu:    cudaVerify(cudaFree(a->ep));
memory_handling.cu:    cudaVerify(cudaFree(a->edotp));
memory_handling.cu:	cudaVerify(cudaFree(a->Tshear));
memory_handling.cu:	cudaVerify(cudaFree(a->T));
memory_handling.cu:	cudaVerify(cudaFree(a->dTdt));
memory_handling.cu:	cudaVerify(cudaFree(a->jc_f));
memory_handling.cu:	cudaVerify(cudaFree(a->pold));
memory_handling.cu:	cudaVerify(cudaFree(a->alpha_jutzi));
memory_handling.cu:	cudaVerify(cudaFree(a->alpha_jutzi_old));
memory_handling.cu:	cudaVerify(cudaFree(a->dalphadt));
memory_handling.cu:	cudaVerify(cudaFree(a->f));
memory_handling.cu:	cudaVerify(cudaFree(a->dalphadp));
memory_handling.cu:	cudaVerify(cudaFree(a->dp));
memory_handling.cu:	cudaVerify(cudaFree(a->delpdelrho));
memory_handling.cu:	cudaVerify(cudaFree(a->delpdele));
memory_handling.cu:	cudaVerify(cudaFree(a->dalphadrho));
memory_handling.cu:    cudaVerify(cudaFree(a->compressive_strength));
memory_handling.cu:    cudaVerify(cudaFree(a->tensile_strength));
memory_handling.cu:    cudaVerify(cudaFree(a->shear_strength));
memory_handling.cu:    cudaVerify(cudaFree(a->K));
memory_handling.cu:    cudaVerify(cudaFree(a->rho_0prime));
memory_handling.cu:    cudaVerify(cudaFree(a->rho_c_plus));
memory_handling.cu:    cudaVerify(cudaFree(a->rho_c_minus));
memory_handling.cu:    cudaVerify(cudaFree(a->flag_rho_0prime));
memory_handling.cu:    cudaVerify(cudaFree(a->flag_plastic));
memory_handling.cu:    cudaVerify(cudaFree(a->alpha_epspor));
memory_handling.cu:    cudaVerify(cudaFree(a->dalpha_epspordt));
memory_handling.cu:    cudaVerify(cudaFree(a->epsilon_v));
memory_handling.cu:    cudaVerify(cudaFree(a->depsilon_vdt));
memory_handling.cu:	cudaVerify(cudaFree(a->d));
memory_handling.cu:	cudaVerify(cudaFree(a->damage_total));
memory_handling.cu:	cudaVerify(cudaFree(a->dddt));
memory_handling.cu:	cudaVerify(cudaFree(a->numFlaws));
memory_handling.cu:	cudaVerify(cudaFree(a->numActiveFlaws));
memory_handling.cu:	    cudaVerify(cudaFree(a->flaws));
memory_handling.cu:	    cudaVerify(cudaFree(a->h0));
memory_handling.cu:	cudaVerify(cudaFree(a->damage_porjutzi));
memory_handling.cu:	cudaVerify(cudaFree(a->ddamage_porjutzidt));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.x, memorySizeForTree));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.vx, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.ax, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.g_ax, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.y, memorySizeForTree));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.vy, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.ay, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.g_ay, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.z, memorySizeForTree));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.vz, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.az, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.g_az, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.m, memorySizeForTree));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.h, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.rho, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.p, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.e, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.cs, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&pointmass_host.x, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&pointmass_host.vx, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&pointmass_host.ax, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&pointmass_device.x, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&pointmass_device.vx, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&pointmass_device.ax, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&pointmass_device.feedback_ax, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&pointmass_host.y, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&pointmass_host.vy, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&pointmass_host.ay, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&pointmass_device.y, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&pointmass_device.vy, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&pointmass_device.ay, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&pointmass_device.feedback_ay, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&pointmass_host.z, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&pointmass_host.vz, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&pointmass_host.az, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&pointmass_device.z, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&pointmass_device.vz, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&pointmass_device.az, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&pointmass_device.feedback_az, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&pointmass_host.rmin, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&pointmass_host.rmax, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&pointmass_device.rmin, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&pointmass_device.rmax, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&pointmass_host.m, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&pointmass_device.m, memorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&pointmass_host.feels_particles, integermemorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&pointmass_device.feels_particles, integermemorySizeForPointmasses));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.p_min, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.p_max, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.rho_min, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.rho_max, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.e_min, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.e_max, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.cs_min, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.cs_max, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.noi, memorySizeForInteractions));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.depth, memorySizeForInteractions));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&interactions_host, memorySizeForInteractions*MAX_NUM_INTERACTIONS));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.materialId, memorySizeForInteractions));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&childList_host, memorySizeForChildren));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.muijmax, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.divv, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.curlv, memorySizeForParticles*DIM));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.beta, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.beta_old, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.divv_old, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.dbetadt, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.tensorialCorrectionMatrix, memorySizeForStress));
memory_handling.cu:	//cudaVerify(cudaMalloc((void**)&p_device.tensorialCorrectiondWdrr, MAX_NUM_INTERACTIONS * maxNumberOfParticles * sizeof(double)));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.shepard_correction, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.dedt, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.dedt, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.drhodt, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.drhodt, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.S, memorySizeForStress));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.dSdt, memorySizeForStress));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.S, memorySizeForStress));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.dSdt, memorySizeForStress));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.local_strain, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&p_device.local_strain, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**) &p_device.sigma, memorySizeForStress));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&p_device.plastic_f, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.ep, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&p_device.ep, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&p_device.edotp, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.Tshear, memorySizeForStress));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.Tshear, memorySizeForStress));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.eta, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**) &p_device.R, memorySizeForStress));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.T, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.T, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.dTdt, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.jc_f, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.d, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.dddt, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.d, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.damage_total, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.dddt, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.numFlaws, memorySizeForInteractions));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.numFlaws, memorySizeForInteractions));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.numActiveFlaws, memorySizeForInteractions));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.numActiveFlaws, memorySizeForInteractions));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.flaws, memorySizeForActivationThreshold));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.flaws, memorySizeForActivationThreshold));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.damage_porjutzi, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.ddamage_porjutzidt, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.damage_porjutzi, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.ddamage_porjutzidt, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.h0, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.h0, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.real_partner, memorySizeForInteractions));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.alpha_jutzi, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.alpha_jutzi_old, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMallocHost((void**)&p_host.pold, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.dalphadt, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.pold, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.alpha_jutzi, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.alpha_jutzi_old, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.dalphadt, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.dalphadp, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.dp, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.dalphadrho, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.f, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.delpdelrho, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.delpdele, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.cs_old, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.compressive_strength, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.tensile_strength, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.shear_strength, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.rho_0prime, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.rho_c_plus, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.rho_c_minus, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.K, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.flag_rho_0prime, memorySizeForInteractions));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.flag_plastic, memorySizeForInteractions));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&p_device.compressive_strength, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&p_device.tensile_strength, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&p_device.shear_strength, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&p_device.K, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&p_device.rho_0prime, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&p_device.rho_c_plus, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&p_device.rho_c_minus, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&p_device.flag_rho_0prime, memorySizeForInteractions));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&p_device.flag_plastic, memorySizeForInteractions));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.alpha_epspor, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.epsilon_v, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&p_device.alpha_epspor, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&p_device.dalpha_epspordt, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&p_device.epsilon_v, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&p_device.depsilon_vdt, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.x, memorySizeForTree));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.g_x, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.g_local_cellsize, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.vx, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.dxdt, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.y, memorySizeForTree));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.g_y, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.vy, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.dydt, memorySizeForParticles));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&p_device.y0, memorySizeForTree));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&p_device.vy0, memorySizeForTree));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.vy0, memorySizeForTree));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&p_device.x0, memorySizeForTree));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&p_device.vx0, memorySizeForTree));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.vx0, memorySizeForTree));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&p_device.z0, memorySizeForTree));
memory_handling.cu:    cudaVerify(cudaMalloc((void**)&p_device.vz0, memorySizeForTree));
memory_handling.cu:    cudaVerify(cudaMallocHost((void**)&p_host.vz0, memorySizeForTree));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.xsphvx, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.xsphvy, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.ax, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.g_ax, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.ay, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.g_ay, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.m, memorySizeForTree));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.h, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.dhdt, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.sml_omega, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.rho, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.p, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.e, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.cs, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.depth, memorySizeForInteractions));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.noi, memorySizeForInteractions));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.materialId, memorySizeForInteractions));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.materialId0, memorySizeForInteractions));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.p_min, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.p_max, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.rho_min, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.rho_max, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.e_min, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.e_max, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.cs_min, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.cs_max, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&interactions, memorySizeForInteractions*MAX_NUM_INTERACTIONS));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&childListd, memorySizeForChildren));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.z, memorySizeForTree));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.g_z, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.dzdt, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.vz, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.az, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.g_az, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMalloc((void**)&p_device.xsphvz, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMemset(p_device.ax, 0, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMemset(p_device.g_ax, 0, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMemset(p_device.ay, 0, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMemset(p_device.g_ay, 0, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMemset(p_device.az, 0, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMemset(p_device.g_az, 0, memorySizeForParticles));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.x0, p_host.x, memorySizeForTree, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.x, p_host.x, memorySizeForTree, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.vx, p_host.vx, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.vx0, p_host.vx0, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.y0, p_host.y, memorySizeForTree, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.y, p_host.y, memorySizeForTree, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.vy, p_host.vy, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.vy0, p_host.vy0, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.z0, p_host.z, memorySizeForTree, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(pointmass_device.x, pointmass_host.x, memorySizeForPointmasses, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(pointmass_device.vx, pointmass_host.vx, memorySizeForPointmasses, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(pointmass_device.y, pointmass_host.y, memorySizeForPointmasses, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(pointmass_device.vy, pointmass_host.vy, memorySizeForPointmasses, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(pointmass_device.z, pointmass_host.z, memorySizeForPointmasses, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(pointmass_device.vz, pointmass_host.vz, memorySizeForPointmasses, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(pointmass_device.rmin, pointmass_host.rmin, memorySizeForPointmasses, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(pointmass_device.rmax, pointmass_host.rmax, memorySizeForPointmasses, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(pointmass_device.m, pointmass_host.m, memorySizeForPointmasses, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(pointmass_device.feels_particles, pointmass_host.feels_particles, integermemorySizeForPointmasses, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.h, p_host.h, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.cs, p_host.cs, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.m, p_host.m, memorySizeForTree, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.rho, p_host.rho, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.e, p_host.e, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.S, p_host.S, memorySizeForStress, cudaMemcpyHostToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(p_device.ep, p_host.ep, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.Tshear, p_host.Tshear, memorySizeForStress, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.alpha_jutzi, p_host.alpha_jutzi, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.alpha_jutzi_old, p_host.alpha_jutzi_old, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.p, p_host.p, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.pold, p_host.pold, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(p_device.p_min, p_host.p_min, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(p_device.p_max, p_host.p_max, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(p_device.rho_min, p_host.rho_min, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(p_device.rho_max, p_host.rho_max, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(p_device.e_min, p_host.e_min, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(p_device.e_max, p_host.e_max, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(p_device.cs_min, p_host.cs_min, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(p_device.cs_max, p_host.cs_max, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(p_device.compressive_strength, p_host.compressive_strength, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(p_device.tensile_strength, p_host.tensile_strength, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(p_device.shear_strength, p_host.shear_strength, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(p_device.rho_0prime, p_host.rho_0prime, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(p_device.rho_c_plus, p_host.rho_c_plus, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(p_device.rho_c_minus, p_host.rho_c_minus, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(p_device.K, p_host.K, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(p_device.flag_rho_0prime, p_host.flag_rho_0prime, memorySizeForInteractions, cudaMemcpyHostToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(p_device.flag_plastic, p_host.flag_plastic, memorySizeForInteractions, cudaMemcpyHostToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(p_device.alpha_epspor, p_host.alpha_epspor, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(p_device.epsilon_v, p_host.epsilon_v, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(p_device.h0, p_host.h0, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.T, p_host.T, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.d, p_host.d, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.numFlaws, p_host.numFlaws, memorySizeForInteractions, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.numActiveFlaws, p_host.numActiveFlaws, memorySizeForInteractions, cudaMemcpyHostToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(p_device.flaws, p_host.flaws, memorySizeForActivationThreshold, cudaMemcpyHostToDevice));
memory_handling.cu:    cudaVerify(cudaMemcpy(p_device.damage_porjutzi, p_host.damage_porjutzi, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.noi, p_host.noi, memorySizeForInteractions, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.materialId, p_host.materialId, memorySizeForInteractions, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.materialId0, p_host.materialId, memorySizeForInteractions, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.z, p_host.z, memorySizeForTree, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemcpy(p_device.vz, p_host.vz, memorySizeForParticles, cudaMemcpyHostToDevice));
memory_handling.cu:	cudaVerify(cudaMemset((void *) childListd, -1, memorySizeForChildren));
memory_handling.cu:	cudaVerify(cudaFree(p_device.x));
memory_handling.cu:	cudaVerify(cudaFree(p_device.g_x));
memory_handling.cu:	cudaVerify(cudaFree(p_device.g_local_cellsize));
memory_handling.cu:	cudaVerify(cudaFree(p_device.depth));
memory_handling.cu:	cudaVerify(cudaFree(p_device.x0));
memory_handling.cu:	cudaVerify(cudaFree(p_device.dxdt));
memory_handling.cu:	cudaVerify(cudaFree(p_device.vx));
memory_handling.cu:	cudaVerify(cudaFree(p_device.vx0));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.vx0));
memory_handling.cu:	cudaVerify(cudaFree(p_device.ax));
memory_handling.cu:	cudaVerify(cudaFree(p_device.g_ax));
memory_handling.cu:	cudaVerify(cudaFree(p_device.m));
memory_handling.cu:	cudaVerify(cudaFree(p_device.vy0));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.vy0));
memory_handling.cu:	cudaVerify(cudaFree(p_device.vz0));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.vz0));
memory_handling.cu:	cudaVerify(cudaFree(p_device.y));
memory_handling.cu:	cudaVerify(cudaFree(p_device.g_y));
memory_handling.cu:	cudaVerify(cudaFree(p_device.y0));
memory_handling.cu:	cudaVerify(cudaFree(p_device.vy));
memory_handling.cu:	cudaVerify(cudaFree(p_device.dydt));
memory_handling.cu:	cudaVerify(cudaFree(p_device.ay));
memory_handling.cu:	cudaVerify(cudaFree(p_device.g_ay));
memory_handling.cu:	cudaVerify(cudaFree(pointmass_device.x));
memory_handling.cu:	cudaVerify(cudaFree(pointmass_device.vx));
memory_handling.cu:	cudaVerify(cudaFree(pointmass_device.ax));
memory_handling.cu:	cudaVerify(cudaFree(pointmass_device.feedback_ax));
memory_handling.cu:	cudaVerify(cudaFree(pointmass_device.y));
memory_handling.cu:	cudaVerify(cudaFree(pointmass_device.vy));
memory_handling.cu:	cudaVerify(cudaFree(pointmass_device.ay));
memory_handling.cu:	cudaVerify(cudaFree(pointmass_device.feedback_ay));
memory_handling.cu:	cudaVerify(cudaFree(pointmass_device.z));
memory_handling.cu:	cudaVerify(cudaFree(pointmass_device.vz));
memory_handling.cu:	cudaVerify(cudaFree(pointmass_device.az));
memory_handling.cu:	cudaVerify(cudaFree(pointmass_device.feedback_az));
memory_handling.cu:	cudaVerify(cudaFree(pointmass_device.m));
memory_handling.cu:	cudaVerify(cudaFree(pointmass_device.feels_particles));
memory_handling.cu:	cudaVerify(cudaFree(pointmass_device.rmin));
memory_handling.cu:	cudaVerify(cudaFree(pointmass_device.rmax));
memory_handling.cu:	cudaVerify(cudaFreeHost(pointmass_host.x));
memory_handling.cu:	cudaVerify(cudaFreeHost(pointmass_host.vx));
memory_handling.cu:	cudaVerify(cudaFreeHost(pointmass_host.ax));
memory_handling.cu:	cudaVerify(cudaFreeHost(pointmass_host.y));
memory_handling.cu:	cudaVerify(cudaFreeHost(pointmass_host.vy));
memory_handling.cu:	cudaVerify(cudaFreeHost(pointmass_host.ay));
memory_handling.cu:	cudaVerify(cudaFreeHost(pointmass_host.z));
memory_handling.cu:	cudaVerify(cudaFreeHost(pointmass_host.vz));
memory_handling.cu:	cudaVerify(cudaFreeHost(pointmass_host.az));
memory_handling.cu:	cudaVerify(cudaFreeHost(pointmass_host.m));
memory_handling.cu:	cudaVerify(cudaFreeHost(pointmass_host.feels_particles));
memory_handling.cu:	cudaVerify(cudaFreeHost(pointmass_host.rmin));
memory_handling.cu:	cudaVerify(cudaFreeHost(pointmass_host.rmax));
memory_handling.cu:	cudaVerify(cudaFree(p_device.xsphvx));
memory_handling.cu:	cudaVerify(cudaFree(p_device.xsphvy));
memory_handling.cu:	cudaVerify(cudaFree(p_device.h));
memory_handling.cu:	cudaVerify(cudaFree(p_device.rho));
memory_handling.cu:	cudaVerify(cudaFree(p_device.p));
memory_handling.cu:	cudaVerify(cudaFree(p_device.e));
memory_handling.cu:	cudaVerify(cudaFree(p_device.cs));
memory_handling.cu:	cudaVerify(cudaFree(p_device.noi));
memory_handling.cu:	cudaVerify(cudaFree(p_device.p_min));
memory_handling.cu:    cudaVerify(cudaFree(p_device.p_max));
memory_handling.cu:    cudaVerify(cudaFree(p_device.rho_min));
memory_handling.cu:    cudaVerify(cudaFree(p_device.rho_max));
memory_handling.cu:	cudaVerify(cudaFree(p_device.e_min));
memory_handling.cu:    cudaVerify(cudaFree(p_device.e_max));
memory_handling.cu:    cudaVerify(cudaFree(p_device.cs_min));
memory_handling.cu:    cudaVerify(cudaFree(p_device.cs_max));
memory_handling.cu:	cudaVerify(cudaFree(p_device.muijmax));
memory_handling.cu:	cudaVerify(cudaFree(p_device.beta));
memory_handling.cu:	cudaVerify(cudaFree(p_device.beta_old));
memory_handling.cu:	cudaVerify(cudaFree(p_device.divv_old));
memory_handling.cu:	cudaVerify(cudaFree(interactions));
memory_handling.cu:	cudaVerify(cudaFree(p_device.materialId));
memory_handling.cu:	cudaVerify(cudaFree(p_device.materialId0));
memory_handling.cu:	cudaVerify(cudaFree(childListd));
memory_handling.cu:	cudaVerify(cudaFree(p_device.z));
memory_handling.cu:	cudaVerify(cudaFree(p_device.g_z));
memory_handling.cu:	cudaVerify(cudaFree(p_device.z0));
memory_handling.cu:	cudaVerify(cudaFree(p_device.dzdt));
memory_handling.cu:	cudaVerify(cudaFree(p_device.vz));
memory_handling.cu:	cudaVerify(cudaFree(p_device.xsphvz));
memory_handling.cu:	cudaVerify(cudaFree(p_device.az));
memory_handling.cu:	cudaVerify(cudaFree(p_device.g_az));
memory_handling.cu:	cudaVerify(cudaFree(p_device.tensorialCorrectionMatrix));
memory_handling.cu:	//cudaVerify(cudaFree(p_device.tensorialCorrectiondWdrr));
memory_handling.cu:	cudaVerify(cudaFree(p_device.shepard_correction));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.dedt));
memory_handling.cu:	cudaVerify(cudaFree(p_device.dedt));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.drhodt));
memory_handling.cu:	cudaVerify(cudaFree(p_device.drhodt));
memory_handling.cu:	cudaVerify(cudaFree(p_device.dhdt));
memory_handling.cu:	cudaVerify(cudaFree(p_device.sml_omega));
memory_handling.cu:	cudaVerify(cudaFree(p_device.Tshear));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.Tshear));
memory_handling.cu:	cudaVerify(cudaFree(p_device.eta));
memory_handling.cu:	cudaVerify(cudaFree(p_device.S));
memory_handling.cu:    cudaVerify(cudaFreeHost(p_host.ep));
memory_handling.cu:	cudaVerify(cudaFree(p_device.dSdt));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.S));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.dSdt));
memory_handling.cu:	cudaVerify(cudaFree(p_device.local_strain));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.local_strain));
memory_handling.cu:    cudaVerify(cudaFree(p_device.plastic_f));
memory_handling.cu:	cudaVerify(cudaFree(p_device.sigma));
memory_handling.cu:    cudaVerify(cudaFree(p_device.ep));
memory_handling.cu:    cudaVerify(cudaFree(p_device.edotp));
memory_handling.cu:	cudaVerify(cudaFree(p_device.R));
memory_handling.cu:	cudaVerify(cudaFree(p_device.T));
memory_handling.cu:	cudaVerify(cudaFree(p_device.dTdt));
memory_handling.cu:	cudaVerify(cudaFree(p_device.jc_f));
memory_handling.cu:	cudaVerify(cudaFree(p_device.real_partner));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.d));
memory_handling.cu:	cudaVerify(cudaFree(p_device.d));
memory_handling.cu:	cudaVerify(cudaFree(p_device.damage_total));
memory_handling.cu:	cudaVerify(cudaFree(p_device.dddt));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.dddt));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.numFlaws));
memory_handling.cu:	cudaVerify(cudaFree(p_device.numFlaws));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.numActiveFlaws));
memory_handling.cu:	cudaVerify(cudaFree(p_device.numActiveFlaws));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.flaws));
memory_handling.cu:	cudaVerify(cudaFree(p_device.flaws));
memory_handling.cu:	cudaVerify(cudaFree(p_device.damage_porjutzi));
memory_handling.cu:	cudaVerify(cudaFree(p_device.cs_old));
memory_handling.cu:	cudaVerify(cudaFree(p_device.ddamage_porjutzidt));
memory_handling.cu:	cudaVerify(cudaFree(p_device.alpha_jutzi));
memory_handling.cu:	cudaVerify(cudaFree(p_device.alpha_jutzi_old));
memory_handling.cu:	cudaVerify(cudaFree(p_device.pold));
memory_handling.cu:	cudaVerify(cudaFree(p_device.dalphadt));
memory_handling.cu:	cudaVerify(cudaFree(p_device.dalphadp));
memory_handling.cu:	cudaVerify(cudaFree(p_device.dp));
memory_handling.cu:	cudaVerify(cudaFree(p_device.dalphadrho));
memory_handling.cu:	cudaVerify(cudaFree(p_device.f));
memory_handling.cu:	cudaVerify(cudaFree(p_device.delpdelrho));
memory_handling.cu:	cudaVerify(cudaFree(p_device.delpdele));
memory_handling.cu:    cudaVerify(cudaFree(p_device.compressive_strength));
memory_handling.cu:    cudaVerify(cudaFree(p_device.tensile_strength));
memory_handling.cu:    cudaVerify(cudaFree(p_device.shear_strength));
memory_handling.cu:    cudaVerify(cudaFree(p_device.K));
memory_handling.cu:    cudaVerify(cudaFree(p_device.rho_0prime));
memory_handling.cu:    cudaVerify(cudaFree(p_device.rho_c_plus));
memory_handling.cu:    cudaVerify(cudaFree(p_device.rho_c_minus));
memory_handling.cu:    cudaVerify(cudaFree(p_device.flag_rho_0prime));
memory_handling.cu:    cudaVerify(cudaFree(p_device.flag_plastic));
memory_handling.cu:    cudaVerify(cudaFree(p_device.alpha_epspor));
memory_handling.cu:    cudaVerify(cudaFree(p_device.dalpha_epspordt));
memory_handling.cu:    cudaVerify(cudaFree(p_device.epsilon_v));
memory_handling.cu:    cudaVerify(cudaFree(p_device.depsilon_vdt));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.x));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.vx));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.ax));
memory_handling.cu:    cudaVerify(cudaFreeHost(p_host.g_ax));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.y));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.vy));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.ay));
memory_handling.cu:    cudaVerify(cudaFreeHost(p_host.g_ay));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.m));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.h));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.rho));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.p));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.e));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.cs));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.noi));
memory_handling.cu:	cudaVerify(cudaFreeHost(interactions_host));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.depth));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.materialId));
memory_handling.cu:	cudaVerify(cudaFreeHost(childList_host));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.p_min));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.p_max));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.rho_min));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.rho_max));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.e_min));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.e_max));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.cs_min));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.cs_max));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.beta));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.beta_old));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.divv_old));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.alpha_jutzi));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.alpha_jutzi_old));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.dalphadt));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.pold));
memory_handling.cu:    cudaVerify(cudaFreeHost(p_host.damage_porjutzi));
memory_handling.cu:    cudaVerify(cudaFreeHost(p_host.ddamage_porjutzidt));
memory_handling.cu:    cudaVerify(cudaFreeHost(p_host.compressive_strength));
memory_handling.cu:    cudaVerify(cudaFreeHost(p_host.tensile_strength));
memory_handling.cu:    cudaVerify(cudaFreeHost(p_host.shear_strength));
memory_handling.cu:    cudaVerify(cudaFreeHost(p_host.rho_0prime));
memory_handling.cu:    cudaVerify(cudaFreeHost(p_host.rho_c_plus));
memory_handling.cu:    cudaVerify(cudaFreeHost(p_host.rho_c_minus));
memory_handling.cu:    cudaVerify(cudaFreeHost(p_host.K));
memory_handling.cu:    cudaVerify(cudaFreeHost(p_host.flag_rho_0prime));
memory_handling.cu:    cudaVerify(cudaFreeHost(p_host.flag_plastic));
memory_handling.cu:    cudaVerify(cudaFreeHost(p_host.alpha_epspor));
memory_handling.cu:    cudaVerify(cudaFreeHost(p_host.epsilon_v));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.T));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.z));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.vz));
memory_handling.cu:	cudaVerify(cudaFreeHost(p_host.az));
memory_handling.cu:    cudaVerify(cudaFreeHost(p_host.g_az));
memory_handling.h: * This file is part of miluphcuda.
memory_handling.h: * miluphcuda is free software: you can redistribute it and/or modify
memory_handling.h: * miluphcuda is distributed in the hope that it will be useful,
memory_handling.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
doc/miluphcuda_documentation.tex:\title{ {\Large \textbf{Documentation miluphCUDA}
doc/miluphcuda_documentation.tex:\fancyhead[LO]{\crule[unirot]{0.7em}{0.5em}~~~ miluphCUDA}
doc/miluphcuda_documentation.tex:\fancyhead[LE]{\crule[blue]{0.7em}{0.5em}~~~ miluphCUDA}
doc/miluphcuda_documentation.tex: pdftitle={Documentation miluphCUDA},    % title
doc/miluphcuda_documentation.tex:\emph{miluphCUDA} was originally the CUDA port of the code \emph{miluph} (see \url{http://www.tat.physik.uni-tuebingen.de/~schaefer/miluph.html}).
doc/miluphcuda_documentation.tex:\emph{miluphCUDA} is capable of using various different equation of state (see Sect.~\ref{section:eos}) and the material strength model and a damage model introduced by Benz \& Asphaug for the simulation of brittle solid materials (see Sect.~\ref{section:damage_model}). Self-gravity can be included and is solved using a Barnes-Hut tree (see Sect.~\ref{section:self-gravity}). The code can be used for the simulation of high-velocity impacts and/or self gravitating astrophysical objects or mixed hydro-solid simulations.
doc/miluphcuda_documentation.tex:Christoph Schäfer and Sven Riecker wrote the basic version of \emph{miluphCUDA}.
doc/miluphcuda_documentation.tex: \item Install CUDA, e.g., from \url{https://developer.nvidia.com/cuda-downloads}
doc/miluphcuda_documentation.tex: \item Check out \emph{miluphCUDA} from the git repo, e.g.,\\ \verb|git clone  https://github.com/christophmschaefer/miluphcuda.git  |\\
doc/miluphcuda_documentation.tex: \item Make relevant changes to the \emph{Makefile}: Change the \verb|CUDA_DIR| variable according to your CUDA version. Make sure to use the correct \verb|GPU_ARCH| corresponding to your Nvidia-GPU model. Otherwise the code will \emph{not} work and crash or give wrong results!
doc/miluphcuda_documentation.tex:       Tested CUDA versions with Nvidia-driver versions are given in Tab.~\ref{tab:cuda_driver}.
doc/miluphcuda_documentation.tex:   CUDA version & Nvidia driver                   \\ \midrule
doc/miluphcuda_documentation.tex:  \caption{Tested CUDA versions with Nvidia-driver versions}
doc/miluphcuda_documentation.tex: Unsupported cuda versions so far: $<= 5.0$. \newline
doc/miluphcuda_documentation.tex: We have one case of an user reporting that the combination of gcc- 4.4.7 and CUDA devkit 7.5 on K20 cards fails to run the SOLID version of the code. If you happen to provide access to one of these cards for us, please contact us.
doc/miluphcuda_documentation.tex: \label{tab:cuda_driver}
doc/miluphcuda_documentation.tex:The number of threads for some CUDA kernels can be set as compile time options in \emph{timeintegration.h}. You may need to change (if so then probably lower) some values depending on your hardware.
doc/miluphcuda_documentation.tex:\verb|./miluphcuda --help|)
doc/miluphcuda_documentation.tex:  \verb|-G, --information| & print information about detected Nvidia GPUs on this host                                                                                                                                                \\
doc/miluphcuda_documentation.tex:  \verb|-d, --device_id <int>| & try to use GPU device with id \verb|<int>| for computation (default: 0)                                                                                                                       \\
doc/miluphcuda_documentation.tex:\verb|./miluphcuda --format|\\
doc/miluphcuda_documentation.tex:Example of miluphCUDA invocation (from the colliding rubber rings example in
doc/miluphcuda_documentation.tex:# set the correct LD_LIBRARY_PATH to the cuda libs
doc/miluphcuda_documentation.tex:export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64
doc/miluphcuda_documentation.tex:# invoke miluphCUDA: use the rk2_adaptive integrator, use HDF5
doc/miluphcuda_documentation.tex:# the used miluphCUDA command line is
doc/miluphcuda_documentation.tex:./miluphcuda -v -I rk2_adaptive -n 500 -H -t 1.0 -f rings.0000 -m material.cfg
doc/miluphcuda_documentation.tex:Table~\ref{tab:file-list} contains a list of the files that are part of the \emph{miluphCUDA} project.
doc/miluphcuda_documentation.tex: \caption{List of files in \emph{miluphCUDA}.}
doc/miluphcuda_documentation.tex:  cuda\_utils.h                                                 & CUDA specific functions                                                                                                                                                                                                                                                                                                                                    \\
doc/miluphcuda_documentation.tex:  Makefile                                                      & Makefile for GNU Make. You'll need to make changes according to your CUDA version and your GPU architecture in this file                                                                                                                                                                                                                                   \\
doc/miluphcuda_documentation.tex:We have implemented the efficient routines for Barnes-Hut tree builds and calculation of the gravitational forces for GPUs introduced by \cite{burtscher:2011}. The algorithm
doc/miluphcuda_documentation.tex:CUDA by \cite{burtscher:2011}. The hierarchical decomposition is recorded in an octree. Since CUDA does not allow the
doc/miluphcuda_documentation.tex:distributed among the threads on the GPU.  Now $N_\mathrm{threads}$ threads on the GPU start to compare the locations of their
doc/miluphcuda_documentation.tex:status $-2$ or "locked" to the {\tt childList} and use the CUDA\-function "atomic compare and save" {\tt atomicCAS} to
doc/miluphcuda_documentation.tex:$N_\mathrm{threads}$ a CUDA kernel that picks the last $N_\mathrm{threads}$ added inner nodes, for which we
doc/miluphcuda_documentation.tex:implementation of Burtscher is especially designed for CUDA, regarding memory access and parallelization.
doc/miluphcuda_documentation.tex:which kernel is implemented is provided by the standard \texttt{miluphCUDA} help option \texttt{-h/-{}-help}.
doc/miluphcuda_documentation.tex:the original definition of the smoothing length, the interaction radius in \texttt{miluphCUDA} is
doc/miluphcuda_documentation.tex:\bibliography{miluphcuda_documentation}
doc/Makefile:	pdflatex miluphcuda_documentation
doc/Makefile:	bibtex miluphcuda_documentation
doc/Makefile:	pdflatex miluphcuda_documentation
doc/Makefile:	pdflatex miluphcuda_documentation
doc/make_doc:	echo "This script lets you generate the documentation for miluphcuda!"
doc/make_doc:	echo "HTML_EXTRA_FILES = $BASHDIR/miluphcuda_documentation.pdf" >> $DOXYDIR/doxyfile.inc
doc/doxygen.log:warning: Tag 'SYMBOL_CACHE_SIZE' at line 327 of file '/Users/Michi/Desktop/Miluph/miluphcuda/doc/DoxygenFiles/Doxyfile' has become obsolete.
doc/doxygen.log:warning: Tag 'SHOW_DIRECTORIES' at line 536 of file '/Users/Michi/Desktop/Miluph/miluphcuda/doc/DoxygenFiles/Doxyfile' has become obsolete.
doc/doxygen.log:warning: Tag 'HTML_ALIGN_MEMBERS' at line 930 of file '/Users/Michi/Desktop/Miluph/miluphcuda/doc/DoxygenFiles/Doxyfile' has become obsolete.
doc/doxygen.log:warning: Tag 'USE_INLINE_TREES' at line 1117 of file '/Users/Michi/Desktop/Miluph/miluphcuda/doc/DoxygenFiles/Doxyfile' has become obsolete.
doc/doxygen.log:warning: Tag 'XML_SCHEMA' at line 1380 of file '/Users/Michi/Desktop/Miluph/miluphcuda/doc/DoxygenFiles/Doxyfile' has become obsolete.
doc/doxygen.log:warning: Tag 'XML_DTD' at line 1386 of file '/Users/Michi/Desktop/Miluph/miluphcuda/doc/DoxygenFiles/Doxyfile' has become obsolete.
doc/doxygen.log:warning: Tag 'PERL_PATH' at line 1551 of file '/Users/Michi/Desktop/Miluph/miluphcuda/doc/DoxygenFiles/Doxyfile' has become obsolete.
doc/doxygen.log:warning: Tag 'MSCGEN_PATH' at line 1572 of file '/Users/Michi/Desktop/Miluph/miluphcuda/doc/DoxygenFiles/Doxyfile' has become obsolete.
doc/doxygen.log:warning: ignoring unknown tag 'Borderliner' at line 1767, file /Users/Michi/Desktop/Miluph/miluphcuda/doc/DoxygenFiles/Doxyfile
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/artificial_stress.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/boundary.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/checks.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/cuda_utils.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/damage.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/density.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/device_tools.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/euler.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/gravity.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/internal_forces.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/io.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:Searching for files in directory /Users/Michi/Desktop/Miluph/miluphcuda
doc/doxygen.log:Searching for files in directory /Users/Michi/Desktop/Miluph/miluphcuda/doc
doc/doxygen.log:Searching for files in directory /Users/Michi/Desktop/Miluph/miluphcuda/doc/DoxygenFiles
doc/doxygen.log:Searching for files in directory /Users/Michi/Desktop/Miluph/miluphcuda/doc/pic
doc/doxygen.log:Searching for files in directory /Users/Michi/Desktop/Miluph/miluphcuda/doc/test_dir
doc/doxygen.log:Searching for files in directory /Users/Michi/Desktop/Miluph/miluphcuda/docs
doc/doxygen.log:Searching for files in directory /Users/Michi/Desktop/Miluph/miluphcuda/material_data
doc/doxygen.log:Searching for files in directory /Users/Michi/Desktop/Miluph/miluphcuda/test_cases
doc/doxygen.log:Searching for files in directory /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/colliding_rings
doc/doxygen.log:Searching for files in directory /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/giant_collision_solid
doc/doxygen.log:Searching for files in directory /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/nakamura
doc/doxygen.log:Searching for files in directory /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/nakamura/input
doc/doxygen.log:Searching for files in directory /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/shocktube
doc/doxygen.log:Searching for files in directory /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/viscously_spreading_ring
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/aneos.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/artificial_stress.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/artificial_stress.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/artificial_stress.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/boundary.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/boundary.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/boundary.h...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/checks.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/checks.h...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/cuda_utils.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/cuda_utils.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/damage.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/damage.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/damage.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/density.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/density.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/density.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/device_tools.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/device_tools.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/device_tools.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/doc/ExtraMarkdown.md...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/doc/Mainpage.md...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/euler.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/euler.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/euler.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/gravity.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/gravity.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/gravity.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/internal_forces.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/internal_forces.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/internal_forces.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/io.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/io.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/io.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/kernel.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/kernel.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda//Users/Michi/Desktop/Miluph/miluphcuda/kernel.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/linalg.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/little_helpers.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/memory_handling.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/miluph.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/parameter.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/plasticity.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/porosity.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/predictor_corrector.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/predictor_corrector_euler.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/pressure.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/rhs.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/rk2adaptive.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/soundspeed.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/stress.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/test_cases/colliding_rings/parameter.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/test_cases/giant_collision_solid/parameter.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/test_cases/nakamura/parameter.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/test_cases/shocktube/parameter.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/test_cases/viscously_spreading_ring/parameter.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/linalg.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/linalg.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/linalg.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/little_helpers.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/little_helpers.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/little_helpers.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/memory_handling.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/memory_handling.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/memory_handling.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/miluph.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/miluph.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/miluph.h...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/parameter.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/parameter.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/plasticity.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/plasticity.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/plasticity.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/porosity.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/porosity.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/porosity.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/predictor_corrector.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/predictor_corrector.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/predictor_corrector.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/predictor_corrector_euler.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/predictor_corrector_euler.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/predictor_corrector_euler.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/pressure.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/pressure.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/pressure.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/README.md...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/rhs.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/rhs.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/rhs.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/rk2adaptive.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/rk2adaptive.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/rk2adaptive.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/soundspeed.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/soundspeed.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/soundspeed.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/stress.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/stress.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/stress.h...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/colliding_rings/parameter.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/colliding_rings/parameter.h...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/giant_collision_solid/parameter.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/giant_collision_solid/parameter.h...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/nakamura/parameter.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/nakamura/parameter.h...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/shocktube/parameter.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/shocktube/parameter.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/viscously_spreading_ring/boundary.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/viscously_spreading_ring/parameter.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/viscously_spreading_ring/parameter.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/timeintegration.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/timeintegration.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/t/Users/Michi/Desktop/Miluph/miluphcuda/timeintegration.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/tree.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/velocity.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/viscosity.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:/Users/Michi/Desktop/Miluph/miluphcuda/xsph.h:4: warning: multiple use of section label 'LICENSE' while adding section, (first occurrence: /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h, line 4)
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/tree.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/tree.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/tree.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/velocity.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/velocity.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/velocity.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/viscosity.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/viscosity.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/viscosity.h...
doc/doxygen.log:Reading /Users/Michi/Desktop/Miluph/miluphcuda/xsph.cu...
doc/doxygen.log:Preprocessing /Users/Michi/Desktop/Miluph/miluphcuda/xsph.h...
doc/doxygen.log:Parsing file /Users/Michi/Desktop/Miluph/miluphcuda/xsph.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/aneos.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/artificial_stress.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/artificial_stress.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/boundary.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/viscously_spreading_ring/boundary.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/boundary.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/checks.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/cuda_utils.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/damage.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/damage.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/density.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/density.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/device_tools.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/device_tools.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/euler.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/euler.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/gravity.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/gravity.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/internal_forces.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/internal_forces.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/io.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/io.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/kernel.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/kernel.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/linalg.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/linalg.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/little_helpers.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/little_helpers.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/memory_handling.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/memory_handling.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/miluph.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/miluph.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/parameter.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/colliding_rings/parameter.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/giant_collision_solid/parameter.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/nakamura/parameter.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/shocktube/parameter.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/viscously_spreading_ring/parameter.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/plasticity.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/plasticity.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/porosity.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/porosity.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/predictor_corrector.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/predictor_corrector.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/predictor_corrector_euler.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/predictor_corrector_euler.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/pressure.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/pressure.h...
doc/doxygen.log:Parsing code for file /Users/Michi/Desktop/Miluph/miluphcuda/README.md...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/rhs.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/rhs.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/rk2adaptive.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/rk2adaptive.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/soundspeed.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/soundspeed.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/stress.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/stress.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/timeintegration.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/timeintegration.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/tree.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/tree.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/velocity.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/velocity.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/viscosity.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/viscosity.h...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/xsph.cu...
doc/doxygen.log:Generating code for file /Users/Michi/Desktop/Miluph/miluphcuda/xsph.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/aneos.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/aneos.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/artificial_stress.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/artificial_stress.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/boundary.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/viscously_spreading_ring/boundary.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/boundary.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/checks.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/cuda_utils.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/damage.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/damage.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/density.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/density.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/device_tools.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/device_tools.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/euler.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/euler.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/gravity.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/gravity.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/internal_forces.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/internal_forces.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/io.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/io.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/kernel.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/kernel.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/linalg.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/linalg.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/little_helpers.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/little_helpers.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/memory_handling.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/memory_handling.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/miluph.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/miluph.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/parameter.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/colliding_rings/parameter.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/giant_collision_solid/parameter.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/nakamura/parameter.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/shocktube/parameter.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/test_cases/viscously_spreading_ring/parameter.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/plasticity.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/plasticity.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/porosity.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/porosity.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/predictor_corrector.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/predictor_corrector.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/predictor_corrector_euler.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/predictor_corrector_euler.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/pressure.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/pressure.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/README.md...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/rhs.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/rhs.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/rk2adaptive.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/rk2adaptive.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/soundspeed.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/soundspeed.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/stress.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/stress.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/timeintegration.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/timeintegration.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/tree.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/tree.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/velocity.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/velocity.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/viscosity.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/viscosity.h...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/xsph.cu...
doc/doxygen.log:Generating docs for file /Users/Michi/Desktop/Miluph/miluphcuda/xsph.h...
doc/doxygen.log:Generating docs for page md__Users_Michi_Desktop_Miluph_miluphcuda_README...
doc/miluphcuda_documentation.bib:    title = {GPU Computing Gems Emerald Edition},
doc/miluphcuda_documentation.bib:    title = "{A Step towards Energy Efficient Computing: Redesigning a Hydrodynamic Application on CPU-GPU}",
doc/miluphcuda_documentation.bib:    title = "{GPUPEGAS: A New GPU-accelerated Hydrodynamic Code for Numerical Simulations of Interacting Galaxies}",
doc/miluphcuda_documentation.bib:    title = "{A sparse octree gravitational N-body code that runs entirely on the GPU processor}",
doc/miluphcuda_documentation.bib:  title		= {GPU programming applied to Smoothed particle Hydrodynamics simulations of planetesimal collisions},
doc/Mainpage.md:# miluphcuda
doc/Mainpage.md:miluphcuda is a smoothed particle hydrodynamics (**SPH**)
doc/Mainpage.md:* [GitHub repository](https://github.com/christophmschaefer/miluphcuda)
doc/Mainpage.md:* [miluphcuda documentation (pdf)](miluphcuda_documentation.pdf) mostly outdated :frown:
doc/Mainpage.md:* **test_cases**: test cases for miluphcuda
doc/Mainpage.md:* miluphcuda is the cuda port of the original miluph code.
doc/Mainpage.md:* miluphcuda can be used to model fluids and solids.
doc/Mainpage.md:* miluphcuda runs on a single Nvidia GPU with compute capability 5.0 and higher.
doc/DoxygenFiles/Doxyfile:PROJECT_NAME           = "Miluphcuda"
doc/DoxygenFiles/Doxyfile:# HTML_EXTRA_FILES       = doc/miluphcuda_documentation.pdf
configure.sh:# check CUDA path
configure.sh:if [ ! -d /usr/local/cuda ]; then
configure.sh:    echo "Warning: /usr/local/cuda not found, set CUDA path manually in Makefile."
configure.sh:GPU_ARCH_FOUND=0
configure.sh:GPU=`nvidia-smi -q -i 0 2>/dev/null | grep "Product Name" | cut -d: -f 2 | cut -d\  -f 2-`
configure.sh:if [ -z "$GPU" ]; then
configure.sh:    echo "Warning: couldn't extract GPU model, set compute capability (GPU_ARCH) manually in Makefile."
configure.sh:    case $GPU in
configure.sh:            GPU_ARCH="-arch=sm_20"
configure.sh:            GPU_ARCH_FOUND=1 ;;
configure.sh:            GPU_ARCH="-arch=sm_30"
configure.sh:            GPU_ARCH_FOUND=1 ;;
configure.sh:            GPU_ARCH="-arch=sm_35"
configure.sh:            GPU_ARCH_FOUND=1 ;;
configure.sh:            GPU_ARCH="-arch=sm_37"
configure.sh:            GPU_ARCH_FOUND=1 ;;
configure.sh:            GPU_ARCH="-arch=sm_50"
configure.sh:            GPU_ARCH_FOUND=1 ;;
configure.sh:            GPU_ARCH="-arch=sm_52"
configure.sh:            GPU_ARCH_FOUND=1 ;;
configure.sh:            GPU_ARCH="-arch=sm_61"
configure.sh:            GPU_ARCH_FOUND=1 ;;
configure.sh:            echo "Warning: didn't find GPU model '$GPU' in lookup list, set compute capability (GPU_ARCH) manually in Makefile, and/or drop the developers a line to add it..."
configure.sh:# set compute capability (GPU_ARCH) in Makefile
configure.sh:if [ $GPU_ARCH_FOUND -eq 1 ]; then
configure.sh:    grep "^GPU_ARCH" $MAKEFILE >/dev/null
configure.sh:        echo "Error: couldn't find setting for compute capability (GPU_ARCH) in Makefile."
configure.sh:        sed -i "/^GPU_ARCH*/c\GPU_ARCH = $GPU_ARCH" $MAKEFILE
configure.sh:        echo "Found GPU model '$GPU' and set compute capability (GPU_ARCH) to '$GPU_ARCH'."
configure.sh:# warning if more than one GPU
configure.sh:NO_GPUS=`nvidia-smi -q 2>/dev/null | grep "Attached GPUs" | cut -d: -f 2 | cut -d\  -f 2`
configure.sh:if [ $GPU_ARCH_FOUND -eq 1 ] && [ $NO_GPUS -gt 1 ]; then
configure.sh:    echo "Warning: more than one GPU on host (found $NO_GPUS), used device with ID 0 to set compute capability."
device_tools.cu: * This file is part of miluphcuda.
device_tools.cu: * miluphcuda is free software: you can redistribute it and/or modify
device_tools.cu: * miluphcuda is distributed in the hope that it will be useful,
device_tools.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
device_tools.cu:        mainly taken from cuda samples
device_tools.cu: *  Helper function to calculate the number of CUDA core.
device_tools.cu: *  Taken from cuda_samples/common/inc/helper_cuda.h
device_tools.cu:    /* Defines for GPU Architecture types (using the SM version to determine the # of cores per SM */
device_tools.cu:    sSMtoCores nGpuArchCoresPerSM[] =
device_tools.cu:    while (nGpuArchCoresPerSM[index].SM != -1) {
device_tools.cu:        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
device_tools.cu:            return nGpuArchCoresPerSM[index].Cores;
device_tools.cu:            major, minor, nGpuArchCoresPerSM[index-1].Cores);
device_tools.cu:    return nGpuArchCoresPerSM[index-1].Cores;
device_tools.cu: *  printfs some basic information about detected CUDA devices. 
device_tools.cu: *  Taken from cuda samples/1_Utilities/deviceQuery
device_tools.cu:    struct cudaDeviceProp prop;
device_tools.cu:    cudaGetDeviceCount(&device_count);
device_tools.cu:        printf("\nNo device(s) that support CUDA found!\n");
device_tools.cu:      //  cudaSetDevice(i);
device_tools.cu:        cudaGetDeviceProperties(&prop, i);
device_tools.cu:        cudaDriverGetVersion(&driverVersion);
device_tools.cu:        cudaRuntimeGetVersion(&runtimeVersion);
device_tools.cu:        printf("  CUDA Driver Version:                           %d.%d\n", driverVersion/1000, (driverVersion%100)/10);
device_tools.cu:        printf("  CUDA Runtime Version:                          %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);
device_tools.cu:        printf("  CUDA Cores / Multiprocessor:                   %d\n", _ConvertSMVer2Cores(prop.major, prop.minor));
device_tools.cu:        printf("  Total amount of CUDA Cores:                    %d\n", _ConvertSMVer2Cores(prop.major, prop.minor)*prop.multiProcessorCount);
device_tools.cu:        printf("  GPU clock rate:                                %0.f MHz\n\n", prop.clockRate * 1e-3f);
device_tools.cu:#if CUDART_VERSION >= 5000
device_tools.cu:        /* This is supported in CUDA 5.0 (runtime API device properties) */
device_tools.cu:            cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
device_tools.cu:        printf("  Integrated GPU sharing Host Memory:            %s\n", prop.integrated ? "Yes" : "No");
pressure.cu: * This file is part of miluphcuda.
pressure.cu: * miluphcuda is free software: you can redistribute it and/or modify
pressure.cu: * miluphcuda is distributed in the hope that it will be useful,
pressure.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
rk4_pointmass.h: * This file is part of miluphcuda.
rk4_pointmass.h: * miluphcuda is free software: you can redistribute it and/or modify
rk4_pointmass.h: * miluphcuda is distributed in the hope that it will be useful,
rk4_pointmass.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
README.md:# miluphcuda
README.md:miluphcuda is a 3D Smoothed Particle Hydrodynamics (SPH) code, mainly developed for modeling astrophysical collision and
README.md:* miluphcuda is the CUDA port of the original miluph code.
README.md:* miluphcuda runs on single Nvidia GPUs with compute capability 5.0 and higher.
README.md:[documentation](https://christophmschaefer.github.io/miluphcuda/index.html).
README.md:2. run `make` to produce the `miluphcuda` executable
README.md:**CUDA**  
README.md:Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) to compile and run code on the GPU.
README.md:The basic usage philosophy for miluphcuda is:
README.md:* All other main options are controlled via cmd-line flags. Check `./miluphcuda -h` once compiled.
README.md:  check `./miluphcuda --format` for the required format
README.md:* suitable cmd-line options for miluphcuda, check `./miluphcuda -h`
predictor_corrector_euler.cu: * This file is part of miluphcuda.
predictor_corrector_euler.cu: * miluphcuda is free software: you can redistribute it and/or modify
predictor_corrector_euler.cu: * miluphcuda is distributed in the hope that it will be useful,
predictor_corrector_euler.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
predictor_corrector_euler.cu:    cudaVerify(cudaMalloc((void**)&courantPerBlock, sizeof(double)*numberOfMultiprocessors));
predictor_corrector_euler.cu:    cudaVerify(cudaMalloc((void**)&forcesPerBlock, sizeof(double)*numberOfMultiprocessors));
predictor_corrector_euler.cu:    cudaVerify(cudaMalloc((void**)&dtSPerBlock, sizeof(double)*numberOfMultiprocessors));
predictor_corrector_euler.cu:    cudaVerify(cudaMalloc((void**)&dtePerBlock, sizeof(double)*numberOfMultiprocessors));
predictor_corrector_euler.cu:    cudaVerify(cudaMalloc((void**)&dtrhoPerBlock, sizeof(double)*numberOfMultiprocessors));
predictor_corrector_euler.cu:    cudaVerify(cudaMalloc((void**)&dtdamagePerBlock, sizeof(double)*numberOfMultiprocessors));
predictor_corrector_euler.cu:    cudaVerify(cudaMalloc((void**)&dtalphaPerBlock, sizeof(double)*numberOfMultiprocessors));
predictor_corrector_euler.cu:    cudaVerify(cudaMalloc((void**)&dtbetaPerBlock, sizeof(double)*numberOfMultiprocessors));
predictor_corrector_euler.cu:    cudaVerify(cudaMalloc((void**)&maxpressureDiffPerBlock, sizeof(double)*numberOfMultiprocessors));
predictor_corrector_euler.cu:    cudaVerify(cudaMalloc((void**)&dtartviscPerBlock, sizeof(double)*numberOfMultiprocessors));
predictor_corrector_euler.cu:    cudaVerify(cudaMalloc((void**)&dtalpha_epsporPerBlock, sizeof(double)*numberOfMultiprocessors));
predictor_corrector_euler.cu:    cudaVerify(cudaMalloc((void**)&dtepsilon_vPerBlock, sizeof(double)*numberOfMultiprocessors));
predictor_corrector_euler.cu:    /* tell the gpu the current time */
predictor_corrector_euler.cu:    cudaVerify(cudaMemcpyToSymbol(currentTimeD, &currentTime, sizeof(double)));
predictor_corrector_euler.cu:    cudaVerify(cudaMemcpyToSymbol(predictor, &predictor_device, sizeof(struct Particle)));
predictor_corrector_euler.cu:    cudaVerify(cudaMemcpyToSymbol(predictor_pointmass, &predictor_pointmass_device, sizeof(struct Pointmass)));
predictor_corrector_euler.cu:        /* tell the gpu the time step */
predictor_corrector_euler.cu:            cudaVerify(cudaMemcpyToSymbol(dt, &param.maxtimestep, sizeof(double)));
predictor_corrector_euler.cu:            cudaVerify(cudaMemcpyToSymbol(dt, &timePerStep, sizeof(double)));
predictor_corrector_euler.cu:        /* tell the gpu the end time */
predictor_corrector_euler.cu:        cudaVerify(cudaMemcpyToSymbol(endTimeD, &endTime, sizeof(double)));
predictor_corrector_euler.cu:			cudaVerify(cudaDeviceSynchronize());
predictor_corrector_euler.cu:	        cudaVerify(cudaMemcpyToSymbol(p, &p_device, sizeof(struct Particle)));
predictor_corrector_euler.cu:	        cudaVerify(cudaMemcpyToSymbol(pointmass, &pointmass_device, sizeof(struct Pointmass)));
predictor_corrector_euler.cu:            cudaVerify(cudaDeviceSynchronize());
predictor_corrector_euler.cu:            cudaVerify(cudaMemcpyFromSymbol(&currentTime, currentTimeD, sizeof(double)));
predictor_corrector_euler.cu:            cudaVerify(cudaMemcpyToSymbol(substep_currentTimeD, &substep_currentTime, sizeof(double)));
predictor_corrector_euler.cu:            cudaVerify(cudaDeviceSynchronize());
predictor_corrector_euler.cu:            cudaVerifyKernel((setTimestep_euler<<<numberOfMultiprocessors, NUM_THREADS_LIMITTIMESTEP>>>(
predictor_corrector_euler.cu:            cudaVerify(cudaDeviceSynchronize());
predictor_corrector_euler.cu:            /* get the time and the time step from the gpu */
predictor_corrector_euler.cu:            cudaVerify(cudaMemcpyFromSymbol(&dt_host, dt, sizeof(double)));
predictor_corrector_euler.cu:                cudaVerify(cudaMemcpyToSymbol(dt, &dt_host, sizeof(double)));
predictor_corrector_euler.cu:			cudaVerify(cudaDeviceSynchronize());
predictor_corrector_euler.cu:	            cudaVerify(cudaMemcpyToSymbol(p, &p_device, sizeof(struct Particle)));
predictor_corrector_euler.cu:	            cudaVerify(cudaMemcpyToSymbol(pointmass, &pointmass_device, sizeof(struct Pointmass)));
predictor_corrector_euler.cu:    	        cudaVerifyKernel((PredictorStep_euler<<<numberOfMultiprocessors, NUM_THREADS_PC_INTEGRATOR>>>()));
predictor_corrector_euler.cu:			    cudaVerify(cudaDeviceSynchronize());
predictor_corrector_euler.cu:		        cudaVerify(cudaMemcpyToSymbol(p, &predictor_device, sizeof(struct Particle)));
predictor_corrector_euler.cu:	            cudaVerify(cudaMemcpyToSymbol(pointmass, &predictor_pointmass_device, sizeof(struct Pointmass)));
predictor_corrector_euler.cu:				cudaVerifyKernel((calculatePressure<<<numberOfMultiprocessors * 4, NUM_THREADS_PRESSURE>>>()));
predictor_corrector_euler.cu:    			cudaVerify(cudaDeviceSynchronize());
predictor_corrector_euler.cu:			    cudaVerify(cudaMemcpyFromSymbol(&dt_host, dt, sizeof(double)));
predictor_corrector_euler.cu:				cudaVerifyKernel((pressureChangeCheck_euler<<<numberOfMultiprocessors, NUM_THREADS_PC_INTEGRATOR>>>(maxpressureDiffPerBlock)));
predictor_corrector_euler.cu:    			cudaVerify(cudaDeviceSynchronize());
predictor_corrector_euler.cu:                cudaVerify(cudaMemcpyFromSymbol(&pressureChangeSmallEnough_host, pressureChangeSmallEnough, sizeof(int)));
predictor_corrector_euler.cu:                cudaVerify(cudaMemcpyFromSymbol(&maxpressureDiff_host, maxpressureDiff, sizeof(double)));
predictor_corrector_euler.cu:			    cudaVerify(cudaMemcpyFromSymbol(&dt_host, dt, sizeof(double)));
predictor_corrector_euler.cu:					cudaVerify(cudaMemcpyToSymbol(currentTimeD, &currentTime, sizeof(double)));
predictor_corrector_euler.cu:		            cudaVerify(cudaMemcpyToSymbol(p, &predictor_device, sizeof(struct Particle)));
predictor_corrector_euler.cu:	                cudaVerify(cudaMemcpyToSymbol(pointmass, &predictor_pointmass_device, sizeof(struct Pointmass)));
predictor_corrector_euler.cu:		            cudaVerify(cudaMemcpyToSymbol(p, &p_device, sizeof(struct Particle)));
predictor_corrector_euler.cu:	                cudaVerify(cudaMemcpyToSymbol(pointmass, &pointmass_device, sizeof(struct Pointmass)));
predictor_corrector_euler.cu:    	            cudaVerifyKernel((CorrectorStep_euler<<<numberOfMultiprocessors, NUM_THREADS_PC_INTEGRATOR>>>()));
predictor_corrector_euler.cu:        cudaVerify(cudaDeviceSynchronize());
predictor_corrector_euler.cu:        cudaVerifyKernel((damageLimit<<<numberOfMultiprocessors*4, NUM_THREADS_PC_INTEGRATOR>>>()));
predictor_corrector_euler.cu:        cudaVerify(cudaDeviceSynchronize());
predictor_corrector_euler.cu:	cudaVerify(cudaFree(courantPerBlock));
predictor_corrector_euler.cu:	cudaVerify(cudaFree(forcesPerBlock));
predictor_corrector_euler.cu:    cudaVerify(cudaFree(dtSPerBlock));
predictor_corrector_euler.cu:	cudaVerify(cudaFree(dtePerBlock));
predictor_corrector_euler.cu:	cudaVerify(cudaFree(dtrhoPerBlock));
predictor_corrector_euler.cu:	cudaVerify(cudaFree(dtdamagePerBlock));
predictor_corrector_euler.cu:    cudaVerify(cudaFree(dtalphaPerBlock));
predictor_corrector_euler.cu:    cudaVerify(cudaFree(dtbetaPerBlock));
predictor_corrector_euler.cu:    cudaVerify(cudaFree(dtalpha_epsporPerBlock));
predictor_corrector_euler.cu:    cudaVerify(cudaFree(dtepsilon_vPerBlock));
predictor_corrector_euler.cu:    cudaVerify(cudaFree(dtartviscPerBlock));
predictor_corrector_euler.cu:    cudaVerify(cudaFree(maxpressureDiffPerBlock));
velocity.h: * This file is part of miluphcuda.
velocity.h: * miluphcuda is free software: you can redistribute it and/or modify
velocity.h: * miluphcuda is distributed in the hope that it will be useful,
velocity.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
cuda_utils.h: * This file is part of miluphcuda.
cuda_utils.h: * miluphcuda is free software: you can redistribute it and/or modify
cuda_utils.h: * miluphcuda is distributed in the hope that it will be useful,
cuda_utils.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
cuda_utils.h:#ifndef om_cuda_utils_
cuda_utils.h:#define om_cuda_utils_
cuda_utils.h:#define cudaVerify(x) do {                                               \
cuda_utils.h:    cudaError_t __cu_result = x;                                         \
cuda_utils.h:    if (__cu_result!=cudaSuccess) {                                      \
cuda_utils.h:      fprintf(stderr,"%s:%i: error: cuda function call failed:\n"        \
cuda_utils.h:              __FILE__,__LINE__,#x,cudaGetErrorString(__cu_result));     \
cuda_utils.h:#define cudaVerifyKernel(x) do {                                         \
cuda_utils.h:    cudaError_t __cu_result = cudaGetLastError();                        \
cuda_utils.h:    if (__cu_result!=cudaSuccess) {                                      \
cuda_utils.h:      fprintf(stderr,"%s:%i: error: cuda function call failed:\n"        \
cuda_utils.h:              __FILE__,__LINE__,#x,cudaGetErrorString(__cu_result));     \
cuda_utils.h:#define cudaVerify(x) do {                                               \
cuda_utils.h:#define cudaVerifyKernel(x) do {                                         \
extrema.cu: * This file is part of miluphcuda.
extrema.cu: * miluphcuda is free software: you can redistribute it and/or modify
extrema.cu: * miluphcuda is distributed in the hope that it will be useful,
extrema.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
soundspeed.h: * This file is part of miluphcuda.
soundspeed.h: * miluphcuda is free software: you can redistribute it and/or modify
soundspeed.h: * miluphcuda is distributed in the hope that it will be useful,
soundspeed.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
gravity.cu: * This file is part of miluphcuda.
gravity.cu: * miluphcuda is free software: you can redistribute it and/or modify
gravity.cu: * miluphcuda is distributed in the hope that it will be useful,
gravity.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
gravity.cu:    cudaMalloc((void **) &g_x, h_blocksize*sizeof(double));
gravity.cu:    cudaMalloc((void **) &g_y, h_blocksize*sizeof(double));
gravity.cu:    cudaMalloc((void **) &g_z, h_blocksize*sizeof(double));
gravity.cu:        cudaVerifyKernel((particles_gravitational_feedback<<<h_blocksize, NUM_THREADS_REDUCTION>>>(n, g_x, g_y, g_z)));
gravity.cu:        cudaVerify(cudaDeviceSynchronize());
gravity.cu:    cudaFree(g_x);
gravity.cu:    cudaFree(g_y);
gravity.cu:    cudaFree(g_z);
coupled_heun_rk4_sph_nbody.h: * This file is part of miluphcuda.
coupled_heun_rk4_sph_nbody.h: * miluphcuda is free software: you can redistribute it and/or modify
coupled_heun_rk4_sph_nbody.h: * miluphcuda is distributed in the hope that it will be useful,
coupled_heun_rk4_sph_nbody.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
kernel.cu: * This file is part of miluphcuda.
kernel.cu: * miluphcuda is free software: you can redistribute it and/or modify
kernel.cu: * miluphcuda is distributed in the hope that it will be useful,
kernel.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
porosity.h: * This file is part of miluphcuda.
porosity.h: * miluphcuda is free software: you can redistribute it and/or modify
porosity.h: * miluphcuda is distributed in the hope that it will be useful,
porosity.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
CodingStyle.txt:Coding Style for miluphCUDA
CodingStyle.txt:That's too much. Really. Don't argue!  That's why we use an indentation level of 4 in the miluphCUDA code. If you use
io.h: * This file is part of miluphcuda.
io.h: * miluphcuda is free software: you can redistribute it and/or modify
io.h: * miluphcuda is distributed in the hope that it will be useful,
io.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
predictor_corrector.h: * This file is part of miluphcuda.
predictor_corrector.h: * miluphcuda is free software: you can redistribute it and/or modify
predictor_corrector.h: * miluphcuda is distributed in the hope that it will be useful,
predictor_corrector.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
euler.h: * This file is part of miluphcuda.
euler.h: * miluphcuda is free software: you can redistribute it and/or modify
euler.h: * miluphcuda is distributed in the hope that it will be useful,
euler.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
material-config/material_data/basalt.fragmentation.cfg:    # from Schäfer et al. (2016), derived by fitting miluphcuda outcomes to the results of the test case by Nakamura & Fujiwara (1991)
material-config/material_data/Regolith_simulant.cfg:# a validation and benchmark study including miluphcuda, iSALE and Bern SPH.
material-config/CREATE-MATERIAL-CONFIG.md:How to set up the material config file for miluphcuda
material-config/CREATE-MATERIAL-CONFIG.md:miluphcuda supports multiple materials, and multiple rheologies. Many important settings for
material-config/CREATE-MATERIAL-CONFIG.md:**Note**: This file is one of three places where you configure miluphcuda settings. The other two are:
material-config/CREATE-MATERIAL-CONFIG.md:The config file is passed to miluphcuda by the `-m` cmd-line option.
material-config/CREATE-MATERIAL-CONFIG.md:                                        https://christophmschaefer.github.io/miluphcuda/pressure_8h.html
material-config/CREATE-MATERIAL-CONFIG.md:                                                                    processed by miluphcuda
material-config/CREATE-MATERIAL-CONFIG.md:        fragmentation.weibull_k     float   none        not processed by miluphcuda
material-config/CREATE-MATERIAL-CONFIG.md:        fragmentation.weibull_m     float   none        not processed by miluphcuda
linalg.cu: * This file is part of miluphcuda.
linalg.cu: * miluphcuda is free software: you can redistribute it and/or modify
linalg.cu: * miluphcuda is distributed in the hope that it will be useful,
linalg.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
coupled_heun_rk4_sph_nbody.cu: * This file is part of miluphcuda.
coupled_heun_rk4_sph_nbody.cu: * miluphcuda is free software: you can redistribute it and/or modify
coupled_heun_rk4_sph_nbody.cu: * miluphcuda is distributed in the hope that it will be useful,
coupled_heun_rk4_sph_nbody.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
coupled_heun_rk4_sph_nbody.cu:    cudaVerify(cudaMalloc((void**)&courantPerBlock, sizeof(double)*numberOfMultiprocessors));
coupled_heun_rk4_sph_nbody.cu:    cudaVerify(cudaMalloc((void**)&forcesPerBlock, sizeof(double)*numberOfMultiprocessors));
coupled_heun_rk4_sph_nbody.cu:    cudaVerify(cudaMalloc((void**)&dtSPerBlock, sizeof(double)*numberOfMultiprocessors));
coupled_heun_rk4_sph_nbody.cu:    cudaVerify(cudaMalloc((void**)&dtePerBlock, sizeof(double)*numberOfMultiprocessors));
coupled_heun_rk4_sph_nbody.cu:    cudaVerify(cudaMalloc((void**)&dtrhoPerBlock, sizeof(double)*numberOfMultiprocessors));
coupled_heun_rk4_sph_nbody.cu:    cudaVerify(cudaMalloc((void**)&dtdamagePerBlock, sizeof(double)*numberOfMultiprocessors));
coupled_heun_rk4_sph_nbody.cu:    cudaVerify(cudaMalloc((void**)&dtalphaPerBlock, sizeof(double)*numberOfMultiprocessors));
coupled_heun_rk4_sph_nbody.cu:    cudaVerify(cudaMalloc((void**)&dtbetaPerBlock, sizeof(double)*numberOfMultiprocessors));
coupled_heun_rk4_sph_nbody.cu:    cudaVerify(cudaMalloc((void**)&maxpressureDiffPerBlock, sizeof(double)*numberOfMultiprocessors));
coupled_heun_rk4_sph_nbody.cu:    cudaVerify(cudaMalloc((void**)&dtartviscPerBlock, sizeof(double)*numberOfMultiprocessors));
coupled_heun_rk4_sph_nbody.cu:    cudaVerify(cudaMalloc((void**)&dtalpha_epsporPerBlock, sizeof(double)*numberOfMultiprocessors));
coupled_heun_rk4_sph_nbody.cu:    cudaVerify(cudaMalloc((void**)&dtepsilon_vPerBlock, sizeof(double)*numberOfMultiprocessors));
coupled_heun_rk4_sph_nbody.cu:    /* tell the gpu the current time */
coupled_heun_rk4_sph_nbody.cu:    cudaVerify(cudaMemcpyToSymbol(currentTimeD, &currentTime, sizeof(double)));
coupled_heun_rk4_sph_nbody.cu:    cudaVerify(cudaMemcpyToSymbol(predictor, &predictor_device, sizeof(struct Particle)));
coupled_heun_rk4_sph_nbody.cu:    cudaVerify(cudaMemcpyToSymbol(rk4_pointmass, &rk4_pointmass_device, sizeof(struct Pointmass) * 4));
coupled_heun_rk4_sph_nbody.cu:        /* tell the gpu the time step */
coupled_heun_rk4_sph_nbody.cu:            cudaVerify(cudaMemcpyToSymbol(dt, &param.maxtimestep, sizeof(double)));
coupled_heun_rk4_sph_nbody.cu:            cudaVerify(cudaMemcpyToSymbol(dt, &timePerStep, sizeof(double)));
coupled_heun_rk4_sph_nbody.cu:        /* tell the gpu the end time */
coupled_heun_rk4_sph_nbody.cu:        cudaVerify(cudaMemcpyToSymbol(endTimeD, &endTime, sizeof(double)));
coupled_heun_rk4_sph_nbody.cu:			cudaVerify(cudaDeviceSynchronize());
coupled_heun_rk4_sph_nbody.cu:	        cudaVerify(cudaMemcpyToSymbol(p, &p_device, sizeof(struct Particle)));
coupled_heun_rk4_sph_nbody.cu:            cudaVerify(cudaDeviceSynchronize());
coupled_heun_rk4_sph_nbody.cu:            cudaVerify(cudaMemcpyFromSymbol(&currentTime, currentTimeD, sizeof(double)));
coupled_heun_rk4_sph_nbody.cu:            cudaVerify(cudaMemcpyToSymbol(substep_currentTimeD, &substep_currentTime, sizeof(double)));
coupled_heun_rk4_sph_nbody.cu:	        cudaVerify(cudaMemcpyToSymbol(pointmass, &pointmass_device, sizeof(struct Pointmass)));
coupled_heun_rk4_sph_nbody.cu:            cudaVerify(cudaDeviceSynchronize());
coupled_heun_rk4_sph_nbody.cu:            cudaVerifyKernel((setTimestep_heun<<<numberOfMultiprocessors, NUM_THREADS_LIMITTIMESTEP>>>(
coupled_heun_rk4_sph_nbody.cu:            cudaVerify(cudaDeviceSynchronize());
coupled_heun_rk4_sph_nbody.cu:            /* get the time and the time step from the gpu */
coupled_heun_rk4_sph_nbody.cu:            cudaVerify(cudaMemcpyFromSymbol(&dt_host, dt, sizeof(double)));
coupled_heun_rk4_sph_nbody.cu:			cudaVerify(cudaDeviceSynchronize());
coupled_heun_rk4_sph_nbody.cu:	            cudaVerify(cudaMemcpyToSymbol(p, &p_device, sizeof(struct Particle)));
coupled_heun_rk4_sph_nbody.cu:	            cudaVerify(cudaMemcpyToSymbol(pointmass, &pointmass_device, sizeof(struct Pointmass)));
coupled_heun_rk4_sph_nbody.cu:    	        cudaVerifyKernel((PredictorStep_heun<<<numberOfMultiprocessors, NUM_THREADS_PC_INTEGRATOR>>>()));
coupled_heun_rk4_sph_nbody.cu:			    cudaVerify(cudaDeviceSynchronize());
coupled_heun_rk4_sph_nbody.cu:		        cudaVerify(cudaMemcpyToSymbol(p, &predictor_device, sizeof(struct Particle)));
coupled_heun_rk4_sph_nbody.cu:				cudaVerifyKernel((calculatePressure<<<numberOfMultiprocessors * 4, NUM_THREADS_PRESSURE>>>()));
coupled_heun_rk4_sph_nbody.cu:    			cudaVerify(cudaDeviceSynchronize());
coupled_heun_rk4_sph_nbody.cu:			    cudaVerify(cudaMemcpyFromSymbol(&dt_host, dt, sizeof(double)));
coupled_heun_rk4_sph_nbody.cu:				cudaVerifyKernel((pressureChangeCheck_heun<<<numberOfMultiprocessors, NUM_THREADS_PC_INTEGRATOR>>>(maxpressureDiffPerBlock)));
coupled_heun_rk4_sph_nbody.cu:    			cudaVerify(cudaDeviceSynchronize());
coupled_heun_rk4_sph_nbody.cu:                cudaVerify(cudaMemcpyFromSymbol(&pressureChangeSmallEnough_host, pressureChangeSmallEnough, sizeof(int)));
coupled_heun_rk4_sph_nbody.cu:                cudaVerify(cudaMemcpyFromSymbol(&maxpressureDiff_host, maxpressureDiff, sizeof(double)));
coupled_heun_rk4_sph_nbody.cu:			    cudaVerify(cudaMemcpyFromSymbol(&dt_host, dt, sizeof(double)));
coupled_heun_rk4_sph_nbody.cu:	                cudaVerify(cudaMemcpyToSymbol(pointmass, &pointmass_device, sizeof(struct Pointmass)));
coupled_heun_rk4_sph_nbody.cu:					cudaVerify(cudaMemcpyToSymbol(currentTimeD, &currentTime, sizeof(double)));
coupled_heun_rk4_sph_nbody.cu:		            cudaVerify(cudaMemcpyToSymbol(p, &predictor_device, sizeof(struct Particle)));
coupled_heun_rk4_sph_nbody.cu:	                cudaVerify(cudaMemcpyToSymbol(pointmass, &pointmass_device, sizeof(struct Pointmass)));
coupled_heun_rk4_sph_nbody.cu:		            cudaVerify(cudaMemcpyToSymbol(p, &p_device, sizeof(struct Particle)));
coupled_heun_rk4_sph_nbody.cu:	                cudaVerify(cudaMemcpyToSymbol(pointmass, &pointmass_device, sizeof(struct Pointmass)));
coupled_heun_rk4_sph_nbody.cu:    	            cudaVerifyKernel((CorrectorStep_heun<<<numberOfMultiprocessors, NUM_THREADS_PC_INTEGRATOR>>>()));
coupled_heun_rk4_sph_nbody.cu:        cudaVerify(cudaDeviceSynchronize());
coupled_heun_rk4_sph_nbody.cu:        cudaVerifyKernel((damageLimit<<<numberOfMultiprocessors*4, NUM_THREADS_PC_INTEGRATOR>>>()));
coupled_heun_rk4_sph_nbody.cu:        cudaVerify(cudaDeviceSynchronize());
coupled_heun_rk4_sph_nbody.cu:	cudaVerify(cudaFree(courantPerBlock));
coupled_heun_rk4_sph_nbody.cu:	cudaVerify(cudaFree(forcesPerBlock));
coupled_heun_rk4_sph_nbody.cu:    cudaVerify(cudaFree(dtSPerBlock));
coupled_heun_rk4_sph_nbody.cu:	cudaVerify(cudaFree(dtePerBlock));
coupled_heun_rk4_sph_nbody.cu:	cudaVerify(cudaFree(dtrhoPerBlock));
coupled_heun_rk4_sph_nbody.cu:	cudaVerify(cudaFree(dtdamagePerBlock));
coupled_heun_rk4_sph_nbody.cu:    cudaVerify(cudaFree(dtalphaPerBlock));
coupled_heun_rk4_sph_nbody.cu:    cudaVerify(cudaFree(dtbetaPerBlock));
coupled_heun_rk4_sph_nbody.cu:    cudaVerify(cudaFree(dtalpha_epsporPerBlock));
coupled_heun_rk4_sph_nbody.cu:    cudaVerify(cudaFree(dtepsilon_vPerBlock));
coupled_heun_rk4_sph_nbody.cu:    cudaVerify(cudaFree(dtartviscPerBlock));
coupled_heun_rk4_sph_nbody.cu:    cudaVerify(cudaFree(maxpressureDiffPerBlock));
artificial_stress.cu: * This file is part of miluphcuda.
artificial_stress.cu: * miluphcuda is free software: you can redistribute it and/or modify
artificial_stress.cu: * miluphcuda is distributed in the hope that it will be useful,
artificial_stress.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
rk2adaptive.h: * This file is part of miluphcuda.
rk2adaptive.h: * miluphcuda is free software: you can redistribute it and/or modify
rk2adaptive.h: * miluphcuda is distributed in the hope that it will be useful,
rk2adaptive.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
pc_values.dat:#        miluphCUDA
tree.cu: * This file is part of miluphcuda.
tree.cu: * miluphcuda is free software: you can redistribute it and/or modify
tree.cu: * miluphcuda is distributed in the hope that it will be useful,
tree.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
density.cu: * This file is part of miluphcuda.
density.cu: * miluphcuda is free software: you can redistribute it and/or modify
density.cu: * miluphcuda is distributed in the hope that it will be useful,
density.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
rk4_pointmass.cu: * This file is part of miluphcuda.
rk4_pointmass.cu: * miluphcuda is free software: you can redistribute it and/or modify
rk4_pointmass.cu: * miluphcuda is distributed in the hope that it will be useful,
rk4_pointmass.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
rk4_pointmass.cu:    cudaVerify(cudaDeviceSynchronize());
rk4_pointmass.cu:    cudaVerify(cudaMemcpyToSymbol(pointmass, &rk4_pointmass_device[RKFIRST], sizeof(struct Pointmass)));
rk4_pointmass.cu:    cudaVerifyKernel((rhs_pointmass<<<numberOfMultiprocessors, NUM_THREADS_RK4_INTEGRATE_STEP>>>()));
rk4_pointmass.cu:    cudaVerify(cudaDeviceSynchronize());
rk4_pointmass.cu:    cudaVerify(cudaDeviceSynchronize());
rk4_pointmass.cu:    cudaVerifyKernel((rhs_pointmass<<<numberOfMultiprocessors,NUM_THREADS_RK4_INTEGRATE_STEP>>>()));
rk4_pointmass.cu:    cudaVerifyKernel((rk4_integrateFirstStep<<<numberOfMultiprocessors, NUM_THREADS_RK4_INTEGRATE_STEP>>>()));
rk4_pointmass.cu:    cudaVerify(cudaDeviceSynchronize());
rk4_pointmass.cu:    cudaVerify(cudaMemcpyToSymbol(pointmass, &rk4_pointmass_device[RKFIRST], sizeof(struct Pointmass)));
rk4_pointmass.cu:    cudaVerifyKernel((rhs_pointmass<<<numberOfMultiprocessors,NUM_THREADS_RK4_INTEGRATE_STEP>>>()));
rk4_pointmass.cu:    cudaVerify(cudaDeviceSynchronize());
rk4_pointmass.cu:    cudaVerifyKernel((rk4_integrateSecondStep<<<numberOfMultiprocessors, NUM_THREADS_RK4_INTEGRATE_STEP>>>()));
rk4_pointmass.cu:    cudaVerify(cudaDeviceSynchronize());
rk4_pointmass.cu:    cudaVerify(cudaMemcpyToSymbol(pointmass, &rk4_pointmass_device[RKSECOND], sizeof(struct Pointmass)));
rk4_pointmass.cu:    cudaVerifyKernel((rhs_pointmass<<<numberOfMultiprocessors,NUM_THREADS_RK4_INTEGRATE_STEP>>>()));
rk4_pointmass.cu:    cudaVerifyKernel((rk4_integrateThirdStep<<<numberOfMultiprocessors, NUM_THREADS_RK4_INTEGRATE_STEP>>>()));
rk4_pointmass.cu:    cudaVerify(cudaMemcpyToSymbol(pointmass, &rk4_pointmass_device[RKTHIRD], sizeof(struct Pointmass)));
rk4_pointmass.cu:    cudaVerifyKernel((rhs_pointmass<<<numberOfMultiprocessors,NUM_THREADS_RK4_INTEGRATE_STEP>>>()));
rk4_pointmass.cu:    cudaVerify(cudaMemcpyToSymbol(pointmass, &pointmass_device, sizeof(struct Pointmass)));
rk4_pointmass.cu:    cudaVerifyKernel((rk4_integrateFourthStep<<<numberOfMultiprocessors, NUM_THREADS_RK4_INTEGRATE_STEP>>>()));
pressure.h: * This file is part of miluphcuda.
pressure.h: * miluphcuda is free software: you can redistribute it and/or modify
pressure.h: * miluphcuda is distributed in the hope that it will be useful,
pressure.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
porosity.cu: * This file is part of miluphcuda.
porosity.cu: * miluphcuda is free software: you can redistribute it and/or modify
porosity.cu: * miluphcuda is distributed in the hope that it will be useful,
porosity.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
kernel.h: * This file is part of miluphcuda.
kernel.h: * miluphcuda is free software: you can redistribute it and/or modify
kernel.h: * miluphcuda is distributed in the hope that it will be useful,
kernel.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
kernel.h:/// cuda allows function pointers since Fermi architecture.
stress.cu: * This file is part of miluphcuda.
stress.cu: * miluphcuda is free software: you can redistribute it and/or modify
stress.cu: * miluphcuda is distributed in the hope that it will be useful,
stress.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
internal_forces.cu: * This file is part of miluphcuda.
internal_forces.cu: * miluphcuda is free software: you can redistribute it and/or modify
internal_forces.cu: * miluphcuda is distributed in the hope that it will be useful,
internal_forces.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
parameter.h: * This file is part of miluphcuda.
parameter.h: * miluphcuda is free software: you can redistribute it and/or modify
parameter.h: * miluphcuda is distributed in the hope that it will be useful,
parameter.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
soundspeed.cu: * This file is part of miluphcuda.
soundspeed.cu: * miluphcuda is free software: you can redistribute it and/or modify
soundspeed.cu: * miluphcuda is distributed in the hope that it will be useful,
soundspeed.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
aneos.h: * This file is part of miluphcuda.
aneos.h: * miluphcuda is free software: you can redistribute it and/or modify
aneos.h: * miluphcuda is distributed in the hope that it will be useful,
aneos.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
extrema.h: * This file is part of miluphcuda.
extrema.h: * miluphcuda is free software: you can redistribute it and/or modify
extrema.h: * miluphcuda is distributed in the hope that it will be useful,
extrema.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
miluph.cu: * This file is part of miluphcuda.
miluph.cu: * miluphcuda is free software: you can redistribute it and/or modify
miluph.cu: * miluphcuda is distributed in the hope that it will be useful,
miluph.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
miluph.cu:#include <cuda_runtime.h>
miluph.cu:            "\nmiluphcuda is a multi-rheology, multi-material SPH code, developed mainly for astrophysical applications.\n"
miluph.cu:            "\t-G, --information\t\t Print information about detected Nvidia GPUs.\n"
miluph.cu:            "More information on github: https://github.com/christophmschaefer/miluphcuda\n\n",
miluph.cu:        MILUPHCUDA_VERSION, name);
miluph.cu:                fprintf(stdout, "Trying to use CUDA device %d\n", wanted_device);
miluph.cu:                cudaSetDevice(wanted_device);
miluph.cu:        cudaMemcpyFromSymbol(&kernel_h, wendlandc2_p, sizeof(SPH_kernel));
miluph.cu:        cudaMemcpyToSymbol(kernel, &kernel_h, sizeof(SPH_kernel));
miluph.cu:        cudaMemcpyFromSymbol(&kernel_h, wendlandc4_p, sizeof(SPH_kernel));
miluph.cu:        cudaMemcpyToSymbol(kernel, &kernel_h, sizeof(SPH_kernel));
miluph.cu:        cudaMemcpyFromSymbol(&kernel_h, wendlandc6_p, sizeof(SPH_kernel));
miluph.cu:        cudaMemcpyToSymbol(kernel, &kernel_h, sizeof(SPH_kernel));
miluph.cu:        cudaMemcpyFromSymbol(&kernel_h, cubic_spline_p, sizeof(SPH_kernel));
miluph.cu:        cudaMemcpyToSymbol(kernel, &kernel_h, sizeof(SPH_kernel));
miluph.cu:        cudaMemcpyFromSymbol(&kernel_h, spiky_p, sizeof(SPH_kernel));
miluph.cu:        cudaMemcpyToSymbol(kernel, &kernel_h, sizeof(SPH_kernel));
miluph.cu:    // query GPU(s)
miluph.cu:    fprintf(stdout, "\nChecking for cuda devices...\n");
miluph.cu:    cudaDeviceProp deviceProp;
miluph.cu:    cudaVerify(cudaGetDeviceProperties(&deviceProp, wanted_device));
miluph.cu:    cudaGetDeviceCount(&cnt);
miluph.cu:        fprintf(stderr, "There is no CUDA capable device. Exiting...\n");
miluph.cu:    fprintf(stdout, "Found #gpus: %d: %s\n", cnt, deviceProp.name);
miluph.cu:    fprintf(stdout, "Found cuda device with %d multiprocessors.\n", numberOfMultiprocessors);
miluph.cu:    // read/initialize material constants and copy them to the GPU + init some values
miluph.cu:    // copy the particles to the GPU
miluph.cu:    if (cudaSuccess != cudaMemcpyToSymbol(childList, &childListd, sizeof(void*))) {
miluph.cu:    cudaProfilerStart();
miluph.cu:    cudaProfilerStop();
miluph.cu:    fprintf(stdout, "Resetting GPU...\n");
miluph.cu:    cudaVerify(cudaDeviceReset());
predictor_corrector_euler.h: * This file is part of miluphcuda.
predictor_corrector_euler.h: * miluphcuda is free software: you can redistribute it and/or modify
predictor_corrector_euler.h: * miluphcuda is distributed in the hope that it will be useful,
predictor_corrector_euler.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
density.h: * This file is part of miluphcuda.
density.h: * miluphcuda is free software: you can redistribute it and/or modify
density.h: * miluphcuda is distributed in the hope that it will be useful,
density.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
examples/impact/run.sh:# If necessary, adapt the paths to the CUDA libs and the miluphcuda executable below, before running it.
examples/impact/run.sh:# set path to CUDA libs [change if necessary]
examples/impact/run.sh:export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
examples/impact/run.sh:# set path to miluphcuda executable [change if necessary]
examples/impact/run.sh:MC=../../miluphcuda
examples/impact/run.sh:# miluphcuda cmd line
examples/impact/run.sh:$MC -v -A -f impact.0000 -H -I rk2_adaptive -Q 1e-4 -m material.cfg -n 200 -t 5e-4 1>miluphcuda.output 2>miluphcuda.error &
examples/impact/USAGE.md:Impact example for miluphcuda
examples/impact/USAGE.md:The scenario uses ~60k SPH particles, with a runtime on the order of one hour on most current GPUs (benchmarked on a GTX 970).
examples/impact/USAGE.md:1. Compile miluphcuda using the `parameter.h` file from this directory.  
examples/impact/USAGE.md:   Don't forget to also adapt the miluphcuda Makefile to your system.
examples/impact/USAGE.md:3. Adapt the start script `run.sh` to your system (path to CUDA libs and to miluphcuda executable) and execute it.
examples/impact/USAGE.md:   Output to stdout and stderr is written to `miluphcuda.output` and `miluphcuda.error`, respectively.
examples/impact/USAGE.md:Take a look at the timestep statistics at the very bottom of *miluphcuda.output*. If you are not satisfied you may try
examples/impact/analyze-results/plot_plastic_yielding.Basalt.py:Plots particles' shear stresses from miluphcuda HDF5 output files + the theoretical
examples/impact/analyze-results/plot_plastic_yielding.Basalt.py:parser = argparse.ArgumentParser(description="Plots particles' shear stresses from miluphcuda HDF5 output files + the theoretical yield stress limit (parameters hardcoded in the script!).")
examples/impact/analyze-results/plot_p_alpha_convergence.py:Plots alpha(p) from miluphcuda HDF5 output files + the theoretical crush curve
examples/impact/analyze-results/plot_p_alpha_convergence.py:parser = argparse.ArgumentParser(description="Plots alpha(p) from miluphcuda HDF5 output files + the theoretical crush curve (parameters hardcoded in the script!).")
examples/impact/parameter.h: * This file is part of miluphcuda.
examples/impact/parameter.h: * miluphcuda is free software: you can redistribute it and/or modify
examples/impact/parameter.h: * miluphcuda is distributed in the hope that it will be useful,
examples/impact/parameter.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
examples/giant_collisions/USAGE.md:Giant Collision examples for miluphcuda
examples/giant_collisions/USAGE.md:and 2h for *solid* on most current GPUs (benchmarked on a GTX 970).
examples/giant_collisions/USAGE.md:1. Compile miluphcuda using the `parameter.h` file from the respective directory (hydro or solid).  
examples/giant_collisions/USAGE.md:   Don't forget to also adapt the miluphcuda Makefile to your system.
examples/giant_collisions/USAGE.md:3. Adapt the start script `run.sh` to your system (path to CUDA libs and to miluphcuda executable) and execute it.
examples/giant_collisions/USAGE.md:   Output to stdout and stderr is written to `miluphcuda.output` and `miluphcuda.error`, respectively.
examples/giant_collisions/USAGE.md:Take a look at the timestep statistics at the very bottom of *miluphcuda.output*. If you are not satisfied you may try
examples/giant_collisions/USAGE.md:        miluphcuda/utils/postprocessing/fast_identify_fragments_and_calc_aggregates/
examples/giant_collisions/hydro/run.sh:# If necessary, adapt the paths to the CUDA libs and the miluphcuda executable below, before running it.
examples/giant_collisions/hydro/run.sh:# set path to CUDA libs [change if necessary]
examples/giant_collisions/hydro/run.sh:export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
examples/giant_collisions/hydro/run.sh:# set path to miluphcuda executable [change if necessary]
examples/giant_collisions/hydro/run.sh:MC=../../../miluphcuda
examples/giant_collisions/hydro/run.sh:# miluphcuda cmd line
examples/giant_collisions/hydro/run.sh:$MC -v -A -f impact.0000 -g -H -I rk2_adaptive -Q 1e-4 -m material.cfg -n 75 -t 100.0 -s 1>miluphcuda.output 2>miluphcuda.error &
examples/giant_collisions/hydro/analyze-results/paraview.pvsm:        <Element index="0" value="/scratch1/burger/simulations/miluphcuda_22Feb2021/test_cases/giant_collisions/solid/paraview.xdmf"/>
examples/giant_collisions/hydro/analyze-results/paraview.pvsm:        <Element index="0" value="/scratch1/burger/simulations/miluphcuda_22Feb2021/test_cases/giant_collisions/solid/paraview.xdmf"/>
examples/giant_collisions/hydro/parameter.h: * This file is part of miluphcuda.
examples/giant_collisions/hydro/parameter.h: * miluphcuda is free software: you can redistribute it and/or modify
examples/giant_collisions/hydro/parameter.h: * miluphcuda is distributed in the hope that it will be useful,
examples/giant_collisions/hydro/parameter.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
examples/giant_collisions/solid/run.sh:# If necessary, adapt the paths to the CUDA libs and the miluphcuda executable below, before running it.
examples/giant_collisions/solid/run.sh:# set path to CUDA libs [change if necessary]
examples/giant_collisions/solid/run.sh:export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
examples/giant_collisions/solid/run.sh:# set path to miluphcuda executable [change if necessary]
examples/giant_collisions/solid/run.sh:MC=../../../miluphcuda
examples/giant_collisions/solid/run.sh:# miluphcuda cmd line
examples/giant_collisions/solid/run.sh:$MC -v -A -f impact.0000 -g -H -I rk2_adaptive -Q 1e-4 -m material.cfg -n 75 -t 100.0 -s 1>miluphcuda.output 2>miluphcuda.error &
examples/giant_collisions/solid/analyze-results/plot_plastic_yielding.Iron.py:Plots particles' shear stresses from miluphcuda HDF5 output files + the theoretical
examples/giant_collisions/solid/analyze-results/plot_plastic_yielding.Iron.py:parser = argparse.ArgumentParser(description="Plots particles' shear stresses from miluphcuda HDF5 output files + the theoretical yield stress limit (parameters hardcoded in the script!).")
examples/giant_collisions/solid/analyze-results/paraview.pvsm:        <Element index="0" value="/scratch1/burger/simulations/miluphcuda_22Feb2021/test_cases/giant_collisions/solid/paraview.xdmf"/>
examples/giant_collisions/solid/analyze-results/paraview.pvsm:        <Element index="0" value="/scratch1/burger/simulations/miluphcuda_22Feb2021/test_cases/giant_collisions/solid/paraview.xdmf"/>
examples/giant_collisions/solid/analyze-results/plot_plastic_yielding.Granite.py:Plots particles' shear stresses from miluphcuda HDF5 output files + the theoretical
examples/giant_collisions/solid/analyze-results/plot_plastic_yielding.Granite.py:parser = argparse.ArgumentParser(description="Plots particles' shear stresses from miluphcuda HDF5 output files + the theoretical yield stress limit (parameters hardcoded in the script!).")
examples/giant_collisions/solid/parameter.h: * This file is part of miluphcuda.
examples/giant_collisions/solid/parameter.h: * miluphcuda is free software: you can redistribute it and/or modify
examples/giant_collisions/solid/parameter.h: * miluphcuda is distributed in the hope that it will be useful,
examples/giant_collisions/solid/parameter.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
predictor_corrector.cu: * This file is part of miluphcuda.
predictor_corrector.cu: * miluphcuda is free software: you can redistribute it and/or modify
predictor_corrector.cu: * miluphcuda is distributed in the hope that it will be useful,
predictor_corrector.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
predictor_corrector.cu:    cudaVerify(cudaMalloc((void**)&courantPerBlock, sizeof(double)*numberOfMultiprocessors));
predictor_corrector.cu:    cudaVerify(cudaMalloc((void**)&forcesPerBlock, sizeof(double)*numberOfMultiprocessors));
predictor_corrector.cu:    cudaVerify(cudaMalloc((void**)&dtSPerBlock, sizeof(double)*numberOfMultiprocessors));
predictor_corrector.cu:    cudaVerify(cudaMalloc((void**)&dtePerBlock, sizeof(double)*numberOfMultiprocessors));
predictor_corrector.cu:    cudaVerify(cudaMalloc((void**)&dtrhoPerBlock, sizeof(double)*numberOfMultiprocessors));
predictor_corrector.cu:    cudaVerify(cudaMalloc((void**)&dtdamagePerBlock, sizeof(double)*numberOfMultiprocessors));
predictor_corrector.cu:    cudaVerify(cudaMalloc((void**)&dtalphaPerBlock, sizeof(double)*numberOfMultiprocessors));
predictor_corrector.cu:    cudaVerify(cudaMalloc((void**)&dtbetaPerBlock, sizeof(double)*numberOfMultiprocessors));
predictor_corrector.cu:    cudaVerify(cudaMalloc((void**)&maxpressureDiffPerBlock, sizeof(double)*numberOfMultiprocessors));
predictor_corrector.cu:    cudaVerify(cudaMalloc((void**)&dtartviscPerBlock, sizeof(double)*numberOfMultiprocessors));
predictor_corrector.cu:    cudaVerify(cudaMalloc((void**)&dtalpha_epsporPerBlock, sizeof(double)*numberOfMultiprocessors));
predictor_corrector.cu:    cudaVerify(cudaMalloc((void**)&dtepsilon_vPerBlock, sizeof(double)*numberOfMultiprocessors));
predictor_corrector.cu:    /* tell the gpu the current time */
predictor_corrector.cu:    cudaVerify(cudaMemcpyToSymbol(currentTimeD, &currentTime, sizeof(double)));
predictor_corrector.cu:    cudaVerify(cudaMemcpyToSymbol(predictor, &predictor_device, sizeof(struct Particle)));
predictor_corrector.cu:    /* tell the gpu the current time */
predictor_corrector.cu:    cudaVerify(cudaMemcpyToSymbol(predictor_pointmass, &predictor_pointmass_device, sizeof(struct Pointmass)));
predictor_corrector.cu:        /* tell the gpu the time step */
predictor_corrector.cu:        cudaVerify(cudaMemcpyToSymbol(dt, &timePerStep, sizeof(double)));
predictor_corrector.cu:        /* tell the gpu the end time */
predictor_corrector.cu:        cudaVerify(cudaMemcpyToSymbol(endTimeD, &endTime, sizeof(double)));
predictor_corrector.cu:			cudaVerify(cudaDeviceSynchronize());
predictor_corrector.cu:	        cudaVerify(cudaMemcpyToSymbol(p, &p_device, sizeof(struct Particle)));
predictor_corrector.cu:	        cudaVerify(cudaMemcpyToSymbol(pointmass, &pointmass_device, sizeof(struct Pointmass)));
predictor_corrector.cu:            cudaVerify(cudaDeviceSynchronize());
predictor_corrector.cu:            cudaVerify(cudaMemcpyFromSymbol(&currentTime, currentTimeD, sizeof(double)));
predictor_corrector.cu:            cudaVerify(cudaMemcpyToSymbol(substep_currentTimeD, &substep_currentTime, sizeof(double)));
predictor_corrector.cu:            cudaVerify(cudaDeviceSynchronize());
predictor_corrector.cu:            cudaVerifyKernel((setTimestep<<<numberOfMultiprocessors, NUM_THREADS_LIMITTIMESTEP>>>(
predictor_corrector.cu:            cudaVerify(cudaDeviceSynchronize());
predictor_corrector.cu:            /* get the time and the time step from the gpu */
predictor_corrector.cu:            cudaVerify(cudaMemcpyFromSymbol(&dt_host, dt, sizeof(double)));
predictor_corrector.cu:			cudaVerify(cudaDeviceSynchronize());
predictor_corrector.cu:    	        cudaVerifyKernel((PredictorStep<<<numberOfMultiprocessors, NUM_THREADS_PC_INTEGRATOR>>>()));
predictor_corrector.cu:			    cudaVerify(cudaDeviceSynchronize());
predictor_corrector.cu:		        cudaVerify(cudaMemcpyToSymbol(p, &predictor_device, sizeof(struct Particle)));
predictor_corrector.cu:		        cudaVerify(cudaMemcpyToSymbol(pointmass, &predictor_pointmass_device, sizeof(struct Pointmass)));
predictor_corrector.cu:            	cudaVerify(cudaMemcpyToSymbol(substep_currentTimeD, &substep_currentTime, sizeof(double)));
predictor_corrector.cu:	        	cudaVerify(cudaMemcpyToSymbol(p, &p_device, sizeof(struct Particle)));
predictor_corrector.cu:	        	cudaVerify(cudaMemcpyToSymbol(pointmass, &pointmass_device, sizeof(struct Pointmass)));
predictor_corrector.cu:            	cudaVerifyKernel((CorrectorStepPorous<<<numberOfMultiprocessors, NUM_THREADS_PC_INTEGRATOR>>>()));
predictor_corrector.cu:				cudaVerify(cudaDeviceSynchronize());
predictor_corrector.cu:				cudaVerify(cudaMemcpyToSymbol(p, &predictor_device, sizeof(struct Particle)));
predictor_corrector.cu:	        	cudaVerify(cudaMemcpyToSymbol(pointmass, &pointmass_device, sizeof(struct Pointmass)));
predictor_corrector.cu:				cudaVerifyKernel((calculatePressure<<<numberOfMultiprocessors * 4, NUM_THREADS_PRESSURE>>>()));
predictor_corrector.cu:    			cudaVerify(cudaDeviceSynchronize());
predictor_corrector.cu:				cudaVerifyKernel((pressureChangeCheck<<<numberOfMultiprocessors, NUM_THREADS_PC_INTEGRATOR>>>(maxpressureDiffPerBlock)));
predictor_corrector.cu:				cudaVerify(cudaDeviceSynchronize());
predictor_corrector.cu:				cudaVerify(cudaMemcpyFromSymbol(&pressureChangeSmallEnough_host, pressureChangeSmallEnough, sizeof(int)));
predictor_corrector.cu:                cudaVerify(cudaMemcpyFromSymbol(&maxpressureDiff_host, maxpressureDiff, sizeof(double)));
predictor_corrector.cu:                    cudaVerify(cudaMemcpyFromSymbol(&dt_host, dt, sizeof(double)));
predictor_corrector.cu:					cudaVerify(cudaMemcpyToSymbol(p, &p_device, sizeof(struct Particle)));
predictor_corrector.cu:	        	    cudaVerify(cudaMemcpyToSymbol(pointmass, &pointmass_device, sizeof(struct Pointmass)));
predictor_corrector.cu:					cudaVerify(cudaMemcpyFromSymbol(&dt_host, dt, sizeof(double)));
predictor_corrector.cu:					cudaVerify(cudaMemcpyToSymbol(currentTimeD, &currentTime, sizeof(double)));
predictor_corrector.cu:					cudaVerify(cudaDeviceSynchronize());
predictor_corrector.cu:            cudaVerifyKernel((CorrectorStep<<<numberOfMultiprocessors, NUM_THREADS_PC_INTEGRATOR>>>()));
predictor_corrector.cu:			cudaVerify(cudaDeviceSynchronize());
predictor_corrector.cu:            /* get the time and the time step from the gpu */
predictor_corrector.cu:            cudaVerify(cudaMemcpyFromSymbol(&currentTime, currentTimeD, sizeof(double)));
predictor_corrector.cu:        cudaVerify(cudaDeviceSynchronize());
predictor_corrector.cu:        cudaVerifyKernel((damageLimit<<<numberOfMultiprocessors*4, NUM_THREADS_PC_INTEGRATOR>>>()));
predictor_corrector.cu:        cudaVerify(cudaDeviceSynchronize());
predictor_corrector.cu:	cudaVerify(cudaFree(courantPerBlock));
predictor_corrector.cu:	cudaVerify(cudaFree(forcesPerBlock));
predictor_corrector.cu:	cudaVerify(cudaFree(dtSPerBlock));
predictor_corrector.cu:	cudaVerify(cudaFree(dtePerBlock));
predictor_corrector.cu:	cudaVerify(cudaFree(dtrhoPerBlock));
predictor_corrector.cu:	cudaVerify(cudaFree(dtdamagePerBlock));
predictor_corrector.cu:	cudaVerify(cudaFree(dtalphaPerBlock));
predictor_corrector.cu:	cudaVerify(cudaFree(dtbetaPerBlock));
predictor_corrector.cu:	cudaVerify(cudaFree(dtartviscPerBlock));
predictor_corrector.cu:    cudaVerify(cudaFree(maxpressureDiffPerBlock));
predictor_corrector.cu:    cudaVerify(cudaFree(dtalpha_epsporPerBlock));
predictor_corrector.cu:    cudaVerify(cudaFree(dtepsilon_vPerBlock));
utils/preprocessing/stl/generate_sph_from_stl.py:to fasten up the processing you might want to install pycuda 
utils/preprocessing/stl/generate_sph_from_stl.py:see https://documen.tician.de/pycuda for instructions
utils/preprocessing/stl/generate_sph_from_stl.py:    import pycuda.driver as cuda
utils/preprocessing/stl/generate_sph_from_stl.py:    import pycuda.autoinit
utils/preprocessing/stl/generate_sph_from_stl.py:    from pycuda.compiler import SourceModule
utils/preprocessing/stl/generate_sph_from_stl.py:    use_gpu = True
utils/preprocessing/stl/generate_sph_from_stl.py:    print("Found cuda support, using gpu. Make sure nvcc is in your PATH.")
utils/preprocessing/stl/generate_sph_from_stl.py:    print("no cuda support found, disabling gpu usage.")
utils/preprocessing/stl/generate_sph_from_stl.py:    use_gpu = False
utils/preprocessing/stl/generate_sph_from_stl.py:    print("Note: the script takes quite a while unless cuda is used.")
utils/preprocessing/stl/generate_sph_from_stl.py:if use_gpu:
utils/preprocessing/stl/generate_sph_from_stl.py:    print("transferring data to the gpu")
utils/preprocessing/stl/generate_sph_from_stl.py:    take_me_gpu = cuda.mem_alloc(take_me.nbytes)
utils/preprocessing/stl/generate_sph_from_stl.py:    x_gpu = cuda.mem_alloc(x.nbytes)
utils/preprocessing/stl/generate_sph_from_stl.py:    y_gpu = cuda.mem_alloc(y.nbytes)
utils/preprocessing/stl/generate_sph_from_stl.py:    z_gpu = cuda.mem_alloc(z.nbytes)
utils/preprocessing/stl/generate_sph_from_stl.py:    xs_gpu = cuda.mem_alloc(xs.nbytes)
utils/preprocessing/stl/generate_sph_from_stl.py:    ys_gpu = cuda.mem_alloc(ys.nbytes)
utils/preprocessing/stl/generate_sph_from_stl.py:    zs_gpu = cuda.mem_alloc(zs.nbytes)
utils/preprocessing/stl/generate_sph_from_stl.py:    nx_gpu = cuda.mem_alloc(nx.nbytes)
utils/preprocessing/stl/generate_sph_from_stl.py:    ny_gpu = cuda.mem_alloc(ny.nbytes)
utils/preprocessing/stl/generate_sph_from_stl.py:    nz_gpu = cuda.mem_alloc(nz.nbytes)
utils/preprocessing/stl/generate_sph_from_stl.py:    nop_gpu = cuda.mem_alloc(N.nbytes)
utils/preprocessing/stl/generate_sph_from_stl.py:    nom_gpu = cuda.mem_alloc(M.nbytes)
utils/preprocessing/stl/generate_sph_from_stl.py:    cuda.memcpy_htod(nop_gpu, N)
utils/preprocessing/stl/generate_sph_from_stl.py:    cuda.memcpy_htod(nom_gpu, M)
utils/preprocessing/stl/generate_sph_from_stl.py:    # now copy everything on the gpu
utils/preprocessing/stl/generate_sph_from_stl.py:    cuda.memcpy_htod(x_gpu, x)
utils/preprocessing/stl/generate_sph_from_stl.py:    cuda.memcpy_htod(y_gpu, y)
utils/preprocessing/stl/generate_sph_from_stl.py:    cuda.memcpy_htod(z_gpu, z)
utils/preprocessing/stl/generate_sph_from_stl.py:    cuda.memcpy_htod(xs_gpu, xs)
utils/preprocessing/stl/generate_sph_from_stl.py:    cuda.memcpy_htod(ys_gpu, ys)
utils/preprocessing/stl/generate_sph_from_stl.py:    cuda.memcpy_htod(zs_gpu, zs)
utils/preprocessing/stl/generate_sph_from_stl.py:    cuda.memcpy_htod(nx_gpu, nx)
utils/preprocessing/stl/generate_sph_from_stl.py:    cuda.memcpy_htod(ny_gpu, ny)
utils/preprocessing/stl/generate_sph_from_stl.py:    cuda.memcpy_htod(nz_gpu, nz)
utils/preprocessing/stl/generate_sph_from_stl.py:    function(x_gpu, y_gpu, z_gpu, xs_gpu, ys_gpu, zs_gpu, nx_gpu, ny_gpu, nz_gpu, take_me_gpu, nop_gpu, nom_gpu, block=(256,1,1))
utils/preprocessing/stl/generate_sph_from_stl.py:    cuda.memcpy_dtoh(take_me, take_me_gpu)
utils/preprocessing/stl/generate_sph_from_stl.py:# end of if use_gpu
utils/postprocessing/p_alpha_model/p_alpha_convergence.py:Plots alpha(p) from miluphcuda HDF5 output files + the theoretical crush curve
utils/postprocessing/p_alpha_model/p_alpha_convergence.py:parser = argparse.ArgumentParser(description="Plots alpha(p) from miluphcuda HDF5 output files + the theoretical crush curve (parameters hardcoded in the script!).")
utils/postprocessing/plasticity_models/plot_plastic_yielding.py:Plots particles' shear stresses from miluphcuda HDF5 output files + the theoretical
utils/postprocessing/plasticity_models/plot_plastic_yielding.py:parser = argparse.ArgumentParser(description="Plots particles' shear stresses from miluphcuda HDF5 output files + the theoretical yield stress limit (parameters hardcoded in the script!).")
utils/postprocessing/create_xdmf.py:Generates xdmf file from .h5 miluphcuda/miluphhpc output files for Paraview postprocessing.
utils/postprocessing/create_xdmf.py:        description='Generates xdmf file from .h5 miluphcuda/miluphpc output files for Paraview postprocessing. '
utils/postprocessing/create_xdmf.py:                    'Usually the defaults will produce what you want for miluphcuda (if cwd contains the .h5 files). '
utils/postprocessing/fast_identify_fragments_and_calc_aggregates/fast_identify_fragments.c:/* Tool for identifying fragments (particles connected by up to a smoothing length) in a miluphcuda output file.
utils/postprocessing/fast_identify_fragments_and_calc_aggregates/fast_identify_fragments.c: * Both, ASCII and HDF5 miluphcuda output files are supported, but currently only a constant smoothing length,
utils/postprocessing/fast_identify_fragments_and_calc_aggregates/fast_identify_fragments.c: * which is read directly from the miluphcuda output file.
utils/postprocessing/fast_identify_fragments_and_calc_aggregates/fast_identify_fragments.c:    fprintf(stdout, "\nTool for identifying fragments (particles connected by up to a smoothing length) in a miluphcuda output file.\n"
utils/postprocessing/fast_identify_fragments_and_calc_aggregates/fast_identify_fragments.c:                    "Both, ASCII and HDF5 miluphcuda output files are supported, but currently only a constant smoothing length,\n"
utils/postprocessing/fast_identify_fragments_and_calc_aggregates/fast_identify_fragments.c:                    "which is read directly from the miluphcuda output file.\n");
utils/postprocessing/fast_identify_fragments_and_calc_aggregates/fast_identify_fragments.c:    fprintf(stdout, "    -H               read from a miluphcuda HDF5 output file, otherwise an ASCII output file is assumed\n");
utils/postprocessing/fast_identify_fragments_and_calc_aggregates/fast_identify_fragments.c:    fprintf(stdout, "    -i inputfile     specify miluphcuda outputfile to read from\n");
utils/postprocessing/fast_identify_fragments_and_calc_aggregates/fast_identify_fragments.c://    fprintf(stdout, "    -s               set this flag if the miluphcuda file to read is from a solid run, otherwise it is assumed to be from a hydro run\n");
utils/postprocessing/fast_identify_fragments_and_calc_aggregates/fast_identify_fragments.c:// Builds the Barnes-Hut tree, following "Burtcher (2011) - An efficient CUDA implementation of the tree-based Barnes-Hut n-body algorithm".
utils/postprocessing/fast_identify_fragments_and_calc_aggregates/QuickStart.md:is intended for postprocessing miluphcuda` SPH simulations.
utils/postprocessing/fast_identify_fragments_and_calc_aggregates/QuickStart.md:This tool identifies *fragments* in a miluphcuda output file.
utils/postprocessing/fast_identify_fragments_and_calc_aggregates/QuickStart.md:* HDF5 and ASCII miluphcuda output files are supported
utils/postprocessing/fast_identify_fragments_and_calc_aggregates/QuickStart.md:* currently only constant sml is supported, which is read from the miluphcuda output file
utils/postprocessing/trace_particle.py:Python3 script to trace a SPH particle's properties over some time in a miluphCUDA run via the HDF5 output files.
utils/postprocessing/trace_particle.py:parser = argparse.ArgumentParser(description="Script to trace a SPH particle's properties over some time in a miluphCUDA run via the HDF5 output files.")
utils/postprocessing/trace_particle.py:parser.add_argument("--path", help = "path of the miluphCUDA directory to process", type = str)
utils/postprocessing/peak_pressures.py:# Extracts the peak pressures for all particles in a range of miluphcuda HDF5 output files and adds them to one or some of these files.
utils/postprocessing/peak_pressures.py:parser = argparse.ArgumentParser(description = "Extracts the peak pressures for all particles in a range of miluphcuda HDF5 output files and adds them to one or some of these files.")
velocity.cu: * This file is part of miluphcuda.
velocity.cu: * miluphcuda is free software: you can redistribute it and/or modify
velocity.cu: * miluphcuda is distributed in the hope that it will be useful,
velocity.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
xsph.cu: * This file is part of miluphcuda.
xsph.cu: * miluphcuda is free software: you can redistribute it and/or modify
xsph.cu: * miluphcuda is distributed in the hope that it will be useful,
xsph.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
rhs.cu: * This file is part of miluphcuda.
rhs.cu: * miluphcuda is free software: you can redistribute it and/or modify
rhs.cu: * miluphcuda is distributed in the hope that it will be useful,
rhs.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
rhs.cu:    cudaEvent_t start, stop;
rhs.cu:    cudaEventCreate(&start);
rhs.cu:    cudaEventCreate(&stop);
rhs.cu:    cudaVerify(cudaMemset(childListd, EMPTY, memorySizeForChildren));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((zero_all_derivatives<<<numberOfMultiprocessors, NUM_THREADS_256>>>(interactions)));
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaVerifyKernel((BoundaryConditionsBeforeRHS<<<16 * numberOfMultiprocessors, NUM_THREADS_BOUNDARY_CONDITIONS>>>(interactions)));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((insertGhostParticles<<<4 * numberOfMultiprocessors, NUM_THREADS_BOUNDARY_CONDITIONS>>>()));
rhs.cu:    //cudaVerifyKernel((insertGhostParticles<<<1, 1>>>()));
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((computationalDomain<<<numberOfMultiprocessors, NUM_THREADS_COMPUTATIONAL_DOMAIN>>>(
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaMemcpyFromSymbol(&xmin, minx, sizeof(double));
rhs.cu:    cudaMemcpyFromSymbol(&xmax, maxx, sizeof(double));
rhs.cu:    cudaMemcpyFromSymbol(&ymin, miny, sizeof(double));
rhs.cu:    cudaMemcpyFromSymbol(&ymax, maxy, sizeof(double));
rhs.cu:    cudaMemcpyFromSymbol(&zmin, minz, sizeof(double));
rhs.cu:    cudaMemcpyFromSymbol(&zmax, maxz, sizeof(double));
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((buildTree<<<numberOfMultiprocessors, NUM_THREADS_BUILD_TREE>>>()));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaMemcpyFromSymbol(&maxNodeIndex_host, maxNodeIndex, sizeof(int));
rhs.cu:    cudaVerify(cudaMalloc((void**)&treeDepthPerBlock, sizeof(int)*numberOfMultiprocessors));
rhs.cu:    cudaVerifyKernel((getTreeDepth<<<numberOfMultiprocessors, NUM_THREADS_TREEDEPTH>>>(treeDepthPerBlock)));
rhs.cu:    cudaMemcpyFromSymbol(&maxtreedepth_host, treeMaxDepth, sizeof(int));
rhs.cu:    cudaVerify(cudaFree(treeDepthPerBlock));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((check_sml_boundary<<<numberOfMultiprocessors * 4, NUM_THREADS_NEIGHBOURSEARCH>>>()));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaVerifyKernel((knnNeighbourSearch<<<numberOfMultiprocessors * 4, NUM_THREADS_NEIGHBOURSEARCH>>>(
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaVerifyKernel((nearNeighbourSearch_modify_sml<<<numberOfMultiprocessors * 4, NUM_THREADS_NEIGHBOURSEARCH>>>(
rhs.cu:    cudaVerifyKernel((nearNeighbourSearch<<<numberOfMultiprocessors * 4, NUM_THREADS_NEIGHBOURSEARCH>>>(
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaVerifyKernel((setEmptyMassForInnerNodes<<<numberOfMultiprocessors * 4, NUM_THREADS_512>>>()));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaMemcpy(p_host.noi, p_device.noi, memorySizeForInteractions, cudaMemcpyDeviceToHost);
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((calculateDensity<<<numberOfMultiprocessors * 4, NUM_THREADS_DENSITY>>>( interactions)));
rhs.cu://    cudaVerifyKernel((calculateDensity<<<1,1>>>( interactions)));
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((shepardCorrection<<<numberOfMultiprocessors*4, NUM_THREADS_256>>>( interactions)));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    //cudaVerifyKernel((printTensorialCorrectionMatrix<<<1,1>>>( interactions)));
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((calculateSoundSpeed<<<numberOfMultiprocessors * 4, NUM_THREADS_PRESSURE>>>()));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((CalcDivvandCurlv<<<numberOfMultiprocessors * 4, NUM_THREADS_128>>>(
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((calculateCompressiveStrength<<<numberOfMultiprocessors * 4, NUM_THREADS_PRESSURE>>>()));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaVerifyKernel((calculateTensileStrength<<<numberOfMultiprocessors * 4, NUM_THREADS_PRESSURE>>>()));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((plasticity<<<numberOfMultiprocessors * 4, NUM_THREADS_PRESSURE>>>()));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((calculatePressure<<<numberOfMultiprocessors * 4, NUM_THREADS_PRESSURE>>>()));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaVerifyKernel((calculateDistensionChange<<<numberOfMultiprocessors * 4, NUM_THREADS_PALPHA_POROSITY>>>()));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:        cudaEventRecord(start, 0);
rhs.cu:        cudaVerifyKernel((calculateCentersOfMass<<<1, NUM_THREADS_CALC_CENTER_OF_MASS>>>()));
rhs.cu:        cudaVerify(cudaDeviceSynchronize());
rhs.cu:        cudaEventRecord(stop, 0);
rhs.cu:        cudaEventSynchronize(stop);
rhs.cu:        cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((betaviscosity<<<numberOfMultiprocessors * 4, NUM_THREADS_128>>>(
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((symmetrizeStress<<<4 * numberOfMultiprocessors, NUM_THREADS_512>>>()));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((damageLimit<<<numberOfMultiprocessors*4, NUM_THREADS_512>>>()));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((plasticityModel<<<numberOfMultiprocessors * 4, NUM_THREADS_512>>>()));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((JohnsonCookPlasticity<<<numberOfMultiprocessors * 4, NUM_THREADS_512>>>()));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((tensorialCorrection<<<numberOfMultiprocessors*4, NUM_THREADS_256>>>( interactions)));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu://    cudaVerifyKernel((printTensorialCorrectionMatrix<<<1,1>>>( interactions)));
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((calculatedeviatoricStress<<<numberOfMultiprocessors*4, NUM_THREADS_256>>>( interactions)));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaVerifyKernel((calculateXSPHchanges<<<4 * numberOfMultiprocessors, NUM_THREADS_512>>>(interactions)));
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((setQuantitiesGhostParticles<<<numberOfMultiprocessors, NUM_THREADS_BOUNDARY_CONDITIONS>>>()));
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaVerifyKernel((checkNaNs<<<numberOfMultiprocessors, NUM_THREADS_128>>>(interactions)));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((set_stress_tensor<<<numberOfMultiprocessors, NUM_THREADS_256>>>()));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((calculate_kinematic_viscosity<<<numberOfMultiprocessors, NUM_THREADS_256>>>()));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((calculate_shear_stress_tensor<<<numberOfMultiprocessors, NUM_THREADS_256>>>(interactions)));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((compute_artificial_stress<<<numberOfMultiprocessors, NUM_THREADS_256>>>(interactions)));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((internalForces<<<numberOfMultiprocessors, NUM_THREADS_128>>>(interactions)));
rhs.cu:    //cudaVerifyKernel((internalForces<<<1, 1 >>>(interactions)));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerifyKernel((gravitation_from_point_masses<<<numberOfMultiprocessors, NUM_THREADS_128>>>(calculate_nbody)));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaEventRecord(start, 0);
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventRecord(stop, 0);
rhs.cu:    cudaEventSynchronize(stop);
rhs.cu:    cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaVerifyKernel((checkNaNs<<<numberOfMultiprocessors, NUM_THREADS_128>>>(interactions)));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaVerifyKernel((removeGhostParticles<<<1,1>>>()));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:        cudaEventRecord(start, 0);
rhs.cu:        cudaVerify(cudaMalloc((void**)&movingparticlesPerBlock, sizeof(int)*numberOfMultiprocessors));
rhs.cu:        cudaVerifyKernel(((measureTreeChange<<<numberOfMultiprocessors, NUM_THREADS_TREECHANGE>>>(movingparticlesPerBlock))));
rhs.cu:        cudaMemcpyFromSymbol(&movingparticles_host, movingparticles, sizeof(int));
rhs.cu:            cudaMemcpyToSymbol(reset_movingparticles, &flag_force_gravity_calc, sizeof(int));
rhs.cu:        cudaVerify(cudaFree(movingparticlesPerBlock));
rhs.cu:        cudaEventRecord(stop, 0);
rhs.cu:        cudaEventSynchronize(stop);
rhs.cu:        cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:        cudaEventRecord(start, 0);
rhs.cu:            cudaVerifyKernel((selfgravity<<<16*numberOfMultiprocessors, NUM_THREADS_SELFGRAVITY>>>()));
rhs.cu:            cudaMemcpyToSymbol(reset_movingparticles, &flag_force_gravity_calc, sizeof(int));
rhs.cu:            cudaVerifyKernel((addoldselfgravity<<<16*numberOfMultiprocessors, NUM_THREADS_SELFGRAVITY>>>()));
rhs.cu:        cudaVerify(cudaDeviceSynchronize());
rhs.cu:        cudaEventRecord(stop, 0);
rhs.cu:        cudaEventSynchronize(stop);
rhs.cu:        cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:        cudaEventRecord(start, 0);
rhs.cu:        cudaVerifyKernel((direct_selfgravity<<<numberOfMultiprocessors, NUM_THREADS_SELFGRAVITY>>>()));
rhs.cu:        cudaVerify(cudaDeviceSynchronize());
rhs.cu:        cudaEventRecord(stop, 0);
rhs.cu:        cudaEventSynchronize(stop);
rhs.cu:        cudaEventElapsedTime(&time[timerCounter], start, stop);
rhs.cu:    cudaVerifyKernel((BoundaryConditionsAfterRHS<<<16 * numberOfMultiprocessors, NUM_THREADS_BOUNDARY_CONDITIONS>>>(interactions)));
rhs.cu:    cudaVerifyKernel((setlocationchanges<<<4 * numberOfMultiprocessors, NUM_THREADS_512>>>(interactions)));
rhs.cu:    cudaVerifyKernel((check_sml_boundary<<<numberOfMultiprocessors * 4, NUM_THREADS_NEIGHBOURSEARCH>>>()));
rhs.cu:    cudaVerify(cudaDeviceSynchronize());
rhs.cu:    cudaEventDestroy(start);
rhs.cu:    cudaEventDestroy(stop);
artificial_stress.h: * This file is part of miluphcuda.
artificial_stress.h: * miluphcuda is free software: you can redistribute it and/or modify
artificial_stress.h: * miluphcuda is distributed in the hope that it will be useful,
artificial_stress.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
boundary.cu: * This file is part of miluphcuda.
boundary.cu: * miluphcuda is free software: you can redistribute it and/or modify
boundary.cu: * miluphcuda is distributed in the hope that it will be useful,
boundary.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
viscosity.cu: * This file is part of miluphcuda.
viscosity.cu: * miluphcuda is free software: you can redistribute it and/or modify
viscosity.cu: * miluphcuda is distributed in the hope that it will be useful,
viscosity.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
tree.h: * This file is part of miluphcuda.
tree.h: * miluphcuda is free software: you can redistribute it and/or modify
tree.h: * miluphcuda is distributed in the hope that it will be useful,
tree.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
sinking.h: * This file is part of miluphcuda.
sinking.h: * miluphcuda is free software: you can redistribute it and/or modify
sinking.h: * miluphcuda is distributed in the hope that it will be useful,
sinking.h: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
rk2adaptive.cu: * This file is part of miluphcuda.
rk2adaptive.cu: * miluphcuda is free software: you can redistribute it and/or modify
rk2adaptive.cu: * miluphcuda is distributed in the hope that it will be useful,
rk2adaptive.cu: * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
rk2adaptive.cu:    cudaVerify(cudaMemcpyToSymbol(rk_epsrel_d, &param.rk_epsrel, sizeof(double)));
rk2adaptive.cu:    cudaVerify(cudaMalloc((void**)&maxPosAbsErrorPerBlock, sizeof(double)*numberOfMultiprocessors));
rk2adaptive.cu:    cudaVerify(cudaMalloc((void**)&maxVelAbsErrorPerBlock, sizeof(double)*numberOfMultiprocessors));
rk2adaptive.cu:    cudaVerify(cudaMalloc((void**)&maxDensityAbsErrorPerBlock , sizeof(double)*numberOfMultiprocessors));
rk2adaptive.cu:    cudaVerify(cudaMalloc((void**)&maxEnergyAbsErrorPerBlock, sizeof(double)*numberOfMultiprocessors));
rk2adaptive.cu:    cudaVerify(cudaMalloc((void**)&courantPerBlock, sizeof(double)*numberOfMultiprocessors));
rk2adaptive.cu:    cudaVerify(cudaMalloc((void**)&forcesPerBlock, sizeof(double)*numberOfMultiprocessors));
rk2adaptive.cu:    cudaVerify(cudaMalloc((void**)&maxDamageTimeStepPerBlock, sizeof(double)*numberOfMultiprocessors));
rk2adaptive.cu:    cudaVerify(cudaMalloc((void**)&maxPressureAbsChangePerBlock, sizeof(double)*numberOfMultiprocessors));
rk2adaptive.cu:    cudaVerify(cudaMalloc((void**)&maxAlphaDiffPerBlock, sizeof(double)*numberOfMultiprocessors));
rk2adaptive.cu:    cudaVerify(cudaMemcpyToSymbol(rk, &rk_device, sizeof(struct Particle) * 3));
rk2adaptive.cu:    cudaVerify(cudaMemcpyToSymbol(rk_pointmass, &rk_pointmass_device, sizeof(struct Pointmass) * 3));
rk2adaptive.cu:    cudaVerify(cudaDeviceSynchronize());
rk2adaptive.cu:    cudaVerify(cudaMemcpyToSymbol(currentTimeD, &currentTime, sizeof(double)));
rk2adaptive.cu:        cudaVerify(cudaMemcpyToSymbol(endTimeD, &endTime, sizeof(double)));
rk2adaptive.cu:        cudaVerify(cudaMemcpyToSymbol(dt, &dt_host, sizeof(double)));
rk2adaptive.cu:            cudaVerify(cudaMemcpyToSymbol(substep_currentTimeD, &substep_currentTime, sizeof(double)));
rk2adaptive.cu:            cudaVerify(cudaDeviceSynchronize());
rk2adaptive.cu:            cudaVerify(cudaDeviceSynchronize());
rk2adaptive.cu:            cudaVerify(cudaDeviceSynchronize());
rk2adaptive.cu:            cudaVerify(cudaMemcpyToSymbol(p, &rk_device[RKFIRST], sizeof(struct Particle)));
rk2adaptive.cu:            cudaVerify(cudaMemcpyToSymbol(pointmass, &rk_pointmass_device[RKFIRST], sizeof(struct Pointmass)));
rk2adaptive.cu:            cudaVerify(cudaDeviceSynchronize());
rk2adaptive.cu:            cudaVerifyKernel((limitTimestepCourant<<<numberOfMultiprocessors, NUM_THREADS_LIMITTIMESTEP>>>(
rk2adaptive.cu:            cudaVerify(cudaDeviceSynchronize());
rk2adaptive.cu:            cudaVerify(cudaMemcpyFromSymbol(&dt_new, dt, sizeof(double)));
rk2adaptive.cu:            cudaVerifyKernel((limitTimestepForces<<<numberOfMultiprocessors, NUM_THREADS_LIMITTIMESTEP>>>(
rk2adaptive.cu:            cudaVerify(cudaDeviceSynchronize());
rk2adaptive.cu:            cudaVerify(cudaMemcpyFromSymbol(&dt_new, dt, sizeof(double)));
rk2adaptive.cu:            cudaVerifyKernel((limitTimestepDamage<<<numberOfMultiprocessors, NUM_THREADS_LIMITTIMESTEP>>>(
rk2adaptive.cu:            cudaVerify(cudaDeviceSynchronize());
rk2adaptive.cu:            cudaVerify(cudaMemcpyFromSymbol(&dt_new, dt, sizeof(double)));
rk2adaptive.cu:                cudaVerify(cudaDeviceSynchronize());
rk2adaptive.cu:                cudaVerifyKernel((integrateFirstStep<<<numberOfMultiprocessors, NUM_THREADS_RK2_INTEGRATE_STEP>>>()));
rk2adaptive.cu:                cudaVerify(cudaDeviceSynchronize());
rk2adaptive.cu:                cudaVerify(cudaMemcpyFromSymbol(&dt_host, dt, sizeof(double)));
rk2adaptive.cu:                cudaVerify(cudaMemcpyToSymbol(substep_currentTimeD, &substep_currentTime, sizeof(double)));
rk2adaptive.cu:                cudaVerify(cudaMemcpyToSymbol(p, &rk_device[RKFIRST], sizeof(struct Particle)));
rk2adaptive.cu:                cudaVerify(cudaMemcpyToSymbol(pointmass, &rk_pointmass_device[RKFIRST], sizeof(struct Pointmass)));
rk2adaptive.cu:                cudaVerify(cudaDeviceSynchronize());
rk2adaptive.cu:                cudaVerifyKernel((integrateSecondStep<<<numberOfMultiprocessors, NUM_THREADS_RK2_INTEGRATE_STEP>>>()));
rk2adaptive.cu:                cudaVerify(cudaDeviceSynchronize());
rk2adaptive.cu:                cudaVerify(cudaMemcpyToSymbol(p, &rk_device[RKSECOND], sizeof(struct Particle)));
rk2adaptive.cu:                cudaVerify(cudaMemcpyToSymbol(pointmass, &rk_pointmass_device[RKSECOND], sizeof(struct Pointmass)));
rk2adaptive.cu:                cudaVerify(cudaMemcpyToSymbol(substep_currentTimeD, &substep_currentTime, sizeof(double)));
rk2adaptive.cu:                cudaVerify(cudaDeviceSynchronize());
rk2adaptive.cu:                cudaVerify(cudaMemcpyToSymbol(p, &p_device, sizeof(struct Particle)));
rk2adaptive.cu:                cudaVerify(cudaMemcpyToSymbol(pointmass, &pointmass_device, sizeof(struct Pointmass)));
rk2adaptive.cu:                cudaVerifyKernel((integrateThirdStep<<<numberOfMultiprocessors, NUM_THREADS_RK2_INTEGRATE_STEP>>>()));
rk2adaptive.cu:                cudaVerify(cudaDeviceSynchronize());
rk2adaptive.cu:                cudaVerifyKernel((checkError<<<numberOfMultiprocessors, NUM_THREADS_ERRORCHECK>>>(
rk2adaptive.cu:                cudaVerify(cudaDeviceSynchronize());
rk2adaptive.cu:                cudaVerify(cudaMemcpyFromSymbol(&dt_suggested, dtNewErrorCheck, sizeof(double)));
rk2adaptive.cu:                cudaVerify(cudaMemcpyFromSymbol(&errorSmallEnough_host, errorSmallEnough, sizeof(int)));
rk2adaptive.cu:                cudaVerify(cudaDeviceSynchronize());
rk2adaptive.cu:                    cudaVerifyKernel((BoundaryConditionsAfterIntegratorStep<<<numberOfMultiprocessors, NUM_THREADS_ERRORCHECK>>>(interactions)));
rk2adaptive.cu:                    cudaVerify(cudaDeviceSynchronize());
rk2adaptive.cu:                    cudaVerify(cudaMemcpyFromSymbol(&errPos, maxPosAbsError, sizeof(double)));
rk2adaptive.cu:                    cudaVerify(cudaMemcpyFromSymbol(&errVel, maxVelAbsError, sizeof(double)));
rk2adaptive.cu:                    cudaVerify(cudaMemcpyFromSymbol(&errDensity, maxDensityAbsError, sizeof(double)));
rk2adaptive.cu:                    cudaVerify(cudaMemcpyFromSymbol(&errEnergy, maxEnergyAbsError, sizeof(double)));
rk2adaptive.cu:                    cudaVerify(cudaDeviceSynchronize());
rk2adaptive.cu:                    cudaVerify(cudaMemcpyFromSymbol(&errPressure, maxPressureAbsChange, sizeof(double)));
rk2adaptive.cu:                    cudaVerify(cudaMemcpyFromSymbol(&errAlpha, maxAlphaDiff, sizeof(double)));
rk2adaptive.cu:                    cudaVerify(cudaDeviceSynchronize());
rk2adaptive.cu:                /* tell the GPU the new timestep and the current time */
rk2adaptive.cu:                cudaVerify(cudaMemcpyToSymbol(currentTimeD, &currentTime, sizeof(double)));
rk2adaptive.cu:                cudaVerify(cudaMemcpyToSymbol(dt, &dt_host, sizeof(double)));
rk2adaptive.cu:                    cudaVerify(cudaDeviceSynchronize());
rk2adaptive.cu:        cudaVerify(cudaDeviceSynchronize());
rk2adaptive.cu:        cudaVerifyKernel((damageLimit<<<numberOfMultiprocessors*4, NUM_THREADS_PC_INTEGRATOR>>>()));
rk2adaptive.cu:        cudaVerify(cudaDeviceSynchronize());
rk2adaptive.cu:    cudaVerify(cudaFree(maxPosAbsErrorPerBlock));
rk2adaptive.cu:    cudaVerify(cudaFree(maxVelAbsErrorPerBlock));
rk2adaptive.cu:    cudaVerify(cudaFree(courantPerBlock));
rk2adaptive.cu:    cudaVerify(cudaFree(forcesPerBlock));
rk2adaptive.cu:    cudaVerify(cudaFree(maxDamageTimeStepPerBlock));
rk2adaptive.cu:    cudaVerify(cudaFree(maxEnergyAbsErrorPerBlock));
rk2adaptive.cu:    cudaVerify(cudaFree(maxDensityAbsErrorPerBlock));
rk2adaptive.cu:    cudaVerify(cudaFree(maxPressureAbsChangePerBlock));
rk2adaptive.cu:    cudaVerify(cudaFree(maxAlphaDiffPerBlock));
rk2adaptive.cu:    cudaVerify(cudaMemcpyFromSymbol(&tmp, max_abs_pressure_change, sizeof(double)));

```
