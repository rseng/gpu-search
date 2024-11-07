# https://github.com/galtay/sphray

```console
src/gadget_owls_input.F90:#ifdef incCloudy
src/main_input.F90:        call read_Gpublic_particles()
src/main_input.F90:        call read_Gpubhdf5_particles()
src/gadget_vbromm_input.F90:!        call ghead%read_Gpublic_header_file(snapfile)
src/gadget_vbromm_input.F90:!        call ghead%print_Gpublic_header_lun(loglun)
src/gadget_vbromm_input.F90:!        call saved_gheads(i,j)%copy_Gpublic_header(ghead)
src/output.F90:#ifdef incCloudy
src/output.F90:#ifdef incCloudy
src/output.F90:#ifdef incCloudy
src/update_particles.F90:     call read_Gpublic_particles()
src/update_particles.F90:     call read_Gpubhdf5_particles()
src/sphpar.F90:#ifdef incCloudy
src/sphpar.F90:#ifdef incCloudy
src/gadget_public_input.F90:public :: read_Gpublic_particles
src/gadget_public_input.F90:!        call ghead%read_Gpublic_header_file(snapfile)
src/gadget_public_input.F90:!        call ghead%print_Gpublic_header_lun(loglun)
src/gadget_public_input.F90:!        call saved_gheads(i,j)%copy_Gpublic_header(ghead)
src/gadget_public_input.F90:subroutine read_Gpublic_particles()
src/gadget_public_input.F90:  character(clen), parameter :: myname="read_Gpublic_particles"
src/gadget_public_input.F90:end subroutine read_Gpublic_particles
src/gadget_public_input_hdf5.F90:public :: read_Gpubhdf5_particles
src/gadget_public_input_hdf5.F90:  subroutine read_Gpubhdf5_particles()
src/gadget_public_input_hdf5.F90:  end subroutine read_Gpubhdf5_particles
src/gadget_public_input_hdf5.F90:!          call ghead%read_Gpublic_header_hdf5_file(snapfile)
src/gadget_public_input_hdf5.F90:!          call ghead%print_Gpublic_header_lun(loglun)
src/gadget_public_input_hdf5.F90:!          call saved_gheads(i,j)%copy_Gpublic_header(ghead)
src/gadget_public_input_hdf5.F90:  subroutine read_Gpubhdf5_particles()
src/gadget_public_input_hdf5.F90:    character(clen), parameter :: myname="read_Gpubhdf5_particles" 
src/gadget_public_input_hdf5.F90:  end subroutine read_Gpubhdf5_particles
src/MakefileTemplate:#OPT += -DincCloudy  # store CLOUDY eq. ionization values for particles
src/MakefileTemplate:   OPT += -DincCloudy
src/MakefileTemplate:   OPT += -DincCloudy
src/particle_system.F90:#ifdef incCloudy
src/particle_system.F90:#ifdef incCloudy
src/particle_system.F90:#ifdef incCloudy

```
