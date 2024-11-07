# https://github.com/mpip/pfft

```console
kernel/sertrafo.c:/* Global infos about procmesh are only enabled in debug mode
kernel/ipfft.h:/* procmesh.c */
kernel/ipfft.h:int PX(is_cart_procmesh)(
kernel/ipfft.h:int PX(is_cart_procmesh_2d)(
kernel/ipfft.h:void PX(split_cart_procmesh)(
kernel/ipfft.h:void PX(split_cart_procmesh_3dto2d_p0q0)(
kernel/ipfft.h:void PX(split_cart_procmesh_3dto2d_p1q1)(
kernel/ipfft.h:void PX(get_procmesh_dims_2d)(
kernel/ipfft.h:void PX(split_cart_procmesh_for_3dto2d_remap_q0)(
kernel/ipfft.h:void PX(split_cart_procmesh_for_3dto2d_remap_q1)(
kernel/procmesh.c:int PX(create_procmesh)(
kernel/procmesh.c:int PX(create_procmesh_1d)(
kernel/procmesh.c:  return PX(create_procmesh)(rnk, comm, &np0,
kernel/procmesh.c:int PX(create_procmesh_2d)(
kernel/procmesh.c:  return PX(create_procmesh)(rnk, comm, np,
kernel/procmesh.c:int PX(is_cart_procmesh_2d)(
kernel/procmesh.c:int PX(is_cart_procmesh)(
kernel/procmesh.c:  if(PX(is_cart_procmesh)(comm)){
kernel/procmesh.c:    PX(create_procmesh_1d)(comm, np, &comm_cart);
kernel/procmesh.c:/* allocate comms_1d before call of split_cart_procmesh */
kernel/procmesh.c:void PX(split_cart_procmesh)(
kernel/procmesh.c:void PX(split_cart_procmesh_3dto2d_p0q0)(
kernel/procmesh.c:  if( !PX(is_cart_procmesh)(comm_cart_3d) )
kernel/procmesh.c:  PX(get_procmesh_dims_2d)(comm_cart_3d, &p0, &p1, &q0, &q1);
kernel/procmesh.c:void PX(split_cart_procmesh_3dto2d_p1q1)(
kernel/procmesh.c:  if( !PX(is_cart_procmesh)(comm_cart_3d) )
kernel/procmesh.c:  PX(get_procmesh_dims_2d)(comm_cart_3d, &p0, &p1, &q0, &q1);
kernel/procmesh.c:void PX(split_cart_procmesh_for_3dto2d_remap_q0)(
kernel/procmesh.c:  if( !PX(is_cart_procmesh)(comm_cart_3d) )
kernel/procmesh.c:  PX(get_procmesh_dims_2d)(comm_cart_3d, &p0, &p1, &q0, &q1);
kernel/procmesh.c:void PX(split_cart_procmesh_for_3dto2d_remap_q1)(
kernel/procmesh.c:  if( !PX(is_cart_procmesh)(comm_cart_3d) )
kernel/procmesh.c:  PX(get_procmesh_dims_2d)(comm_cart_3d, &p0, &p1, &q0, &q1);
kernel/procmesh.c:void PX(get_procmesh_dims_2d)(
kernel/partrafo.c:static void malloc_and_split_cart_procmesh(
kernel/partrafo.c:  malloc_and_split_cart_procmesh(rnk_n, transp_flag, comm_cart,
kernel/partrafo.c:  /* dimension of FFT is not allowed to be smaller than procmesh dimension */
kernel/partrafo.c:  /* equal dimension of FFT and procmesh only implemented for 3 dimensions */
kernel/partrafo.c:  malloc_and_split_cart_procmesh(rnk_n, transp_flag, comm_cart,
kernel/partrafo.c:static void malloc_and_split_cart_procmesh(
kernel/partrafo.c:    PX(split_cart_procmesh_3dto2d_p0q0)(comm_cart,
kernel/partrafo.c:    PX(split_cart_procmesh_3dto2d_p1q1)(comm_cart,
kernel/partrafo.c:    PX(split_cart_procmesh)(comm_cart, *comms_pm);
kernel/partrafo.c:    PX(get_procmesh_dims_2d)(comm_cart, &p0, &p1, &q0, &q1);
kernel/Makefile.am:	procmesh.c \
kernel/remap_3dto2d.c:/* Global infos about procmesh are only enabled in debug mode
kernel/remap_3dto2d.c:  /* remap only works for 3d data on 3d procmesh */
kernel/remap_3dto2d.c:  PX(get_procmesh_dims_2d)(comm_cart_3d, &p0, &p1, &q0, &q1);
kernel/remap_3dto2d.c:  /* remap only works for 3d data on 3d procmesh */
kernel/remap_3dto2d.c:  PX(get_procmesh_dims_2d)(comm_cart_3d, &p0, &p1, &q0, &q1);
kernel/remap_3dto2d.c:  PX(split_cart_procmesh_for_3dto2d_remap_q1)(comm_cart_3d, &comm_q1);
kernel/remap_3dto2d.c:  PX(split_cart_procmesh_for_3dto2d_remap_q0)(comm_cart_3d, &comm_q0);
kernel/remap_3dto2d.c:  /* remap only works for 3d data on 3d procmesh */
kernel/remap_3dto2d.c:  PX(get_procmesh_dims_2d)(comm_cart_3d, &p0, &p1, &ths->q0, &ths->q1);
kernel/remap_3dto2d.c:  PX(split_cart_procmesh_for_3dto2d_remap_q1)(comm_cart_3d, &comm_q1);
kernel/remap_3dto2d.c:  PX(split_cart_procmesh_for_3dto2d_remap_q0)(comm_cart_3d, &comm_q0);
kernel/remap_3dto2d.c:  PX(get_procmesh_dims_2d)(comm_cart_3d, &p0, &p1, &q0, &q1);
kernel/remap_3dto2d.c:  PX(split_cart_procmesh)(comm_cart_3d, icomms);
kernel/remap_3dto2d.c:  PX(split_cart_procmesh_3dto2d_p0q0)(comm_cart_3d, &ocomms[0]);
kernel/remap_3dto2d.c:  PX(split_cart_procmesh_3dto2d_p1q1)(comm_cart_3d, &ocomms[1]);
kernel/remap_3dto2d.c:  PX(split_cart_procmesh_for_3dto2d_remap_q0)(comm_cart_3d, &mcomms[2]);
api/pfft.f03.in:    integer(C_INT) function pfft_create_procmesh(rnk_n,comm,np,comm_cart) bind(C, name='pfft_create_procmesh_f03')
api/pfft.f03.in:    end function pfft_create_procmesh
api/pfft.f03.in:    integer(C_INT) function pfft_create_procmesh_1d(comm,np0,comm_cart_1d) bind(C, name='pfft_create_procmesh_1d_f03')
api/pfft.f03.in:    end function pfft_create_procmesh_1d
api/pfft.f03.in:    integer(C_INT) function pfft_create_procmesh_2d(comm,np0,np1,comm_cart_2d) bind(C, name='pfft_create_procmesh_2d_f03')
api/pfft.f03.in:    end function pfft_create_procmesh_2d
api/pfft.f03.in:    integer(C_INT) function pfftf_create_procmesh(rnk_n,comm,np,comm_cart) bind(C, name='pfftf_create_procmesh_f03')
api/pfft.f03.in:    end function pfftf_create_procmesh
api/pfft.f03.in:    integer(C_INT) function pfftf_create_procmesh_1d(comm,np0,comm_cart_1d) bind(C, name='pfftf_create_procmesh_1d_f03')
api/pfft.f03.in:    end function pfftf_create_procmesh_1d
api/pfft.f03.in:    integer(C_INT) function pfftf_create_procmesh_2d(comm,np0,np1,comm_cart_2d) bind(C, name='pfftf_create_procmesh_2d_f03')
api/pfft.f03.in:    end function pfftf_create_procmesh_2d
api/api-adv.c:  PX(split_cart_procmesh)(comm_cart, comms_pm);
api/api-adv.c:  PX(get_procmesh_dims_2d)(comm_cart_3d, &p0, &p1, &q0, &q1);
api/fortran-prototypes.h:PFFT_VOIDFUNC FORT(create_procmesh, CREATE_PROCMESH)(
api/fortran-prototypes.h:PFFT_VOIDFUNC FORT(create_procmesh_2d, CREATE_PROCMESH_2D)(
api/fortran-wrappers.h:PFFT_VOIDFUNC FORT(create_procmesh, CREATE_PROCMESH)(
api/fortran-wrappers.h:  *ierror = PX(create_procmesh)(*rnk, MPI_Comm_f2c(*comm), np_rev, &comm_cart_clike);
api/fortran-wrappers.h:PFFT_VOIDFUNC FORT(create_procmesh_2d, CREATE_PROCMESH_2D)(
api/fortran-wrappers.h:  *ierror = PX(create_procmesh_2d)(MPI_Comm_f2c(*comm), *np3, *np2, &comm_cart_2d_clike);
api/f03-wrap.c:PFFT_EXTERN int PX(create_procmesh_f03)(int rnk_n, MPI_Fint f_comm, const int * np, MPI_Fint * f_comm_cart);
api/f03-wrap.c:PFFT_EXTERN int PX(create_procmesh_1d_f03)(MPI_Fint f_comm, int np0, MPI_Fint * f_comm_cart_1d);
api/f03-wrap.c:PFFT_EXTERN int PX(create_procmesh_2d_f03)(MPI_Fint f_comm, int np0, int np1, MPI_Fint * f_comm_cart_2d);
api/f03-wrap.c:int PX(create_procmesh_f03)(int rnk_n, MPI_Fint f_comm, const int * np, MPI_Fint * f_comm_cart)
api/f03-wrap.c:  int ret = PX(create_procmesh)(rnk_n, comm, np, &comm_cart);
api/f03-wrap.c:int PX(create_procmesh_1d_f03)(MPI_Fint f_comm, int np0, MPI_Fint * f_comm_cart_1d)
api/f03-wrap.c:  int ret = PX(create_procmesh_1d)(comm, np0, &comm_cart_1d);
api/f03-wrap.c:int PX(create_procmesh_2d_f03)(MPI_Fint f_comm, int np0, int np1, MPI_Fint * f_comm_cart_2d)
api/f03-wrap.c:  int ret = PX(create_procmesh_2d)(comm, np0, np1, &comm_cart_2d);
api/pfft.h:  PFFT_EXTERN int PX(create_procmesh)(                                                  \
api/pfft.h:  PFFT_EXTERN int PX(create_procmesh_1d)(                                               \
api/pfft.h:  PFFT_EXTERN int PX(create_procmesh_2d)(                                               \
api/api-basic.c:/* Global infos about procmesh are only enabled in debug mode
api/pfftl.f03.in:    integer(C_INT) function pfftl_create_procmesh(rnk_n,comm,np,comm_cart) bind(C, name='pfftl_create_procmesh_f03')
api/pfftl.f03.in:    end function pfftl_create_procmesh
api/pfftl.f03.in:    integer(C_INT) function pfftl_create_procmesh_1d(comm,np0,comm_cart_1d) bind(C, name='pfftl_create_procmesh_1d_f03')
api/pfftl.f03.in:    end function pfftl_create_procmesh_1d
api/pfftl.f03.in:    integer(C_INT) function pfftl_create_procmesh_2d(comm,np0,np1,comm_cart_2d) bind(C, name='pfftl_create_procmesh_2d_f03')
api/pfftl.f03.in:    end function pfftl_create_procmesh_2d
ChangeLog:	* kernel/procmesh.c: bugfix remap of 3d cart. comm. into 2d cart. comm
tests/simple_check_c2c_3d_on_3d_transposed.c:  if( pfft_create_procmesh(3, MPI_COMM_WORLD, np, &comm_cart_3d) ){
tests/simple_check_ousam_c2c_4d_transposed_newarray.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_r2c_3d_on_3d_transposed.c:  if( pfft_create_procmesh(3, MPI_COMM_WORLD, np, &comm_cart_3d) ){
tests/simple_check_ousam_r2c.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_ousam_c2r.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_r2c_padded_newarray.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_r2c_transposed_newarray.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/time_c2c_transposed.c:  if( pfft_create_procmesh_2d(comm, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_c2c_4d_transposed.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_ousam_c2c_4d_on_3d_transposed.c:  if( pfft_create_procmesh(3, MPI_COMM_WORLD, np, &comm_cart_3d) ){
tests/run_checks.sh:## Run tests with 2d procmesh
tests/run_checks.sh:## Run tests with 2d procmesh
tests/simple_check_r2r.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_r2c_newarray.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_r2r_3d_on_3d_transposed.c:  if( pfft_create_procmesh(3, MPI_COMM_WORLD, np, &comm_cart_3d) ){
tests/simple_check_r2r_4d_on_3d.c:  if( pfft_create_procmesh(3, MPI_COMM_WORLD, np, &comm_cart_3d) ){
tests/simple_check_ghost_c2c.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_c2c_4d_on_3d_transposed.c:  if( pfft_create_procmesh(3, MPI_COMM_WORLD, np, &comm_cart_3d) ){
tests/simple_check_r2c_4d_transposed.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_r2r_3d_on_3d.c:  if( pfft_create_procmesh(3, MPI_COMM_WORLD, np, &comm_cart_3d) ){
tests/simple_check_ousam_r2c_4d_newarray.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_ghost_r2c_input_padded.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_ousam_r2c_4d_transposed_newarray.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/f03/time_c2c.F03:  ierror =  pfft_create_procmesh_2d(MPI_COMM_WORLD, np(1), np(2), comm_cart_2d)
tests/f03/time_c2c_transposed.F03:  ierror =  pfft_create_procmesh_2d(MPI_COMM_WORLD, np(1), np(2), comm_cart_2d)
tests/f03/simple_check_r2c_4d_on_3d.F03:  ierror =  pfft_create_procmesh(3, MPI_COMM_WORLD, np, comm_cart_3d)
tests/f03/simple_check_ousam_c2c_4d_on_3d.F03:  ierror =  pfft_create_procmesh(3, MPI_COMM_WORLD, np, comm_cart_3d)
tests/f03/simple_check_r2c_4d_on_3d_transposed.F03:  ierror =  pfft_create_procmesh(3, MPI_COMM_WORLD, np, comm_cart_3d)
tests/f03/simple_check_ousam_c2c_4d_on_3d_transposed.F03:  ierror =  pfft_create_procmesh(3, MPI_COMM_WORLD, np, comm_cart_3d)
tests/f03/simple_check_c2c.F03:  ierror =  pfft_create_procmesh_2d(MPI_COMM_WORLD, np(1), np(2), comm_cart_2d)
tests/f03/simple_check_ousam_r2c_4d_on_3d_transposed.F03:  ierror =  pfft_create_procmesh(3, MPI_COMM_WORLD, np, comm_cart_3d)
tests/f03/simple_check_c2c_transposed.F03:  ierror =  pfft_create_procmesh_2d(MPI_COMM_WORLD, np(1), np(2), comm_cart_2d)
tests/f03/simple_check_ousam_r2c_4d_on_3d.F03:  ierror =  pfft_create_procmesh(3, MPI_COMM_WORLD, np, comm_cart_3d)
tests/simple_check_r2c_4d.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_c2c.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_r2c_3d_on_3d_transposed_many.c:  if( pfft_create_procmesh(3, MPI_COMM_WORLD, np, &comm_cart_3d) ){
tests/bench_c2c.c:  if( pfft_create_procmesh(3, MPI_COMM_WORLD, np, &comm_cart_3d) ){
tests/bench_c2c.c:    pfft_fprintf(MPI_COMM_WORLD, stderr, "Error: Procmesh of size %d x %d x %d does not fit to number of allocated processes.\n", np[0], np[1], np[2]);
tests/bench_c2c.c:    pfft_fprintf(MPI_COMM_WORLD, stderr, "       Please allocate %d processes (mpiexec -np %d ...) or change the procmesh (with -pfft_np * * *).\n", np[0]*np[1]*np[2], np[0]*np[1]*np[2]);
tests/bench_c2c.c:      if( pfft_create_procmesh(2, MPI_COMM_WORLD, np, &comm_cart_2d) )
tests/bench_c2c.c:        pfft_printf(MPI_COMM_WORLD, "Error in creation of 2d procmesh of size %d x %d\n", np[0], np[1]);
tests/bench_c2c.c:    if( pfft_create_procmesh(1, MPI_COMM_WORLD, np, &comm_cart_1d) )
tests/bench_c2c.c:      pfft_printf(MPI_COMM_WORLD, "Error in creation of 2d procmesh of size %d\n", np[0]);
tests/fortran/simple_check_ousam_r2c.F90:      call dpfft_create_procmesh_2d(ierror, MPI_COMM_WORLD, &
tests/fortran/simple_check_r2c_3d_on_3d_transposed.F90:      call dpfft_create_procmesh(ierror, 3, MPI_COMM_WORLD, &
tests/fortran/simple_check_ghost_c2c_3d_on_3d.F90:      call dpfft_create_procmesh(ierror, 3, MPI_COMM_WORLD, np,&
tests/fortran/simple_check_c2c_transposed.F90:      call dpfft_create_procmesh_2d(ierror, MPI_COMM_WORLD, &
tests/fortran/simple_check_ghost_c2c.F90:      call dpfft_create_procmesh_2d(ierror, MPI_COMM_WORLD, &
tests/fortran/simple_check_ousam_c2c_transposed.F90:      call dpfft_create_procmesh_2d(ierror, MPI_COMM_WORLD, &
tests/fortran/simple_check_r2r.F90:      call dpfft_create_procmesh_2d(ierror, MPI_COMM_WORLD, &
tests/fortran/simple_test.F90:      call dpfft_create_procmesh_2d(ierror, MPI_COMM_WORLD, &
tests/fortran/simple_check_ousam_c2c.F90:      call dpfft_create_procmesh_2d(ierror, MPI_COMM_WORLD, &
tests/fortran/simple_check_ousam_r2r.F90:      call dpfft_create_procmesh_2d(ierror, MPI_COMM_WORLD, &
tests/fortran/simple_check_r2r_transposed.F90:      call dpfft_create_procmesh_2d(ierror, MPI_COMM_WORLD, &
tests/fortran/simple_check_ousam_r2r_transposed.F90:      call dpfft_create_procmesh_2d(ierror, MPI_COMM_WORLD, &
tests/fortran/simple_check_r2c.F90:      call dpfft_create_procmesh_2d(ierror, MPI_COMM_WORLD, &
tests/fortran/simple_check_c2c.F90:      call dpfft_create_procmesh_2d(ierror, MPI_COMM_WORLD, &
tests/fortran/simple_check_ousam_r2c_transposed.F90:      call dpfft_create_procmesh_2d(ierror, MPI_COMM_WORLD, &
tests/fortran/simple_check_r2c_transposed.F90:      call dpfft_create_procmesh_2d(ierror, MPI_COMM_WORLD, &
tests/fortran/simple_check_r2c_3d_on_3d.F90:      call dpfft_create_procmesh(ierror, 3, MPI_COMM_WORLD, &
tests/simple_check_r2r_4d.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_r2c.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/openmp/simple_check_c2c.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/openmp/simple_check_c2c_omp.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/openmp/omp_bench_c2c.c:  if( pfft_create_procmesh(3, MPI_COMM_WORLD, np, &comm_cart_3d) ){
tests/openmp/omp_bench_c2c.c:    pfft_fprintf(MPI_COMM_WORLD, stderr, "Error: Procmesh of size %d x %d x %d does not fit to number of allocated processes.\n", np[0], np[1], np[2]);
tests/openmp/omp_bench_c2c.c:    pfft_fprintf(MPI_COMM_WORLD, stderr, "       Please allocate %d processes (mpiexec -np %d ...) or change the procmesh (with -pfft_np * * *).\n", np[0]*np[1]*np[2], np[0]*np[1]*np[2]);
tests/openmp/omp_bench_c2c.c:      if( pfft_create_procmesh(2, MPI_COMM_WORLD, np, &comm_cart_2d) )
tests/openmp/omp_bench_c2c.c:        pfft_printf(MPI_COMM_WORLD, "Error in creation of 2d procmesh of size %d x %d\n", np[0], np[1]);
tests/openmp/omp_bench_c2c.c:    if( pfft_create_procmesh(1, MPI_COMM_WORLD, np, &comm_cart_1d) )
tests/openmp/omp_bench_c2c.c:      pfft_printf(MPI_COMM_WORLD, "Error in creation of 2d procmesh of size %d\n", np[0]);
tests/openmp/simple_check_c2c_transposed.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_c2c_inplace.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_c2r_c2c.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_ousam_r2c_4d_on_3d.c:  if( pfft_create_procmesh(3, MPI_COMM_WORLD, np, &comm_cart_3d) ){
tests/simple_check_ousam_r2c_4d_on_3d_transposed.c:  if( pfft_create_procmesh(3, MPI_COMM_WORLD, np, &comm_cart_3d) ){
tests/simple_check_ousam_c2c_4d.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_ousam_r2c_transposed.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/minimal_check_c2c_transposed.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/scale_c2c.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/scale_c2c.c:    pfft_fprintf(MPI_COMM_WORLD, stderr, "Error: Procmesh %d x %d requires MPI launch with %d processes.\n",
tests/simple_check_ousam_c2c_transposed.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_ousam_c2c.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_r2r_4d_on_3d_transposed.c:  if( pfft_create_procmesh(3, MPI_COMM_WORLD, np, &comm_cart_3d) ){
tests/simple_check_c2r_c2c_shifted.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/time_c2c.c:  if( pfft_create_procmesh_2d(comm, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_c2c_4d.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/scale_c2c_1d.c:  if( pfft_create_procmesh(1, MPI_COMM_WORLD, np, &comm_cart_1d) ){
tests/scale_c2c_1d.c:    pfft_fprintf(MPI_COMM_WORLD, stderr, "Error: Procmesh %d x %d requires MPI launch with %d processes.\n",
tests/scale_c2c_1d.c:    pfft_fprintf(MPI_COMM_WORLD, stderr, "Error: Procmesh %d x %d is not one-dimensional.\n",
tests/simple_check_ousam_r2r.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_r2r_transposed.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_ghost_r2c_input.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_c2c_transposed_inplace.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_c2c_newarray.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_r2c_transposed.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_r2r_4d_transposed.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_c2c_4d_on_3d.c:  if( pfft_create_procmesh(3, MPI_COMM_WORLD, np, &comm_cart_3d) ){
tests/simple_check_ousam_c2c_4d_on_3d.c:  if( pfft_create_procmesh(3, MPI_COMM_WORLD, np, &comm_cart_3d) ){
tests/simple_check_r2c_3d_on_3d.c:  if( pfft_create_procmesh(3, MPI_COMM_WORLD, np, &comm_cart_3d) ){
tests/debug_simple_check_ghost_c2c_3d_on_3d.c:  if( pfft_create_procmesh(3, MPI_COMM_WORLD, np, &comm_cart_3d) ){
tests/simple_check_ghost_c2c_3d_on_3d.c:  if( pfft_create_procmesh(3, MPI_COMM_WORLD, np, &comm_cart_3d) ){
tests/simple_check_c2c_transposed.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_ousam_r2c_padded.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_c2c_transposed_newarray.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/minimal_check_c2c.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_r2c_4d_on_3d.c:  if( pfft_create_procmesh(3, MPI_COMM_WORLD, np, &comm_cart_3d) ){
tests/simple_check_r2c_4d_on_3d_transposed.c:  if( pfft_create_procmesh(3, MPI_COMM_WORLD, np, &comm_cart_3d) ){
tests/simple_check_r2c_padded_transposed_newarray.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_ousam_r2r_transposed.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_c2c_ldouble.c:  if( pfftl_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_ghost_r2c_output.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/debug_simple_check_r2c_3d_on_3d_transposed_many.c:  if( pfft_create_procmesh(3, MPI_COMM_WORLD, np, &comm_cart_3d) ){
tests/adv_check_ghost_c2c.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_c2r_c2c_ousam_shifted.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_c2c_3d_on_3d.c:  if( pfft_create_procmesh(3, MPI_COMM_WORLD, np, &comm_cart_3d) ){
tests/manual_c2c_3d.c:  pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1],
tests/simple_check_ousam_r2c_4d.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_ousam_c2c_4d_transposed.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_ousam_r2c_4d_transposed.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_ousam_c2c_4d_newarray.c:  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
tests/simple_check_c2c_float.c:  if( pfftf_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
doc/reference.tex:int pfft_create_procmesh_1d(
doc/reference.tex:int pfft_create_procmesh_2d(
doc/reference.tex:int pfft_create_procmesh(
doc/shortcuts.tex:% 	              pfft_create_procmesh_2d, pfft_create_procmesh,
doc/tutorial.tex:int pfft_create_procmesh_2d(
doc/tutorial.tex:to free the communicator allocated by \code{pfft_create_procmesh_2d} and
doc/tutorial.tex:    (red@pfft_create_procmesh_2d(MPI_COMM_WORLD, np0, np1,@*)
doc/tutorial.tex:int pfft_create_procmesh_2d(
doc/tutorial.tex:if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1],
doc/tutorial.tex:int pfft_create_procmesh(
doc/intro.tex:  \item PFFT does not support GPU parallelization.
gcell/gcells_plan.c:  /* procmesh remains the same for transposed layout */

```
