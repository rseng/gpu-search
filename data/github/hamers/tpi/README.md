# https://github.com/hamers/tpi

```console
input/S.txt:==OpenMP__________OMPnumthread____OMPminN_________resume__________N_resume________GPUminN_________
input/run1.txt:==OpenMP__________OMPnumthread____OMPminN_________resume__________N_resume________GPUminN_________
tpi.cpp:#ifdef USE_GPU
tpi.cpp:  #ifdef USE_GPU
tpi.cpp:    //cout << "Sapporo 2 is enabled; using GPU if N_tp_calc > " << parameters.minimum_N_tp_calc_for_GPU << endl;
tpi.cpp:  #ifdef USE_GPU
tpi.cpp:      cout << "open GPU" << endl;
tpi.cpp:    GPU_set_j_particles(&fp_data,&parameters);
tpi.cpp:      cout << "open GPU - done" << endl;
tpi.cpp:  #ifdef USE_GPU
tpi.cpp:    #ifdef USE_GPU
tpi.cpp:  #ifdef USE_GPU
tpi.cpp:      timespec start_GPU,end_GPU;
tpi.cpp:      clock_gettime(CLOCK_MONOTONIC,&start_GPU);        
tpi.cpp:      clock_gettime(CLOCK_MONOTONIC,&end_GPU);
tpi.cpp:      parameters->execution_times.allocate_arrays_for_sapporo += elapsed_time(&start_GPU,&end_GPU);
tpi.cpp:      clock_gettime(CLOCK_MONOTONIC,&start_GPU);
tpi.cpp:    get_fp_gravity_GPU_start(N_tp_calc,tp_calc,fp_data,parameters,ids,r,v,h2,pot,a_fp_on_tp,j_fp_on_tp);
tpi.cpp:  #ifdef USE_GPU
tpi.cpp:    get_fp_gravity_GPU_retrieve(N_tp_calc,tp_calc,fp_data,parameters,ids,r,v,h2,pot,a_fp_on_tp,j_fp_on_tp);
tpi.cpp:      clock_gettime(CLOCK_MONOTONIC,&end_GPU);
tpi.cpp:      parameters->execution_times.gg_fp_GPU += elapsed_time(&start_GPU,&end_GPU);
tpi.cpp:#ifdef USE_GPU
tpi.cpp:void GPU_set_j_particles(std::vector<fp_data_t> *fp_data, parameters_t *parameters)
tpi.cpp:    cout << "GPU_set_j_particles" << endl;
tpi.cpp:    parameters->execution_times.GPU_set_j_particles += elapsed_time(&start,&end);
tpi.cpp:    cout << "GPU_set_j_particles - done" << endl;
tpi.cpp:void get_fp_gravity_GPU_start(int N_tp_calc,std::vector<tp_calc_t> *tp_calc,std::vector<fp_data_t> *fp_data,parameters_t *parameters, int *ids, double (*r)[3], double (*v)[3], double *h2, double *pot, double (*a_fp_on_tp)[3], double (*j_fp_on_tp)[3])
tpi.cpp:    cout << "fp_gravity_GPU_start" << endl;
tpi.cpp:    //cout << "fp_gravity_GPU_start r " << r[i_tp_calc][0] << " " << r[i_tp_calc][1] << " " << r[i_tp_calc][2] << endl;
tpi.cpp:    //cout << "fp_gravity_GPU_start ids " << ids[i_tp_calc] << endl;
tpi.cpp:    //cout << "fp_gravity_GPU_start h2 " << h2[i_tp_calc] << endl;
tpi.cpp:    //cout << "fp_gravity_GPU_start pot " << pot[i_tp_calc] << endl;    
tpi.cpp:    cout << "fp_gravity_GPU_start - done" << endl;
tpi.cpp:void get_fp_gravity_GPU_retrieve(int N_tp_calc,std::vector<tp_calc_t> *tp_calc,std::vector<fp_data_t> *fp_data,parameters_t *parameters, int *ids, double (*r)[3], double (*v)[3], double *h2, double *pot, double (*a_fp_on_tp)[3], double (*j_fp_on_tp)[3])
tpi.cpp:    cout << "fp_gravity_GPU_retrieve" << endl;
tpi.cpp:    cout << "fp_gravity_GPU_retrieve - done" << endl;
README.md:A code to compute the gravitational dynamics of particles orbiting a supermassive black hole (SBH). A distinction is made to two types of particles: test particles and field particles. Field particles are assumed to move in quasi-static Keplerian orbits around the SBH that precess due to the enclosed mass (Newtonian `mass precession') and relativistic effects. Otherwise, field-particle-field-particle interactions are neglected. Test particles are integrated in the time-dependent potential of the field particles and the SBH. Relativistic effects are included in the equations of motion (including the effects of SBH spin), and test-particle-test-particle interactions are neglected. Supports OpenMP; legacy GPU support (not tested recently).
helper_routines.cpp:  minimum_N_tp_calc_for_GPU = 50;
helper_routines.cpp:          if (i==0) iss >> parameters->enable_OpenMP; if (i==1) iss >> parameters->OpenMP_max_threads; if (i==2) iss >> parameters->minimum_N_for_OpenMP; if (i==3) iss >> parameters->enable_resume_functionality; if (i==4) iss >> parameters->N_resume; if (i==5) iss >> parameters->minimum_N_tp_calc_for_GPU;
helper_routines.cpp:  cout << "GPU_set_j_particles: " << parameters->execution_times.GPU_set_j_particles << endl;
helper_routines.cpp:  cout <<  parameters->execution_times.GPU_set_j_particles << " " << parameters->execution_times.GPU_set_j_particles/parameters->execution_times.total << "\t GPU_set_j_particles" << endl;
helper_routines.cpp:  cout <<  parameters->execution_times.gg_fp_GPU << " " << parameters->execution_times.gg_fp_GPU/parameters->execution_times.total << "\t gg_fp_GPU" << endl;  
helper_routines.cpp:    + parameters->execution_times.predict + parameters->execution_times.correct + parameters->execution_times.evaluate + parameters->execution_times.GPU_set_j_particles \
helper_routines.cpp:    + parameters->execution_times.allocate_arrays_for_sapporo + parameters->execution_times.gg_fp_CPU + parameters->execution_times.gg_fp_GPU + parameters->execution_times.gg_PN_acc \
helper_routines.h:  double total,shift_fp,rotate_fp,determine_i_tp_calc,predict,correct,evaluate,GPU_set_j_particles,allocate_arrays_for_sapporo,gg_fp_CPU,gg_fp_GPU,gg_PN_acc,gg_PN_jerk;
helper_routines.h:  int OpenMP_max_threads,minimum_N_for_OpenMP,minimum_N_tp_calc_for_GPU,N_resume,resume_loop_index;
tpi.h://#define USE_GPU
tpi.h:void GPU_set_j_particles(std::vector<fp_data_t> *fp_data, parameters_t *parameters);
tpi.h:void get_fp_gravity_GPU_start(int N_tp_calc,std::vector<tp_calc_t> *tp_calc,std::vector<fp_data_t> *fp_data,parameters_t *parameters, int *ids, double (*r)[3], double (*v)[3], double *h2, double *pot, double (*a_fp_on_tp)[3], double (*j_fp_on_tp)[3]);
tpi.h:void get_fp_gravity_GPU_retrieve(int N_tp_calc,std::vector<tp_calc_t> *tp_calc,std::vector<fp_data_t> *fp_data,parameters_t *parameters, int *ids, double (*r)[3], double (*v)[3], double *h2, double *pot, double (*a_fp_on_tp)[3], double (*j_fp_on_tp)[3]);

```
