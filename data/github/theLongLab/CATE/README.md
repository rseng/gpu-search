# https://github.com/theLongLab/CATE

```console
fu_li.cu:fu_li::fu_li(string gene_List, string input_Folder, string ouput_Path, int cuda_ID, string intermediate_Path, int ploidy)
fu_li.cu:    cout << "Initiating CUDA powered Fu and Li's D, D*, F and F* calculator" << endl
fu_li.cu:    set_Values(gene_List, input_Folder, ouput_Path, cuda_ID, intermediate_Path, ploidy);
fu_li.cu:fu_li::fu_li(string gene_List, string input_Folder, string ouput_Path, int cuda_ID, string intermediate_Path, int ploidy, string prometheus_Activate, string Multi_read, int number_of_genes, int CPU_cores, int SNPs_per_Run)
fu_li.cu:    cout << "Initiating CUDA powered Fu and Li's D, D*, F and F* calculator on PROMETHEUS" << endl
fu_li.cu:    set_Values(gene_List, input_Folder, ouput_Path, cuda_ID, intermediate_Path, ploidy);
fu_li.cu:fu_li::fu_li(string calc_Mode, int window_Size, int step_Size, string input_Folder, string ouput_Path, int cuda_ID, int ploidy, string prometheus_Activate, string Multi_read, int number_of_genes, int CPU_cores, int SNPs_per_Run)
fu_li.cu:    cout << "Initiating CUDA powered Fu and Li's D, D*, F and F* calculator on PROMETHEUS" << endl
fu_li.cu:    set_Values("", input_Folder, ouput_Path, cuda_ID, "", ploidy);
fu_li.cu:fu_li::fu_li(string calc_Mode, int window_Size, int step_Size, string input_Folder, string ouput_Path, int cuda_ID, int ploidy)
fu_li.cu:    cout << "Initiating CUDA powered Fu and Li's D, D*, F and F* calculator" << endl
fu_li.cu:    set_Values("", input_Folder, ouput_Path, cuda_ID, "", ploidy);
fu_li.cu:void fu_li::set_Values(string gene_List, string input_Folder, string ouput_Path, int cuda_ID, string intermediate_Path, int ploidy)
fu_li.cu:     * Here the first call to the selected CUDA device occurs.
fu_li.cu:    cudaSetDevice(cuda_ID);
fu_li.cu:    cout << "Properties of selected CUDA GPU:" << endl;
fu_li.cu:    cudaDeviceProp prop;
fu_li.cu:    cudaGetDeviceProperties(&prop, cuda_ID);
fu_li.cu:    cout << "GPU number\t: " << cuda_ID << endl;
fu_li.cu:    cout << "GPU name\t: " << prop.name << endl;
fu_li.cu:    cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);
fu_li.cu:    cout << "GPU memory (GB)\t: " << l_Total / (1000 * 1000 * 1000) << endl;
fu_li.cu:    cout << "GPU number of multiprocessor(s)\t: " << prop.multiProcessorCount << endl;
fu_li.cu:    cout << "GPU block(s) per multiprocessor\t: " << prop.maxBlocksPerMultiProcessor << endl;
fu_li.cu:    cout << "GPU thread(s) per block\t: " << tot_ThreadsperBlock << endl
fu_li.cu:                                         * Information from the SNP is extracted via the GPU.
fu_li.cu:__global__ void fuli_Calculation(int N, float *a1_CUDA, float *a2_CUDA)
fu_li.cu:        a1_CUDA[tid] = (float)1 / (tid + 1);
fu_li.cu:        a2_CUDA[tid] = (float)1 / ((tid + 1) * (tid + 1));
fu_li.cu:    float *a1_CUDA, *a2_CUDA;
fu_li.cu:    cudaMallocManaged(&a1_CUDA, N * sizeof(int));
fu_li.cu:    cudaMallocManaged(&a2_CUDA, N * sizeof(int));
fu_li.cu:    fuli_Calculation<<<tot_Blocks, tot_ThreadsperBlock>>>(N, a1_CUDA, a2_CUDA);
fu_li.cu:    cudaDeviceSynchronize();
fu_li.cu:    cudaMemcpy(a1_partial, a1_CUDA, N * sizeof(float), cudaMemcpyDeviceToHost);
fu_li.cu:    cudaMemcpy(a2_partial, a2_CUDA, N * sizeof(float), cudaMemcpyDeviceToHost);
fu_li.cu:    cudaFree(a1_CUDA);
fu_li.cu:    cudaFree(a2_CUDA);
node_within_host.cuh:#include "cuda_runtime.h"
node_within_host.cuh:#include <thrust/system/cuda/error.h>
node_within_host.cuh:    void run_Generation(functions_library &functions, string &multi_Read, int &max_Cells_at_a_time, int &gpu_Limit, int *CUDA_device_IDs, int &num_Cuda_devices, int &genome_Length,
node_within_host.cuh:    void simulate_Cell_replication(functions_library &functions, string &multi_Read, int &gpu_Limit, int *CUDA_device_IDs, int &num_Cuda_devices, string &source_sequence_Data_folder, vector<pair<int, int>> &indexed_Tissue_Folder,
node_within_host.cuh:                                             vector<string> &collected_Sequences, int *CUDA_device_IDs, int &num_Cuda_devices, int &genome_Length,
node_within_host.cuh:                              float **cuda_sequence_Configuration_standard, int recombination_Hotspots,
node_within_host.cuh:                              int **progeny_Configuration, int *cuda_progeny_Stride, int progeny_Total, int remove_Back);
node_within_host.cuh:                            int *CUDA_device_IDs, int &num_Cuda_devices,
neutral.cuh:#include "cuda_runtime.h"
neutral.cuh:     * @param tot_Blocks defines number of GPU blocks that are available
neutral.cuh:     * @param tot_ThreadsperBlock defines number of threads that are available per GPU block.
neutral.cuh:     * @param SNPs_per_Run defines the number of SNP sites the GPU will process at a time.
neutral.cuh:    neutral(string gene_List, string input_Folder, string output_Path, int cuda_ID, string intermediate_Path, int ploidy);
neutral.cuh:    neutral(string calc_Mode, int window_Size, int step_Size, string input_Folder, string output_Path, int cuda_ID, int ploidy);
neutral.cuh:    neutral(string gene_List, string input_Folder, string output_Path, int cuda_ID, string intermediate_Path, int ploidy, string prometheus_Activate, string Multi_read, int number_of_genes, int CPU_cores, int SNPs_per_Run);
neutral.cuh:    neutral(string calc_Mode, int window_Size, int step_Size, string input_Folder, string output_Path, int cuda_ID, int ploidy, string prometheus_Activate, string Multi_read, int number_of_genes, int CPU_cores, int SNPs_per_Run);
neutral.cuh:    void set_Values(string gene_List, string input_Folder, string ouput_Path, int cuda_ID, string intermediate_Path, int ploidy);
neutral.cuh:     * Function directly calls upon the GPU function.
segmatch.cuh:#include "cuda_runtime.h"
segmatch.cuh:#include <thrust/system/cuda/error.h>
parameters.json:    # Cuda device
parameters.json:    "CUDA Device ID":0,
parameters.json:    "Split SNPs per_time_GPU":5000,
cudaDevices.cu:#include "cudaDevices.cuh"
cudaDevices.cu:cudaDevices::cudaDevices()
cudaDevices.cu:     * Prints all CUDA devices.
cudaDevices.cu:    cout << "Listing all CUDA capable devices:" << endl;
cudaDevices.cu:    cudaGetDeviceCount(&nDevices);
cudaDevices.cu:         * Once a CUDA enabled device has been detected its details will be printed.
cudaDevices.cu:         * NVIDIA's CUDA toolkit is properly communicating with the query device.
cudaDevices.cu:        cudaDeviceProp prop;
cudaDevices.cu:        cudaGetDeviceProperties(&prop, i);
cudaDevices.cu:        cout << "GPU number\t: " << i << endl;
cudaDevices.cu:        cout << "GPU name\t: " << prop.name << endl;
cudaDevices.cu:        cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);
cudaDevices.cu:        cout << "GPU memory (GB)\t: " << l_Total / (1000 * 1000 * 1000) << endl;
cudaDevices.cu:        cout << "GPU number of multiprocessor(s)\t: " << prop.multiProcessorCount << endl;
cudaDevices.cu:        cout << "GPU block(s) per multiprocessor\t: " << prop.maxBlocksPerMultiProcessor << endl;
cudaDevices.cu:        cout << "GPU thread(s) per block\t: " << prop.maxThreadsPerBlock << endl;
cudaDevices.cu:         * If there is an ERROR in the execution of the CUDA device the error will be printed.
cudaDevices.cu:        cudaError_t err = cudaGetLastError();
cudaDevices.cu:        if (err != cudaSuccess)
cudaDevices.cu:            printf("CUDA Error: %s\n", cudaGetErrorString(err));
ehh.cuh:#include "cuda_runtime.h"
ehh.cuh:    ehh(string range_Mode, string file_Mode_path, string fixed_Mode_value, string input_Folder, string output_Path, int cuda_ID, string intermediate_Path, int ploidy, int default_SNP_count, int EHH_CPU_cores, int default_SNP_BP_count);
Test_main.cpp:#include "cudaDevices.cuh"
Test_main.cpp:#include <cuda.h>
Test_main.cpp:#include "cuda_runtime.h"
Test_main.cpp:     cout << "CATE: CUDA Accelerated Testing of Evolution" << endl;
Test_main.cpp:          << "CATE: A fast and scalable CUDA implementation to conduct highly parallelized evolutionary tests on large scale genomic data.\n"
Test_main.cpp:      * C.A.T.E. stands for CUDA Accelerated Testing of Evolution.
Test_main.cpp:      * It is a software designed with the aim of utilizing the computer's multiprocessing technologies of both the CPU and GPU.
Test_main.cpp:      * ! At present, CATE is compatible only with CUDA enabled NVIDIA GPU's.
Test_main.cpp:          else if (function == "--cuda" || function == "-c")
Test_main.cpp:                * Prints all available CUDA devices present on the current system.
Test_main.cpp:                * User can use this list to determine which CUDA device to be used via the CUDA ID.
Test_main.cpp:               cudaDevices cudaList = cudaDevices();
Test_main.cpp:               cout << "All CUDA capable devices have been listed" << endl;
Test_main.cpp:                                   // vcf_splitter_2(string input_vcf_Folder, string output_Folder, int cores, int SNPs_per_time_CPU, int SNPs_per_time_GPU, int allele_Count_REF, int allele_Count_ALT);
Test_main.cpp:                                   vcf_splitter_2 split_CHR = vcf_splitter_2(properties.where_Int("CUDA Device ID"), properties.where("Input path"), output_Path, properties.where_Int("Split cores"), properties.where_Int("Split SNPs per_time_CPU"), properties.where_Int("Split SNPs per_time_GPU"), properties.where_Int("Reference allele count"), properties.where_Int("Alternate allele count"), properties.where_Int("Ploidy"), summary);
Test_main.cpp:                                        vcf_splitter_2 CATE_split = vcf_splitter_2(properties.where_Int("CUDA Device ID"), properties.where("Input path"), output_Path, properties.where("Population file path"), properties.where_Int("Sample_ID Column number"), properties.where_Int("Population_ID Column number"), properties.where_Int("Split cores"), properties.where_Int("Split SNPs per_time_CPU"), properties.where_Int("Split SNPs per_time_GPU"), properties.where_Int("Ploidy"), properties.where_Int("SNP count per file"), MAF_logic, stod(MAF));
Test_main.cpp:                              hap_extract haplotype_Extractor = hap_extract(gene_List, properties.where("Input path"), output_Path, properties.where_Int("CUDA Device ID"), intermediate_Path, properties.where_Int("Ploidy"), properties.where("Reference genome hap"), properties.where("Population out"));
Test_main.cpp:                                   << "Completed CUDA powered Haplotype extractor." << endl;
Test_main.cpp:                                        tajima tajimasD = tajima(gene_List, properties.where("Input path"), output_Path, properties.where_Int("CUDA Device ID"), intermediate_Path, properties.where_Int("Ploidy"));
Test_main.cpp:                                        tajima tajimasD_Window = tajima(calc_Mode, properties.where_Int("Window size"), properties.where_Int("Step size"), properties.where("Input path"), output_Path, properties.where_Int("CUDA Device ID"), properties.where_Int("Ploidy"));
Test_main.cpp:                                        tajima tajimasD_Prometheus = tajima(gene_List, properties.where("Input path"), output_Path, properties.where_Int("CUDA Device ID"), intermediate_Path, properties.where_Int("Ploidy"), prometheus_Activate, properties.where("Multi read"), properties.where_Int("Number of genes"), properties.where_Int("CPU cores"), properties.where_Int("SNPs per time"));
Test_main.cpp:                                        tajima tajimasD_Prometheus_Window = tajima(calc_Mode, properties.where_Int("Window size"), properties.where_Int("Step size"), properties.where("Input path"), output_Path, properties.where_Int("CUDA Device ID"), properties.where_Int("Ploidy"), prometheus_Activate, properties.where("Multi read"), properties.where_Int("Number of genes"), properties.where_Int("CPU cores"), properties.where_Int("SNPs per time"));
Test_main.cpp:                                   << "CUDA powered Tajima's D calculator has been completed." << endl;
Test_main.cpp:                                        fu_li fuli = fu_li(gene_List, properties.where("Input path"), output_Path, properties.where_Int("CUDA Device ID"), intermediate_Path, properties.where_Int("Ploidy"));
Test_main.cpp:                                        fu_li fuli_Window = fu_li(calc_Mode, properties.where_Int("Window size"), properties.where_Int("Step size"), properties.where("Input path"), output_Path, properties.where_Int("CUDA Device ID"), properties.where_Int("Ploidy"));
Test_main.cpp:                                        fu_li fuli_Prometheus = fu_li(gene_List, properties.where("Input path"), output_Path, properties.where_Int("CUDA Device ID"), intermediate_Path, properties.where_Int("Ploidy"), prometheus_Activate, properties.where("Multi read"), properties.where_Int("Number of genes"), properties.where_Int("CPU cores"), properties.where_Int("SNPs per time"));
Test_main.cpp:                                        fu_li fuli_Prometheus_Window = fu_li(calc_Mode, properties.where_Int("Window size"), properties.where_Int("Step size"), properties.where("Input path"), output_Path, properties.where_Int("CUDA Device ID"), properties.where_Int("Ploidy"), prometheus_Activate, properties.where("Multi read"), properties.where_Int("Number of genes"), properties.where_Int("CPU cores"), properties.where_Int("SNPs per time"));
Test_main.cpp:                              cout << "CUDA powered Fu and Li's D, D*, F and F* calculator has been completed." << endl;
Test_main.cpp:                                        fay_wu faywu = fay_wu(gene_List, properties.where("Input path"), output_Path, properties.where_Int("CUDA Device ID"), intermediate_Path, properties.where_Int("Ploidy"));
Test_main.cpp:                                        fay_wu faywu_Window = fay_wu(calc_Mode, properties.where_Int("Window size"), properties.where_Int("Step size"), properties.where("Input path"), output_Path, properties.where_Int("CUDA Device ID"), properties.where_Int("Ploidy"));
Test_main.cpp:                                        fay_wu faywu_Prometheus = fay_wu(gene_List, properties.where("Input path"), output_Path, properties.where_Int("CUDA Device ID"), intermediate_Path, properties.where_Int("Ploidy"), prometheus_Activate, properties.where("Multi read"), properties.where_Int("Number of genes"), properties.where_Int("CPU cores"), properties.where_Int("SNPs per time"));
Test_main.cpp:                                        fay_wu faywu_Prometheus_Window = fay_wu(calc_Mode, properties.where_Int("Window size"), properties.where_Int("Step size"), properties.where("Input path"), output_Path, properties.where_Int("CUDA Device ID"), properties.where_Int("Ploidy"), prometheus_Activate, properties.where("Multi read"), properties.where_Int("Number of genes"), properties.where_Int("CPU cores"), properties.where_Int("SNPs per time"));
Test_main.cpp:                              cout << "CUDA powered Fay and Wu's normalized H and E calculator has been completed." << endl;
Test_main.cpp:                                        neutral neutrality = neutral(gene_List, properties.where("Input path"), output_Path, properties.where_Int("CUDA Device ID"), intermediate_Path, properties.where_Int("Ploidy"));
Test_main.cpp:                                        neutral neutrality_Window = neutral(calc_Mode, properties.where_Int("Window size"), properties.where_Int("Step size"), properties.where("Input path"), output_Path, properties.where_Int("CUDA Device ID"), properties.where_Int("Ploidy"));
Test_main.cpp:                                        neutral neutrality_Prometheus = neutral(gene_List, properties.where("Input path"), output_Path, properties.where_Int("CUDA Device ID"), intermediate_Path, properties.where_Int("Ploidy"), prometheus_Activate, properties.where("Multi read"), properties.where_Int("Number of genes"), properties.where_Int("CPU cores"), properties.where_Int("SNPs per time"));
Test_main.cpp:                                        neutral neutrality_Prometheus = neutral(calc_Mode, properties.where_Int("Window size"), properties.where_Int("Step size"), properties.where("Input path"), output_Path, properties.where_Int("CUDA Device ID"), properties.where_Int("Ploidy"), prometheus_Activate, properties.where("Multi read"), properties.where_Int("Number of genes"), properties.where_Int("CPU cores"), properties.where_Int("SNPs per time"));
Test_main.cpp:                              cout << "CUDA powered complete neutrality test calculator has completed" << endl;
Test_main.cpp:                              mk_test mk = mk_test(properties.where("Reference genome mk"), properties.where("Alignment file"), gene_List, properties.where("Input path"), output_Path, properties.where_Int("CUDA Device ID"), intermediate_Path, properties.where_Int("Ploidy"), properties.where("Genetic code"), properties.where("Start codon(s)"), properties.where("Stop codon(s)"), properties.where("Alignment mode"), properties.where("ORF known"));
Test_main.cpp:                              cout << "CUDA powered McDonald–Kreitman Neutrality Index (NI) test has been completed." << endl;
Test_main.cpp:                                   fst fs = fst(gene_List, properties.where("Input path"), output_Path, properties.where_Int("CUDA Device ID"), intermediate_Path, properties.where_Int("Ploidy"), properties.where("Population index file path"), properties.where("Population ID"));
Test_main.cpp:                                   // fst(string calc_Mode, int window_Size, int step_Size, string gene_List, string input_Folder, string output_Path, int cuda_ID, int ploidy, string pop_Index_path, string pop_List);
Test_main.cpp:                                   fst fst_Window = fst(calc_Mode, properties.where_Int("Window size"), properties.where_Int("Step size"), properties.where("Input path"), output_Path, properties.where_Int("CUDA Device ID"), properties.where_Int("Ploidy"), properties.where("Population index file path"), properties.where("Population ID"));
Test_main.cpp:                              cout << "CUDA powered Fst (Fixation Index) calculator has been completed." << endl;
Test_main.cpp:                                   ehh ehh_ = ehh(mode, file_mode_Path, fixed_mode_Value, properties.where("Input path"), output_Path, properties.where_Int("CUDA Device ID"), intermediate_Path, properties.where_Int("Ploidy"), default_SNP_count, EHH_CPU_cores, default_SNP_BP_count);
Test_main.cpp:                                   cout << "CUDA powered Extended Haplotype Homozygosity (EHH) calculator has been completed." << endl;
Test_main.cpp:          << "PROMETHEUS uses a CUDA powered engine, therefore, requires a CUDA capable GPU." << endl
Test_main.cpp:          << "3. SNPs per time      : Controls the max number of SNPs that will be processed on the GPU at a time." << endl
Test_main.cpp:          << "                          Uses a CUDA powered engine, therefore, requires a CUDA capable GPU." << endl
Test_main.cpp:          << "              \t  Uses a CUDA powered engine, therefore, requires a CUDA capable GPU." << endl
Test_main.cpp:          << "              \t  Uses a CUDA powered engine, therefore, requires a CUDA capable GPU." << endl
Test_main.cpp:          << "              \t  Uses a CUDA powered engine, therefore, requires a CUDA capable GPU." << endl
Test_main.cpp:          << "                    Uses a CUDA powered engine, therefore, requires a CUDA capable GPU." << endl
Test_main.cpp:          << "              \t  Uses a CUDA powered engine, therefore, requires a CUDA capable GPU." << endl
Test_main.cpp:          << "              \t  Uses a CUDA powered engine, therefore, requires a CUDA capable GPU." << endl
Test_main.cpp:          << "              \t  Uses a CUDA powered engine, therefore, requires a CUDA capable GPU." << endl
Test_main.cpp:          << "-c or --cuda\t: Lists all available CUDA capable devices on machine." << endl
Test_main.cpp:          << "            \t  Use the \"GPU number\" to select the desired CUDA device in the parameter file." << endl
prometheus.cuh:#include "cuda_runtime.h"
prometheus.cuh:     * @param tot_Blocks defines number of GPU blocks that are available
prometheus.cuh:     * @param tot_ThreadsperBlock defines number of threads that are available per GPU block.
prometheus.cuh:     * @param SNPs_per_Run defines the maximum number of SNPs that can be processed by the GPU at any given time.
prometheus.cuh:     * @param pre_MA, @param pre_Theta_partials, @param pre_ne, @param pre_ns: GPU processed SNP values that will be carried forward.
prometheus.cuh:     * @param same_Files acts as a Boolean variable and will indicate the GPU to process the new list of SNPs or whether to use the existing list.
prometheus.cuh:     * @param start_stop used to keep track of the start and stop seg numbers for each thread when concatenating the seg data for GPU use.
prometheus.cuh:     * @param concat_Segs stores the concatenated Seg sites for GPU processing
prometheus.cuh:     * Concatenates the Segregating sites for processing by each GPU round.
parameters_Apollo/parameters_MASTER.json:  # Configuration of the Cuda device IDs accesible to Apollo
parameters_Apollo/parameters_MASTER.json:  # Here only one CUDA device is assigned for Apollo's use
parameters_Apollo/parameters_MASTER.json:  "CUDA Device IDs":"0",
parameters_Apollo/parameters_MASTER.json:  "GPU max units":1000,
cudaDevices.cuh:#include "cuda_runtime.h"
cudaDevices.cuh:#include <thrust/system/cuda/error.h>
cudaDevices.cuh:class cudaDevices
cudaDevices.cuh:     * Prints all available CUDA devices present on the current system.
cudaDevices.cuh:     * User can use this list to determine which CUDA device to be used via the CUDA ID.
cudaDevices.cuh:     * It also prints all CUDA devices.
cudaDevices.cuh:    cudaDevices();
ehh.cu:ehh::ehh(string range_Mode, string file_Mode_path, string fixed_Mode_value, string input_Folder, string output_Path, int cuda_ID, string intermediate_Path, int ploidy, int default_SNP_count, int EHH_CPU_cores, int default_SNP_BP_count)
ehh.cu:    cout << "Initiating CUDA powered Extended Haplotype Homozygosity (EHH) calculator" << endl
ehh.cu:    cudaSetDevice(cuda_ID);
ehh.cu:    cout << "Properties of selected CUDA GPU:" << endl;
ehh.cu:    cudaDeviceProp prop;
ehh.cu:    cudaGetDeviceProperties(&prop, cuda_ID);
ehh.cu:    cout << "GPU number\t: " << cuda_ID << endl;
ehh.cu:    cout << "GPU name\t: " << prop.name << endl;
ehh.cu:    cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);
ehh.cu:    cout << "GPU memory (GB)\t: " << l_Total / (1000 * 1000 * 1000) << endl;
ehh.cu:    cout << "GPU number of multiprocessor(s)\t: " << prop.multiProcessorCount << endl;
ehh.cu:    cout << "GPU block(s) per multiprocessor\t: " << prop.maxBlocksPerMultiProcessor << endl;
ehh.cu:    cout << "GPU thread(s) per block\t: " << tot_ThreadsperBlock << endl
ehh.cu:__global__ void cuda_SNP_grid_0_1_BP(int total_Segs, char *sites, int *index, char *Hap_array, int core_SNP_index, int *core_SNP_alleles, int *SNP_counts)
ehh.cu:__global__ void cuda_SNP_grid_0_1(int total_Segs, char *sites, int *index, char *Hap_array, int core_SNP_index, int *core_SNP_alleles, int *SNP_counts, int *cuda_pos_start_Index, int *cuda_pos_end_Index)
ehh.cu:        cuda_pos_start_Index[tid] = i;
ehh.cu:        cuda_pos_end_Index[tid] = i - 1;
ehh.cu:__global__ void cuda_EHH_up_0(int total_Segs_UP, char **grid, int core_SNP_index, int *core_SNP_alleles, int N, int **Indexes_found_Zero, int zero_Count, float *EHH_Zero, int combo_Zero)
ehh.cu:    cout << "STEP 1 of 4: Organizing SNPs for GPU" << endl;
ehh.cu:    char *cuda_full_Char;
ehh.cu:    cudaMallocManaged(&cuda_full_Char, (Seg_sites.size() + 1) * sizeof(char));
ehh.cu:    int *cuda_site_Index;
ehh.cu:    cudaMallocManaged(&cuda_site_Index, (num_segregrating_Sites + 1) * sizeof(int));
ehh.cu:    int *cuda_core_SNP_alleles;
ehh.cu:    int *SNP_counts, *cuda_SNP_counts;
ehh.cu:    cudaMallocManaged(&cuda_SNP_counts, 2 * sizeof(int));
ehh.cu:    cudaMallocManaged(&cuda_core_SNP_alleles, N * sizeof(int));
ehh.cu:    cudaMemcpy(cuda_full_Char, full_Char, (Seg_sites.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
ehh.cu:    cudaMemcpy(cuda_site_Index, site_Index, (num_segregrating_Sites + 1) * sizeof(int), cudaMemcpyHostToDevice);
ehh.cu:     * @param cuda_Hap_array stores the forged Haplotypes for the region under study.
ehh.cu:     * @param Hap_array is used by the CPU. Is a COPY of cuda_Hap_array.
ehh.cu:    char *Hap_array, *cuda_Hap_array;
ehh.cu:    cudaMallocManaged(&cuda_Hap_array, ((N * num_segregrating_Sites) + 1) * sizeof(char));
ehh.cu:    // cuda_SNP_grid_0_1_BP(int total_Segs, char *sites, int *index, char *Hap_array, int core_SNP_index, int *core_SNP_alleles, int *SNP_counts)
ehh.cu:    cuda_SNP_grid_0_1_BP<<<tot_Blocks, tot_ThreadsperBlock>>>(num_segregrating_Sites, cuda_full_Char, cuda_site_Index, cuda_Hap_array, SNP_Index_in_FULL, cuda_core_SNP_alleles, cuda_SNP_counts);
ehh.cu:    cudaError_t err = cudaGetLastError();
ehh.cu:    if (err != cudaSuccess)
ehh.cu:        printf("CUDA Error: %s\n", cudaGetErrorString(err));
ehh.cu:    cudaDeviceSynchronize();
ehh.cu:    cudaMemcpy(core_SNP_alleles, cuda_core_SNP_alleles, N * sizeof(int), cudaMemcpyDeviceToHost);
ehh.cu:    cudaMemcpy(Hap_array, cuda_Hap_array, ((N * num_segregrating_Sites) + 1) * sizeof(char), cudaMemcpyDeviceToHost);
ehh.cu:    cudaMemcpy(SNP_counts, cuda_SNP_counts, 2 * sizeof(int), cudaMemcpyDeviceToHost);
ehh.cu:    cudaFree(cuda_full_Char);
ehh.cu:    cudaFree(cuda_site_Index);
ehh.cu:    cudaFree(cuda_core_SNP_alleles);
ehh.cu:    cudaFree(cuda_SNP_counts);
ehh.cu:    cudaFree(cuda_Hap_array);
ehh.cu:    cout << "STEP 2 of 6: Organizing SNPs for GPU" << endl;
ehh.cu:    char *cuda_full_Char;
ehh.cu:    cudaMallocManaged(&cuda_full_Char, (Seg_sites.size() + 1) * sizeof(char));
ehh.cu:    int *cuda_site_Index;
ehh.cu:    cudaMallocManaged(&cuda_site_Index, (num_segregrating_Sites + 1) * sizeof(int));
ehh.cu:    int *cuda_core_SNP_alleles;
ehh.cu:    int *SNP_counts, *cuda_SNP_counts;
ehh.cu:    cudaMallocManaged(&cuda_SNP_counts, 2 * sizeof(int));
ehh.cu:    cudaMallocManaged(&cuda_core_SNP_alleles, N * sizeof(int));
ehh.cu:    cudaMemcpy(cuda_full_Char, full_Char, (Seg_sites.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
ehh.cu:    cudaMemcpy(cuda_site_Index, site_Index, (num_segregrating_Sites + 1) * sizeof(int), cudaMemcpyHostToDevice);
ehh.cu:     * @param cuda_Hap_array stores the forged Haplotypes for the region under study.
ehh.cu:     * @param Hap_array is used by the CPU. Is a COPY of cuda_Hap_array.
ehh.cu:    char *Hap_array, *cuda_Hap_array;
ehh.cu:    cudaMallocManaged(&cuda_Hap_array, ((N * num_segregrating_Sites) + 1) * sizeof(char));
ehh.cu:    int *pos_start_Index, *pos_end_Index, *cuda_pos_start_Index, *cuda_pos_end_Index;
ehh.cu:    cudaMallocManaged(&cuda_pos_start_Index, num_segregrating_Sites * sizeof(int));
ehh.cu:    cudaMallocManaged(&cuda_pos_end_Index, num_segregrating_Sites * sizeof(int));
ehh.cu:    // char **cuda_snp_N_grid;
ehh.cu:    // cudaMallocManaged(&cuda_snp_N_grid, N * num_segregrating_Sites * sizeof(char));
ehh.cu:    //     cudaMalloc((void **)&tmp[i], N * sizeof(tmp[0][0]));
ehh.cu:    // cudaMemcpy(cuda_snp_N_grid, tmp, num_segregrating_Sites * sizeof(char *), cudaMemcpyHostToDevice);
ehh.cu:    cuda_SNP_grid_0_1<<<tot_Blocks, tot_ThreadsperBlock>>>(num_segregrating_Sites, cuda_full_Char, cuda_site_Index, cuda_Hap_array, SNP_position_in_full, cuda_core_SNP_alleles, cuda_SNP_counts, cuda_pos_start_Index, cuda_pos_end_Index);
ehh.cu:    cudaError_t err = cudaGetLastError();
ehh.cu:    if (err != cudaSuccess)
ehh.cu:        printf("CUDA Error: %s\n", cudaGetErrorString(err));
ehh.cu:    cudaDeviceSynchronize();
ehh.cu:    cudaMemcpy(core_SNP_alleles, cuda_core_SNP_alleles, N * sizeof(int), cudaMemcpyDeviceToHost);
ehh.cu:    cudaMemcpy(Hap_array, cuda_Hap_array, ((N * num_segregrating_Sites) + 1) * sizeof(char), cudaMemcpyDeviceToHost);
ehh.cu:    cudaMemcpy(SNP_counts, cuda_SNP_counts, 2 * sizeof(int), cudaMemcpyDeviceToHost);
ehh.cu:    cudaMemcpy(pos_start_Index, cuda_pos_start_Index, num_segregrating_Sites * sizeof(int), cudaMemcpyDeviceToHost);
ehh.cu:    cudaMemcpy(pos_end_Index, cuda_pos_end_Index, num_segregrating_Sites * sizeof(int), cudaMemcpyDeviceToHost);
ehh.cu:    cudaFree(cuda_full_Char);
ehh.cu:    cudaFree(cuda_site_Index);
ehh.cu:    cudaFree(cuda_core_SNP_alleles);
ehh.cu:    cudaFree(cuda_SNP_counts);
ehh.cu:    cudaFree(cuda_pos_start_Index);
ehh.cu:    cudaFree(cuda_pos_end_Index);
ehh.cu:    cudaFree(cuda_Hap_array);
ehh.cu:    // int **cuda_Indexes_found_Zero;
ehh.cu:    // cudaMallocManaged(&cuda_Indexes_found_Zero, SNP_counts[0] * SNPs_above * sizeof(int));
ehh.cu:    //     cudaMalloc((void **)&tmp_2[i], SNP_counts[0] * sizeof(tmp_2[0][0]));
ehh.cu:    // cudaMemcpy(cuda_Indexes_found_Zero, tmp_2, SNPs_above * sizeof(int *), cudaMemcpyHostToDevice);
ehh.cu:    // float *cuda_EHH_Zero, *EHH_Zero;
ehh.cu:    // cudaMallocManaged(&cuda_EHH_Zero, SNPs_above * sizeof(float));
ehh.cu:    // cuda_EHH_up_0(int total_Segs_UP, char **grid, int core_SNP_index, int *core_SNP_alleles, int N, int **Indexes_found_Zero, int zero_Count, float *EHH_Zero, int combo_Zero)
ehh.cu:    // cuda_EHH_up_0<<<tot_Blocks, tot_ThreadsperBlock>>>(SNPs_above, cuda_snp_N_grid, SNP_position_in_full, cuda_core_SNP_alleles, (int)N, cuda_Indexes_found_Zero, SNP_counts[0], cuda_EHH_Zero, combo_Zero);
ehh.cu:    // cudaError_t err2 = cudaGetLastError();
ehh.cu:    // if (err2 != cudaSuccess)
ehh.cu:    //     printf("CUDA Error: %s\n", cudaGetErrorString(err2));
ehh.cu:    // cudaDeviceSynchronize();
ehh.cu:    // cudaMemcpy(EHH_Zero, cuda_EHH_Zero, SNPs_above * sizeof(float), cudaMemcpyDeviceToHost);
ehh.cu:__global__ void cuda_SNP_grid(int total_Segs, char *sites, int *index, char **grid)
ehh.cu:__global__ void cuda_Core_Haplotype_concat(int N, int total_Segs, char **grid, int *core_OR_ext, int core_Count, char *core_Hap_array, char *ext_Hap_array)
ehh.cu:__global__ void cuda_search_Extended_Haplotypes(int core_Count, char **grid, int **core_Map_grid, int *core_Sizes, int total_Segs, int *cores_Hap_Sums)
ehh.cu:    int *core_OR_ext_Array, *cuda_core_OR_ext_Array;
ehh.cu:    char **cuda_snp_N_grid;
ehh.cu:    cudaMallocManaged(&cuda_snp_N_grid, (N + 1) * num_segregrating_Sites * sizeof(char));
ehh.cu:        cudaMalloc((void **)&tmp[i], (N + 1) * sizeof(tmp[0][0]));
ehh.cu:    cudaMemcpy(cuda_snp_N_grid, tmp, num_segregrating_Sites * sizeof(char *), cudaMemcpyHostToDevice);
ehh.cu:    char *cuda_full_Char;
ehh.cu:    cudaMallocManaged(&cuda_full_Char, (Seg_sites.size() + 1) * sizeof(char));
ehh.cu:    int *cuda_site_Index;
ehh.cu:    cudaMallocManaged(&cuda_site_Index, (num_segregrating_Sites + 1) * sizeof(int));
ehh.cu:    cudaMemcpy(cuda_full_Char, full_Char, (Seg_sites.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
ehh.cu:    cudaMemcpy(cuda_site_Index, site_Index, (num_segregrating_Sites + 1) * sizeof(int), cudaMemcpyHostToDevice);
ehh.cu:    // cuda_SNP_grid(int total_Segs, char *sites, int *index, int **grid)
ehh.cu:    cuda_SNP_grid<<<tot_Blocks, tot_ThreadsperBlock>>>(num_segregrating_Sites, cuda_full_Char, cuda_site_Index, cuda_snp_N_grid);
ehh.cu:    cudaDeviceSynchronize();
ehh.cu:    cudaFree(cuda_full_Char);
ehh.cu:    cudaFree(cuda_site_Index);
ehh.cu:    //     cudaMemcpy(snp_N_grid[i], cuda_snp_N_grid[i], N * sizeof(cuda_snp_N_grid[0][0]), cudaMemcpyDeviceToHost);
ehh.cu:    char *core_Hap_array, *cuda_core_Hap_array;
ehh.cu:    char *ext_Hap_array, *cuda_ext_Hap_array;
ehh.cu:    cudaMallocManaged(&cuda_core_Hap_array, ((N * core_Count) + 1) * sizeof(char));
ehh.cu:    cudaMallocManaged(&cuda_ext_Hap_array, ((N * num_segregrating_Sites) + 1) * sizeof(char));
ehh.cu:    cudaMallocManaged(&cuda_core_OR_ext_Array, num_segregrating_Sites * sizeof(int));
ehh.cu:    cudaMemcpy(cuda_core_OR_ext_Array, core_OR_ext_Array, num_segregrating_Sites * sizeof(int), cudaMemcpyHostToDevice);
ehh.cu:    // cuda_Haplotype_core_ext(int N, int total_Segs, int **grid, int *core_OR_ext, int core_Count, int ext_Count, int *core_Hap_array, int *ext_Hap_array)
ehh.cu:    cuda_Core_Haplotype_concat<<<tot_Blocks, tot_ThreadsperBlock>>>(N, num_segregrating_Sites, cuda_snp_N_grid, cuda_core_OR_ext_Array, core_Count, cuda_core_Hap_array, cuda_ext_Hap_array);
ehh.cu:    cudaDeviceSynchronize();
ehh.cu:    cudaMemcpy(core_Hap_array, cuda_core_Hap_array, ((N * core_Count) + 1) * sizeof(char), cudaMemcpyDeviceToHost);
ehh.cu:    cudaMemcpy(ext_Hap_array, cuda_ext_Hap_array, ((N * num_segregrating_Sites) + 1) * sizeof(char), cudaMemcpyDeviceToHost);
ehh.cu:    // PROCESS EACH CORE HAPLOTYPE SEPERATELY IN THE GPU
ehh.cu:    // int **core_Map_grid, **cuda_core_Map_grid;
ehh.cu:    // int *core_Sizes, *cuda_core_Sizes;
ehh.cu:    // cudaMallocManaged(&cuda_core_Sizes, unique_Haplotypes_num * sizeof(int));
ehh.cu:    // cudaMemcpy(cuda_core_Sizes, core_Sizes, unique_Haplotypes_num * sizeof(int), cudaMemcpyHostToDevice);
ehh.cu:    // cudaMallocManaged(&cuda_core_Map_grid, unique_Haplotypes_num * max_Count * sizeof(int));
ehh.cu:    //     cudaMalloc((void **)&tmp_2[i], max_Count * sizeof(tmp_2[0][0]));
ehh.cu:    // cudaMemcpy(cuda_core_Map_grid, tmp_2, unique_Haplotypes_num * sizeof(int *), cudaMemcpyHostToDevice);
ehh.cu:    //     cudaMemcpy(tmp_2[i], core_Map_grid[i], max_Count * sizeof(cuda_core_Map_grid[0][0]), cudaMemcpyHostToDevice);
ehh.cu:    // int *core_Hap_sums, *cuda_core_Hap_sums;
ehh.cu:    // cudaMallocManaged(&cuda_core_Hap_sums, unique_Haplotypes_num * sizeof(int));
ehh.cu:    // search_Extended_Haplotypes<<<tot_Blocks, tot_ThreadsperBlock>>>(unique_Haplotypes_num, cuda_snp_N_grid, cuda_core_Map_grid, cuda_core_Sizes, num_segregrating_Sites, cuda_core_Hap_sums);
ehh.cu:    // cudaDeviceSynchronize();
ehh.cu:    // cudaMemcpy(core_Hap_sums, cuda_core_Hap_sums, unique_Haplotypes_num * sizeof(int), cudaMemcpyDeviceToHost);
ehh.cu:    cudaFree(cuda_snp_N_grid);
ehh.cu:    cudaFree(cuda_core_Hap_array);
ehh.cu:    cudaFree(cuda_ext_Hap_array);
functions_library.cu:functions_library::functions_library(int tot_Blocks, int tot_ThreadsperBlock, int gpu_Limit, int CPU_cores)
functions_library.cu:    this->gpu_Limit = gpu_Limit;
functions_library.cu:functions_library::functions_library(int *tot_Blocks_array, int *tot_ThreadsperBlock_array, int *CUDA_device_IDs, int num_Cuda_devices, int gpu_Limit, int CPU_cores)
functions_library.cu:    this->CUDA_device_IDs = CUDA_device_IDs;
functions_library.cu:    this->num_Cuda_devices = num_Cuda_devices;
functions_library.cu:    this->gpu_Limit = gpu_Limit;
functions_library.cu:void functions_library::print_Cuda_device(int cuda_ID, int &tot_Blocks, int &tot_ThreadsperBlock)
functions_library.cu:    cudaSetDevice(cuda_ID);
functions_library.cu:    cout << "Properties of selected CUDA GPU:" << endl;
functions_library.cu:    cudaDeviceProp prop;
functions_library.cu:    cudaGetDeviceProperties(&prop, cuda_ID);
functions_library.cu:    cout << "GPU number\t: " << cuda_ID << endl;
functions_library.cu:    cout << "GPU name\t: " << prop.name << endl;
functions_library.cu:    cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);
functions_library.cu:    cout << "GPU memory (GB)\t: " << l_Total / (1000 * 1000 * 1000) << endl;
functions_library.cu:    cout << "GPU number of multiprocessor(s)\t: " << prop.multiProcessorCount << endl;
functions_library.cu:    cout << "GPU block(s) per multiprocessor\t: " << prop.maxBlocksPerMultiProcessor << endl;
functions_library.cu:    cout << "GPU thread(s) per block\t: " << tot_ThreadsperBlock << endl;
functions_library.cu:void functions_library::print_Cuda_devices(vector<string> cuda_IDs, int *CUDA_device_IDs, int num_Cuda_devices, int *tot_Blocks, int *tot_ThreadsperBlock)
functions_library.cu:    cudaGetDeviceCount(&nDevices);
functions_library.cu:    if (nDevices >= num_Cuda_devices)
functions_library.cu:        cout << "Properties of selected " << num_Cuda_devices << " CUDA GPU(s):" << endl;
functions_library.cu:        for (int device = 0; device < num_Cuda_devices; device++)
functions_library.cu:            cudaDeviceProp prop;
functions_library.cu:            CUDA_device_IDs[device] = stoi(cuda_IDs[device]);
functions_library.cu:            cudaGetDeviceProperties(&prop, CUDA_device_IDs[device]);
functions_library.cu:            cout << "\nGPU number\t: " << CUDA_device_IDs[device] << endl;
functions_library.cu:            cout << "GPU name\t: " << prop.name << endl;
functions_library.cu:            cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);
functions_library.cu:            cout << "GPU memory (GB)\t: " << l_Total / (1000 * 1000 * 1000) << endl;
functions_library.cu:            cout << "GPU number of multiprocessor(s)\t: " << prop.multiProcessorCount << endl;
functions_library.cu:            cout << "GPU block(s) per multiprocessor\t: " << tot_Blocks[device] << endl;
functions_library.cu:            cout << "GPU thread(s) per block\t: " << tot_ThreadsperBlock[device] << endl;
functions_library.cu:        cout << "ERROR: THERE MORE CUDA DEVICES THAN PRESENT HAVE BEEN SELECTED\n";
functions_library.cu:        cout << "USER HAS SELECTED " << num_Cuda_devices << " BUT THERE IS/ ARE ONLY " << nDevices << " PRESENT IN THE SYSTEM\n";
functions_library.cu:int **functions_library::create_INT_2D_arrays_for_GPU(int rows, int columns)
functions_library.cu:float **functions_library::create_FLOAT_2D_arrays_for_GPU(int rows, int columns)
functions_library.cu:__global__ void CUDA_normal_distribution(curandState *state, int num_Values, float *values, float mean, float st_deviation, int start_Index)
functions_library.cu:__global__ void CUDA_poisson_distribution(curandState *state, int num_Values, int *values, float mean, int start_Index)
functions_library.cu:// __global__ void CUDA_gamma_distribution(curandState *state, int num_Values, float *values, float shape, float scale, int start_Index)
functions_library.cu:// __global__ void CUDA_gamma_distribution_PROGENY(curandState *state, int num_Values, int **Progeny_values, float shape, float scale, int start_Index, int hotspot_Number, float **CUDA_current_gen_Parent_data)
functions_library.cu://         float progeny_Base_fitness = CUDA_current_gen_Parent_data[parent_ID][0];
functions_library.cu://             progeny_Base_fitness = progeny_Base_fitness * CUDA_current_gen_Parent_data[parent_ID][(hotspot * 3) + 2];
functions_library.cu://                 CUDA_current_gen_Parent_data[parent_ID][selectivity_Index] = 0;
functions_library.cu://                 if (curand_uniform(&local_state) < CUDA_current_gen_Parent_data[parent_ID][ratio_Index])
functions_library.cu:__global__ void CUDA_binomial_distribution(curandState *state, int num_Values, int *values, float prob, int trials, int start_Index)
functions_library.cu:__global__ void CUDA_negative_binomial_distribution(curandState *state, int num_Values, int *values, int r, float p, int start_Index)
functions_library.cu:__global__ void CUDA_negative_binomial_distribution_PROGENY(curandState *state, int num_Values, int *Progeny_values, int r, float p, int start_Index, float *cuda__Parent_finess)
functions_library.cu:                Progeny_values[parent_ID] = (int)((k + x) * cuda__Parent_finess[parent_ID]);
functions_library.cu:int *functions_library::negative_binomial_distribution_CUDA(int num_of_values, float mean, float dispersion_Parameter)
functions_library.cu:    int *values, *cuda_Values;
functions_library.cu:    cudaMallocManaged(&cuda_Values, num_of_values * sizeof(int));
functions_library.cu:    int full_Rounds = num_of_values / this->gpu_Limit;
functions_library.cu:    int partial_Rounds = num_of_values % this->gpu_Limit;
functions_library.cu:        int start = full * this->gpu_Limit;
functions_library.cu:        int stop = start + this->gpu_Limit;
functions_library.cu:        cudaMalloc((void **)&state, num_of_values_current * sizeof(curandState));
functions_library.cu:        // CUDA_negative_binomial_distribution(curandState *state, int num_Values, int *values, float r, float p, int start_Index)
functions_library.cu:        CUDA_negative_binomial_distribution<<<tot_Blocks, tot_ThreadsperBlock>>>(state, num_of_values_current, cuda_Values, r, prob, start_stops[i].first);
functions_library.cu:        cudaError_t err = cudaGetLastError();
functions_library.cu:        if (err != cudaSuccess)
functions_library.cu:            printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:        cudaDeviceSynchronize();
functions_library.cu:        cudaFree(state);
functions_library.cu:    cudaMemcpy(values, cuda_Values, num_of_values * sizeof(int), cudaMemcpyDeviceToHost);
functions_library.cu:    cudaFree(cuda_Values);
functions_library.cu:    cout << "Completed generation via " << start_stops.size() << " GPU rounds" << endl;
functions_library.cu:int *functions_library::binomial_distribution_CUDA(int num_of_values, float prob, int progeny_Number)
functions_library.cu:    int *values, *cuda_Values;
functions_library.cu:    cudaMallocManaged(&cuda_Values, num_of_values * sizeof(int));
functions_library.cu:    int full_Rounds = num_of_values / this->gpu_Limit;
functions_library.cu:    int partial_Rounds = num_of_values % this->gpu_Limit;
functions_library.cu:        int start = full * this->gpu_Limit;
functions_library.cu:        int stop = start + this->gpu_Limit;
functions_library.cu:        cudaMalloc((void **)&state, num_of_values_current * sizeof(curandState));
functions_library.cu:        CUDA_binomial_distribution<<<tot_Blocks, tot_ThreadsperBlock>>>(state, num_of_values_current, cuda_Values, prob, progeny_Number, start_stops[i].first);
functions_library.cu:        cudaError_t err = cudaGetLastError();
functions_library.cu:        if (err != cudaSuccess)
functions_library.cu:            printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:        cudaDeviceSynchronize();
functions_library.cu:        cudaFree(state);
functions_library.cu:    cudaMemcpy(values, cuda_Values, num_of_values * sizeof(int), cudaMemcpyDeviceToHost);
functions_library.cu:    cudaFree(cuda_Values);
functions_library.cu:    cout << "Completed generation via " << start_stops.size() << " GPU rounds" << endl;
functions_library.cu:// float *functions_library::gamma_distribution_CUDA(int num_of_values, float shape, float scale)
functions_library.cu://     float *values, *cuda_Values;
functions_library.cu://     cudaMallocManaged(&cuda_Values, num_of_values * sizeof(float));
functions_library.cu://     int full_Rounds = num_of_values / this->gpu_Limit;
functions_library.cu://     int partial_Rounds = num_of_values % this->gpu_Limit;
functions_library.cu://         int start = full * this->gpu_Limit;
functions_library.cu://         int stop = start + this->gpu_Limit;
functions_library.cu://         cudaMalloc((void **)&state, num_of_values_current * sizeof(curandState));
functions_library.cu://         //CUDA_gamma_distribution<<<tot_Blocks, tot_ThreadsperBlock>>>(state, num_of_values_current, cuda_Values, shape, scale, start_stops[i].first);
functions_library.cu://         cudaError_t err = cudaGetLastError();
functions_library.cu://         if (err != cudaSuccess)
functions_library.cu://             printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu://         cudaDeviceSynchronize();
functions_library.cu://         cudaFree(state);
functions_library.cu://     cout << "Completed generation via " << start_stops.size() << " GPU rounds" << endl;
functions_library.cu://     cudaMemcpy(values, cuda_Values, num_of_values * sizeof(float), cudaMemcpyDeviceToHost);
functions_library.cu://     cudaFree(cuda_Values);
functions_library.cu:int *functions_library::poisson_distribution_CUDA(int num_of_values, float mean)
functions_library.cu:    int *values, *cuda_Values;
functions_library.cu:    cudaMallocManaged(&cuda_Values, num_of_values * sizeof(int));
functions_library.cu:    int full_Rounds = num_of_values / this->gpu_Limit;
functions_library.cu:    int partial_Rounds = num_of_values % this->gpu_Limit;
functions_library.cu:        int start = full * this->gpu_Limit;
functions_library.cu:        int stop = start + this->gpu_Limit;
functions_library.cu:        cudaMalloc((void **)&state, num_of_values_current * sizeof(curandState));
functions_library.cu:        CUDA_poisson_distribution<<<tot_Blocks, tot_ThreadsperBlock>>>(state, num_of_values_current, cuda_Values, mean, start_stops[i].first);
functions_library.cu:        cudaError_t err = cudaGetLastError();
functions_library.cu:        if (err != cudaSuccess)
functions_library.cu:            printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:        cudaDeviceSynchronize();
functions_library.cu:        cudaFree(state);
functions_library.cu:    cudaMemcpy(values, cuda_Values, num_of_values * sizeof(int), cudaMemcpyDeviceToHost);
functions_library.cu:    cudaFree(cuda_Values);
functions_library.cu:    cout << "Completed generation via " << start_stops.size() << " GPU rounds" << endl;
functions_library.cu:int *functions_library::copy_1D_to_CUDA_INT(int *host_Array, int num_Values)
functions_library.cu:    int *cuda_Device_array;
functions_library.cu:    cudaMallocManaged(&cuda_Device_array, num_Values * sizeof(int));
functions_library.cu:    cudaMemcpy(cuda_Device_array, host_Array, num_Values * sizeof(int), cudaMemcpyHostToDevice);
functions_library.cu:    return cuda_Device_array;
functions_library.cu:float *functions_library::copy_1D_to_CUDA_FLOAT(float *host_Array, int num_Values)
functions_library.cu:    float *cuda_Device_array;
functions_library.cu:    cudaMallocManaged(&cuda_Device_array, num_Values * sizeof(float));
functions_library.cu:    cudaMemcpy(cuda_Device_array, host_Array, num_Values * sizeof(float), cudaMemcpyHostToDevice);
functions_library.cu:    return cuda_Device_array;
functions_library.cu:float *functions_library::normal_distribution_CUDA(int num_of_values, float mean, float st_deviation)
functions_library.cu:    float *values, *cuda_Values;
functions_library.cu:    cudaMallocManaged(&cuda_Values, num_of_values * sizeof(float));
functions_library.cu:    int full_Rounds = num_of_values / this->gpu_Limit;
functions_library.cu:    int partial_Rounds = num_of_values % this->gpu_Limit;
functions_library.cu:        int start = full * this->gpu_Limit;
functions_library.cu:        int stop = start + this->gpu_Limit;
functions_library.cu:        cudaMalloc((void **)&state, num_of_values_current * sizeof(curandState));
functions_library.cu:        CUDA_normal_distribution<<<tot_Blocks, tot_ThreadsperBlock>>>(state, num_of_values_current, cuda_Values, mean, st_deviation, start_stops[i].first);
functions_library.cu:        cudaError_t err = cudaGetLastError();
functions_library.cu:        if (err != cudaSuccess)
functions_library.cu:            printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:        cudaDeviceSynchronize();
functions_library.cu:        cudaFree(state);
functions_library.cu:    cudaMemcpy(values, cuda_Values, num_of_values * sizeof(float), cudaMemcpyDeviceToHost);
functions_library.cu:    cudaFree(cuda_Values);
functions_library.cu:    cout << "Completed generation via " << start_stops.size() << " GPU rounds" << endl;
functions_library.cu:int **functions_library::progeny_distribution_CUDA(string &distribution_Type, int &num_of_parents, float &shape, float &scale, float &mean, float &dispersion_Parameter, float *cuda__Parent_finess, float **CUDA_current_gen_Parent_data, int recombination_hotspots)
functions_library.cu:    int **cuda_Progeny_numbers;
functions_library.cu:    cudaMallocManaged(&cuda_Progeny_numbers, (num_of_parents + 1) * (recombination_hotspots + 1) * sizeof(int));
functions_library.cu:        cudaMalloc((void **)&tmp[i], (num_of_parents + 1) * sizeof(tmp[0][0]));
functions_library.cu:    cudaMemcpy(cuda_Progeny_numbers, tmp, (recombination_hotspots + 1) * sizeof(int *), cudaMemcpyHostToDevice);
functions_library.cu:    int full_Rounds = num_of_parents / this->gpu_Limit;
functions_library.cu:    int partial_Rounds = num_of_parents % this->gpu_Limit;
functions_library.cu:        int start = full * this->gpu_Limit;
functions_library.cu:        int stop = start + this->gpu_Limit;
functions_library.cu:        cudaMalloc((void **)&state, num_of_values_current * sizeof(curandState));
functions_library.cu:            // CUDA_gamma_distribution_PROGENY(curandState *state, int num_Values, int **Progeny_values, float shape, float scale, int start_Index, float *cuda__Parent_finess, int hotspot_Number, float **CUDA_current_gen_Parent_data)
functions_library.cu:            // CUDA_gamma_distribution_PROGENY<<<tot_Blocks, tot_ThreadsperBlock>>>(state, num_of_values_current, cuda_Progeny_numbers, shape, scale, start_stops[i].first, cuda__Parent_finess, recombination_hotspots, CUDA_current_gen_Parent_data);
functions_library.cu:            // CUDA_negative_binomial_distribution_PROGENY<<<tot_Blocks, tot_ThreadsperBlock>>>(state, num_of_values_current, cuda_Progeny_numbers, r, prob, start_stops[i].first, cuda__Parent_finess);
functions_library.cu:            // CUDA_negative_binomial_distribution_PROGENY
functions_library.cu:        cudaError_t err = cudaGetLastError();
functions_library.cu:        if (err != cudaSuccess)
functions_library.cu:            printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:        cudaDeviceSynchronize();
functions_library.cu:        cudaFree(state);
functions_library.cu:    cout << "Completed generation via " << start_stops.size() << " GPU rounds" << endl;
functions_library.cu:    return cuda_Progeny_numbers;
functions_library.cu:__global__ void cuda_Fill_2D_array(int total, int columns, int fill_Value, int **array_2D)
functions_library.cu:__global__ void cuda_Fill_2D_array_Float(int total, int columns, float fill_Value, float **array_2D)
functions_library.cu:__global__ void cuda_Fill_2D_array_INT_Device(int total, int columns, int fill_Value, int **array_2D, int start_Index)
functions_library.cu:__global__ void progeny_Array_CUDA(int total_Progeny_current, int parent_ID, int num_Recombination_hotspots, int **cuda_Progeny_numbers, int **cuda_progeny_Array, int fill_Value, int start_Index, float **CUDA_fitness_distribution, int count_Parents)
functions_library.cu:        cuda_progeny_Array[progeny_Index][0] = parent_ID;
functions_library.cu:            int number_of_recombinant_Progeny = cuda_Progeny_numbers[hotspot + 1][parent_ID];
functions_library.cu:                    cumulative_prob += CUDA_fitness_distribution[hotspot][check];
functions_library.cu:                cuda_progeny_Array[progeny_Index][hotspot + 1] = parent;
functions_library.cu:                cuda_progeny_Array[progeny_Index][hotspot + 1] = fill_Value;
functions_library.cu:__global__ void progeny_Array_CUDA_CELLS(int total_Progeny_current, int parent_Start, int start_Index, int parent_ID, int num_Recombination_hotspots, int **cuda_Progeny_numbers, int **cuda_progeny_Array, int fill_Value, float **CUDA_fitness_distribution, int *CUDA_per_Cell_parents_Stride, int **CUDA_parent_Indexes_parents_and_their_Cells)
functions_library.cu:        cuda_progeny_Array[progeny_Array_Index][0] = parent_ID;
functions_library.cu:        int cell_ID = CUDA_parent_Indexes_parents_and_their_Cells[1][parent_ID];
functions_library.cu:        int start_Cell = CUDA_per_Cell_parents_Stride[cell_ID];
functions_library.cu:        int stop_Cell = CUDA_per_Cell_parents_Stride[cell_ID + 1];
functions_library.cu:            int number_of_recombinant_Progeny = cuda_Progeny_numbers[hotspot + 1][parent_ID];
functions_library.cu:                    cumulative_prob += CUDA_fitness_distribution[check][hotspot];
functions_library.cu:                cuda_progeny_Array[progeny_Array_Index][hotspot + 1] = parent;
functions_library.cu:                cuda_progeny_Array[progeny_Array_Index][hotspot + 1] = fill_Value;
functions_library.cu:__global__ void CUDA_summation_Selectivity(int num_Hotspots, int count_Parents, float **CUDA_current_gen_Parent_data, float *CUDA_hotspot_selectivity_Summations)
functions_library.cu:            total_Selectivity = total_Selectivity + CUDA_current_gen_Parent_data[parent][selectivity_Index];
functions_library.cu:        CUDA_hotspot_selectivity_Summations[tid] = total_Selectivity;
functions_library.cu:__global__ void CUDA_summation_Selectivity_CELLS(int cells, int start_Index, int num_Hotspots, float **CUDA_current_gen_Parent_data, float **CUDA_hotspot_selectivity_Summations, int *CUDA_per_Cell_parents_Stride)
functions_library.cu:        int start = CUDA_per_Cell_parents_Stride[cell_ID];
functions_library.cu:        int stop = CUDA_per_Cell_parents_Stride[cell_ID + 1];
functions_library.cu:                total_Selectivity = total_Selectivity + CUDA_current_gen_Parent_data[parent][selectivity_Index];
functions_library.cu:            CUDA_hotspot_selectivity_Summations[cell_ID][hotspot] = total_Selectivity;
functions_library.cu:        //     total_Selectivity = total_Selectivity + CUDA_current_gen_Parent_data[parent][selectivity_Index];
functions_library.cu:        // CUDA_hotspot_selectivity_Summations[tid] = total_Selectivity;
functions_library.cu:__global__ void CUDA_Selectivity_Distribution_CELLS(int num_Hotspots, int count_Parents, float **CUDA_current_gen_Parent_data, float **CUDA_hotspot_selectivity_Summations, float **CUDA_fitness_distribution, int start_Index, int **CUDA_parent_Indexes_parents_and_their_Cells)
functions_library.cu:        int cell_ID = CUDA_parent_Indexes_parents_and_their_Cells[1][parent_Index];
functions_library.cu:            CUDA_fitness_distribution[parent_Index][hotspot] = CUDA_current_gen_Parent_data[parent_Index][selectivity_Index] / CUDA_hotspot_selectivity_Summations[cell_ID][hotspot];
functions_library.cu:__global__ void CUDA_Selectivity_Distribution(int num_Hotspots, int count_Parents, float **CUDA_current_gen_Parent_data, float *CUDA_hotspot_selectivity_Summations, float **CUDA_fitness_distribution, int start_Index)
functions_library.cu:            CUDA_fitness_distribution[hotspot][parent_Index] = CUDA_current_gen_Parent_data[parent_Index][selectivity_Index] / CUDA_hotspot_selectivity_Summations[hotspot];
functions_library.cu:__global__ void CUDA_assign_parents_Recombination(curandState *states, int total_Progeny, int **progeny_recom_Index_Cuda, int num_Hotspots, float **cell_hotspot_selectivity_distribution, int *parent_and_their_cell_CUDA, int **cell_and_their_viruses_CUDA, int *per_Cell_max_viruses_CUDA, float **CUDA_current_gen_Parent_data, int start_Index)
functions_library.cu:        int parent = progeny_recom_Index_Cuda[progeny_Index][0];
functions_library.cu:        int cell_of_Parent = parent_and_their_cell_CUDA[parent];
functions_library.cu:        int putative_parents_in_Cell = per_Cell_max_viruses_CUDA[cell_of_Parent];
functions_library.cu:            if (progeny_recom_Index_Cuda[progeny_Index][recom_Hotspot + 1] != -1)
functions_library.cu:                progeny_recom_Index_Cuda[progeny_Index][recom_Hotspot + 1] = cell_and_their_viruses_CUDA[cell_of_Parent][virus_Parent_Index];
functions_library.cu:void functions_library::progeny_Recombination_parents_array(int **progeny_recom_Index_Cuda, int total_Progeny, int num_Hotspots,
functions_library.cu:                                                            int *parent_and_their_cell_CUDA, int **cell_and_their_viruses_CUDA, int *per_Cell_max_viruses_CUDA,
functions_library.cu:                                                            float **CUDA_current_gen_Parent_data, int max_Count, int num_Unique_cells)
functions_library.cu:    float **cell_hotspot_selectivity_distribution_CUDA;
functions_library.cu:    cudaMallocManaged(&cell_hotspot_selectivity_distribution_CUDA, ((max_Count * num_Hotspots) + 1) * num_Unique_cells * sizeof(int));
functions_library.cu:        cudaMalloc((void **)&tmp[i], ((max_Count * num_Hotspots) + 1) * sizeof(tmp[0][0]));
functions_library.cu:    cudaMemcpy(cell_hotspot_selectivity_distribution_CUDA, tmp, num_Unique_cells * sizeof(float *), cudaMemcpyHostToDevice);
functions_library.cu:    int full_Rounds = total_Cells / this->gpu_Limit;
functions_library.cu:    int partial_Rounds = total_Cells % this->gpu_Limit;
functions_library.cu:        int start = full * this->gpu_Limit;
functions_library.cu:        int stop = start + this->gpu_Limit;
functions_library.cu:        // CUDA_distribution_Cells_Selectivity(int num_Cells, int num_Hotspots, int **cell_and_their_viruses_CUDA, int *per_Cell_max_viruses_CUDA, float **CUDA_current_gen_Parent_data, float **cell_hotspot_selectivity_distribution, int start_Index)
functions_library.cu:        // CUDA_distribution_Cells_Selectivity<<<tot_Blocks, tot_ThreadsperBlock>>>(num_of_values_current, num_Hotspots, cell_and_their_viruses_CUDA, per_Cell_max_viruses_CUDA, CUDA_current_gen_Parent_data, cell_hotspot_selectivity_distribution_CUDA, start_stops[i].first);
functions_library.cu:        cudaError_t err = cudaGetLastError();
functions_library.cu:        if (err != cudaSuccess)
functions_library.cu:            printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:        cudaDeviceSynchronize();
functions_library.cu:    //     cudaMemcpy(test[i], cell_hotspot_selectivity_distribution_CUDA[i], ((max_Count * num_Hotspots) + 1) * sizeof(cell_hotspot_selectivity_distribution_CUDA[0][0]), cudaMemcpyDeviceToHost);
functions_library.cu:    full_Rounds = total_Cells / this->gpu_Limit;
functions_library.cu:    partial_Rounds = total_Cells % this->gpu_Limit;
functions_library.cu:        int start = full * this->gpu_Limit;
functions_library.cu:        int stop = start + this->gpu_Limit;
functions_library.cu:        cudaMalloc((void **)&state, num_of_values_current * num_Hotspots * sizeof(curandState));
functions_library.cu:        // CUDA_assign_parents_Recombination(curandState *states, int total_Progeny, int **progeny_recom_Index_Cuda, int num_Hotspots, float **cell_hotspot_selectivity_distribution, int *parent_and_their_cell_CUDA, int **cell_and_their_viruses_CUDA, int *per_Cell_max_viruses_CUDA, float **CUDA_current_gen_Parent_data, int start_Index)
functions_library.cu:        // CUDA_assign_parents_Recombination<<<tot_Blocks, tot_ThreadsperBlock>>>(state, num_of_values_current, progeny_recom_Index_Cuda, num_Hotspots, cell_hotspot_selectivity_distribution_CUDA, parent_and_their_cell_CUDA, cell_and_their_viruses_CUDA, per_Cell_max_viruses_CUDA, CUDA_current_gen_Parent_data, start_stops[i].first);
functions_library.cu:        cudaError_t err = cudaGetLastError();
functions_library.cu:        if (err != cudaSuccess)
functions_library.cu:            printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:        cudaDeviceSynchronize();
functions_library.cu:        cudaFree(state);
functions_library.cu:__global__ void CUDA_progeny_shuffle(int num_Hotspots, int **progeny_recom_Index_Cuda, int num_Progeny)
functions_library.cu:            int temp = progeny_recom_Index_Cuda[i][tid + 1];
functions_library.cu:            progeny_recom_Index_Cuda[i][tid + 1] = progeny_recom_Index_Cuda[j][tid + 1];
functions_library.cu:            progeny_recom_Index_Cuda[j][tid + 1] = temp;
functions_library.cu:__global__ void CUDA_progeny_shuffle_CELLs(int parents, int start_Index, int num_Hotspots, int **progeny_recom_Index_Cuda, int *CUDA_progeny_Stride_Index)
functions_library.cu:        int start = CUDA_progeny_Stride_Index[parent_Index];
functions_library.cu:        int stop = CUDA_progeny_Stride_Index[parent_Index + 1];
functions_library.cu:                int temp = progeny_recom_Index_Cuda[i][hotspot + 1];
functions_library.cu:                progeny_recom_Index_Cuda[i][hotspot + 1] = progeny_recom_Index_Cuda[j][hotspot + 1];
functions_library.cu:                progeny_recom_Index_Cuda[j][hotspot + 1] = temp;
functions_library.cu:void functions_library::progeny_Shuffle(int **progeny_recom_Index_Cuda, int num_Hotspots, int parents_in_current_generation, int *stride_Progeny_Index_CUDA)
functions_library.cu:    int full_Rounds = total_Cells / this->gpu_Limit;
functions_library.cu:    int partial_Rounds = total_Cells % this->gpu_Limit;
functions_library.cu:        int start = full * this->gpu_Limit;
functions_library.cu:        int stop = start + this->gpu_Limit;
functions_library.cu:        cudaMalloc((void **)&state, num_of_values_current * num_Hotspots * sizeof(curandState));
functions_library.cu:        // GPU shuffle function
functions_library.cu:        // CUDA_progeny_shuffle(curandState *states, int iterations, int **progeny_recom_Index_Cuda, int num_Hotspots, int *stride_Progeny_Index_CUDA, int start_Index)
functions_library.cu:        // CUDA_progeny_shuffle<<<tot_Blocks, tot_ThreadsperBlock>>>(state, num_of_values_current, progeny_recom_Index_Cuda, num_Hotspots, stride_Progeny_Index_CUDA, start_stops[i].first);
functions_library.cu:        cudaError_t err = cudaGetLastError();
functions_library.cu:        if (err != cudaSuccess)
functions_library.cu:            printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:        cudaDeviceSynchronize();
functions_library.cu:        cudaFree(state);
functions_library.cu:int **functions_library::create_Progeny_Array(int parents_in_current_generation, int *stride_Progeny_Index, int total_Progeny, int num_Hotspots, int **cuda_Progeny_numbers, int fill_Value)
functions_library.cu:    int **cuda_progeny_Array;
functions_library.cu:    cudaMallocManaged(&cuda_progeny_Array, (num_Hotspots + 2) * total_Progeny * sizeof(int));
functions_library.cu:        cudaMalloc((void **)&tmp[i], (num_Hotspots + 2) * sizeof(tmp[0][0]));
functions_library.cu:    cudaMemcpy(cuda_progeny_Array, tmp, total_Progeny * sizeof(int *), cudaMemcpyHostToDevice);
functions_library.cu:        cudaMemcpy(Progeny_numbers[i], cuda_Progeny_numbers[i], (parents_in_current_generation + 1) * sizeof(cuda_Progeny_numbers[0][0]), cudaMemcpyDeviceToHost);
functions_library.cu:        int full_Rounds = total_Cells / this->gpu_Limit;
functions_library.cu:        int partial_Rounds = total_Cells % this->gpu_Limit;
functions_library.cu:            int start = full * this->gpu_Limit;
functions_library.cu:            int stop = start + this->gpu_Limit;
functions_library.cu:            // progeny_Array_CUDA(int total_Progeny, int parent_ID, int parent_Start, int num_Recombination_hotspots, int **cuda_Progeny_numbers, int **cuda_progeny_Array, int fill_Value, int start_Index)
functions_library.cu:            // progeny_Array_CUDA<<<tot_Blocks, tot_ThreadsperBlock>>>(num_of_values_current, parent, parent_Start, num_Hotspots, cuda_Progeny_numbers, cuda_progeny_Array, fill_Value, start_stops[i].first);
functions_library.cu:            cudaError_t err = cudaGetLastError();
functions_library.cu:            if (err != cudaSuccess)
functions_library.cu:                printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:            cudaDeviceSynchronize();
functions_library.cu:    return cuda_progeny_Array;
functions_library.cu:__global__ void CUDA_progeny_Profiles_fill(int total_Progeny, float **CUDA_current_gen_Progeny_data, int num_Hotspots, int **progeny_recom_Index_Cuda, float **CUDA_current_gen_Parent_data, float *CUDA_parent_Proof_reading_probability, float *CUDA_progeny_Proof_reading_probability, int start_Index, int proof_reading_Activate_parent)
functions_library.cu:        int parent = progeny_recom_Index_Cuda[progeny_Index][0];
functions_library.cu:        CUDA_current_gen_Progeny_data[tid][0] = CUDA_current_gen_Parent_data[parent][0];
functions_library.cu:            CUDA_progeny_Proof_reading_probability[tid] = CUDA_parent_Proof_reading_probability[parent];
functions_library.cu:            int hotspot_Parent = progeny_recom_Index_Cuda[progeny_Index][hotspot + 1];
functions_library.cu:                CUDA_current_gen_Progeny_data[tid][i] = CUDA_current_gen_Parent_data[hotspot_Parent][i];
functions_library.cu:__global__ void CUDA_progeny_Profiles_fill_CELLs(int total_Progeny, float **CUDA_current_gen_Progeny_data, int num_Hotspots, int **progeny_recom_Index_Cuda, float **CUDA_current_gen_Parent_data, float *CUDA_parent_Proof_reading_probability, float *CUDA_progeny_Proof_reading_probability, int start_Index, int proof_reading_Activate_parent, float **CUDA_parent_survivability_Probabilities, float **CUDA_progeny_survivability_Probabilities)
functions_library.cu:        int parent = progeny_recom_Index_Cuda[progeny_Index][0];
functions_library.cu:        CUDA_current_gen_Progeny_data[progeny_Index][0] = CUDA_current_gen_Parent_data[parent][0];
functions_library.cu:            CUDA_progeny_Proof_reading_probability[progeny_Index] = CUDA_parent_Proof_reading_probability[parent];
functions_library.cu:        CUDA_progeny_survivability_Probabilities[progeny_Index][0] = CUDA_parent_survivability_Probabilities[parent][0];
functions_library.cu:            int hotspot_Parent = progeny_recom_Index_Cuda[progeny_Index][hotspot + 1];
functions_library.cu:            CUDA_progeny_survivability_Probabilities[progeny_Index][hotspot + 1] = CUDA_parent_survivability_Probabilities[hotspot_Parent][hotspot + 1];
functions_library.cu:                CUDA_current_gen_Progeny_data[progeny_Index][i] = CUDA_current_gen_Parent_data[hotspot_Parent][i];
functions_library.cu:float **functions_library::create_current_Progeny_data(int total_Progeny, int num_Hotspots, int **progeny_recom_Index_Cuda, float **CUDA_current_gen_Parent_data, float *CUDA_parent_Proof_reading_probability, float *CUDA_progeny_Proof_reading_probability)
functions_library.cu:    float **CUDA_current_gen_Progeny_data;
functions_library.cu:    cudaMallocManaged(&CUDA_progeny_Proof_reading_probability, total_Progeny * sizeof(float));
functions_library.cu:    // cudaMallocManaged(&CUDA_progeny_Proof_reading_probability, total_Progeny * sizeof(float));
functions_library.cu:    cudaMallocManaged(&CUDA_current_gen_Progeny_data, ((1 + (3 * num_Hotspots)) + 1) * total_Progeny * sizeof(float));
functions_library.cu:        cudaMalloc((void **)&tmp[i], ((1 + (3 * num_Hotspots)) + 1) * sizeof(tmp[0][0]));
functions_library.cu:    cudaMemcpy(CUDA_current_gen_Progeny_data, tmp, total_Progeny * sizeof(float *), cudaMemcpyHostToDevice);
functions_library.cu:    int full_Rounds = total_Cells / this->gpu_Limit;
functions_library.cu:    int partial_Rounds = total_Cells % this->gpu_Limit;
functions_library.cu:        int start = full * this->gpu_Limit;
functions_library.cu:        int stop = start + this->gpu_Limit;
functions_library.cu:        // CUDA_progeny_Profiles_fill(int total_Progeny, float **CUDA_current_gen_Progeny_data, int num_Hotspots, int **progeny_recom_Index_Cuda, float **CUDA_current_gen_Parent_data, int start_Index, float *CUDA_parent_Proof_reading_probability, float *CUDA_progeny_Proof_reading_probability)
functions_library.cu:        // CUDA_progeny_Profiles_fill<<<tot_Blocks, tot_ThreadsperBlock>>>(num_of_values_current, CUDA_current_gen_Progeny_data, num_Hotspots, progeny_recom_Index_Cuda, CUDA_current_gen_Parent_data, start_stops[i].first, CUDA_parent_Proof_reading_probability, CUDA_progeny_Proof_reading_probability);
functions_library.cu:        cudaError_t err = cudaGetLastError();
functions_library.cu:        if (err != cudaSuccess)
functions_library.cu:            printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:        cudaDeviceSynchronize();
functions_library.cu:    return CUDA_current_gen_Progeny_data;
functions_library.cu:__global__ void CUDA_progeny_sequence_generation(int total_Progeny, int start_Index, int **cuda_parent_sequences, int **cuda_progeny_Sequences, int **progeny_recom_Index_Cuda, int num_Hotspots, int genome_SIZE, int **CUDA_recombination_hotspots_start_stop)
functions_library.cu:        int parent = progeny_recom_Index_Cuda[progeny_Index][0];
functions_library.cu:        //     printf("%d %d %d %d \n", parent, CUDA_parent_Indexes[0], num_Unique_Parents, parent_Sequence_Index);
functions_library.cu:            cuda_progeny_Sequences[tid][base] = cuda_parent_sequences[parent][base];
functions_library.cu:            int hotspot_Parent = progeny_recom_Index_Cuda[progeny_Index][hotspot + 1];
functions_library.cu:                int start_Hotspot = CUDA_recombination_hotspots_start_stop[hotspot][0] - 1;
functions_library.cu:                int stop_Hotspot = CUDA_recombination_hotspots_start_stop[hotspot][1];
functions_library.cu:                    cuda_progeny_Sequences[tid][base] = cuda_parent_sequences[hotspot_Parent][base];
functions_library.cu:__global__ void CUDA_progeny_sequence_generation_CELLS(int total_Progeny, int start_Index, int **cuda_parent_sequences, int **cuda_progeny_Sequences, int **progeny_recom_Index_Cuda, int num_Hotspots, int genome_SIZE, int **CUDA_recombination_hotspots_start_stop)
functions_library.cu:        int parent = progeny_recom_Index_Cuda[progeny_Index][0];
functions_library.cu:        //     printf("%d %d %d %d \n", parent, CUDA_parent_Indexes[0], num_Unique_Parents, parent_Sequence_Index);
functions_library.cu:            cuda_progeny_Sequences[progeny_Index][base] = cuda_parent_sequences[parent][base];
functions_library.cu:            int hotspot_Parent = progeny_recom_Index_Cuda[progeny_Index][hotspot + 1];
functions_library.cu:                int start_Hotspot = CUDA_recombination_hotspots_start_stop[hotspot][0] - 1;
functions_library.cu:                int stop_Hotspot = CUDA_recombination_hotspots_start_stop[hotspot][1];
functions_library.cu:                    cuda_progeny_Sequences[progeny_Index][base] = cuda_parent_sequences[hotspot_Parent][base];
functions_library.cu:int **functions_library::create_progeny_Sequences(int **cuda_parent_Sequences, int **progeny_recom_Index_Cuda, int num_Hotspots, int total_Progeny, int genome_SIZE, int **CUDA_recombination_hotspots_start_stop)
functions_library.cu:    int **cuda_progeny_Sequences;
functions_library.cu:    cudaMallocManaged(&cuda_progeny_Sequences, (genome_SIZE + 1) * total_Progeny * sizeof(int));
functions_library.cu:        cudaMalloc((void **)&tmp[i], (genome_SIZE + 1) * sizeof(tmp[0][0]));
functions_library.cu:    cudaMemcpy(cuda_progeny_Sequences, tmp, total_Progeny * sizeof(int *), cudaMemcpyHostToDevice);
functions_library.cu:    int full_Rounds = total_Cells / this->gpu_Limit;
functions_library.cu:    int partial_Rounds = total_Cells % this->gpu_Limit;
functions_library.cu:        int start = full * this->gpu_Limit;
functions_library.cu:        int stop = start + this->gpu_Limit;
functions_library.cu:        // CUDA_progeny_sequence_generation(int total_Progeny, int start_Index, int **cuda_parent_sequences, int **cuda_progeny_Sequences, int **progeny_recom_Index_Cuda, int num_Hotspots, int genome_SIZE, int **CUDA_recombination_hotspots_start_stop)
functions_library.cu:        // CUDA_progeny_sequence_generation<<<tot_Blocks, tot_ThreadsperBlock>>>(num_of_values_current, start_stops[i].first, cuda_parent_Sequences, cuda_progeny_Sequences, progeny_recom_Index_Cuda, num_Hotspots, genome_SIZE, CUDA_recombination_hotspots_start_stop);
functions_library.cu:        cudaError_t err = cudaGetLastError();
functions_library.cu:        if (err != cudaSuccess)
functions_library.cu:            printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:        cudaDeviceSynchronize();
functions_library.cu:    return cuda_progeny_Sequences;
functions_library.cu:int **functions_library::Fill_2D_array_CUDA(int rows, int columns, int fill_Value, int **cuda_Progeny_numbers)
functions_library.cu:    int **cuda_Array_2D;
functions_library.cu:    cudaMallocManaged(&cuda_Array_2D, rows * sizeof(int *));
functions_library.cu:        cudaMalloc((void **)&(cuda_Array_2D[row]), (1 + columns) * sizeof(int));
functions_library.cu:    // cudaMallocManaged(&cuda_Array_2D, (columns + 1) * rows * sizeof(int));
functions_library.cu:    //     cudaMalloc((void **)&tmp[i], (columns + 1) * sizeof(tmp[0][0]));
functions_library.cu:    // cudaMemcpy(cuda_Array_2D, tmp, rows * sizeof(int *), cudaMemcpyHostToDevice);
functions_library.cu:    int full_Rounds = total_Cells / this->gpu_Limit;
functions_library.cu:    int partial_Rounds = total_Cells % this->gpu_Limit;
functions_library.cu:        int start = full * this->gpu_Limit;
functions_library.cu:        int stop = start + this->gpu_Limit;
functions_library.cu:        // cuda_Fill_2D_array_INT_Device(int total, int columns, int fill_Value, int **array_2D, int start_Index)
functions_library.cu:        cuda_Fill_2D_array_INT_Device<<<tot_Blocks, tot_ThreadsperBlock>>>(num_of_values_current, columns, fill_Value, cuda_Array_2D, start_stops[i].first);
functions_library.cu:        cudaError_t err = cudaGetLastError();
functions_library.cu:        if (err != cudaSuccess)
functions_library.cu:            printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:        cudaDeviceSynchronize();
functions_library.cu:    return cuda_Array_2D;
functions_library.cu:    cudaSetDevice(CUDA_device_IDs[0]);
functions_library.cu:    int **cuda_Array_2D;
functions_library.cu:    cudaMallocManaged(&cuda_Array_2D, rows * sizeof(int *));
functions_library.cu:        cudaMalloc((void **)&(cuda_Array_2D[row]), (1 + columns) * sizeof(int));
functions_library.cu:    // cudaMallocManaged(&cuda_Array_2D, (columns + 1) * rows * sizeof(int));
functions_library.cu:    //     cudaMalloc((void **)&tmp[i], (columns + 1) * sizeof(tmp[0][0]));
functions_library.cu:    // cudaMemcpy(cuda_Array_2D, tmp, rows * sizeof(int *), cudaMemcpyHostToDevice);
functions_library.cu:    cuda_Fill_2D_array<<<tot_Blocks, tot_ThreadsperBlock>>>((rows * columns), columns, fill_Value, cuda_Array_2D);
functions_library.cu:    cudaError_t err = cudaGetLastError();
functions_library.cu:    if (err != cudaSuccess)
functions_library.cu:        printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:    cudaDeviceSynchronize();
functions_library.cu:    // cudaMemcpy(array_2D, cuda_Array_2D, rows * sizeof(int), cudaMemcpyDeviceToHost);
functions_library.cu:        cudaMemcpy(array_2D[i], cuda_Array_2D[i], (columns + 1) * sizeof(int), cudaMemcpyDeviceToHost);
functions_library.cu:        cudaFree(cuda_Array_2D[row]);
functions_library.cu:    cudaFree(cuda_Array_2D);
functions_library.cu:float **functions_library::float_2D_Array_load_to_CUDA(float **host_Array, int rows, int columns)
functions_library.cu:    float **cuda_Array_2D;
functions_library.cu:    // cudaMallocManaged(&cuda_Array_2D, (columns + 1) * rows * sizeof(float));
functions_library.cu:    //     cudaMalloc((void **)&tmp[i], (columns + 1) * sizeof(tmp[0][0]));
functions_library.cu:    // cudaMemcpy(cuda_Array_2D, tmp, rows * sizeof(float *), cudaMemcpyHostToDevice);
functions_library.cu:    //     cudaMemcpy(tmp[i], host_Array[i], (columns + 1) * sizeof(cuda_Array_2D[0][0]), cudaMemcpyHostToDevice);
functions_library.cu:    cudaMallocManaged(&cuda_Array_2D, rows * sizeof(float *));
functions_library.cu:        cudaMalloc((void **)&(cuda_Array_2D[row]), columns * sizeof(float));
functions_library.cu:        cudaMemcpy(cuda_Array_2D[row], host_Array[row], columns * sizeof(float), cudaMemcpyHostToDevice);
functions_library.cu:    return cuda_Array_2D;
functions_library.cu:int **functions_library::int_2D_Array_load_to_CUDA(int **host_Array, int rows, int columns)
functions_library.cu:    int **cuda_Array_2D;
functions_library.cu:    // cudaMallocManaged(&cuda_Array_2D, (columns + 1) * rows * sizeof(int));
functions_library.cu:    //     cudaMalloc((void **)&tmp[i], (columns + 1) * sizeof(tmp[0][0]));
functions_library.cu:    // cudaMemcpy(cuda_Array_2D, tmp, rows * sizeof(int *), cudaMemcpyHostToDevice);
functions_library.cu:    //     cudaMemcpy(tmp[i], host_Array[i], (columns + 1) * sizeof(cuda_Array_2D[0][0]), cudaMemcpyHostToDevice);
functions_library.cu:    cudaMallocManaged(&cuda_Array_2D, rows * sizeof(int *));
functions_library.cu:        cudaMalloc((void **)&(cuda_Array_2D[row]), columns * sizeof(int));
functions_library.cu:        cudaMemcpy(cuda_Array_2D[row], host_Array[row], columns * sizeof(int), cudaMemcpyHostToDevice);
functions_library.cu:    return cuda_Array_2D;
functions_library.cu:    cudaSetDevice(CUDA_device_IDs[0]);
functions_library.cu:    float **cuda_Array_2D;
functions_library.cu:    cudaMallocManaged(&cuda_Array_2D, rows * sizeof(float *));
functions_library.cu:        cudaMalloc((void **)&(cuda_Array_2D[row]), (1 + columns) * sizeof(float));
functions_library.cu:    // cudaMallocManaged(&cuda_Array_2D, (columns + 1) * rows * sizeof(float));
functions_library.cu:    //     cudaMalloc((void **)&tmp[i], (columns + 1) * sizeof(tmp[0][0]));
functions_library.cu:    // cudaMemcpy(cuda_Array_2D, tmp, rows * sizeof(float *), cudaMemcpyHostToDevice);
functions_library.cu:    cuda_Fill_2D_array_Float<<<tot_Blocks, tot_ThreadsperBlock>>>((rows * columns), columns, fill_Value, cuda_Array_2D);
functions_library.cu:    cudaError_t err = cudaGetLastError();
functions_library.cu:    if (err != cudaSuccess)
functions_library.cu:        printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:    cudaDeviceSynchronize();
functions_library.cu:    // cudaMemcpy(array_2D, cuda_Array_2D, rows * sizeof(int), cudaMemcpyDeviceToHost);
functions_library.cu:        cudaMemcpy(array_2D[i], cuda_Array_2D[i], (columns + 1) * sizeof(float), cudaMemcpyDeviceToHost);
functions_library.cu:    // cudaFree(cuda_Array_2D);
functions_library.cu:        cudaFree(cuda_Array_2D[row]);
functions_library.cu:    cudaFree(cuda_Array_2D);
functions_library.cu:__global__ void CUDA_sum(int **input, int *output, int size)
functions_library.cu:int functions_library::sum_CUDA(int **cuda_Array_input, int num_Elements)
functions_library.cu:    int *CUDA_output;
functions_library.cu:    cudaMalloc((void **)&CUDA_output, num_Elements * sizeof(float));
functions_library.cu:    CUDA_sum<<<blocks_per_grid, threads_per_block>>>(cuda_Array_input, CUDA_output, num_Elements);
functions_library.cu:    cudaError_t err = cudaGetLastError();
functions_library.cu:    if (err != cudaSuccess)
functions_library.cu:        printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:    cudaDeviceSynchronize();
functions_library.cu:    cudaMemcpy(host_output, CUDA_output, blocks_per_grid * sizeof(int), cudaMemcpyDeviceToHost);
functions_library.cu:    cudaFree(CUDA_output);
functions_library.cu:__global__ void CUDA_mutate_Progeny_CELLS(int total_Progeny, int num_mutation_Hotspots, int generation,
functions_library.cu:                                          float **CUDA_mutation_rates_Hotspot_generation, int **CUDA_mutation_Regions_start_stop,
functions_library.cu:                                          int **cuda_progeny_Sequences, float *CUDA_progeny_Proof_reading_probability, int proof_reading_Activate,
functions_library.cu:                                          float **CUDA_A_0_mutation, float **CUDA_T_1_mutation, float **CUDA_G_2_mutation, float **CUDA_C_3_mutation,
functions_library.cu:                                          int **CUDA_sequence_Mutation_tracker, float **CUDA_current_gen_Progeny_data,
functions_library.cu:                                          float **CUDA_A_0_fitness, float **CUDA_T_1_fitness, float **CUDA_G_2_fitness, float **CUDA_C_3_fitness,
functions_library.cu:                                          float **CUDA_A_0_probability_Proof_reading, float **CUDA_T_1_probability_Proof_reading, float **CUDA_G_2_probability_Proof_reading, float **CUDA_C_3_probability_Proof_reading,
functions_library.cu:                                          float **CUDA_A_0_Recombination, float **CUDA_T_1_Recombination, float **CUDA_G_2_Recombination, float **CUDA_C_3_Recombination,
functions_library.cu:                                          int *CUDA_stride_Array, int start_Index,
functions_library.cu:                                          float **CUDA_progeny_survivability_Probabilities,
functions_library.cu:                                          float **CUDA_A_0_survivability, float **CUDA_T_1_survivability, float **CUDA_G_2_survivability, float **CUDA_C_3_survivability)
functions_library.cu:            float mean = CUDA_mutation_rates_Hotspot_generation[mutation_Hotspot][generation];
functions_library.cu:            int start = CUDA_mutation_Regions_start_stop[mutation_Hotspot][0] - 1;
functions_library.cu:            int stop = CUDA_mutation_Regions_start_stop[mutation_Hotspot][1] - 1;
functions_library.cu:                float proof_Reading_probability = CUDA_progeny_Proof_reading_probability[progeny_Index];
functions_library.cu:                int original_BASE = cuda_progeny_Sequences[progeny_Index][position];
functions_library.cu:                    cumulative_prob += (original_BASE == 0)   ? CUDA_A_0_mutation[mutation_Hotspot][base]
functions_library.cu:                                       : (original_BASE == 1) ? CUDA_T_1_mutation[mutation_Hotspot][base]
functions_library.cu:                                       : (original_BASE == 2) ? CUDA_G_2_mutation[mutation_Hotspot][base]
functions_library.cu:                                       : (original_BASE == 3) ? CUDA_C_3_mutation[mutation_Hotspot][base]
functions_library.cu:                    //     cumulative_prob += CUDA_A_0_mutation[mutation_Hotspot][base];
functions_library.cu:                    //     cumulative_prob += CUDA_T_1_mutation[mutation_Hotspot][base];
functions_library.cu:                    //     cumulative_prob += CUDA_G_2_mutation[mutation_Hotspot][base];
functions_library.cu:                    //     cumulative_prob += CUDA_C_3_mutation[mutation_Hotspot][base];
functions_library.cu:                    cuda_progeny_Sequences[progeny_Index][position] = new_BASE;
functions_library.cu:                        if (CUDA_sequence_Mutation_tracker[mutation_Tracker][position] != -1)
functions_library.cu:                                int fitness_Point = CUDA_sequence_Mutation_tracker[mutation_Tracker][position];
functions_library.cu:                                fitness_Change = (original_BASE == 0) ? CUDA_A_0_fitness[fitness_Point][new_BASE] : (original_BASE == 1) ? CUDA_T_1_fitness[fitness_Point][new_BASE]
functions_library.cu:                                                                                                                : (original_BASE == 2)   ? CUDA_G_2_fitness[fitness_Point][new_BASE]
functions_library.cu:                                                                                                                : (original_BASE == 3)   ? CUDA_C_3_fitness[fitness_Point][new_BASE]
functions_library.cu:                                CUDA_current_gen_Progeny_data[progeny_Index][0] = CUDA_current_gen_Progeny_data[progeny_Index][0] * fitness_Change;
functions_library.cu:                                int proof_Point = CUDA_sequence_Mutation_tracker[mutation_Tracker][position];
functions_library.cu:                                proof_Change = (original_BASE == 0) ? CUDA_A_0_probability_Proof_reading[proof_Point][new_BASE] : (original_BASE == 1) ? CUDA_T_1_probability_Proof_reading[proof_Point][new_BASE]
functions_library.cu:                                                                                                                              : (original_BASE == 2)   ? CUDA_G_2_probability_Proof_reading[proof_Point][new_BASE]
functions_library.cu:                                                                                                                              : (original_BASE == 3)   ? CUDA_C_3_probability_Proof_reading[proof_Point][new_BASE]
functions_library.cu:                                CUDA_progeny_Proof_reading_probability[progeny_Index] = CUDA_progeny_Proof_reading_probability[progeny_Index] + proof_Change;
functions_library.cu:                                if (CUDA_progeny_Proof_reading_probability[progeny_Index] < 0)
functions_library.cu:                                    CUDA_progeny_Proof_reading_probability[progeny_Index] = 0;
functions_library.cu:                                else if (CUDA_progeny_Proof_reading_probability[progeny_Index] > 1)
functions_library.cu:                                    CUDA_progeny_Proof_reading_probability[progeny_Index] = 1;
functions_library.cu:                                int proof_Point = CUDA_sequence_Mutation_tracker[mutation_Tracker][position];
functions_library.cu:                                proof_Change = (original_BASE == 0) ? CUDA_A_0_survivability[proof_Point][new_BASE] : (original_BASE == 1) ? CUDA_T_1_survivability[proof_Point][new_BASE]
functions_library.cu:                                                                                                                  : (original_BASE == 2)   ? CUDA_G_2_survivability[proof_Point][new_BASE]
functions_library.cu:                                                                                                                  : (original_BASE == 3)   ? CUDA_C_3_survivability[proof_Point][new_BASE]
functions_library.cu:                                CUDA_progeny_survivability_Probabilities[progeny_Index][0] = CUDA_progeny_survivability_Probabilities[progeny_Index][0] + proof_Change;
functions_library.cu:                                // if (CUDA_progeny_survivability_Probabilities[progeny_Index][0] < 0)
functions_library.cu:                                //     CUDA_progeny_survivability_Probabilities[progeny_Index][0] = 0;
functions_library.cu:                                // else if (CUDA_progeny_survivability_Probabilities[progeny_Index][0] > 1)
functions_library.cu:                                //     CUDA_progeny_survivability_Probabilities[progeny_Index][0] = 1;
functions_library.cu:                                int num_of_Recombination_events = CUDA_sequence_Mutation_tracker[mutation_Tracker][position];
functions_library.cu:                                    start = CUDA_stride_Array[3];
functions_library.cu:                                    stop = CUDA_stride_Array[4];
functions_library.cu:                                    start = CUDA_stride_Array[mutation_Tracker - 1];
functions_library.cu:                                    stop = CUDA_stride_Array[mutation_Tracker];
functions_library.cu:                                    if (original_BASE == 0 && CUDA_A_0_Recombination[rows][1] == (position + 1))
functions_library.cu:                                        hotspot = (int)CUDA_A_0_Recombination[rows][0];
functions_library.cu:                                        change = CUDA_A_0_Recombination[rows][new_BASE + 2];
functions_library.cu:                                        // CUDA_current_gen_Progeny_data[progeny_Index][index_Change] = CUDA_current_gen_Progeny_data[progeny_Index][index_Change] + probability_Change;
functions_library.cu:                                    else if (original_BASE == 1 && CUDA_T_1_Recombination[rows][1] == (position + 1))
functions_library.cu:                                        hotspot = (int)CUDA_T_1_Recombination[rows][0];
functions_library.cu:                                        change = CUDA_T_1_Recombination[rows][new_BASE + 2];
functions_library.cu:                                        // CUDA_current_gen_Progeny_data[progeny_Index][index_Change] = CUDA_current_gen_Progeny_data[progeny_Index][index_Change] + probability_Change;
functions_library.cu:                                    else if (original_BASE == 2 && CUDA_G_2_Recombination[rows][1] == (position + 1))
functions_library.cu:                                        hotspot = (int)CUDA_G_2_Recombination[rows][0];
functions_library.cu:                                        change = CUDA_G_2_Recombination[rows][new_BASE + 2];
functions_library.cu:                                        // CUDA_current_gen_Progeny_data[progeny_Index][index_Change] = CUDA_current_gen_Progeny_data[progeny_Index][index_Change] + probability_Change;
functions_library.cu:                                    else if (original_BASE == 2 && CUDA_C_3_Recombination[rows][1] == (position + 1))
functions_library.cu:                                        hotspot = (int)CUDA_C_3_Recombination[rows][0];
functions_library.cu:                                        change = CUDA_C_3_Recombination[rows][new_BASE + 2];
functions_library.cu:                                        // CUDA_current_gen_Progeny_data[progeny_Index][index_Change] = CUDA_current_gen_Progeny_data[progeny_Index][index_Change] + probability_Change;
functions_library.cu:                                            CUDA_current_gen_Progeny_data[progeny_Index][index_Change] = CUDA_current_gen_Progeny_data[progeny_Index][index_Change] + change;
functions_library.cu:                                            if (CUDA_current_gen_Progeny_data[progeny_Index][index_Change] < 0)
functions_library.cu:                                                CUDA_current_gen_Progeny_data[progeny_Index][index_Change] = 0;
functions_library.cu:                                            else if (CUDA_current_gen_Progeny_data[progeny_Index][index_Change] > 1)
functions_library.cu:                                                CUDA_current_gen_Progeny_data[progeny_Index][index_Change] = 1;
functions_library.cu:                                            CUDA_progeny_survivability_Probabilities[progeny_Index][hotspot + 1] = CUDA_progeny_survivability_Probabilities[progeny_Index][hotspot + 1] + change;
functions_library.cu:                                            // if (CUDA_progeny_survivability_Probabilities[progeny_Index][hotspot + 1] < 0)
functions_library.cu:                                            //     CUDA_progeny_survivability_Probabilities[progeny_Index][hotspot + 1] = 0;
functions_library.cu:                                            // else if (CUDA_progeny_survivability_Probabilities[progeny_Index][hotspot + 1] > 1)
functions_library.cu:                                            //     CUDA_progeny_survivability_Probabilities[progeny_Index][hotspot + 1] = 1;
functions_library.cu:                                            CUDA_current_gen_Progeny_data[progeny_Index][index_Change] = CUDA_current_gen_Progeny_data[progeny_Index][index_Change] * change;
functions_library.cu:__global__ void CUDA_mutate_Progeny(int total_Progeny, int num_mutation_Hotspots, int generation,
functions_library.cu:                                    float **CUDA_mutation_rates_Hotspot_generation, int **CUDA_mutation_Regions_start_stop,
functions_library.cu:                                    int **cuda_progeny_Sequences, float *CUDA_progeny_Proof_reading_probability, int proof_reading_Activate,
functions_library.cu:                                    float **CUDA_A_0_mutation, float **CUDA_T_1_mutation, float **CUDA_G_2_mutation, float **CUDA_C_3_mutation,
functions_library.cu:                                    int **CUDA_sequence_Mutation_tracker, float **CUDA_current_gen_Progeny_data,
functions_library.cu:                                    float **CUDA_A_0_fitness, float **CUDA_T_1_fitness, float **CUDA_G_2_fitness, float **CUDA_C_3_fitness,
functions_library.cu:                                    float **CUDA_A_0_probability_Proof_reading, float **CUDA_T_1_probability_Proof_reading, float **CUDA_G_2_probability_Proof_reading, float **CUDA_C_3_probability_Proof_reading,
functions_library.cu:                                    float **CUDA_A_0_Recombination, float **CUDA_T_1_Recombination, float **CUDA_G_2_Recombination, float **CUDA_C_3_Recombination,
functions_library.cu:                                    int *CUDA_stride_Array)
functions_library.cu:            float mean = CUDA_mutation_rates_Hotspot_generation[mutation_Hotspot][generation];
functions_library.cu:            int start = CUDA_mutation_Regions_start_stop[mutation_Hotspot][0] - 1;
functions_library.cu:            int stop = CUDA_mutation_Regions_start_stop[mutation_Hotspot][1] - 1;
functions_library.cu:                float proof_Reading_probability = CUDA_progeny_Proof_reading_probability[tid];
functions_library.cu:                int original_BASE = cuda_progeny_Sequences[tid][position];
functions_library.cu:                    cumulative_prob += (original_BASE == 0)   ? CUDA_A_0_mutation[mutation_Hotspot][base]
functions_library.cu:                                       : (original_BASE == 1) ? CUDA_T_1_mutation[mutation_Hotspot][base]
functions_library.cu:                                       : (original_BASE == 2) ? CUDA_G_2_mutation[mutation_Hotspot][base]
functions_library.cu:                                       : (original_BASE == 3) ? CUDA_C_3_mutation[mutation_Hotspot][base]
functions_library.cu:                    //     cumulative_prob += CUDA_A_0_mutation[mutation_Hotspot][base];
functions_library.cu:                    //     cumulative_prob += CUDA_T_1_mutation[mutation_Hotspot][base];
functions_library.cu:                    //     cumulative_prob += CUDA_G_2_mutation[mutation_Hotspot][base];
functions_library.cu:                    //     cumulative_prob += CUDA_C_3_mutation[mutation_Hotspot][base];
functions_library.cu:                    cuda_progeny_Sequences[tid][position] = new_BASE;
functions_library.cu:                        if (CUDA_sequence_Mutation_tracker[mutation_Tracker][position] != -1)
functions_library.cu:                                int fitness_Point = CUDA_sequence_Mutation_tracker[mutation_Tracker][position];
functions_library.cu:                                fitness_Change = (original_BASE == 0) ? CUDA_A_0_fitness[fitness_Point][new_BASE] : (original_BASE == 1) ? CUDA_T_1_fitness[fitness_Point][new_BASE]
functions_library.cu:                                                                                                                : (original_BASE == 2)   ? CUDA_G_2_fitness[fitness_Point][new_BASE]
functions_library.cu:                                                                                                                : (original_BASE == 3)   ? CUDA_C_3_fitness[fitness_Point][new_BASE]
functions_library.cu:                                CUDA_current_gen_Progeny_data[tid][0] = CUDA_current_gen_Progeny_data[tid][0] * fitness_Change;
functions_library.cu:                                int proof_Point = CUDA_sequence_Mutation_tracker[mutation_Tracker][position];
functions_library.cu:                                proof_Change = (original_BASE == 0) ? CUDA_A_0_probability_Proof_reading[proof_Point][new_BASE] : (original_BASE == 1) ? CUDA_T_1_probability_Proof_reading[proof_Point][new_BASE]
functions_library.cu:                                                                                                                              : (original_BASE == 2)   ? CUDA_G_2_probability_Proof_reading[proof_Point][new_BASE]
functions_library.cu:                                                                                                                              : (original_BASE == 3)   ? CUDA_C_3_probability_Proof_reading[proof_Point][new_BASE]
functions_library.cu:                                CUDA_progeny_Proof_reading_probability[tid] = CUDA_progeny_Proof_reading_probability[tid] + proof_Change;
functions_library.cu:                                if (CUDA_progeny_Proof_reading_probability[tid] < 0)
functions_library.cu:                                    CUDA_progeny_Proof_reading_probability[tid] = 0;
functions_library.cu:                                else if (CUDA_progeny_Proof_reading_probability[tid] > 1)
functions_library.cu:                                    CUDA_progeny_Proof_reading_probability[tid] = 1;
functions_library.cu:                                int num_of_Recombination_events = CUDA_sequence_Mutation_tracker[mutation_Tracker][position];
functions_library.cu:                                int start = CUDA_stride_Array[mutation_Tracker - 1];
functions_library.cu:                                int stop = CUDA_stride_Array[mutation_Tracker];
functions_library.cu:                                    if (original_BASE == 0 && CUDA_A_0_Recombination[rows][1] == (position + 1))
functions_library.cu:                                        hotspot = (int)CUDA_A_0_Recombination[rows][0];
functions_library.cu:                                        change = CUDA_A_0_Recombination[rows][new_BASE + 2];
functions_library.cu:                                        // CUDA_current_gen_Progeny_data[progeny_Index][index_Change] = CUDA_current_gen_Progeny_data[progeny_Index][index_Change] + probability_Change;
functions_library.cu:                                    else if (original_BASE == 1 && CUDA_T_1_Recombination[rows][1] == (position + 1))
functions_library.cu:                                        hotspot = (int)CUDA_T_1_Recombination[rows][0];
functions_library.cu:                                        change = CUDA_T_1_Recombination[rows][new_BASE + 2];
functions_library.cu:                                        // CUDA_current_gen_Progeny_data[progeny_Index][index_Change] = CUDA_current_gen_Progeny_data[progeny_Index][index_Change] + probability_Change;
functions_library.cu:                                    else if (original_BASE == 2 && CUDA_G_2_Recombination[rows][1] == (position + 1))
functions_library.cu:                                        hotspot = (int)CUDA_G_2_Recombination[rows][0];
functions_library.cu:                                        change = CUDA_G_2_Recombination[rows][new_BASE + 2];
functions_library.cu:                                        // CUDA_current_gen_Progeny_data[progeny_Index][index_Change] = CUDA_current_gen_Progeny_data[progeny_Index][index_Change] + probability_Change;
functions_library.cu:                                    else if (original_BASE == 2 && CUDA_C_3_Recombination[rows][1] == (position + 1))
functions_library.cu:                                        hotspot = (int)CUDA_C_3_Recombination[rows][0];
functions_library.cu:                                        change = CUDA_C_3_Recombination[rows][new_BASE + 2];
functions_library.cu:                                        // CUDA_current_gen_Progeny_data[progeny_Index][index_Change] = CUDA_current_gen_Progeny_data[progeny_Index][index_Change] + probability_Change;
functions_library.cu:                                            CUDA_current_gen_Progeny_data[tid][index_Change] = CUDA_current_gen_Progeny_data[tid][index_Change] + change;
functions_library.cu:                                            if (CUDA_current_gen_Progeny_data[tid][index_Change] < 0)
functions_library.cu:                                                CUDA_current_gen_Progeny_data[tid][index_Change] = 0;
functions_library.cu:                                            else if (CUDA_current_gen_Progeny_data[tid][index_Change] > 1)
functions_library.cu:                                                CUDA_current_gen_Progeny_data[tid][index_Change] = 1;
functions_library.cu:                                            CUDA_current_gen_Progeny_data[tid][index_Change] = CUDA_current_gen_Progeny_data[tid][index_Change] * change;
functions_library.cu:void functions_library::mutate_Sequences(int **cuda_progeny_Sequences, float **CUDA_current_gen_Progeny_data, float *CUDA_progeny_Proof_reading_probability,
functions_library.cu:                                         float **CUDA_mutation_rates_Hotspot_generation, int **CUDA_mutation_Regions_start_stop, int **CUDA_sequence_Mutation_tracker,
functions_library.cu:    int full_Rounds = total_Cells / this->gpu_Limit;
functions_library.cu:    int partial_Rounds = total_Cells % this->gpu_Limit;
functions_library.cu:        int start = full * this->gpu_Limit;
functions_library.cu:        int stop = start + this->gpu_Limit;
functions_library.cu:        // CUDA_mutate_Progeny(int total_Progeny, int start_Index, int num_mutation_Hotspots, int generation,
functions_library.cu:        //                             float **CUDA_mutation_rates_Hotspot_generation, int **CUDA_mutation_Regions_start_stop,
functions_library.cu:        //                             int **cuda_progeny_Sequences, float *CUDA_progeny_Proof_reading_probability, int proof_reading_Activate,
functions_library.cu:        //                             float **CUDA_A_0_mutation, float **CUDA_T_1_mutation, float **CUDA_G_2_mutation, float **CUDA_C_3_mutation,
functions_library.cu:        //                             int **CUDA_sequence_Mutation_tracker, float **CUDA_current_gen_Progeny_data,
functions_library.cu:        //                             float **CUDA_A_0_fitness, float **CUDA_T_1_fitness, float **CUDA_G_2_fitness, float **CUDA_C_3_fitness,
functions_library.cu:        //                             float **CUDA_A_0_probability_Proof_reading, float **CUDA_T_1_probability_Proof_reading, float **CUDA_G_2_probability_Proof_reading, float **CUDA_C_3_probability_Proof_reading,
functions_library.cu:        //                             float **CUDA_A_0_Recombination, float **CUDA_T_1_Recombination, float **CUDA_G_2_Recombination, float **CUDA_C_3_Recombination,
functions_library.cu:        //                             int *CUDA_stride_Array)
functions_library.cu:        // CUDA_mutate_Progeny<<<tot_Blocks, tot_ThreadsperBlock>>>(num_of_values_current, start_stops[i].first, num_Mutation_hotspots, current_Generation,
functions_library.cu:        //                                                          CUDA_mutation_rates_Hotspot_generation, CUDA_mutation_Regions_start_stop,
functions_library.cu:        //                                                          cuda_progeny_Sequences, CUDA_progeny_Proof_reading_probability, proof_Reading_Activate,
functions_library.cu:        //                                                          CUDA_A_0_mutation, CUDA_T_1_mutation, CUDA_G_2_mutation, CUDA_C_3_mutation,
functions_library.cu:        //                                                          CUDA_sequence_Mutation_tracker, CUDA_current_gen_Progeny_data,
functions_library.cu:        //                                                          CUDA_A_0_fitness, CUDA_T_1_fitness, CUDA_G_2_fitness, CUDA_C_3_fitness,
functions_library.cu:        //                                                          CUDA_A_0_probability_Proof_reading, CUDA_T_1_probability_Proof_reading, CUDA_G_2_probability_Proof_reading, CUDA_C_3_probability_Proof_reading,
functions_library.cu:        //                                                          CUDA_A_0_Recombination, CUDA_T_1_Recombination, CUDA_G_2_Recombination, CUDA_C_3_Recombination,
functions_library.cu:        //                                                          CUDA_stride_Array);
functions_library.cu:        cudaError_t err = cudaGetLastError();
functions_library.cu:        if (err != cudaSuccess)
functions_library.cu:            printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:        cudaDeviceSynchronize();
functions_library.cu:__global__ void cuda_fill_parent_Sequences(int genome_Size, int **master_Sequences, char *cuda_reference, int ref_Num)
functions_library.cu:        char base = cuda_reference[tid];
functions_library.cu:void functions_library::find_Unique_values(int **progeny_recom_Index_Cuda, int total_Elements, int num_Recom_hotspots,
functions_library.cu:                                           int **CUDA_recombination_hotspots_start_stop,
functions_library.cu:                                           float **CUDA_current_gen_Progeny_data, float *CUDA_progeny_Proof_reading_probability, int current_Generation, float **CUDA_mutation_rates_Hotspot_generation, int **CUDA_mutation_Regions_start_stop, int **CUDA_sequence_Mutation_tracker, int proof_Reading_Activate,
functions_library.cu:    // int **cuda_Test_Array = int_2D_Array_load_to_CUDA(Array_2D, 5, 20);
functions_library.cu:    // int *CUDA_num_Unique;
functions_library.cu:    // cudaMalloc(&CUDA_num_Unique, sizeof(int));
functions_library.cu:    int *CUDA_unique_values_Array, *unique_values_Array;
functions_library.cu:    // cudaMemcpy(CUDA_num_Unique, &num_Unique, sizeof(int), cudaMemcpyHostToDevice);
functions_library.cu:    cudaMalloc(&CUDA_unique_values_Array, (total_Elements * columns_Total) * sizeof(int));
functions_library.cu:    unique_Values<<<tot_Blocks, tot_ThreadsperBlock>>>(total_Elements, start_Index, columns_Total, progeny_recom_Index_Cuda, CUDA_unique_values_Array);
functions_library.cu:    cudaError_t err = cudaGetLastError();
functions_library.cu:    if (err != cudaSuccess)
functions_library.cu:        printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:    cudaDeviceSynchronize();
functions_library.cu:    cudaMemcpy(unique_values_Array, CUDA_unique_values_Array, sizeof(int) * (total_Elements * columns_Total), cudaMemcpyDeviceToHost);
functions_library.cu:    // cudaMemcpy(&num_Unique, CUDA_num_Unique, sizeof(int), cudaMemcpyDeviceToHost);
functions_library.cu:    cudaFree(CUDA_unique_values_Array);
functions_library.cu:    // cudaFree(CUDA_num_Unique);
functions_library.cu:    int **CUDA_all_Parent_Sequences;
functions_library.cu:    cudaMallocManaged(&CUDA_all_Parent_Sequences, (genome_Size + 1) * num_Unique_Parents * sizeof(int));
functions_library.cu:        cudaMalloc((void **)&tmp[i], (genome_Size + 1) * sizeof(tmp[0][0]));
functions_library.cu:    cudaMemcpy(CUDA_all_Parent_Sequences, tmp, num_Unique_Parents * sizeof(int *), cudaMemcpyHostToDevice);
functions_library.cu:    cout << "Loading parents to the GPU" << endl;
functions_library.cu:        char *reference_full, *cuda_reference;
functions_library.cu:        cudaMallocManaged(&cuda_reference, (sequence.size() + 1) * sizeof(char));
functions_library.cu:        cudaMemcpy(cuda_reference, reference_full, (sequence.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
functions_library.cu:        cuda_fill_parent_Sequences<<<tot_Blocks, tot_ThreadsperBlock>>>(genome_Size, CUDA_all_Parent_Sequences, cuda_reference, sequence_i);
functions_library.cu:        cudaError_t err = cudaGetLastError();
functions_library.cu:        if (err != cudaSuccess)
functions_library.cu:            printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:        cudaDeviceSynchronize();
functions_library.cu:    cout << num_Unique_Parents << " parents loaded to the GPU" << endl;
functions_library.cu:    //     cudaMemcpy(test[i], CUDA_all_Parent_Sequences[i], (genome_Size + 1) * sizeof(CUDA_all_Parent_Sequences[0][0]), cudaMemcpyDeviceToHost);
functions_library.cu:    int *CUDA_parent_Indexes;
functions_library.cu:    cudaMallocManaged(&CUDA_parent_Indexes, num_Unique_Parents * sizeof(int));
functions_library.cu:    cudaMemcpy(CUDA_parent_Indexes, parent_Indexes, num_Unique_Parents * sizeof(int), cudaMemcpyHostToDevice);
functions_library.cu:    //         cudaMemcpy(progeny_recom_Index[i], progeny_recom_Index_Cuda[i], (4) * sizeof(progeny_recom_Index_Cuda[0][0]), cudaMemcpyDeviceToHost);
functions_library.cu:    int **cuda_progeny_Sequences;
functions_library.cu:    cudaMallocManaged(&cuda_progeny_Sequences, (genome_Size + 1) * total_Elements * sizeof(int));
functions_library.cu:        cudaMalloc((void **)&tmp[i], (genome_Size + 1) * sizeof(tmp[0][0]));
functions_library.cu:    cudaMemcpy(cuda_progeny_Sequences, tmp, total_Elements * sizeof(int *), cudaMemcpyHostToDevice);
functions_library.cu:    // CUDA_progeny_sequence_generation<<<tot_Blocks, tot_ThreadsperBlock>>>(total_Elements, start_Index, CUDA_all_Parent_Sequences, cuda_progeny_Sequences, progeny_recom_Index_Cuda, num_Recom_hotspots, genome_Size, CUDA_recombination_hotspots_start_stop, CUDA_parent_Indexes, num_Unique_Parents);
functions_library.cu:    err = cudaGetLastError();
functions_library.cu:    if (err != cudaSuccess)
functions_library.cu:        printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:    cudaDeviceSynchronize();
functions_library.cu:    //         cudaMemcpy(progeny_Sequences[i], cuda_progeny_Sequences[i], (genome_Size + 1) * sizeof(cuda_progeny_Sequences[0][0]), cudaMemcpyDeviceToHost);
functions_library.cu:    cudaFree(CUDA_all_Parent_Sequences);
functions_library.cu:    cudaFree(CUDA_parent_Indexes);
functions_library.cu:        // CUDA_mutate_Progeny<<<tot_Blocks, tot_ThreadsperBlock>>>(total_Elements, start_Index, mutation_Activate, current_Generation,
functions_library.cu:        //                                                          CUDA_mutation_rates_Hotspot_generation, CUDA_mutation_Regions_start_stop,
functions_library.cu:        //                                                          cuda_progeny_Sequences, CUDA_progeny_Proof_reading_probability, proof_Reading_Activate,
functions_library.cu:        //                                                          CUDA_A_0_mutation, CUDA_T_1_mutation, CUDA_G_2_mutation, CUDA_C_3_mutation,
functions_library.cu:        //                                                          CUDA_sequence_Mutation_tracker, CUDA_current_gen_Progeny_data,
functions_library.cu:        //                                                          CUDA_A_0_fitness, CUDA_T_1_fitness, CUDA_G_2_fitness, CUDA_C_3_fitness,
functions_library.cu:        //                                                          CUDA_A_0_probability_Proof_reading, CUDA_T_1_probability_Proof_reading, CUDA_G_2_probability_Proof_reading, CUDA_C_3_probability_Proof_reading,
functions_library.cu:        //                                                          CUDA_A_0_Recombination, CUDA_T_1_Recombination, CUDA_G_2_Recombination, CUDA_C_3_Recombination,
functions_library.cu:        //                                                          CUDA_stride_Array);
functions_library.cu:        err = cudaGetLastError();
functions_library.cu:        if (err != cudaSuccess)
functions_library.cu:            printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:        cudaDeviceSynchronize();
functions_library.cu:        cudaMemcpy(progeny_Sequences[i], cuda_progeny_Sequences[i], (genome_Size + 1) * sizeof(cuda_progeny_Sequences[0][0]), cudaMemcpyDeviceToHost);
functions_library.cu:    cudaFree(cuda_progeny_Sequences);
functions_library.cu:                                          int **progeny_recom_Index_Cuda, int num_Hotspots, int **CUDA_recombination_hotspots_start_stop,
functions_library.cu:                                          float **CUDA_current_gen_Progeny_data, float *CUDA_progeny_Proof_reading_probability, float **CUDA_mutation_rates_Hotspot_generation, int **CUDA_mutation_Regions_start_stop, int **CUDA_sequence_Mutation_tracker, int proof_Reading_Activate,
functions_library.cu:    int full_Rounds = total_Cells / this->gpu_Limit;
functions_library.cu:    int partial_Rounds = total_Cells % this->gpu_Limit;
functions_library.cu:        int start = full * this->gpu_Limit;
functions_library.cu:        int stop = start + this->gpu_Limit;
functions_library.cu:        // find_Unique_values(int **CUDA_Array_2D, int total_Elements, int num_Recom_hotspots,
functions_library.cu:        find_Unique_values(progeny_recom_Index_Cuda, num_of_values_current, num_Hotspots,
functions_library.cu:                           CUDA_recombination_hotspots_start_stop,
functions_library.cu:                           CUDA_current_gen_Progeny_data, CUDA_progeny_Proof_reading_probability, generation, CUDA_mutation_rates_Hotspot_generation, CUDA_mutation_Regions_start_stop, CUDA_sequence_Mutation_tracker, proof_Reading_Activate,
functions_library.cu:void functions_library::clear_Array_INT(int **CUDA_2D_array, int rows)
functions_library.cu:    cudaSetDevice(CUDA_device_IDs[0]);
functions_library.cu:        cudaFree(CUDA_2D_array[i]);
functions_library.cu:    cudaFree(CUDA_2D_array);
functions_library.cu:void functions_library::clear_Array_FLOAT(float **CUDA_2D_array, int rows)
functions_library.cu:    cudaSetDevice(CUDA_device_IDs[0]);
functions_library.cu:        cudaFree(CUDA_2D_array[i]);
functions_library.cu:    cudaFree(CUDA_2D_array);
functions_library.cu:        //! clear gpu = DONE
functions_library.cu:        float **CUDA_current_gen_Parent_data = float_2D_Array_load_to_CUDA(current_gen_Parent_data, count_Parents, 1 + (3 * recombination_hotspots));
functions_library.cu:        cout << "Parents loaded to GPU" << endl;
functions_library.cu:        //! clear gpu = done
functions_library.cu:        int **cuda_Progeny_numbers = create_CUDA_2D_int(recombination_hotspots + 1, count_Parents);
functions_library.cu:        int full_Rounds = count_Parents / this->gpu_Limit;
functions_library.cu:        int partial_Rounds = count_Parents % this->gpu_Limit;
functions_library.cu:            int start = full * this->gpu_Limit;
functions_library.cu:            int stop = start + this->gpu_Limit;
functions_library.cu:            cudaMalloc((void **)&state, num_of_values_current * sizeof(curandState));
functions_library.cu:                //CUDA_gamma_distribution_PROGENY<<<tot_Blocks, tot_ThreadsperBlock>>>(state, num_of_values_current, cuda_Progeny_numbers, progeny_shape, progeny_scale, start_stops[i].first, recombination_hotspots, CUDA_current_gen_Parent_data);
functions_library.cu:            cudaError_t err = cudaGetLastError();
functions_library.cu:            if (err != cudaSuccess)
functions_library.cu:                printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:            cudaDeviceSynchronize();
functions_library.cu:            cudaFree(state);
functions_library.cu:        cout << "Completed progeny generation via " << start_stops.size() << " GPU rounds" << endl;
functions_library.cu:        int sum_Progeny = sum_CUDA(cuda_Progeny_numbers, count_Parents);
functions_library.cu:        //! clear gpu = done
functions_library.cu:        float **CUDA_fitness_distribution;
functions_library.cu:        //! clear gpu = done
functions_library.cu:        int *CUDA_per_Cell_parents_Stride;
functions_library.cu:        cudaMallocManaged(&CUDA_per_Cell_parents_Stride, (num_of_Cells + 1) * sizeof(int));
functions_library.cu:        cudaMemcpy(CUDA_per_Cell_parents_Stride, per_Cell_parents_Stride, (num_of_Cells + 1) * sizeof(int), cudaMemcpyHostToDevice);
functions_library.cu:        //! clear gpu = done
functions_library.cu:        int **CUDA_parent_Indexes_parents_and_their_Cells = int_2D_Array_load_to_CUDA(parent_Indexes_parents_and_their_Cells, 2, count_Parents);
functions_library.cu:            //! clear gpu = DONE;
functions_library.cu:            float **CUDA_hotspot_selectivity_Summations = create_CUDA_2D_FLOAT(num_of_Cells, recombination_hotspots);
functions_library.cu:            full_Rounds = num_of_Cells / this->gpu_Limit;
functions_library.cu:            partial_Rounds = num_of_Cells % this->gpu_Limit;
functions_library.cu:                int start = full * this->gpu_Limit;
functions_library.cu:                int stop = start + this->gpu_Limit;
functions_library.cu:                // CUDA_summation_Selectivity_CELLS(int cells, int start_Index, int num_Hotspots, float **CUDA_current_gen_Parent_data, float **CUDA_hotspot_selectivity_Summations, int *CUDA_per_Cell_parents_Stride)
functions_library.cu:                CUDA_summation_Selectivity_CELLS<<<tot_Blocks, tot_ThreadsperBlock>>>(num_of_values_current, start_stops[i].first, recombination_hotspots, CUDA_current_gen_Parent_data, CUDA_hotspot_selectivity_Summations, CUDA_per_Cell_parents_Stride);
functions_library.cu:                cudaError_t err = cudaGetLastError();
functions_library.cu:                if (err != cudaSuccess)
functions_library.cu:                    printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:                cudaDeviceSynchronize();
functions_library.cu:            // float **test_2d = load_to_Host_FLOAT(CUDA_hotspot_selectivity_Summations, num_of_Cells, recombination_hotspots);
functions_library.cu:            CUDA_fitness_distribution = create_CUDA_2D_FLOAT(count_Parents, recombination_hotspots);
functions_library.cu:            full_Rounds = total_Cells / this->gpu_Limit;
functions_library.cu:            partial_Rounds = total_Cells % this->gpu_Limit;
functions_library.cu:                int start = full * this->gpu_Limit;
functions_library.cu:                int stop = start + this->gpu_Limit;
functions_library.cu:                CUDA_Selectivity_Distribution_CELLS<<<tot_Blocks, tot_ThreadsperBlock>>>(recombination_hotspots, num_of_values_current, CUDA_current_gen_Parent_data, CUDA_hotspot_selectivity_Summations, CUDA_fitness_distribution, start_stops[i].first, CUDA_parent_Indexes_parents_and_their_Cells);
functions_library.cu:                cudaError_t err = cudaGetLastError();
functions_library.cu:                if (err != cudaSuccess)
functions_library.cu:                    printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:                cudaDeviceSynchronize();
functions_library.cu:            // float **test_Distributions = load_to_Host_FLOAT(CUDA_fitness_distribution, count_Parents, recombination_hotspots);
functions_library.cu:            clear_Array_FLOAT(CUDA_hotspot_selectivity_Summations, num_of_Cells);
functions_library.cu:        int **Progeny_numbers = load_to_Host(cuda_Progeny_numbers, (recombination_hotspots + 1), count_Parents);
functions_library.cu:        //! clear gpu = DONE
functions_library.cu:        int **progeny_recom_Index_Cuda = create_CUDA_2D_int(sum_Progeny, recombination_hotspots + 1);
functions_library.cu:            full_Rounds = total_Cells / this->gpu_Limit;
functions_library.cu:            partial_Rounds = total_Cells % this->gpu_Limit;
functions_library.cu:                int start = full * this->gpu_Limit;
functions_library.cu:                int stop = start + this->gpu_Limit;
functions_library.cu:                progeny_Array_CUDA_CELLS<<<tot_Blocks, tot_ThreadsperBlock>>>(num_of_values_current, progeny_Stride_Index[parent], start_stops[i].first, parent, recombination_hotspots, cuda_Progeny_numbers, progeny_recom_Index_Cuda, -1, CUDA_fitness_distribution, CUDA_per_Cell_parents_Stride, CUDA_parent_Indexes_parents_and_their_Cells);
functions_library.cu:                cudaError_t err = cudaGetLastError();
functions_library.cu:                if (err != cudaSuccess)
functions_library.cu:                    printf("CUDA Error 1: %s\n", cudaGetErrorString(err));
functions_library.cu:                cudaDeviceSynchronize();
functions_library.cu:        cudaFree(CUDA_per_Cell_parents_Stride);
functions_library.cu:        clear_Array_INT(CUDA_parent_Indexes_parents_and_their_Cells, 2);
functions_library.cu:        // int **test_Load = load_to_Host(progeny_recom_Index_Cuda, sum_Progeny, recombination_hotspots + 1);
functions_library.cu:            clear_Array_FLOAT(CUDA_fitness_distribution, count_Parents);
functions_library.cu:            int *CUDA_progeny_Stride_Index;
functions_library.cu:            cudaMallocManaged(&CUDA_progeny_Stride_Index, (count_Parents + 1) * sizeof(int));
functions_library.cu:            cudaMemcpy(CUDA_progeny_Stride_Index, progeny_Stride_Index, (count_Parents + 1) * sizeof(int), cudaMemcpyHostToDevice);
functions_library.cu:            full_Rounds = total_Cells / this->gpu_Limit;
functions_library.cu:            partial_Rounds = total_Cells % this->gpu_Limit;
functions_library.cu:                int start = full * this->gpu_Limit;
functions_library.cu:                int stop = start + this->gpu_Limit;
functions_library.cu:                CUDA_progeny_shuffle_CELLs<<<tot_Blocks, tot_ThreadsperBlock>>>(num_of_values_current, start_stops[i].first, recombination_hotspots, progeny_recom_Index_Cuda, CUDA_progeny_Stride_Index);
functions_library.cu:                cudaError_t err = cudaGetLastError();
functions_library.cu:                if (err != cudaSuccess)
functions_library.cu:                    printf("CUDA Error 1: %s\n", cudaGetErrorString(err));
functions_library.cu:                cudaDeviceSynchronize();
functions_library.cu:            cudaFree(CUDA_progeny_Stride_Index);
functions_library.cu:        clear_Array_INT(cuda_Progeny_numbers, (recombination_hotspots + 1));
functions_library.cu:        // int **test_Load = load_to_Host(progeny_recom_Index_Cuda, sum_Progeny, recombination_hotspots + 1);
functions_library.cu:        //! clear gpu = DONE
functions_library.cu:        float *CUDA_parent_Proof_reading_probability;
functions_library.cu:            CUDA_parent_Proof_reading_probability = copy_1D_to_CUDA_FLOAT(parent_Proof_reading_probability, count_Parents);
functions_library.cu:        //! CLEAR GPU = done
functions_library.cu:        float **CUDA_parent_survivability_Probabilities;
functions_library.cu:        CUDA_parent_survivability_Probabilities = float_2D_Array_load_to_CUDA(parent_survivability_Probabilities, count_Parents, recombination_hotspots + 1);
functions_library.cu:        //! CLEAR GPU DEFINITELY = DONE
functions_library.cu:        float **CUDA_current_gen_Progeny_data;
functions_library.cu:        CUDA_current_gen_Progeny_data = create_CUDA_2D_FLOAT(sum_Progeny, 1 + (3 * recombination_hotspots));
functions_library.cu:        float *CUDA_progeny_Proof_reading_probability;
functions_library.cu:            cudaMallocManaged(&CUDA_progeny_Proof_reading_probability, sum_Progeny * sizeof(float));
functions_library.cu:        //! CLEAR GPU = done
functions_library.cu:        float **CUDA_progeny_survivability_Probabilities = create_CUDA_2D_FLOAT(sum_Progeny, 1 + recombination_hotspots);
functions_library.cu:        full_Rounds = total_Cells / this->gpu_Limit;
functions_library.cu:        partial_Rounds = total_Cells % this->gpu_Limit;
functions_library.cu:            int start = full * this->gpu_Limit;
functions_library.cu:            int stop = start + this->gpu_Limit;
functions_library.cu:            CUDA_progeny_Profiles_fill_CELLs<<<tot_Blocks, tot_ThreadsperBlock>>>(num_of_values_current, CUDA_current_gen_Progeny_data, recombination_hotspots, progeny_recom_Index_Cuda, CUDA_current_gen_Parent_data, CUDA_parent_Proof_reading_probability, CUDA_progeny_Proof_reading_probability, start_stops[i].first, proof_reading_Activate_parent, CUDA_parent_survivability_Probabilities, CUDA_progeny_survivability_Probabilities);
functions_library.cu:            cudaError_t err = cudaGetLastError();
functions_library.cu:            if (err != cudaSuccess)
functions_library.cu:                printf("CUDA Error 1: %s\n", cudaGetErrorString(err));
functions_library.cu:            cudaDeviceSynchronize();
functions_library.cu:        clear_Array_FLOAT(CUDA_current_gen_Parent_data, count_Parents);
functions_library.cu:            cudaFree(CUDA_parent_Proof_reading_probability);
functions_library.cu:        clear_Array_FLOAT(CUDA_parent_survivability_Probabilities, count_Parents);
functions_library.cu:        // float **test_Load_2 = load_to_Host_FLOAT(CUDA_progeny_survivability_Probabilities, sum_Progeny, recombination_hotspots + 1);
functions_library.cu:        // cudaMemcpy(test_prob, CUDA_progeny_Proof_reading_probability, sizeof(float) * sum_Progeny, cudaMemcpyDeviceToHost);
functions_library.cu:        // float **test_Load_3 = load_to_Host_FLOAT(CUDA_current_gen_Progeny_data, sum_Progeny, 1 + (3 * recombination_hotspots));
functions_library.cu:        cout << "Loading parents to the GPU" << endl;
functions_library.cu:        //! CLEAR from gpu = DONE
functions_library.cu:        int **CUDA_all_Parent_Sequences;
functions_library.cu:        CUDA_all_Parent_Sequences = create_CUDA_2D_int(count_Parents, genome_Size);
functions_library.cu:            char *reference_full, *cuda_reference;
functions_library.cu:            cudaMallocManaged(&cuda_reference, (sequence.size() + 1) * sizeof(char));
functions_library.cu:            cudaMemcpy(cuda_reference, reference_full, (sequence.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
functions_library.cu:            cuda_fill_parent_Sequences<<<tot_Blocks, tot_ThreadsperBlock>>>(genome_Size, CUDA_all_Parent_Sequences, cuda_reference, parent);
functions_library.cu:            cudaError_t err = cudaGetLastError();
functions_library.cu:            if (err != cudaSuccess)
functions_library.cu:                printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:            cudaDeviceSynchronize();
functions_library.cu:        cout << count_Parents << " parents loaded to the GPU" << endl;
functions_library.cu:        // int **sequences_Test = load_to_Host(CUDA_all_Parent_Sequences, count_Parents, genome_Size);
functions_library.cu:        //! CLEAR FROM GPU = DONE
functions_library.cu:        int **cuda_progeny_Sequences = create_CUDA_2D_int(sum_Progeny, genome_Size);
functions_library.cu:        full_Rounds = total_Cells / this->gpu_Limit;
functions_library.cu:        partial_Rounds = total_Cells % this->gpu_Limit;
functions_library.cu:            int start = full * this->gpu_Limit;
functions_library.cu:            int stop = start + this->gpu_Limit;
functions_library.cu:            CUDA_progeny_sequence_generation_CELLS<<<tot_Blocks, tot_ThreadsperBlock>>>(num_of_values_current, start_stops[i].first, CUDA_all_Parent_Sequences, cuda_progeny_Sequences, progeny_recom_Index_Cuda, recombination_hotspots, genome_Size, CUDA_recombination_hotspots_start_stop);
functions_library.cu:            cudaError_t err = cudaGetLastError();
functions_library.cu:            if (err != cudaSuccess)
functions_library.cu:                printf("CUDA Error 1: %s\n", cudaGetErrorString(err));
functions_library.cu:            cudaDeviceSynchronize();
functions_library.cu:        clear_Array_INT(CUDA_all_Parent_Sequences, count_Parents);
functions_library.cu:        int **progeny_recom_Index = load_to_Host(progeny_recom_Index_Cuda, sum_Progeny, recombination_hotspots + 1);
functions_library.cu:        clear_Array_INT(progeny_recom_Index_Cuda, sum_Progeny);
functions_library.cu:        // int **sequences_Test = load_to_Host(cuda_progeny_Sequences, sum_Progeny, genome_Size);
functions_library.cu:                CUDA_mutate_Progeny_CELLS<<<tot_Blocks, tot_ThreadsperBlock>>>(num_of_values_current, mutation_hotspots, generation_Current,
functions_library.cu:                                                                               CUDA_mutation_rates_Hotspot_generation, CUDA_mutation_Regions_start_stop,
functions_library.cu:                                                                               cuda_progeny_Sequences, CUDA_progeny_Proof_reading_probability, proof_reading_Activate_parent,
functions_library.cu:                                                                               CUDA_A_0_mutation, CUDA_T_1_mutation, CUDA_G_2_mutation, CUDA_C_3_mutation,
functions_library.cu:                                                                               CUDA_sequence_Mutation_tracker, CUDA_current_gen_Progeny_data,
functions_library.cu:                                                                               CUDA_A_0_fitness, CUDA_T_1_fitness, CUDA_G_2_fitness, CUDA_C_3_fitness,
functions_library.cu:                                                                               CUDA_A_0_probability_Proof_reading, CUDA_T_1_probability_Proof_reading, CUDA_G_2_probability_Proof_reading, CUDA_C_3_probability_Proof_reading,
functions_library.cu:                                                                               CUDA_A_0_Recombination, CUDA_T_1_Recombination, CUDA_G_2_Recombination, CUDA_C_3_Recombination,
functions_library.cu:                                                                               CUDA_stride_Array, start_stops[i].first,
functions_library.cu:                                                                               CUDA_progeny_survivability_Probabilities,
functions_library.cu:                                                                               CUDA_A_0_survivability, CUDA_T_1_survivability, CUDA_G_2_survivability, CUDA_C_3_survivability);
functions_library.cu:                cudaError_t err = cudaGetLastError();
functions_library.cu:                if (err != cudaSuccess)
functions_library.cu:                    printf("CUDA Error 1: %s\n", cudaGetErrorString(err));
functions_library.cu:                cudaDeviceSynchronize();
functions_library.cu:        // int **sequences_Test_change = load_to_Host(cuda_progeny_Sequences, sum_Progeny, genome_Size);
functions_library.cu:        // float **test_Load_2 = load_to_Host_FLOAT(CUDA_current_gen_Progeny_data, sum_Progeny, 1 + (3 * recombination_hotspots));
functions_library.cu:        // test_Load_2 = load_to_Host_FLOAT(CUDA_progeny_survivability_Probabilities, sum_Progeny, recombination_hotspots + 1);
functions_library.cu:        float **current_gen_Progeny_data = load_to_Host_FLOAT(CUDA_current_gen_Progeny_data, sum_Progeny, 1 + (3 * recombination_hotspots));
functions_library.cu:        clear_Array_FLOAT(CUDA_current_gen_Progeny_data, sum_Progeny);
functions_library.cu:        float **progeny_survivability_Probabilities = load_to_Host_FLOAT(CUDA_progeny_survivability_Probabilities, sum_Progeny, 1 + recombination_hotspots);
functions_library.cu:        clear_Array_FLOAT(CUDA_progeny_survivability_Probabilities, sum_Progeny);
functions_library.cu:            cudaMemcpy(progeny_Proof_reading_probability, CUDA_progeny_Proof_reading_probability, sum_Progeny * sizeof(float), cudaMemcpyDeviceToHost);
functions_library.cu:            cudaFree(CUDA_progeny_Proof_reading_probability);
functions_library.cu:        int **progeny_Sequences = load_to_Host(cuda_progeny_Sequences, sum_Progeny, genome_Size);
functions_library.cu:        clear_Array_INT(cuda_progeny_Sequences, sum_Progeny);
functions_library.cu:            float **CUDA_current_gen_Parent_data = float_2D_Array_load_to_CUDA(current_gen_Parent_data, count_Parents, 1 + (3 * recombination_hotspots));
functions_library.cu:            float *CUDA_parent_Proof_reading_probability;
functions_library.cu:                CUDA_parent_Proof_reading_probability = copy_1D_to_CUDA_FLOAT(parent_Proof_reading_probability, count_Parents);
functions_library.cu:            int **cuda_Progeny_numbers = create_CUDA_2D_int(recombination_hotspots + 1, count_Parents);
functions_library.cu:            // cudaMallocManaged(&cuda_Progeny_numbers, (count_Parents + 1) * (recombination_hotspots + 1) * sizeof(int));
functions_library.cu:            //     cudaMalloc((void **)&tmp[i], (count_Parents + 1) * sizeof(tmp[0][0]));
functions_library.cu:            // cudaMemcpy(cuda_Progeny_numbers, tmp, (recombination_hotspots + 1) * sizeof(int *), cudaMemcpyHostToDevice);
functions_library.cu:            int full_Rounds = count_Parents / this->gpu_Limit;
functions_library.cu:            int partial_Rounds = count_Parents % this->gpu_Limit;
functions_library.cu:                int start = full * this->gpu_Limit;
functions_library.cu:                int stop = start + this->gpu_Limit;
functions_library.cu:                cudaMalloc((void **)&state, num_of_values_current * sizeof(curandState));
functions_library.cu:                   // CUDA_gamma_distribution_PROGENY<<<tot_Blocks, tot_ThreadsperBlock>>>(state, num_of_values_current, cuda_Progeny_numbers, progeny_shape, progeny_scale, start_stops[i].first, recombination_hotspots, CUDA_current_gen_Parent_data);
functions_library.cu:                cudaError_t err = cudaGetLastError();
functions_library.cu:                if (err != cudaSuccess)
functions_library.cu:                    printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:                cudaDeviceSynchronize();
functions_library.cu:                cudaFree(state);
functions_library.cu:            cout << "Completed progeny generation via " << start_stops.size() << " GPU rounds" << endl;
functions_library.cu:            int **Progeny_numbers = load_to_Host(cuda_Progeny_numbers, (recombination_hotspots + 1), count_Parents);
functions_library.cu:            int sum_Progeny = sum_CUDA(cuda_Progeny_numbers, count_Parents);
functions_library.cu:            float **CUDA_fitness_distribution;
functions_library.cu:            CUDA_fitness_distribution = create_CUDA_2D_FLOAT(recombination_hotspots, count_Parents);
functions_library.cu:                float *CUDA_hotspot_selectivity_Summations;
functions_library.cu:                cudaMallocManaged(&CUDA_hotspot_selectivity_Summations, recombination_hotspots * sizeof(float));
functions_library.cu:                // run gpu to get summations
functions_library.cu:                CUDA_summation_Selectivity<<<tot_Blocks, tot_ThreadsperBlock>>>(recombination_hotspots, count_Parents, CUDA_current_gen_Parent_data, CUDA_hotspot_selectivity_Summations);
functions_library.cu:                cudaError_t err = cudaGetLastError();
functions_library.cu:                if (err != cudaSuccess)
functions_library.cu:                    printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:                cudaDeviceSynchronize();
functions_library.cu:                // cudaMemcpy(summations, CUDA_hotspot_selectivity_Summations, recombination_hotspots * sizeof(float), cudaMemcpyDeviceToHost);
functions_library.cu:                full_Rounds = total_Cells / this->gpu_Limit;
functions_library.cu:                partial_Rounds = total_Cells % this->gpu_Limit;
functions_library.cu:                    int start = full * this->gpu_Limit;
functions_library.cu:                    int stop = start + this->gpu_Limit;
functions_library.cu:                    CUDA_Selectivity_Distribution<<<tot_Blocks, tot_ThreadsperBlock>>>(recombination_hotspots, num_of_values_current, CUDA_current_gen_Parent_data, CUDA_hotspot_selectivity_Summations, CUDA_fitness_distribution, start_stops[i].first);
functions_library.cu:                    err = cudaGetLastError();
functions_library.cu:                    if (err != cudaSuccess)
functions_library.cu:                        printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:                    cudaDeviceSynchronize();
functions_library.cu:                cudaFree(CUDA_hotspot_selectivity_Summations);
functions_library.cu:            // float **test_Array = load_to_Host_FLOAT(CUDA_fitness_distribution, recombination_hotspots, count_Parents);
functions_library.cu:            cout << "Loading parents to the GPU" << endl;
functions_library.cu:            int **CUDA_all_Parent_Sequences;
functions_library.cu:            CUDA_all_Parent_Sequences = create_CUDA_2D_int(count_Parents, genome_Size);
functions_library.cu:                char *reference_full, *cuda_reference;
functions_library.cu:                cudaMallocManaged(&cuda_reference, (sequence.size() + 1) * sizeof(char));
functions_library.cu:                cudaMemcpy(cuda_reference, reference_full, (sequence.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
functions_library.cu:                cuda_fill_parent_Sequences<<<tot_Blocks, tot_ThreadsperBlock>>>(genome_Size, CUDA_all_Parent_Sequences, cuda_reference, parent);
functions_library.cu:                cudaError_t err = cudaGetLastError();
functions_library.cu:                if (err != cudaSuccess)
functions_library.cu:                    printf("CUDA Error: %s\n", cudaGetErrorString(err));
functions_library.cu:                cudaDeviceSynchronize();
functions_library.cu:            cout << count_Parents << " parents loaded to the GPU" << endl;
functions_library.cu:                int **progeny_recom_Index_Cuda = create_CUDA_2D_int(Progeny_numbers[0][parent], recombination_hotspots + 1);
functions_library.cu:                full_Rounds = total_Cells / this->gpu_Limit;
functions_library.cu:                partial_Rounds = total_Cells % this->gpu_Limit;
functions_library.cu:                    int start = full * this->gpu_Limit;
functions_library.cu:                    int stop = start + this->gpu_Limit;
functions_library.cu:                    progeny_Array_CUDA<<<tot_Blocks, tot_ThreadsperBlock>>>(num_of_values_current, parent, recombination_hotspots, cuda_Progeny_numbers, progeny_recom_Index_Cuda, fill_Value, start_stops[i].first, CUDA_fitness_distribution, count_Parents);
functions_library.cu:                    cudaError_t err = cudaGetLastError();
functions_library.cu:                    if (err != cudaSuccess)
functions_library.cu:                        printf("CUDA Error 1: %s\n", cudaGetErrorString(err));
functions_library.cu:                    cudaDeviceSynchronize();
functions_library.cu:                // int **test_Load = load_to_Host(progeny_recom_Index_Cuda, total_Cells, recombination_hotspots + 1);
functions_library.cu:                    CUDA_progeny_shuffle<<<tot_Blocks, tot_ThreadsperBlock>>>(recombination_hotspots, progeny_recom_Index_Cuda, Progeny_numbers[0][parent]);
functions_library.cu:                    cudaError_t err = cudaGetLastError();
functions_library.cu:                    if (err != cudaSuccess)
functions_library.cu:                        printf("CUDA Error 2: %s\n", cudaGetErrorString(err));
functions_library.cu:                    cudaDeviceSynchronize();
functions_library.cu:                    // int** test_Load = load_to_Host(progeny_recom_Index_Cuda, total_Cells, recombination_hotspots + 1);
functions_library.cu:                int **progeny_recom_Index = load_to_Host(progeny_recom_Index_Cuda, Progeny_numbers[0][parent], recombination_hotspots + 1);
functions_library.cu:                    float **CUDA_current_gen_Progeny_data = create_CUDA_2D_FLOAT(num_of_values_current, 1 + (3 * recombination_hotspots));
functions_library.cu:                    float *CUDA_progeny_Proof_reading_probability;
functions_library.cu:                        cudaMallocManaged(&CUDA_progeny_Proof_reading_probability, num_of_values_current * sizeof(float));
functions_library.cu:                    CUDA_progeny_Profiles_fill<<<tot_Blocks, tot_ThreadsperBlock>>>(num_of_values_current, CUDA_current_gen_Progeny_data, recombination_hotspots, progeny_recom_Index_Cuda, CUDA_current_gen_Parent_data, CUDA_parent_Proof_reading_probability, CUDA_progeny_Proof_reading_probability, start_stops[i].first, proof_reading_Activate_parent);
functions_library.cu:                    cudaError_t err = cudaGetLastError();
functions_library.cu:                    if (err != cudaSuccess)
functions_library.cu:                        printf("CUDA Error 3: %s\n", cudaGetErrorString(err));
functions_library.cu:                    cudaDeviceSynchronize();
functions_library.cu:                    int **cuda_progeny_Sequences = create_CUDA_2D_int(num_of_values_current, genome_Size);
functions_library.cu:                    CUDA_progeny_sequence_generation<<<tot_Blocks, tot_ThreadsperBlock>>>(num_of_values_current, start_stops[i].first, CUDA_all_Parent_Sequences, cuda_progeny_Sequences, progeny_recom_Index_Cuda, recombination_hotspots, genome_Size, CUDA_recombination_hotspots_start_stop);
functions_library.cu:                    err = cudaGetLastError();
functions_library.cu:                    if (err != cudaSuccess)
functions_library.cu:                        printf("CUDA Error 4: %s\n", cudaGetErrorString(err));
functions_library.cu:                    cudaDeviceSynchronize();
functions_library.cu:                    // int **progeny_Sequences = load_to_Host(cuda_progeny_Sequences, num_of_values_current, genome_Size);
functions_library.cu:                    // cudaMemcpy(progeny_Proof_reading_probability, CUDA_progeny_Proof_reading_probability, num_of_values_current * sizeof(float), cudaMemcpyDeviceToHost);
functions_library.cu:                        CUDA_mutate_Progeny<<<tot_Blocks, tot_ThreadsperBlock>>>(num_of_values_current, mutation_hotspots, generation_Current,
functions_library.cu:                                                                                 CUDA_mutation_rates_Hotspot_generation, CUDA_mutation_Regions_start_stop,
functions_library.cu:                                                                                 cuda_progeny_Sequences, CUDA_progeny_Proof_reading_probability, proof_reading_Activate_parent,
functions_library.cu:                                                                                 CUDA_A_0_mutation, CUDA_T_1_mutation, CUDA_G_2_mutation, CUDA_C_3_mutation,
functions_library.cu:                                                                                 CUDA_sequence_Mutation_tracker, CUDA_current_gen_Progeny_data,
functions_library.cu:                                                                                 CUDA_A_0_fitness, CUDA_T_1_fitness, CUDA_G_2_fitness, CUDA_C_3_fitness,
functions_library.cu:                                                                                 CUDA_A_0_probability_Proof_reading, CUDA_T_1_probability_Proof_reading, CUDA_G_2_probability_Proof_reading, CUDA_C_3_probability_Proof_reading,
functions_library.cu:                                                                                 CUDA_A_0_Recombination, CUDA_T_1_Recombination, CUDA_G_2_Recombination, CUDA_C_3_Recombination,
functions_library.cu:                                                                                 CUDA_stride_Array);
functions_library.cu:                        err = cudaGetLastError();
functions_library.cu:                        if (err != cudaSuccess)
functions_library.cu:                            printf("CUDA Error 5: %s\n", cudaGetErrorString(err));
functions_library.cu:                        cudaDeviceSynchronize();
functions_library.cu:                    float **current_gen_Progeny_data = load_to_Host_FLOAT(CUDA_current_gen_Progeny_data, num_of_values_current, 1 + (3 * recombination_hotspots));
functions_library.cu:                    clear_Array_FLOAT(CUDA_current_gen_Progeny_data, num_of_values_current);
functions_library.cu:                    // cudaFree(CUDA_current_gen_Progeny_data);
functions_library.cu:                        cudaMemcpy(progeny_Proof_reading_probability, CUDA_progeny_Proof_reading_probability, num_of_values_current * sizeof(float), cudaMemcpyDeviceToHost);
functions_library.cu:                        cudaFree(CUDA_progeny_Proof_reading_probability);
functions_library.cu:                    int **progeny_Sequences = load_to_Host(cuda_progeny_Sequences, num_of_values_current, genome_Size);
functions_library.cu:                    clear_Array_INT(cuda_progeny_Sequences, num_of_values_current);
functions_library.cu:                    // cudaFree(cuda_progeny_Sequences);
functions_library.cu:                    //     // progeny_Sequences = load_to_Host(cuda_progeny_Sequences, num_of_values_current, genome_Size);
functions_library.cu:                    //     // cudaMemcpy(progeny_Proof_reading_probability, CUDA_progeny_Proof_reading_probability, num_of_values_current * sizeof(float), cudaMemcpyDeviceToHost);
functions_library.cu:                    //     // current_gen_Progeny_data = load_to_Host_FLOAT(CUDA_current_gen_Progeny_data, num_of_values_current, 1 + (3 * recombination_hotspots));
functions_library.cu:                // cudaFree(progeny_recom_Index_Cuda);
functions_library.cu:                clear_Array_INT(progeny_recom_Index_Cuda, Progeny_numbers[0][parent]);
functions_library.cu:            // cudaFree(cuda_Progeny_numbers);
functions_library.cu:            clear_Array_INT(cuda_Progeny_numbers, recombination_hotspots + 1);
functions_library.cu:            // cudaFree(CUDA_all_Parent_Sequences);
functions_library.cu:            clear_Array_INT(CUDA_all_Parent_Sequences, count_Parents);
functions_library.cu:            cudaFree(CUDA_parent_Proof_reading_probability);
functions_library.cu:                // cudaFree(CUDA_fitness_distribution);
functions_library.cu:                clear_Array_FLOAT(CUDA_fitness_distribution, recombination_hotspots);
functions_library.cu:                cudaFree(CUDA_parent_Proof_reading_probability);
functions_library.cu:int **functions_library::create_CUDA_2D_int(int rows, int columns)
functions_library.cu:    cudaSetDevice(CUDA_device_IDs[0]);
functions_library.cu:    int **CUDA_array;
functions_library.cu:    cudaMallocManaged(&CUDA_array, rows * sizeof(int *));
functions_library.cu:        cudaMalloc((void **)&(CUDA_array[row]), (1 + columns) * sizeof(int));
functions_library.cu:    // cudaMallocManaged(&CUDA_array, (columns + 1) * rows * sizeof(int));
functions_library.cu:    //     cudaMalloc((void **)&tmp[i], (columns + 1) * sizeof(tmp[0][0]));
functions_library.cu:    // cudaMemcpy(CUDA_array, tmp, rows * sizeof(int *), cudaMemcpyHostToDevice);
functions_library.cu:    return CUDA_array;
functions_library.cu:float **functions_library::create_CUDA_2D_FLOAT(int rows, int columns)
functions_library.cu:    cudaSetDevice(CUDA_device_IDs[0]);
functions_library.cu:    float **CUDA_array;
functions_library.cu:    cudaMallocManaged(&CUDA_array, rows * sizeof(float *));
functions_library.cu:        cudaMalloc((void **)&(CUDA_array[row]), (1 + columns) * sizeof(float));
functions_library.cu:    // cudaMallocManaged(&CUDA_array, (columns + 1) * rows * sizeof(float));
functions_library.cu:    //     cudaMalloc((void **)&tmp[i], (columns + 1) * sizeof(tmp[0][0]));
functions_library.cu:    // cudaMemcpy(CUDA_array, tmp, rows * sizeof(float *), cudaMemcpyHostToDevice);
functions_library.cu:    return CUDA_array;
functions_library.cu:float **functions_library::load_to_Host_FLOAT(float **cuda_2D_Array, int rows, int columns)
functions_library.cu:    cudaSetDevice(CUDA_device_IDs[0]);
functions_library.cu:        cudaMemcpy(Array_host_2D[row], cuda_2D_Array[row], (columns + 1) * sizeof(cuda_2D_Array[0][0]), cudaMemcpyDeviceToHost);
functions_library.cu:int **functions_library::load_to_Host(int **cuda_2D_Array, int rows, int columns)
functions_library.cu:    cudaSetDevice(CUDA_device_IDs[0]);
functions_library.cu:        cudaMemcpy(Array_host_2D[row], cuda_2D_Array[row], (columns + 1) * sizeof(cuda_2D_Array[0][0]), cudaMemcpyDeviceToHost);
functions_library.cu:__global__ void cuda_Sequences_to_INT(int num_Sequences, int **sequence_INT, int genome_Length, char *sites)
functions_library.cu:    cout << "Configuring multi gpu distribution of " << num_of_Sequences_current << " sequence(s)\n";
functions_library.cu:    int standard_num_per_GPU = num_of_Sequences_current / num_Cuda_devices;
functions_library.cu:    int remainder = num_of_Sequences_current % num_Cuda_devices;
functions_library.cu:    vector<pair<int, int>> start_stop_Per_GPU;
functions_library.cu:    for (int gpu = 0; gpu < num_Cuda_devices; gpu++)
functions_library.cu:        int start = gpu * standard_num_per_GPU;
functions_library.cu:        int stop = start + standard_num_per_GPU;
functions_library.cu:        start_stop_Per_GPU.push_back(make_pair(start, stop));
functions_library.cu:    start_stop_Per_GPU[num_Cuda_devices - 1].second = start_stop_Per_GPU[num_Cuda_devices - 1].second + remainder;
functions_library.cu:    for (int gpu = 0; gpu < num_Cuda_devices; gpu++)
functions_library.cu:        for (int sequence = start_stop_Per_GPU[gpu].first; sequence < start_stop_Per_GPU[gpu].second; sequence++)
functions_library.cu:    cudaStream_t streams[num_Cuda_devices];
functions_library.cu:    cudaDeviceProp deviceProp;
functions_library.cu:    char *cuda_full_Char[num_Cuda_devices];
functions_library.cu:    int **cuda_Sequence[num_Cuda_devices];
functions_library.cu:    for (int gpu = 0; gpu < num_Cuda_devices; gpu++)
functions_library.cu:        cudaSetDevice(CUDA_device_IDs[gpu]);
functions_library.cu:        cudaGetDeviceProperties(&deviceProp, gpu);
functions_library.cu:        cout << "Intializing GPU " << CUDA_device_IDs[gpu] << "'s stream: " << deviceProp.name << endl;
functions_library.cu:        cudaMalloc(&cuda_full_Char[gpu], (start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first) * genome_Length * sizeof(char));
functions_library.cu:        cudaMemcpy(cuda_full_Char[gpu], full_Char + (start_stop_Per_GPU[gpu].first * genome_Length), (start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first) * genome_Length * sizeof(char), cudaMemcpyHostToDevice);
functions_library.cu:        // cudaMalloc(&cuda_Sequence[gpu], (start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first) * genome_Length * sizeof(int));
functions_library.cu:        cudaMallocManaged(&cuda_Sequence[gpu], (start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first) * sizeof(int *));
functions_library.cu:        for (int row = 0; row < (start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first); row++)
functions_library.cu:            cudaMalloc((void **)&cuda_Sequence[gpu][row], genome_Length * sizeof(int));
functions_library.cu:        cudaStreamCreate(&streams[gpu]);
functions_library.cu:    cout << "Loaded " << num_of_Sequences_current << " sequence(s) to the GPU(s)\n";
functions_library.cu:    // int devID[num_Cuda_devices];
functions_library.cu:    for (int gpu = 0; gpu < num_Cuda_devices; gpu++)
functions_library.cu:        cudaSetDevice(CUDA_device_IDs[gpu]);
functions_library.cu:        cuda_Sequences_to_INT<<<tot_Blocks_array[gpu], tot_ThreadsperBlock_array[gpu], 0, streams[gpu]>>>(start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first, cuda_Sequence[gpu], genome_Length, cuda_full_Char[gpu]);
functions_library.cu:    for (int gpu = 0; gpu < num_Cuda_devices; gpu++)
functions_library.cu:        cudaSetDevice(CUDA_device_IDs[gpu]);
functions_library.cu:        cudaStreamSynchronize(streams[gpu]);
functions_library.cu:    cout << "GPU(s) streams completed and synchronized\nCopying data from GPU to Host memory\n";
functions_library.cu:    for (int gpu = 0; gpu < num_Cuda_devices; gpu++)
functions_library.cu:        cudaSetDevice(CUDA_device_IDs[gpu]);
functions_library.cu:        // cudaMemcpy(sequence + (start_stop_Per_GPU[gpu].first * genome_Length), cuda_Sequence[gpu], (start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first) * genome_Length * sizeof(int), cudaMemcpyDeviceToHost);
functions_library.cu:        // cudaMemcpy(sequence[start_stop_Per_GPU[gpu].first], cuda_Sequence[gpu], (start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first) * sizeof(int *), cudaMemcpyDeviceToHost);
functions_library.cu:        for (int row = 0; row < (start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first); row++)
functions_library.cu:            cudaMemcpy(sequence[start_stop_Per_GPU[gpu].first + row], cuda_Sequence[gpu][row], genome_Length * sizeof(int), cudaMemcpyDeviceToHost);
functions_library.cu:    for (int gpu = 0; gpu < num_Cuda_devices; gpu++)
functions_library.cu:        cudaSetDevice(CUDA_device_IDs[gpu]);
functions_library.cu:        cudaFree(cuda_full_Char[gpu]);
functions_library.cu:        // cudaFree(cuda_Sequence);
functions_library.cu:        for (int row = 0; row < (start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first); row++)
functions_library.cu:            cudaFree(cuda_Sequence[gpu][row]);
functions_library.cu:        cudaFree(cuda_Sequence[gpu]);
functions_library.cu:        cudaStreamDestroy(streams[gpu]);
functions_library.cu:    // cudaError_t err = cudaGetLastError();
functions_library.cu:    // if (err != cudaSuccess)
functions_library.cu:    //     printf("CUDA Error: %s\n", cudaGetErrorString(err));
vcf_splitter_2.cuh:#include "cuda_runtime.h"
vcf_splitter_2.cuh:    int SNPs_per_time_GPU = 100;
vcf_splitter_2.cuh:     * @param concat_Segs stores the concatenated Seg sites for GPU processing
vcf_splitter_2.cuh:    int *sample_ID_population_ID, *cuda_sample_ID_population_ID;
vcf_splitter_2.cuh:    int *MAF_count_per_Population, *cuda_MAF_count_per_Population;
vcf_splitter_2.cuh:    vcf_splitter_2(int cuda_ID, string input_vcf_Folder, string output_Folder, int cores, int SNPs_per_time_CPU, int SNPs_per_time_GPU, int allele_Count_REF, int allele_Count_ALT, int ploidy, int summary_Individuals);
vcf_splitter_2.cuh:    vcf_splitter_2(int cuda_ID, string input_vcf_Folder, string output_Folder, string population_File, int sampled_ID_col, int pop_ID_column, int cores, int SNPs_per_time_CPU, int SNPs_per_time_GPU, int ploidy, int max_SNPs_per_file, int logic_MAF, double MAF);
vcf_splitter_2.cuh:    void cuda_Set_device(int cuda_ID);
vcf_splitter_2.cuh:     * Concatenates the Segregating sites for processing by each GPU round.
simulator_Master.cuh:#include "cuda_runtime.h"
simulator_Master.cuh:#include <thrust/system/cuda/error.h>
simulator_Master.cuh:    int CUDA_device_number;
simulator_Master.cuh:    int *CUDA_device_IDs;
simulator_Master.cuh:    int num_Cuda_devices;
simulator_Master.cuh:    int gpu_Limit;
functions_library.cuh:#include "cuda_runtime.h"
functions_library.cuh:#include <thrust/system/cuda/error.h>
functions_library.cuh:    int *CUDA_device_IDs;
functions_library.cuh:    int num_Cuda_devices;
functions_library.cuh:    int gpu_Limit;
functions_library.cuh:    float **CUDA_mutation_rates_Hotspot_generation;
functions_library.cuh:    int **CUDA_mutation_Regions_start_stop;
functions_library.cuh:    float **CUDA_A_0_mutation;
functions_library.cuh:    float **CUDA_T_1_mutation;
functions_library.cuh:    float **CUDA_G_2_mutation;
functions_library.cuh:    float **CUDA_C_3_mutation;
functions_library.cuh:    int *CUDA_stride_Array;
functions_library.cuh:    int **CUDA_recombination_hotspots_start_stop;
functions_library.cuh:    float **CUDA_A_0_Recombination;
functions_library.cuh:    float **CUDA_T_1_Recombination;
functions_library.cuh:    float **CUDA_G_2_Recombination;
functions_library.cuh:    float **CUDA_C_3_Recombination;
functions_library.cuh:    float **CUDA_A_0_fitness;
functions_library.cuh:    float **CUDA_T_1_fitness;
functions_library.cuh:    float **CUDA_G_2_fitness;
functions_library.cuh:    float **CUDA_C_3_fitness;
functions_library.cuh:    float **CUDA_A_0_survivability;
functions_library.cuh:    float **CUDA_T_1_survivability;
functions_library.cuh:    float **CUDA_G_2_survivability;
functions_library.cuh:    float **CUDA_C_3_survivability;
functions_library.cuh:    float **CUDA_A_0_probability_Proof_reading;
functions_library.cuh:    float **CUDA_T_1_probability_Proof_reading;
functions_library.cuh:    float **CUDA_G_2_probability_Proof_reading;
functions_library.cuh:    float **CUDA_C_3_probability_Proof_reading;
functions_library.cuh:    int **CUDA_sequence_Mutation_tracker;
functions_library.cuh:    functions_library(int tot_Blocks, int tot_ThreadsperBlock, int gpu_Limit, int CPU_cores);
functions_library.cuh:    functions_library(int *tot_Blocks_array, int *tot_ThreadsperBlock_array, int *CUDA_device_IDs, int num_Cuda_devices, int gpu_Limit, int CPU_cores);
functions_library.cuh:    void print_Cuda_device(int cuda_ID, int &tot_Blocks, int &tot_ThreadsperBlock);
functions_library.cuh:    void print_Cuda_devices(vector<string> cuda_IDs, int *CUDA_device_IDs, int num_Cuda_devices, int *tot_Blocks, int *tot_ThreadsperBlock);
functions_library.cuh:    int **create_INT_2D_arrays_for_GPU(int rows, int columns);
functions_library.cuh:    float **create_FLOAT_2D_arrays_for_GPU(int rows, int columns);
functions_library.cuh:    int **Fill_2D_array_CUDA(int rows, int columns, int fill_Value, int **cuda_Progeny_numbers);
functions_library.cuh:    int **create_Progeny_Array(int parents_in_current_generation, int *stride_Progeny_Index, int total_Progeny, int num_Hotspots, int **cuda_Progeny_numbers, int fill_Value);
functions_library.cuh:    void progeny_Recombination_parents_array(int **progeny_recom_Index_Cuda, int total_Progeny, int num_Hotspots,
functions_library.cuh:                                             int *parent_and_their_cell_CUDA, int **cell_and_their_viruses_CUDA, int *per_Cell_max_viruses_CUDA,
functions_library.cuh:                                             float **CUDA_current_gen_Parent_data, int max_Count, int num_Unique_cells);
functions_library.cuh:    void progeny_Shuffle(int **progeny_recom_Index_Cuda, int num_Hotspots, int parents_in_current_generation, int *stride_Progeny_Index_CUDA);
functions_library.cuh:    float **create_current_Progeny_data(int total_Progeny, int num_Hotspots, int **progeny_recom_Index_Cuda, float **CUDA_current_gen_Parent_data, float *CUDA_parent_Proof_reading_probability, float *CUDA_progeny_Proof_reading_probability);
functions_library.cuh:    float *normal_distribution_CUDA(int num_of_values, float mean, float st_deviation);
functions_library.cuh:    int *poisson_distribution_CUDA(int num_of_values, float mean);
functions_library.cuh:    float *gamma_distribution_CUDA(int num_of_values, float shape, float scale);
functions_library.cuh:    int *binomial_distribution_CUDA(int num_of_values, float prob, int progeny_Number);
functions_library.cuh:    int *negative_binomial_distribution_CUDA(int num_of_values, float mean, float dispersion_Parameter);
functions_library.cuh:    float **float_2D_Array_load_to_CUDA(float **host_Array, int rows, int columns);
functions_library.cuh:    int **int_2D_Array_load_to_CUDA(int **host_Array, int rows, int columns);
functions_library.cuh:    int *copy_1D_to_CUDA_INT(int *host_Array, int num_Values);
functions_library.cuh:    float *copy_1D_to_CUDA_FLOAT(float *host_Array, int num_Values);
functions_library.cuh:    int **load_to_Host(int **cuda_2D_Array, int rows, int columns);
functions_library.cuh:    float **load_to_Host_FLOAT(float **cuda_2D_Array, int rows, int columns);
functions_library.cuh:    int **create_CUDA_2D_int(int rows, int columns);
functions_library.cuh:    float **create_CUDA_2D_FLOAT(int rows, int columns);
functions_library.cuh:    void clear_Array_INT(int **CUDA_2D_array, int rows);
functions_library.cuh:    void clear_Array_FLOAT(float **CUDA_2D_array, int rows);
functions_library.cuh:    int **progeny_distribution_CUDA(string &distribution_Type,
functions_library.cuh:                                    float *cuda__Parent_finess,
functions_library.cuh:                                    float **CUDA_current_gen_Parent_data, int recombination_hotspots);
functions_library.cuh:    int sum_CUDA(int **cuda_Array_input, int num_Elements);
functions_library.cuh:    void mutate_Sequences(int **cuda_progeny_Sequences, float **CUDA_current_gen_Progeny_data, float *CUDA_progeny_Proof_reading_probability,
functions_library.cuh:                          float **CUDA_mutation_rates_Hotspot_generation, int **CUDA_mutation_Regions_start_stop, int **CUDA_sequence_Mutation_tracker,
functions_library.cuh:    int **create_progeny_Sequences(int **cuda_parent_Sequences, int **progeny_recom_Index_Cuda, int num_Hotspots, int total_Progeny, int genome_SIZE, int **CUDA_recombination_hotspots_start_stop);
functions_library.cuh:    void find_Unique_values(int **CUDA_Array_2D, int total_Elements, int num_Recom_hotspots,
functions_library.cuh:                            int **CUDA_recombination_hotspots_start_stop,
functions_library.cuh:                            float **CUDA_current_gen_Progeny_data, float *CUDA_progeny_Proof_reading_probability, int current_Generation, float **CUDA_mutation_rates_Hotspot_generation, int **CUDA_mutation_Regions_start_stop, int **CUDA_sequence_Mutation_tracker, int proof_Reading_Activate,
functions_library.cuh:                           int **progeny_recom_Index_Cuda, int num_Hotspots, int **CUDA_recombination_hotspots_start_stop,
functions_library.cuh:                           float **CUDA_current_gen_Progeny_data, float *CUDA_progeny_Proof_reading_probability, float **CUDA_mutation_rates_Hotspot_generation, int **CUDA_mutation_Regions_start_stop, int **CUDA_sequence_Mutation_tracker, int proof_Reading_Activate,
hap_counter.cuh:#include "cuda_runtime.h"
hap_counter.cuh:#include <thrust/system/cuda/error.h>
hap_counter.cuh:    int CUDA_device_number;
hap_counter.cuh:    int *CUDA_device_IDs;
hap_counter.cuh:    int num_Cuda_devices;
hap_counter.cuh:    int gpu_Limit;
README.md:# CATE (CUDA Accelerated Testing of Evolution)
README.md:A fast and scalable CUDA implementation to conduct highly parallelized evolutionary tests on large-scale genomic data.
README.md:The CATE software is a CUDA based solution to enable rapid processing of large-scale VCF files to conduct a series of six different tests on evolution.
README.md:1. CUDA capable hardware
README.md:3. NVIDIA's CUDA toolkit (nvcc compiler)
README.md:![C/C++ CUDA CI](https://github.com/theLongLab/CATE/actions/workflows/c-cpp.yml/badge.svg?event=push)
README.md:*cuda 11.3.0 or higher*
README.md:module load cuda/11.3.0
README.md:**CATE** has been successfully published in the journal Methods in Ecology and Evolution (MEE). If you find this framework or the software solution useful in your analyses, please CITE the published article available in [MEE, CATE: A fast and scalable CUDA implementation to conduct highly parallelized evolutionary tests on large scale genomic data](https://doi.org/10.1111/2041-210X.14168).
README.md:CATE: A fast and scalable CUDA implementation to conduct highly parallelized evolutionary tests on large scale genomic data. 
README.md:Apollo: A comprehensive GPU-powered within-host simulator for viral evolution and infection dynamics across population, tissue, and cell
hap_extract.cu:hap_extract::hap_extract(string gene_List, string input_Folder, string output_Path, int cuda_ID, string intermediate_Path, int ploidy, string reference_File, string pop_Out)
hap_extract.cu:    cout << "Initiating CUDA powered Haplotype extractor" << endl
hap_extract.cu:    set_Values(gene_List, input_Folder, output_Path, cuda_ID, intermediate_Path, ploidy);
hap_extract.cu:void hap_extract::set_Values(string gene_List, string input_Folder, string output_Path, int cuda_ID, string intermediate_Path, int ploidy)
hap_extract.cu:     * Here the first call to the selected CUDA device occurs.
hap_extract.cu:    cudaSetDevice(cuda_ID);
hap_extract.cu:    cout << "Properties of selected CUDA GPU:" << endl;
hap_extract.cu:    cudaDeviceProp prop;
hap_extract.cu:    cudaGetDeviceProperties(&prop, cuda_ID);
hap_extract.cu:    cout << "GPU number\t: " << cuda_ID << endl;
hap_extract.cu:    cout << "GPU name\t: " << prop.name << endl;
hap_extract.cu:    cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);
hap_extract.cu:    cout << "GPU memory (GB)\t: " << l_Total / (1000 * 1000 * 1000) << endl;
hap_extract.cu:    cout << "GPU number of multiprocessor(s)\t: " << prop.multiProcessorCount << endl;
hap_extract.cu:    cout << "GPU block(s) per multiprocessor\t: " << prop.maxBlocksPerMultiProcessor << endl;
hap_extract.cu:    cout << "GPU thread(s) per block\t: " << tot_ThreadsperBlock << endl
hap_extract.cu:     * Read the reference file and load it into the GPU memory for processing.
hap_extract.cu:     * @param reference_full is used to convert the string into a char pointer that can then be transferred into the GPU memory.
hap_extract.cu:    cudaMallocManaged(&cuda_reference, (full_Reference.size() + 1) * sizeof(char));
hap_extract.cu:    cudaMemcpy(cuda_reference, reference_full, (full_Reference.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
hap_extract.cu:__global__ void cuda_hap_Forge_with_alleles(int total_Segs, char *sites, int *index, char *Hap_array, char *REF_all, char *ALT_all)
hap_extract.cu:// __global__ void cuda_haplotype_Forge(int N, int total_Segs, char **grid, char *Hap_array)
hap_extract.cu:__global__ void cuda_sequence_Generation(int sequence_Size, int num_of_Segs, int start, char *ref, char *haplotype, int *pos_Allele, int *index_Allele, char *REF, char *ALT, char *sequence_Full)
hap_extract.cu:         * EACH thread in this GPU call is a position on the FASTA sequence.
hap_extract.cu:         * This enables the entire sequence to be reconstructed at once inside the GPU.
hap_extract.cu:     * 1. Conversion of SNP strings into char pointers for GPU accessability.
hap_extract.cu:     * 2. Call GPU for Haplotype reconstruction.
hap_extract.cu:     * 4. Call GPU Sequence reconstruction.
hap_extract.cu:     * This track is vital for navigating through the data in the GPU. For the data is stored in the form of a 1D array.
hap_extract.cu:    //*cuda_pos;
hap_extract.cu:     * @param cuda_pos_Allele is used by the GPU. Is a COPY of pos_Allele.
hap_extract.cu:     * @param cuda_index_Allele is used by the GPU. Is a COPY of index_Allele.
hap_extract.cu:    int *pos_Allele, *cuda_pos_Allele, *index_Allele, *cuda_index_Allele;
hap_extract.cu:    // char **cuda_snp_N_grid;
hap_extract.cu:    // cudaMallocManaged(&cuda_snp_N_grid, this->N * num_segregrating_Sites * sizeof(char));
hap_extract.cu:    //     cudaMalloc((void **)&tmp[i], this->N * sizeof(tmp[0][0]));
hap_extract.cu:    // cudaMemcpy(cuda_snp_N_grid, tmp, num_segregrating_Sites * sizeof(char *), cudaMemcpyHostToDevice);
hap_extract.cu:     * @param cuda_full_Char is used by the GPU. Is a COPY of full_Char.
hap_extract.cu:     * @param cuda_site_Index is used by the GPU. Is a COPY of site_Index.
hap_extract.cu:    char *cuda_full_Char;
hap_extract.cu:    cudaMallocManaged(&cuda_full_Char, (Seg_sites.size() + 1) * sizeof(char));
hap_extract.cu:    int *cuda_site_Index;
hap_extract.cu:    cudaMallocManaged(&cuda_site_Index, (num_segregrating_Sites + 1) * sizeof(int));
hap_extract.cu:     * Transfer of data to the GPU.
hap_extract.cu:    cudaMemcpy(cuda_full_Char, full_Char, (Seg_sites.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
hap_extract.cu:    cudaMemcpy(cuda_site_Index, site_Index, (num_segregrating_Sites + 1) * sizeof(int), cudaMemcpyHostToDevice);
hap_extract.cu:     * @param cuda_REF_char used to capture the Reference allele in the SNP position.
hap_extract.cu:     * @param cuda_ALT_char used to capture the Alternate allele in the SNP position.
hap_extract.cu:    char *cuda_REF_char, *cuda_ALT_char;
hap_extract.cu:    cudaMallocManaged(&cuda_REF_char, (num_segregrating_Sites + 1) * sizeof(char));
hap_extract.cu:    cudaMallocManaged(&cuda_ALT_char, (num_segregrating_Sites + 1) * sizeof(char));
hap_extract.cu:     * @param cuda_Hap_array stores the forged Haplotypes for the region under study.
hap_extract.cu:     * @param Hap_array is used by the CPU. Is a COPY of cuda_Hap_array.
hap_extract.cu:    char *Hap_array, *cuda_Hap_array;
hap_extract.cu:    cudaMallocManaged(&cuda_Hap_array, ((this->N * num_segregrating_Sites) + 1) * sizeof(char));
hap_extract.cu:     * CALL THE GPU.
hap_extract.cu:     * * GPU WILL CONDUCT HAPLOTYPE RECONSTRUCTION AND COLLECT THE ALLELIC INFORMATION REQUIRED FOR POST PROCESSING.
hap_extract.cu:    cuda_hap_Forge_with_alleles<<<tot_Blocks, tot_ThreadsperBlock>>>(num_segregrating_Sites, cuda_full_Char, cuda_site_Index, cuda_Hap_array, cuda_REF_char, cuda_ALT_char);
hap_extract.cu:    cudaError_t err = cudaGetLastError();
hap_extract.cu:    if (err != cudaSuccess)
hap_extract.cu:        printf("CUDA Error: %s\n", cudaGetErrorString(err));
hap_extract.cu:    cudaDeviceSynchronize();
hap_extract.cu:    cudaMemcpy(Hap_array, cuda_Hap_array, ((this->N * num_segregrating_Sites) + 1) * sizeof(char), cudaMemcpyDeviceToHost);
hap_extract.cu:    // cudaMemcpy(REF_char, cuda_REF_char, (num_segregrating_Sites + 1) * sizeof(char), cudaMemcpyDeviceToHost);
hap_extract.cu:    // cudaMemcpy(ALT_char, cuda_ALT_char, (num_segregrating_Sites + 1) * sizeof(char), cudaMemcpyDeviceToHost);
hap_extract.cu:    cudaFree(cuda_full_Char);
hap_extract.cu:    cudaFree(cuda_site_Index);
hap_extract.cu:    cudaFree(cuda_Hap_array);
hap_extract.cu:    // cuda_haplotype_Forge<<<tot_Blocks, tot_ThreadsperBlock>>>(this->N, num_segregrating_Sites, cuda_snp_N_grid, cuda_Hap_array);
hap_extract.cu:    // // cudaError_t err = cudaGetLastError();
hap_extract.cu:    // if (err != cudaSuccess)
hap_extract.cu:    //     printf("CUDA Error: %s\n", cudaGetErrorString(err));
hap_extract.cu:    // cudaDeviceSynchronize();
hap_extract.cu:    // cudaMallocManaged(&cuda_pos, (num_segregrating_Sites * sizeof(int)));
hap_extract.cu:    // cudaMemcpy(cuda_pos, pos, num_segregrating_Sites * sizeof(int), cudaMemcpyHostToDevice);
hap_extract.cu:    cudaMallocManaged(&cuda_pos_Allele, (num_segregrating_Sites * sizeof(int)));
hap_extract.cu:    cudaMallocManaged(&cuda_index_Allele, (num_segregrating_Sites * sizeof(int)));
hap_extract.cu:    cudaMemcpy(cuda_pos_Allele, pos_Allele, num_segregrating_Sites * sizeof(int), cudaMemcpyHostToDevice);
hap_extract.cu:    cudaMemcpy(cuda_index_Allele, index_Allele, num_segregrating_Sites * sizeof(int), cudaMemcpyHostToDevice);
hap_extract.cu:             * @param cuda_sequence is used by the CPU. Is a COPY of sequence.
hap_extract.cu:            char *sequence, *cuda_sequence;
hap_extract.cu:            cudaMallocManaged(&cuda_sequence, (sequence_Size + 1) * sizeof(char));
hap_extract.cu:             * @param cuda_haplotype is used by the CPU. Is a COPY of haplotype.
hap_extract.cu:            char *haplotye, *cuda_haplotype;
hap_extract.cu:            cudaMallocManaged(&cuda_haplotype, (num_segregrating_Sites + 1) * sizeof(char));
hap_extract.cu:            cudaMemcpy(cuda_haplotype, haplotye, (num_segregrating_Sites + 1) * sizeof(char), cudaMemcpyHostToDevice);
hap_extract.cu:            // cuda_sequence_Generation(int sequence_Size, int num_of_Segs, int start, char *ref, char *haplotype, int *pos_Allele, char *index_Allele, char *REF, char *ALT, char *sequence_Full)
hap_extract.cu:             * CALL THE GPU.
hap_extract.cu:             * * GPU WILL CONDUCT SEQUENCE RECONSTRUCTION.
hap_extract.cu:            cuda_sequence_Generation<<<tot_Blocks, tot_ThreadsperBlock>>>(sequence_Size, num_segregrating_Sites, start_Pos, cuda_reference, cuda_haplotype, cuda_pos_Allele, cuda_index_Allele, cuda_REF_char, cuda_ALT_char, cuda_sequence);
hap_extract.cu:            if (err != cudaSuccess)
hap_extract.cu:                printf("CUDA Error: %s\n", cudaGetErrorString(err));
hap_extract.cu:            cudaDeviceSynchronize();
hap_extract.cu:            cudaMemcpy(sequence, cuda_sequence, (sequence_Size + 1) * sizeof(char), cudaMemcpyDeviceToHost);
hap_extract.cu:            cudaFree(cuda_haplotype);
hap_extract.cu:            cudaFree(cuda_sequence);
hap_extract.cu:    cudaFree(cuda_index_Allele);
hap_extract.cu:    cudaFree(cuda_pos_Allele);
hap_extract.cu:    cudaFree(cuda_REF_char);
hap_extract.cu:    cudaFree(cuda_ALT_char);
hap_extract.cu:    // cudaFree(cuda_snp_N_grid);
node_within_host.cu:void node_within_host::run_Generation(functions_library &functions, string &multi_Read, int &max_Cells_at_a_time, int &gpu_Limit, int *CUDA_device_IDs, int &num_Cuda_devices, int &genome_Length,
node_within_host.cu:                                simulate_Cell_replication(functions, multi_Read, gpu_Limit, CUDA_device_IDs, num_Cuda_devices, source_sequence_Data_folder, indexed_Source_Folders[tissue],
node_within_host.cu:void node_within_host::simulate_Cell_replication(functions_library &functions, string &multi_Read, int &gpu_Limit, int *CUDA_device_IDs, int &num_Cuda_devices, string &source_sequence_Data_folder, vector<pair<int, int>> &indexed_Tissue_Folder,
node_within_host.cu:    // gpu_Limit = 5;
node_within_host.cu:    int full_Rounds = Total_seqeunces_to_Process / gpu_Limit;
node_within_host.cu:    int partial_Rounds = Total_seqeunces_to_Process % gpu_Limit;
node_within_host.cu:        int start = full * gpu_Limit;
node_within_host.cu:        int stop = start + gpu_Limit;
node_within_host.cu:                                                    collected_Sequences, CUDA_device_IDs, num_Cuda_devices, genome_Length,
node_within_host.cu:    cout << "Intializing GPU memory structures\n";
node_within_host.cu:    cudaSetDevice(CUDA_device_IDs[0]);
node_within_host.cu:    // int **cuda_progeny_Configuration = functions.create_CUDA_2D_int(total_Progeny, 1 + recombination_Hotspots);
node_within_host.cu:    int *cuda_progeny_Stride;
node_within_host.cu:    cudaMallocManaged(&cuda_progeny_Stride, (Total_seqeunces_to_Process + 1) * sizeof(int));
node_within_host.cu:    cudaMemcpy(cuda_progeny_Stride, progeny_Stride, (Total_seqeunces_to_Process + 1) * sizeof(int), cudaMemcpyHostToDevice);
node_within_host.cu:    float **cuda_sequence_Configuration_standard;
node_within_host.cu:    // functions.float_2D_Array_load_to_CUDA(sequence_Configuration_standard, Total_seqeunces_to_Process, 2 + (2 * recombination_Hotspots));
node_within_host.cu:    cudaMallocManaged(&cuda_sequence_Configuration_standard, Total_seqeunces_to_Process * sizeof(float *));
node_within_host.cu:        cudaMalloc((void **)&(cuda_sequence_Configuration_standard[row]), (2 + (2 * recombination_Hotspots)) * sizeof(float));
node_within_host.cu:        cudaMemcpy(cuda_sequence_Configuration_standard[row], sequence_Configuration_standard[row], (2 + (2 * recombination_Hotspots)) * sizeof(float), cudaMemcpyHostToDevice);
node_within_host.cu:    //     cudaMemcpy(cuda_sequence_Configuration_standard[row], sequence_Configuration_standard[row], (2 + (2 * recombination_Hotspots)) * sizeof(float), cudaMemcpyHostToDevice);
node_within_host.cu:                             cuda_sequence_Configuration_standard, recombination_Hotspots,
node_within_host.cu:                             progeny_Configuration, cuda_progeny_Stride, cuda_progeny_Stride[start_stops[round].second] - cuda_progeny_Stride[start_stops[round].first], cuda_progeny_Stride[start_stops[round].first]);
node_within_host.cu:    //  int **progeny_Configuration = functions.load_to_Host(cuda_progeny_Configuration, total_Progeny, 1 + recombination_Hotspots);
node_within_host.cu:    // functions.clear_Array_INT(cuda_progeny_Configuration, total_Progeny);
node_within_host.cu:    cudaFree(cuda_progeny_Stride);
node_within_host.cu:        cudaFree(cuda_sequence_Configuration_standard[row]);
node_within_host.cu:    cudaFree(cuda_sequence_Configuration_standard);
node_within_host.cu:    full_Rounds = total_Progeny / gpu_Limit;
node_within_host.cu:    partial_Rounds = total_Progeny % gpu_Limit;
node_within_host.cu:        int start = full * gpu_Limit;
node_within_host.cu:        int stop = start + gpu_Limit;
node_within_host.cu:                           CUDA_device_IDs, num_Cuda_devices,
node_within_host.cu:__global__ void cuda_Progeny_Complete_Configuration(int genome_Length,
node_within_host.cu:                                                    float *cuda_Reference_fitness_survivability_proof_reading,
node_within_host.cu:                                                    int *cuda_num_effect_Segregating_sites,
node_within_host.cu:                                                    float **cuda_sequence_Survivability_changes,
node_within_host.cu:                                                    float **cuda_recombination_hotspot_parameters,
node_within_host.cu:                                                    int *cuda_tot_prob_selectivity,
node_within_host.cu:                                                    float **cuda_A_0_mutation,
node_within_host.cu:                                                    float **cuda_T_1_mutation,
node_within_host.cu:                                                    float **cuda_G_2_mutation,
node_within_host.cu:                                                    float **cuda_C_3_mutation,
node_within_host.cu:                                                    float **cuda_mutation_hotspot_parameters,
node_within_host.cu:                                                    int **cuda_parent_Sequences, int **cuda_parent_IDs,
node_within_host.cu:                                                    float **cuda_sequence_Configuration_standard,
node_within_host.cu:                                                    int *cuda_cell_Index, int num_Cells,
node_within_host.cu:                                                    float **cuda_totals_Progeny_Selectivity,
node_within_host.cu:                                                    int **cuda_progeny_Configuration,
node_within_host.cu:                                                    int **cuda_progeny_Sequences,
node_within_host.cu:                                                    int *cuda_Dead_or_Alive,
node_within_host.cu:                                                    int per_gpu_Progeny)
node_within_host.cu:    while (tid < per_gpu_Progeny)
node_within_host.cu:            cuda_progeny_Sequences[tid][base] = cuda_parent_Sequences[cuda_progeny_Configuration[tid][0]][base];
node_within_host.cu:            int get_Cell = cuda_parent_IDs[1][cuda_progeny_Configuration[tid][0]];
node_within_host.cu:                if (cuda_progeny_Configuration[tid][hotspot + 1] != -1)
node_within_host.cu:                    int recomb_parent = cuda_progeny_Configuration[tid][0];
node_within_host.cu:                    for (int check = cuda_cell_Index[get_Cell]; check < cuda_cell_Index[get_Cell + 1]; check++)
node_within_host.cu:                        cumulative_prob += (cuda_sequence_Configuration_standard[check][(hotspot * 2) + 3] / cuda_totals_Progeny_Selectivity[get_Cell][hotspot]);
node_within_host.cu:                    cuda_progeny_Configuration[tid][hotspot + 1] = recomb_parent;
node_within_host.cu:                    if (recomb_parent != cuda_progeny_Configuration[tid][0])
node_within_host.cu:                        for (int base = ((int)cuda_recombination_hotspot_parameters[hotspot][0] - 1); base < (int)cuda_recombination_hotspot_parameters[hotspot][1]; base++)
node_within_host.cu:                            cuda_progeny_Sequences[tid][base] = cuda_parent_Sequences[recomb_parent][base];
node_within_host.cu:                if (cuda_mutation_hotspot_parameters[hotspot][2] == 0)
node_within_host.cu:                    num_Mutations = curand_poisson(&localState, cuda_mutation_hotspot_parameters[hotspot][3]);
node_within_host.cu:                else if (cuda_mutation_hotspot_parameters[hotspot][2] == 1)
node_within_host.cu:                    while (successes < cuda_mutation_hotspot_parameters[hotspot][3])
node_within_host.cu:                        if (rand_num < cuda_mutation_hotspot_parameters[hotspot][4])
node_within_host.cu:                    int bases_in_Region = cuda_mutation_hotspot_parameters[hotspot][1] - (cuda_mutation_hotspot_parameters[hotspot][0] - 1);
node_within_host.cu:                        if (curand_uniform(&localState) < cuda_mutation_hotspot_parameters[hotspot][3])
node_within_host.cu:                    if (cuda_sequence_Configuration_standard[cuda_progeny_Configuration[tid][0]][1] != -1)
node_within_host.cu:                        // int bases_in_Region = cuda_mutation_hotspot_parameters[hotspot][1] - (cuda_mutation_hotspot_parameters[hotspot][0] - 1);
node_within_host.cu:                            if (curand_uniform(&localState) < cuda_sequence_Configuration_standard[cuda_progeny_Configuration[tid][0]][1])
node_within_host.cu:                            int position = (int)(curand_uniform(&localState) * (((int)cuda_mutation_hotspot_parameters[hotspot][1] - 1) - ((int)cuda_mutation_hotspot_parameters[hotspot][0] - 1) + 1)) + ((int)cuda_mutation_hotspot_parameters[hotspot][0] - 1);
node_within_host.cu:                            int original_BASE = cuda_progeny_Sequences[tid][position];
node_within_host.cu:                                cumulative_prob += (original_BASE == 0)   ? cuda_A_0_mutation[hotspot][base]
node_within_host.cu:                                                   : (original_BASE == 1) ? cuda_T_1_mutation[hotspot][base]
node_within_host.cu:                                                   : (original_BASE == 2) ? cuda_G_2_mutation[hotspot][base]
node_within_host.cu:                                                   : (original_BASE == 3) ? cuda_C_3_mutation[hotspot][base]
node_within_host.cu:                            cuda_progeny_Sequences[tid][position] = new_Base;
node_within_host.cu:        float survivability = cuda_Reference_fitness_survivability_proof_reading[1];
node_within_host.cu:        for (int pos = 0; pos < cuda_num_effect_Segregating_sites[1]; pos++)
node_within_host.cu:            if (cuda_progeny_Sequences[tid][(int)cuda_sequence_Survivability_changes[pos][0] - 1] == 0)
node_within_host.cu:                survivability = survivability + cuda_sequence_Survivability_changes[pos][1];
node_within_host.cu:            else if (cuda_progeny_Sequences[tid][(int)cuda_sequence_Survivability_changes[pos][0] - 1] == 1)
node_within_host.cu:                survivability = survivability + cuda_sequence_Survivability_changes[pos][2];
node_within_host.cu:            else if (cuda_progeny_Sequences[tid][(int)cuda_sequence_Survivability_changes[pos][0] - 1] == 2)
node_within_host.cu:                survivability = survivability + cuda_sequence_Survivability_changes[pos][3];
node_within_host.cu:            else if (cuda_progeny_Sequences[tid][(int)cuda_sequence_Survivability_changes[pos][0] - 1] == 3)
node_within_host.cu:                survivability = survivability + cuda_sequence_Survivability_changes[pos][4];
node_within_host.cu:            cuda_Dead_or_Alive[tid] = 1;
node_within_host.cu:            cuda_Dead_or_Alive[tid] = 0;
node_within_host.cu:            cuda_Dead_or_Alive[tid] = (survivability_Check < survivability) ? 1 : 0;
node_within_host.cu:                                          int *CUDA_device_IDs, int &num_Cuda_devices,
node_within_host.cu:    cout << "\nConfiguring multi gpu distribution of " << num_of_Sequences_current << " sequence(s)\n";
node_within_host.cu:    int standard_num_per_GPU = num_of_Sequences_current / num_Cuda_devices;
node_within_host.cu:    int remainder = num_of_Sequences_current % num_Cuda_devices;
node_within_host.cu:    vector<pair<int, int>> start_stop_Per_GPU;
node_within_host.cu:    for (int gpu = 0; gpu < num_Cuda_devices; gpu++)
node_within_host.cu:        int start = gpu * standard_num_per_GPU;
node_within_host.cu:        int stop = start + standard_num_per_GPU;
node_within_host.cu:        start_stop_Per_GPU.push_back(make_pair(start, stop));
node_within_host.cu:    start_stop_Per_GPU[num_Cuda_devices - 1].second = start_stop_Per_GPU[num_Cuda_devices - 1].second + remainder;
node_within_host.cu:    cudaStream_t streams[num_Cuda_devices];
node_within_host.cu:    cudaDeviceProp deviceProp;
node_within_host.cu:    float *cuda_Reference_fitness_survivability_proof_reading[num_Cuda_devices];
node_within_host.cu:    int *cuda_num_effect_Segregating_sites[num_Cuda_devices];
node_within_host.cu:    float **cuda_sequence_Survivability_changes[num_Cuda_devices];
node_within_host.cu:    float **cuda_recombination_hotspot_parameters[num_Cuda_devices];
node_within_host.cu:    int *cuda_tot_prob_selectivity[num_Cuda_devices];
node_within_host.cu:    float **cuda_A_0_mutation[num_Cuda_devices];
node_within_host.cu:    float **cuda_T_1_mutation[num_Cuda_devices];
node_within_host.cu:    float **cuda_G_2_mutation[num_Cuda_devices];
node_within_host.cu:    float **cuda_C_3_mutation[num_Cuda_devices];
node_within_host.cu:    float **cuda_mutation_hotspot_parameters[num_Cuda_devices];
node_within_host.cu:    int **cuda_parent_Sequences[num_Cuda_devices];
node_within_host.cu:    float **cuda_sequence_Configuration_standard[num_Cuda_devices];
node_within_host.cu:    int **cuda_parent_IDs[num_Cuda_devices];
node_within_host.cu:    int *cuda_cell_Index[num_Cuda_devices];
node_within_host.cu:    float **cuda_totals_Progeny_Selectivity[num_Cuda_devices];
node_within_host.cu:    int **cuda_progeny_Configuration[num_Cuda_devices];
node_within_host.cu:    int **cuda_progeny_Sequences[num_Cuda_devices];
node_within_host.cu:    int *cuda_Dead_or_Alive[num_Cuda_devices];
node_within_host.cu:    for (int gpu = 0; gpu < num_Cuda_devices; gpu++)
node_within_host.cu:        cudaSetDevice(CUDA_device_IDs[gpu]);
node_within_host.cu:        cudaGetDeviceProperties(&deviceProp, gpu);
node_within_host.cu:        cout << "Intializing GPU " << CUDA_device_IDs[gpu] << "'s stream: " << deviceProp.name << endl;
node_within_host.cu:        cudaMallocManaged(&cuda_progeny_Sequences[gpu], (start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first) * sizeof(int *));
node_within_host.cu:        for (int row = 0; row < (start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first); row++)
node_within_host.cu:            cudaMalloc((void **)&cuda_progeny_Sequences[gpu][row], genome_Length * sizeof(int));
node_within_host.cu:        cudaMallocManaged(&cuda_Dead_or_Alive[gpu], (start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first) * sizeof(int));
node_within_host.cu:        cudaMallocManaged(&cuda_Reference_fitness_survivability_proof_reading[gpu], 3 * sizeof(float));
node_within_host.cu:        cudaMemcpy(cuda_Reference_fitness_survivability_proof_reading[gpu], Reference_fitness_survivability_proof_reading, 3 * sizeof(float), cudaMemcpyHostToDevice);
node_within_host.cu:        cudaMallocManaged(&cuda_num_effect_Segregating_sites[gpu], 3 * sizeof(int));
node_within_host.cu:        cudaMemcpy(cuda_num_effect_Segregating_sites[gpu], num_effect_Segregating_sites, 3 * sizeof(int), cudaMemcpyHostToDevice);
node_within_host.cu:        cudaMallocManaged(&cuda_sequence_Survivability_changes[gpu], num_effect_Segregating_sites[1] * sizeof(float *));
node_within_host.cu:            cudaMalloc((void **)&cuda_sequence_Survivability_changes[gpu][row], 5 * sizeof(float));
node_within_host.cu:            cudaMemcpy(cuda_sequence_Survivability_changes[gpu][row], sequence_Survivability_changes[row], 5 * sizeof(float), cudaMemcpyHostToDevice);
node_within_host.cu:        cudaMallocManaged(&cuda_tot_prob_selectivity[gpu], 2 * sizeof(int));
node_within_host.cu:        cudaMallocManaged(&cuda_recombination_hotspot_parameters[gpu], recombination_Hotspots * sizeof(float *));
node_within_host.cu:            cudaMemcpy(cuda_tot_prob_selectivity[gpu], tot_prob_selectivity, 2 * sizeof(int), cudaMemcpyHostToDevice);
node_within_host.cu:                cudaMalloc((void **)&cuda_recombination_hotspot_parameters[gpu][row], 4 * sizeof(float));
node_within_host.cu:                cudaMemcpy(cuda_recombination_hotspot_parameters[gpu][row], recombination_hotspot_parameters[row], 4 * sizeof(float), cudaMemcpyHostToDevice);
node_within_host.cu:        cudaMallocManaged(&cuda_A_0_mutation[gpu], mutation_Hotspots * sizeof(float *));
node_within_host.cu:        cudaMallocManaged(&cuda_T_1_mutation[gpu], mutation_Hotspots * sizeof(float *));
node_within_host.cu:        cudaMallocManaged(&cuda_G_2_mutation[gpu], mutation_Hotspots * sizeof(float *));
node_within_host.cu:        cudaMallocManaged(&cuda_C_3_mutation[gpu], mutation_Hotspots * sizeof(float *));
node_within_host.cu:        cudaMallocManaged(&cuda_mutation_hotspot_parameters[gpu], mutation_Hotspots * sizeof(float *));
node_within_host.cu:                cudaMalloc((void **)&cuda_A_0_mutation[gpu][row], 4 * sizeof(float));
node_within_host.cu:                cudaMemcpy(cuda_A_0_mutation[gpu][row], A_0_mutation[row], 4 * sizeof(float), cudaMemcpyHostToDevice);
node_within_host.cu:                cudaMalloc((void **)&cuda_T_1_mutation[gpu][row], 4 * sizeof(float));
node_within_host.cu:                cudaMemcpy(cuda_T_1_mutation[gpu][row], T_1_mutation[row], 4 * sizeof(float), cudaMemcpyHostToDevice);
node_within_host.cu:                cudaMalloc((void **)&cuda_G_2_mutation[gpu][row], 4 * sizeof(float));
node_within_host.cu:                cudaMemcpy(cuda_G_2_mutation[gpu][row], G_2_mutation[row], 4 * sizeof(float), cudaMemcpyHostToDevice);
node_within_host.cu:                cudaMalloc((void **)&cuda_C_3_mutation[gpu][row], 4 * sizeof(float));
node_within_host.cu:                cudaMemcpy(cuda_C_3_mutation[gpu][row], C_3_mutation[row], 4 * sizeof(float), cudaMemcpyHostToDevice);
node_within_host.cu:                cudaMalloc((void **)&cuda_mutation_hotspot_parameters[gpu][row], 5 * sizeof(float));
node_within_host.cu:                cudaMemcpy(cuda_mutation_hotspot_parameters[gpu][row], mutation_hotspot_parameters[row], 5 * sizeof(float), cudaMemcpyHostToDevice);
node_within_host.cu:        cudaMallocManaged(&cuda_parent_Sequences[gpu], num_Parent_sequence * sizeof(int *));
node_within_host.cu:        cudaMallocManaged(&cuda_sequence_Configuration_standard[gpu], num_Parent_sequence * sizeof(float *));
node_within_host.cu:            cudaMalloc((void **)&cuda_parent_Sequences[gpu][row], genome_Length * sizeof(int));
node_within_host.cu:            cudaMemcpy(cuda_parent_Sequences[gpu][row], parent_Sequences[row], genome_Length * sizeof(int), cudaMemcpyHostToDevice);
node_within_host.cu:            cudaMalloc((void **)&cuda_sequence_Configuration_standard[gpu][row], (2 + (2 * recombination_Hotspots)) * sizeof(float));
node_within_host.cu:            cudaMemcpy(cuda_sequence_Configuration_standard[gpu][row], sequence_Configuration_standard[row], (2 + (2 * recombination_Hotspots)) * sizeof(float), cudaMemcpyHostToDevice);
node_within_host.cu:        cudaMallocManaged(&cuda_parent_IDs[gpu], 2 * sizeof(int *));
node_within_host.cu:            cudaMalloc((void **)&cuda_parent_IDs[gpu][row], num_Parent_sequence * sizeof(int));
node_within_host.cu:            cudaMemcpy(cuda_parent_IDs[gpu][row], parent_IDs[row], num_Parent_sequence * sizeof(int), cudaMemcpyHostToDevice);
node_within_host.cu:        // cudaMalloc(&cuda_cell_Index[gpu], (num_Cells + 1) * sizeof(int));
node_within_host.cu:        cudaMallocManaged(&cuda_cell_Index[gpu], (num_Cells + 1) * sizeof(int));
node_within_host.cu:        cudaMemcpy(cuda_cell_Index[gpu], cell_Index, (num_Cells + 1) * sizeof(int), cudaMemcpyHostToDevice);
node_within_host.cu:        cudaMallocManaged(&cuda_totals_Progeny_Selectivity[gpu], num_Cells * sizeof(float *));
node_within_host.cu:            cudaMalloc((void **)&cuda_totals_Progeny_Selectivity[gpu][row], recombination_Hotspots * sizeof(float));
node_within_host.cu:            cudaMemcpy(cuda_totals_Progeny_Selectivity[gpu][row], totals_Progeny_Selectivity[row], recombination_Hotspots * sizeof(float), cudaMemcpyHostToDevice);
node_within_host.cu:        cudaMallocManaged(&cuda_progeny_Configuration[gpu], (start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first) * sizeof(int *));
node_within_host.cu:        for (int row = 0; row < (start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first); row++)
node_within_host.cu:            cudaMalloc((void **)&cuda_progeny_Configuration[gpu][row], (1 + recombination_Hotspots) * sizeof(int));
node_within_host.cu:            cudaMemcpy(cuda_progeny_Configuration[gpu][row], progeny_Configuration[row + start_stop_Per_GPU[gpu].first + start_Progeny], (1 + recombination_Hotspots) * sizeof(int), cudaMemcpyHostToDevice);
node_within_host.cu:        cudaStreamCreate(&streams[gpu]);
node_within_host.cu:    cout << "Loaded " << num_Progeny_being_Processed << " sequence(s) and all pre-requisites to the GPU(s)\nInitiating GPU(s) execution\n";
node_within_host.cu:    for (int gpu = 0; gpu < num_Cuda_devices; gpu++)
node_within_host.cu:        cudaSetDevice(CUDA_device_IDs[gpu]);
node_within_host.cu:        cuda_Progeny_Complete_Configuration<<<functions.tot_Blocks_array[gpu], functions.tot_ThreadsperBlock_array[gpu], 0, streams[gpu]>>>(genome_Length,
node_within_host.cu:                                                                                                                                            cuda_Reference_fitness_survivability_proof_reading[gpu],
node_within_host.cu:                                                                                                                                            cuda_num_effect_Segregating_sites[gpu],
node_within_host.cu:                                                                                                                                            cuda_sequence_Survivability_changes[gpu],
node_within_host.cu:                                                                                                                                            cuda_recombination_hotspot_parameters[gpu],
node_within_host.cu:                                                                                                                                            cuda_tot_prob_selectivity[gpu],
node_within_host.cu:                                                                                                                                            cuda_A_0_mutation[gpu],
node_within_host.cu:                                                                                                                                            cuda_T_1_mutation[gpu],
node_within_host.cu:                                                                                                                                            cuda_G_2_mutation[gpu],
node_within_host.cu:                                                                                                                                            cuda_C_3_mutation[gpu],
node_within_host.cu:                                                                                                                                            cuda_mutation_hotspot_parameters[gpu],
node_within_host.cu:                                                                                                                                            cuda_parent_Sequences[gpu], cuda_parent_IDs[gpu],
node_within_host.cu:                                                                                                                                            cuda_sequence_Configuration_standard[gpu],
node_within_host.cu:                                                                                                                                            cuda_cell_Index[gpu], num_Cells,
node_within_host.cu:                                                                                                                                            cuda_totals_Progeny_Selectivity[gpu],
node_within_host.cu:                                                                                                                                            cuda_progeny_Configuration[gpu],
node_within_host.cu:                                                                                                                                            cuda_progeny_Sequences[gpu],
node_within_host.cu:                                                                                                                                            cuda_Dead_or_Alive[gpu],
node_within_host.cu:                                                                                                                                            start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first);
node_within_host.cu:    for (int gpu = 0; gpu < num_Cuda_devices; gpu++)
node_within_host.cu:        cudaSetDevice(CUDA_device_IDs[gpu]);
node_within_host.cu:        cudaStreamSynchronize(streams[gpu]);
node_within_host.cu:    cout << "GPU(s) streams completed and synchronized\nCopying data from GPU to Host memory\n";
node_within_host.cu:    // = functions.create_INT_2D_arrays_for_GPU(num_Progeny_being_Processed, (1 + recombination_Hotspots));
node_within_host.cu:    // functions.create_INT_2D_arrays_for_GPU(num_Progeny_being_Processed, genome_Length);
node_within_host.cu:    for (int gpu = 0; gpu < num_Cuda_devices; gpu++)
node_within_host.cu:        cudaSetDevice(CUDA_device_IDs[gpu]);
node_within_host.cu:        for (int row = 0; row < (start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first); row++)
node_within_host.cu:            cudaMemcpy(progeny_Configuration_Filled[start_stop_Per_GPU[gpu].first + row], cuda_progeny_Configuration[gpu][row], (1 + recombination_Hotspots) * sizeof(int), cudaMemcpyDeviceToHost);
node_within_host.cu:            cudaMemcpy(progeny_Sequences[start_stop_Per_GPU[gpu].first + row], cuda_progeny_Sequences[gpu][row], genome_Length * sizeof(int), cudaMemcpyDeviceToHost);
node_within_host.cu:        cudaMemcpy(Dead_or_Alive + start_stop_Per_GPU[gpu].first, cuda_Dead_or_Alive[gpu], (start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first) * sizeof(int), cudaMemcpyDeviceToHost);
node_within_host.cu:    cout << "Terminating GPU streams: ";
node_within_host.cu:    for (int gpu = 0; gpu < num_Cuda_devices; gpu++)
node_within_host.cu:        cudaSetDevice(CUDA_device_IDs[gpu]);
node_within_host.cu:        cudaFree(cuda_Reference_fitness_survivability_proof_reading[gpu]);
node_within_host.cu:            cudaFree(cuda_sequence_Survivability_changes[gpu][row]);
node_within_host.cu:        cudaFree(cuda_sequence_Survivability_changes[gpu]);
node_within_host.cu:            cudaFree(cuda_recombination_hotspot_parameters[gpu][row]);
node_within_host.cu:        cudaFree(cuda_recombination_hotspot_parameters[gpu]);
node_within_host.cu:        cudaFree(cuda_tot_prob_selectivity[gpu]);
node_within_host.cu:            cudaFree(cuda_A_0_mutation[gpu][row]);
node_within_host.cu:            cudaFree(cuda_T_1_mutation[gpu][row]);
node_within_host.cu:            cudaFree(cuda_G_2_mutation[gpu][row]);
node_within_host.cu:            cudaFree(cuda_C_3_mutation[gpu][row]);
node_within_host.cu:            cudaFree(cuda_mutation_hotspot_parameters[gpu][row]);
node_within_host.cu:        cudaFree(cuda_A_0_mutation[gpu]);
node_within_host.cu:        cudaFree(cuda_T_1_mutation[gpu]);
node_within_host.cu:        cudaFree(cuda_G_2_mutation[gpu]);
node_within_host.cu:        cudaFree(cuda_C_3_mutation[gpu]);
node_within_host.cu:        cudaFree(cuda_mutation_hotspot_parameters[gpu]);
node_within_host.cu:            cudaFree(cuda_parent_Sequences[gpu][row]);
node_within_host.cu:            cudaFree(cuda_sequence_Configuration_standard[gpu][row]);
node_within_host.cu:        cudaFree(cuda_parent_Sequences[gpu]);
node_within_host.cu:        cudaFree(cuda_sequence_Configuration_standard[gpu]);
node_within_host.cu:            cudaFree(cuda_parent_IDs[gpu][row]);
node_within_host.cu:        cudaFree(cuda_parent_IDs[gpu]);
node_within_host.cu:        cudaFree(cuda_cell_Index[gpu]);
node_within_host.cu:            cudaFree(cuda_totals_Progeny_Selectivity[gpu][row]);
node_within_host.cu:        cudaFree(cuda_totals_Progeny_Selectivity[gpu]);
node_within_host.cu:        for (int row = 0; row < (start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first); row++)
node_within_host.cu:            cudaFree(cuda_progeny_Configuration[gpu][row]);
node_within_host.cu:            cudaFree(cuda_progeny_Sequences[gpu][row]);
node_within_host.cu:        cudaFree(cuda_progeny_Configuration[gpu]);
node_within_host.cu:        cudaFree(cuda_progeny_Sequences[gpu]);
node_within_host.cu:        cudaFree(cuda_Dead_or_Alive[gpu]);
node_within_host.cu:        cudaStreamDestroy(streams[gpu]);
node_within_host.cu:__global__ void cuda_Progeny_Configurator(int num_Parents_to_Process, int start_Index,
node_within_host.cu:                                          float **cuda_sequence_Configuration_standard, int recombination_Hotspots,
node_within_host.cu:                                          int **cuda_progeny_Configuration, int *cuda_progeny_Stride, int remove_Back)
node_within_host.cu:        int progeny_Fill_start = cuda_progeny_Stride[parent_Index] - remove_Back;
node_within_host.cu:        int progeny_Fill_end = cuda_progeny_Stride[parent_Index + 1] - remove_Back;
node_within_host.cu:        for (int progeny = 0; progeny < cuda_sequence_Configuration_standard[parent_Index][0]; progeny++)
node_within_host.cu:            cuda_progeny_Configuration[progeny_Fill_start + progeny][0] = parent_Index;
node_within_host.cu:                if (progeny < cuda_sequence_Configuration_standard[parent_Index][(hotspot * 2) + 2])
node_within_host.cu:                    cuda_progeny_Configuration[progeny_Fill_start + progeny][hotspot + 1] = parent_Index;
node_within_host.cu:                    cuda_progeny_Configuration[progeny_Fill_start + progeny][hotspot + 1] = -1;
node_within_host.cu:            if (cuda_sequence_Configuration_standard[parent_Index][(hotspot * 2) + 2] > 0)
node_within_host.cu:                if (cuda_sequence_Configuration_standard[parent_Index][(hotspot * 2) + 2] != cuda_sequence_Configuration_standard[parent_Index][0])
node_within_host.cu:                        int temp = cuda_progeny_Configuration[i][hotspot + 1];
node_within_host.cu:                        cuda_progeny_Configuration[i][hotspot + 1] = cuda_progeny_Configuration[j][hotspot + 1];
node_within_host.cu:                        cuda_progeny_Configuration[j][hotspot + 1] = temp;
node_within_host.cu:                                            float **cuda_sequence_Configuration_standard, int recombination_Hotspots,
node_within_host.cu:                                            int **progeny_Configuration, int *cuda_progeny_Stride, int progeny_Total, int remove_Back)
node_within_host.cu:    int **cuda_progeny_Configuration;
node_within_host.cu:    //= functions.create_CUDA_2D_int(progeny_Total, 1 + recombination_Hotspots);
node_within_host.cu:    cudaMallocManaged(&cuda_progeny_Configuration, progeny_Total * sizeof(int *));
node_within_host.cu:        cudaMalloc((void **)&(cuda_progeny_Configuration[row]), (1 + recombination_Hotspots) * sizeof(int));
node_within_host.cu:    cuda_Progeny_Configurator<<<functions.tot_Blocks_array[0], functions.tot_ThreadsperBlock_array[0]>>>(num_Parents_to_Process, start_Index,
node_within_host.cu:                                                                                                         cuda_sequence_Configuration_standard, recombination_Hotspots,
node_within_host.cu:                                                                                                         cuda_progeny_Configuration, cuda_progeny_Stride, remove_Back);
node_within_host.cu:    cudaDeviceSynchronize();
node_within_host.cu:        cudaMemcpy(progeny_Configuration[row + remove_Back], cuda_progeny_Configuration[row], (recombination_Hotspots + 1) * sizeof(cuda_progeny_Configuration[0][0]), cudaMemcpyDeviceToHost);
node_within_host.cu:    functions.clear_Array_INT(cuda_progeny_Configuration, progeny_Total);
node_within_host.cu:__global__ void cuda_Parent_configuration(int num_Sequences, int **sequence_INT, int genome_Length, char *sites, float **cuda_sequence_Configuration_standard,
node_within_host.cu:                                          float *cuda_Reference_fitness_survivability_proof_reading, int *cuda_num_effect_Segregating_sites,
node_within_host.cu:                                          float **cuda_sequence_Fitness_changes, float **cuda_sequence_Proof_reading_changes,
node_within_host.cu:                                          int recombination_Hotspots, float **cuda_recombination_hotspot_parameters,
node_within_host.cu:                                          int *cuda_recombination_prob_Stride, float **cuda_recombination_Prob_matrix,
node_within_host.cu:                                          int *cuda_recombination_select_Stride, float **cuda_recombination_Select_matrix,
node_within_host.cu:                                          float *cuda_progeny_distribution_parameters_Array)
node_within_host.cu:        float fitness = cuda_Reference_fitness_survivability_proof_reading[0];
node_within_host.cu:        for (int pos = 0; pos < cuda_num_effect_Segregating_sites[0]; pos++)
node_within_host.cu:            if (sequence_INT[tid][(int)cuda_sequence_Fitness_changes[pos][0] - 1] == 0)
node_within_host.cu:                fitness = fitness * cuda_sequence_Fitness_changes[pos][1];
node_within_host.cu:            else if (sequence_INT[tid][(int)cuda_sequence_Fitness_changes[pos][0] - 1] == 1)
node_within_host.cu:                fitness = fitness * cuda_sequence_Fitness_changes[pos][2];
node_within_host.cu:            else if (sequence_INT[tid][(int)cuda_sequence_Fitness_changes[pos][0] - 1] == 2)
node_within_host.cu:                fitness = fitness * cuda_sequence_Fitness_changes[pos][3];
node_within_host.cu:            else if (sequence_INT[tid][(int)cuda_sequence_Fitness_changes[pos][0] - 1] == 3)
node_within_host.cu:                fitness = fitness * cuda_sequence_Fitness_changes[pos][4];
node_within_host.cu:        if (cuda_progeny_distribution_parameters_Array[0] == 0)
node_within_host.cu:            while (successes < cuda_progeny_distribution_parameters_Array[1])
node_within_host.cu:                if (rand_num < cuda_progeny_distribution_parameters_Array[2])
node_within_host.cu:        else if (cuda_progeny_distribution_parameters_Array[0] == 1)
node_within_host.cu:            // progeny = (int)rand_gamma_node(&localState, cuda_progeny_distribution_parameters_Array[1], cuda_progeny_distribution_parameters_Array[2]);
node_within_host.cu:            for (int j = 0; j < cuda_progeny_distribution_parameters_Array[1]; ++j)
node_within_host.cu:                sum += generateExponential(&localState, 1.0f / cuda_progeny_distribution_parameters_Array[2]);
node_within_host.cu:        else if (cuda_progeny_distribution_parameters_Array[0] == 2)
node_within_host.cu:            progeny = curand_poisson(&localState, cuda_progeny_distribution_parameters_Array[1]);
node_within_host.cu:        cuda_sequence_Configuration_standard[tid][0] = (int)(progeny * fitness);
node_within_host.cu:        if (cuda_Reference_fitness_survivability_proof_reading[2] != -1)
node_within_host.cu:            float proof_Reading = cuda_Reference_fitness_survivability_proof_reading[2];
node_within_host.cu:            for (int pos = 0; pos < cuda_num_effect_Segregating_sites[2]; pos++)
node_within_host.cu:                if (sequence_INT[tid][(int)cuda_sequence_Proof_reading_changes[pos][0] - 1] == 0)
node_within_host.cu:                    proof_Reading = proof_Reading + cuda_sequence_Proof_reading_changes[pos][1];
node_within_host.cu:                else if (sequence_INT[tid][(int)cuda_sequence_Proof_reading_changes[pos][0] - 1] == 1)
node_within_host.cu:                    proof_Reading = proof_Reading + cuda_sequence_Proof_reading_changes[pos][2];
node_within_host.cu:                else if (sequence_INT[tid][(int)cuda_sequence_Proof_reading_changes[pos][0] - 1] == 2)
node_within_host.cu:                    proof_Reading = proof_Reading + cuda_sequence_Proof_reading_changes[pos][3];
node_within_host.cu:                else if (sequence_INT[tid][(int)cuda_sequence_Proof_reading_changes[pos][0] - 1] == 3)
node_within_host.cu:                    proof_Reading = proof_Reading + cuda_sequence_Proof_reading_changes[pos][4];
node_within_host.cu:            cuda_sequence_Configuration_standard[tid][1] = proof_Reading;
node_within_host.cu:            cuda_sequence_Configuration_standard[tid][1] = -1;
node_within_host.cu:                float probability = cuda_recombination_hotspot_parameters[hotspot][2];
node_within_host.cu:                float selectivity = cuda_recombination_hotspot_parameters[hotspot][3];
node_within_host.cu:                for (int stride = cuda_recombination_prob_Stride[hotspot]; stride < cuda_recombination_prob_Stride[hotspot + 1]; stride++)
node_within_host.cu:                    if (sequence_INT[tid][(int)cuda_recombination_Prob_matrix[stride][0] - 1] == 0)
node_within_host.cu:                        probability = probability + cuda_recombination_Prob_matrix[stride][1];
node_within_host.cu:                    else if (sequence_INT[tid][(int)cuda_recombination_Prob_matrix[stride][0] - 1] == 1)
node_within_host.cu:                        probability = probability + cuda_recombination_Prob_matrix[stride][2];
node_within_host.cu:                    else if (sequence_INT[tid][(int)cuda_recombination_Prob_matrix[stride][0] - 1] == 2)
node_within_host.cu:                        probability = probability + cuda_recombination_Prob_matrix[stride][3];
node_within_host.cu:                    else if (sequence_INT[tid][(int)cuda_recombination_Prob_matrix[stride][0] - 1] == 3)
node_within_host.cu:                        probability = probability + cuda_recombination_Prob_matrix[stride][4];
node_within_host.cu:                for (int stride = cuda_recombination_select_Stride[hotspot]; stride < cuda_recombination_select_Stride[hotspot + 1]; stride++)
node_within_host.cu:                    if (sequence_INT[tid][(int)cuda_recombination_Select_matrix[stride][0] - 1] == 0)
node_within_host.cu:                        selectivity = selectivity * cuda_recombination_Select_matrix[stride][1];
node_within_host.cu:                    else if (sequence_INT[tid][(int)cuda_recombination_Select_matrix[stride][0] - 1] == 1)
node_within_host.cu:                        selectivity = selectivity * cuda_recombination_Select_matrix[stride][2];
node_within_host.cu:                    else if (sequence_INT[tid][(int)cuda_recombination_Select_matrix[stride][0] - 1] == 2)
node_within_host.cu:                        selectivity = selectivity * cuda_recombination_Select_matrix[stride][3];
node_within_host.cu:                    else if (sequence_INT[tid][(int)cuda_recombination_Select_matrix[stride][0] - 1] == 3)
node_within_host.cu:                        selectivity = selectivity * cuda_recombination_Select_matrix[stride][4];
node_within_host.cu:                for (int trial = 0; trial < cuda_sequence_Configuration_standard[tid][0]; trial++)
node_within_host.cu:                cuda_sequence_Configuration_standard[tid][index_Progeny] = hotspot_Progeny;
node_within_host.cu:                cuda_sequence_Configuration_standard[tid][index_Selectivity] = selectivity;
node_within_host.cu:                                                           vector<string> &collected_Sequences, int *CUDA_device_IDs, int &num_Cuda_devices, int &genome_Length,
node_within_host.cu:    cout << "\nConfiguring multi gpu distribution of " << num_of_Sequences_current << " sequence(s)\n";
node_within_host.cu:    int standard_num_per_GPU = num_of_Sequences_current / num_Cuda_devices;
node_within_host.cu:    int remainder = num_of_Sequences_current % num_Cuda_devices;
node_within_host.cu:    vector<pair<int, int>> start_stop_Per_GPU;
node_within_host.cu:    for (int gpu = 0; gpu < num_Cuda_devices; gpu++)
node_within_host.cu:        int start = gpu * standard_num_per_GPU;
node_within_host.cu:        int stop = start + standard_num_per_GPU;
node_within_host.cu:        start_stop_Per_GPU.push_back(make_pair(start, stop));
node_within_host.cu:    start_stop_Per_GPU[num_Cuda_devices - 1].second = start_stop_Per_GPU[num_Cuda_devices - 1].second + remainder;
node_within_host.cu:    for (int gpu = 0; gpu < num_Cuda_devices; gpu++)
node_within_host.cu:        for (int sequence = start_stop_Per_GPU[gpu].first; sequence < start_stop_Per_GPU[gpu].second; sequence++)
node_within_host.cu:    for (int gpu = 0; gpu < num_Cuda_devices; gpu++)
node_within_host.cu:        for (int sequence = start_stop_Per_GPU[gpu].first; sequence < start_stop_Per_GPU[gpu].second; sequence++)
node_within_host.cu:    cudaStream_t streams[num_Cuda_devices];
node_within_host.cu:    cudaDeviceProp deviceProp;
node_within_host.cu:    char *cuda_full_Char[num_Cuda_devices];
node_within_host.cu:    int **cuda_Sequence[num_Cuda_devices];
node_within_host.cu:    float **cuda_sequence_Configuration_standard[num_Cuda_devices];
node_within_host.cu:    float *cuda_Reference_fitness_survivability_proof_reading[num_Cuda_devices];
node_within_host.cu:    int *cuda_mutation_recombination_proof_Reading_availability[num_Cuda_devices];
node_within_host.cu:    int *cuda_num_effect_Segregating_sites[num_Cuda_devices];
node_within_host.cu:    float **cuda_sequence_Fitness_changes[num_Cuda_devices];
node_within_host.cu:    float **cuda_sequence_Proof_reading_changes[num_Cuda_devices];
node_within_host.cu:    float **cuda_recombination_hotspot_parameters[num_Cuda_devices];
node_within_host.cu:    int *cuda_tot_prob_selectivity[num_Cuda_devices];
node_within_host.cu:    int *cuda_recombination_prob_Stride[num_Cuda_devices];
node_within_host.cu:    int *cuda_recombination_select_Stride[num_Cuda_devices];
node_within_host.cu:    float **cuda_recombination_Prob_matrix[num_Cuda_devices];
node_within_host.cu:    float **cuda_recombination_Select_matrix[num_Cuda_devices];
node_within_host.cu:    float *cuda_progeny_distribution_parameters_Array[num_Cuda_devices];
node_within_host.cu:    for (int gpu = 0; gpu < num_Cuda_devices; gpu++)
node_within_host.cu:        cudaSetDevice(CUDA_device_IDs[gpu]);
node_within_host.cu:        cudaGetDeviceProperties(&deviceProp, gpu);
node_within_host.cu:        cout << "Intializing GPU " << CUDA_device_IDs[gpu] << "'s stream: " << deviceProp.name << endl;
node_within_host.cu:        cudaMalloc(&cuda_full_Char[gpu], (start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first) * genome_Length * sizeof(char));
node_within_host.cu:        cudaMemcpy(cuda_full_Char[gpu], full_Char + (start_stop_Per_GPU[gpu].first * genome_Length), (start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first) * genome_Length * sizeof(char), cudaMemcpyHostToDevice);
node_within_host.cu:        cudaMallocManaged(&cuda_Sequence[gpu], (start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first) * sizeof(int *));
node_within_host.cu:        cudaMallocManaged(&cuda_sequence_Configuration_standard[gpu], (start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first) * sizeof(float *));
node_within_host.cu:        for (int row = 0; row < (start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first); row++)
node_within_host.cu:            cudaMalloc((void **)&cuda_Sequence[gpu][row], genome_Length * sizeof(int));
node_within_host.cu:            cudaMalloc((void **)&cuda_sequence_Configuration_standard[gpu][row], (3 + (2 * recombination_Hotspots)) * sizeof(float));
node_within_host.cu:        cudaMallocManaged(&cuda_Reference_fitness_survivability_proof_reading[gpu], 3 * sizeof(float));
node_within_host.cu:        cudaMemcpy(cuda_Reference_fitness_survivability_proof_reading[gpu], Reference_fitness_survivability_proof_reading, 3 * sizeof(float), cudaMemcpyHostToDevice);
node_within_host.cu:        cudaMallocManaged(&cuda_mutation_recombination_proof_Reading_availability[gpu], 3 * sizeof(int));
node_within_host.cu:        cudaMemcpy(cuda_mutation_recombination_proof_Reading_availability[gpu], mutation_recombination_proof_Reading_availability, 3 * sizeof(int), cudaMemcpyHostToDevice);
node_within_host.cu:        cudaMallocManaged(&cuda_num_effect_Segregating_sites[gpu], 3 * sizeof(int));
node_within_host.cu:        cudaMemcpy(cuda_num_effect_Segregating_sites[gpu], num_effect_Segregating_sites, 3 * sizeof(int), cudaMemcpyHostToDevice);
node_within_host.cu:        cudaMallocManaged(&cuda_progeny_distribution_parameters_Array[gpu], 3 * sizeof(float));
node_within_host.cu:        cudaMemcpy(cuda_progeny_distribution_parameters_Array[gpu], progeny_distribution_parameters_Array, 3 * sizeof(float), cudaMemcpyHostToDevice);
node_within_host.cu:        cudaMallocManaged(&cuda_sequence_Fitness_changes[gpu], num_effect_Segregating_sites[0] * sizeof(float *));
node_within_host.cu:            cudaMalloc((void **)&cuda_sequence_Fitness_changes[gpu][row], 5 * sizeof(float));
node_within_host.cu:            cudaMemcpy(cuda_sequence_Fitness_changes[gpu][row], sequence_Fitness_changes[row], 5 * sizeof(float), cudaMemcpyHostToDevice);
node_within_host.cu:        //  cudaMallocManaged(&cuda_sequence_Survivability_changes[gpu], num_effect_Segregating_sites[1] * sizeof(float *));
node_within_host.cu:        //      cudaMalloc((void **)&cuda_sequence_Survivability_changes[gpu][row], 5 * sizeof(float));
node_within_host.cu:        //      cudaMemcpy(cuda_sequence_Survivability_changes[gpu][row], sequence_Survivability_changes[row], 5 * sizeof(float), cudaMemcpyHostToDevice);
node_within_host.cu:        cudaMallocManaged(&cuda_sequence_Proof_reading_changes[gpu], num_effect_Segregating_sites[2] * sizeof(float *));
node_within_host.cu:            cudaMalloc((void **)&cuda_sequence_Proof_reading_changes[gpu][row], 5 * sizeof(float));
node_within_host.cu:            cudaMemcpy(cuda_sequence_Proof_reading_changes[gpu][row], sequence_Proof_reading_changes[row], 5 * sizeof(float), cudaMemcpyHostToDevice);
node_within_host.cu:        cudaMallocManaged(&cuda_tot_prob_selectivity[gpu], 2 * sizeof(int));
node_within_host.cu:        cudaMallocManaged(&cuda_recombination_prob_Stride[gpu], (recombination_Hotspots + 1) * sizeof(int));
node_within_host.cu:        cudaMallocManaged(&cuda_recombination_select_Stride[gpu], (recombination_Hotspots + 1) * sizeof(int));
node_within_host.cu:        cudaMallocManaged(&cuda_recombination_hotspot_parameters[gpu], recombination_Hotspots * sizeof(float *));
node_within_host.cu:            cudaMallocManaged(&cuda_recombination_Prob_matrix[gpu], tot_prob_selectivity[0] * sizeof(float *));
node_within_host.cu:            cudaMallocManaged(&cuda_recombination_Select_matrix[gpu], tot_prob_selectivity[1] * sizeof(float *));
node_within_host.cu:            cudaMemcpy(cuda_tot_prob_selectivity[gpu], tot_prob_selectivity, 2 * sizeof(int), cudaMemcpyHostToDevice);
node_within_host.cu:            cudaMemcpy(cuda_recombination_prob_Stride[gpu], recombination_prob_Stride, (recombination_Hotspots + 1) * sizeof(int), cudaMemcpyHostToDevice);
node_within_host.cu:            cudaMemcpy(cuda_recombination_select_Stride[gpu], recombination_select_Stride, (recombination_Hotspots + 1) * sizeof(int), cudaMemcpyHostToDevice);
node_within_host.cu:                cudaMalloc((void **)&cuda_recombination_hotspot_parameters[gpu][row], 4 * sizeof(float));
node_within_host.cu:                cudaMemcpy(cuda_recombination_hotspot_parameters[gpu][row], recombination_hotspot_parameters[row], 4 * sizeof(float), cudaMemcpyHostToDevice);
node_within_host.cu:                cudaMalloc((void **)&cuda_recombination_Prob_matrix[gpu][row], 5 * sizeof(float));
node_within_host.cu:                cudaMemcpy(cuda_recombination_Prob_matrix[gpu][row], recombination_Prob_matrix[row], 5 * sizeof(float), cudaMemcpyHostToDevice);
node_within_host.cu:                cudaMalloc((void **)&cuda_recombination_Select_matrix[gpu][row], 5 * sizeof(float));
node_within_host.cu:                cudaMemcpy(cuda_recombination_Select_matrix[gpu][row], recombination_Select_matrix[row], 5 * sizeof(float), cudaMemcpyHostToDevice);
node_within_host.cu:            cudaMallocManaged(&cuda_recombination_Prob_matrix[gpu], 0 * sizeof(float *));
node_within_host.cu:            cudaMallocManaged(&cuda_recombination_Select_matrix[gpu], 0 * sizeof(float *));
node_within_host.cu:        cudaStreamCreate(&streams[gpu]);
node_within_host.cu:    cout << "Loaded " << num_of_Sequences_current << " sequence(s) and all pre-requisites to the GPU(s)\nInitiating GPU(s) execution\n";
node_within_host.cu:    for (int gpu = 0; gpu < num_Cuda_devices; gpu++)
node_within_host.cu:        cudaSetDevice(CUDA_device_IDs[gpu]);
node_within_host.cu:        // (start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first, cuda_Sequence[gpu], genome_Length, cuda_full_Char[gpu]);
node_within_host.cu:        cuda_Parent_configuration<<<functions.tot_Blocks_array[gpu], functions.tot_ThreadsperBlock_array[gpu], 0, streams[gpu]>>>(start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first, cuda_Sequence[gpu], genome_Length, cuda_full_Char[gpu], cuda_sequence_Configuration_standard[gpu],
node_within_host.cu:                                                                                                                                  cuda_Reference_fitness_survivability_proof_reading[gpu], cuda_num_effect_Segregating_sites[gpu],
node_within_host.cu:                                                                                                                                  cuda_sequence_Fitness_changes[gpu], cuda_sequence_Proof_reading_changes[gpu],
node_within_host.cu:                                                                                                                                  recombination_Hotspots, cuda_recombination_hotspot_parameters[gpu],
node_within_host.cu:                                                                                                                                  cuda_recombination_prob_Stride[gpu], cuda_recombination_Prob_matrix[gpu],
node_within_host.cu:                                                                                                                                  cuda_recombination_select_Stride[gpu], cuda_recombination_Select_matrix[gpu],
node_within_host.cu:                                                                                                                                  cuda_progeny_distribution_parameters_Array[gpu]);
node_within_host.cu:    for (int gpu = 0; gpu < num_Cuda_devices; gpu++)
node_within_host.cu:        cudaSetDevice(CUDA_device_IDs[gpu]);
node_within_host.cu:        cudaStreamSynchronize(streams[gpu]);
node_within_host.cu:    cout << "GPU(s) streams completed and synchronized\nCopying data from GPU to Host memory\n";
node_within_host.cu:    for (int gpu = 0; gpu < num_Cuda_devices; gpu++)
node_within_host.cu:        cudaSetDevice(CUDA_device_IDs[gpu]);
node_within_host.cu:        for (int row = 0; row < (start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first); row++)
node_within_host.cu:            cudaMemcpy(parent_Sequences[start_stop_Per_GPU[gpu].first + row + start_Index], cuda_Sequence[gpu][row], genome_Length * sizeof(int), cudaMemcpyDeviceToHost);
node_within_host.cu:            cudaMemcpy(sequence_Configuration_standard[start_stop_Per_GPU[gpu].first + row + start_Index], cuda_sequence_Configuration_standard[gpu][row], (2 + (2 * recombination_Hotspots)) * sizeof(float), cudaMemcpyDeviceToHost);
node_within_host.cu:    cout << "Terminating GPU streams: ";
node_within_host.cu:    for (int gpu = 0; gpu < num_Cuda_devices; gpu++)
node_within_host.cu:        cudaSetDevice(CUDA_device_IDs[gpu]);
node_within_host.cu:        cudaFree(cuda_full_Char[gpu]);
node_within_host.cu:        for (int row = 0; row < (start_stop_Per_GPU[gpu].second - start_stop_Per_GPU[gpu].first); row++)
node_within_host.cu:            cudaFree(cuda_Sequence[gpu][row]);
node_within_host.cu:            cudaFree(cuda_sequence_Configuration_standard[gpu][row]);
node_within_host.cu:        cudaFree(cuda_Sequence[gpu]);
node_within_host.cu:        cudaFree(cuda_sequence_Configuration_standard[gpu]);
node_within_host.cu:        cudaFree(cuda_Reference_fitness_survivability_proof_reading[gpu]);
node_within_host.cu:        cudaFree(cuda_mutation_recombination_proof_Reading_availability[gpu]);
node_within_host.cu:        cudaFree(cuda_num_effect_Segregating_sites[gpu]);
node_within_host.cu:            cudaFree(cuda_sequence_Fitness_changes[gpu][row]);
node_within_host.cu:        cudaFree(cuda_sequence_Fitness_changes[gpu]);
node_within_host.cu:            cudaFree(cuda_sequence_Proof_reading_changes[gpu][row]);
node_within_host.cu:        cudaFree(cuda_sequence_Proof_reading_changes[gpu]);
node_within_host.cu:        cudaFree(cuda_tot_prob_selectivity[gpu]);
node_within_host.cu:        cudaFree(cuda_recombination_prob_Stride[gpu]);
node_within_host.cu:        cudaFree(cuda_recombination_select_Stride[gpu]);
node_within_host.cu:                cudaFree(cuda_recombination_hotspot_parameters[gpu][row]);
node_within_host.cu:                cudaFree(cuda_recombination_Prob_matrix[gpu][row]);
node_within_host.cu:                cudaFree(cuda_recombination_Select_matrix[gpu][row]);
node_within_host.cu:        cudaFree(cuda_recombination_hotspot_parameters[gpu]);
node_within_host.cu:        cudaFree(cuda_recombination_Prob_matrix[gpu]);
node_within_host.cu:        cudaFree(cuda_recombination_Select_matrix[gpu]);
node_within_host.cu:        cudaFree(cuda_progeny_distribution_parameters_Array[gpu]);
node_within_host.cu:        cudaStreamDestroy(streams[gpu]);
prometheus.cu:                             * The GPU is permitted to handle only a certain max number of SNPs at a time.
prometheus.cu:                             * Therefore the number of rounds of GPU processing and,
prometheus.cu:                             * @param GPU_rounds_full rounds requiring the max set of SNPs to be processed.
prometheus.cu:                             * @param GPU_rounds_partial rounds requiring the remaining set of SNPs to be processed.
prometheus.cu:                            int GPU_rounds_full = tot_Segs / SNPs_per_Run;
prometheus.cu:                            int GPU_rounds_partial = tot_Segs % SNPs_per_Run;
prometheus.cu:                            for (int i = 0; i < GPU_rounds_full; i++)
prometheus.cu:                            if (GPU_rounds_partial != 0)
prometheus.cu:                                int start = tot_Segs - GPU_rounds_partial;
prometheus.cu:                             * Concatenation of SNPs for GPU processing is also done in parallel,
prometheus.cu:                             * Used to indicate that no GPU based processing needs to be done.
prometheus.cu:            int GPU_rounds_full = tot_Segs / SNPs_per_Run;
prometheus.cu:            int GPU_rounds_partial = tot_Segs % SNPs_per_Run;
prometheus.cu:            for (int i = 0; i < GPU_rounds_full; i++)
prometheus.cu:            if (GPU_rounds_partial != 0)
prometheus.cu:                int start = tot_Segs - GPU_rounds_partial;
prometheus.cu:                 * The GPU is permitted to handle only a certain max number of SNPs at a time.
prometheus.cu:                 * Therefore the number of rounds of GPU processing and,
prometheus.cu:                 * @param GPU_rounds_full rounds requiring the max set of SNPs to be processed.
prometheus.cu:                 * @param GPU_rounds_partial rounds requiring the remaining set of SNPs to be processed.
prometheus.cu:                int GPU_rounds_full = tot_Segs / SNPs_per_Run;
prometheus.cu:                int GPU_rounds_partial = tot_Segs % SNPs_per_Run;
prometheus.cu:                for (int i = 0; i < GPU_rounds_full; i++)
prometheus.cu:                if (GPU_rounds_partial != 0)
prometheus.cu:                    int start = tot_Segs - GPU_rounds_partial;
prometheus.cu:                 * Concatenation of SNPs for GPU processing is also done in parallel,
prometheus.cu:                 * Used to indicate that no GPU based processing needs to be done.
prometheus.cu:            int GPU_rounds_full = tot_Segs / SNPs_per_Run;
prometheus.cu:            int GPU_rounds_partial = tot_Segs % SNPs_per_Run;
prometheus.cu:            for (int i = 0; i < GPU_rounds_full; i++)
prometheus.cu:            if (GPU_rounds_partial != 0)
prometheus.cu:                int start = tot_Segs - GPU_rounds_partial;
prometheus.cu:     * The GPU is permitted to handle only a certain max number of SNPs at a time.
prometheus.cu:     * Therefore the number of rounds of GPU processing and,
prometheus.cu:     * @param GPU_rounds_full rounds requiring the max set of SNPs to be processed.
prometheus.cu:     * @param GPU_rounds_partial rounds requiring the remaining set of SNPs to be processed.
prometheus.cu:    int GPU_rounds_full = tot_Segs / SNPs_per_Run;
prometheus.cu:    int GPU_rounds_partial = tot_Segs % SNPs_per_Run;
prometheus.cu:    for (int i = 0; i < GPU_rounds_full; i++)
prometheus.cu:    if (GPU_rounds_partial != 0)
prometheus.cu:        int start = tot_Segs - GPU_rounds_partial;
prometheus.cu:     * Concatenation of SNPs for GPU processing is also done in parallel,
prometheus.cu:__global__ void cuda_neutrality_Prometheus(char *sites, int *index, int num_Segregrating_sites, int *theta_Partials, int *VALID_or_NOT, int *MA_count, int *ne, int *ns, int *cuda_pos_start_Index, int *cuda_pos_end_Index, int start)
prometheus.cu:     * ! All GPU processing is similar to their normal mode counterparts except for the extraction of the POS data in the POS column.
prometheus.cu:     * @param cuda_pos_start_Index stores the start position in the array.
prometheus.cu:     * @param cuda_pos_end_Index stores the stop position in the array.
prometheus.cu:        cuda_pos_start_Index[tid] = i;
prometheus.cu:        cuda_pos_end_Index[tid] = i - 1;
prometheus.cu:        int *cuda_MA_Count;
prometheus.cu:        cudaMallocManaged(&cuda_MA_Count, tot_Segs * sizeof(int));
prometheus.cu:        int *ne_CUDA, *ns_CUDA;
prometheus.cu:        cudaMallocManaged(&ne_CUDA, tot_Segs * sizeof(int));
prometheus.cu:        cudaMallocManaged(&ns_CUDA, tot_Segs * sizeof(int));
prometheus.cu:        int *cuda_Theta_partials;
prometheus.cu:        cudaMallocManaged(&cuda_Theta_partials, tot_Segs * sizeof(int));
prometheus.cu:         * To prevent GPU overloading the SNPs are processed in batches.
prometheus.cu:            char *cuda_full_Char;
prometheus.cu:            cudaMallocManaged(&cuda_full_Char, (Seg_sites.size() + 1) * sizeof(char));
prometheus.cu:            int *cuda_site_Index;
prometheus.cu:            cudaMallocManaged(&cuda_site_Index, (total_Segs + 1) * sizeof(int));
prometheus.cu:            cudaMemcpy(cuda_full_Char, full_Char, (Seg_sites.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
prometheus.cu:            cudaMemcpy(cuda_site_Index, site_Index, (total_Segs + 1) * sizeof(int), cudaMemcpyHostToDevice);
prometheus.cu:            int *cuda_VALID_or_NOT, *VALID_or_NOT;
prometheus.cu:            cudaMallocManaged(&cuda_VALID_or_NOT, total_Segs * sizeof(int));
prometheus.cu:            int *pos_start_Index, *pos_end_Index, *cuda_pos_start_Index, *cuda_pos_end_Index;
prometheus.cu:            cudaMallocManaged(&cuda_pos_start_Index, total_Segs * sizeof(int));
prometheus.cu:            cudaMallocManaged(&cuda_pos_end_Index, total_Segs * sizeof(int));
prometheus.cu:            cuda_neutrality_Prometheus<<<tot_Blocks, tot_ThreadsperBlock>>>(cuda_full_Char, cuda_site_Index, total_Segs, cuda_Theta_partials, cuda_VALID_or_NOT, cuda_MA_Count, ne_CUDA, ns_CUDA, cuda_pos_start_Index, cuda_pos_end_Index, start);
prometheus.cu:            cudaError_t err = cudaGetLastError();
prometheus.cu:            if (err != cudaSuccess)
prometheus.cu:                printf("CUDA Error: %s\n", cudaGetErrorString(err));
prometheus.cu:            cudaDeviceSynchronize();
prometheus.cu:            cudaMemcpy(VALID_or_NOT, cuda_VALID_or_NOT, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:            cudaMemcpy(pos_start_Index, cuda_pos_start_Index, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:            cudaMemcpy(pos_end_Index, cuda_pos_end_Index, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:            cudaFree(cuda_site_Index);
prometheus.cu:            cudaFree(cuda_full_Char);
prometheus.cu:            cudaFree(cuda_VALID_or_NOT);
prometheus.cu:            cudaFree(cuda_pos_start_Index);
prometheus.cu:            cudaFree(cuda_pos_end_Index);
prometheus.cu:             * ! After each GPU round positions are extracted from the SNP's and they are indexed.
prometheus.cu:        cudaMemcpy(MA_Count, cuda_MA_Count, tot_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:        cudaMemcpy(ne, ne_CUDA, tot_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:        cudaMemcpy(ns, ns_CUDA, tot_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:        cudaMemcpy(Theta_partials, cuda_Theta_partials, tot_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:            cudaMemcpy(pre_MA, cuda_MA_Count, tot_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:            cudaMemcpy(pre_Theta_partials, cuda_Theta_partials, tot_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:            cudaMemcpy(pre_ne, ne_CUDA, tot_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:            cudaMemcpy(pre_ns, ns_CUDA, tot_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:        cudaFree(cuda_MA_Count);
prometheus.cu:        cudaFree(ne_CUDA);
prometheus.cu:        cudaFree(ns_CUDA);
prometheus.cu:        cudaFree(cuda_Theta_partials);
prometheus.cu:__global__ void cuda_fay_wu_Prometheus(char *sites, int *index, int num_Segregrating_sites, int *theta_Partials, int *VALID_or_NOT, int *MA_count, int *cuda_pos_start_Index, int *cuda_pos_end_Index, int start)
prometheus.cu:     * ! All GPU processing is similar to their normal mode counterparts except for the extraction of the POS data in the POS column.
prometheus.cu:     * @param cuda_pos_start_Index stores the start position in the array.
prometheus.cu:     * @param cuda_pos_end_Index stores the stop position in the array.
prometheus.cu:        cuda_pos_start_Index[tid] = i;
prometheus.cu:        cuda_pos_end_Index[tid] = i - 1;
prometheus.cu:     * If the previous file segment list and the current one are the same no GPU processing will be conducted.
prometheus.cu:        int *cuda_MA_Count;
prometheus.cu:        cudaMallocManaged(&cuda_MA_Count, tot_Segs * sizeof(int));
prometheus.cu:        int *cuda_Theta_partials;
prometheus.cu:        cudaMallocManaged(&cuda_Theta_partials, tot_Segs * sizeof(int));
prometheus.cu:         * To prevent GPU overloading the SNPs are processed in batches.
prometheus.cu:            char *cuda_full_Char;
prometheus.cu:            cudaMallocManaged(&cuda_full_Char, (Seg_sites.size() + 1) * sizeof(char));
prometheus.cu:            int *cuda_site_Index;
prometheus.cu:            cudaMallocManaged(&cuda_site_Index, (total_Segs + 1) * sizeof(int));
prometheus.cu:            cudaMemcpy(cuda_full_Char, full_Char, (Seg_sites.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
prometheus.cu:            cudaMemcpy(cuda_site_Index, site_Index, (total_Segs + 1) * sizeof(int), cudaMemcpyHostToDevice);
prometheus.cu:            int *cuda_VALID_or_NOT, *VALID_or_NOT;
prometheus.cu:            cudaMallocManaged(&cuda_VALID_or_NOT, total_Segs * sizeof(int));
prometheus.cu:            int *pos_start_Index, *pos_end_Index, *cuda_pos_start_Index, *cuda_pos_end_Index;
prometheus.cu:            cudaMallocManaged(&cuda_pos_start_Index, total_Segs * sizeof(int));
prometheus.cu:            cudaMallocManaged(&cuda_pos_end_Index, total_Segs * sizeof(int));
prometheus.cu:            cuda_fay_wu_Prometheus<<<tot_Blocks, tot_ThreadsperBlock>>>(cuda_full_Char, cuda_site_Index, total_Segs, cuda_Theta_partials, cuda_VALID_or_NOT, cuda_MA_Count, cuda_pos_start_Index, cuda_pos_end_Index, start);
prometheus.cu:            cudaError_t err = cudaGetLastError();
prometheus.cu:            if (err != cudaSuccess)
prometheus.cu:                printf("CUDA Error: %s\n", cudaGetErrorString(err));
prometheus.cu:            cudaDeviceSynchronize();
prometheus.cu:            cudaMemcpy(VALID_or_NOT, cuda_VALID_or_NOT, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:            cudaMemcpy(pos_start_Index, cuda_pos_start_Index, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:            cudaMemcpy(pos_end_Index, cuda_pos_end_Index, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:            cudaFree(cuda_site_Index);
prometheus.cu:            cudaFree(cuda_full_Char);
prometheus.cu:            cudaFree(cuda_VALID_or_NOT);
prometheus.cu:            cudaFree(cuda_pos_start_Index);
prometheus.cu:            cudaFree(cuda_pos_end_Index);
prometheus.cu:             * ! After each GPU round positions are extracted from the SNP's and they are indexed.
prometheus.cu:        cudaMemcpy(MA_Count, cuda_MA_Count, tot_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:        cudaMemcpy(Theta_partials, cuda_Theta_partials, tot_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:            cudaMemcpy(pre_MA, cuda_MA_Count, tot_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:            cudaMemcpy(pre_Theta_partials, cuda_Theta_partials, tot_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:        cudaFree(cuda_MA_Count);
prometheus.cu:        cudaFree(cuda_Theta_partials);
prometheus.cu:__global__ void cuda_fu_li_Prometheus(char *sites, int *index, int tot_Segregrating_sites, int *VALID_or_NOT, int *MA_count, int *ne, int *ns, int *cuda_pos_start_Index, int *cuda_pos_end_Index, int start)
prometheus.cu:     * ! All GPU processing is similar to their normal mode counterparts except for the extraction of the POS data in the POS column.
prometheus.cu:     * @param cuda_pos_start_Index stores the start position in the array.
prometheus.cu:     * @param cuda_pos_end_Index stores the stop position in the array.
prometheus.cu:        cuda_pos_start_Index[tid] = i;
prometheus.cu:        cuda_pos_end_Index[tid] = i - 1;
prometheus.cu:     * If the previous file segment list and the current one are the same no GPU processing will be conducted.
prometheus.cu:        int *cuda_MA_Count;
prometheus.cu:        cudaMallocManaged(&cuda_MA_Count, tot_Segs * sizeof(int));
prometheus.cu:        int *ne_CUDA, *ns_CUDA;
prometheus.cu:        cudaMallocManaged(&ne_CUDA, tot_Segs * sizeof(int));
prometheus.cu:        cudaMallocManaged(&ns_CUDA, tot_Segs * sizeof(int));
prometheus.cu:         * To prevent GPU overloading the SNPs are processed in batches.
prometheus.cu:            char *cuda_full_Char;
prometheus.cu:            cudaMallocManaged(&cuda_full_Char, (Seg_sites.size() + 1) * sizeof(char));
prometheus.cu:            int *cuda_site_Index;
prometheus.cu:            cudaMallocManaged(&cuda_site_Index, (total_Segs + 1) * sizeof(int));
prometheus.cu:            cudaMemcpy(cuda_full_Char, full_Char, (Seg_sites.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
prometheus.cu:            cudaMemcpy(cuda_site_Index, site_Index, (total_Segs + 1) * sizeof(int), cudaMemcpyHostToDevice);
prometheus.cu:            int *cuda_VALID_or_NOT, *VALID_or_NOT;
prometheus.cu:            cudaMallocManaged(&cuda_VALID_or_NOT, total_Segs * sizeof(int));
prometheus.cu:            int *pos_start_Index, *pos_end_Index, *cuda_pos_start_Index, *cuda_pos_end_Index;
prometheus.cu:            cudaMallocManaged(&cuda_pos_start_Index, total_Segs * sizeof(int));
prometheus.cu:            cudaMallocManaged(&cuda_pos_end_Index, total_Segs * sizeof(int));
prometheus.cu:            cuda_fu_li_Prometheus<<<tot_Blocks, tot_ThreadsperBlock>>>(cuda_full_Char, cuda_site_Index, total_Segs, cuda_VALID_or_NOT, cuda_MA_Count, ne_CUDA, ns_CUDA, cuda_pos_start_Index, cuda_pos_end_Index, start);
prometheus.cu:            cudaError_t err = cudaGetLastError();
prometheus.cu:            if (err != cudaSuccess)
prometheus.cu:                printf("CUDA Error: %s\n", cudaGetErrorString(err));
prometheus.cu:            cudaDeviceSynchronize();
prometheus.cu:            cudaMemcpy(VALID_or_NOT, cuda_VALID_or_NOT, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:            cudaMemcpy(pos_start_Index, cuda_pos_start_Index, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:            cudaMemcpy(pos_end_Index, cuda_pos_end_Index, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:            cudaFree(cuda_site_Index);
prometheus.cu:            cudaFree(cuda_full_Char);
prometheus.cu:            cudaFree(cuda_VALID_or_NOT);
prometheus.cu:            cudaFree(cuda_pos_start_Index);
prometheus.cu:            cudaFree(cuda_pos_end_Index);
prometheus.cu:             * ! After each GPU round positions are extracted from the SNP's and they are indexed.
prometheus.cu:        cudaMemcpy(MA_Count, cuda_MA_Count, tot_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:        cudaMemcpy(ne, ne_CUDA, tot_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:        cudaMemcpy(ns, ns_CUDA, tot_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:            cudaMemcpy(pre_MA, cuda_MA_Count, tot_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:            cudaMemcpy(pre_ne, ne_CUDA, tot_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:            cudaMemcpy(pre_ns, ns_CUDA, tot_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:        cudaFree(cuda_MA_Count);
prometheus.cu:        cudaFree(ne_CUDA);
prometheus.cu:        cudaFree(ns_CUDA);
prometheus.cu:__global__ void cuda_tajima_Prometheus(char *sites, int *index, int tot_Segregrating_sites, int *VALID_or_NOT, int *MA_count, int *cuda_pos_start_Index, int *cuda_pos_end_Index, int start)
prometheus.cu:     * ! All GPU processing is similar to their normal mode counterparts except for the extraction of the POS data in the POS column.
prometheus.cu:     * @param cuda_pos_start_Index stores the start position in the array.
prometheus.cu:     * @param cuda_pos_end_Index stores the stop position in the array.
prometheus.cu:        cuda_pos_start_Index[tid] = i;
prometheus.cu:        cuda_pos_end_Index[tid] = i - 1;
prometheus.cu:     * Will spawn threads based on the umber of GPU rounds needed.
prometheus.cu:     * Will concat the segments for GPU processing per GPU rounds.
prometheus.cu:     * If the previous file segment list and the current one are the same no GPU processing will be conducted.
prometheus.cu:        int *cuda_MA_Count;
prometheus.cu:        cudaMallocManaged(&cuda_MA_Count, tot_Segs * sizeof(int));
prometheus.cu:         * To prevent GPU overloading the SNPs are processed in batches.
prometheus.cu:            char *cuda_full_Char;
prometheus.cu:            cudaMallocManaged(&cuda_full_Char, (Seg_sites.size() + 1) * sizeof(char));
prometheus.cu:            int *cuda_site_Index;
prometheus.cu:            cudaMallocManaged(&cuda_site_Index, (total_Segs + 1) * sizeof(int));
prometheus.cu:            cudaMemcpy(cuda_full_Char, full_Char, (Seg_sites.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
prometheus.cu:            cudaMemcpy(cuda_site_Index, site_Index, (total_Segs + 1) * sizeof(int), cudaMemcpyHostToDevice);
prometheus.cu:            int *cuda_VALID_or_NOT, *VALID_or_NOT;
prometheus.cu:            cudaMallocManaged(&cuda_VALID_or_NOT, total_Segs * sizeof(int));
prometheus.cu:            int *pos_start_Index, *pos_end_Index, *cuda_pos_start_Index, *cuda_pos_end_Index;
prometheus.cu:            cudaMallocManaged(&cuda_pos_start_Index, total_Segs * sizeof(int));
prometheus.cu:            cudaMallocManaged(&cuda_pos_end_Index, total_Segs * sizeof(int));
prometheus.cu:            cuda_tajima_Prometheus<<<tot_Blocks, tot_ThreadsperBlock>>>(cuda_full_Char, cuda_site_Index, total_Segs, cuda_VALID_or_NOT, cuda_MA_Count, cuda_pos_start_Index, cuda_pos_end_Index, start);
prometheus.cu:            cudaError_t err = cudaGetLastError();
prometheus.cu:            if (err != cudaSuccess)
prometheus.cu:                printf("CUDA Error: %s\n", cudaGetErrorString(err));
prometheus.cu:            cudaDeviceSynchronize();
prometheus.cu:            cudaMemcpy(VALID_or_NOT, cuda_VALID_or_NOT, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:            cudaMemcpy(pos_start_Index, cuda_pos_start_Index, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:            cudaMemcpy(pos_end_Index, cuda_pos_end_Index, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:            cudaFree(cuda_site_Index);
prometheus.cu:            cudaFree(cuda_full_Char);
prometheus.cu:            cudaFree(cuda_VALID_or_NOT);
prometheus.cu:            cudaFree(cuda_pos_start_Index);
prometheus.cu:            cudaFree(cuda_pos_end_Index);
prometheus.cu:             * ! After each GPU round positions are extracted from the SNP's and they are indexed.
prometheus.cu:        cudaMemcpy(MA_Count, cuda_MA_Count, tot_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:            cudaMemcpy(pre_MA, cuda_MA_Count, tot_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:        // char *cuda_full_Char;
prometheus.cu:        // cudaMallocManaged(&cuda_full_Char, (Seg_sites.size() + 1) * sizeof(char));
prometheus.cu:        // int *cuda_site_Index;
prometheus.cu:        // cudaMallocManaged(&cuda_site_Index, (tot_Segs + 1) * sizeof(int));
prometheus.cu:        // cudaMemcpy(cuda_full_Char, full_Char, (Seg_sites.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
prometheus.cu:        // cudaMemcpy(cuda_site_Index, site_Index, (tot_Segs + 1) * sizeof(int), cudaMemcpyHostToDevice);
prometheus.cu:        // int *cuda_VALID_or_NOT, *VALID_or_NOT;
prometheus.cu:        // cudaMallocManaged(&cuda_VALID_or_NOT, tot_Segs * sizeof(int));
prometheus.cu:        // int *cuda_MA_Count, *MA_Count;
prometheus.cu:        // cudaMallocManaged(&cuda_MA_Count, tot_Segs * sizeof(int));
prometheus.cu:        // int *pos_start_Index, *pos_end_Index, *cuda_pos_start_Index, *cuda_pos_end_Index;
prometheus.cu:        // cudaMallocManaged(&cuda_pos_start_Index, tot_Segs * sizeof(int));
prometheus.cu:        // cudaMallocManaged(&cuda_pos_end_Index, tot_Segs * sizeof(int));
prometheus.cu:        // cuda_tajima_Prometheus<<<tot_Blocks, tot_ThreadsperBlock>>>(cuda_full_Char, cuda_site_Index, tot_Segs, cuda_VALID_or_NOT, cuda_MA_Count, cuda_pos_start_Index, cuda_pos_end_Index);
prometheus.cu:        // cudaDeviceSynchronize();
prometheus.cu:        // cudaMemcpy(VALID_or_NOT, cuda_VALID_or_NOT, tot_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:        // cudaMemcpy(MA_Count, cuda_MA_Count, tot_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:        // cudaMemcpy(pos_start_Index, cuda_pos_start_Index, tot_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:        // cudaMemcpy(pos_end_Index, cuda_pos_end_Index, tot_Segs * sizeof(int), cudaMemcpyDeviceToHost);
prometheus.cu:        // cudaFree(cuda_site_Index);
prometheus.cu:        cudaFree(cuda_MA_Count);
prometheus.cu:        // cudaFree(cuda_full_Char);
prometheus.cu:     * Responsible for extracting the positions from the GPU processed SNP data and,
mk_test.cuh:#include "cuda_runtime.h"
mk_test.cuh:    char *cuda_stop_Codons;
mk_test.cuh:    char *cuda_reference;
mk_test.cuh:    mk_test(string reference_Path, string alignment_Path, string gene_List, string input_Folder, string ouput_Path, int cuda_ID, string intermediate_Path, int ploidy, string genetic_Code, string start_Codons, string stop_Codons, string mode, string ORF_mode);
parameter_load.cpp:void parameter_load::get_parameters(int &CUDA_device_ID, string &parent_SEQ_folder,
parameter_load.cpp:                    if (line_Data[0] == "\"CUDA Device ID\"")
parameter_load.cpp:                        CUDA_device_ID = get_INT(line_Data[1]);
parameter_load.cpp:                        // cout << CUDA_device_ID << endl;
mk_test.cu:mk_test::mk_test(string reference_Path, string alignment_Path, string gene_List, string input_Folder, string ouput_Path, int cuda_ID, string intermediate_Path, int ploidy, string genetic_Code, string start_Codons, string stop_Codons, string mode, string ORF_mode)
mk_test.cu:    cout << "Initiating CUDA powered McDonald–Kreitman Neutrality Index (NI) test calculator" << endl
mk_test.cu:    cudaSetDevice(cuda_ID);
mk_test.cu:    cout << "Properties of selected CUDA GPU:" << endl;
mk_test.cu:    cudaDeviceProp prop;
mk_test.cu:    cudaGetDeviceProperties(&prop, cuda_ID);
mk_test.cu:    cout << "GPU number\t: " << cuda_ID << endl;
mk_test.cu:    cout << "GPU name\t: " << prop.name << endl;
mk_test.cu:    cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);
mk_test.cu:    cout << "GPU memory (GB)\t: " << l_Total / (1000 * 1000 * 1000) << endl;
mk_test.cu:    cout << "GPU number of multiprocessor(s)\t: " << prop.multiProcessorCount << endl;
mk_test.cu:    cout << "GPU block(s) per multiprocessor\t: " << prop.maxBlocksPerMultiProcessor << endl;
mk_test.cu:    cout << "GPU thread(s) per block\t: " << tot_ThreadsperBlock << endl
mk_test.cu:    cudaMallocManaged(&cuda_stop_Codons, (stop_Codon_All.size() + 1) * sizeof(char));
mk_test.cu:    cudaMemcpy(cuda_stop_Codons, stop_Codons, (stop_Codon_All.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
mk_test.cu:                        // calc mk syn and nonsy in cuda. Cant use MAF data cause we dont know which one is the MA
mk_test.cu:__global__ void cuda_process_SNPS(char *sites, int *index, int tot_Segregrating_sites, int *REF_Count_all, int *ALT_Count_all, char *REF_all, char *ALT_all)
mk_test.cu:__global__ void cuda_process_Codons(int codon_Number, int *positions, char *REF, char *Outgroup, char *seg_REF, char *seg_ALT, int SEG_size, int *SEG_positions, int *seg_REF_count, int *seg_ALT_count, int codon_Start, int size_of_alignment_File, int genetic_Code_size, char *index_Genetic_code, int *VALID_or_NOT, int *Ds, int *Dn, int *Ps, int *Pn)
mk_test.cu:    // GET SEG SITE POSITIONS FROM PREVIOUS CUDA FUNCTION
mk_test.cu:    char *cuda_full_Char;
mk_test.cu:    cudaMallocManaged(&cuda_full_Char, (Seg_sites.size() + 1) * sizeof(char));
mk_test.cu:    int *cuda_site_Index;
mk_test.cu:    cudaMallocManaged(&cuda_site_Index, (num_segregrating_Sites + 1) * sizeof(int));
mk_test.cu:    cudaMemcpy(cuda_full_Char, full_Char, (Seg_sites.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
mk_test.cu:    cudaMemcpy(cuda_site_Index, site_Index, (num_segregrating_Sites + 1) * sizeof(int), cudaMemcpyHostToDevice);
mk_test.cu:    int *cuda_REF_Count, *cuda_ALT_Count;
mk_test.cu:    cudaMallocManaged(&cuda_REF_Count, num_segregrating_Sites * sizeof(int));
mk_test.cu:    cudaMallocManaged(&cuda_ALT_Count, num_segregrating_Sites * sizeof(int));
mk_test.cu:    char *cuda_REF, *cuda_ALT;
mk_test.cu:    cudaMallocManaged(&cuda_REF, num_segregrating_Sites * sizeof(char));
mk_test.cu:    cudaMallocManaged(&cuda_ALT, num_segregrating_Sites * sizeof(char));
mk_test.cu:    cuda_process_SNPS<<<tot_Blocks, tot_ThreadsperBlock>>>(cuda_full_Char, cuda_site_Index, num_segregrating_Sites, cuda_REF_Count, cuda_ALT_Count, cuda_REF, cuda_ALT);
mk_test.cu:    cudaDeviceSynchronize();
mk_test.cu:    // cudaMemcpy(REF_Count, cuda_REF_Count, num_segregrating_Sites * sizeof(int), cudaMemcpyDeviceToHost);
mk_test.cu:    // cudaMemcpy(ALT_Count, cuda_ALT_Count, num_segregrating_Sites * sizeof(int), cudaMemcpyDeviceToHost);
mk_test.cu:    // cudaMemcpy(REF, cuda_REF, num_segregrating_Sites * sizeof(char), cudaMemcpyDeviceToHost);
mk_test.cu:    // cudaMemcpy(ALT, cuda_ALT, num_segregrating_Sites * sizeof(char), cudaMemcpyDeviceToHost);
mk_test.cu:    cudaFree(cuda_full_Char);
mk_test.cu:    cudaFree(cuda_site_Index);
mk_test.cu:    int *positions_ARRAY, *cuda_positions_ARRAY;
mk_test.cu:    char *REF_array, *cuda_REF_array, *Outroup_array, *cuda_Outroup_array;
mk_test.cu:    cout << "             Priming GPU" << endl;
mk_test.cu:    cudaMallocManaged(&cuda_positions_ARRAY, size_of_alignment_File * sizeof(int));
mk_test.cu:    cudaMemcpy(cuda_positions_ARRAY, positions_ARRAY, size_of_alignment_File * sizeof(int), cudaMemcpyHostToDevice);
mk_test.cu:    cudaMallocManaged(&cuda_REF_array, size_of_alignment_File * sizeof(char));
mk_test.cu:    cudaMemcpy(cuda_REF_array, REF_array, size_of_alignment_File * sizeof(char), cudaMemcpyHostToDevice);
mk_test.cu:    cudaMallocManaged(&cuda_Outroup_array, size_of_alignment_File * sizeof(char));
mk_test.cu:    cudaMemcpy(cuda_Outroup_array, Outroup_array, size_of_alignment_File * sizeof(char), cudaMemcpyHostToDevice);
mk_test.cu:    char *cuda_index_Gen_code;
mk_test.cu:    cudaMallocManaged(&cuda_index_Gen_code, size_of_genetic_Code * sizeof(char));
mk_test.cu:    cudaMemcpy(cuda_index_Gen_code, index_Gen_code, size_of_genetic_Code * sizeof(char), cudaMemcpyHostToDevice);
mk_test.cu:    int *cuda_Seg_positions; // int *SEG_positions;
mk_test.cu:    cudaMallocManaged(&cuda_Seg_positions, num_segregrating_Sites * sizeof(int));
mk_test.cu:    cudaMemcpy(cuda_Seg_positions, SEG_positions, num_segregrating_Sites * sizeof(int), cudaMemcpyHostToDevice);
mk_test.cu:    int *VALID_or_NOT, *cuda_VALID_or_NOT;
mk_test.cu:    int *Ds, *Dn, *Ps, *Pn, *cuda_Ds, *cuda_Dn, *cuda_Ps, *cuda_Pn;
mk_test.cu:    cudaMallocManaged(&cuda_VALID_or_NOT, num_of_Codons * sizeof(int));
mk_test.cu:    cudaMallocManaged(&cuda_Dn, num_of_Codons * sizeof(int));
mk_test.cu:    cudaMallocManaged(&cuda_Ds, num_of_Codons * sizeof(int));
mk_test.cu:    cudaMallocManaged(&cuda_Pn, num_of_Codons * sizeof(int));
mk_test.cu:    cudaMallocManaged(&cuda_Ps, num_of_Codons * sizeof(int));
mk_test.cu:    // GPU load here
mk_test.cu:    // cuda_process_Codons(int codon_Number, int *positions, char *REF, char *Outgroup, char *seg_REF, char *seg_ALT, int SEG_size, int *SEG_positions, int *seg_REF_count, int *seg_ALT_count, int codon_Start, int size_of_alignment_File, int genetic_Code_size, char *index_Genetic_code, int *VALID_or_NOT, int *Ds, int *Dn, int *Ps, int *Pn)
mk_test.cu:    cout << "             Launching GPU" << endl;
mk_test.cu:    cuda_process_Codons<<<tot_Blocks, tot_ThreadsperBlock>>>(num_of_Codons, cuda_positions_ARRAY, cuda_REF_array, cuda_Outroup_array, cuda_REF, cuda_ALT, num_segregrating_Sites, cuda_Seg_positions, cuda_REF_Count, cuda_ALT_Count, codon_Start, size_of_alignment_File, size_of_genetic_Code, cuda_index_Gen_code, cuda_VALID_or_NOT, cuda_Ds, cuda_Dn, cuda_Ps, cuda_Pn);
mk_test.cu:    cudaDeviceSynchronize();
mk_test.cu:    cudaError_t err = cudaGetLastError();
mk_test.cu:    if (err != cudaSuccess)
mk_test.cu:        printf("CUDA Error: %s\n", cudaGetErrorString(err));
mk_test.cu:    cudaMemcpy(VALID_or_NOT, cuda_VALID_or_NOT, num_of_Codons * sizeof(int), cudaMemcpyDeviceToHost);
mk_test.cu:    cudaMemcpy(Dn, cuda_Dn, num_of_Codons * sizeof(int), cudaMemcpyDeviceToHost);
mk_test.cu:    cudaMemcpy(Ds, cuda_Ds, num_of_Codons * sizeof(int), cudaMemcpyDeviceToHost);
mk_test.cu:    cudaMemcpy(Pn, cuda_Pn, num_of_Codons * sizeof(int), cudaMemcpyDeviceToHost);
mk_test.cu:    cudaMemcpy(Ps, cuda_Ps, num_of_Codons * sizeof(int), cudaMemcpyDeviceToHost);
mk_test.cu:    cudaFree(cuda_Seg_positions);
mk_test.cu:    cudaFree(cuda_REF_Count);
mk_test.cu:    cudaFree(cuda_ALT_Count);
mk_test.cu:    cudaFree(cuda_REF);
mk_test.cu:    cudaFree(cuda_ALT);
mk_test.cu:    cudaFree(cuda_positions_ARRAY);
mk_test.cu:    cudaFree(cuda_REF_array);
mk_test.cu:    cudaFree(cuda_Outroup_array);
mk_test.cu:    cudaFree(cuda_index_Gen_code);
mk_test.cu:    cudaFree(cuda_VALID_or_NOT);
mk_test.cu:    cudaFree(cuda_Dn);
mk_test.cu:    cudaFree(cuda_Ds);
mk_test.cu:    cudaFree(cuda_Pn);
mk_test.cu:    cudaFree(cuda_Ps);
mk_test.cu:    cout << "             GPU launched" << endl;
mk_test.cu:    cudaMallocManaged(&cuda_reference, (full_Reference.size() + 1) * sizeof(char));
mk_test.cu:    cudaMemcpy(cuda_reference, reference_full, (full_Reference.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
mk_test.cu:    cudaFree(cuda_stop_Codons);
mk_test.cu:    cudaFree(cuda_reference);
mk_test.cu:    cudaMallocManaged(&cuda_reference, (full_Reference.size() + 1) * sizeof(char));
mk_test.cu:    cudaMemcpy(cuda_reference, reference_full, (full_Reference.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
mk_test.cu:    cudaFree(cuda_stop_Codons);
mk_test.cu:    cudaFree(cuda_reference);
mk_test.cu:__global__ void cuda_ORF_search(int ORF_nums, int *start_ORFs, char *reference_Full, int reference_Length, char *stop_Codons, int stop_Codons_Length, int gene_End, int *VALID_or_NOT, int *end_ORFs)
mk_test.cu:    int *start_ORF_array, *cuda_start_ORF_array;
mk_test.cu:    cudaMallocManaged(&cuda_start_ORF_array, potential_ORFs * sizeof(int));
mk_test.cu:    int *VALID_or_NOT, *cuda_VALID_or_NOT, *end_ORFs, *cuda_end_ORFs;
mk_test.cu:    cudaMallocManaged(&cuda_VALID_or_NOT, potential_ORFs * sizeof(int));
mk_test.cu:    cudaMallocManaged(&cuda_end_ORFs, potential_ORFs * sizeof(int));
mk_test.cu:    cudaMemcpy(cuda_start_ORF_array, start_ORF_array, potential_ORFs * sizeof(int), cudaMemcpyHostToDevice);
mk_test.cu:    // cuda_ORF_search(int ORF_nums, int *start_ORFs, char *reference_Full, int reference_Length, char *stop_Codons, int stop_Codons_Length, int gene_End, int *VALID_or_NOT, int *end_ORFs)
mk_test.cu:    cuda_ORF_search<<<tot_Blocks, tot_ThreadsperBlock>>>(potential_ORFs, cuda_start_ORF_array, cuda_reference, reference_size, cuda_stop_Codons, stop_Codon_size, gene_End, cuda_VALID_or_NOT, cuda_end_ORFs);
mk_test.cu:    cudaDeviceSynchronize();
mk_test.cu:    cudaMemcpy(VALID_or_NOT, cuda_VALID_or_NOT, potential_ORFs * sizeof(int), cudaMemcpyDeviceToHost);
mk_test.cu:    cudaMemcpy(end_ORFs, cuda_end_ORFs, potential_ORFs * sizeof(int), cudaMemcpyDeviceToHost);
mk_test.cu:    cudaFree(cuda_VALID_or_NOT);
mk_test.cu:    cudaFree(cuda_end_ORFs);
mk_test.cu:    cudaFree(cuda_start_ORF_array);
fay_wu.cu:fay_wu::fay_wu(string gene_List, string input_Folder, string ouput_Path, int cuda_ID, string intermediate_Path, int ploidy)
fay_wu.cu:     cout << "Initiating CUDA powered Fay and Wu's normalized H and E calculator" << endl
fay_wu.cu:     set_Values(gene_List, input_Folder, ouput_Path, cuda_ID, intermediate_Path, ploidy);
fay_wu.cu:     // cudaSetDevice(cuda_ID);
fay_wu.cu:     // cout << "Properties of selected CUDA GPU:" << endl;
fay_wu.cu:     // cudaDeviceProp prop;
fay_wu.cu:     // cudaGetDeviceProperties(&prop, cuda_ID);
fay_wu.cu:     // cout << "GPU number\t: " << cuda_ID << endl;
fay_wu.cu:     // cout << "GPU name\t: " << prop.name << endl;
fay_wu.cu:     // cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);
fay_wu.cu:     // cout << "GPU memory (GB)\t: " << l_Total / (1000 * 1000 * 1000) << endl;
fay_wu.cu:     // cout << "GPU number of multiprocessor(s)\t: " << prop.multiProcessorCount << endl;
fay_wu.cu:     // cout << "GPU block(s) per multiprocessor\t: " << prop.maxBlocksPerMultiProcessor << endl;
fay_wu.cu:     // cout << "GPU thread(s) per block\t: " << tot_ThreadsperBlock << endl
fay_wu.cu:fay_wu::fay_wu(string gene_List, string input_Folder, string ouput_Path, int cuda_ID, string intermediate_Path, int ploidy, string prometheus_Activate, string Multi_read, int number_of_genes, int CPU_cores, int SNPs_per_Run)
fay_wu.cu:     cout << "Initiating CUDA powered Fay and Wu's normalized H and E calculator on PROMETHEUS" << endl
fay_wu.cu:     set_Values(gene_List, input_Folder, ouput_Path, cuda_ID, intermediate_Path, ploidy);
fay_wu.cu:fay_wu::fay_wu(string calc_Mode, int window_Size, int step_Size, string input_Folder, string ouput_Path, int cuda_ID, int ploidy, string prometheus_Activate, string Multi_read, int number_of_genes, int CPU_cores, int SNPs_per_Run)
fay_wu.cu:     cout << "Initiating CUDA powered Fay and Wu's normalized H and E calculator on PROMETHEUS" << endl
fay_wu.cu:     set_Values("", input_Folder, ouput_Path, cuda_ID, "", ploidy);
fay_wu.cu:fay_wu::fay_wu(string calc_Mode, int window_Size, int step_Size, string input_Folder, string ouput_Path, int cuda_ID, int ploidy)
fay_wu.cu:     cout << "Initiating CUDA powered Fay and Wu's normalized H and E calculator" << endl
fay_wu.cu:     set_Values("", input_Folder, ouput_Path, cuda_ID, "", ploidy);
fay_wu.cu:void fay_wu::set_Values(string gene_List, string input_Folder, string ouput_Path, int cuda_ID, string intermediate_Path, int ploidy)
fay_wu.cu:      * Here the first call to the selected CUDA device occurs.
fay_wu.cu:     cudaSetDevice(cuda_ID);
fay_wu.cu:     cout << "Properties of selected CUDA GPU:" << endl;
fay_wu.cu:     cudaDeviceProp prop;
fay_wu.cu:     cudaGetDeviceProperties(&prop, cuda_ID);
fay_wu.cu:     cout << "GPU number\t: " << cuda_ID << endl;
fay_wu.cu:     cout << "GPU name\t: " << prop.name << endl;
fay_wu.cu:     cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);
fay_wu.cu:     cout << "GPU memory (GB)\t: " << l_Total / (1000 * 1000 * 1000) << endl;
fay_wu.cu:     cout << "GPU number of multiprocessor(s)\t: " << prop.multiProcessorCount << endl;
fay_wu.cu:     cout << "GPU block(s) per multiprocessor\t: " << prop.maxBlocksPerMultiProcessor << endl;
fay_wu.cu:     cout << "GPU thread(s) per block\t: " << tot_ThreadsperBlock << endl
fay_wu.cu:                                                   * Information from the SNP is extracted via the GPU.
fay_wu.cu:                                                   * Information from the SNP is extracted via the GPU.
fay_wu.cu:                               * Information from the SNP is extracted via the GPU.
fay_wu.cu:__global__ void cuda_theta_L(char *sites, int *index, int num_Segregrating_sites, int *theta_Partials, int *VALID_or_NOT, int *MA_count)
fay_wu.cu:      * 1. Conversion of SNP strings into char pointers for GPU accessability.
fay_wu.cu:      * 2. Call GPU for extracting MAs (Minor allele) and MAF's (Minor Allele Frequencies).
fay_wu.cu:      * This track is vital for navigating through the data in the GPU. For the data is stored in the form of a 1D array.
fay_wu.cu:      * @param cuda_full_Char is used by the GPU. Is a COPY of full_Char.
fay_wu.cu:      * @param cuda_site_Index is used by the GPU. Is a COPY of site_Index.
fay_wu.cu:     char *cuda_full_Char;
fay_wu.cu:     cudaMallocManaged(&cuda_full_Char, (Seg_sites.size() + 1) * sizeof(char));
fay_wu.cu:     int *cuda_site_Index;
fay_wu.cu:     cudaMallocManaged(&cuda_site_Index, (num_segregrating_Sites + 1) * sizeof(int));
fay_wu.cu:      * @param cuda_VALID_or_NOT is used to determine if a site is seg site which is VALID or NOT.
fay_wu.cu:      * @param VALID_or_NOT is used by the CPU. Is a COPY of cuda_VALID_or_NOT.
fay_wu.cu:     int *cuda_VALID_or_NOT, *VALID_or_NOT;
fay_wu.cu:     cudaMallocManaged(&cuda_VALID_or_NOT, num_segregrating_Sites * sizeof(int));
fay_wu.cu:      * @param cuda_MA_Count is used to record the MA's count.
fay_wu.cu:      * @param MA_Count is used by the CPU. Is a COPY of cuda_MA_Count.
fay_wu.cu:     int *cuda_MA_Count, *MA_Count;
fay_wu.cu:     cudaMallocManaged(&cuda_MA_Count, num_segregrating_Sites * sizeof(int));
fay_wu.cu:     int *cuda_Theta_partials, *Theta_partials;
fay_wu.cu:     cudaMallocManaged(&cuda_Theta_partials, num_segregrating_Sites * sizeof(int));
fay_wu.cu:      * Transfer of data to the GPU.
fay_wu.cu:     cudaMemcpy(cuda_full_Char, full_Char, (Seg_sites.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
fay_wu.cu:     cudaMemcpy(cuda_site_Index, site_Index, (num_segregrating_Sites + 1) * sizeof(int), cudaMemcpyHostToDevice);
fay_wu.cu:     // cout << "GPU" << endl;
fay_wu.cu:      * CALL THE GPU.
fay_wu.cu:      * * GPU WILL PROCESS THE COLLECTED SEG SITES
fay_wu.cu:     cuda_theta_L<<<tot_Blocks, tot_ThreadsperBlock>>>(cuda_full_Char, cuda_site_Index, num_segregrating_Sites, cuda_Theta_partials, cuda_VALID_or_NOT, cuda_MA_Count);
fay_wu.cu:     cudaDeviceSynchronize();
fay_wu.cu:     cudaMemcpy(VALID_or_NOT, cuda_VALID_or_NOT, num_segregrating_Sites * sizeof(int), cudaMemcpyDeviceToHost);
fay_wu.cu:     cudaMemcpy(MA_Count, cuda_MA_Count, num_segregrating_Sites * sizeof(int), cudaMemcpyDeviceToHost);
fay_wu.cu:     cudaMemcpy(Theta_partials, cuda_Theta_partials, num_segregrating_Sites * sizeof(int), cudaMemcpyDeviceToHost);
fay_wu.cu:     cudaFree(cuda_full_Char);
fay_wu.cu:     cudaFree(cuda_site_Index);
fay_wu.cu:     cudaFree(cuda_MA_Count);
fay_wu.cu:     cudaFree(cuda_VALID_or_NOT);
fay_wu.cu:     cudaFree(cuda_Theta_partials);
fay_wu.cu:     // replace with CUDA addition
fay_wu.cu:     // cout << "GPU DONE" << endl;
fay_wu.cu:__global__ void faywu_Calculation(int N, float *a1_CUDA, float *a2_CUDA)
fay_wu.cu:          a1_CUDA[tid] = (float)1 / (tid + 1);
fay_wu.cu:          a2_CUDA[tid] = (float)1 / ((tid + 1) * (tid + 1));
fay_wu.cu:     float *a1_CUDA, *a2_CUDA;
fay_wu.cu:     cudaMallocManaged(&a1_CUDA, N * sizeof(int));
fay_wu.cu:     cudaMallocManaged(&a2_CUDA, N * sizeof(int));
fay_wu.cu:     faywu_Calculation<<<tot_Blocks, tot_ThreadsperBlock>>>(N, a1_CUDA, a2_CUDA);
fay_wu.cu:     cudaDeviceSynchronize();
fay_wu.cu:     cudaMemcpy(a1_partial, a1_CUDA, N * sizeof(float), cudaMemcpyDeviceToHost);
fay_wu.cu:     cudaMemcpy(a2_partial, a2_CUDA, N * sizeof(float), cudaMemcpyDeviceToHost);
fay_wu.cu:     cudaFree(a1_CUDA);
fay_wu.cu:     cudaFree(a2_CUDA);
fst.cu:fst::fst(string gene_List, string input_Folder, string output_Path, int cuda_ID, string intermediate_Path, int ploidy, string pop_Index_path, string pop_List)
fst.cu:    cout << "Initiating CUDA powered Fst (Fixation Index) calculator" << endl
fst.cu:    set_Values(gene_List, input_Folder, output_Path, cuda_ID, intermediate_Path, ploidy);
fst.cu:    // cudaSetDevice(cuda_ID);
fst.cu:    // cout << "Properties of selected CUDA GPU:" << endl;
fst.cu:    // cudaDeviceProp prop;
fst.cu:    // cudaGetDeviceProperties(&prop, cuda_ID);
fst.cu:    // cout << "GPU number\t: " << cuda_ID << endl;
fst.cu:    // cout << "GPU name\t: " << prop.name << endl;
fst.cu:    // cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);
fst.cu:    // cout << "GPU memory (GB)\t: " << l_Total / (1000 * 1000 * 1000) << endl;
fst.cu:    // cout << "GPU number of multiprocessor(s)\t: " << prop.multiProcessorCount << endl;
fst.cu:    // cout << "GPU block(s) per multiprocessor\t: " << prop.maxBlocksPerMultiProcessor << endl;
fst.cu:    // cout << "GPU thread(s) per block\t: " << tot_ThreadsperBlock << endl
fst.cu:fst::fst(string calc_Mode, int window_Size, int step_Size, string input_Folder, string output_Path, int cuda_ID, int ploidy, string pop_Index_path, string pop_List)
fst.cu:    cout << "Initiating CUDA powered Fst (Fixation Index) calculator" << endl
fst.cu:    set_Values("", input_Folder, output_Path, cuda_ID, "", ploidy);
fst.cu:void fst::set_Values(string gene_List, string input_Folder, string output_Path, int cuda_ID, string intermediate_Path, int ploidy)
fst.cu:    cudaSetDevice(cuda_ID);
fst.cu:    cout << "Properties of selected CUDA GPU:" << endl;
fst.cu:    cudaDeviceProp prop;
fst.cu:    cudaGetDeviceProperties(&prop, cuda_ID);
fst.cu:    cout << "GPU number\t: " << cuda_ID << endl;
fst.cu:    cout << "GPU name\t: " << prop.name << endl;
fst.cu:    cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);
fst.cu:    cout << "GPU memory (GB)\t: " << l_Total / (1000 * 1000 * 1000) << endl;
fst.cu:    cout << "GPU number of multiprocessor(s)\t: " << prop.multiProcessorCount << endl;
fst.cu:    cout << "GPU block(s) per multiprocessor\t: " << prop.maxBlocksPerMultiProcessor << endl;
fst.cu:    cout << "GPU thread(s) per block\t: " << tot_ThreadsperBlock << endl
fst.cu:            // make a list and use in the gpu to validate if we want to process it or not based on pos
fst.cu:                // make a list and use in the gpu to validate if we want to process it or not based on pos
fst.cu:    cudaFree(cuda_sample_Location_array);
fst.cu:    cudaFree(cuda_locations_Size);
fst.cu:    cudaFree(cuda_pop_seqeunce_Size_Array);
fst.cu:    //  cudaMallocManaged(&cuda_pop_Sample_Size_Array, num_Pop_Ids * sizeof(int));
fst.cu:    cudaMallocManaged(&cuda_pop_seqeunce_Size_Array, num_Pop_Ids * sizeof(int));
fst.cu:    cudaMallocManaged(&cuda_locations_Size, num_Pop_Ids * sizeof(int));
fst.cu:    cudaMemcpy(cuda_locations_Size, locations_Size, num_Pop_Ids * sizeof(int), cudaMemcpyHostToDevice);
fst.cu:    cudaMemcpy(cuda_pop_seqeunce_Size_Array, pop_seqeunce_Size_Array, num_Pop_Ids * sizeof(int), cudaMemcpyHostToDevice);
fst.cu:    cudaMallocManaged(&cuda_sample_Location_array, max_Location_Size * (num_Pop_Ids + 1) * sizeof(int));
fst.cu:        cudaMalloc((void **)&tmp_3[i], (num_Pop_Ids + 1) * sizeof(tmp_3[0][0]));
fst.cu:    cudaMemcpy(cuda_sample_Location_array, tmp_3, max_Location_Size * sizeof(int *), cudaMemcpyHostToDevice);
fst.cu:        cudaMemcpy(tmp_3[i], sample_Location_array[i], (num_Pop_Ids + 1) * sizeof(cuda_sample_Location_array[0][0]), cudaMemcpyHostToDevice);
fst.cu:__global__ void cuda_position_Filter(int threads_Needed, int *sites_Size_Array, int pop_ID_count, int **seg_Position_Array, int *Present_or_Not, int **found_Relationships)
fst.cu:__global__ void cuda_process_FST(int segs_Seperate, int num_Pop_Ids, int *pop_Sample_Size_Array, int *pop_seqeunce_Size_Array, int *VALID_or_NOT_ALL, int *VALID_or_NOT_FST, int *REF_Count_all, int *ALT_Count_all, float *Fst, float *CUDA_numerator, float *CUDA_denominators)
fst.cu:            CUDA_numerator[tid] = Ht_calc - Hs_calc;
fst.cu:            CUDA_denominators[tid] = Ht_calc;
fst.cu:__global__ void cuda_process_Segs(int total_Segs, char *sites, int *index, int *pop_Sample_Size_Array, int *seg_Site_pop_ID, int **sample_Location_array, int *VALID_or_NOT, int *REF_Count_all, int *ALT_Count_all)
fst.cu:    int *pop_Seg_size_Array, *cuda_pop_Seg_size_Array;
fst.cu:    int **cuda_seg_Positions, *cuda_PRESENT_or_NOT, **cuda_first_match_Relationships;
fst.cu:    // cudaMallocManaged(&cuda_site_Index, (num_segregrating_Sites + 1) * sizeof(int));
fst.cu:    cudaMallocManaged(&cuda_pop_Seg_size_Array, num_Pop_Ids * sizeof(int));
fst.cu:    cudaMallocManaged(&cuda_seg_Positions, (num_Pop_Ids + 1) * max_Segs * sizeof(int));
fst.cu:        cudaMalloc((void **)&tmp_2[i], (num_Pop_Ids + 1) * sizeof(tmp_2[0][0]));
fst.cu:    cudaMemcpy(cuda_seg_Positions, tmp_2, max_Segs * sizeof(int *), cudaMemcpyHostToDevice);
fst.cu:    cudaMallocManaged(&cuda_PRESENT_or_NOT, first_Seg_sites * sizeof(int));
fst.cu:    //     cudaMalloc((void **)&tmp_3[i], num_Pop_Ids * sizeof(tmp_2[0][0]));
fst.cu:    // cudaMemcpy(cuda_PRESENT_or_NOT, tmp_3, max_Segs * sizeof(int *), cudaMemcpyHostToDevice);
fst.cu:    cudaMallocManaged(&cuda_first_match_Relationships, (num_Pop_Ids + 1) * first_Seg_sites * sizeof(int));
fst.cu:    // cudaMalloc((void **)&cuda_first_match_Relationships, first_Seg_sites * sizeof(int *));
fst.cu:        cudaMalloc((void **)&tmp[i], (num_Pop_Ids + 1) * sizeof(tmp[0][0]));
fst.cu:    cudaMemcpy(cuda_first_match_Relationships, tmp, first_Seg_sites * sizeof(int *), cudaMemcpyHostToDevice);
fst.cu:    //     cudaMalloc((void **)&cuda_first_match_Relationships[i], num_Pop_Ids * sizeof(int));
fst.cu:    // cudaMemcpy(cuda_full_Char, full_Char, (Seg_sites.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
fst.cu:    // gpu
fst.cu:    cudaMemcpy(cuda_pop_Seg_size_Array, pop_Seg_size_Array, num_Pop_Ids * sizeof(int), cudaMemcpyHostToDevice);
fst.cu:    // cudaMemcpy(cuda_seg_Positions, seg_Positions, num_Pop_Ids * max_Segs * sizeof(int), cudaMemcpyHostToDevice);
fst.cu:        cudaMemcpy(tmp_2[i], seg_Positions[i], (num_Pop_Ids + 1) * sizeof(cuda_seg_Positions[0][0]), cudaMemcpyHostToDevice);
fst.cu:    // cudaMemcpy(cuda_PRESENT_or_NOT, PRESENT_or_NOT, num_Pop_Ids * max_Segs * sizeof(int), cudaMemcpyHostToDevice);
fst.cu:    //     cudaMemcpy(tmp_3[i], PRESENT_or_NOT[i], num_Pop_Ids * sizeof(cuda_PRESENT_or_NOT[0][0]), cudaMemcpyHostToDevice);
fst.cu:    //     cudaMemcpy(tmp[i], first_match_Relationships[i], num_Pop_Ids * sizeof(cuda_first_match_Relationships[0][0]), cudaMemcpyHostToDevice);
fst.cu:    // cuda_position_Filter(int *sites_Size_Array, int pop_ID_count, int *seg_Position_Array, int *Present_or_Not, int *found_Relationships);
fst.cu:    cuda_position_Filter<<<tot_Blocks, tot_ThreadsperBlock>>>(pop_Seg_size_Array[0], cuda_pop_Seg_size_Array, num_Pop_Ids, cuda_seg_Positions, cuda_PRESENT_or_NOT, cuda_first_match_Relationships);
fst.cu:    cudaDeviceSynchronize();
fst.cu:    cudaMemcpy(PRESENT_or_NOT, cuda_PRESENT_or_NOT, first_Seg_sites * sizeof(int), cudaMemcpyDeviceToHost);
fst.cu:    // cudaMemcpy(first_match_Relationships, cuda_first_match_Relationships, num_Pop_Ids * first_Seg_sites * sizeof(int), cudaMemcpyDeviceToHost);
fst.cu:    //     cudaMemcpy(PRESENT_or_NOT[i], cuda_PRESENT_or_NOT[i], num_Pop_Ids * sizeof(cuda_PRESENT_or_NOT[0][0]), cudaMemcpyDeviceToHost);
fst.cu:        cudaMemcpy(first_match_Relationships[i], cuda_first_match_Relationships[i], (num_Pop_Ids + 1) * sizeof(cuda_first_match_Relationships[0][0]), cudaMemcpyDeviceToHost);
fst.cu:        int *seg_Site_pop_ID, *cuda_seg_Site_pop_ID;
fst.cu:        cudaMallocManaged(&cuda_seg_Site_pop_ID, tot_num_segregrating_Sites * sizeof(int));
fst.cu:        cudaFree(cuda_PRESENT_or_NOT);
fst.cu:        cudaFree(cuda_pop_Seg_size_Array);
fst.cu:        cudaFree(cuda_seg_Positions);
fst.cu:        cudaFree(cuda_first_match_Relationships);
fst.cu:        char *cuda_full_Char;
fst.cu:        cudaMallocManaged(&cuda_full_Char, (Seg_sites.size() + 1) * sizeof(char));
fst.cu:        int *cuda_site_Index;
fst.cu:        cudaMallocManaged(&cuda_site_Index, (tot_num_segregrating_Sites + 1) * sizeof(int));
fst.cu:        cudaMemcpy(cuda_full_Char, full_Char, (Seg_sites.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
fst.cu:        cudaMemcpy(cuda_site_Index, site_Index, (tot_num_segregrating_Sites + 1) * sizeof(int), cudaMemcpyHostToDevice);
fst.cu:        cudaMemcpy(cuda_seg_Site_pop_ID, seg_Site_pop_ID, tot_num_segregrating_Sites * sizeof(int), cudaMemcpyHostToDevice);
fst.cu:        // int *cuda_pop_seqeunce_Size_Array, *cuda_locations_Size;
fst.cu:        // int **cuda_sample_Location_array;
fst.cu:        // cudaMallocManaged(&cuda_pop_seqeunce_Size_Array, num_Pop_Ids * sizeof(int));
fst.cu:        // cudaMallocManaged(&cuda_locations_Size, num_Pop_Ids * sizeof(int));
fst.cu:        // int *locations_Size, *cuda_locations_Size;
fst.cu:        // int *pop_seqeunce_Size_Array, *cuda_pop_seqeunce_Size_Array;
fst.cu:        // // cudaMallocManaged(&cuda_pop_Sample_Size_Array, num_Pop_Ids * sizeof(int));
fst.cu:        // cudaMallocManaged(&cuda_pop_seqeunce_Size_Array, num_Pop_Ids * sizeof(int));
fst.cu:        // cudaMallocManaged(&cuda_locations_Size, num_Pop_Ids * sizeof(int));
fst.cu:        // cudaMemcpy(cuda_locations_Size, locations_Size, num_Pop_Ids * sizeof(int), cudaMemcpyHostToDevice);
fst.cu:        // cudaMemcpy(cuda_pop_seqeunce_Size_Array, pop_seqeunce_Size_Array, num_Pop_Ids * sizeof(int), cudaMemcpyHostToDevice);
fst.cu:        // int **sample_Location_array, **cuda_sample_Location_array;
fst.cu:        // cudaMemcpy(cuda_locations_Size, locations_Size, num_Pop_Ids * sizeof(int), cudaMemcpyHostToDevice);
fst.cu:        // cudaMemcpy(cuda_pop_seqeunce_Size_Array, pop_seqeunce_Size_Array, num_Pop_Ids * sizeof(int), cudaMemcpyHostToDevice);
fst.cu:        // // copy location array to CUDA kernal
fst.cu:        // cudaMallocManaged(&cuda_sample_Location_array, max_Location_Size * num_Pop_Ids * sizeof(int));
fst.cu:        //     cudaMalloc((void **)&tmp_3[i], num_Pop_Ids * sizeof(tmp_3[0][0]));
fst.cu:        // cudaMemcpy(cuda_sample_Location_array, tmp_3, max_Location_Size * sizeof(int *), cudaMemcpyHostToDevice);
fst.cu:        //     cudaMemcpy(tmp_3[i], sample_Location_array[i], num_Pop_Ids * sizeof(cuda_sample_Location_array[0][0]), cudaMemcpyHostToDevice);
fst.cu:        int *cuda_VALID_or_NOT;
fst.cu:        cudaMallocManaged(&cuda_VALID_or_NOT, tot_num_segregrating_Sites * sizeof(int));
fst.cu:        int *cuda_VALID_or_NOT_FST, *VALID_or_NOT_FST;
fst.cu:        cudaMallocManaged(&cuda_VALID_or_NOT_FST, num_segregrating_Sites * sizeof(int));
fst.cu:        int *cuda_REF_Count, *cuda_ALT_Count;
fst.cu:        float *Fst_per_Seg, *cuda_Fst_per_Seg;
fst.cu:        float *CUDA_numerator, *CUDA_denominators;
fst.cu:        cudaMallocManaged(&cuda_REF_Count, tot_num_segregrating_Sites * sizeof(int));
fst.cu:        cudaMallocManaged(&cuda_ALT_Count, tot_num_segregrating_Sites * sizeof(int));
fst.cu:        cudaMallocManaged(&cuda_Fst_per_Seg, num_segregrating_Sites * sizeof(float));
fst.cu:        cudaMallocManaged(&CUDA_numerator, num_segregrating_Sites * sizeof(float));
fst.cu:        cudaMallocManaged(&CUDA_denominators, num_segregrating_Sites * sizeof(float));
fst.cu:        // cudaMallocManaged(&cuda_Ht_per_Seg, num_segregrating_Sites * sizeof(float));
fst.cu:        // cudaMallocManaged(&cuda_Hs_per_Seg, num_segregrating_Sites * sizeof(float));
fst.cu:        // cuda_process_segs
fst.cu:        // cuda_process_Segs(int total_Segs, char *sites, int *index, int *pop_Sample_Size_Array, int *seg_Site_pop_ID, int **sample_Location_array, int *VALID_or_NOT, int *REF_Count_all, int *ALT_Count_all)
fst.cu:        cuda_process_Segs<<<tot_Blocks, tot_ThreadsperBlock>>>(tot_num_segregrating_Sites, cuda_full_Char, cuda_site_Index, cuda_locations_Size, cuda_seg_Site_pop_ID, cuda_sample_Location_array, cuda_VALID_or_NOT, cuda_REF_Count, cuda_ALT_Count);
fst.cu:        cudaDeviceSynchronize();
fst.cu:        // cudaMemcpy(VALID_or_NOT, cuda_VALID_or_NOT, tot_num_segregrating_Sites * sizeof(int), cudaMemcpyDeviceToHost);
fst.cu:        // cudaError_t error = cudaGetLastError();
fst.cu:        // if (error != cudaSuccess)
fst.cu:        //     // print the CUDA error message and exit
fst.cu:        //     printf("CUDA error: %s\n", cudaGetErrorString(error));
fst.cu:        //__global__ void cuda_process_FST(int segs_Seperate, int num_Pop_Ids, int *pop_Sample_Size_Array, int *pop_seqeunce_Size_Array, int *VALID_or_NOT_ALL, int *VALID_or_NOT_FST, int *REF_Count_all, int *ALT_Count_all, float *Fst)
fst.cu:        cuda_process_FST<<<tot_Blocks, tot_ThreadsperBlock>>>(num_segregrating_Sites, num_Pop_Ids, cuda_locations_Size, cuda_pop_seqeunce_Size_Array, cuda_VALID_or_NOT, cuda_VALID_or_NOT_FST, cuda_REF_Count, cuda_ALT_Count, cuda_Fst_per_Seg,
fst.cu:                                                              CUDA_numerator, CUDA_denominators);
fst.cu:        cudaDeviceSynchronize();
fst.cu:        cudaMemcpy(VALID_or_NOT_FST, cuda_VALID_or_NOT_FST, num_segregrating_Sites * sizeof(int), cudaMemcpyDeviceToHost);
fst.cu:        // cudaMemcpy(Ht_per_Seg, cuda_Ht_per_Seg, num_segregrating_Sites * sizeof(float), cudaMemcpyDeviceToHost);
fst.cu:        // cudaMemcpy(Hs_per_Seg, cuda_Hs_per_Seg, num_segregrating_Sites * sizeof(float), cudaMemcpyDeviceToHost);
fst.cu:        cudaMemcpy(Fst_per_Seg, cuda_Fst_per_Seg, num_segregrating_Sites * sizeof(float), cudaMemcpyDeviceToHost);
fst.cu:        cudaMemcpy(numerator, CUDA_numerator, num_segregrating_Sites * sizeof(float), cudaMemcpyDeviceToHost);
fst.cu:        cudaMemcpy(denominators, CUDA_denominators, num_segregrating_Sites * sizeof(float), cudaMemcpyDeviceToHost);
fst.cu:        cudaFree(cuda_VALID_or_NOT_FST);
fst.cu:        cudaFree(cuda_VALID_or_NOT);
fst.cu:        cudaFree(cuda_Fst_per_Seg);
fst.cu:        cudaFree(cuda_REF_Count);
fst.cu:        cudaFree(cuda_ALT_Count);
fst.cu:        cudaFree(cuda_full_Char);
fst.cu:        cudaFree(cuda_site_Index);
fst.cu:        cudaFree(cuda_seg_Site_pop_ID);
fst.cu:        cudaFree(CUDA_numerator);
fst.cu:        cudaFree(CUDA_denominators);
fst.cu:        cudaFree(cuda_PRESENT_or_NOT);
fst.cu:        cudaFree(cuda_pop_Seg_size_Array);
fst.cu:        cudaFree(cuda_seg_Positions);
fst.cu:        cudaFree(cuda_first_match_Relationships);
test.cu:    cout << "Are we in GPU testing" << endl;
test.cu:    cudaGetDeviceCount(&nDevices);
test.cu:        cudaDeviceProp prop;
test.cu:        cudaGetDeviceProperties(&prop, i);
test.cu:        cout << "GPU number\t: " << i << endl;
test.cu:        cout << "GPU name\t: " << prop.name << endl;
test.cu:        cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);
test.cu:        cout << "GPU memory (GB)\t: " << l_Total / (1000 * 1000 * 1000) << endl;
test.cu:        cout << "GPU number of multiprocessor(s)\t: " << prop.multiProcessorCount << endl;
test.cu:        cout << "GPU number of blocks per multiprocessor\t: " << prop.maxBlocksPerMultiProcessor << endl;
test.cu:        cout << "GPU threads per block\t: " << prop.maxThreadsPerBlock << endl;
test.cu:        cout << "GPU thread(s) per multiProcessorCount\t: " << prop.maxThreadsPerMultiProcessor << endl
test.cu:    // cudaGetDevice(&device);
test.cu:__global__ void cuda_hello(int n, float *x, float *y)
test.cu:__global__ void cuda_hello_2(const float *a, float *out, int arraySize)
test.cu:    float *x_Cuda, *y_Cuda;
test.cu:    cudaMalloc((void **)&x_Cuda, N * sizeof(float));
test.cu:    cudaMemcpy(x_Cuda, array, N * sizeof(float), cudaMemcpyHostToDevice);
test.cu:    cudaMalloc((void **)&y_Cuda, sizeof(float));
test.cu:    cuda_hello_2<<<1, 1024>>>(x_Cuda, y_Cuda, N);
test.cu:    cudaDeviceSynchronize();
test.cu:    cudaMemcpy(y_partial, y_Cuda, sizeof(float), cudaMemcpyDeviceToHost);
test.cu:    cudaFree(x_Cuda);
test.cu:    cudaFree(y_Cuda);
test.cu:    cudaMallocManaged(&x, N * sizeof(float));
test.cu:    cudaMallocManaged(&y, N * sizeof(float));
test.cu:    cuda_hello<<<1, 1024>>>(N, x, y);
test.cu:    cudaDeviceSynchronize();
test.cu:    cudaFree(x);
test.cu:    cudaFree(y);
fst_test_pop.cuh:#include "cuda_runtime.h"
test.h:#include "cuda_runtime.h"
test.h:    void cuda_hello();
mutations_T_json.cuh:#include "cuda_runtime.h"
mutations_T_json.cuh:#include <thrust/system/cuda/error.h>
print_param.cpp:              output << "    # Cuda device\n"
print_param.cpp:                     << "    \"CUDA Device ID\":0,\n\n";
print_param.cpp:                     << "    \"Split SNPs per_time_GPU\":100000\n"
fu_li.cuh:#include "cuda_runtime.h"
fu_li.cuh:     * @param tot_Blocks defines number of GPU blocks that are available
fu_li.cuh:     * @param tot_ThreadsperBlock defines number of threads that are available per GPU block.
fu_li.cuh:     * @param SNPs_per_Run defines the number of SNP sites the GPU will process at a time.
fu_li.cuh:    fu_li(string gene_List, string input_Folder, string ouput_Path, int cuda_ID, string intermediate_Path, int ploidy);
fu_li.cuh:    fu_li(string calc_Mode, int window_Size, int step_Size, string input_Folder, string ouput_Path, int cuda_ID, int ploidy);
fu_li.cuh:    fu_li(string gene_List, string input_Folder, string ouput_Path, int cuda_ID, string intermediate_Path, int ploidy, string prometheus_Activate, string Multi_read, int number_of_genes, int CPU_cores, int SNPs_per_Run);
fu_li.cuh:    fu_li(string calc_Mode, int window_Size, int step_Size, string input_Folder, string ouput_Path, int cuda_ID, int ploidy, string prometheus_Activate, string Multi_read, int number_of_genes, int CPU_cores, int SNPs_per_Run);
fu_li.cuh:    void set_Values(string gene_List, string input_Folder, string ouput_Path, int cuda_ID, string intermediate_Path, int ploidy);
functions.cuh:#include "cuda_runtime.h"
functions.cuh:     * Function directly calls upon the GPU function.
functions.cuh:     * Function directly calls upon the GPU function.
tajima.cuh:#include "cuda_runtime.h"
tajima.cuh:     * @param tot_Blocks defines number of GPU blocks that are available
tajima.cuh:     * @param tot_ThreadsperBlock defines number of threads that are available per GPU block.
tajima.cuh:     * @param SNPs_per_Run defines the number of SNP sites the GPU will process at a time.
tajima.cuh:    tajima(string gene_List, string input_Folder, string ouput_Path, int cuda_ID, string intermediate_Path, int ploidy);
tajima.cuh:    tajima(string calc_Mode, int window_Size, int step_Size, string input_Folder, string ouput_Path, int cuda_ID, int ploidy);
tajima.cuh:    tajima(string gene_List, string input_Folder, string ouput_Path, int cuda_ID, string intermediate_Path, int ploidy, string prometheus_Activate, string Multi_read, int number_of_genes, int CPU_cores, int SNPs_per_Run);
tajima.cuh:    tajima(string calc_Mode, int window_Size, int step_Size, string input_Folder, string ouput_Path, int cuda_ID, int ploidy, string prometheus_Activate, string Multi_read, int number_of_genes, int CPU_cores, int SNPs_per_Run);
tajima.cuh:    void set_Values(string gene_List, string input_Folder, string ouput_Path, int cuda_ID, string intermediate_Path, int ploidy);
bfs.cuh:#include "cuda_runtime.h"
bfs.cuh:#include <thrust/system/cuda/error.h>
functions.cu:__global__ void cuda_process_Seg_tajima(char *sites, int *index, int tot_Segregrating_sites, int *VALID_or_NOT, int *MA_count)
functions.cu:     * 1. Conversion of SNP strings into char pointers for GPU accessability.
functions.cu:     * 2. Call GPU for extracting MAs (Minor allele) and MAF's (Minor Allele Frequencies).
functions.cu:     * This track is vital for navigating through the data in the GPU. For the data is stored in the form of a 1D array.
functions.cu:    // cuda_process_Seg(char *sites, int *index, int tot_Segregrating_sites, int *VALID_or_NOT, int *MA_count)
functions.cu:     * @param cuda_full_Char is used by the GPU. Is a COPY of full_Char.
functions.cu:     * @param cuda_site_Index is used by the GPU. Is a COPY of site_Index.
functions.cu:    char *cuda_full_Char;
functions.cu:    cudaMallocManaged(&cuda_full_Char, (Seg_sites.size() + 1) * sizeof(char));
functions.cu:    int *cuda_site_Index;
functions.cu:    cudaMallocManaged(&cuda_site_Index, (num_segregrating_Sites + 1) * sizeof(int));
functions.cu:     * Transfer of data to the GPU.
functions.cu:    cudaMemcpy(cuda_full_Char, full_Char, (Seg_sites.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
functions.cu:    cudaMemcpy(cuda_site_Index, site_Index, (num_segregrating_Sites + 1) * sizeof(int), cudaMemcpyHostToDevice);
functions.cu:     * @param cuda_VALID_or_NOT is used to determine if a site is seg site which is VALID or NOT.
functions.cu:     * @param VALID_or_NOT is used by the CPU. Is a COPY of cuda_VALID_or_NOT.
functions.cu:    int *cuda_VALID_or_NOT, *VALID_or_NOT;
functions.cu:    cudaMallocManaged(&cuda_VALID_or_NOT, num_segregrating_Sites * sizeof(int));
functions.cu:     * @param cuda_MA_Count is used to record the MA's count.
functions.cu:     * @param MA_Count is used by the CPU. Is a COPY of cuda_MA_Count.
functions.cu:    int *cuda_MA_Count, *MA_Count;
functions.cu:    cudaMallocManaged(&cuda_MA_Count, num_segregrating_Sites * sizeof(int));
functions.cu:     * CALL THE GPU.
functions.cu:     * * GPU WILL PROCESS THE COLLECTED SEG SITES
functions.cu:    cuda_process_Seg_tajima<<<tot_Blocks, tot_ThreadsperBlock>>>(cuda_full_Char, cuda_site_Index, num_segregrating_Sites, cuda_VALID_or_NOT, cuda_MA_Count);
functions.cu:    cudaDeviceSynchronize();
functions.cu:    cudaMemcpy(VALID_or_NOT, cuda_VALID_or_NOT, num_segregrating_Sites * sizeof(int), cudaMemcpyDeviceToHost);
functions.cu:    cudaMemcpy(MA_Count, cuda_MA_Count, num_segregrating_Sites * sizeof(int), cudaMemcpyDeviceToHost);
functions.cu:    cudaFree(cuda_site_Index);
functions.cu:    cudaFree(cuda_MA_Count);
functions.cu:    cudaFree(cuda_VALID_or_NOT);
functions.cu:__global__ void cuda_process_Seg_fu_li(char *sites, int *index, int tot_Segregrating_sites, int *VALID_or_NOT, int *MA_count, int *ne, int *ns)
functions.cu:    char *cuda_full_Char;
functions.cu:    cudaMallocManaged(&cuda_full_Char, (Seg_sites.size() + 1) * sizeof(char));
functions.cu:    int *cuda_site_Index;
functions.cu:    cudaMallocManaged(&cuda_site_Index, (num_segregrating_Sites + 1) * sizeof(int));
functions.cu:    cudaMemcpy(cuda_full_Char, full_Char, (Seg_sites.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
functions.cu:    cudaMemcpy(cuda_site_Index, site_Index, (num_segregrating_Sites + 1) * sizeof(int), cudaMemcpyHostToDevice);
functions.cu:    int *cuda_VALID_or_NOT, *VALID_or_NOT;
functions.cu:    cudaMallocManaged(&cuda_VALID_or_NOT, num_segregrating_Sites * sizeof(int));
functions.cu:    int *cuda_MA_Count, *MA_Count;
functions.cu:    cudaMallocManaged(&cuda_MA_Count, num_segregrating_Sites * sizeof(int));
functions.cu:    int *ne_CUDA, *ne, *ns_CUDA, *ns;
functions.cu:    cudaMallocManaged(&ne_CUDA, num_segregrating_Sites * sizeof(int));
functions.cu:    cudaMallocManaged(&ns_CUDA, num_segregrating_Sites * sizeof(int));
functions.cu:    // cuda_process_Seg_fu_li(char *sites, int *index, int tot_Segregrating_sites, int *VALID_or_NOT, int *MA_count, int *ne, int *ns)
functions.cu:    cuda_process_Seg_fu_li<<<tot_Blocks, tot_ThreadsperBlock>>>(cuda_full_Char, cuda_site_Index, num_segregrating_Sites, cuda_VALID_or_NOT, cuda_MA_Count, ne_CUDA, ns_CUDA);
functions.cu:    cudaDeviceSynchronize();
functions.cu:    cudaMemcpy(VALID_or_NOT, cuda_VALID_or_NOT, num_segregrating_Sites * sizeof(int), cudaMemcpyDeviceToHost);
functions.cu:    cudaMemcpy(MA_Count, cuda_MA_Count, num_segregrating_Sites * sizeof(int), cudaMemcpyDeviceToHost);
functions.cu:    cudaMemcpy(ne, ne_CUDA, num_segregrating_Sites * sizeof(int), cudaMemcpyDeviceToHost);
functions.cu:    cudaMemcpy(ns, ns_CUDA, num_segregrating_Sites * sizeof(int), cudaMemcpyDeviceToHost);
functions.cu:    cudaFree(cuda_full_Char);
functions.cu:    cudaFree(cuda_site_Index);
functions.cu:    cudaFree(cuda_MA_Count);
functions.cu:    cudaFree(cuda_VALID_or_NOT);
functions.cu:    cudaFree(ns_CUDA);
functions.cu:    cudaFree(ne_CUDA);
functions.cu:__global__ void cuda_array_ADD(const float *a, float *out, int arraySize)
functions.cu:    float *x_Cuda, *y_Cuda;
functions.cu:    cudaMalloc((void **)&x_Cuda, N * sizeof(float));
functions.cu:    cudaMemcpy(x_Cuda, array, N * sizeof(float), cudaMemcpyHostToDevice);
functions.cu:    cudaMalloc((void **)&y_Cuda, sizeof(float));
functions.cu:    cuda_array_ADD<<<1, 1024>>>(x_Cuda, y_Cuda, N);
functions.cu:    cudaDeviceSynchronize();
functions.cu:    cudaMemcpy(y_partial, y_Cuda, sizeof(float), cudaMemcpyDeviceToHost);
functions.cu:    cudaFree(x_Cuda);
functions.cu:    cudaFree(y_Cuda);
functions.cu:__global__ void pairwise_Cuda_(int N, int *SNP, int *differences)
functions.cu:    int *cuda_line_Data;
functions.cu:    cudaMallocManaged(&cuda_line_Data, N * sizeof(int));
functions.cu:    int *differences, *cuda_Differences;
functions.cu:    cudaMallocManaged(&cuda_Differences, N * sizeof(int));
functions.cu:    cudaMemcpy(cuda_line_Data, line_temp, (N * sizeof(int)), cudaMemcpyHostToDevice);
functions.cu:    pairwise_Cuda_<<<tot_Blocks, tot_ThreadsperBlock>>>(N, cuda_line_Data, cuda_Differences);
functions.cu:    cudaDeviceSynchronize();
functions.cu:    cudaMemcpy(differences, cuda_Differences, N * sizeof(int), cudaMemcpyDeviceToHost);
functions.cu:    cudaFree(cuda_line_Data);
functions.cu:    cudaFree(cuda_Differences);
simulator_Master.cu:        "\"CUDA Device IDs\"",
simulator_Master.cu:        "\"GPU max units\"",
simulator_Master.cu:    // this->CUDA_device_number = Parameters.get_INT(found_Parameters[0]);
simulator_Master.cu:    string cuda_IDs_String = Parameters.get_STRING(found_Parameters[0]);
simulator_Master.cu:    vector<string> cuda_IDs;
simulator_Master.cu:    function.split(cuda_IDs, cuda_IDs_String, ',');
simulator_Master.cu:    if (cuda_IDs.size() > 0)
simulator_Master.cu:        this->num_Cuda_devices = cuda_IDs.size();
simulator_Master.cu:        CUDA_device_IDs = (int *)malloc(sizeof(int) * num_Cuda_devices);
simulator_Master.cu:        tot_Blocks = (int *)malloc(sizeof(int) * num_Cuda_devices);
simulator_Master.cu:        tot_ThreadsperBlock = (int *)malloc(sizeof(int) * num_Cuda_devices);
simulator_Master.cu:        function.print_Cuda_devices(cuda_IDs, this->CUDA_device_IDs, num_Cuda_devices, this->tot_Blocks, this->tot_ThreadsperBlock);
simulator_Master.cu:        cout << "ERROR: THERE HAS TO BE AT LEAST ONE CUDA DEVICE SELECTED\n";
simulator_Master.cu:    // function.print_Cuda_device(this->CUDA_device_number, this->tot_Blocks, this->tot_ThreadsperBlock);
simulator_Master.cu:    this->gpu_Limit = Parameters.get_INT(found_Parameters[2]);
simulator_Master.cu:    cout << "Per round GPU max unit: " << this->gpu_Limit << endl
simulator_Master.cu:    functions_library functions = functions_library(tot_Blocks, tot_ThreadsperBlock, CUDA_device_IDs, num_Cuda_devices, gpu_Limit, CPU_cores);
simulator_Master.cu:                Hosts[infected_Population[host]].run_Generation(functions, this->multi_Read, this->max_Cells_at_a_time, this->gpu_Limit, CUDA_device_IDs, this->num_Cuda_devices, this->genome_Length,
simulator_Master.cu:        // cudaError_t err = cudaGetLastError();
simulator_Master.cu:        // if (err != cudaSuccess)
simulator_Master.cu:        //     printf("CUDA Error: %s\n", cudaGetErrorString(err));
simulator_Master.cu:        int full_Rounds = total_Sequences / this->gpu_Limit;
simulator_Master.cu:        int partial_Rounds = total_Sequences % this->gpu_Limit;
simulator_Master.cu:            int start = full * this->gpu_Limit;
simulator_Master.cu:            int stop = start + this->gpu_Limit;
simulator_Master.cu:            int full_Rounds = collect_Sequences_Tissue[tissue].size() / this->gpu_Limit;
simulator_Master.cu:            int partial_Rounds = collect_Sequences_Tissue[tissue].size() % this->gpu_Limit;
simulator_Master.cu:                int start = full * this->gpu_Limit;
simulator_Master.cu:                int stop = start + this->gpu_Limit;
vcf_splitter_2.cu:vcf_splitter_2::vcf_splitter_2(int cuda_ID, string input_vcf_Folder, string output_Folder, int cores, int SNPs_per_time_CPU, int SNPs_per_time_GPU, int allele_Count_REF, int allele_Count_ALT, int ploidy, int summary_Individuals)
vcf_splitter_2.cu:    this->SNPs_per_time_GPU = SNPs_per_time_GPU;
vcf_splitter_2.cu:    cuda_Set_device(cuda_ID);
vcf_splitter_2.cu:vcf_splitter_2::vcf_splitter_2(int cuda_ID, string input_vcf_Folder, string output_Folder, string population_File, int sampled_ID_col, int pop_ID_column, int cores, int SNPs_per_time_CPU, int SNPs_per_time_GPU, int ploidy, int max_SNPs_per_file, int logic_MAF, double MAF)
vcf_splitter_2.cu:    this->SNPs_per_time_GPU = SNPs_per_time_GPU;
vcf_splitter_2.cu:    cuda_Set_device(cuda_ID);
vcf_splitter_2.cu:void vcf_splitter_2::cuda_Set_device(int cuda_ID)
vcf_splitter_2.cu:    cudaSetDevice(cuda_ID);
vcf_splitter_2.cu:    cout << "Properties of selected CUDA GPU:" << endl;
vcf_splitter_2.cu:    cudaDeviceProp prop;
vcf_splitter_2.cu:    cudaGetDeviceProperties(&prop, cuda_ID);
vcf_splitter_2.cu:    cout << "GPU number\t: " << cuda_ID << endl;
vcf_splitter_2.cu:    cout << "GPU name\t: " << prop.name << endl;
vcf_splitter_2.cu:    cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);
vcf_splitter_2.cu:    cout << "GPU memory (GB)\t: " << l_Total / (1000 * 1000 * 1000) << endl;
vcf_splitter_2.cu:    cout << "GPU number of multiprocessor(s)\t: " << prop.multiProcessorCount << endl;
vcf_splitter_2.cu:    cout << "GPU block(s) per multiprocessor\t: " << prop.maxBlocksPerMultiProcessor << endl;
vcf_splitter_2.cu:    cout << "GPU thread(s) per block\t: " << tot_ThreadsperBlock << endl
vcf_splitter_2.cu:            cudaMallocManaged(&cuda_MAF_count_per_Population, population_Unique_IDs.size() * sizeof(int));
vcf_splitter_2.cu:            cudaMallocManaged(&cuda_sample_ID_population_ID, N * sizeof(int));
vcf_splitter_2.cu:            cudaMemcpy(cuda_MAF_count_per_Population, MAF_count_per_Population, population_Unique_IDs.size() * sizeof(int), cudaMemcpyHostToDevice);
vcf_splitter_2.cu:            cudaMemcpy(cuda_sample_ID_population_ID, sample_ID_population_ID, N * sizeof(int), cudaMemcpyHostToDevice);
vcf_splitter_2.cu:            // cudaError_t err = cudaGetLastError();
vcf_splitter_2.cu:            // if (err != cudaSuccess)
vcf_splitter_2.cu:            //     printf("CUDA Error 1: %s\n", cudaGetErrorString(err));
vcf_splitter_2.cu:            // cudaDeviceSynchronize();
vcf_splitter_2.cu:        cudaFree(cuda_sample_ID_population_ID);
vcf_splitter_2.cu:        cudaFree(cuda_MAF_count_per_Population);
vcf_splitter_2.cu:__global__ void cuda_seg_Pop_process(char *sites, int *index, int num_Segregrating_sites, int ploidy, int tot_N_individuals, int *cuda_CHR_start_Index, int *cuda_CHR_end_Index, int *cuda_pos_start_Index, int *cuda_pos_end_Index, int *cuda_ID_start_Index, int *cuda_ID_end_Index, int *cuda_REF_start_Index, int *cuda_REF_end_Index, int *cuda_ALT_start_Index, int *cuda_ALT_end_Index, int *cuda_six_9_start_Index, int *cuda_six_9_end_Index, int *cuda_VALID_or_NOT, char *seg_Array, int num_pop, int **cuda_REF_populations, int **cuda_ALT_populations, int **cuda_VALID_or_NOT_populations, int *cuda_sample_ID_population_ID, int *cuda_MAF_count_per_Population, int logic_MAF)
vcf_splitter_2.cu:        cuda_CHR_start_Index[tid] = site_Start;
vcf_splitter_2.cu:        cuda_CHR_end_Index[tid] = i - 1;
vcf_splitter_2.cu:        cuda_pos_start_Index[tid] = i;
vcf_splitter_2.cu:        cuda_pos_end_Index[tid] = i - 1;
vcf_splitter_2.cu:        cuda_ID_start_Index[tid] = i;
vcf_splitter_2.cu:        cuda_ID_end_Index[tid] = i - 1;
vcf_splitter_2.cu:        cuda_REF_start_Index[tid] = i;
vcf_splitter_2.cu:        cuda_REF_end_Index[tid] = i - 1;
vcf_splitter_2.cu:        int num_REF = cuda_REF_end_Index[tid] - cuda_REF_start_Index[tid];
vcf_splitter_2.cu:            cuda_ALT_start_Index[tid] = i;
vcf_splitter_2.cu:            cuda_ALT_end_Index[tid] = i - 1;
vcf_splitter_2.cu:            int num_ALT = cuda_ALT_end_Index[tid] - cuda_ALT_start_Index[tid];
vcf_splitter_2.cu:                cuda_six_9_start_Index[tid] = i;
vcf_splitter_2.cu:                cuda_six_9_end_Index[tid] = i - 1;
vcf_splitter_2.cu:                cuda_VALID_or_NOT[tid] = 1;
vcf_splitter_2.cu:                int pop_ID = cuda_sample_ID_population_ID[sample_ID];
vcf_splitter_2.cu:                    cuda_REF_populations[tid][pop] = 0;
vcf_splitter_2.cu:                    cuda_ALT_populations[tid][pop] = 0;
vcf_splitter_2.cu:                    cuda_VALID_or_NOT_populations[tid][pop] = 1;
vcf_splitter_2.cu:                        if (cuda_VALID_or_NOT_populations[tid][pop_ID] == 1)
vcf_splitter_2.cu:                                cuda_ALT_populations[tid][pop_ID] = cuda_ALT_populations[tid][pop_ID] + 1;
vcf_splitter_2.cu:                                cuda_REF_populations[tid][pop_ID] = cuda_REF_populations[tid][pop_ID] + 1;
vcf_splitter_2.cu:                                cuda_VALID_or_NOT_populations[tid][pop_ID] = 0;
vcf_splitter_2.cu:                            pop_ID = cuda_sample_ID_population_ID[sample_ID];
vcf_splitter_2.cu:                    if (cuda_VALID_or_NOT_populations[tid][pop] == 1)
vcf_splitter_2.cu:                        if (cuda_REF_populations[tid][pop] > cuda_ALT_populations[tid][pop])
vcf_splitter_2.cu:                            MA_Count = cuda_ALT_populations[tid][pop];
vcf_splitter_2.cu:                            MA_Count = cuda_REF_populations[tid][pop];
vcf_splitter_2.cu:                            if (MA_Count == cuda_MAF_count_per_Population[pop])
vcf_splitter_2.cu:                                // cuda_VALID_or_NOT_populations[tid][pop] = 1;
vcf_splitter_2.cu:                                cuda_VALID_or_NOT_populations[tid][pop] = 0;
vcf_splitter_2.cu:                            if (MA_Count > cuda_MAF_count_per_Population[pop])
vcf_splitter_2.cu:                                // cuda_VALID_or_NOT_populations[tid][pop] = 1;
vcf_splitter_2.cu:                                cuda_VALID_or_NOT_populations[tid][pop] = 0;
vcf_splitter_2.cu:                            if (MA_Count < cuda_MAF_count_per_Population[pop])
vcf_splitter_2.cu:                                //  cuda_VALID_or_NOT_populations[tid][pop] = 1;
vcf_splitter_2.cu:                                cuda_VALID_or_NOT_populations[tid][pop] = 0;
vcf_splitter_2.cu:                            if (MA_Count >= cuda_MAF_count_per_Population[pop])
vcf_splitter_2.cu:                                //  cuda_VALID_or_NOT_populations[tid][pop] = 1;
vcf_splitter_2.cu:                                cuda_VALID_or_NOT_populations[tid][pop] = 0;
vcf_splitter_2.cu:                            if (MA_Count <= cuda_MAF_count_per_Population[pop])
vcf_splitter_2.cu:                                // cuda_VALID_or_NOT_populations[tid][pop] = 1;
vcf_splitter_2.cu:                                cuda_VALID_or_NOT_populations[tid][pop] = 0;
vcf_splitter_2.cu:                        //     cuda_VALID_or_NOT_populations[tid][pop] = 0;
vcf_splitter_2.cu:                    cuda_VALID_or_NOT[tid] = 0;
vcf_splitter_2.cu:                cuda_VALID_or_NOT[tid] = 0;
vcf_splitter_2.cu:            cuda_VALID_or_NOT[tid] = 0;
vcf_splitter_2.cu:        if (cuda_VALID_or_NOT[tid] == 0)
vcf_splitter_2.cu:     * The GPU is permitted to handle only a certain max number of SNPs at a time.
vcf_splitter_2.cu:     * Therefore the number of rounds of GPU processing and,
vcf_splitter_2.cu:     * @param GPU_rounds_full rounds requiring the max set of SNPs to be processed.
vcf_splitter_2.cu:     * @param GPU_rounds_partial rounds requiring the remaining set of SNPs to be processed.
vcf_splitter_2.cu:    int GPU_rounds_full = tot_Segs_Round / SNPs_per_time_GPU;
vcf_splitter_2.cu:    int GPU_rounds_partial = tot_Segs_Round % SNPs_per_time_GPU;
vcf_splitter_2.cu:    for (int i = 0; i < GPU_rounds_full; i++)
vcf_splitter_2.cu:        int start = i * SNPs_per_time_GPU;
vcf_splitter_2.cu:        int stop = start + SNPs_per_time_GPU;
vcf_splitter_2.cu:    if (GPU_rounds_partial != 0)
vcf_splitter_2.cu:        int start = tot_Segs_Round - GPU_rounds_partial;
vcf_splitter_2.cu:     * Concatenation of SNPs for GPU processing is also done in parallel,
vcf_splitter_2.cu:        char *cuda_full_Char;
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_full_Char, (Seg_sites.size() + 1) * sizeof(char));
vcf_splitter_2.cu:        int *cuda_site_Index;
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_site_Index, (total_Segs + 1) * sizeof(int));
vcf_splitter_2.cu:        cudaMemcpy(cuda_full_Char, full_Char, (Seg_sites.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
vcf_splitter_2.cu:        cudaMemcpy(cuda_site_Index, site_Index, (total_Segs + 1) * sizeof(int), cudaMemcpyHostToDevice);
vcf_splitter_2.cu:        int *chr_start_Index, *chr_end_Index, *cuda_chr_start_Index, *cuda_chr_end_Index;
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_chr_start_Index, total_Segs * sizeof(int));
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_chr_end_Index, total_Segs * sizeof(int));
vcf_splitter_2.cu:        int *pos_start_Index, *pos_end_Index, *cuda_pos_start_Index, *cuda_pos_end_Index;
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_pos_start_Index, total_Segs * sizeof(int));
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_pos_end_Index, total_Segs * sizeof(int));
vcf_splitter_2.cu:        int *ID_start_Index, *ID_end_Index, *cuda_ID_start_Index, *cuda_ID_end_Index;
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_ID_start_Index, total_Segs * sizeof(int));
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_ID_end_Index, total_Segs * sizeof(int));
vcf_splitter_2.cu:        int *REF_start, *REF_stop, *ALT_start, *ALT_stop, *cuda_REF_start, *cuda_REF_stop, *cuda_ALT_start, *cuda_ALT_stop;
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_REF_start, total_Segs * sizeof(int));
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_REF_stop, total_Segs * sizeof(int));
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_ALT_start, total_Segs * sizeof(int));
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_ALT_stop, total_Segs * sizeof(int));
vcf_splitter_2.cu:        int *six_9_start_Index, *six_9_stop_Index, *cuda_six_9_start_Index, *cuda_six_9_stop_Index;
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_six_9_start_Index, total_Segs * sizeof(int));
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_six_9_stop_Index, total_Segs * sizeof(int));
vcf_splitter_2.cu:        int *VALID_or_NOT, *cuda_VALID_or_NOT;
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_VALID_or_NOT, total_Segs * sizeof(int));
vcf_splitter_2.cu:        //  cudaError_t err4 = cudaGetLastError();
vcf_splitter_2.cu:        // if (err4 != cudaSuccess)
vcf_splitter_2.cu:        //     printf("CUDA Error 4: %s\n", cudaGetErrorString(err4));
vcf_splitter_2.cu:        // cudaDeviceSynchronize();
vcf_splitter_2.cu:        int **VALID_or_NOT_populations, **cuda_VALID_or_NOT_populations;
vcf_splitter_2.cu:        int **cuda_REF_populations, **cuda_ALT_populations;
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_VALID_or_NOT_populations, total_Segs * (num_Populations + 1) * sizeof(int));
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_REF_populations, total_Segs * (num_Populations + 1) * sizeof(int));
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_ALT_populations, total_Segs * (num_Populations + 1) * sizeof(int));
vcf_splitter_2.cu:        // cudaError_t err5 = cudaGetLastError();
vcf_splitter_2.cu:        // if (err5 != cudaSuccess)
vcf_splitter_2.cu:        //     printf("CUDA Error 5: %s\n", cudaGetErrorString(err5));
vcf_splitter_2.cu:        // cudaDeviceSynchronize();
vcf_splitter_2.cu:            cudaMalloc((void **)&tmp[i], (num_Populations + 1) * sizeof(tmp[0][0]));
vcf_splitter_2.cu:            cudaMalloc((void **)&tmp_2[i], (num_Populations + 1) * sizeof(tmp_2[0][0]));
vcf_splitter_2.cu:            cudaMalloc((void **)&tmp_3[i], (num_Populations + 1) * sizeof(tmp_3[0][0]));
vcf_splitter_2.cu:        // cudaError_t err6 = cudaGetLastError();
vcf_splitter_2.cu:        // if (err6 != cudaSuccess)
vcf_splitter_2.cu:        //     printf("CUDA Error 6: %s\n", cudaGetErrorString(err6));
vcf_splitter_2.cu:        // cudaDeviceSynchronize();
vcf_splitter_2.cu:        cudaMemcpy(cuda_VALID_or_NOT_populations, tmp, total_Segs * sizeof(int *), cudaMemcpyHostToDevice);
vcf_splitter_2.cu:        cudaMemcpy(cuda_REF_populations, tmp_2, total_Segs * sizeof(int *), cudaMemcpyHostToDevice);
vcf_splitter_2.cu:        cudaMemcpy(cuda_ALT_populations, tmp_3, total_Segs * sizeof(int *), cudaMemcpyHostToDevice);
vcf_splitter_2.cu:        // cudaError_t err2 = cudaGetLastError();
vcf_splitter_2.cu:        // if (err2 != cudaSuccess)
vcf_splitter_2.cu:        //     printf("CUDA Error 2: %s\n", cudaGetErrorString(err2));
vcf_splitter_2.cu:        // cudaDeviceSynchronize();
vcf_splitter_2.cu:        char *Hap_array, *cuda_Hap_array;
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_Hap_array, (((N * ((2 * ploidy) - 1)) * total_Segs) + 1) * sizeof(char));
vcf_splitter_2.cu:        // cuda_seg_Pop_process(char *sites, int *index, int num_Segregrating_sites, int ploidy, int tot_N_individuals, int *cuda_CHR_start_Index, int *cuda_CHR_end_Index, int *cuda_pos_start_Index, int *cuda_pos_end_Index, int *cuda_ID_start_Index, int *cuda_ID_end_Index, int *cuda_REF_start_Index, int *cuda_REF_end_Index, int *cuda_ALT_start_Index, int *cuda_ALT_end_Index, int *cuda_six_9_start_Index, int *cuda_six_9_end_Index, int *cuda_VALID_or_NOT, char *seg_Array, int num_pop, int **cuda_REF_populations, int **cuda_ALT_populations, int **cuda_VALID_or_NOT_populations, int *cuda_sample_ID_population_ID, int *cuda_MAF_count_per_Population, int logic_MAF)
vcf_splitter_2.cu:        cuda_seg_Pop_process<<<tot_Blocks, tot_ThreadsperBlock>>>(cuda_full_Char, cuda_site_Index, total_Segs, ploidy, N, cuda_chr_start_Index, cuda_chr_end_Index, cuda_pos_start_Index, cuda_pos_end_Index, cuda_ID_start_Index, cuda_ID_end_Index, cuda_REF_start, cuda_REF_stop, cuda_ALT_start, cuda_ALT_stop, cuda_six_9_start_Index, cuda_six_9_stop_Index, cuda_VALID_or_NOT, cuda_Hap_array, num_Populations, cuda_REF_populations, cuda_ALT_populations, cuda_VALID_or_NOT_populations, cuda_sample_ID_population_ID, cuda_MAF_count_per_Population, logic_MAF);
vcf_splitter_2.cu:        cudaError_t err3 = cudaGetLastError();
vcf_splitter_2.cu:        if (err3 != cudaSuccess)
vcf_splitter_2.cu:            printf("CUDA Error: %s\n", cudaGetErrorString(err3));
vcf_splitter_2.cu:        cudaDeviceSynchronize();
vcf_splitter_2.cu:        cudaMemcpy(VALID_or_NOT, cuda_VALID_or_NOT, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(chr_start_Index, cuda_chr_start_Index, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(chr_end_Index, cuda_chr_end_Index, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(pos_start_Index, cuda_pos_start_Index, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(pos_end_Index, cuda_pos_end_Index, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(ID_start_Index, cuda_ID_start_Index, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(ID_end_Index, cuda_ID_end_Index, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(REF_start, cuda_REF_start, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(REF_stop, cuda_REF_stop, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(ALT_start, cuda_ALT_start, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(ALT_stop, cuda_ALT_stop, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(six_9_start_Index, cuda_six_9_start_Index, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(six_9_stop_Index, cuda_six_9_stop_Index, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(Hap_array, cuda_Hap_array, (((N * ((2 * ploidy) - 1)) * total_Segs) + 1) * sizeof(char), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:                cudaMemcpy(VALID_or_NOT_populations[i], cuda_VALID_or_NOT_populations[i], (num_Populations + 1) * sizeof(cuda_VALID_or_NOT_populations[0][0]), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaFree(cuda_full_Char);
vcf_splitter_2.cu:        cudaFree(cuda_site_Index);
vcf_splitter_2.cu:        cudaFree(cuda_VALID_or_NOT_populations);
vcf_splitter_2.cu:        cudaFree(cuda_chr_start_Index);
vcf_splitter_2.cu:        cudaFree(cuda_chr_end_Index);
vcf_splitter_2.cu:        cudaFree(cuda_pos_start_Index);
vcf_splitter_2.cu:        cudaFree(cuda_pos_end_Index);
vcf_splitter_2.cu:        cudaFree(cuda_ID_start_Index);
vcf_splitter_2.cu:        cudaFree(cuda_ID_end_Index);
vcf_splitter_2.cu:        cudaFree(cuda_REF_start);
vcf_splitter_2.cu:        cudaFree(cuda_REF_stop);
vcf_splitter_2.cu:        cudaFree(cuda_ALT_start);
vcf_splitter_2.cu:        cudaFree(cuda_ALT_stop);
vcf_splitter_2.cu:        cudaFree(cuda_six_9_start_Index);
vcf_splitter_2.cu:        cudaFree(cuda_six_9_stop_Index);
vcf_splitter_2.cu:        cudaFree(cuda_REF_populations);
vcf_splitter_2.cu:        cudaFree(cuda_ALT_populations);
vcf_splitter_2.cu:        cudaFree(cuda_Hap_array);
vcf_splitter_2.cu:__global__ void cuda_seg_Info_extract(char *sites, int *index, int num_Segregrating_sites, int ploidy, int N_individuals, int *VALID_or_NOT, int REF_count, int ALT_count, int *cuda_CHR_start_Index, int *cuda_CHR_end_Index, int *cuda_pos_start_Index, int *cuda_pos_end_Index, int *cuda_ID_start_Index, int *cuda_ID_end_Index, int *cuda_REF_start_Index, int *cuda_REF_end_Index, int *cuda_ALT_start_Index, int *cuda_ALT_end_Index, int *cuda_six_8_start_Index, int *cuda_six_8_end_Index, int **sample_sequence_Tracker, char *seg_Array)
vcf_splitter_2.cu:        cuda_CHR_start_Index[tid] = site_Start;
vcf_splitter_2.cu:        cuda_CHR_end_Index[tid] = i - 1;
vcf_splitter_2.cu:        cuda_pos_start_Index[tid] = i;
vcf_splitter_2.cu:        cuda_pos_end_Index[tid] = i - 1;
vcf_splitter_2.cu:        cuda_ID_start_Index[tid] = i;
vcf_splitter_2.cu:        cuda_ID_end_Index[tid] = i - 1;
vcf_splitter_2.cu:        cuda_REF_start_Index[tid] = i;
vcf_splitter_2.cu:        cuda_REF_end_Index[tid] = i - 1;
vcf_splitter_2.cu:        int num_REF = cuda_REF_end_Index[tid] - cuda_REF_start_Index[tid];
vcf_splitter_2.cu:            cuda_ALT_start_Index[tid] = i;
vcf_splitter_2.cu:            cuda_ALT_end_Index[tid] = i - 1;
vcf_splitter_2.cu:            int num_ALT = cuda_ALT_end_Index[tid] - cuda_ALT_start_Index[tid];
vcf_splitter_2.cu:                cuda_six_8_start_Index[tid] = i;
vcf_splitter_2.cu:                cuda_six_8_end_Index[tid] = i - 1;
vcf_splitter_2.cu:     * The GPU is permitted to handle only a certain max number of SNPs at a time.
vcf_splitter_2.cu:     * Therefore the number of rounds of GPU processing and,
vcf_splitter_2.cu:     * @param GPU_rounds_full rounds requiring the max set of SNPs to be processed.
vcf_splitter_2.cu:     * @param GPU_rounds_partial rounds requiring the remaining set of SNPs to be processed.
vcf_splitter_2.cu:    int GPU_rounds_full = tot_Segs_Round / SNPs_per_time_GPU;
vcf_splitter_2.cu:    int GPU_rounds_partial = tot_Segs_Round % SNPs_per_time_GPU;
vcf_splitter_2.cu:    for (int i = 0; i < GPU_rounds_full; i++)
vcf_splitter_2.cu:        int start = i * SNPs_per_time_GPU;
vcf_splitter_2.cu:        int stop = start + SNPs_per_time_GPU;
vcf_splitter_2.cu:    if (GPU_rounds_partial != 0)
vcf_splitter_2.cu:        int start = tot_Segs_Round - GPU_rounds_partial;
vcf_splitter_2.cu:     * Concatenation of SNPs for GPU processing is also done in parallel,
vcf_splitter_2.cu:        char *cuda_full_Char;
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_full_Char, (Seg_sites.size() + 1) * sizeof(char));
vcf_splitter_2.cu:        int *cuda_site_Index;
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_site_Index, (total_Segs + 1) * sizeof(int));
vcf_splitter_2.cu:        cudaMemcpy(cuda_full_Char, full_Char, (Seg_sites.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
vcf_splitter_2.cu:        cudaMemcpy(cuda_site_Index, site_Index, (total_Segs + 1) * sizeof(int), cudaMemcpyHostToDevice);
vcf_splitter_2.cu:        int *cuda_VALID_or_NOT, *VALID_or_NOT;
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_VALID_or_NOT, total_Segs * sizeof(int));
vcf_splitter_2.cu:        int *chr_start_Index, *chr_end_Index, *cuda_chr_start_Index, *cuda_chr_end_Index;
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_chr_start_Index, total_Segs * sizeof(int));
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_chr_end_Index, total_Segs * sizeof(int));
vcf_splitter_2.cu:        int *pos_start_Index, *pos_end_Index, *cuda_pos_start_Index, *cuda_pos_end_Index;
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_pos_start_Index, total_Segs * sizeof(int));
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_pos_end_Index, total_Segs * sizeof(int));
vcf_splitter_2.cu:        int *ID_start_Index, *ID_end_Index, *cuda_ID_start_Index, *cuda_ID_end_Index;
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_ID_start_Index, total_Segs * sizeof(int));
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_ID_end_Index, total_Segs * sizeof(int));
vcf_splitter_2.cu:        int *REF_start, *REF_stop, *ALT_start, *ALT_stop, *cuda_REF_start, *cuda_REF_stop, *cuda_ALT_start, *cuda_ALT_stop;
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_REF_start, total_Segs * sizeof(int));
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_REF_stop, total_Segs * sizeof(int));
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_ALT_start, total_Segs * sizeof(int));
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_ALT_stop, total_Segs * sizeof(int));
vcf_splitter_2.cu:        int *six_8_start_Index, *six_8_stop_Index, *cuda_six_8_start_Index, *cuda_six_8_stop_Index;
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_six_8_start_Index, total_Segs * sizeof(int));
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_six_8_stop_Index, total_Segs * sizeof(int));
vcf_splitter_2.cu:        char *Hap_array, *cuda_Hap_array;
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_Hap_array, (((N * ((2 * ploidy) - 1)) * total_Segs) + 1) * sizeof(char));
vcf_splitter_2.cu:        int **cuda_sample_sequence_Tracker;
vcf_splitter_2.cu:        cudaMallocManaged(&cuda_sample_sequence_Tracker, (N + 1) * total_Segs * sizeof(int));
vcf_splitter_2.cu:            cudaMalloc((void **)&tmp[i], (N + 1) * sizeof(tmp[0][0]));
vcf_splitter_2.cu:        cudaMemcpy(cuda_sample_sequence_Tracker, tmp, total_Segs * sizeof(int *), cudaMemcpyHostToDevice);
vcf_splitter_2.cu:        // cuda_seg_Info_extract(char *sites, int *index, int num_Segregrating_sites, int ploidy, int N_individuals, int *VALID_or_NOT, int REF_count, int ALT_count, int *cuda_CHR_start_Index, int *cuda_CHR_end_Index, int *cuda_pos_start_Index, int *cuda_pos_end_Index, int *cuda_ID_start_Index, int *cuda_ID_end_Index, int *cuda_REF_start_Index, int *cuda_REF_end_Index, int *cuda_ALT_start_Index, int *cuda_ALT_end_Index, int *cuda_six_8_start_Index, int *cuda_six_8_end_Index, int **sample_sequence_Tracker, char *seg_Array)
vcf_splitter_2.cu:        cuda_seg_Info_extract<<<tot_Blocks, tot_ThreadsperBlock>>>(cuda_full_Char, cuda_site_Index, total_Segs, ploidy, N, cuda_VALID_or_NOT, allele_Count_REF, allele_Count_ALT, cuda_chr_start_Index, cuda_chr_end_Index, cuda_pos_start_Index, cuda_pos_end_Index, cuda_ID_start_Index, cuda_ID_end_Index, cuda_REF_start, cuda_REF_stop, cuda_ALT_start, cuda_ALT_stop, cuda_six_8_start_Index, cuda_six_8_stop_Index, cuda_sample_sequence_Tracker, cuda_Hap_array);
vcf_splitter_2.cu:        cudaError_t err = cudaGetLastError();
vcf_splitter_2.cu:        if (err != cudaSuccess)
vcf_splitter_2.cu:            printf("CUDA Error: %s\n", cudaGetErrorString(err));
vcf_splitter_2.cu:        cudaDeviceSynchronize();
vcf_splitter_2.cu:        cudaMemcpy(VALID_or_NOT, cuda_VALID_or_NOT, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(chr_start_Index, cuda_chr_start_Index, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(chr_end_Index, cuda_chr_end_Index, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(pos_start_Index, cuda_pos_start_Index, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(pos_end_Index, cuda_pos_end_Index, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(ID_start_Index, cuda_ID_start_Index, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(ID_end_Index, cuda_ID_end_Index, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(REF_start, cuda_REF_start, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(REF_stop, cuda_REF_stop, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(ALT_start, cuda_ALT_start, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(ALT_stop, cuda_ALT_stop, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(six_8_start_Index, cuda_six_8_start_Index, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(six_8_stop_Index, cuda_six_8_stop_Index, total_Segs * sizeof(int), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaMemcpy(Hap_array, cuda_Hap_array, (((N * ((2 * ploidy) - 1)) * total_Segs) + 1) * sizeof(char), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:                cudaMemcpy(sample_sequence_Tracker[i], cuda_sample_sequence_Tracker[i], (N + 1) * sizeof(cuda_sample_sequence_Tracker[0][0]), cudaMemcpyDeviceToHost);
vcf_splitter_2.cu:        cudaFree(cuda_full_Char);
vcf_splitter_2.cu:        cudaFree(cuda_site_Index);
vcf_splitter_2.cu:        cudaFree(cuda_VALID_or_NOT);
vcf_splitter_2.cu:        cudaFree(cuda_chr_start_Index);
vcf_splitter_2.cu:        cudaFree(cuda_chr_end_Index);
vcf_splitter_2.cu:        cudaFree(cuda_pos_start_Index);
vcf_splitter_2.cu:        cudaFree(cuda_pos_end_Index);
vcf_splitter_2.cu:        cudaFree(cuda_ID_start_Index);
vcf_splitter_2.cu:        cudaFree(cuda_ID_end_Index);
vcf_splitter_2.cu:        cudaFree(cuda_REF_start);
vcf_splitter_2.cu:        cudaFree(cuda_REF_stop);
vcf_splitter_2.cu:        cudaFree(cuda_ALT_start);
vcf_splitter_2.cu:        cudaFree(cuda_ALT_stop);
vcf_splitter_2.cu:        cudaFree(cuda_six_8_start_Index);
vcf_splitter_2.cu:        cudaFree(cuda_six_8_stop_Index);
vcf_splitter_2.cu:        cudaFree(cuda_Hap_array);
vcf_splitter_2.cu:        cudaFree(cuda_sample_sequence_Tracker);
vcf_splitter_2.cu:     * Will spawn threads based on the umber of GPU rounds needed.
vcf_splitter_2.cu:     * Will concat the segments for GPU processing per GPU rounds.
tajima.cu:tajima::tajima(string gene_List, string input_Folder, string ouput_Path, int cuda_ID, string intermediate_Path, int ploidy)
tajima.cu:    cout << "Initiating CUDA powered Tajima's D calculator" << endl
tajima.cu:    set_Values(gene_List, input_Folder, ouput_Path, cuda_ID, intermediate_Path, ploidy);
tajima.cu:    // cudaSetDevice(cuda_ID);
tajima.cu:    // cout << "Properties of selected CUDA GPU:" << endl;
tajima.cu:    // cudaDeviceProp prop;
tajima.cu:    // cudaGetDeviceProperties(&prop, cuda_ID);
tajima.cu:    // cout << "GPU number\t: " << cuda_ID << endl;
tajima.cu:    // cout << "GPU name\t: " << prop.name << endl;
tajima.cu:    // cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);
tajima.cu:    // cout << "GPU memory (GB)\t: " << l_Total / (1000 * 1000 * 1000) << endl;
tajima.cu:    // cout << "GPU number of multiprocessor(s)\t: " << prop.multiProcessorCount << endl;
tajima.cu:    // cout << "GPU block(s) per multiprocessor\t: " << prop.maxBlocksPerMultiProcessor << endl;
tajima.cu:    // cout << "GPU thread(s) per block\t: " << tot_ThreadsperBlock << endl
tajima.cu:tajima::tajima(string gene_List, string input_Folder, string ouput_Path, int cuda_ID, string intermediate_Path, int ploidy, string prometheus_Activate, string Multi_read, int number_of_genes, int CPU_cores, int SNPs_per_Run)
tajima.cu:    cout << "Initiating CUDA powered Tajima's D calculator on PROMETHEUS" << endl
tajima.cu:    set_Values(gene_List, input_Folder, ouput_Path, cuda_ID, intermediate_Path, ploidy);
tajima.cu:    // cudaSetDevice(cuda_ID);
tajima.cu:    // cout << "Properties of selected CUDA GPU:" << endl;
tajima.cu:    // cudaDeviceProp prop;
tajima.cu:    // cudaGetDeviceProperties(&prop, cuda_ID);
tajima.cu:    // cout << "GPU number\t: " << cuda_ID << endl;
tajima.cu:    // cout << "GPU name\t: " << prop.name << endl;
tajima.cu:    // cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);
tajima.cu:    // cout << "GPU memory (GB)\t: " << l_Total / (1000 * 1000 * 1000) << endl;
tajima.cu:    // cout << "GPU number of multiprocessor(s)\t: " << prop.multiProcessorCount << endl;
tajima.cu:    // cout << "GPU block(s) per multiprocessor\t: " << prop.maxBlocksPerMultiProcessor << endl;
tajima.cu:    // cout << "GPU thread(s) per block\t: " << tot_ThreadsperBlock << endl
tajima.cu:tajima::tajima(string calc_Mode, int window_Size, int step_Size, string input_Folder, string ouput_Path, int cuda_ID, int ploidy, string prometheus_Activate, string Multi_read, int number_of_genes, int CPU_cores, int SNPs_per_Run)
tajima.cu:    cout << "Initiating CUDA powered Tajima's D calculator on PROMETHEUS" << endl
tajima.cu:    set_Values("", input_Folder, ouput_Path, cuda_ID, "", ploidy);
tajima.cu:tajima::tajima(string calc_Mode, int window_Size, int step_Size, string input_Folder, string ouput_Path, int cuda_ID, int ploidy)
tajima.cu:    cout << "Initiating CUDA powered Tajima's D calculator" << endl
tajima.cu:    set_Values("", input_Folder, ouput_Path, cuda_ID, "", ploidy);
tajima.cu:void tajima::set_Values(string gene_List, string input_Folder, string ouput_Path, int cuda_ID, string intermediate_Path, int ploidy)
tajima.cu:     * Here the first call to the selected CUDA device occurs.
tajima.cu:    cudaSetDevice(cuda_ID);
tajima.cu:    cout << "Properties of selected CUDA GPU:" << endl;
tajima.cu:    cudaDeviceProp prop;
tajima.cu:    cudaGetDeviceProperties(&prop, cuda_ID);
tajima.cu:    cout << "GPU number\t: " << cuda_ID << endl;
tajima.cu:    cout << "GPU name\t: " << prop.name << endl;
tajima.cu:    cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);
tajima.cu:    cout << "GPU memory (GB)\t: " << l_Total / (1000 * 1000 * 1000) << endl;
tajima.cu:    cout << "GPU number of multiprocessor(s)\t: " << prop.multiProcessorCount << endl;
tajima.cu:    cout << "GPU block(s) per multiprocessor\t: " << prop.maxBlocksPerMultiProcessor << endl;
tajima.cu:    cout << "GPU thread(s) per block\t: " << tot_ThreadsperBlock << endl
tajima.cu:                                         * Information from the SNP is extracted via the GPU.
tajima.cu:                                         * Information from the SNP is extracted via the GPU.
tajima.cu:                         * Information from the SNP is extracted via the GPU.
tajima.cu:__global__ void pairwise_Cuda(int N, int *SNP, int *differences)
tajima.cu://     int *cuda_line_Data;
tajima.cu://     cudaMallocManaged(&cuda_line_Data, N * sizeof(int));
tajima.cu://     int *differences, *cuda_Differences;
tajima.cu://     cudaMallocManaged(&cuda_Differences, N * sizeof(int));
tajima.cu://     cudaMemcpy(cuda_line_Data, line_temp, (N * sizeof(int)), cudaMemcpyHostToDevice);
tajima.cu://     pairwise_Cuda<<<tot_Blocks, tot_ThreadsperBlock>>>(N, cuda_line_Data, cuda_Differences);
tajima.cu://     cudaDeviceSynchronize();
tajima.cu://     cudaMemcpy(differences, cuda_Differences, N * sizeof(int), cudaMemcpyDeviceToHost);
tajima.cu://     cudaFree(cuda_line_Data);
tajima.cu://     cudaFree(cuda_Differences);
tajima.cu:__global__ void a_Calculation(int N, float *a1_CUDA, float *a2_CUDA)
tajima.cu:        a1_CUDA[tid] = (float)1 / (tid + 1);
tajima.cu:        a2_CUDA[tid] = (float)1 / ((tid + 1) * (tid + 1));
tajima.cu:__global__ void add_Cuda(const float *a, float *out, int arraySize)
tajima.cu:     * CUDA based reduction add function for large arrays
tajima.cu:    float *a1_CUDA, *a2_CUDA;
tajima.cu:    cudaMallocManaged(&a1_CUDA, N * sizeof(int));
tajima.cu:    cudaMallocManaged(&a2_CUDA, N * sizeof(int));
tajima.cu:    a_Calculation<<<tot_Blocks, tot_ThreadsperBlock>>>(N, a1_CUDA, a2_CUDA);
tajima.cu:    cudaDeviceSynchronize();
tajima.cu:    cudaMemcpy(a1_partial, a1_CUDA, N * sizeof(float), cudaMemcpyDeviceToHost);
tajima.cu:    cudaMemcpy(a2_partial, a2_CUDA, N * sizeof(float), cudaMemcpyDeviceToHost);
tajima.cu:    cudaFree(a1_CUDA);
tajima.cu:    cudaFree(a2_CUDA);
fay_wu.cuh:#include "cuda_runtime.h"
fay_wu.cuh:     * @param tot_Blocks defines number of GPU blocks that are available
fay_wu.cuh:     * @param tot_ThreadsperBlock defines number of threads that are available per GPU block.
fay_wu.cuh:     * @param SNPs_per_Run defines the number of SNP sites the GPU will process at a time.
fay_wu.cuh:    fay_wu(string gene_List, string input_Folder, string ouput_Path, int cuda_ID, string intermediate_Path, int ploidy);
fay_wu.cuh:    fay_wu(string calc_Mode, int window_Size, int step_Size, string input_Folder, string ouput_Path, int cuda_ID, int ploidy);
fay_wu.cuh:    fay_wu(string gene_List, string input_Folder, string ouput_Path, int cuda_ID, string intermediate_Path, int ploidy, string prometheus_Activate, string Multi_read, int number_of_genes, int CPU_cores, int SNPs_per_Run);
fay_wu.cuh:    fay_wu(string calc_Mode, int window_Size, int step_Size, string input_Folder, string ouput_Path, int cuda_ID, int ploidy, string prometheus_Activate, string Multi_read, int number_of_genes, int CPU_cores, int SNPs_per_Run);
fay_wu.cuh:    void set_Values(string gene_List, string input_Folder, string ouput_Path, int cuda_ID, string intermediate_Path, int ploidy);
fay_wu.cuh:     * Function directly calls upon the GPU function.
parameter_load.h:    void get_parameters(int &CUDA_device_ID, string &parent_SEQ_folder,
neutral.cu:neutral::neutral(string gene_List, string input_Folder, string output_Path, int cuda_ID, string intermediate_Path, int ploidy)
neutral.cu:     cout << "Initiating CUDA powered complete neutrality test calculator" << endl
neutral.cu:     set_Values(gene_List, input_Folder, output_Path, cuda_ID, intermediate_Path, ploidy);
neutral.cu:neutral::neutral(string gene_List, string input_Folder, string output_Path, int cuda_ID, string intermediate_Path, int ploidy, string prometheus_Activate, string Multi_read, int number_of_genes, int CPU_cores, int SNPs_per_Run)
neutral.cu:     cout << "Initiating CUDA powered complete neutrality test calculator on PROMETHEUS" << endl
neutral.cu:     set_Values(gene_List, input_Folder, output_Path, cuda_ID, intermediate_Path, ploidy);
neutral.cu:neutral::neutral(string calc_Mode, int window_Size, int step_Size, string input_Folder, string output_Path, int cuda_ID, int ploidy, string prometheus_Activate, string Multi_read, int number_of_genes, int CPU_cores, int SNPs_per_Run)
neutral.cu:     cout << "Initiating CUDA powered complete neutrality test calculator on PROMETHEUS" << endl
neutral.cu:     set_Values("", input_Folder, output_Path, cuda_ID, "", ploidy);
neutral.cu:neutral::neutral(string calc_Mode, int window_Size, int step_Size, string input_Folder, string output_Path, int cuda_ID, int ploidy)
neutral.cu:     cout << "Initiating CUDA powered complete neutrality test calculator" << endl
neutral.cu:     set_Values("", input_Folder, output_Path, cuda_ID, "", ploidy);
neutral.cu:void neutral::set_Values(string gene_List, string input_Folder, string output_Path, int cuda_ID, string intermediate_Path, int ploidy)
neutral.cu:      * Here the first call to the selected CUDA device occurs.
neutral.cu:     cudaSetDevice(cuda_ID);
neutral.cu:     cout << "Properties of selected CUDA GPU:" << endl;
neutral.cu:     cudaDeviceProp prop;
neutral.cu:     cudaGetDeviceProperties(&prop, cuda_ID);
neutral.cu:     cout << "GPU number\t: " << cuda_ID << endl;
neutral.cu:     cout << "GPU name\t: " << prop.name << endl;
neutral.cu:     cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);
neutral.cu:     cout << "GPU memory (GB)\t: " << l_Total / (1000 * 1000 * 1000) << endl;
neutral.cu:     cout << "GPU number of multiprocessor(s)\t: " << prop.multiProcessorCount << endl;
neutral.cu:     cout << "GPU block(s) per multiprocessor\t: " << prop.maxBlocksPerMultiProcessor << endl;
neutral.cu:     cout << "GPU thread(s) per block\t: " << tot_ThreadsperBlock << endl
neutral.cu:                              // CUDA combined function
neutral.cu:                                                   * Information from the SNP is extracted via the GPU.
neutral.cu:                              // CUDA combined function
neutral.cu:          // CUDA combined function
neutral.cu:__global__ void cuda_process_Segs(char *sites, int *index, int num_Segregrating_sites, int *theta_Partials, int *VALID_or_NOT, int *MA_count, int *ne, int *ns)
neutral.cu:     char *cuda_full_Char;
neutral.cu:     cudaMallocManaged(&cuda_full_Char, (Seg_sites.size() + 1) * sizeof(char));
neutral.cu:     int *cuda_site_Index;
neutral.cu:     cudaMallocManaged(&cuda_site_Index, (num_segregrating_Sites + 1) * sizeof(int));
neutral.cu:     int *cuda_VALID_or_NOT, *VALID_or_NOT;
neutral.cu:     cudaMallocManaged(&cuda_VALID_or_NOT, num_segregrating_Sites * sizeof(int));
neutral.cu:     int *cuda_MA_Count, *MA_Count;
neutral.cu:     cudaMallocManaged(&cuda_MA_Count, num_segregrating_Sites * sizeof(int));
neutral.cu:     int *cuda_Theta_partials, *Theta_partials;
neutral.cu:     cudaMallocManaged(&cuda_Theta_partials, num_segregrating_Sites * sizeof(int));
neutral.cu:     int *ne_CUDA, *ne, *ns_CUDA, *ns;
neutral.cu:     cudaMallocManaged(&ne_CUDA, num_segregrating_Sites * sizeof(int));
neutral.cu:     cudaMallocManaged(&ns_CUDA, num_segregrating_Sites * sizeof(int));
neutral.cu:     cudaMemcpy(cuda_full_Char, full_Char, (Seg_sites.size() + 1) * sizeof(char), cudaMemcpyHostToDevice);
neutral.cu:     cudaMemcpy(cuda_site_Index, site_Index, (num_segregrating_Sites + 1) * sizeof(int), cudaMemcpyHostToDevice);
neutral.cu:     // cuda_process_Segs(char *sites, int *index, int num_Segregrating_sites, int *theta_Partials, int *VALID_or_NOT, int *MA_count, int *ne, int *ns)
neutral.cu:     cuda_process_Segs<<<tot_Blocks, tot_ThreadsperBlock>>>(cuda_full_Char, cuda_site_Index, num_segregrating_Sites, cuda_Theta_partials, cuda_VALID_or_NOT, cuda_MA_Count, ne_CUDA, ns_CUDA);
neutral.cu:     cudaDeviceSynchronize();
neutral.cu:     cudaMemcpy(VALID_or_NOT, cuda_VALID_or_NOT, num_segregrating_Sites * sizeof(int), cudaMemcpyDeviceToHost);
neutral.cu:     cudaMemcpy(MA_Count, cuda_MA_Count, num_segregrating_Sites * sizeof(int), cudaMemcpyDeviceToHost);
neutral.cu:     cudaMemcpy(ne, ne_CUDA, num_segregrating_Sites * sizeof(int), cudaMemcpyDeviceToHost);
neutral.cu:     cudaMemcpy(ns, ns_CUDA, num_segregrating_Sites * sizeof(int), cudaMemcpyDeviceToHost);
neutral.cu:     cudaMemcpy(Theta_partials, cuda_Theta_partials, num_segregrating_Sites * sizeof(int), cudaMemcpyDeviceToHost);
neutral.cu:     cudaFree(cuda_full_Char);
neutral.cu:     cudaFree(cuda_site_Index);
neutral.cu:     cudaFree(cuda_MA_Count);
neutral.cu:     cudaFree(cuda_VALID_or_NOT);
neutral.cu:     cudaFree(cuda_Theta_partials);
neutral.cu:     cudaFree(ns_CUDA);
neutral.cu:     cudaFree(ne_CUDA);
neutral.cu:__global__ void cuda_pre_Calculation(int N, float *a1_CUDA, float *a2_CUDA)
neutral.cu:          a1_CUDA[tid] = (float)1 / (tid + 1);
neutral.cu:          a2_CUDA[tid] = (float)1 / ((tid + 1) * (tid + 1));
neutral.cu:     float *a1_CUDA, *a2_CUDA;
neutral.cu:     cudaMallocManaged(&a1_CUDA, N * sizeof(int));
neutral.cu:     cudaMallocManaged(&a2_CUDA, N * sizeof(int));
neutral.cu:     cuda_pre_Calculation<<<tot_Blocks, tot_ThreadsperBlock>>>(N, a1_CUDA, a2_CUDA);
neutral.cu:     cudaDeviceSynchronize();
neutral.cu:     cudaMemcpy(a1_partial, a1_CUDA, N * sizeof(float), cudaMemcpyDeviceToHost);
neutral.cu:     cudaMemcpy(a2_partial, a2_CUDA, N * sizeof(float), cudaMemcpyDeviceToHost);
neutral.cu:     cudaFree(a1_CUDA);
neutral.cu:     cudaFree(a2_CUDA);
fst.cuh:#include "cuda_runtime.h"
fst.cuh:    int *locations_Size, *cuda_locations_Size;
fst.cuh:    int *pop_seqeunce_Size_Array, *cuda_pop_seqeunce_Size_Array;
fst.cuh:    int **sample_Location_array, **cuda_sample_Location_array;
fst.cuh:    fst(string gene_List, string input_Folder, string output_Path, int cuda_ID, string intermediate_Path, int ploidy, string pop_Index_path, string pop_List);
fst.cuh:    fst(string calc_Mode, int window_Size, int step_Size, string input_Folder, string output_Path, int cuda_ID, int ploidy, string pop_Index_path, string pop_List);
fst.cuh:    void set_Values(string gene_List, string input_Folder, string output_Path, int cuda_ID, string intermediate_Path, int ploidy);
hap_extract.cuh:#include "cuda_runtime.h"
hap_extract.cuh:     * @param tot_Blocks defines number of GPU blocks that are available
hap_extract.cuh:     * @param tot_ThreadsperBlock defines number of threads that are available per GPU block.
hap_extract.cuh:     * @param cuda_reference used to capture and store the reference genome sequence in GPU memory.
hap_extract.cuh:    char *cuda_reference;
hap_extract.cuh:    hap_extract(string gene_List, string input_Folder, string output_Path, int cuda_ID, string intermediate_Path, int ploidy, string reference_File, string pop_Out);
hap_extract.cuh:    void set_Values(string gene_List, string input_Folder, string output_Path, int cuda_ID, string intermediate_Path, int ploidy);
hap_extract.cuh:     * Function directly calls upon the GPU function.
Example_data/Script files/neutrality_script.sh:#SBATCH --gres=gpu:a100:1
Example_data/Script files/neutrality_script.sh:module load cuda/11.4
Example_data/Script files/split_script.sh:#SBATCH --gres=gpu:a100:1
Example_data/Script files/split_script.sh:module load cuda/11.4
test.cuh:#include "cuda_runtime.h"

```
