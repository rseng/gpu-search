# https://github.com/VU-BEAM-Lab/GENRE

```console
GENRE_Code/test_GENRE.m:% This function is used to test that the GENRE (GPU Elastic-Net REgression) 
GENRE_Code/test_GENRE.m:% all of the model fits (1 = standardize the predictors on the GPU and 
GENRE_Code/test_GENRE.m:% predictors on the GPU and return the unnormalized model coefficients,
GENRE_Code/test_GENRE.m:% Call the GENRE.m function to perform the model fits on the GPU using
GENRE_Code/test_GENRE.m:% most likely indicates that the GPU processing was not successful)
GENRE_Code/test_GENRE.m:% Call the GENRE.m function to perform the model fits on the GPU using
GENRE_Code/test_GENRE.m:% most likely indicates that the GPU processing was not successful)
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:// Description of GENRE_GPU_double_precision.cu: 
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:// This file contains the MEX-interface that calls the C/CUDA code for 
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:#include "GPU_kernels_double_precision.cu"
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:static int transformed = 0;              // Specifies whether the predictors have been transformed or not if it is selected for the predictors to be transformed on the GPU           
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:static int shared_memory_flag;           // Flag that determines whether to use GPU shared memory or not
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:// Declare the pointers to the GPU device arrays
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:// Define the function that frees allocated memory on the GPU when the MEX-interface is exited
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:// Set the current GPU device
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaSetDevice(DEVICE_ID);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:// Free the GPU device arrays
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaFree(y_d);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaFree(residual_y_d);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaFree(y_std_d);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaFree(num_observations_d);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaFree(observation_thread_stride_d);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaFree(num_predictors_d);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaFree(X_matrix_thread_stride_d);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaFree(X_matrix_d);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaFree(B_d);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaFree(B_thread_stride_d);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaFree(alpha_values_d);            
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaFree(standardized_lambda_values_d);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaFree(tolerance_values_d);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaFree(max_iterations_values_d);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaFree(intercept_flag_d);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaFree(scaling_factors_d);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaFree(mean_X_matrix_d);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaFree(model_fit_flag_d);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:// Reset the GPU device (need this for profiling the MEX-file using the Nvidia Visual Profiler)
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaDeviceReset();
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   // Set the curret GPU device
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaSetDevice(DEVICE_ID);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   double * GPU_params_h;
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   // Obtain the array that contains the GPU parameters
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   GPU_params_h = (double*)mxGetData(prhs[0]);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   transformation_flag = (int)GPU_params_h[0];
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   num_fits = (int)GPU_params_h[1];
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   total_num_y_observations = (int)GPU_params_h[2];
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   total_num_X_matrix_values = (int)GPU_params_h[3];
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   total_num_B_values = (int)GPU_params_h[4];
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   max_num_observations = (int)GPU_params_h[5];
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   shared_memory_flag = (int)GPU_params_h[6];
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   num_threads_per_block = (int)GPU_params_h[7];
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   // Allocate the GPU device arrays
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaMalloc(&y_d, total_num_y_observations * sizeof(double));
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaMalloc(&residual_y_d, total_num_y_observations * sizeof(double));
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaMalloc(&y_std_d, num_fits * sizeof(double));
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaMalloc(&num_observations_d, num_fits * sizeof(double));
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaMalloc(&observation_thread_stride_d, num_fits * sizeof(double));
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaMalloc(&num_predictors_d, num_fits * sizeof(double));
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaMalloc(&X_matrix_thread_stride_d, num_fits * sizeof(double));
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaMalloc(&X_matrix_d, total_num_X_matrix_values * sizeof(double));
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaMalloc(&B_d, total_num_B_values * sizeof(double));
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaMalloc(&B_thread_stride_d, num_fits * sizeof(double));
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaMalloc(&alpha_values_d, num_fits * sizeof(double));
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaMalloc(&standardized_lambda_values_d, num_fits * sizeof(double));
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaMalloc(&tolerance_values_d, num_fits * sizeof(double));
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaMalloc(&max_iterations_values_d, num_fits * sizeof(double));
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaMalloc(&intercept_flag_d, num_fits * sizeof(double));
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaMalloc(&scaling_factors_d, total_num_B_values * sizeof(double));
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaMalloc(&mean_X_matrix_d, total_num_B_values * sizeof(double));
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaMalloc(&model_fit_flag_d, num_fits * sizeof(double));
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   // Transfer the data from the host arrays to the GPU device arrays
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaMemcpy(num_observations_d, num_observations_h, num_fits * sizeof(double), cudaMemcpyHostToDevice);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaMemcpy(observation_thread_stride_d, observation_thread_stride_h, num_fits * sizeof(double), cudaMemcpyHostToDevice);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaMemcpy(num_predictors_d, num_predictors_h, num_fits * sizeof(double), cudaMemcpyHostToDevice);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaMemcpy(X_matrix_thread_stride_d, X_matrix_thread_stride_h, num_fits * sizeof(double), cudaMemcpyHostToDevice);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaMemcpy(X_matrix_d, X_matrix_h, total_num_X_matrix_values * sizeof(double), cudaMemcpyHostToDevice);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaMemcpy(B_thread_stride_d, B_thread_stride_h, num_fits * sizeof(double), cudaMemcpyHostToDevice);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   cudaMemcpy(intercept_flag_d, intercept_flag_h, num_fits * sizeof(double), cudaMemcpyHostToDevice);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:// Set the current GPU device
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaSetDevice(DEVICE_ID);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:// Transfer the data from the host arrays to the GPU device arrays
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaMemcpy(alpha_values_d, alpha_values_h, num_fits * sizeof(double), cudaMemcpyHostToDevice);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaMemcpy(standardized_lambda_values_d, lambda_values_h, num_fits * sizeof(double), cudaMemcpyHostToDevice);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaMemcpy(tolerance_values_d, tolerance_values_h, num_fits * sizeof(double), cudaMemcpyHostToDevice);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaMemcpy(max_iterations_values_d, max_iterations_values_h, num_fits * sizeof(double), cudaMemcpyHostToDevice);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:// Transfer the input data from the host array to the GPU device array
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaMemcpy(y_d, y_h, total_num_y_observations * sizeof(double), cudaMemcpyHostToDevice);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaMemset(B_d, 0, total_num_B_values * sizeof(double));
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaMemset(model_fit_flag_d, 0, num_fits * sizeof(double));
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:// Set num_threads_per_block to num_fits if the total number of model fits is less than the number of model fits per GPU block
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:// Calculate the number of GPU blocks that are required to perform all of the model fits
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:// Calculate the number of model fits that are performed within the last GPU block
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   // Define the grid and block dimensions for the predictor_standardization GPU kernel
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   // Call the predictor_standardization GPU kernel in order to standardize the predictors of the model matrices
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   // Define the grid and block dimensions for the predictor_normalization GPU kernel
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:   // Call the predictor_normalization GPU kernel in order to normalize the predictors of the model matrices
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:// Define the grid and block dimensions for the model_fit_preparation GPU kernel
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:// Call the model_fit_preparation GPU kernel in order to standardize the y data and the lambda values
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:// Define the grid and block dimensions for the model_fit_reconstruction GPU kernel
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:// Call the model_fit GPU kernel in order to fit the models to the y data     
GENRE_Code/GENRE_GPU_Double_Precision_Code/GENRE_GPU_double_precision.cu:cudaMemcpy(B_h, B_d, total_num_B_values * sizeof(double), cudaMemcpyDeviceToHost);
GENRE_Code/GENRE_GPU_Double_Precision_Code/GPU_kernels_double_precision.cu:// Description of GPU_kernels_double_precision.cu: 
GENRE_Code/GENRE_GPU_Double_Precision_Code/GPU_kernels_double_precision.cu:// This file contains the CUDA code that allows for performing the computations
GENRE_Code/GENRE_GPU_Double_Precision_Code/GPU_kernels_double_precision.cu:// for GENRE on a GPU using double precision
GENRE_Code/GENRE_GPU_Double_Precision_Code/GPU_kernels_double_precision.cu:// Define the GPU kernel that performs predictor normalization
GENRE_Code/GENRE_GPU_Double_Precision_Code/GPU_kernels_double_precision.cu:// Define the GPU kernel that performs predictor standardization
GENRE_Code/GENRE_GPU_Double_Precision_Code/GPU_kernels_double_precision.cu:// Define the GPU kernel that calculates the standard deviations for each portion of the y_d array, standardizes the y_d array, and calculates the standardized lambda values
GENRE_Code/GENRE_GPU_Double_Precision_Code/GPU_kernels_double_precision.cu:// Define the GPU kernel that performs least-squares regression with elastic-net regularization using the cyclic coordinate descent optimization algorithm in order to fit the model matrices to the data
GENRE_Code/GENRE_GPU_Double_Precision_Code/GPU_kernels_double_precision.cu:// Define the GPU kernel that performs least-squares regression with elastic-net regularization using the cyclic coordinate descent optimization algorithm in order to fit the model matrices to the data
GENRE_Code/GENRE_GPU_Double_Precision_Code/GPU_kernels_double_precision.cu:// Define the shared memory array that stores the residual values of the model fits within one block (the amount of bytes is declared in the GPU kernel call)
GENRE_Code/GENRE_GPU_Double_Precision_Code/GPU_kernels_double_precision.cu:// Define the GPU kernel that performs predictor coefficient unnormalization
GENRE_Code/GENRE_GPU_Double_Precision_Code/GPU_kernels_double_precision.cu:// Define the GPU kernel that performs predictor coefficient unstandardization
GENRE_Code/GPU_memory_estimator.m:% Description of GPU_memory_estimator.m:
GENRE_Code/GPU_memory_estimator.m:% This script estimates the amount of GPU memory that is required to
GENRE_Code/GPU_memory_estimator.m:% exceed the amount of memory that is available on the GPU
GENRE_Code/GPU_memory_estimator.m:% GPU calculations
GENRE_Code/GPU_memory_estimator.m:% Calculate the size in bytes of each array that is allocated on the GPU
GENRE_Code/GPU_memory_estimator.m:%% Estimate Required GPU Memory %%
GENRE_Code/GPU_memory_estimator.m:% Calculate the estimate of GPU memory that is required to perform the
GENRE_Code/GPU_memory_estimator.m:estimated_GPU_memory_required_GB = (y_d_size + residual_y_d_size + y_std_d_size + num_observations_d_size ...
GENRE_Code/GPU_memory_estimator.m:%% Print Estimate of Required GPU Memory %%
GENRE_Code/GPU_memory_estimator.m:% Print the estimate of GPU memory that is required to perform the model fits
GENRE_Code/GPU_memory_estimator.m:fprintf('Estimated GPU memory required: %f GB\n', estimated_GPU_memory_required_GB);
GENRE_Code/GPU_memory_estimator.m:%% Compare Estimate of Required GPU Memory to Available GPU Memory %%
GENRE_Code/GPU_memory_estimator.m:% This if statement makes sure that the estimate of GPU memory that is
GENRE_Code/GPU_memory_estimator.m:% available memory on the GPU
GENRE_Code/GPU_memory_estimator.m:gpu_properties = gpuDevice;
GENRE_Code/GPU_memory_estimator.m:if estimated_GPU_memory_required_GB > (gpu_properties.AvailableMemory ./ 1E9) 
GENRE_Code/GPU_memory_estimator.m:    error('Estimate of required GPU memory exceeds available GPU memory.');
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:// Description of GENRE_GPU_single_precision.cu: 
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:// This file contains the MEX-interface that calls the C/CUDA code for 
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:#include "GPU_kernels_single_precision.cu"
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:static int transformed = 0;              // Specifies whether the predictors have been transformed or not if it is selected for the predictors to be transformed on the GPU      
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:static int shared_memory_flag;           // Flag that determines whether to use GPU shared memory or not
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:// Declare the pointers to the GPU device arrays
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:// Define the function that frees allocated memory on the GPU when the MEX-interface is exited
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:// Set the current GPU device
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaSetDevice(DEVICE_ID);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:// Free the GPU device arrays
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaFree(y_d);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaFree(residual_y_d);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaFree(y_std_d);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaFree(num_observations_d);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaFree(observation_thread_stride_d);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaFree(num_predictors_d);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaFree(X_matrix_thread_stride_d);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaFree(X_matrix_d);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaFree(B_d);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaFree(B_thread_stride_d);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaFree(alpha_values_d);            
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaFree(standardized_lambda_values_d);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaFree(tolerance_values_d);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaFree(max_iterations_values_d);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaFree(intercept_flag_d);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaFree(scaling_factors_d);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaFree(mean_X_matrix_d);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaFree(model_fit_flag_d);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:// Reset the GPU device (need this for profiling the MEX-file using the Nvidia Visual Profiler)
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaDeviceReset();
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   // Set the current GPU device
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaSetDevice(DEVICE_ID);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   double * GPU_params_h;
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   // Obtain the array that contains the GPU parameters
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   GPU_params_h = (double*)mxGetData(prhs[0]);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   transformation_flag = (int)GPU_params_h[0];
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   num_fits = (int)GPU_params_h[1];
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   total_num_y_observations = (int)GPU_params_h[2];
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   total_num_X_matrix_values = (int)GPU_params_h[3];
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   total_num_B_values = (int)GPU_params_h[4];
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   max_num_observations = (int)GPU_params_h[5];
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   shared_memory_flag = (int)GPU_params_h[6];
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   num_threads_per_block = (int)GPU_params_h[7];
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   // Allocate the GPU device arrays
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaMalloc(&y_d, total_num_y_observations * sizeof(float));
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaMalloc(&residual_y_d, total_num_y_observations * sizeof(float));
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaMalloc(&y_std_d, num_fits * sizeof(float));
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaMalloc(&num_observations_d, num_fits * sizeof(double));
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaMalloc(&observation_thread_stride_d, num_fits * sizeof(double));
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaMalloc(&num_predictors_d, num_fits * sizeof(double));
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaMalloc(&X_matrix_thread_stride_d, num_fits * sizeof(double));
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaMalloc(&X_matrix_d, total_num_X_matrix_values * sizeof(float));
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaMalloc(&B_d, total_num_B_values * sizeof(float));
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaMalloc(&B_thread_stride_d, num_fits * sizeof(double));
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaMalloc(&alpha_values_d, num_fits * sizeof(float));
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaMalloc(&standardized_lambda_values_d, num_fits * sizeof(float));
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaMalloc(&tolerance_values_d, num_fits * sizeof(float));
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaMalloc(&max_iterations_values_d, num_fits * sizeof(float));
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaMalloc(&intercept_flag_d, num_fits * sizeof(float));
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaMalloc(&scaling_factors_d, total_num_B_values * sizeof(float));
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaMalloc(&mean_X_matrix_d, total_num_B_values * sizeof(float));
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaMalloc(&model_fit_flag_d, num_fits * sizeof(float));
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   // Transfer the data from the host arrays to the GPU device arrays
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaMemcpy(num_observations_d, num_observations_h, num_fits * sizeof(double), cudaMemcpyHostToDevice);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaMemcpy(observation_thread_stride_d, observation_thread_stride_h, num_fits * sizeof(double), cudaMemcpyHostToDevice);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaMemcpy(num_predictors_d, num_predictors_h, num_fits * sizeof(double), cudaMemcpyHostToDevice);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaMemcpy(X_matrix_thread_stride_d, X_matrix_thread_stride_h, num_fits * sizeof(double), cudaMemcpyHostToDevice);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaMemcpy(X_matrix_d, X_matrix_h, total_num_X_matrix_values * sizeof(float), cudaMemcpyHostToDevice);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaMemcpy(B_thread_stride_d, B_thread_stride_h, num_fits * sizeof(double), cudaMemcpyHostToDevice);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   cudaMemcpy(intercept_flag_d, intercept_flag_h, num_fits * sizeof(float), cudaMemcpyHostToDevice);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:// Set the current GPU device
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaSetDevice(DEVICE_ID);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:// Transfer the data from the host arrays to the GPU device arrays
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaMemcpy(alpha_values_d, alpha_values_h, num_fits * sizeof(float), cudaMemcpyHostToDevice);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaMemcpy(standardized_lambda_values_d, lambda_values_h, num_fits * sizeof(float), cudaMemcpyHostToDevice);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaMemcpy(tolerance_values_d, tolerance_values_h, num_fits * sizeof(float), cudaMemcpyHostToDevice);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaMemcpy(max_iterations_values_d, max_iterations_values_h, num_fits * sizeof(float), cudaMemcpyHostToDevice);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:// Transfer the input data from the host array to the GPU device array
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaMemcpy(y_d, y_h, total_num_y_observations * sizeof(float), cudaMemcpyHostToDevice);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaMemset(B_d, 0, total_num_B_values * sizeof(float));
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaMemset(model_fit_flag_d, 0, num_fits * sizeof(float));
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:// Set num_threads_per_block to num_fits if the total number of model fits is less than the number of model fits per GPU block
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:// Calculate the number of GPU blocks that are required to perform all of the model fits
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:// Calculate the number of model fits that are performed within the last GPU block
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   // Define the grid and block dimensions for the predictor_standardization GPU kernel
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   // Call the predictor_standardization GPU kernel in order to standardize the predictors of the model matrices
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   // Define the grid and block dimensions for the predictor_normalization GPU kernel
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:   // Call the predictor_normalization GPU kernel in order to normalize the predictors of the model matrices
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:// Define the grid and block dimensions for the model_fit_preparation GPU kernel
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:// Call the model_fit_preparation GPU kernel in order to standardize the y data and the lambda values
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:// Define the grid and block dimensions for the model_fit_reconstruction GPU kernel
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:// Call the model_fit GPU kernel in order to fit the models to the y data     
GENRE_Code/GENRE_GPU_Single_Precision_Code/GENRE_GPU_single_precision.cu:cudaMemcpy(B_h, B_d, total_num_B_values * sizeof(float), cudaMemcpyDeviceToHost);
GENRE_Code/GENRE_GPU_Single_Precision_Code/GPU_kernels_single_precision.cu:// Description of GPU_kernels_single_precision.cu: 
GENRE_Code/GENRE_GPU_Single_Precision_Code/GPU_kernels_single_precision.cu:// This file contains the CUDA code that allows for performing the computations
GENRE_Code/GENRE_GPU_Single_Precision_Code/GPU_kernels_single_precision.cu:// for GENRE on a GPU using single precision
GENRE_Code/GENRE_GPU_Single_Precision_Code/GPU_kernels_single_precision.cu:// Define the GPU kernel that performs predictor normalization
GENRE_Code/GENRE_GPU_Single_Precision_Code/GPU_kernels_single_precision.cu:// Define the GPU kernel that performs predictor standardization
GENRE_Code/GENRE_GPU_Single_Precision_Code/GPU_kernels_single_precision.cu:// Define the GPU kernel that calculates the standard deviations for each portion of the y_d array, standardizes the y_d array, and calculates the standardized lambda values
GENRE_Code/GENRE_GPU_Single_Precision_Code/GPU_kernels_single_precision.cu:// Define the GPU kernel that performs least-squares regression with elastic-net regularization using the cyclic coordinate descent optimization algorithm in order to fit the model matrices to the data
GENRE_Code/GENRE_GPU_Single_Precision_Code/GPU_kernels_single_precision.cu:// Define the GPU kernel that performs least-squares regression with elastic-net regularization using the cyclic coordinate descent optimization algorithm in order to fit the model matrices to the data
GENRE_Code/GENRE_GPU_Single_Precision_Code/GPU_kernels_single_precision.cu:// Define the shared memory array that stores the residual values of the model fits within one block (the amount of bytes is declared in the GPU kernel call)
GENRE_Code/GENRE_GPU_Single_Precision_Code/GPU_kernels_single_precision.cu:// Define the GPU kernel that performs predictor coefficient unnormalization
GENRE_Code/GENRE_GPU_Single_Precision_Code/GPU_kernels_single_precision.cu:// Define the GPU kernel that performs predictor coefficient unstandardization
GENRE_Code/data_organizer.m:% passed to the GPU
GENRE_Code/test_GENRE_shared_memory.m:% This function is used to test that the GENRE (GPU Elastic-Net REgression) 
GENRE_Code/test_GENRE_shared_memory.m:% all of the model fits (1 = standardize the predictors on the GPU and 
GENRE_Code/test_GENRE_shared_memory.m:% predictors on the GPU and return the unnormalized model coefficients,
GENRE_Code/test_GENRE_shared_memory.m:% Call the GENRE.m function to perform the model fits on the GPU using
GENRE_Code/test_GENRE_shared_memory.m:% most likely indicates that the GPU processing was not successful)
GENRE_Code/test_GENRE_shared_memory.m:% Call the GENRE.m function to perform the model fits on the GPU using
GENRE_Code/test_GENRE_shared_memory.m:% most likely indicates that the GPU processing was not successful)
GENRE_Code/GENRE.m:% This is the main function that is used to call the GENRE (GPU Elastic-Net 
GENRE_Code/GENRE.m:% single precision on GPUs, but using single precision has the trade-off of
GENRE_Code/GENRE.m:% the predictors on the GPU and return the unstandardized model 
GENRE_Code/GENRE.m:% coefficients, 2 = normalize the predictors on the GPU and return the 
GENRE_Code/GENRE.m:%% GPU Memory Estimation %%
GENRE_Code/GENRE.m:fprintf('Beginning GPU memory estimation.\n');
GENRE_Code/GENRE.m:GPU_memory_estimator;
GENRE_Code/GENRE.m:fprintf('GPU memory estimation complete.\n');
GENRE_Code/GENRE.m:%% GPU Thread Block Configuration Calculations %%
GENRE_Code/GENRE.m:% Determine whether GPU shared memory can be utilized for the model fits or
GENRE_Code/GENRE.m:% 32,000 bytes of shared memory per CUDA block)
GENRE_Code/GENRE.m:if gpu_properties.MaxShmemPerBlock >= shared_memory_per_block_max
GENRE_Code/GENRE.m:%% Call the C/CUDA Code to Perform the Model Fits %%
GENRE_Code/GENRE.m:% Create the GPU parameters array
GENRE_Code/GENRE.m:GPU_params_h = [transformation_flag, num_fits, total_num_y_observations, ...
GENRE_Code/GENRE.m:    GPU_params_h = double(GPU_params_h);
GENRE_Code/GENRE.m:    fprintf('Beginning GPU processing.\n');
GENRE_Code/GENRE.m:    [B] = GENRE_GPU_single_precision(GPU_params_h, num_observations_h, ...
GENRE_Code/GENRE.m:    fprintf('GPU processing complete.\n');
GENRE_Code/GENRE.m:    GPU_params_h = double(GPU_params_h);
GENRE_Code/GENRE.m:    fprintf('Beginning GPU processing.\n');
GENRE_Code/GENRE.m:    [B] = GENRE_GPU_double_precision(GPU_params_h, num_observations_h, ...
GENRE_Code/GENRE.m:    fprintf('GPU processing complete.\n');
Paper/paper.bib:@article{H2O4GPU_2020,
Paper/paper.bib:	title={H2O4GPU}, 
Paper/paper.bib:	url={https://github.com/h2oai/h2o4gpu}
Paper/paper.md:title: 'GENRE (GPU Elastic-Net REgression): A CUDA-Accelerated Package for Massively Parallel Linear Regression with Elastic-Net Regularization'
Paper/paper.md:  - CUDA
Paper/paper.md:  - GPU computing
Paper/paper.md:GENRE (GPU Elastic-Net REgression) is a package that allows for many instances of linear
Paper/paper.md:regression with elastic-net regularization to be processed in parallel on a GPU by using the C programming 
Paper/paper.md:language and NVIDIA's (NVIDIA Corporation, Santa Clara, CA, USA) Compute Unified Device Architecture (CUDA) parallel
Paper/paper.md:in a serial fashion on a CPU. However, by using GENRE to perform massively parallel processing on a GPU, a significant speedup 
Paper/paper.md:can potentially be achieved. This is due to the fact that modern GPUs consist of thousands of computational cores
Paper/paper.md:CUDA, a MEX-interface is included to allow for this code to be called within the MATLAB (The MathWorks, Inc., Natick, MA, USA) 
Paper/paper.md:another interface if it is desired to call the C/CUDA code in another language, or the C/CUDA code can be utilized 
Paper/paper.md:without an interface. Note that other packages have been developed that can utilize GPUs for linear regression with
Paper/paper.md:elastic-net regularization, such as H2O4GPU [@H2O4GPU_2020]. However, for this application, these packages typically 
Paper/paper.md:focus on performing parallel computations on the GPU for one model fit at a time in order to achieve acceleration when 
Paper/paper.md:GPU. Instead, many model fits on the GPU are executed in parallel, where each model fit is performed by one computational thread.
Paper/paper.md:The core motivation for developing GENRE was that many of the available packages for performing linear regression with elastic-net regularization focus on achieving high performance in terms of computational time or resource consumption for single model fits. However, they often do not address the case in which there is a need to perform many model fits in parallel. For example, the research project that laid the foundation for GENRE involved performing ultrasound image reconstruction using an algorithm called Aperture Domain Model Image REconstruction (ADMIRE) [@byram_jakovljevic_2014; @byram_dei_tierney_dumont_2015; @dei_byram_2017]. This algorithm is computationally expensive due to the fact that in one stage, it requires thousands of instances of linear regression with elastic-net regularization to be performed in order to fit models of ultrasound data. When this algorithm was implemented on a CPU, it typically required an amount of time that was on the scale of minutes to reconstruct one ultrasound image. The primary bottleneck was performing all of the required model fits due to the fact that a custom C implementation of cyclic coordinate descent was used to compute each fit serially. However, a GPU implementation of the algorithm was developed, and this implementation provided a speedup of over two orders of magnitude, which allowed for multiple ultrasound images to be reconstructed per second. For example, on a computer containing dual Intel (Intel Corporation, Santa Clara, CA) Xeon Silver 4114 CPUs @ 2.20 GHz with 10 cores each along with an NVIDIA GeForce GTX 1080 Ti GPU and an NVIDIA GeForce RTX 2080 Ti GPU, the CPU implementation of ADMIRE had an average processing time of 94.326 $\pm$ 0.437 seconds for one frame of ultrasound channel data while the GPU implementation had an average processing time of 0.436 $\pm$ 0.001 seconds. The average processing time was obtained for each case by taking the average of 10 runs for the same dataset, and timing was performed using MATLAB's built-in timing capabilities. The 2080 Ti GPU was used to perform GPU processing, and the number of processing threads was set to 1 for the CPU implementation. The main contributor to this speedup was the fact that the model fits were performed in parallel on the GPU. For this particular case, 152,832 model fits were performed. Note that double precision was used for the CPU implementation while single precision was utilized for the GPU implementation due to the fact there is typically a performance penalty when using double precision on a GPU. Moreover, for the CPU implementation, MATLAB was used, and a MEX-file was used to call the C implementation of cyclic coordinate descent for the model fitting stage. In addition, note that one additional optimization when performing the model fits on the GPU in the case of ADMIRE is that groups of model fits can use the same model matrix, which allows for improved coalesced memory access and GPU memory bandwidth use. This particular optimization is not used by GENRE.
README.md:# GENRE (GPU Elastic-Net REgression): A CUDA-Accelerated Package for Massively Parallel Linear Regression with Elastic-Net Regularization
README.md:```GENRE``` (GPU Elastic-Net REgression) is a CUDA-accelerated package that allows for many instances of linear regression with elastic-net regularization to be performed in parallel on a GPU. The specific objective function that is minimized is shown below.
README.md:The description provided above describes the process of performing one model fit, but ```GENRE``` allows for many of these fits to be performed in parallel on the GPU by using the CUDA parallel programming framework. GPUs have many computational cores, which allows for a large number of threads to execute operations in parallel. In the case of ```GENRE```, each GPU thread handles one model fit. For example, if 100 individual model fits need to be performed, then 100 computational threads will be required. Performing the fits in parallel on a GPU rather than in a sequential fashion on a CPU can potentially provide a significant speedup in terms of computational time (speedup varies depending on the GPU that is utilized).
README.md:* CUDA-capable NVIDIA GPU (code was tested using an NVIDIA GeForce GTX 1080 Ti GPU, an NVIDIA GeForce GTX 2080 Ti GPU, and an NVIDIA GeForce GTX 1660 Ti laptop GPU)
README.md:  * The speedup that is obtained using ```GENRE``` can vary depending on the GPU that is used.
README.md:  * Note that a MEX-interface is only being used to allow for the C/CUDA code to be called within MATLAB for convenience. With modification, a different interface can be utilized to allow for the C/CUDA code to be called from within another programming language, or the C/CUDA code can be utilized without an interface.
README.md:* Parallel Computing Toolbox for MATLAB in order to allow for the compilation of MEX-files containing CUDA code
README.md:* CUDA toolkit that is compatible with the release of MATLAB (compatibility can be found at https://www.mathworks.com/help/parallel-computing/gpu-support-by-release.html)
README.md:  * Once the compatibility is determined, go to https://developer.nvidia.com/cuda-toolkit-archive and install the particular CUDA toolkit version. Note that the installation process for the toolkit will also allow for the option to install a new graphics driver. If you do not desire to install a new driver, then you must ensure that your current driver supports the toolkit version that is being installed. For driver and toolkit compatability, refer to page 4 of https://docs.nvidia.com/pdf/CUDA_Compatibility.pdf.
README.md:### MATLAB GPU Check
README.md:* Before compiling the files that contain the C/CUDA code into MEX-files, you should first check to see that MATLAB recognizes your GPU card. To do so, go to the command prompt and type ```gpuDevice```. If successful, the properties of the GPU will be displayed. If an error is returned, then possible causes will most likely be related to the graphics driver or the toolkit version that is installed.
README.md:  * The ```-v``` flag can also be included as an argument to each mexcuda command to display compilation details. When included as an argument, it should be wrapped with single quotes like the other arguments. If the compilation process is successful, then it will display a success message for each compilation in the command prompt. In addition, a compiled MEX-file will appear in each folder. The compilation process is important, and it is recommended to recompile any time a different release of MATLAB is utilized.
README.md:cd GENRE_GPU_Single_Precision_Code
README.md:mexcuda('GENRE_GPU_single_precision.cu', 'NVCC_FLAGS=-Xptxas -dlcm=ca')
README.md:cd ..\GENRE_GPU_Double_Precision_Code
README.md:mexcuda('GENRE_GPU_double_precision.cu', 'NVCC_FLAGS=-Xptxas -dlcm=ca')
README.md:  * These commands are similar to the commands that are used for code compilation for Windows OS, but the path to the CUDA toolkit library must also be included. Note that mexcuda might find the CUDA toolkit library even if you do not explicitly type out its path. In addition, note that there might be differences in your path compared to the one shown above, such as in regards to the version of the CUDA toolkit that is being used. The ```-v``` flag can also be included as an argument to each mexcuda command to display compilation details. When included as an argument, it should be wrapped with single quotes like the other arguments. If the compilation process is successful, then it will display a success message for each compilation in the command prompt. In addition, a compiled MEX-file will appear in each folder. The compilation process is important, and it is recommended to recompile any time a different release of MATLAB is utilized.
README.md:cd GENRE_GPU_Single_Precision_Code
README.md:mexcuda('GENRE_GPU_single_precision.cu', '-L/usr/local/cuda-10.0/lib64', 'NVCC_FLAGS=-Xptxas -dlcm=ca')
README.md:cd ../GENRE_GPU_Double_Precision_Code
README.md:mexcuda('GENRE_GPU_double_precision.cu', '-L/usr/local/cuda-10.0/lib64', 'NVCC_FLAGS=-Xptxas -dlcm=ca')
README.md:As previously stated, ```GENRE``` allows for many models to run in parallel on the GPU. The data for each model fit needs to be saved as a ```.mat``` file. For example, if there are 100 model fits that need to be performed, then there should be 100 ```.mat``` files. Each file should contain the following 3 variables. 
README.md:All of the ```.mat``` files should be saved in a directory. In terms of the naming convention of the files, the code assumes that the file for the first model fit is called ```model_data_1.mat```, the second file is called ```model_data_2.mat```, and so on. However, if desired, this naming convention can be changed by modifying the way the ```filename``` variable is defined in the ```data_organizer.m``` script. Note that ```GENRE``` allows for either single precision or double precision to be utilized for the model fit calculations. However, the input data for each model fit can be saved as either ```single``` or ```double``` data type. For example, if the variables in the files are saved as ```double``` data type, the model fits can still be performed using either single precision or double precision because ```GENRE``` converts the input data to the precision that is selected for the model fit calculations before it is passed to the GPU. The model coefficients that ```GENRE``` returns are converted to the same data type as the original input data. This means that if the data in the model files is saved as ```double``` data type and single precision is selected for the GPU calculations, then the returned model coefficients will be converted to ```double``` data type.
README.md:* ```precision```: Specifies which numerical precision to use for the model fit calculations on the GPU. The two options are either ```precision = 'single'``` or ```precision = 'double'```. Using double precision instead of single precision on GPUs typically results in a performance penalty due to there being fewer FP64 units than FP32 units and double precision requiring more memory resources as a result of one value of type ```double``` being 64 bits versus one value of type ```single``` being 32 bits. However, using single precision has the trade-off of reduced numerical precision. Depending on factors such as the conditioning of the model matrices, this reduced precision can lead to significantly different results. Therefore, if you select single precision, then you should ensure that this precision is sufficient for your application. If you are uncertain, then it is recommended to use double precision.
README.md:* ```transformation_flag```: A flag that specifies which transformation option to use for the model fits. Note that all of the model fits need to use the same option for this flag, which is why it is not a vector. In addition, note that all of these transformation options are only applied to the model matrices. When the model fits are performed on the GPU, the data corresponding to ```y``` for each model fit is divided by its standard deviation using the 1/N variance formula regardless of which transformation option is selected, and the input ![lambda](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Clambda) value for each model fit is also divided by this standard deviation value. However, before they are returned, the coefficients for each model fit are multiplied by these standard deviation values, where there is one standard deviation value for each fit. This means that the coefficients should reflect the scale of the data corresponding to ```y``` before it was divided by its standard deviation. This is similar to the 'gaussian' fit family in ```glmnet```. 
README.md:  * ```transformation_flag = 1``` means that each predictor column in the model matrix for each model fit will be standardized on the GPU. The mean of each predictor column is subtracted off from each observation in the column, and each observation in the column is then divided by the standard deviation of the column. Note that the 1/N variance formula is used when calculating the standard deviation similar to the ```glmnet``` software package. Once the model fits are performed, the coefficients will be unstandardized before they are returned due to the fact that the original model matrices were unstandardized. A column vector of ones corresponding to an intercept term must be included in every model matrix in order to select this option. Note that this requirement is only for this option and does not apply to the other options. The intercept term for each fit is not standardized. 
README.md:  * ```transformation_flag = 2``` means that each predictor column in the model matrix for each model fit will be normalized on the GPU. Each observation in each predictor column will be divided by a scaling factor. The scaling factor is computed by squaring each observation in the predictor column, summing the squared observations, and taking the square root of the sum. Once the model fits are performed, the coefficients will be unnormalized before they are returned due to the fact that the original model matrices were unnormalized. If an intercept term is included for a particular model fit, it is not normalized. 
README.md:Once the user-defined parameters are specified, the ```GENRE.m``` function can be called within MATLAB. In terms of the processing pipeline, the ```data_organizer.m``` script will be called within this function. This script loops through all of the model data files and organizes the data before it is passed to the GPU. For example, a 1-D array called ```X_matrix_h``` is created that contains the model matrices across all of the model fits in column-major order. As an illustration, if 2 model fits need to be performed and one model matrix is 100 x 1,000 while the other model matrix is 200 x 2,000, then the 1-D array will contain 500,000 elements. The first 100,000 elements will correspond to ```X``` for the first model fit in column-major order, and the remaining 400,000 elements will correspond to ```X``` for the second model fit in column-major order. In addition, a 1-D array called ```y_h``` is also created that contains the sets of observations to which the model matrices are fit. Using the same example just mentioned, the 1-D array will contain 300 elements. The first 100 elements will correspond to ```y``` for the first model fit, and the remaining 200 elements will correspond to ```y``` for the second model fit. Moreover, additional arrays must be created that contain the number of observations for each model fit, the number of predictors for each model fit, and the zero-based indices for where the data for each model fit begins. For example, each model fit is performed by one computational thread on a GPU, so the these arrays are used to ensure that each thread is accessing the elements in the arrays that correspond to the data for its specific model fit.
README.md:After the data is organized, the ```GPU_memory_estimator.m``` script will be called in order to estimate the amount of GPU memory that is required to perform the model fits. A check within the script is performed to ensure that the estimate of required memory does not exceed the amount of memory that is available on the GPU. Once this memory check is performed, the ```GENRE.m``` function will then call either the ```GENRE_GPU_single_precision``` MEX-file or the ```GENRE_GPU_double_precision``` MEX-file depending on which option is selected for ```precision```. These two files contain the C/CUDA code that allows for the model fits to be performed in parallel on the GPU. The output of both of these functions is ```B```, which is a 1-D array that contains the computed model coefficients across all of the model fits. The model coefficients for each model fit are then stored into ```B_cell``` so that each entry in the cell contains the model coefficients for one model fit. ```B_cell``` is saved to a ```.mat``` file along with ```precision```, ```alpha_values_h```, ```lambda_values_h```, ```tolerance_values_h```, ```max_iterations_values_h```, and ```transformation_flag```. The name of the file and the directory to which the file is saved are specified as user-defined parameters. In addition, ```B_cell``` is assigned as an output for the ```GENRE.m``` function.
README.md:% decreased if you do not have enough RAM or GPU VRAM for this many model 
README.md:% single precision on GPUs, but using single precision has the trade-off of
README.md:% all of the model fits (1 = standardize the predictors on the GPU and 
README.md:% predictors on the GPU and return the unnormalized model coefficients,
README.md:% Call the GENRE.m function to perform the model fits on the GPU
README.md:Once you are finished typing the lines of code above, run the ```run_GENRE.m``` script. This will perform the model fits on the GPU, and it will save out the parameters and the computed model coefficients for the model fits to the specified directory. The variable containing the coefficients that is saved to the file is ```B_cell```, and it should also be available within the MATLAB workspace. Each entry in this cell contains the computed model coefficients for a specific fit. For example, to view the coefficients for the first model fit, type the following command within the MATLAB command prompt.
README.md:Note that since we included an intercept term in every model, the first model coefficient is the value of the intercept term. In addition, also note that ```transformation_flag = 1``` for this tutorial, which means that unstandardized model matrices were transferred to the GPU, where they were then standardized. As a result, the coefficients that were returned represent the unstandardized coefficients. To obtain standardized coefficients, you would need to standardize all of your model matrices before saving them in the model data files. Then, you would need to set ```transformation_flag = 3``` in the ```run_GENRE.m``` script to indicate that the input model matrices are already standardized.
README.md:1. As previously stated, ```y``` for each model fit is always standardized on the GPU by dividing it by its standard deviation using the 1/N
README.md:3. The ```GENRE.m``` function calls either the ```GENRE_GPU_single_precision``` MEX-file or the ```GENRE_GPU_double_precision``` MEX-file
README.md:   first call, all of the input arrays generated by the ```data_organizer.m``` script are transferred to the GPU. For example the model
README.md:   they are already on the GPU. An example of when this might be relevant is if the ```GENRE.m``` function is modified to call either of 
README.md:   standardization or normalization is applied to the model matrices on the GPU, then this is only done the first time either MEX-file is
README.md:   MEX-functions from memory and to free the memory allocated on the GPU. The ```GENRE.m``` function, as is, does not call the MEX-files in a 
README.md:   ![lambda](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Clambda) values will be computed in parallel on the GPU. In addition, another way of calculating the coefficients for 
README.md:   modified to call either the ```GENRE_GPU_single_precision``` MEX-file or the ```GENRE_GPU_double_precision``` MEX-file in a for loop. In each iteration of the for loop, the
README.md:   GPU at once, one benefit is that it will require less memory because multiple copies of the model will not be required.
README.md:5. When possible, ```GENRE``` uses shared memory on the GPU in addition to global memory when performing the model fits. This memory has lower latency than global memory, so it can 
README.md:   shared memory available per GPU block and if for the model matrix with the largest number of observations, the number of observations is less than or equal to 250 observations for 
README.md:```GENRE``` has the potential to provide significant speedup due to the fact that many model fits can be performed in parallel on a GPU. Therefore, an example benchmark was performed where we compared ```GENRE``` with ```glmnet```, which is written in Fortran and performs the model fits in a serial fashion on a CPU. In this benchmark, 20,000 model matrices were randomly generated within MATLAB. Each model matrix consisted of 50 observations and 200 predictors (50x200), and an intercept term was included for all of the models. Note that to add an intercept term in ```GENRE```, a column of ones was appended at the beginning of each model matrix to make the predictor dimension 201 (adding a column of ones is not required for ```glmnet```). For each model matrix, the model coefficients were randomly generated, and the matrix multiplication of the model matrix and the coefficients was performed to obtain the observation vector. Therefore, this provided 20,000 observation vectors with each containing 50 observations. Once the data was generated, both ```GENRE``` and ```glmnet``` were used to perform the model fits and return the computed model coefficients. An ![alpha](https://latex.codecogs.com/svg.latex?%5Calpha) value of 0.5 and a ![lambda](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Clambda) value of 0.001 were used for all of the model fits. The tolerance convergence criterion for both packages was set to 1E-4. It was also specified for each package to standardize the model matrices, which means that the unstandardized model coefficients were returned. Note that the column of ones for each model matrix corresponding to the intercept term is not standardized in the case of ```GENRE```. ```GENRE``` allows for the user to select either single precision or double precision for performing the model fits on the GPU, so processing was done for both cases. The MATLAB version of the ```glmnet``` software package includes a compiled executable MEX-file that allows for Fortran code to be called, and it uses double precision for the calculations. In addition, due to the fact that all of the model matrices have a small number of observations (50) in this case, ```GENRE``` is also able to use shared memory in addition to global memory when performing the model fits. Shared memory has lower latency than global memory, so utilizing it can provide performance benefits. Therefore, processing was performed both with and without using shared memory. 
README.md:The computer that was used for the benchmarks contained dual Intel Xeon Silver 4114 CPUs @ 2.20 GHz with 10 cores each along with an NVIDIA GeForce GTX 1080 Ti GPU and an NVIDIA GeForce RTX 2080 Ti GPU. The 2080 Ti GPU was used to perform GPU processing. For each case, the average of 10 runs was taken, and timing was performed using MATLAB's built-in timing capabilities. Note that ```GENRE``` has a data organization step that loads the data for the model fits from files and organizes it into the format that is used by the GPU. For this benchmark, this step was not counted in the timing due to the fact that it was assumed that all of the data was already loaded into MATLAB on the host system for both ```GENRE``` and ```glmnet```. The GPU times include the time it takes to transfer data for the model fits from the host system to the GPU, standardize the model matrices, perform the model fits, unstandardize the model coefficients, transfer the computed model coefficients back from the GPU to the host system, and store the coefficients into a MATLAB cell structure. The CPU time includes the time it takes to standardize the model matrices, perform the model fits, unstandardize the model coefficients, and store the coefficients into a MATLAB cell structure. The benchmark results are shown in [Table 1](Table_1_Benchmark_Results.png) below. Note that DP, SP, and SMEM correspond to double precision, single precision, and shared memory, respectively. In addition, note that the input data for the model fits was of type ```double``` for this benchmark. Therefore, in the case of ```GENRE```, some of the inputs would need to be converted to type ```single``` before they are passed to the GPU when using single precision for the computations. Moreover, ```GENRE``` also converts the data type of the computed model coefficients to the data type of the original input data. This means that for the single precision cases, the computed model coefficients would need to be converted to be type ```double``` after they are passed back to the host system from the GPU. For purposes of benchmarking the single precision cases, the time to perform the type conversions of the inputs to type ```single``` was not included, and the returned model coefficients were just kept as type ```single```. This is due to the fact that including these times would increase the benchmark times for the single precision cases in this scenario, and if it were a different scenario, the double precision cases could be impacted instead of the single precision cases. For example, if the type of the original input data was ```single``` and double precision was used for the calculations, then these data type conversions would have to be made for the double precision cases, but they would not have to be made for the single precision cases.
README.md:As shown in [Table 1](Table_1_Benchmark_Results.png), ```GENRE``` provides an order of magnitude speedup when compared to ```glmnet```, and the best performance was achieved by using single precision with shared memory. For ```glmnet```, the benchmark result that is shown was obtained by using the ```naive``` algorithm option for the package because this option was faster than the ```covariance``` algorithm option. For example, the benchmark result that was obtained when using the ```covariance``` algorithm option was 32.271 ![plus_minus](https://latex.codecogs.com/svg.latex?\pm) 0.176 seconds. In addition, it is important to note that in these benchmarks, most of the time for ```GENRE``` was spent transferring the model matrices from the host system to the GPU. However, there are cases when once the model matrices have been used in one call, they can be reused in subsequent calls. For example, a user might want to reuse the same model matrices except just change the ![alpha](https://latex.codecogs.com/svg.latex?%5Calpha) value or the ![lambda](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Clambda) value that is used in elastic-net regularization, or they might want to just change the observation vectors that the model matrices are fit to. By default, each time ```GENRE``` is called, the ```clear mex``` command is executed, and the ```GENRE``` MEX-files are setup so that all allocated memory on the GPU is freed when this command is called. However, in a case where the model matrices can be reused after they are transferred once, the ```clear mex``` command can be removed. Essentially, every time one of the MEX-files for ```GENRE``` is called for the first time, all of the data for the model fits will be transferred to the GPU. However, if the ```clear mex``` command is removed, then for subsequent calls, all of the data for the model fits will still be transferred except for the model matrices, which will be kept on the GPU from the first call. By not having to transfer the model matrices again, performance can be significantly increased. To demonstrate this, the same benchmark from above was repeated, but for each case this time, ```GENRE``` was called before performing the 10 runs. This is to replicate the case where the model matrices are reused in subsequent calls. The benchmark results are shown in 
README.md:As shown in [Table 2](Table_2_Benchmark_Results.png), when the model matrices can be reused and do not have to be transferred again, ```GENRE``` provides a speedup of over two orders of magnitude when compared with ```glmnet```, and using single precision with shared memory provides the best performance. This type of performance gain would most likely be difficult to achieve even when using a multi-CPU implementation of cyclic coordinate descent on a single host system. In addition, it is important to note that this benchmark was just to illustrate an example of when using ```GENRE``` provides performance benefits, but whether or not performance benefits are achieved depends on the problem. For example, in ```GENRE```, one computational thread on the GPU is used to perform each model fit. Therefore, when many model fits are performed on a GPU, the parallelism of the GPU can be utilized. However, if only one model fit needs to be performed, then using a serial CPU implementation such as ```glmnet``` will most likely provide better performance than ```GENRE``` due to factors such as CPU cores having higher clock rates and more resources per core than GPU cores.
README.md:  title = {GENRE (GPU Elastic-Net REgression): A CUDA-Accelerated Package for Massively Parallel Linear Regression with Elastic-Net Regularization},
NOTICE:GENRE (GPU Elastic-Net REgression) software package

```
