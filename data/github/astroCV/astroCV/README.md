# https://github.com/astroCV/astroCV

```console
galaxy_detection/data/darknet/data/9k.names:barracuda
galaxy_detection/data/darknet/data/9k.names:great barracuda
galaxy_detection/data/darknet/data/imagenet.shortnames.list:great barracuda
galaxy_detection/data/darknet/data/imagenet.shortnames.list:rangpur
galaxy_detection/data/darknet/data/imagenet.shortnames.list:barracuda
galaxy_detection/data/darknet/Makefile:GPU=1
galaxy_detection/data/darknet/Makefile:ifeq ($(GPU), 1) 
galaxy_detection/data/darknet/Makefile:COMMON+= -DGPU -I/usr/local/cuda/include/
galaxy_detection/data/darknet/Makefile:CFLAGS+= -DGPU
galaxy_detection/data/darknet/Makefile:LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
galaxy_detection/data/darknet/Makefile:OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o detection_layer.o route_layer.o box.o normalization_layer.o avgpool_layer.o layer.o local_layer.o shortcut_layer.o activation_layer.o rnn_layer.o gru_layer.o crnn_layer.o demo.o batchnorm_layer.o region_layer.o reorg_layer.o tree.o  lstm_layer.o
galaxy_detection/data/darknet/Makefile:ifeq ($(GPU), 1) 
galaxy_detection/data/darknet/README.md:Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.
galaxy_detection/data/darknet/include/darknet.h:extern int gpu_index;
galaxy_detection/data/darknet/include/darknet.h:#ifdef GPU
galaxy_detection/data/darknet/include/darknet.h:    #include "cuda_runtime.h"
galaxy_detection/data/darknet/include/darknet.h:    void (*forward_gpu)   (struct layer, struct network);
galaxy_detection/data/darknet/include/darknet.h:    void (*backward_gpu)  (struct layer, struct network);
galaxy_detection/data/darknet/include/darknet.h:    void (*update_gpu)    (struct layer, update_args);
galaxy_detection/data/darknet/include/darknet.h:#ifdef GPU
galaxy_detection/data/darknet/include/darknet.h:    int *indexes_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float *z_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float *r_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float *h_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float *temp_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float *temp2_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float *temp3_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float *dh_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float *hh_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float *prev_cell_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float *cell_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float *f_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float *i_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float *g_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float *o_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float *c_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float *dc_gpu; 
galaxy_detection/data/darknet/include/darknet.h:    float *m_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float *v_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float *bias_m_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float *scale_m_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float *bias_v_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float *scale_v_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * combine_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * combine_delta_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * prev_state_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * forgot_state_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * forgot_delta_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * state_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * state_delta_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * gate_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * gate_delta_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * save_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * save_delta_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * concat_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * concat_delta_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * binary_input_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * binary_weights_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * mean_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * variance_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * rolling_mean_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * rolling_variance_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * variance_delta_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * mean_delta_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * x_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * x_norm_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * weights_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * weight_updates_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * weight_change_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * biases_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * bias_updates_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * bias_change_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * scales_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * scale_updates_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * scale_change_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * output_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * delta_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * rand_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * squared_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float * norms_gpu;
galaxy_detection/data/darknet/include/darknet.h:    int gpu_index;
galaxy_detection/data/darknet/include/darknet.h:#ifdef GPU
galaxy_detection/data/darknet/include/darknet.h:    float *input_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float *truth_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float *delta_gpu;
galaxy_detection/data/darknet/include/darknet.h:    float *output_gpu;
galaxy_detection/data/darknet/include/darknet.h:#ifdef GPU
galaxy_detection/data/darknet/include/darknet.h:void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY);
galaxy_detection/data/darknet/include/darknet.h:void fill_gpu(int N, float ALPHA, float * X, int INCX);
galaxy_detection/data/darknet/include/darknet.h:void scal_gpu(int N, float ALPHA, float * X, int INCX);
galaxy_detection/data/darknet/include/darknet.h:void copy_gpu(int N, float * X, int INCX, float * Y, int INCY);
galaxy_detection/data/darknet/include/darknet.h:void cuda_set_device(int n);
galaxy_detection/data/darknet/include/darknet.h:void cuda_free(float *x_gpu);
galaxy_detection/data/darknet/include/darknet.h:float *cuda_make_array(float *x, size_t n);
galaxy_detection/data/darknet/include/darknet.h:void cuda_pull_array(float *x_gpu, float *x, size_t n);
galaxy_detection/data/darknet/include/darknet.h:float cuda_mag_array(float *x_gpu, size_t n);
galaxy_detection/data/darknet/include/darknet.h:void cuda_push_array(float *x_gpu, float *x, size_t n);
galaxy_detection/data/darknet/include/darknet.h:void forward_network_gpu(network *net);
galaxy_detection/data/darknet/include/darknet.h:void backward_network_gpu(network *net);
galaxy_detection/data/darknet/include/darknet.h:void update_network_gpu(network *net);
galaxy_detection/data/darknet/include/darknet.h:void harmless_update_network_gpu(network *net);
galaxy_detection/data/darknet/examples/darknet.c:    gpu_index = -1;
galaxy_detection/data/darknet/examples/darknet.c:    gpu_index = -1;
galaxy_detection/data/darknet/examples/darknet.c:    gpu_index = -1;
galaxy_detection/data/darknet/examples/darknet.c:    gpu_index = -1;
galaxy_detection/data/darknet/examples/darknet.c:    gpu_index = -1;
galaxy_detection/data/darknet/examples/darknet.c:    gpu_index = -1;
galaxy_detection/data/darknet/examples/darknet.c:    gpu_index = -1;
galaxy_detection/data/darknet/examples/darknet.c:    gpu_index = -1;
galaxy_detection/data/darknet/examples/darknet.c:    gpu_index = -1;
galaxy_detection/data/darknet/examples/darknet.c:    gpu_index = -1;
galaxy_detection/data/darknet/examples/darknet.c:    gpu_index = -1;
galaxy_detection/data/darknet/examples/darknet.c:    gpu_index = find_int_arg(argc, argv, "-i", 0);
galaxy_detection/data/darknet/examples/darknet.c:    if(find_arg(argc, argv, "-nogpu")) {
galaxy_detection/data/darknet/examples/darknet.c:        gpu_index = -1;
galaxy_detection/data/darknet/examples/darknet.c:#ifndef GPU
galaxy_detection/data/darknet/examples/darknet.c:    gpu_index = -1;
galaxy_detection/data/darknet/examples/darknet.c:    if(gpu_index >= 0){
galaxy_detection/data/darknet/examples/darknet.c:        cuda_set_device(gpu_index);
galaxy_detection/data/darknet/examples/go.c:void train_go(char *cfgfile, char *weightfile, char *filename, int *gpus, int ngpus, int clear)
galaxy_detection/data/darknet/examples/go.c:    printf("%d\n", ngpus);
galaxy_detection/data/darknet/examples/go.c:    network **nets = calloc(ngpus, sizeof(network*));
galaxy_detection/data/darknet/examples/go.c:    for(i = 0; i < ngpus; ++i){
galaxy_detection/data/darknet/examples/go.c:#ifdef GPU
galaxy_detection/data/darknet/examples/go.c:        cuda_set_device(gpus[i]);
galaxy_detection/data/darknet/examples/go.c:        nets[i]->learning_rate *= ngpus;
galaxy_detection/data/darknet/examples/go.c:        data train = random_go_moves(m, net->batch*net->subdivisions*ngpus);
galaxy_detection/data/darknet/examples/go.c:#ifdef GPU
galaxy_detection/data/darknet/examples/go.c:        if(ngpus == 1){
galaxy_detection/data/darknet/examples/go.c:            loss = train_networks(nets, ngpus, train, 4);
galaxy_detection/data/darknet/examples/go.c:    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
galaxy_detection/data/darknet/examples/go.c:    int *gpus = 0;
galaxy_detection/data/darknet/examples/go.c:    int gpu = 0;
galaxy_detection/data/darknet/examples/go.c:    int ngpus = 0;
galaxy_detection/data/darknet/examples/go.c:    if(gpu_list){
galaxy_detection/data/darknet/examples/go.c:        printf("%s\n", gpu_list);
galaxy_detection/data/darknet/examples/go.c:        int len = strlen(gpu_list);
galaxy_detection/data/darknet/examples/go.c:        ngpus = 1;
galaxy_detection/data/darknet/examples/go.c:            if (gpu_list[i] == ',') ++ngpus;
galaxy_detection/data/darknet/examples/go.c:        gpus = calloc(ngpus, sizeof(int));
galaxy_detection/data/darknet/examples/go.c:        for(i = 0; i < ngpus; ++i){
galaxy_detection/data/darknet/examples/go.c:            gpus[i] = atoi(gpu_list);
galaxy_detection/data/darknet/examples/go.c:            gpu_list = strchr(gpu_list, ',')+1;
galaxy_detection/data/darknet/examples/go.c:        gpu = gpu_index;
galaxy_detection/data/darknet/examples/go.c:        gpus = &gpu;
galaxy_detection/data/darknet/examples/go.c:        ngpus = 1;
galaxy_detection/data/darknet/examples/go.c:    if(0==strcmp(argv[2], "train")) train_go(cfg, weights, c2, gpus, ngpus, clear);
galaxy_detection/data/darknet/examples/segmenter.c:void train_segmenter(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int display)
galaxy_detection/data/darknet/examples/segmenter.c:    printf("%d\n", ngpus);
galaxy_detection/data/darknet/examples/segmenter.c:    network **nets = calloc(ngpus, sizeof(network*));
galaxy_detection/data/darknet/examples/segmenter.c:    for(i = 0; i < ngpus; ++i){
galaxy_detection/data/darknet/examples/segmenter.c:#ifdef GPU
galaxy_detection/data/darknet/examples/segmenter.c:        cuda_set_device(gpus[i]);
galaxy_detection/data/darknet/examples/segmenter.c:        nets[i]->learning_rate *= ngpus;
galaxy_detection/data/darknet/examples/segmenter.c:    int imgs = net->batch * net->subdivisions * ngpus;
galaxy_detection/data/darknet/examples/segmenter.c:#ifdef GPU
galaxy_detection/data/darknet/examples/segmenter.c:        if(ngpus == 1){
galaxy_detection/data/darknet/examples/segmenter.c:            loss = train_networks(nets, ngpus, train, 4);
galaxy_detection/data/darknet/examples/segmenter.c:    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
galaxy_detection/data/darknet/examples/segmenter.c:    int *gpus = 0;
galaxy_detection/data/darknet/examples/segmenter.c:    int gpu = 0;
galaxy_detection/data/darknet/examples/segmenter.c:    int ngpus = 0;
galaxy_detection/data/darknet/examples/segmenter.c:    if(gpu_list){
galaxy_detection/data/darknet/examples/segmenter.c:        printf("%s\n", gpu_list);
galaxy_detection/data/darknet/examples/segmenter.c:        int len = strlen(gpu_list);
galaxy_detection/data/darknet/examples/segmenter.c:        ngpus = 1;
galaxy_detection/data/darknet/examples/segmenter.c:            if (gpu_list[i] == ',') ++ngpus;
galaxy_detection/data/darknet/examples/segmenter.c:        gpus = calloc(ngpus, sizeof(int));
galaxy_detection/data/darknet/examples/segmenter.c:        for(i = 0; i < ngpus; ++i){
galaxy_detection/data/darknet/examples/segmenter.c:            gpus[i] = atoi(gpu_list);
galaxy_detection/data/darknet/examples/segmenter.c:            gpu_list = strchr(gpu_list, ',')+1;
galaxy_detection/data/darknet/examples/segmenter.c:        gpu = gpu_index;
galaxy_detection/data/darknet/examples/segmenter.c:        gpus = &gpu;
galaxy_detection/data/darknet/examples/segmenter.c:        ngpus = 1;
galaxy_detection/data/darknet/examples/segmenter.c:    else if(0==strcmp(argv[2], "train")) train_segmenter(data, cfg, weights, gpus, ngpus, clear, display);
galaxy_detection/data/darknet/examples/rnn.c:        #ifdef GPU
galaxy_detection/data/darknet/examples/rnn.c:        cuda_pull_array(l.output_gpu, l.output, l.outputs);
galaxy_detection/data/darknet/examples/nightmare.c:#ifdef GPU
galaxy_detection/data/darknet/examples/nightmare.c:    net->delta_gpu = cuda_make_array(delta.data, im.w*im.h*im.c);
galaxy_detection/data/darknet/examples/nightmare.c:    cuda_push_array(net->input_gpu, im.data, net->inputs);
galaxy_detection/data/darknet/examples/nightmare.c:    forward_network_gpu(net);
galaxy_detection/data/darknet/examples/nightmare.c:    copy_gpu(last.outputs, last.output_gpu, 1, last.delta_gpu, 1);
galaxy_detection/data/darknet/examples/nightmare.c:    cuda_pull_array(last.delta_gpu, last.delta, last.outputs);
galaxy_detection/data/darknet/examples/nightmare.c:    cuda_push_array(last.delta_gpu, last.delta, last.outputs);
galaxy_detection/data/darknet/examples/nightmare.c:    backward_network_gpu(net);
galaxy_detection/data/darknet/examples/nightmare.c:    cuda_pull_array(net->delta_gpu, delta.data, im.w*im.h*im.c);
galaxy_detection/data/darknet/examples/nightmare.c:    cuda_free(net->delta_gpu);
galaxy_detection/data/darknet/examples/nightmare.c:    net->delta_gpu = 0;
galaxy_detection/data/darknet/examples/nightmare.c:#ifdef GPU
galaxy_detection/data/darknet/examples/nightmare.c:        cuda_push_array(net->input_gpu, recon.data, recon.w*recon.h*recon.c);
galaxy_detection/data/darknet/examples/nightmare.c:        //cuda_push_array(net->truth_gpu, features, net->truths);
galaxy_detection/data/darknet/examples/nightmare.c:        net->delta_gpu = cuda_make_array(delta.data, delta.w*delta.h*delta.c);
galaxy_detection/data/darknet/examples/nightmare.c:        forward_network_gpu(net);
galaxy_detection/data/darknet/examples/nightmare.c:        cuda_push_array(l.delta_gpu, features, l.outputs);
galaxy_detection/data/darknet/examples/nightmare.c:        axpy_gpu(l.outputs, -1, l.output_gpu, 1, l.delta_gpu, 1);
galaxy_detection/data/darknet/examples/nightmare.c:        backward_network_gpu(net);
galaxy_detection/data/darknet/examples/nightmare.c:        cuda_pull_array(net->delta_gpu, delta.data, delta.w*delta.h*delta.c);
galaxy_detection/data/darknet/examples/nightmare.c:        cuda_free(net->delta_gpu);
galaxy_detection/data/darknet/examples/detector.c:void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
galaxy_detection/data/darknet/examples/detector.c:    network **nets = calloc(ngpus, sizeof(network));
galaxy_detection/data/darknet/examples/detector.c:    for(i = 0; i < ngpus; ++i){
galaxy_detection/data/darknet/examples/detector.c:#ifdef GPU
galaxy_detection/data/darknet/examples/detector.c:        cuda_set_device(gpus[i]);
galaxy_detection/data/darknet/examples/detector.c:        nets[i]->learning_rate *= ngpus;
galaxy_detection/data/darknet/examples/detector.c:    int imgs = net->batch * net->subdivisions * ngpus;
galaxy_detection/data/darknet/examples/detector.c:            for(i = 0; i < ngpus; ++i){
galaxy_detection/data/darknet/examples/detector.c:#ifdef GPU
galaxy_detection/data/darknet/examples/detector.c:        if(ngpus == 1){
galaxy_detection/data/darknet/examples/detector.c:            loss = train_networks(nets, ngpus, train, 4);
galaxy_detection/data/darknet/examples/detector.c:#ifdef GPU
galaxy_detection/data/darknet/examples/detector.c:            if(ngpus != 1) sync_nets(nets, ngpus, 0);
galaxy_detection/data/darknet/examples/detector.c:#ifdef GPU
galaxy_detection/data/darknet/examples/detector.c:            if(ngpus != 1) sync_nets(nets, ngpus, 0);
galaxy_detection/data/darknet/examples/detector.c:#ifdef GPU
galaxy_detection/data/darknet/examples/detector.c:    if(ngpus != 1) sync_nets(nets, ngpus, 0);
galaxy_detection/data/darknet/examples/detector.c:    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
galaxy_detection/data/darknet/examples/detector.c:    int *gpus = 0;
galaxy_detection/data/darknet/examples/detector.c:    int gpu = 0;
galaxy_detection/data/darknet/examples/detector.c:    int ngpus = 0;
galaxy_detection/data/darknet/examples/detector.c:    if(gpu_list){
galaxy_detection/data/darknet/examples/detector.c:        printf("%s\n", gpu_list);
galaxy_detection/data/darknet/examples/detector.c:        int len = strlen(gpu_list);
galaxy_detection/data/darknet/examples/detector.c:        ngpus = 1;
galaxy_detection/data/darknet/examples/detector.c:            if (gpu_list[i] == ',') ++ngpus;
galaxy_detection/data/darknet/examples/detector.c:        gpus = calloc(ngpus, sizeof(int));
galaxy_detection/data/darknet/examples/detector.c:        for(i = 0; i < ngpus; ++i){
galaxy_detection/data/darknet/examples/detector.c:            gpus[i] = atoi(gpu_list);
galaxy_detection/data/darknet/examples/detector.c:            gpu_list = strchr(gpu_list, ',')+1;
galaxy_detection/data/darknet/examples/detector.c:        gpu = gpu_index;
galaxy_detection/data/darknet/examples/detector.c:        gpus = &gpu;
galaxy_detection/data/darknet/examples/detector.c:        ngpus = 1;
galaxy_detection/data/darknet/examples/detector.c:    else if(0==strcmp(argv[2], "train")) train_detector(datacfg, cfg, weights, gpus, ngpus, clear);
galaxy_detection/data/darknet/examples/classifier.c:void train_classifier(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
galaxy_detection/data/darknet/examples/classifier.c:    printf("%d\n", ngpus);
galaxy_detection/data/darknet/examples/classifier.c:    network **nets = calloc(ngpus, sizeof(network*));
galaxy_detection/data/darknet/examples/classifier.c:    for(i = 0; i < ngpus; ++i){
galaxy_detection/data/darknet/examples/classifier.c:#ifdef GPU
galaxy_detection/data/darknet/examples/classifier.c:        cuda_set_device(gpus[i]);
galaxy_detection/data/darknet/examples/classifier.c:        nets[i]->learning_rate *= ngpus;
galaxy_detection/data/darknet/examples/classifier.c:    int imgs = net->batch * net->subdivisions * ngpus;
galaxy_detection/data/darknet/examples/classifier.c:            for(i = 0; i < ngpus; ++i){
galaxy_detection/data/darknet/examples/classifier.c:#ifdef GPU
galaxy_detection/data/darknet/examples/classifier.c:        if(ngpus == 1){
galaxy_detection/data/darknet/examples/classifier.c:            loss = train_networks(nets, ngpus, train, 4);
galaxy_detection/data/darknet/examples/classifier.c:#ifdef GPU
galaxy_detection/data/darknet/examples/classifier.c:        cuda_pull_array(l.output_gpu, l.output, l.outputs);
galaxy_detection/data/darknet/examples/classifier.c:    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
galaxy_detection/data/darknet/examples/classifier.c:    int ngpus;
galaxy_detection/data/darknet/examples/classifier.c:    int *gpus = read_intlist(gpu_list, &ngpus, gpu_index);
galaxy_detection/data/darknet/examples/classifier.c:    else if(0==strcmp(argv[2], "train")) train_classifier(data, cfg, weights, gpus, ngpus, clear);
galaxy_detection/data/darknet/examples/regressor.c:void train_regressor(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
galaxy_detection/data/darknet/examples/regressor.c:    printf("%d\n", ngpus);
galaxy_detection/data/darknet/examples/regressor.c:    network **nets = calloc(ngpus, sizeof(network*));
galaxy_detection/data/darknet/examples/regressor.c:    for(i = 0; i < ngpus; ++i){
galaxy_detection/data/darknet/examples/regressor.c:#ifdef GPU
galaxy_detection/data/darknet/examples/regressor.c:        cuda_set_device(gpus[i]);
galaxy_detection/data/darknet/examples/regressor.c:        nets[i]->learning_rate *= ngpus;
galaxy_detection/data/darknet/examples/regressor.c:    int imgs = net->batch * net->subdivisions * ngpus;
galaxy_detection/data/darknet/examples/regressor.c:#ifdef GPU
galaxy_detection/data/darknet/examples/regressor.c:        if(ngpus == 1){
galaxy_detection/data/darknet/examples/regressor.c:            loss = train_networks(nets, ngpus, train, 4);
galaxy_detection/data/darknet/examples/regressor.c:    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
galaxy_detection/data/darknet/examples/regressor.c:    int *gpus = 0;
galaxy_detection/data/darknet/examples/regressor.c:    int gpu = 0;
galaxy_detection/data/darknet/examples/regressor.c:    int ngpus = 0;
galaxy_detection/data/darknet/examples/regressor.c:    if(gpu_list){
galaxy_detection/data/darknet/examples/regressor.c:        printf("%s\n", gpu_list);
galaxy_detection/data/darknet/examples/regressor.c:        int len = strlen(gpu_list);
galaxy_detection/data/darknet/examples/regressor.c:        ngpus = 1;
galaxy_detection/data/darknet/examples/regressor.c:            if (gpu_list[i] == ',') ++ngpus;
galaxy_detection/data/darknet/examples/regressor.c:        gpus = calloc(ngpus, sizeof(int));
galaxy_detection/data/darknet/examples/regressor.c:        for(i = 0; i < ngpus; ++i){
galaxy_detection/data/darknet/examples/regressor.c:            gpus[i] = atoi(gpu_list);
galaxy_detection/data/darknet/examples/regressor.c:            gpu_list = strchr(gpu_list, ',')+1;
galaxy_detection/data/darknet/examples/regressor.c:        gpu = gpu_index;
galaxy_detection/data/darknet/examples/regressor.c:        gpus = &gpu;
galaxy_detection/data/darknet/examples/regressor.c:        ngpus = 1;
galaxy_detection/data/darknet/examples/regressor.c:    else if(0==strcmp(argv[2], "train")) train_regressor(data, cfg, weights, gpus, ngpus, clear);
galaxy_detection/data/darknet/examples/attention.c:void train_attention(char *datacfg, char *cfgfile, char *weightfile, char *cfgfile2, char *weightfile2, int *gpus, int ngpus, int clear)
galaxy_detection/data/darknet/examples/attention.c:    printf("%d\n", ngpus);
galaxy_detection/data/darknet/examples/attention.c:    network **attnets = calloc(ngpus, sizeof(network*));
galaxy_detection/data/darknet/examples/attention.c:    network **clsnets = calloc(ngpus, sizeof(network*));
galaxy_detection/data/darknet/examples/attention.c:    for(i = 0; i < ngpus; ++i){
galaxy_detection/data/darknet/examples/attention.c:#ifdef GPU
galaxy_detection/data/darknet/examples/attention.c:        cuda_set_device(gpus[i]);
galaxy_detection/data/darknet/examples/attention.c:        attnets[i]->learning_rate *= ngpus;
galaxy_detection/data/darknet/examples/attention.c:        clsnets[i]->learning_rate *= ngpus;
galaxy_detection/data/darknet/examples/attention.c:    int imgs = net->batch * net->subdivisions * ngpus;
galaxy_detection/data/darknet/examples/attention.c:#ifdef GPU
galaxy_detection/data/darknet/examples/attention.c:        if(ngpus == 1){
galaxy_detection/data/darknet/examples/attention.c:            loss = train_networks(attnets, ngpus, train, 4);
galaxy_detection/data/darknet/examples/attention.c:    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
galaxy_detection/data/darknet/examples/attention.c:    int ngpus;
galaxy_detection/data/darknet/examples/attention.c:    int *gpus = read_intlist(gpu_list, &ngpus, gpu_index);
galaxy_detection/data/darknet/examples/attention.c:    else if(0==strcmp(argv[2], "train")) train_attention(data, cfg, weights, filename, layer_s, gpus, ngpus, clear);
galaxy_detection/data/darknet/examples/lsd.c:#ifdef GPU
galaxy_detection/data/darknet/examples/lsd.c:    fill_gpu(ay_size, .9, anet->truth_gpu, 1);
galaxy_detection/data/darknet/examples/lsd.c:    anet->delta_gpu = cuda_make_array(0, ax_size);
galaxy_detection/data/darknet/examples/lsd.c:    gstate.input = cuda_make_array(0, gx_size);
galaxy_detection/data/darknet/examples/lsd.c:            cuda_push_array(fstate.input, X, x_size);
galaxy_detection/data/darknet/examples/lsd.c:            cuda_push_array(gstate.input, X, gx_size);
galaxy_detection/data/darknet/examples/lsd.c:            forward_network_gpu(fnet, fstate);
galaxy_detection/data/darknet/examples/lsd.c:            float *feats = fnet->layers[fnet->n - 2].output_gpu;
galaxy_detection/data/darknet/examples/lsd.c:            copy_gpu(y_size, feats, 1, fstate.truth, 1);
galaxy_detection/data/darknet/examples/lsd.c:            forward_network_gpu(gnet, gstate);
galaxy_detection/data/darknet/examples/lsd.c:            float *gen = gnet->layers[gnet->n-1].output_gpu;
galaxy_detection/data/darknet/examples/lsd.c:            copy_gpu(x_size, gen, 1, fstate.input, 1);
galaxy_detection/data/darknet/examples/lsd.c:            fill_gpu(x_size, 0, fstate.delta, 1);
galaxy_detection/data/darknet/examples/lsd.c:            forward_network_gpu(fnet, fstate);
galaxy_detection/data/darknet/examples/lsd.c:            backward_network_gpu(fnet, fstate);
galaxy_detection/data/darknet/examples/lsd.c:            fill_gpu(ax_size, 0, astate.delta, 1);
galaxy_detection/data/darknet/examples/lsd.c:            forward_network_gpu(anet, astate);
galaxy_detection/data/darknet/examples/lsd.c:            backward_network_gpu(anet, astate);
galaxy_detection/data/darknet/examples/lsd.c:            float *delta = imlayer.delta_gpu;
galaxy_detection/data/darknet/examples/lsd.c:            fill_gpu(x_size, 0, delta, 1);
galaxy_detection/data/darknet/examples/lsd.c:            scal_gpu(x_size, 100, astate.delta, 1);
galaxy_detection/data/darknet/examples/lsd.c:            scal_gpu(x_size, .001, fstate.delta, 1);
galaxy_detection/data/darknet/examples/lsd.c:            axpy_gpu(x_size, 1, fstate.delta, 1, delta, 1);
galaxy_detection/data/darknet/examples/lsd.c:            axpy_gpu(x_size, 1, astate.delta, 1, delta, 1);
galaxy_detection/data/darknet/examples/lsd.c:            //fill_gpu(x_size, 0, delta, 1);
galaxy_detection/data/darknet/examples/lsd.c:            //cuda_push_array(delta, X, x_size);
galaxy_detection/data/darknet/examples/lsd.c:            //axpy_gpu(x_size, -1, imlayer.output_gpu, 1, delta, 1);
galaxy_detection/data/darknet/examples/lsd.c:            //printf("pix error: %f\n", cuda_mag_array(delta, x_size));
galaxy_detection/data/darknet/examples/lsd.c:            printf("fea error: %f\n", cuda_mag_array(fstate.delta, x_size));
galaxy_detection/data/darknet/examples/lsd.c:            printf("adv error: %f\n", cuda_mag_array(astate.delta, x_size));
galaxy_detection/data/darknet/examples/lsd.c:            //axpy_gpu(x_size, 1, astate.delta, 1, delta, 1);
galaxy_detection/data/darknet/examples/lsd.c:            backward_network_gpu(gnet, gstate);
galaxy_detection/data/darknet/examples/lsd.c:            cuda_pull_array(imlayer.output_gpu, imlayer.output, imlayer.outputs*imlayer.batch);
galaxy_detection/data/darknet/examples/lsd.c:        harmless_update_network_gpu(anet);
galaxy_detection/data/darknet/examples/lsd.c:        update_network_gpu(gnet);
galaxy_detection/data/darknet/examples/lsd.c:#ifdef GPU
galaxy_detection/data/darknet/examples/lsd.c:    gstate.input = cuda_make_array(0, x_size);
galaxy_detection/data/darknet/examples/lsd.c:    gstate.truth = cuda_make_array(0, y_size);
galaxy_detection/data/darknet/examples/lsd.c:    float *imerror = cuda_make_array(0, imlayer.outputs);
galaxy_detection/data/darknet/examples/lsd.c:    float *ones_gpu = cuda_make_array(0, ay_size);
galaxy_detection/data/darknet/examples/lsd.c:    fill_gpu(ay_size, .9, ones_gpu, 1);
galaxy_detection/data/darknet/examples/lsd.c:            cuda_push_array(gstate.input, graypixs, x_size);
galaxy_detection/data/darknet/examples/lsd.c:            cuda_push_array(gstate.truth, pixs, y_size);
galaxy_detection/data/darknet/examples/lsd.c:            forward_network_gpu(net, gstate);
galaxy_detection/data/darknet/examples/lsd.c:            fill_gpu(imlayer.outputs, 0, imerror, 1);
galaxy_detection/data/darknet/examples/lsd.c:            astate.input = imlayer.output_gpu;
galaxy_detection/data/darknet/examples/lsd.c:            astate.truth = ones_gpu;
galaxy_detection/data/darknet/examples/lsd.c:            forward_network_gpu(anet, astate);
galaxy_detection/data/darknet/examples/lsd.c:            backward_network_gpu(anet, astate);
galaxy_detection/data/darknet/examples/lsd.c:            scal_gpu(imlayer.outputs, .1, net->layers[net->n-1].delta_gpu, 1);
galaxy_detection/data/darknet/examples/lsd.c:            backward_network_gpu(net, gstate);
galaxy_detection/data/darknet/examples/lsd.c:            scal_gpu(imlayer.outputs, 1000, imerror, 1);
galaxy_detection/data/darknet/examples/lsd.c:            printf("realness %f\n", cuda_mag_array(imerror, imlayer.outputs));
galaxy_detection/data/darknet/examples/lsd.c:            printf("features %f\n", cuda_mag_array(net->layers[net->n-1].delta_gpu, imlayer.outputs));
galaxy_detection/data/darknet/examples/lsd.c:            axpy_gpu(imlayer.outputs, 1, imerror, 1, imlayer.delta_gpu, 1);
galaxy_detection/data/darknet/examples/lsd.c:            cuda_pull_array(imlayer.output_gpu, imlayer.output, imlayer.outputs*imlayer.batch);
galaxy_detection/data/darknet/examples/lsd.c:        harmless_update_network_gpu(anet);
galaxy_detection/data/darknet/examples/lsd.c:        update_network_gpu(net);
galaxy_detection/data/darknet/examples/lsd.c:        update_network_gpu(anet);
galaxy_detection/data/darknet/examples/lsd.c:#ifdef GPU
galaxy_detection/data/darknet/examples/lsd.c:    float *imerror = cuda_make_array(0, y_size);
galaxy_detection/data/darknet/examples/lsd.c:            cuda_push_array(gnet->input_gpu, gnet->input, x_size);
galaxy_detection/data/darknet/examples/lsd.c:            cuda_push_array(gnet->truth_gpu, gnet->truth, y_size);
galaxy_detection/data/darknet/examples/lsd.c:            forward_network_gpu(gnet);
galaxy_detection/data/darknet/examples/lsd.c:            fill_gpu(imlayer.outputs*imlayer.batch, 0, imerror, 1);
galaxy_detection/data/darknet/examples/lsd.c:            fill_gpu(anet->truths*anet->batch, .95, anet->truth_gpu, 1);
galaxy_detection/data/darknet/examples/lsd.c:            copy_gpu(anet->inputs*anet->batch, imlayer.output_gpu, 1, anet->input_gpu, 1);
galaxy_detection/data/darknet/examples/lsd.c:            anet->delta_gpu = imerror;
galaxy_detection/data/darknet/examples/lsd.c:            forward_network_gpu(anet);
galaxy_detection/data/darknet/examples/lsd.c:            backward_network_gpu(anet);
galaxy_detection/data/darknet/examples/lsd.c:            scal_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1);
galaxy_detection/data/darknet/examples/lsd.c:            scal_gpu(imlayer.outputs*imlayer.batch, .00, gnet->layers[gnet->n-1].delta_gpu, 1);
galaxy_detection/data/darknet/examples/lsd.c:            printf("realness %f\n", cuda_mag_array(imerror, imlayer.outputs*imlayer.batch));
galaxy_detection/data/darknet/examples/lsd.c:            printf("features %f\n", cuda_mag_array(gnet->layers[gnet->n-1].delta_gpu, imlayer.outputs*imlayer.batch));
galaxy_detection/data/darknet/examples/lsd.c:            axpy_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1, gnet->layers[gnet->n-1].delta_gpu, 1);
galaxy_detection/data/darknet/examples/lsd.c:            backward_network_gpu(gnet);
galaxy_detection/data/darknet/examples/lsd.c:        harmless_update_network_gpu(anet);
galaxy_detection/data/darknet/examples/lsd.c:        update_network_gpu(gnet);
galaxy_detection/data/darknet/examples/lsd.c:#ifdef GPU
galaxy_detection/data/darknet/examples/lsd.c:    float *imerror = cuda_make_array(0, imlayer.outputs*imlayer.batch);
galaxy_detection/data/darknet/examples/lsd.c:            cuda_push_array(net->input_gpu, graypixs, net->inputs*net->batch);
galaxy_detection/data/darknet/examples/lsd.c:            cuda_push_array(net->truth_gpu, pixs, net->truths*net->batch);
galaxy_detection/data/darknet/examples/lsd.c:            forward_network_gpu(net);
galaxy_detection/data/darknet/examples/lsd.c:            fill_gpu(imlayer.outputs*imlayer.batch, 0, imerror, 1);
galaxy_detection/data/darknet/examples/lsd.c:            copy_gpu(anet->inputs*anet->batch, imlayer.output_gpu, 1, anet->input_gpu, 1);
galaxy_detection/data/darknet/examples/lsd.c:            fill_gpu(anet->inputs*anet->batch, .95, anet->truth_gpu, 1);
galaxy_detection/data/darknet/examples/lsd.c:            anet->delta_gpu = imerror;
galaxy_detection/data/darknet/examples/lsd.c:            forward_network_gpu(anet);
galaxy_detection/data/darknet/examples/lsd.c:            backward_network_gpu(anet);
galaxy_detection/data/darknet/examples/lsd.c:            scal_gpu(imlayer.outputs*imlayer.batch, 1./100., net->layers[net->n-1].delta_gpu, 1);
galaxy_detection/data/darknet/examples/lsd.c:            scal_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1);
galaxy_detection/data/darknet/examples/lsd.c:            printf("realness %f\n", cuda_mag_array(imerror, imlayer.outputs*imlayer.batch));
galaxy_detection/data/darknet/examples/lsd.c:            printf("features %f\n", cuda_mag_array(net->layers[net->n-1].delta_gpu, imlayer.outputs*imlayer.batch));
galaxy_detection/data/darknet/examples/lsd.c:            axpy_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1, net->layers[net->n-1].delta_gpu, 1);
galaxy_detection/data/darknet/examples/lsd.c:            backward_network_gpu(net);
galaxy_detection/data/darknet/examples/lsd.c:        harmless_update_network_gpu(anet);
galaxy_detection/data/darknet/examples/lsd.c:        update_network_gpu(net);
galaxy_detection/data/darknet/examples/lsd.c:#ifdef GPU
galaxy_detection/data/darknet/examples/lsd.c:    gstate.input = cuda_make_array(0, x_size);
galaxy_detection/data/darknet/examples/lsd.c:    float *imerror = cuda_make_array(0, imlayer.outputs);
galaxy_detection/data/darknet/examples/lsd.c:    float *ones_gpu = cuda_make_array(0, ay_size);
galaxy_detection/data/darknet/examples/lsd.c:    fill_gpu(ay_size, 1, ones_gpu, 1);
galaxy_detection/data/darknet/examples/lsd.c:            cuda_push_array(gstate.input, X, x_size);
galaxy_detection/data/darknet/examples/lsd.c:            forward_network_gpu(net, gstate);
galaxy_detection/data/darknet/examples/lsd.c:            fill_gpu(imlayer.outputs, 0, imerror, 1);
galaxy_detection/data/darknet/examples/lsd.c:            astate.input = imlayer.output_gpu;
galaxy_detection/data/darknet/examples/lsd.c:            astate.truth = ones_gpu;
galaxy_detection/data/darknet/examples/lsd.c:            forward_network_gpu(anet, astate);
galaxy_detection/data/darknet/examples/lsd.c:            backward_network_gpu(anet, astate);
galaxy_detection/data/darknet/examples/lsd.c:            scal_gpu(imlayer.outputs, 1, imerror, 1);
galaxy_detection/data/darknet/examples/lsd.c:            axpy_gpu(imlayer.outputs, 1, imerror, 1, imlayer.delta_gpu, 1);
galaxy_detection/data/darknet/examples/lsd.c:            backward_network_gpu(net, gstate);
galaxy_detection/data/darknet/examples/lsd.c:            printf("features %f\n", cuda_mag_array(imlayer.delta_gpu, imlayer.outputs));
galaxy_detection/data/darknet/examples/lsd.c:            printf("realness %f\n", cuda_mag_array(imerror, imlayer.outputs));
galaxy_detection/data/darknet/examples/lsd.c:            cuda_pull_array(imlayer.output_gpu, imlayer.output, imlayer.outputs*imlayer.batch);
galaxy_detection/data/darknet/examples/lsd.c:        harmless_update_network_gpu(anet);
galaxy_detection/data/darknet/examples/lsd.c:        update_network_gpu(net);
galaxy_detection/data/darknet/examples/lsd.c:        update_network_gpu(anet);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:#include "cuda_runtime.h"
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:#include "cuda.h"
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:extern "C" void forward_deconvolutional_layer_gpu(layer l, network net)
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        float *a = l.weights_gpu;
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        float *b = net.input_gpu + i*l.c*l.h*l.w;
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        gemm_gpu(1,0,m,n,k,1,a,m,b,n,0,c,n);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        col2im_gpu(net.workspace, l.out_c, l.out_h, l.out_w, l.size, l.stride, l.pad, l.output_gpu+i*l.outputs);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        forward_batchnorm_layer_gpu(l, net);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:    activate_array_gpu(l.output_gpu, l.batch*l.n*l.out_w*l.out_h, l.activation);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:extern "C" void backward_deconvolutional_layer_gpu(layer l, network net)
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:    constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        backward_batchnorm_layer_gpu(l, net);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:    //if(net.delta_gpu) memset(net.delta_gpu, 0, l.batch*l.h*l.w*l.c*sizeof(float));
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        float *a = net.input_gpu + i*m*k;
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        float *c = l.weight_updates_gpu;
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        im2col_gpu(l.delta_gpu + i*l.outputs, l.out_c, l.out_h, l.out_w, 
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        if(net.delta_gpu){
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:            float *a = l.weights_gpu;
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:            float *c = net.delta_gpu + i*n*m;
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:    cuda_pull_array(l.weights_gpu, l.weights, l.c*l.n*l.size*l.size);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:    cuda_pull_array(l.biases_gpu, l.biases, l.n);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.c*l.n*l.size*l.size);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        cuda_pull_array(l.scales_gpu, l.scales, l.n);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:    cuda_push_array(l.weights_gpu, l.weights, l.c*l.n*l.size*l.size);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:    cuda_push_array(l.biases_gpu, l.biases, l.n);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.c*l.n*l.size*l.size);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        cuda_push_array(l.scales_gpu, l.scales, l.n);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:void update_deconvolutional_layer_gpu(layer l, update_args a)
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, size, batch, a.t);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        if(l.scales_gpu){
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        axpy_gpu(size, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        axpy_gpu(size, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        scal_gpu(size, momentum, l.weight_updates_gpu, 1);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:        if(l.scales_gpu){
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:            axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
galaxy_detection/data/darknet/src/deconvolutional_kernels.cu:            scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
galaxy_detection/data/darknet/src/maxpool_layer.h:#include "cuda.h"
galaxy_detection/data/darknet/src/maxpool_layer.h:#ifdef GPU
galaxy_detection/data/darknet/src/maxpool_layer.h:void forward_maxpool_layer_gpu(maxpool_layer l, network net);
galaxy_detection/data/darknet/src/maxpool_layer.h:void backward_maxpool_layer_gpu(maxpool_layer l, network net);
galaxy_detection/data/darknet/src/crop_layer.c:#include "cuda.h"
galaxy_detection/data/darknet/src/crop_layer.c:void backward_crop_layer_gpu(const crop_layer l, network net){}
galaxy_detection/data/darknet/src/crop_layer.c:    #ifdef GPU
galaxy_detection/data/darknet/src/crop_layer.c:    l.forward_gpu = forward_crop_layer_gpu;
galaxy_detection/data/darknet/src/crop_layer.c:    l.backward_gpu = backward_crop_layer_gpu;
galaxy_detection/data/darknet/src/crop_layer.c:    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
galaxy_detection/data/darknet/src/crop_layer.c:    l.rand_gpu   = cuda_make_array(0, l.batch*8);
galaxy_detection/data/darknet/src/crop_layer.c:    #ifdef GPU
galaxy_detection/data/darknet/src/crop_layer.c:    cuda_free(l->output_gpu);
galaxy_detection/data/darknet/src/crop_layer.c:    l->output_gpu = cuda_make_array(l->output, l->outputs*l->batch);
galaxy_detection/data/darknet/src/softmax_layer.c:#include "cuda.h"
galaxy_detection/data/darknet/src/softmax_layer.c:    #ifdef GPU
galaxy_detection/data/darknet/src/softmax_layer.c:    l.forward_gpu = forward_softmax_layer_gpu;
galaxy_detection/data/darknet/src/softmax_layer.c:    l.backward_gpu = backward_softmax_layer_gpu;
galaxy_detection/data/darknet/src/softmax_layer.c:    l.output_gpu = cuda_make_array(l.output, inputs*batch); 
galaxy_detection/data/darknet/src/softmax_layer.c:    l.delta_gpu = cuda_make_array(l.delta, inputs*batch); 
galaxy_detection/data/darknet/src/softmax_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/softmax_layer.c:    cuda_pull_array(layer.output_gpu, layer.output, layer.inputs*layer.batch);
galaxy_detection/data/darknet/src/softmax_layer.c:void forward_softmax_layer_gpu(const softmax_layer l, network net)
galaxy_detection/data/darknet/src/softmax_layer.c:            softmax_gpu(net.input_gpu + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output_gpu + count);
galaxy_detection/data/darknet/src/softmax_layer.c:            softmax_gpu(net.input_gpu, l.c, l.batch*l.c, l.inputs/l.c, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu);
galaxy_detection/data/darknet/src/softmax_layer.c:            softmax_gpu(net.input_gpu, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output_gpu);
galaxy_detection/data/darknet/src/softmax_layer.c:void backward_softmax_layer_gpu(const softmax_layer layer, network net)
galaxy_detection/data/darknet/src/softmax_layer.c:    axpy_gpu(layer.batch*layer.inputs, 1, layer.delta_gpu, 1, net.delta_gpu, 1);
galaxy_detection/data/darknet/src/image.c:#include "cuda.h"
galaxy_detection/data/darknet/src/avgpool_layer.h:#include "cuda.h"
galaxy_detection/data/darknet/src/avgpool_layer.h:#ifdef GPU
galaxy_detection/data/darknet/src/avgpool_layer.h:void forward_avgpool_layer_gpu(avgpool_layer l, network net);
galaxy_detection/data/darknet/src/avgpool_layer.h:void backward_avgpool_layer_gpu(avgpool_layer l, network net);
galaxy_detection/data/darknet/src/gemm.h:#ifdef GPU
galaxy_detection/data/darknet/src/gemm.h:void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
galaxy_detection/data/darknet/src/gemm.h:        float *A_gpu, int lda, 
galaxy_detection/data/darknet/src/gemm.h:        float *B_gpu, int ldb,
galaxy_detection/data/darknet/src/gemm.h:        float *C_gpu, int ldc);
galaxy_detection/data/darknet/src/gemm.h:void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
galaxy_detection/data/darknet/src/network.h:#ifdef GPU
galaxy_detection/data/darknet/src/shortcut_layer.h:#ifdef GPU
galaxy_detection/data/darknet/src/shortcut_layer.h:void forward_shortcut_layer_gpu(const layer l, network net);
galaxy_detection/data/darknet/src/shortcut_layer.h:void backward_shortcut_layer_gpu(const layer l, network net);
galaxy_detection/data/darknet/src/crop_layer_kernels.cu:#include "cuda_runtime.h"
galaxy_detection/data/darknet/src/crop_layer_kernels.cu:#include "cuda.h"
galaxy_detection/data/darknet/src/crop_layer_kernels.cu:extern "C" void forward_crop_layer_gpu(crop_layer layer, network net)
galaxy_detection/data/darknet/src/crop_layer_kernels.cu:    cuda_random(layer.rand_gpu, layer.batch*8);
galaxy_detection/data/darknet/src/crop_layer_kernels.cu:    levels_image_kernel<<<cuda_gridsize(size), BLOCK>>>(net.input_gpu, layer.rand_gpu, layer.batch, layer.w, layer.h, net.train, layer.saturation, layer.exposure, translate, scale, layer.shift);
galaxy_detection/data/darknet/src/crop_layer_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/crop_layer_kernels.cu:    forward_crop_layer_kernel<<<cuda_gridsize(size), BLOCK>>>(net.input_gpu, layer.rand_gpu, size, layer.c, layer.h, layer.w, layer.out_h, layer.out_w, net.train, layer.flip, radians, layer.output_gpu);
galaxy_detection/data/darknet/src/crop_layer_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/crop_layer_kernels.cu:       cuda_pull_array(layer.output_gpu, layer.output, size);
galaxy_detection/data/darknet/src/connected_layer.c:#include "cuda.h"
galaxy_detection/data/darknet/src/connected_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/connected_layer.c:    l.forward_gpu = forward_connected_layer_gpu;
galaxy_detection/data/darknet/src/connected_layer.c:    l.backward_gpu = backward_connected_layer_gpu;
galaxy_detection/data/darknet/src/connected_layer.c:    l.update_gpu = update_connected_layer_gpu;
galaxy_detection/data/darknet/src/connected_layer.c:    l.weights_gpu = cuda_make_array(l.weights, outputs*inputs);
galaxy_detection/data/darknet/src/connected_layer.c:    l.biases_gpu = cuda_make_array(l.biases, outputs);
galaxy_detection/data/darknet/src/connected_layer.c:    l.weight_updates_gpu = cuda_make_array(l.weight_updates, outputs*inputs);
galaxy_detection/data/darknet/src/connected_layer.c:    l.bias_updates_gpu = cuda_make_array(l.bias_updates, outputs);
galaxy_detection/data/darknet/src/connected_layer.c:    l.output_gpu = cuda_make_array(l.output, outputs*batch);
galaxy_detection/data/darknet/src/connected_layer.c:    l.delta_gpu = cuda_make_array(l.delta, outputs*batch);
galaxy_detection/data/darknet/src/connected_layer.c:        l.m_gpu =       cuda_make_array(0, inputs*outputs);
galaxy_detection/data/darknet/src/connected_layer.c:        l.v_gpu =       cuda_make_array(0, inputs*outputs);
galaxy_detection/data/darknet/src/connected_layer.c:        l.bias_m_gpu =  cuda_make_array(0, outputs);
galaxy_detection/data/darknet/src/connected_layer.c:        l.bias_v_gpu =  cuda_make_array(0, outputs);
galaxy_detection/data/darknet/src/connected_layer.c:        l.scale_m_gpu = cuda_make_array(0, outputs);
galaxy_detection/data/darknet/src/connected_layer.c:        l.scale_v_gpu = cuda_make_array(0, outputs);
galaxy_detection/data/darknet/src/connected_layer.c:        l.mean_gpu = cuda_make_array(l.mean, outputs);
galaxy_detection/data/darknet/src/connected_layer.c:        l.variance_gpu = cuda_make_array(l.variance, outputs);
galaxy_detection/data/darknet/src/connected_layer.c:        l.rolling_mean_gpu = cuda_make_array(l.mean, outputs);
galaxy_detection/data/darknet/src/connected_layer.c:        l.rolling_variance_gpu = cuda_make_array(l.variance, outputs);
galaxy_detection/data/darknet/src/connected_layer.c:        l.mean_delta_gpu = cuda_make_array(l.mean, outputs);
galaxy_detection/data/darknet/src/connected_layer.c:        l.variance_delta_gpu = cuda_make_array(l.variance, outputs);
galaxy_detection/data/darknet/src/connected_layer.c:        l.scales_gpu = cuda_make_array(l.scales, outputs);
galaxy_detection/data/darknet/src/connected_layer.c:        l.scale_updates_gpu = cuda_make_array(l.scale_updates, outputs);
galaxy_detection/data/darknet/src/connected_layer.c:        l.x_gpu = cuda_make_array(l.output, l.batch*outputs);
galaxy_detection/data/darknet/src/connected_layer.c:        l.x_norm_gpu = cuda_make_array(l.output, l.batch*outputs);
galaxy_detection/data/darknet/src/connected_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/connected_layer.c:    cuda_pull_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
galaxy_detection/data/darknet/src/connected_layer.c:    cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
galaxy_detection/data/darknet/src/connected_layer.c:    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
galaxy_detection/data/darknet/src/connected_layer.c:    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
galaxy_detection/data/darknet/src/connected_layer.c:        cuda_pull_array(l.scales_gpu, l.scales, l.outputs);
galaxy_detection/data/darknet/src/connected_layer.c:        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
galaxy_detection/data/darknet/src/connected_layer.c:        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
galaxy_detection/data/darknet/src/connected_layer.c:    cuda_push_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
galaxy_detection/data/darknet/src/connected_layer.c:    cuda_push_array(l.biases_gpu, l.biases, l.outputs);
galaxy_detection/data/darknet/src/connected_layer.c:    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
galaxy_detection/data/darknet/src/connected_layer.c:    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
galaxy_detection/data/darknet/src/connected_layer.c:        cuda_push_array(l.scales_gpu, l.scales, l.outputs);
galaxy_detection/data/darknet/src/connected_layer.c:        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
galaxy_detection/data/darknet/src/connected_layer.c:        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
galaxy_detection/data/darknet/src/connected_layer.c:void update_connected_layer_gpu(layer l, update_args a)
galaxy_detection/data/darknet/src/connected_layer.c:        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.inputs*l.outputs, batch, a.t);
galaxy_detection/data/darknet/src/connected_layer.c:        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
galaxy_detection/data/darknet/src/connected_layer.c:        if(l.scales_gpu){
galaxy_detection/data/darknet/src/connected_layer.c:            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
galaxy_detection/data/darknet/src/connected_layer.c:        axpy_gpu(l.outputs, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
galaxy_detection/data/darknet/src/connected_layer.c:        scal_gpu(l.outputs, momentum, l.bias_updates_gpu, 1);
galaxy_detection/data/darknet/src/connected_layer.c:            axpy_gpu(l.outputs, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
galaxy_detection/data/darknet/src/connected_layer.c:            scal_gpu(l.outputs, momentum, l.scale_updates_gpu, 1);
galaxy_detection/data/darknet/src/connected_layer.c:        axpy_gpu(l.inputs*l.outputs, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
galaxy_detection/data/darknet/src/connected_layer.c:        axpy_gpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
galaxy_detection/data/darknet/src/connected_layer.c:        scal_gpu(l.inputs*l.outputs, momentum, l.weight_updates_gpu, 1);
galaxy_detection/data/darknet/src/connected_layer.c:void forward_connected_layer_gpu(layer l, network net)
galaxy_detection/data/darknet/src/connected_layer.c:    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
galaxy_detection/data/darknet/src/connected_layer.c:    float * a = net.input_gpu;
galaxy_detection/data/darknet/src/connected_layer.c:    float * b = l.weights_gpu;
galaxy_detection/data/darknet/src/connected_layer.c:    float * c = l.output_gpu;
galaxy_detection/data/darknet/src/connected_layer.c:    gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);
galaxy_detection/data/darknet/src/connected_layer.c:        forward_batchnorm_layer_gpu(l, net);
galaxy_detection/data/darknet/src/connected_layer.c:        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.outputs, 1);
galaxy_detection/data/darknet/src/connected_layer.c:    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
galaxy_detection/data/darknet/src/connected_layer.c:void backward_connected_layer_gpu(layer l, network net)
galaxy_detection/data/darknet/src/connected_layer.c:    constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
galaxy_detection/data/darknet/src/connected_layer.c:    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
galaxy_detection/data/darknet/src/connected_layer.c:        backward_batchnorm_layer_gpu(l, net);
galaxy_detection/data/darknet/src/connected_layer.c:        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.outputs, 1);
galaxy_detection/data/darknet/src/connected_layer.c:    float * a = l.delta_gpu;
galaxy_detection/data/darknet/src/connected_layer.c:    float * b = net.input_gpu;
galaxy_detection/data/darknet/src/connected_layer.c:    float * c = l.weight_updates_gpu;
galaxy_detection/data/darknet/src/connected_layer.c:    gemm_gpu(1,0,m,n,k,1,a,m,b,n,1,c,n);
galaxy_detection/data/darknet/src/connected_layer.c:    a = l.delta_gpu;
galaxy_detection/data/darknet/src/connected_layer.c:    b = l.weights_gpu;
galaxy_detection/data/darknet/src/connected_layer.c:    c = net.delta_gpu;
galaxy_detection/data/darknet/src/connected_layer.c:    if(c) gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
galaxy_detection/data/darknet/src/batchnorm_layer.h:#ifdef GPU
galaxy_detection/data/darknet/src/batchnorm_layer.h:void forward_batchnorm_layer_gpu(layer l, network net);
galaxy_detection/data/darknet/src/batchnorm_layer.h:void backward_batchnorm_layer_gpu(layer l, network net);
galaxy_detection/data/darknet/src/route_layer.h:#ifdef GPU
galaxy_detection/data/darknet/src/route_layer.h:void forward_route_layer_gpu(const route_layer l, network net);
galaxy_detection/data/darknet/src/route_layer.h:void backward_route_layer_gpu(const route_layer l, network net);
galaxy_detection/data/darknet/src/region_layer.c:#include "cuda.h"
galaxy_detection/data/darknet/src/region_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/region_layer.c:    l.forward_gpu = forward_region_layer_gpu;
galaxy_detection/data/darknet/src/region_layer.c:    l.backward_gpu = backward_region_layer_gpu;
galaxy_detection/data/darknet/src/region_layer.c:    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
galaxy_detection/data/darknet/src/region_layer.c:    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
galaxy_detection/data/darknet/src/region_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/region_layer.c:    cuda_free(l->delta_gpu);
galaxy_detection/data/darknet/src/region_layer.c:    cuda_free(l->output_gpu);
galaxy_detection/data/darknet/src/region_layer.c:    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
galaxy_detection/data/darknet/src/region_layer.c:    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
galaxy_detection/data/darknet/src/region_layer.c:#ifndef GPU
galaxy_detection/data/darknet/src/region_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/region_layer.c:void forward_region_layer_gpu(const layer l, network net)
galaxy_detection/data/darknet/src/region_layer.c:    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
galaxy_detection/data/darknet/src/region_layer.c:            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
galaxy_detection/data/darknet/src/region_layer.c:                activate_array_gpu(l.output_gpu + index, (l.coords - 4)*l.w*l.h, LOGISTIC);
galaxy_detection/data/darknet/src/region_layer.c:            if(!l.background) activate_array_gpu(l.output_gpu + index,   l.w*l.h, LOGISTIC);
galaxy_detection/data/darknet/src/region_layer.c:            if(!l.softmax && !l.softmax_tree) activate_array_gpu(l.output_gpu + index, l.classes*l.w*l.h, LOGISTIC);
galaxy_detection/data/darknet/src/region_layer.c:        softmax_tree(net.input_gpu + index, l.w*l.h, l.batch*l.n, l.inputs/l.n, 1, l.output_gpu + index, *l.softmax_tree);
galaxy_detection/data/darknet/src/region_layer.c:        softmax_tree(net.input_gpu + index, l.w*l.h, l.batch*l.n, l.inputs/l.n, 1, l.output_gpu + index, *l.softmax_tree);
galaxy_detection/data/darknet/src/region_layer.c:        cudaDeviceSynchronize();
galaxy_detection/data/darknet/src/region_layer.c:        printf("Good GPU Timing: %f\n", what_time_is_it_now() - then);
galaxy_detection/data/darknet/src/region_layer.c:        softmax_gpu(net.input_gpu + index, group_size, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu + index);
galaxy_detection/data/darknet/src/region_layer.c:        cudaDeviceSynchronize();
galaxy_detection/data/darknet/src/region_layer.c:        printf("Bad GPU Timing: %f\n", what_time_is_it_now() - then);
galaxy_detection/data/darknet/src/region_layer.c:        cudaDeviceSynchronize();
galaxy_detection/data/darknet/src/region_layer.c:           softmax_gpu(net.input_gpu + index, group_size, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu + index);
galaxy_detection/data/darknet/src/region_layer.c:        softmax_gpu(net.input_gpu + index, l.classes + l.background, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu + index);
galaxy_detection/data/darknet/src/region_layer.c:        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
galaxy_detection/data/darknet/src/region_layer.c:    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
galaxy_detection/data/darknet/src/region_layer.c:    //cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
galaxy_detection/data/darknet/src/region_layer.c:    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
galaxy_detection/data/darknet/src/region_layer.c:void backward_region_layer_gpu(const layer l, network net)
galaxy_detection/data/darknet/src/region_layer.c:            gradient_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC, l.delta_gpu + index);
galaxy_detection/data/darknet/src/region_layer.c:                gradient_array_gpu(l.output_gpu + index, (l.coords - 4)*l.w*l.h, LOGISTIC, l.delta_gpu + index);
galaxy_detection/data/darknet/src/region_layer.c:            if(!l.background) gradient_array_gpu(l.output_gpu + index,   l.w*l.h, LOGISTIC, l.delta_gpu + index);
galaxy_detection/data/darknet/src/region_layer.c:    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
galaxy_detection/data/darknet/src/normalization_layer.c:    #ifdef GPU
galaxy_detection/data/darknet/src/normalization_layer.c:    layer.forward_gpu = forward_normalization_layer_gpu;
galaxy_detection/data/darknet/src/normalization_layer.c:    layer.backward_gpu = backward_normalization_layer_gpu;
galaxy_detection/data/darknet/src/normalization_layer.c:    layer.output_gpu =  cuda_make_array(layer.output, h * w * c * batch);
galaxy_detection/data/darknet/src/normalization_layer.c:    layer.delta_gpu =   cuda_make_array(layer.delta, h * w * c * batch);
galaxy_detection/data/darknet/src/normalization_layer.c:    layer.squared_gpu = cuda_make_array(layer.squared, h * w * c * batch);
galaxy_detection/data/darknet/src/normalization_layer.c:    layer.norms_gpu =   cuda_make_array(layer.norms, h * w * c * batch);
galaxy_detection/data/darknet/src/normalization_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/normalization_layer.c:    cuda_free(layer->output_gpu);
galaxy_detection/data/darknet/src/normalization_layer.c:    cuda_free(layer->delta_gpu); 
galaxy_detection/data/darknet/src/normalization_layer.c:    cuda_free(layer->squared_gpu); 
galaxy_detection/data/darknet/src/normalization_layer.c:    cuda_free(layer->norms_gpu);   
galaxy_detection/data/darknet/src/normalization_layer.c:    layer->output_gpu =  cuda_make_array(layer->output, h * w * c * batch);
galaxy_detection/data/darknet/src/normalization_layer.c:    layer->delta_gpu =   cuda_make_array(layer->delta, h * w * c * batch);
galaxy_detection/data/darknet/src/normalization_layer.c:    layer->squared_gpu = cuda_make_array(layer->squared, h * w * c * batch);
galaxy_detection/data/darknet/src/normalization_layer.c:    layer->norms_gpu =   cuda_make_array(layer->norms, h * w * c * batch);
galaxy_detection/data/darknet/src/normalization_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/normalization_layer.c:void forward_normalization_layer_gpu(const layer layer, network net)
galaxy_detection/data/darknet/src/normalization_layer.c:    scal_gpu(w*h*c*layer.batch, 0, layer.squared_gpu, 1);
galaxy_detection/data/darknet/src/normalization_layer.c:        float *squared = layer.squared_gpu + w*h*c*b;
galaxy_detection/data/darknet/src/normalization_layer.c:        float *norms   = layer.norms_gpu + w*h*c*b;
galaxy_detection/data/darknet/src/normalization_layer.c:        float *input   = net.input_gpu + w*h*c*b;
galaxy_detection/data/darknet/src/normalization_layer.c:        pow_gpu(w*h*c, 2, input, 1, squared, 1);
galaxy_detection/data/darknet/src/normalization_layer.c:        const_gpu(w*h, layer.kappa, norms, 1);
galaxy_detection/data/darknet/src/normalization_layer.c:            axpy_gpu(w*h, layer.alpha, squared + w*h*k, 1, norms, 1);
galaxy_detection/data/darknet/src/normalization_layer.c:            copy_gpu(w*h, norms + w*h*(k-1), 1, norms + w*h*k, 1);
galaxy_detection/data/darknet/src/normalization_layer.c:            if(prev >= 0)      axpy_gpu(w*h, -layer.alpha, squared + w*h*prev, 1, norms + w*h*k, 1);
galaxy_detection/data/darknet/src/normalization_layer.c:            if(next < layer.c) axpy_gpu(w*h,  layer.alpha, squared + w*h*next, 1, norms + w*h*k, 1);
galaxy_detection/data/darknet/src/normalization_layer.c:    pow_gpu(w*h*c*layer.batch, -layer.beta, layer.norms_gpu, 1, layer.output_gpu, 1);
galaxy_detection/data/darknet/src/normalization_layer.c:    mul_gpu(w*h*c*layer.batch, net.input_gpu, 1, layer.output_gpu, 1);
galaxy_detection/data/darknet/src/normalization_layer.c:void backward_normalization_layer_gpu(const layer layer, network net)
galaxy_detection/data/darknet/src/normalization_layer.c:    pow_gpu(w*h*c*layer.batch, -layer.beta, layer.norms_gpu, 1, net.delta_gpu, 1);
galaxy_detection/data/darknet/src/normalization_layer.c:    mul_gpu(w*h*c*layer.batch, layer.delta_gpu, 1, net.delta_gpu, 1);
galaxy_detection/data/darknet/src/gemm.c:#include "cuda.h"
galaxy_detection/data/darknet/src/gemm.c:#ifdef GPU
galaxy_detection/data/darknet/src/gemm.c:void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
galaxy_detection/data/darknet/src/gemm.c:        float *A_gpu, int lda, 
galaxy_detection/data/darknet/src/gemm.c:        float *B_gpu, int ldb,
galaxy_detection/data/darknet/src/gemm.c:        float *C_gpu, int ldc)
galaxy_detection/data/darknet/src/gemm.c:    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
galaxy_detection/data/darknet/src/gemm.c:            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
galaxy_detection/data/darknet/src/gemm.c:void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
galaxy_detection/data/darknet/src/gemm.c:        gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
galaxy_detection/data/darknet/src/gemm.c:void time_gpu(int TA, int TB, int m, int k, int n)
galaxy_detection/data/darknet/src/gemm.c:    float *a_cl = cuda_make_array(a, m*k);
galaxy_detection/data/darknet/src/gemm.c:    float *b_cl = cuda_make_array(b, k*n);
galaxy_detection/data/darknet/src/gemm.c:    float *c_cl = cuda_make_array(c, m*n);
galaxy_detection/data/darknet/src/gemm.c:        gemm_gpu(TA,TB,m,n,k,1,a_cl,lda,b_cl,ldb,1,c_cl,n);
galaxy_detection/data/darknet/src/gemm.c:        cudaThreadSynchronize();
galaxy_detection/data/darknet/src/gemm.c:    cuda_free(a_cl);
galaxy_detection/data/darknet/src/gemm.c:    cuda_free(b_cl);
galaxy_detection/data/darknet/src/gemm.c:    cuda_free(c_cl);
galaxy_detection/data/darknet/src/gemm.c:void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
galaxy_detection/data/darknet/src/gemm.c:    float *c_gpu = random_matrix(m,n);
galaxy_detection/data/darknet/src/gemm.c:    memset(c_gpu, 0, m*n*sizeof(float));
galaxy_detection/data/darknet/src/gemm.c:    gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c_gpu,n);
galaxy_detection/data/darknet/src/gemm.c:    //printf("GPU\n");
galaxy_detection/data/darknet/src/gemm.c:    //pm(m, n, c_gpu);
galaxy_detection/data/darknet/src/gemm.c:        //printf("%f %f\n", c[i], c_gpu[i]);
galaxy_detection/data/darknet/src/gemm.c:        sse += pow(c[i]-c_gpu[i], 2);
galaxy_detection/data/darknet/src/gemm.c:    free(c_gpu);
galaxy_detection/data/darknet/src/gemm.c:int test_gpu_blas()
galaxy_detection/data/darknet/src/gemm.c:       test_gpu_accuracy(0,0,10,576,75); 
galaxy_detection/data/darknet/src/gemm.c:       test_gpu_accuracy(0,0,17,10,10); 
galaxy_detection/data/darknet/src/gemm.c:       test_gpu_accuracy(1,0,17,10,10); 
galaxy_detection/data/darknet/src/gemm.c:       test_gpu_accuracy(0,1,17,10,10); 
galaxy_detection/data/darknet/src/gemm.c:       test_gpu_accuracy(1,1,17,10,10); 
galaxy_detection/data/darknet/src/gemm.c:       test_gpu_accuracy(0,0,1000,10,100); 
galaxy_detection/data/darknet/src/gemm.c:       test_gpu_accuracy(1,0,1000,10,100); 
galaxy_detection/data/darknet/src/gemm.c:       test_gpu_accuracy(0,1,1000,10,100); 
galaxy_detection/data/darknet/src/gemm.c:       test_gpu_accuracy(1,1,1000,10,100); 
galaxy_detection/data/darknet/src/gemm.c:       test_gpu_accuracy(0,0,10,10,10); 
galaxy_detection/data/darknet/src/gemm.c:       time_gpu(0,0,64,2916,363); 
galaxy_detection/data/darknet/src/gemm.c:       time_gpu(0,0,64,2916,363); 
galaxy_detection/data/darknet/src/gemm.c:       time_gpu(0,0,64,2916,363); 
galaxy_detection/data/darknet/src/gemm.c:       time_gpu(0,0,192,729,1600); 
galaxy_detection/data/darknet/src/gemm.c:       time_gpu(0,0,384,196,1728); 
galaxy_detection/data/darknet/src/gemm.c:       time_gpu(0,0,256,196,3456); 
galaxy_detection/data/darknet/src/gemm.c:       time_gpu(0,0,256,196,2304); 
galaxy_detection/data/darknet/src/gemm.c:       time_gpu(0,0,128,4096,12544); 
galaxy_detection/data/darknet/src/gemm.c:       time_gpu(0,0,128,4096,4096); 
galaxy_detection/data/darknet/src/gemm.c:    time_gpu(0,0,64,75,12544); 
galaxy_detection/data/darknet/src/gemm.c:    time_gpu(0,0,64,75,12544); 
galaxy_detection/data/darknet/src/gemm.c:    time_gpu(0,0,64,75,12544); 
galaxy_detection/data/darknet/src/gemm.c:    time_gpu(0,0,64,576,12544); 
galaxy_detection/data/darknet/src/gemm.c:    time_gpu(0,0,256,2304,784); 
galaxy_detection/data/darknet/src/gemm.c:    time_gpu(1,1,2304,256,784); 
galaxy_detection/data/darknet/src/gemm.c:    time_gpu(0,0,512,4608,196); 
galaxy_detection/data/darknet/src/gemm.c:    time_gpu(1,1,4608,512,196); 
galaxy_detection/data/darknet/src/activation_layer.c:#include "cuda.h"
galaxy_detection/data/darknet/src/activation_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/activation_layer.c:    l.forward_gpu = forward_activation_layer_gpu;
galaxy_detection/data/darknet/src/activation_layer.c:    l.backward_gpu = backward_activation_layer_gpu;
galaxy_detection/data/darknet/src/activation_layer.c:    l.output_gpu = cuda_make_array(l.output, inputs*batch);
galaxy_detection/data/darknet/src/activation_layer.c:    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
galaxy_detection/data/darknet/src/activation_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/activation_layer.c:void forward_activation_layer_gpu(layer l, network net)
galaxy_detection/data/darknet/src/activation_layer.c:    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
galaxy_detection/data/darknet/src/activation_layer.c:    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
galaxy_detection/data/darknet/src/activation_layer.c:void backward_activation_layer_gpu(layer l, network net)
galaxy_detection/data/darknet/src/activation_layer.c:    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
galaxy_detection/data/darknet/src/activation_layer.c:    copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, net.delta_gpu, 1);
galaxy_detection/data/darknet/src/cost_layer.h:#ifdef GPU
galaxy_detection/data/darknet/src/cost_layer.h:void forward_cost_layer_gpu(cost_layer l, network net);
galaxy_detection/data/darknet/src/cost_layer.h:void backward_cost_layer_gpu(const cost_layer l, network net);
galaxy_detection/data/darknet/src/detection_layer.c:#include "cuda.h"
galaxy_detection/data/darknet/src/detection_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/detection_layer.c:    l.forward_gpu = forward_detection_layer_gpu;
galaxy_detection/data/darknet/src/detection_layer.c:    l.backward_gpu = backward_detection_layer_gpu;
galaxy_detection/data/darknet/src/detection_layer.c:    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
galaxy_detection/data/darknet/src/detection_layer.c:    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
galaxy_detection/data/darknet/src/detection_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/detection_layer.c:void forward_detection_layer_gpu(const detection_layer l, network net)
galaxy_detection/data/darknet/src/detection_layer.c:        copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
galaxy_detection/data/darknet/src/detection_layer.c:    cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
galaxy_detection/data/darknet/src/detection_layer.c:    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
galaxy_detection/data/darknet/src/detection_layer.c:void backward_detection_layer_gpu(detection_layer l, network net)
galaxy_detection/data/darknet/src/detection_layer.c:    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
galaxy_detection/data/darknet/src/detection_layer.c:    //copy_gpu(l.batch*l.inputs, l.delta_gpu, 1, net.delta_gpu, 1);
galaxy_detection/data/darknet/src/local_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/local_layer.c:    l.forward_gpu = forward_local_layer_gpu;
galaxy_detection/data/darknet/src/local_layer.c:    l.backward_gpu = backward_local_layer_gpu;
galaxy_detection/data/darknet/src/local_layer.c:    l.update_gpu = update_local_layer_gpu;
galaxy_detection/data/darknet/src/local_layer.c:    l.weights_gpu = cuda_make_array(l.weights, c*n*size*size*locations);
galaxy_detection/data/darknet/src/local_layer.c:    l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*n*size*size*locations);
galaxy_detection/data/darknet/src/local_layer.c:    l.biases_gpu = cuda_make_array(l.biases, l.outputs);
galaxy_detection/data/darknet/src/local_layer.c:    l.bias_updates_gpu = cuda_make_array(l.bias_updates, l.outputs);
galaxy_detection/data/darknet/src/local_layer.c:    l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
galaxy_detection/data/darknet/src/local_layer.c:    l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
galaxy_detection/data/darknet/src/local_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/local_layer.c:void forward_local_layer_gpu(const local_layer l, network net)
galaxy_detection/data/darknet/src/local_layer.c:        copy_gpu(l.outputs, l.biases_gpu, 1, l.output_gpu + i*l.outputs, 1);
galaxy_detection/data/darknet/src/local_layer.c:        float *input = net.input_gpu + i*l.w*l.h*l.c;
galaxy_detection/data/darknet/src/local_layer.c:        im2col_gpu(input, l.c, l.h, l.w, 
galaxy_detection/data/darknet/src/local_layer.c:        float *output = l.output_gpu + i*l.outputs;
galaxy_detection/data/darknet/src/local_layer.c:            float *a = l.weights_gpu + j*l.size*l.size*l.c*l.n;
galaxy_detection/data/darknet/src/local_layer.c:            gemm_gpu(0,0,m,n,k,1,a,k,b,locations,1,c,locations);
galaxy_detection/data/darknet/src/local_layer.c:    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
galaxy_detection/data/darknet/src/local_layer.c:void backward_local_layer_gpu(local_layer l, network net)
galaxy_detection/data/darknet/src/local_layer.c:    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
galaxy_detection/data/darknet/src/local_layer.c:        axpy_gpu(l.outputs, 1, l.delta_gpu + i*l.outputs, 1, l.bias_updates_gpu, 1);
galaxy_detection/data/darknet/src/local_layer.c:        float *input = net.input_gpu + i*l.w*l.h*l.c;
galaxy_detection/data/darknet/src/local_layer.c:        im2col_gpu(input, l.c, l.h, l.w, 
galaxy_detection/data/darknet/src/local_layer.c:            float *a = l.delta_gpu + i*l.outputs + j;
galaxy_detection/data/darknet/src/local_layer.c:            float *c = l.weight_updates_gpu + j*l.size*l.size*l.c*l.n;
galaxy_detection/data/darknet/src/local_layer.c:            gemm_gpu(0,1,m,n,k,1,a,locations,b,locations,1,c,n);
galaxy_detection/data/darknet/src/local_layer.c:        if(net.delta_gpu){
galaxy_detection/data/darknet/src/local_layer.c:                float *a = l.weights_gpu + j*l.size*l.size*l.c*l.n;
galaxy_detection/data/darknet/src/local_layer.c:                float *b = l.delta_gpu + i*l.outputs + j;
galaxy_detection/data/darknet/src/local_layer.c:                gemm_gpu(1,0,m,n,k,1,a,m,b,locations,0,c,locations);
galaxy_detection/data/darknet/src/local_layer.c:            col2im_gpu(net.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, net.delta_gpu+i*l.c*l.h*l.w);
galaxy_detection/data/darknet/src/local_layer.c:void update_local_layer_gpu(local_layer l, update_args a)
galaxy_detection/data/darknet/src/local_layer.c:    axpy_gpu(l.outputs, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
galaxy_detection/data/darknet/src/local_layer.c:    scal_gpu(l.outputs, momentum, l.bias_updates_gpu, 1);
galaxy_detection/data/darknet/src/local_layer.c:    axpy_gpu(size, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
galaxy_detection/data/darknet/src/local_layer.c:    axpy_gpu(size, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
galaxy_detection/data/darknet/src/local_layer.c:    scal_gpu(size, momentum, l.weight_updates_gpu, 1);
galaxy_detection/data/darknet/src/local_layer.c:    cuda_pull_array(l.weights_gpu, l.weights, size);
galaxy_detection/data/darknet/src/local_layer.c:    cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
galaxy_detection/data/darknet/src/local_layer.c:    cuda_push_array(l.weights_gpu, l.weights, size);
galaxy_detection/data/darknet/src/local_layer.c:    cuda_push_array(l.biases_gpu, l.biases, l.outputs);
galaxy_detection/data/darknet/src/col2im_kernels.cu:#include "cuda_runtime.h"
galaxy_detection/data/darknet/src/col2im_kernels.cu:#include "cuda.h"
galaxy_detection/data/darknet/src/col2im_kernels.cu:__global__ void col2im_gpu_kernel(const int n, const float* data_col,
galaxy_detection/data/darknet/src/col2im_kernels.cu:void col2im_gpu(float *data_col,
galaxy_detection/data/darknet/src/col2im_kernels.cu:    col2im_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
galaxy_detection/data/darknet/src/blas.h:void constrain_gpu(int N, float ALPHA, float * X, int INCX);
galaxy_detection/data/darknet/src/blas.h:int test_gpu_blas();
galaxy_detection/data/darknet/src/blas.h:#ifdef GPU
galaxy_detection/data/darknet/src/blas.h:#include "cuda.h"
galaxy_detection/data/darknet/src/blas.h:void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY);
galaxy_detection/data/darknet/src/blas.h:void axpy_gpu_offset(int N, float ALPHA, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY);
galaxy_detection/data/darknet/src/blas.h:void copy_gpu(int N, float * X, int INCX, float * Y, int INCY);
galaxy_detection/data/darknet/src/blas.h:void copy_gpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY);
galaxy_detection/data/darknet/src/blas.h:void add_gpu(int N, float ALPHA, float * X, int INCX);
galaxy_detection/data/darknet/src/blas.h:void supp_gpu(int N, float ALPHA, float * X, int INCX);
galaxy_detection/data/darknet/src/blas.h:void mask_gpu(int N, float * X, float mask_num, float * mask);
galaxy_detection/data/darknet/src/blas.h:void scale_mask_gpu(int N, float * X, float mask_num, float * mask, float scale);
galaxy_detection/data/darknet/src/blas.h:void const_gpu(int N, float ALPHA, float *X, int INCX);
galaxy_detection/data/darknet/src/blas.h:void pow_gpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
galaxy_detection/data/darknet/src/blas.h:void mul_gpu(int N, float *X, int INCX, float *Y, int INCY);
galaxy_detection/data/darknet/src/blas.h:void mean_gpu(float *x, int batch, int filters, int spatial, float *mean);
galaxy_detection/data/darknet/src/blas.h:void variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
galaxy_detection/data/darknet/src/blas.h:void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
galaxy_detection/data/darknet/src/blas.h:void normalize_delta_gpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta);
galaxy_detection/data/darknet/src/blas.h:void fast_mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
galaxy_detection/data/darknet/src/blas.h:void fast_variance_delta_gpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta);
galaxy_detection/data/darknet/src/blas.h:void fast_variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
galaxy_detection/data/darknet/src/blas.h:void fast_mean_gpu(float *x, int batch, int filters, int spatial, float *mean);
galaxy_detection/data/darknet/src/blas.h:void shortcut_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out);
galaxy_detection/data/darknet/src/blas.h:void scale_bias_gpu(float *output, float *biases, int batch, int n, int size);
galaxy_detection/data/darknet/src/blas.h:void backward_scale_gpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);
galaxy_detection/data/darknet/src/blas.h:void scale_bias_gpu(float *output, float *biases, int batch, int n, int size);
galaxy_detection/data/darknet/src/blas.h:void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
galaxy_detection/data/darknet/src/blas.h:void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);
galaxy_detection/data/darknet/src/blas.h:void smooth_l1_gpu(int n, float *pred, float *truth, float *delta, float *error);
galaxy_detection/data/darknet/src/blas.h:void l2_gpu(int n, float *pred, float *truth, float *delta, float *error);
galaxy_detection/data/darknet/src/blas.h:void l1_gpu(int n, float *pred, float *truth, float *delta, float *error);
galaxy_detection/data/darknet/src/blas.h:void weighted_delta_gpu(float *a, float *b, float *s, float *da, float *db, float *ds, int num, float *dc);
galaxy_detection/data/darknet/src/blas.h:void weighted_sum_gpu(float *a, float *b, float *s, int num, float *c);
galaxy_detection/data/darknet/src/blas.h:void mult_add_into_gpu(int num, float *a, float *b, float *c);
galaxy_detection/data/darknet/src/blas.h:void inter_gpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
galaxy_detection/data/darknet/src/blas.h:void deinter_gpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
galaxy_detection/data/darknet/src/blas.h:void reorg_gpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out);
galaxy_detection/data/darknet/src/blas.h:void softmax_gpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);
galaxy_detection/data/darknet/src/blas.h:void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t);
galaxy_detection/data/darknet/src/blas.h:void adam_gpu(int n, float *x, float *m, float *v, float B1, float B2, float rate, float eps, int t);
galaxy_detection/data/darknet/src/blas.h:void flatten_gpu(float *x, int spatial, int layers, int batch, int forward, float *out);
galaxy_detection/data/darknet/src/dropout_layer.c:#include "cuda.h"
galaxy_detection/data/darknet/src/dropout_layer.c:    #ifdef GPU
galaxy_detection/data/darknet/src/dropout_layer.c:    l.forward_gpu = forward_dropout_layer_gpu;
galaxy_detection/data/darknet/src/dropout_layer.c:    l.backward_gpu = backward_dropout_layer_gpu;
galaxy_detection/data/darknet/src/dropout_layer.c:    l.rand_gpu = cuda_make_array(l.rand, inputs*batch);
galaxy_detection/data/darknet/src/dropout_layer.c:    #ifdef GPU
galaxy_detection/data/darknet/src/dropout_layer.c:    cuda_free(l->rand_gpu);
galaxy_detection/data/darknet/src/dropout_layer.c:    l->rand_gpu = cuda_make_array(l->rand, inputs*l->batch);
galaxy_detection/data/darknet/src/data.c:#include "cuda.h"
galaxy_detection/data/darknet/src/layer.c:#include "cuda.h"
galaxy_detection/data/darknet/src/layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/layer.c:        if(l.rand_gpu)             cuda_free(l.rand_gpu);
galaxy_detection/data/darknet/src/layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/layer.c:    if(l.indexes_gpu)           cuda_free((float *)l.indexes_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.z_gpu)                   cuda_free(l.z_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.r_gpu)                   cuda_free(l.r_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.h_gpu)                   cuda_free(l.h_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.m_gpu)                   cuda_free(l.m_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.v_gpu)                   cuda_free(l.v_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.prev_state_gpu)          cuda_free(l.prev_state_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.forgot_state_gpu)        cuda_free(l.forgot_state_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.forgot_delta_gpu)        cuda_free(l.forgot_delta_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.state_gpu)               cuda_free(l.state_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.state_delta_gpu)         cuda_free(l.state_delta_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.gate_gpu)                cuda_free(l.gate_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.gate_delta_gpu)          cuda_free(l.gate_delta_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.save_gpu)                cuda_free(l.save_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.save_delta_gpu)          cuda_free(l.save_delta_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.concat_gpu)              cuda_free(l.concat_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.concat_delta_gpu)        cuda_free(l.concat_delta_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.binary_input_gpu)        cuda_free(l.binary_input_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.binary_weights_gpu)      cuda_free(l.binary_weights_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.mean_gpu)                cuda_free(l.mean_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.variance_gpu)            cuda_free(l.variance_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.rolling_mean_gpu)        cuda_free(l.rolling_mean_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.rolling_variance_gpu)    cuda_free(l.rolling_variance_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.variance_delta_gpu)      cuda_free(l.variance_delta_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.mean_delta_gpu)          cuda_free(l.mean_delta_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.x_gpu)                   cuda_free(l.x_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.x_norm_gpu)              cuda_free(l.x_norm_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.weights_gpu)             cuda_free(l.weights_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.weight_updates_gpu)      cuda_free(l.weight_updates_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.biases_gpu)              cuda_free(l.biases_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.bias_updates_gpu)        cuda_free(l.bias_updates_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.scales_gpu)              cuda_free(l.scales_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.scale_updates_gpu)       cuda_free(l.scale_updates_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.output_gpu)              cuda_free(l.output_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.delta_gpu)               cuda_free(l.delta_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.rand_gpu)                cuda_free(l.rand_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.squared_gpu)             cuda_free(l.squared_gpu);
galaxy_detection/data/darknet/src/layer.c:    if(l.norms_gpu)               cuda_free(l.norms_gpu);
galaxy_detection/data/darknet/src/parser.c:    net->gpu_index = gpu_index;
galaxy_detection/data/darknet/src/parser.c:#ifdef GPU
galaxy_detection/data/darknet/src/parser.c:            l.output_gpu = net->layers[count-1].output_gpu;
galaxy_detection/data/darknet/src/parser.c:            l.delta_gpu = net->layers[count-1].delta_gpu;
galaxy_detection/data/darknet/src/parser.c:#ifdef GPU
galaxy_detection/data/darknet/src/parser.c:    net->output_gpu = out.output_gpu;
galaxy_detection/data/darknet/src/parser.c:    net->input_gpu = cuda_make_array(net->input, net->inputs*net->batch);
galaxy_detection/data/darknet/src/parser.c:    net->truth_gpu = cuda_make_array(net->truth, net->truths*net->batch);
galaxy_detection/data/darknet/src/parser.c:#ifdef GPU
galaxy_detection/data/darknet/src/parser.c:        if(gpu_index >= 0){
galaxy_detection/data/darknet/src/parser.c:            net->workspace = cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
galaxy_detection/data/darknet/src/parser.c:#ifdef GPU
galaxy_detection/data/darknet/src/parser.c:    if(gpu_index >= 0){
galaxy_detection/data/darknet/src/parser.c:#ifdef GPU
galaxy_detection/data/darknet/src/parser.c:    if(gpu_index >= 0){
galaxy_detection/data/darknet/src/parser.c:#ifdef GPU
galaxy_detection/data/darknet/src/parser.c:    if(gpu_index >= 0){
galaxy_detection/data/darknet/src/parser.c:#ifdef GPU
galaxy_detection/data/darknet/src/parser.c:    if(gpu_index >= 0){
galaxy_detection/data/darknet/src/parser.c:#ifdef GPU
galaxy_detection/data/darknet/src/parser.c:    if(net->gpu_index >= 0){
galaxy_detection/data/darknet/src/parser.c:        cuda_set_device(net->gpu_index);
galaxy_detection/data/darknet/src/parser.c:#ifdef GPU
galaxy_detection/data/darknet/src/parser.c:            if(gpu_index >= 0){
galaxy_detection/data/darknet/src/parser.c:#ifdef GPU
galaxy_detection/data/darknet/src/parser.c:    if(gpu_index >= 0){
galaxy_detection/data/darknet/src/parser.c:#ifdef GPU
galaxy_detection/data/darknet/src/parser.c:    if(gpu_index >= 0){
galaxy_detection/data/darknet/src/parser.c:#ifdef GPU
galaxy_detection/data/darknet/src/parser.c:    if(gpu_index >= 0){
galaxy_detection/data/darknet/src/parser.c:#ifdef GPU
galaxy_detection/data/darknet/src/parser.c:    if(gpu_index >= 0){
galaxy_detection/data/darknet/src/parser.c:#ifdef GPU
galaxy_detection/data/darknet/src/parser.c:    if(net->gpu_index >= 0){
galaxy_detection/data/darknet/src/parser.c:        cuda_set_device(net->gpu_index);
galaxy_detection/data/darknet/src/parser.c:#ifdef GPU
galaxy_detection/data/darknet/src/parser.c:            if(gpu_index >= 0){
galaxy_detection/data/darknet/src/activation_kernels.cu:#include "cuda_runtime.h"
galaxy_detection/data/darknet/src/activation_kernels.cu:#include "cuda.h"
galaxy_detection/data/darknet/src/activation_kernels.cu:extern "C" void activate_array_gpu(float *x, int n, ACTIVATION a) 
galaxy_detection/data/darknet/src/activation_kernels.cu:    activate_array_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, a);
galaxy_detection/data/darknet/src/activation_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/activation_kernels.cu:extern "C" void gradient_array_gpu(float *x, int n, ACTIVATION a, float *delta) 
galaxy_detection/data/darknet/src/activation_kernels.cu:    gradient_array_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, a, delta);
galaxy_detection/data/darknet/src/activation_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/maxpool_layer.c:#include "cuda.h"
galaxy_detection/data/darknet/src/maxpool_layer.c:    #ifdef GPU
galaxy_detection/data/darknet/src/maxpool_layer.c:    l.forward_gpu = forward_maxpool_layer_gpu;
galaxy_detection/data/darknet/src/maxpool_layer.c:    l.backward_gpu = backward_maxpool_layer_gpu;
galaxy_detection/data/darknet/src/maxpool_layer.c:    l.indexes_gpu = cuda_make_int_array(0, output_size);
galaxy_detection/data/darknet/src/maxpool_layer.c:    l.output_gpu  = cuda_make_array(l.output, output_size);
galaxy_detection/data/darknet/src/maxpool_layer.c:    l.delta_gpu   = cuda_make_array(l.delta, output_size);
galaxy_detection/data/darknet/src/maxpool_layer.c:    #ifdef GPU
galaxy_detection/data/darknet/src/maxpool_layer.c:    cuda_free((float *)l->indexes_gpu);
galaxy_detection/data/darknet/src/maxpool_layer.c:    cuda_free(l->output_gpu);
galaxy_detection/data/darknet/src/maxpool_layer.c:    cuda_free(l->delta_gpu);
galaxy_detection/data/darknet/src/maxpool_layer.c:    l->indexes_gpu = cuda_make_int_array(0, output_size);
galaxy_detection/data/darknet/src/maxpool_layer.c:    l->output_gpu  = cuda_make_array(l->output, output_size);
galaxy_detection/data/darknet/src/maxpool_layer.c:    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
galaxy_detection/data/darknet/src/crop_layer.h:#ifdef GPU
galaxy_detection/data/darknet/src/crop_layer.h:void forward_crop_layer_gpu(crop_layer l, network net);
galaxy_detection/data/darknet/src/dropout_layer_kernels.cu:#include "cuda_runtime.h"
galaxy_detection/data/darknet/src/dropout_layer_kernels.cu:#include "cuda.h"
galaxy_detection/data/darknet/src/dropout_layer_kernels.cu:void forward_dropout_layer_gpu(dropout_layer layer, network net)
galaxy_detection/data/darknet/src/dropout_layer_kernels.cu:    cuda_random(layer.rand_gpu, size);
galaxy_detection/data/darknet/src/dropout_layer_kernels.cu:    cuda_push_array(layer.rand_gpu, layer.rand, size);
galaxy_detection/data/darknet/src/dropout_layer_kernels.cu:    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(net.input_gpu, size, layer.rand_gpu, layer.probability, layer.scale);
galaxy_detection/data/darknet/src/dropout_layer_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/dropout_layer_kernels.cu:void backward_dropout_layer_gpu(dropout_layer layer, network net)
galaxy_detection/data/darknet/src/dropout_layer_kernels.cu:    if(!net.delta_gpu) return;
galaxy_detection/data/darknet/src/dropout_layer_kernels.cu:    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(net.delta_gpu, size, layer.rand_gpu, layer.probability, layer.scale);
galaxy_detection/data/darknet/src/dropout_layer_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/network.c:        #ifdef GPU
galaxy_detection/data/darknet/src/network.c:        if(l.state_gpu){
galaxy_detection/data/darknet/src/network.c:            fill_gpu(l.outputs, 0, l.state_gpu + l.outputs*b, 1);
galaxy_detection/data/darknet/src/network.c:        if(l.h_gpu){
galaxy_detection/data/darknet/src/network.c:            fill_gpu(l.outputs, 0, l.h_gpu + l.outputs*b, 1);
galaxy_detection/data/darknet/src/network.c:#ifdef GPU
galaxy_detection/data/darknet/src/network.c:    if(netp->gpu_index >= 0){
galaxy_detection/data/darknet/src/network.c:        forward_network_gpu(netp);   
galaxy_detection/data/darknet/src/network.c:#ifdef GPU
galaxy_detection/data/darknet/src/network.c:    if(netp->gpu_index >= 0){
galaxy_detection/data/darknet/src/network.c:        update_network_gpu(netp);   
galaxy_detection/data/darknet/src/network.c:#ifdef GPU
galaxy_detection/data/darknet/src/network.c:    if(netp->gpu_index >= 0){
galaxy_detection/data/darknet/src/network.c:        backward_network_gpu(netp);   
galaxy_detection/data/darknet/src/network.c:#ifdef GPU
galaxy_detection/data/darknet/src/network.c:    cuda_set_device(net->gpu_index);
galaxy_detection/data/darknet/src/network.c:    cuda_free(net->workspace);
galaxy_detection/data/darknet/src/network.c:#ifdef GPU
galaxy_detection/data/darknet/src/network.c:    if(gpu_index >= 0){
galaxy_detection/data/darknet/src/network.c:        cuda_free(net->input_gpu);
galaxy_detection/data/darknet/src/network.c:        cuda_free(net->truth_gpu);
galaxy_detection/data/darknet/src/network.c:        net->input_gpu = cuda_make_array(net->input, net->inputs*net->batch);
galaxy_detection/data/darknet/src/network.c:        net->truth_gpu = cuda_make_array(net->truth, net->truths*net->batch);
galaxy_detection/data/darknet/src/network.c:        net->workspace = cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
galaxy_detection/data/darknet/src/network.c:#ifdef GPU
galaxy_detection/data/darknet/src/network.c:    //cuda_pull_array(l.output_gpu, l.output, l.outputs);
galaxy_detection/data/darknet/src/network.c:#ifdef GPU
galaxy_detection/data/darknet/src/network.c:    if(net->input_gpu) cuda_free(net->input_gpu);
galaxy_detection/data/darknet/src/network.c:    if(net->truth_gpu) cuda_free(net->truth_gpu);
galaxy_detection/data/darknet/src/network.c:#ifdef GPU
galaxy_detection/data/darknet/src/network.c:void forward_network_gpu(network *netp)
galaxy_detection/data/darknet/src/network.c:    cuda_set_device(net.gpu_index);
galaxy_detection/data/darknet/src/network.c:    cuda_push_array(net.input_gpu, net.input, net.inputs*net.batch);
galaxy_detection/data/darknet/src/network.c:        cuda_push_array(net.truth_gpu, net.truth, net.truths*net.batch);
galaxy_detection/data/darknet/src/network.c:        if(l.delta_gpu){
galaxy_detection/data/darknet/src/network.c:            fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
galaxy_detection/data/darknet/src/network.c:        l.forward_gpu(l, net);
galaxy_detection/data/darknet/src/network.c:        net.input_gpu = l.output_gpu;
galaxy_detection/data/darknet/src/network.c:            net.truth_gpu = l.output_gpu;
galaxy_detection/data/darknet/src/network.c:void backward_network_gpu(network *netp)
galaxy_detection/data/darknet/src/network.c:    cuda_set_device(net.gpu_index);
galaxy_detection/data/darknet/src/network.c:            net.input_gpu = prev.output_gpu;
galaxy_detection/data/darknet/src/network.c:            net.delta_gpu = prev.delta_gpu;
galaxy_detection/data/darknet/src/network.c:        l.backward_gpu(l, net);
galaxy_detection/data/darknet/src/network.c:void update_network_gpu(network *netp)
galaxy_detection/data/darknet/src/network.c:    cuda_set_device(net.gpu_index);
galaxy_detection/data/darknet/src/network.c:        if(l.update_gpu){
galaxy_detection/data/darknet/src/network.c:            l.update_gpu(l, a);
galaxy_detection/data/darknet/src/network.c:void harmless_update_network_gpu(network *netp)
galaxy_detection/data/darknet/src/network.c:    cuda_set_device(net.gpu_index);
galaxy_detection/data/darknet/src/network.c:        if(l.weight_updates_gpu) fill_gpu(l.nweights, 0, l.weight_updates_gpu, 1);
galaxy_detection/data/darknet/src/network.c:        if(l.bias_updates_gpu) fill_gpu(l.nbiases, 0, l.bias_updates_gpu, 1);
galaxy_detection/data/darknet/src/network.c:        if(l.scale_updates_gpu) fill_gpu(l.nbiases, 0, l.scale_updates_gpu, 1);
galaxy_detection/data/darknet/src/network.c:    cuda_set_device(args.net->gpu_index);
galaxy_detection/data/darknet/src/network.c:        cuda_pull_array(l.biases_gpu, l.bias_updates, l.n);
galaxy_detection/data/darknet/src/network.c:        cuda_pull_array(l.weights_gpu, l.weight_updates, l.nweights);
galaxy_detection/data/darknet/src/network.c:        if(l.scales) cuda_pull_array(l.scales_gpu, l.scale_updates, l.n);
galaxy_detection/data/darknet/src/network.c:        cuda_pull_array(l.biases_gpu, l.bias_updates, l.outputs);
galaxy_detection/data/darknet/src/network.c:        cuda_pull_array(l.weights_gpu, l.weight_updates, l.outputs*l.inputs);
galaxy_detection/data/darknet/src/network.c:        cuda_push_array(l.biases_gpu, l.biases, l.n);
galaxy_detection/data/darknet/src/network.c:        cuda_push_array(l.weights_gpu, l.weights, l.nweights);
galaxy_detection/data/darknet/src/network.c:        if(l.scales) cuda_push_array(l.scales_gpu, l.scales, l.n);
galaxy_detection/data/darknet/src/network.c:        cuda_push_array(l.biases_gpu, l.biases, l.outputs);
galaxy_detection/data/darknet/src/network.c:        cuda_push_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
galaxy_detection/data/darknet/src/network.c:        cuda_push_array(l.biases_gpu, base.biases, l.n);
galaxy_detection/data/darknet/src/network.c:        cuda_push_array(l.weights_gpu, base.weights, l.nweights);
galaxy_detection/data/darknet/src/network.c:        if (base.scales) cuda_push_array(l.scales_gpu, base.scales, l.n);
galaxy_detection/data/darknet/src/network.c:        cuda_push_array(l.biases_gpu, base.biases, l.outputs);
galaxy_detection/data/darknet/src/network.c:        cuda_push_array(l.weights_gpu, base.weights, l.outputs*l.inputs);
galaxy_detection/data/darknet/src/network.c:   cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
galaxy_detection/data/darknet/src/network.c:   cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
galaxy_detection/data/darknet/src/network.c:   if(l.scale_updates) cuda_pull_array(l.scale_updates_gpu, l.scale_updates, l.n);
galaxy_detection/data/darknet/src/network.c:   cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
galaxy_detection/data/darknet/src/network.c:   cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
galaxy_detection/data/darknet/src/network.c:   cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
galaxy_detection/data/darknet/src/network.c:   cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
galaxy_detection/data/darknet/src/network.c:   if(l.scale_updates) cuda_push_array(l.scale_updates_gpu, l.scale_updates, l.n);
galaxy_detection/data/darknet/src/network.c:   cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
galaxy_detection/data/darknet/src/network.c:   cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
galaxy_detection/data/darknet/src/network.c:   if(l.update_gpu){
galaxy_detection/data/darknet/src/network.c:   l.update_gpu(l, update_batch, rate*l.learning_rate_scale, net.momentum, net.decay);
galaxy_detection/data/darknet/src/network.c:   cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.n);
galaxy_detection/data/darknet/src/network.c:   cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.nweights);
galaxy_detection/data/darknet/src/network.c:   if(base.scale_updates) cuda_push_array(l.scale_updates_gpu, base.scale_updates, l.n);
galaxy_detection/data/darknet/src/network.c:   cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.outputs);
galaxy_detection/data/darknet/src/network.c:   cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.outputs*l.inputs);
galaxy_detection/data/darknet/src/network.c:   cuda_set_device(nets[i].gpu_index);
galaxy_detection/data/darknet/src/network.c:   cuda_set_device(nets[i].gpu_index);
galaxy_detection/data/darknet/src/network.c:        cuda_set_device(nets[i]->gpu_index);
galaxy_detection/data/darknet/src/network.c:        cuda_set_device(nets[i]->gpu_index);
galaxy_detection/data/darknet/src/network.c:    //cudaDeviceSynchronize();
galaxy_detection/data/darknet/src/network.c:    //cudaDeviceSynchronize();
galaxy_detection/data/darknet/src/network.c:    cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
galaxy_detection/data/darknet/src/utils.c:int *read_intlist(char *gpu_list, int *ngpus, int d)
galaxy_detection/data/darknet/src/utils.c:    int *gpus = 0;
galaxy_detection/data/darknet/src/utils.c:    if(gpu_list){
galaxy_detection/data/darknet/src/utils.c:        int len = strlen(gpu_list);
galaxy_detection/data/darknet/src/utils.c:        *ngpus = 1;
galaxy_detection/data/darknet/src/utils.c:            if (gpu_list[i] == ',') ++*ngpus;
galaxy_detection/data/darknet/src/utils.c:        gpus = calloc(*ngpus, sizeof(int));
galaxy_detection/data/darknet/src/utils.c:        for(i = 0; i < *ngpus; ++i){
galaxy_detection/data/darknet/src/utils.c:            gpus[i] = atoi(gpu_list);
galaxy_detection/data/darknet/src/utils.c:            gpu_list = strchr(gpu_list, ',')+1;
galaxy_detection/data/darknet/src/utils.c:        gpus = calloc(1, sizeof(float));
galaxy_detection/data/darknet/src/utils.c:        *gpus = d;
galaxy_detection/data/darknet/src/utils.c:        *ngpus = 1;
galaxy_detection/data/darknet/src/utils.c:    return gpus;
galaxy_detection/data/darknet/src/crnn_layer.h:#ifdef GPU
galaxy_detection/data/darknet/src/crnn_layer.h:void forward_crnn_layer_gpu(layer l, network net);
galaxy_detection/data/darknet/src/crnn_layer.h:void backward_crnn_layer_gpu(layer l, network net);
galaxy_detection/data/darknet/src/crnn_layer.h:void update_crnn_layer_gpu(layer l, update_args a);
galaxy_detection/data/darknet/src/col2im.h:#ifdef GPU
galaxy_detection/data/darknet/src/col2im.h:void col2im_gpu(float *data_col,
galaxy_detection/data/darknet/src/maxpool_layer_kernels.cu:#include "cuda_runtime.h"
galaxy_detection/data/darknet/src/maxpool_layer_kernels.cu:#include "cuda.h"
galaxy_detection/data/darknet/src/maxpool_layer_kernels.cu:extern "C" void forward_maxpool_layer_gpu(maxpool_layer layer, network net)
galaxy_detection/data/darknet/src/maxpool_layer_kernels.cu:    forward_maxpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.h, layer.w, layer.c, layer.stride, layer.size, layer.pad, net.input_gpu, layer.output_gpu, layer.indexes_gpu);
galaxy_detection/data/darknet/src/maxpool_layer_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/maxpool_layer_kernels.cu:extern "C" void backward_maxpool_layer_gpu(maxpool_layer layer, network net)
galaxy_detection/data/darknet/src/maxpool_layer_kernels.cu:    backward_maxpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.h, layer.w, layer.c, layer.stride, layer.size, layer.pad, layer.delta_gpu, net.delta_gpu, layer.indexes_gpu);
galaxy_detection/data/darknet/src/maxpool_layer_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/rnn_layer.c:#include "cuda.h"
galaxy_detection/data/darknet/src/rnn_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/rnn_layer.c:    l->output_gpu += num;
galaxy_detection/data/darknet/src/rnn_layer.c:    l->delta_gpu += num;
galaxy_detection/data/darknet/src/rnn_layer.c:    l->x_gpu += num;
galaxy_detection/data/darknet/src/rnn_layer.c:    l->x_norm_gpu += num;
galaxy_detection/data/darknet/src/rnn_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/rnn_layer.c:    l.forward_gpu = forward_rnn_layer_gpu;
galaxy_detection/data/darknet/src/rnn_layer.c:    l.backward_gpu = backward_rnn_layer_gpu;
galaxy_detection/data/darknet/src/rnn_layer.c:    l.update_gpu = update_rnn_layer_gpu;
galaxy_detection/data/darknet/src/rnn_layer.c:    l.state_gpu = cuda_make_array(0, batch*outputs);
galaxy_detection/data/darknet/src/rnn_layer.c:    l.prev_state_gpu = cuda_make_array(0, batch*outputs);
galaxy_detection/data/darknet/src/rnn_layer.c:    l.output_gpu = l.output_layer->output_gpu;
galaxy_detection/data/darknet/src/rnn_layer.c:    l.delta_gpu = l.output_layer->delta_gpu;
galaxy_detection/data/darknet/src/rnn_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/rnn_layer.c:void update_rnn_layer_gpu(layer l, update_args a)
galaxy_detection/data/darknet/src/rnn_layer.c:    update_connected_layer_gpu(*(l.input_layer),  a);
galaxy_detection/data/darknet/src/rnn_layer.c:    update_connected_layer_gpu(*(l.self_layer),   a);
galaxy_detection/data/darknet/src/rnn_layer.c:    update_connected_layer_gpu(*(l.output_layer), a);
galaxy_detection/data/darknet/src/rnn_layer.c:void forward_rnn_layer_gpu(layer l, network net)
galaxy_detection/data/darknet/src/rnn_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, output_layer.delta_gpu, 1);
galaxy_detection/data/darknet/src/rnn_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, self_layer.delta_gpu, 1);
galaxy_detection/data/darknet/src/rnn_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, input_layer.delta_gpu, 1);
galaxy_detection/data/darknet/src/rnn_layer.c:        fill_gpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1);
galaxy_detection/data/darknet/src/rnn_layer.c:        copy_gpu(l.outputs*l.batch, l.state_gpu, 1, l.prev_state_gpu, 1);
galaxy_detection/data/darknet/src/rnn_layer.c:        s.input_gpu = net.input_gpu;
galaxy_detection/data/darknet/src/rnn_layer.c:        forward_connected_layer_gpu(input_layer, s);
galaxy_detection/data/darknet/src/rnn_layer.c:        s.input_gpu = l.state_gpu;
galaxy_detection/data/darknet/src/rnn_layer.c:        forward_connected_layer_gpu(self_layer, s);
galaxy_detection/data/darknet/src/rnn_layer.c:        fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
galaxy_detection/data/darknet/src/rnn_layer.c:        axpy_gpu(l.outputs * l.batch, 1, input_layer.output_gpu, 1, l.state_gpu, 1);
galaxy_detection/data/darknet/src/rnn_layer.c:        axpy_gpu(l.outputs * l.batch, 1, self_layer.output_gpu, 1, l.state_gpu, 1);
galaxy_detection/data/darknet/src/rnn_layer.c:        s.input_gpu = l.state_gpu;
galaxy_detection/data/darknet/src/rnn_layer.c:        forward_connected_layer_gpu(output_layer, s);
galaxy_detection/data/darknet/src/rnn_layer.c:        net.input_gpu += l.inputs*l.batch;
galaxy_detection/data/darknet/src/rnn_layer.c:void backward_rnn_layer_gpu(layer l, network net)
galaxy_detection/data/darknet/src/rnn_layer.c:    float *last_input = input_layer.output_gpu;
galaxy_detection/data/darknet/src/rnn_layer.c:    float *last_self = self_layer.output_gpu;
galaxy_detection/data/darknet/src/rnn_layer.c:        fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
galaxy_detection/data/darknet/src/rnn_layer.c:        axpy_gpu(l.outputs * l.batch, 1, input_layer.output_gpu, 1, l.state_gpu, 1);
galaxy_detection/data/darknet/src/rnn_layer.c:        axpy_gpu(l.outputs * l.batch, 1, self_layer.output_gpu, 1, l.state_gpu, 1);
galaxy_detection/data/darknet/src/rnn_layer.c:        s.input_gpu = l.state_gpu;
galaxy_detection/data/darknet/src/rnn_layer.c:        s.delta_gpu = self_layer.delta_gpu;
galaxy_detection/data/darknet/src/rnn_layer.c:        backward_connected_layer_gpu(output_layer, s);
galaxy_detection/data/darknet/src/rnn_layer.c:            fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
galaxy_detection/data/darknet/src/rnn_layer.c:            axpy_gpu(l.outputs * l.batch, 1, input_layer.output_gpu - l.outputs*l.batch, 1, l.state_gpu, 1);
galaxy_detection/data/darknet/src/rnn_layer.c:            axpy_gpu(l.outputs * l.batch, 1, self_layer.output_gpu - l.outputs*l.batch, 1, l.state_gpu, 1);
galaxy_detection/data/darknet/src/rnn_layer.c:            copy_gpu(l.outputs*l.batch, l.prev_state_gpu, 1, l.state_gpu, 1);
galaxy_detection/data/darknet/src/rnn_layer.c:        copy_gpu(l.outputs*l.batch, self_layer.delta_gpu, 1, input_layer.delta_gpu, 1);
galaxy_detection/data/darknet/src/rnn_layer.c:        s.input_gpu = l.state_gpu;
galaxy_detection/data/darknet/src/rnn_layer.c:        s.delta_gpu = (i > 0) ? self_layer.delta_gpu - l.outputs*l.batch : 0;
galaxy_detection/data/darknet/src/rnn_layer.c:        if (i == 0) s.delta_gpu = 0;
galaxy_detection/data/darknet/src/rnn_layer.c:        backward_connected_layer_gpu(self_layer, s);
galaxy_detection/data/darknet/src/rnn_layer.c:        s.input_gpu = net.input_gpu + i*l.inputs*l.batch;
galaxy_detection/data/darknet/src/rnn_layer.c:        if(net.delta_gpu) s.delta_gpu = net.delta_gpu + i*l.inputs*l.batch;
galaxy_detection/data/darknet/src/rnn_layer.c:        else s.delta_gpu = 0;
galaxy_detection/data/darknet/src/rnn_layer.c:        backward_connected_layer_gpu(input_layer, s);
galaxy_detection/data/darknet/src/rnn_layer.c:    fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
galaxy_detection/data/darknet/src/rnn_layer.c:    axpy_gpu(l.outputs * l.batch, 1, last_input, 1, l.state_gpu, 1);
galaxy_detection/data/darknet/src/rnn_layer.c:    axpy_gpu(l.outputs * l.batch, 1, last_self, 1, l.state_gpu, 1);
galaxy_detection/data/darknet/src/rnn_layer.h:#ifdef GPU
galaxy_detection/data/darknet/src/rnn_layer.h:void forward_rnn_layer_gpu(layer l, network net);
galaxy_detection/data/darknet/src/rnn_layer.h:void backward_rnn_layer_gpu(layer l, network net);
galaxy_detection/data/darknet/src/rnn_layer.h:void update_rnn_layer_gpu(layer l, update_args a);
galaxy_detection/data/darknet/src/region_layer.h:#ifdef GPU
galaxy_detection/data/darknet/src/region_layer.h:void forward_region_layer_gpu(const layer l, network net);
galaxy_detection/data/darknet/src/region_layer.h:void backward_region_layer_gpu(layer l, network net);
galaxy_detection/data/darknet/src/softmax_layer.h:#ifdef GPU
galaxy_detection/data/darknet/src/softmax_layer.h:void forward_softmax_layer_gpu(const softmax_layer l, network net);
galaxy_detection/data/darknet/src/softmax_layer.h:void backward_softmax_layer_gpu(const softmax_layer l, network net);
galaxy_detection/data/darknet/src/lstm_layer.c:#include "cuda.h"
galaxy_detection/data/darknet/src/lstm_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/lstm_layer.c:    l->output_gpu += num;
galaxy_detection/data/darknet/src/lstm_layer.c:    l->delta_gpu += num;
galaxy_detection/data/darknet/src/lstm_layer.c:    l->x_gpu += num;
galaxy_detection/data/darknet/src/lstm_layer.c:    l->x_norm_gpu += num;
galaxy_detection/data/darknet/src/lstm_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/lstm_layer.c:    l.forward_gpu = forward_lstm_layer_gpu;
galaxy_detection/data/darknet/src/lstm_layer.c:    l.backward_gpu = backward_lstm_layer_gpu;
galaxy_detection/data/darknet/src/lstm_layer.c:    l.update_gpu = update_lstm_layer_gpu;
galaxy_detection/data/darknet/src/lstm_layer.c:    l.output_gpu = cuda_make_array(0, batch*outputs*steps);
galaxy_detection/data/darknet/src/lstm_layer.c:    l.delta_gpu = cuda_make_array(0, batch*l.outputs*steps);
galaxy_detection/data/darknet/src/lstm_layer.c:    l.prev_state_gpu = cuda_make_array(0, batch*outputs);
galaxy_detection/data/darknet/src/lstm_layer.c:    l.prev_cell_gpu = cuda_make_array(0, batch*outputs);
galaxy_detection/data/darknet/src/lstm_layer.c:    l.cell_gpu = cuda_make_array(0, batch*outputs*steps);
galaxy_detection/data/darknet/src/lstm_layer.c:    l.f_gpu = cuda_make_array(0, batch*outputs);
galaxy_detection/data/darknet/src/lstm_layer.c:    l.i_gpu = cuda_make_array(0, batch*outputs);
galaxy_detection/data/darknet/src/lstm_layer.c:    l.g_gpu = cuda_make_array(0, batch*outputs);
galaxy_detection/data/darknet/src/lstm_layer.c:    l.o_gpu = cuda_make_array(0, batch*outputs);
galaxy_detection/data/darknet/src/lstm_layer.c:    l.c_gpu = cuda_make_array(0, batch*outputs);
galaxy_detection/data/darknet/src/lstm_layer.c:    l.h_gpu = cuda_make_array(0, batch*outputs);
galaxy_detection/data/darknet/src/lstm_layer.c:    l.temp_gpu =  cuda_make_array(0, batch*outputs);
galaxy_detection/data/darknet/src/lstm_layer.c:    l.temp2_gpu = cuda_make_array(0, batch*outputs);
galaxy_detection/data/darknet/src/lstm_layer.c:    l.temp3_gpu = cuda_make_array(0, batch*outputs);
galaxy_detection/data/darknet/src/lstm_layer.c:    l.dc_gpu = cuda_make_array(0, batch*outputs);
galaxy_detection/data/darknet/src/lstm_layer.c:    l.dh_gpu = cuda_make_array(0, batch*outputs);
galaxy_detection/data/darknet/src/lstm_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/lstm_layer.c:void update_lstm_layer_gpu(layer l, update_args a)
galaxy_detection/data/darknet/src/lstm_layer.c:    update_connected_layer_gpu(*(l.wf), a);
galaxy_detection/data/darknet/src/lstm_layer.c:    update_connected_layer_gpu(*(l.wi), a);
galaxy_detection/data/darknet/src/lstm_layer.c:    update_connected_layer_gpu(*(l.wg), a);
galaxy_detection/data/darknet/src/lstm_layer.c:    update_connected_layer_gpu(*(l.wo), a);
galaxy_detection/data/darknet/src/lstm_layer.c:    update_connected_layer_gpu(*(l.uf), a);
galaxy_detection/data/darknet/src/lstm_layer.c:    update_connected_layer_gpu(*(l.ui), a);
galaxy_detection/data/darknet/src/lstm_layer.c:    update_connected_layer_gpu(*(l.ug), a);
galaxy_detection/data/darknet/src/lstm_layer.c:    update_connected_layer_gpu(*(l.uo), a);
galaxy_detection/data/darknet/src/lstm_layer.c:void forward_lstm_layer_gpu(layer l, network state)
galaxy_detection/data/darknet/src/lstm_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, wf.delta_gpu, 1);
galaxy_detection/data/darknet/src/lstm_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, wi.delta_gpu, 1);
galaxy_detection/data/darknet/src/lstm_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, wg.delta_gpu, 1);
galaxy_detection/data/darknet/src/lstm_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, wo.delta_gpu, 1);
galaxy_detection/data/darknet/src/lstm_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, uf.delta_gpu, 1);
galaxy_detection/data/darknet/src/lstm_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, ui.delta_gpu, 1);
galaxy_detection/data/darknet/src/lstm_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, ug.delta_gpu, 1);
galaxy_detection/data/darknet/src/lstm_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, uo.delta_gpu, 1);
galaxy_detection/data/darknet/src/lstm_layer.c:        fill_gpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1);
galaxy_detection/data/darknet/src/lstm_layer.c:        s.input_gpu = l.h_gpu;
galaxy_detection/data/darknet/src/lstm_layer.c:        forward_connected_layer_gpu(wf, s);							
galaxy_detection/data/darknet/src/lstm_layer.c:        forward_connected_layer_gpu(wi, s);							
galaxy_detection/data/darknet/src/lstm_layer.c:        forward_connected_layer_gpu(wg, s);							
galaxy_detection/data/darknet/src/lstm_layer.c:        forward_connected_layer_gpu(wo, s);							
galaxy_detection/data/darknet/src/lstm_layer.c:        s.input_gpu = state.input_gpu;
galaxy_detection/data/darknet/src/lstm_layer.c:        forward_connected_layer_gpu(uf, s);							
galaxy_detection/data/darknet/src/lstm_layer.c:        forward_connected_layer_gpu(ui, s);							
galaxy_detection/data/darknet/src/lstm_layer.c:        forward_connected_layer_gpu(ug, s);							
galaxy_detection/data/darknet/src/lstm_layer.c:        forward_connected_layer_gpu(uo, s);							
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, wf.output_gpu, 1, l.f_gpu, 1);
galaxy_detection/data/darknet/src/lstm_layer.c:        axpy_gpu(l.outputs*l.batch, 1, uf.output_gpu, 1, l.f_gpu, 1);
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, wi.output_gpu, 1, l.i_gpu, 1);	
galaxy_detection/data/darknet/src/lstm_layer.c:        axpy_gpu(l.outputs*l.batch, 1, ui.output_gpu, 1, l.i_gpu, 1);	
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, wg.output_gpu, 1, l.g_gpu, 1);	
galaxy_detection/data/darknet/src/lstm_layer.c:        axpy_gpu(l.outputs*l.batch, 1, ug.output_gpu, 1, l.g_gpu, 1);	
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, wo.output_gpu, 1, l.o_gpu, 1);	
galaxy_detection/data/darknet/src/lstm_layer.c:        axpy_gpu(l.outputs*l.batch, 1, uo.output_gpu, 1, l.o_gpu, 1);	
galaxy_detection/data/darknet/src/lstm_layer.c:        activate_array_gpu(l.f_gpu, l.outputs*l.batch, LOGISTIC);		
galaxy_detection/data/darknet/src/lstm_layer.c:        activate_array_gpu(l.i_gpu, l.outputs*l.batch, LOGISTIC);		
galaxy_detection/data/darknet/src/lstm_layer.c:        activate_array_gpu(l.g_gpu, l.outputs*l.batch, TANH);			
galaxy_detection/data/darknet/src/lstm_layer.c:        activate_array_gpu(l.o_gpu, l.outputs*l.batch, LOGISTIC);		
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.i_gpu, 1, l.temp_gpu, 1);		
galaxy_detection/data/darknet/src/lstm_layer.c:        mul_gpu(l.outputs*l.batch, l.g_gpu, 1, l.temp_gpu, 1);		
galaxy_detection/data/darknet/src/lstm_layer.c:        mul_gpu(l.outputs*l.batch, l.f_gpu, 1, l.c_gpu, 1);			
galaxy_detection/data/darknet/src/lstm_layer.c:        axpy_gpu(l.outputs*l.batch, 1, l.temp_gpu, 1, l.c_gpu, 1);	
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.h_gpu, 1);			
galaxy_detection/data/darknet/src/lstm_layer.c:        activate_array_gpu(l.h_gpu, l.outputs*l.batch, TANH);		
galaxy_detection/data/darknet/src/lstm_layer.c:        mul_gpu(l.outputs*l.batch, l.o_gpu, 1, l.h_gpu, 1);	
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.cell_gpu, 1);		
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.h_gpu, 1, l.output_gpu, 1);
galaxy_detection/data/darknet/src/lstm_layer.c:        state.input_gpu += l.inputs*l.batch;
galaxy_detection/data/darknet/src/lstm_layer.c:        l.output_gpu    += l.outputs*l.batch;
galaxy_detection/data/darknet/src/lstm_layer.c:        l.cell_gpu      += l.outputs*l.batch;
galaxy_detection/data/darknet/src/lstm_layer.c:void backward_lstm_layer_gpu(layer l, network state)
galaxy_detection/data/darknet/src/lstm_layer.c:    state.input_gpu += l.inputs*l.batch*(l.steps - 1);
galaxy_detection/data/darknet/src/lstm_layer.c:    if (state.delta_gpu) state.delta_gpu += l.inputs*l.batch*(l.steps - 1);
galaxy_detection/data/darknet/src/lstm_layer.c:    l.output_gpu += l.outputs*l.batch*(l.steps - 1);
galaxy_detection/data/darknet/src/lstm_layer.c:    l.cell_gpu += l.outputs*l.batch*(l.steps - 1);
galaxy_detection/data/darknet/src/lstm_layer.c:    l.delta_gpu += l.outputs*l.batch*(l.steps - 1);
galaxy_detection/data/darknet/src/lstm_layer.c:        if (i != 0) copy_gpu(l.outputs*l.batch, l.cell_gpu - l.outputs*l.batch, 1, l.prev_cell_gpu, 1);
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.cell_gpu, 1, l.c_gpu, 1);
galaxy_detection/data/darknet/src/lstm_layer.c:        if (i != 0) copy_gpu(l.outputs*l.batch, l.output_gpu - l.outputs*l.batch, 1, l.prev_state_gpu, 1);
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.h_gpu, 1);
galaxy_detection/data/darknet/src/lstm_layer.c:        l.dh_gpu = (i == 0) ? 0 : l.delta_gpu - l.outputs*l.batch;
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, wf.output_gpu, 1, l.f_gpu, 1);			
galaxy_detection/data/darknet/src/lstm_layer.c:        axpy_gpu(l.outputs*l.batch, 1, uf.output_gpu, 1, l.f_gpu, 1);			
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, wi.output_gpu, 1, l.i_gpu, 1);			
galaxy_detection/data/darknet/src/lstm_layer.c:        axpy_gpu(l.outputs*l.batch, 1, ui.output_gpu, 1, l.i_gpu, 1);			
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, wg.output_gpu, 1, l.g_gpu, 1);			
galaxy_detection/data/darknet/src/lstm_layer.c:        axpy_gpu(l.outputs*l.batch, 1, ug.output_gpu, 1, l.g_gpu, 1);			
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, wo.output_gpu, 1, l.o_gpu, 1);			
galaxy_detection/data/darknet/src/lstm_layer.c:        axpy_gpu(l.outputs*l.batch, 1, uo.output_gpu, 1, l.o_gpu, 1);			
galaxy_detection/data/darknet/src/lstm_layer.c:        activate_array_gpu(l.f_gpu, l.outputs*l.batch, LOGISTIC);			
galaxy_detection/data/darknet/src/lstm_layer.c:        activate_array_gpu(l.i_gpu, l.outputs*l.batch, LOGISTIC);		
galaxy_detection/data/darknet/src/lstm_layer.c:        activate_array_gpu(l.g_gpu, l.outputs*l.batch, TANH);			
galaxy_detection/data/darknet/src/lstm_layer.c:        activate_array_gpu(l.o_gpu, l.outputs*l.batch, LOGISTIC);		
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, l.temp3_gpu, 1);		
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.temp_gpu, 1);			
galaxy_detection/data/darknet/src/lstm_layer.c:        activate_array_gpu(l.temp_gpu, l.outputs*l.batch, TANH);			
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp3_gpu, 1, l.temp2_gpu, 1);		
galaxy_detection/data/darknet/src/lstm_layer.c:        mul_gpu(l.outputs*l.batch, l.o_gpu, 1, l.temp2_gpu, 1);			
galaxy_detection/data/darknet/src/lstm_layer.c:        gradient_array_gpu(l.temp_gpu, l.outputs*l.batch, TANH, l.temp2_gpu);
galaxy_detection/data/darknet/src/lstm_layer.c:        axpy_gpu(l.outputs*l.batch, 1, l.dc_gpu, 1, l.temp2_gpu, 1);		
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.temp_gpu, 1);			
galaxy_detection/data/darknet/src/lstm_layer.c:        activate_array_gpu(l.temp_gpu, l.outputs*l.batch, TANH);			
galaxy_detection/data/darknet/src/lstm_layer.c:        mul_gpu(l.outputs*l.batch, l.temp3_gpu, 1, l.temp_gpu, 1);		
galaxy_detection/data/darknet/src/lstm_layer.c:        gradient_array_gpu(l.o_gpu, l.outputs*l.batch, LOGISTIC, l.temp_gpu);
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wo.delta_gpu, 1);
galaxy_detection/data/darknet/src/lstm_layer.c:        s.input_gpu = l.prev_state_gpu;
galaxy_detection/data/darknet/src/lstm_layer.c:        s.delta_gpu = l.dh_gpu;															
galaxy_detection/data/darknet/src/lstm_layer.c:        backward_connected_layer_gpu(wo, s);	
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, uo.delta_gpu, 1);
galaxy_detection/data/darknet/src/lstm_layer.c:        s.input_gpu = state.input_gpu;
galaxy_detection/data/darknet/src/lstm_layer.c:        s.delta_gpu = state.delta_gpu;
galaxy_detection/data/darknet/src/lstm_layer.c:        backward_connected_layer_gpu(uo, s);									
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);			
galaxy_detection/data/darknet/src/lstm_layer.c:        mul_gpu(l.outputs*l.batch, l.i_gpu, 1, l.temp_gpu, 1);				
galaxy_detection/data/darknet/src/lstm_layer.c:        gradient_array_gpu(l.g_gpu, l.outputs*l.batch, TANH, l.temp_gpu);		
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wg.delta_gpu, 1);
galaxy_detection/data/darknet/src/lstm_layer.c:        s.input_gpu = l.prev_state_gpu;
galaxy_detection/data/darknet/src/lstm_layer.c:        s.delta_gpu = l.dh_gpu;														
galaxy_detection/data/darknet/src/lstm_layer.c:        backward_connected_layer_gpu(wg, s);	
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, ug.delta_gpu, 1);
galaxy_detection/data/darknet/src/lstm_layer.c:        s.input_gpu = state.input_gpu;
galaxy_detection/data/darknet/src/lstm_layer.c:        s.delta_gpu = state.delta_gpu;
galaxy_detection/data/darknet/src/lstm_layer.c:        backward_connected_layer_gpu(ug, s);																
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);			
galaxy_detection/data/darknet/src/lstm_layer.c:        mul_gpu(l.outputs*l.batch, l.g_gpu, 1, l.temp_gpu, 1);				
galaxy_detection/data/darknet/src/lstm_layer.c:        gradient_array_gpu(l.i_gpu, l.outputs*l.batch, LOGISTIC, l.temp_gpu);	
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wi.delta_gpu, 1);
galaxy_detection/data/darknet/src/lstm_layer.c:        s.input_gpu = l.prev_state_gpu;
galaxy_detection/data/darknet/src/lstm_layer.c:        s.delta_gpu = l.dh_gpu;
galaxy_detection/data/darknet/src/lstm_layer.c:        backward_connected_layer_gpu(wi, s);						
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, ui.delta_gpu, 1);
galaxy_detection/data/darknet/src/lstm_layer.c:        s.input_gpu = state.input_gpu;
galaxy_detection/data/darknet/src/lstm_layer.c:        s.delta_gpu = state.delta_gpu;
galaxy_detection/data/darknet/src/lstm_layer.c:        backward_connected_layer_gpu(ui, s);									
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);		
galaxy_detection/data/darknet/src/lstm_layer.c:        mul_gpu(l.outputs*l.batch, l.prev_cell_gpu, 1, l.temp_gpu, 1);
galaxy_detection/data/darknet/src/lstm_layer.c:        gradient_array_gpu(l.f_gpu, l.outputs*l.batch, LOGISTIC, l.temp_gpu);
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wf.delta_gpu, 1);
galaxy_detection/data/darknet/src/lstm_layer.c:        s.input_gpu = l.prev_state_gpu;
galaxy_detection/data/darknet/src/lstm_layer.c:        s.delta_gpu = l.dh_gpu;
galaxy_detection/data/darknet/src/lstm_layer.c:        backward_connected_layer_gpu(wf, s);						
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, uf.delta_gpu, 1);
galaxy_detection/data/darknet/src/lstm_layer.c:        s.input_gpu = state.input_gpu;
galaxy_detection/data/darknet/src/lstm_layer.c:        s.delta_gpu = state.delta_gpu;
galaxy_detection/data/darknet/src/lstm_layer.c:        backward_connected_layer_gpu(uf, s);									
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);			
galaxy_detection/data/darknet/src/lstm_layer.c:        mul_gpu(l.outputs*l.batch, l.f_gpu, 1, l.temp_gpu, 1);				
galaxy_detection/data/darknet/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, l.dc_gpu, 1);				
galaxy_detection/data/darknet/src/lstm_layer.c:        state.input_gpu -= l.inputs*l.batch;
galaxy_detection/data/darknet/src/lstm_layer.c:        if (state.delta_gpu) state.delta_gpu -= l.inputs*l.batch;
galaxy_detection/data/darknet/src/lstm_layer.c:        l.output_gpu -= l.outputs*l.batch;
galaxy_detection/data/darknet/src/lstm_layer.c:        l.cell_gpu -= l.outputs*l.batch;
galaxy_detection/data/darknet/src/lstm_layer.c:        l.delta_gpu -= l.outputs*l.batch;
galaxy_detection/data/darknet/src/crnn_layer.c:#include "cuda.h"
galaxy_detection/data/darknet/src/crnn_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/crnn_layer.c:    l->output_gpu += num;
galaxy_detection/data/darknet/src/crnn_layer.c:    l->delta_gpu += num;
galaxy_detection/data/darknet/src/crnn_layer.c:    l->x_gpu += num;
galaxy_detection/data/darknet/src/crnn_layer.c:    l->x_norm_gpu += num;
galaxy_detection/data/darknet/src/crnn_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/crnn_layer.c:    l.forward_gpu = forward_crnn_layer_gpu;
galaxy_detection/data/darknet/src/crnn_layer.c:    l.backward_gpu = backward_crnn_layer_gpu;
galaxy_detection/data/darknet/src/crnn_layer.c:    l.update_gpu = update_crnn_layer_gpu;
galaxy_detection/data/darknet/src/crnn_layer.c:    l.state_gpu = cuda_make_array(l.state, l.hidden*batch*(steps+1));
galaxy_detection/data/darknet/src/crnn_layer.c:    l.output_gpu = l.output_layer->output_gpu;
galaxy_detection/data/darknet/src/crnn_layer.c:    l.delta_gpu = l.output_layer->delta_gpu;
galaxy_detection/data/darknet/src/crnn_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/crnn_layer.c:void update_crnn_layer_gpu(layer l, update_args a)
galaxy_detection/data/darknet/src/crnn_layer.c:    update_convolutional_layer_gpu(*(l.input_layer),  a);
galaxy_detection/data/darknet/src/crnn_layer.c:    update_convolutional_layer_gpu(*(l.self_layer),   a);
galaxy_detection/data/darknet/src/crnn_layer.c:    update_convolutional_layer_gpu(*(l.output_layer), a);
galaxy_detection/data/darknet/src/crnn_layer.c:void forward_crnn_layer_gpu(layer l, network net)
galaxy_detection/data/darknet/src/crnn_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, output_layer.delta_gpu, 1);
galaxy_detection/data/darknet/src/crnn_layer.c:    fill_gpu(l.hidden * l.batch * l.steps, 0, self_layer.delta_gpu, 1);
galaxy_detection/data/darknet/src/crnn_layer.c:    fill_gpu(l.hidden * l.batch * l.steps, 0, input_layer.delta_gpu, 1);
galaxy_detection/data/darknet/src/crnn_layer.c:    if(net.train) fill_gpu(l.hidden * l.batch, 0, l.state_gpu, 1);
galaxy_detection/data/darknet/src/crnn_layer.c:        s.input_gpu = net.input_gpu;
galaxy_detection/data/darknet/src/crnn_layer.c:        forward_convolutional_layer_gpu(input_layer, s);
galaxy_detection/data/darknet/src/crnn_layer.c:        s.input_gpu = l.state_gpu;
galaxy_detection/data/darknet/src/crnn_layer.c:        forward_convolutional_layer_gpu(self_layer, s);
galaxy_detection/data/darknet/src/crnn_layer.c:        float *old_state = l.state_gpu;
galaxy_detection/data/darknet/src/crnn_layer.c:        if(net.train) l.state_gpu += l.hidden*l.batch;
galaxy_detection/data/darknet/src/crnn_layer.c:            copy_gpu(l.hidden * l.batch, old_state, 1, l.state_gpu, 1);
galaxy_detection/data/darknet/src/crnn_layer.c:            fill_gpu(l.hidden * l.batch, 0, l.state_gpu, 1);
galaxy_detection/data/darknet/src/crnn_layer.c:        axpy_gpu(l.hidden * l.batch, 1, input_layer.output_gpu, 1, l.state_gpu, 1);
galaxy_detection/data/darknet/src/crnn_layer.c:        axpy_gpu(l.hidden * l.batch, 1, self_layer.output_gpu, 1, l.state_gpu, 1);
galaxy_detection/data/darknet/src/crnn_layer.c:        s.input_gpu = l.state_gpu;
galaxy_detection/data/darknet/src/crnn_layer.c:        forward_convolutional_layer_gpu(output_layer, s);
galaxy_detection/data/darknet/src/crnn_layer.c:        net.input_gpu += l.inputs*l.batch;
galaxy_detection/data/darknet/src/crnn_layer.c:void backward_crnn_layer_gpu(layer l, network net)
galaxy_detection/data/darknet/src/crnn_layer.c:    l.state_gpu += l.hidden*l.batch*l.steps;
galaxy_detection/data/darknet/src/crnn_layer.c:        copy_gpu(l.hidden * l.batch, input_layer.output_gpu, 1, l.state_gpu, 1);
galaxy_detection/data/darknet/src/crnn_layer.c:        axpy_gpu(l.hidden * l.batch, 1, self_layer.output_gpu, 1, l.state_gpu, 1);
galaxy_detection/data/darknet/src/crnn_layer.c:        s.input_gpu = l.state_gpu;
galaxy_detection/data/darknet/src/crnn_layer.c:        s.delta_gpu = self_layer.delta_gpu;
galaxy_detection/data/darknet/src/crnn_layer.c:        backward_convolutional_layer_gpu(output_layer, s);
galaxy_detection/data/darknet/src/crnn_layer.c:        l.state_gpu -= l.hidden*l.batch;
galaxy_detection/data/darknet/src/crnn_layer.c:        s.input_gpu = l.state_gpu;
galaxy_detection/data/darknet/src/crnn_layer.c:        s.delta_gpu = self_layer.delta_gpu - l.hidden*l.batch;
galaxy_detection/data/darknet/src/crnn_layer.c:        if (i == 0) s.delta_gpu = 0;
galaxy_detection/data/darknet/src/crnn_layer.c:        backward_convolutional_layer_gpu(self_layer, s);
galaxy_detection/data/darknet/src/crnn_layer.c:        copy_gpu(l.hidden*l.batch, self_layer.delta_gpu, 1, input_layer.delta_gpu, 1);
galaxy_detection/data/darknet/src/crnn_layer.c:        if (i > 0 && l.shortcut) axpy_gpu(l.hidden*l.batch, 1, self_layer.delta_gpu, 1, self_layer.delta_gpu - l.hidden*l.batch, 1);
galaxy_detection/data/darknet/src/crnn_layer.c:        s.input_gpu = net.input_gpu + i*l.inputs*l.batch;
galaxy_detection/data/darknet/src/crnn_layer.c:        if(net.delta_gpu) s.delta_gpu = net.delta_gpu + i*l.inputs*l.batch;
galaxy_detection/data/darknet/src/crnn_layer.c:        else s.delta_gpu = 0;
galaxy_detection/data/darknet/src/crnn_layer.c:        backward_convolutional_layer_gpu(input_layer, s);
galaxy_detection/data/darknet/src/gru_layer.c:#include "cuda.h"
galaxy_detection/data/darknet/src/gru_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/gru_layer.c:    l->output_gpu += num;
galaxy_detection/data/darknet/src/gru_layer.c:    l->delta_gpu += num;
galaxy_detection/data/darknet/src/gru_layer.c:    l->x_gpu += num;
galaxy_detection/data/darknet/src/gru_layer.c:    l->x_norm_gpu += num;
galaxy_detection/data/darknet/src/gru_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/gru_layer.c:    l.forward_gpu = forward_gru_layer_gpu;
galaxy_detection/data/darknet/src/gru_layer.c:    l.backward_gpu = backward_gru_layer_gpu;
galaxy_detection/data/darknet/src/gru_layer.c:    l.update_gpu = update_gru_layer_gpu;
galaxy_detection/data/darknet/src/gru_layer.c:    l.forgot_state_gpu = cuda_make_array(0, batch*outputs);
galaxy_detection/data/darknet/src/gru_layer.c:    l.forgot_delta_gpu = cuda_make_array(0, batch*outputs);
galaxy_detection/data/darknet/src/gru_layer.c:    l.prev_state_gpu = cuda_make_array(0, batch*outputs);
galaxy_detection/data/darknet/src/gru_layer.c:    l.state_gpu = cuda_make_array(0, batch*outputs);
galaxy_detection/data/darknet/src/gru_layer.c:    l.output_gpu = cuda_make_array(0, batch*outputs*steps);
galaxy_detection/data/darknet/src/gru_layer.c:    l.delta_gpu = cuda_make_array(0, batch*outputs*steps);
galaxy_detection/data/darknet/src/gru_layer.c:    l.r_gpu = cuda_make_array(0, batch*outputs);
galaxy_detection/data/darknet/src/gru_layer.c:    l.z_gpu = cuda_make_array(0, batch*outputs);
galaxy_detection/data/darknet/src/gru_layer.c:    l.h_gpu = cuda_make_array(0, batch*outputs);
galaxy_detection/data/darknet/src/gru_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/gru_layer.c:void update_gru_layer_gpu(layer l, update_args a)
galaxy_detection/data/darknet/src/gru_layer.c:    update_connected_layer_gpu(*(l.ur), a);
galaxy_detection/data/darknet/src/gru_layer.c:    update_connected_layer_gpu(*(l.uz), a);
galaxy_detection/data/darknet/src/gru_layer.c:    update_connected_layer_gpu(*(l.uh), a);
galaxy_detection/data/darknet/src/gru_layer.c:    update_connected_layer_gpu(*(l.wr), a);
galaxy_detection/data/darknet/src/gru_layer.c:    update_connected_layer_gpu(*(l.wz), a);
galaxy_detection/data/darknet/src/gru_layer.c:    update_connected_layer_gpu(*(l.wh), a);
galaxy_detection/data/darknet/src/gru_layer.c:void forward_gru_layer_gpu(layer l, network net)
galaxy_detection/data/darknet/src/gru_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, uz.delta_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, ur.delta_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, uh.delta_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, wz.delta_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, wr.delta_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, wh.delta_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:        fill_gpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, l.state_gpu, 1, l.prev_state_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:        s.input_gpu = l.state_gpu;
galaxy_detection/data/darknet/src/gru_layer.c:        forward_connected_layer_gpu(wz, s);
galaxy_detection/data/darknet/src/gru_layer.c:        forward_connected_layer_gpu(wr, s);
galaxy_detection/data/darknet/src/gru_layer.c:        s.input_gpu = net.input_gpu;
galaxy_detection/data/darknet/src/gru_layer.c:        forward_connected_layer_gpu(uz, s);
galaxy_detection/data/darknet/src/gru_layer.c:        forward_connected_layer_gpu(ur, s);
galaxy_detection/data/darknet/src/gru_layer.c:        forward_connected_layer_gpu(uh, s);
galaxy_detection/data/darknet/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, uz.output_gpu, 1, l.z_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:        axpy_gpu(l.outputs*l.batch, 1, wz.output_gpu, 1, l.z_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, ur.output_gpu, 1, l.r_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:        axpy_gpu(l.outputs*l.batch, 1, wr.output_gpu, 1, l.r_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:        activate_array_gpu(l.z_gpu, l.outputs*l.batch, LOGISTIC);
galaxy_detection/data/darknet/src/gru_layer.c:        activate_array_gpu(l.r_gpu, l.outputs*l.batch, LOGISTIC);
galaxy_detection/data/darknet/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, l.state_gpu, 1, l.forgot_state_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:        mul_gpu(l.outputs*l.batch, l.r_gpu, 1, l.forgot_state_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:        s.input_gpu = l.forgot_state_gpu;
galaxy_detection/data/darknet/src/gru_layer.c:        forward_connected_layer_gpu(wh, s);
galaxy_detection/data/darknet/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, uh.output_gpu, 1, l.h_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:        axpy_gpu(l.outputs*l.batch, 1, wh.output_gpu, 1, l.h_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:            activate_array_gpu(l.h_gpu, l.outputs*l.batch, TANH);
galaxy_detection/data/darknet/src/gru_layer.c:            activate_array_gpu(l.h_gpu, l.outputs*l.batch, LOGISTIC);
galaxy_detection/data/darknet/src/gru_layer.c:        weighted_sum_gpu(l.state_gpu, l.h_gpu, l.z_gpu, l.outputs*l.batch, l.output_gpu);
galaxy_detection/data/darknet/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.state_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:        net.input_gpu += l.inputs*l.batch;
galaxy_detection/data/darknet/src/gru_layer.c:        l.output_gpu += l.outputs*l.batch;
galaxy_detection/data/darknet/src/gru_layer.c:void backward_gru_layer_gpu(layer l, network net)
galaxy_detection/data/darknet/src/gru_layer.c:    net.input_gpu += l.inputs*l.batch*(l.steps-1);
galaxy_detection/data/darknet/src/gru_layer.c:    if(net.delta_gpu) net.delta_gpu += l.inputs*l.batch*(l.steps-1);
galaxy_detection/data/darknet/src/gru_layer.c:    l.output_gpu += l.outputs*l.batch*(l.steps-1);
galaxy_detection/data/darknet/src/gru_layer.c:    l.delta_gpu += l.outputs*l.batch*(l.steps-1);
galaxy_detection/data/darknet/src/gru_layer.c:    float *end_state = l.output_gpu;
galaxy_detection/data/darknet/src/gru_layer.c:        if(i != 0) copy_gpu(l.outputs*l.batch, l.output_gpu - l.outputs*l.batch, 1, l.state_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:        else copy_gpu(l.outputs*l.batch, l.prev_state_gpu, 1, l.state_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:        float *prev_delta_gpu = (i == 0) ? 0 : l.delta_gpu - l.outputs*l.batch;
galaxy_detection/data/darknet/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, uz.output_gpu, 1, l.z_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:        axpy_gpu(l.outputs*l.batch, 1, wz.output_gpu, 1, l.z_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, ur.output_gpu, 1, l.r_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:        axpy_gpu(l.outputs*l.batch, 1, wr.output_gpu, 1, l.r_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:        activate_array_gpu(l.z_gpu, l.outputs*l.batch, LOGISTIC);
galaxy_detection/data/darknet/src/gru_layer.c:        activate_array_gpu(l.r_gpu, l.outputs*l.batch, LOGISTIC);
galaxy_detection/data/darknet/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, uh.output_gpu, 1, l.h_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:        axpy_gpu(l.outputs*l.batch, 1, wh.output_gpu, 1, l.h_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:            activate_array_gpu(l.h_gpu, l.outputs*l.batch, TANH);
galaxy_detection/data/darknet/src/gru_layer.c:            activate_array_gpu(l.h_gpu, l.outputs*l.batch, LOGISTIC);
galaxy_detection/data/darknet/src/gru_layer.c:        weighted_delta_gpu(l.state_gpu, l.h_gpu, l.z_gpu, prev_delta_gpu, uh.delta_gpu, uz.delta_gpu, l.outputs*l.batch, l.delta_gpu);
galaxy_detection/data/darknet/src/gru_layer.c:            gradient_array_gpu(l.h_gpu, l.outputs*l.batch, TANH, uh.delta_gpu);
galaxy_detection/data/darknet/src/gru_layer.c:            gradient_array_gpu(l.h_gpu, l.outputs*l.batch, LOGISTIC, uh.delta_gpu);
galaxy_detection/data/darknet/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, uh.delta_gpu, 1, wh.delta_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, l.state_gpu, 1, l.forgot_state_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:        mul_gpu(l.outputs*l.batch, l.r_gpu, 1, l.forgot_state_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:        fill_gpu(l.outputs*l.batch, 0, l.forgot_delta_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:        s.input_gpu = l.forgot_state_gpu;
galaxy_detection/data/darknet/src/gru_layer.c:        s.delta_gpu = l.forgot_delta_gpu;
galaxy_detection/data/darknet/src/gru_layer.c:        backward_connected_layer_gpu(wh, s);
galaxy_detection/data/darknet/src/gru_layer.c:        if(prev_delta_gpu) mult_add_into_gpu(l.outputs*l.batch, l.forgot_delta_gpu, l.r_gpu, prev_delta_gpu);
galaxy_detection/data/darknet/src/gru_layer.c:        mult_add_into_gpu(l.outputs*l.batch, l.forgot_delta_gpu, l.state_gpu, ur.delta_gpu);
galaxy_detection/data/darknet/src/gru_layer.c:        gradient_array_gpu(l.r_gpu, l.outputs*l.batch, LOGISTIC, ur.delta_gpu);
galaxy_detection/data/darknet/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, ur.delta_gpu, 1, wr.delta_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:        gradient_array_gpu(l.z_gpu, l.outputs*l.batch, LOGISTIC, uz.delta_gpu);
galaxy_detection/data/darknet/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, uz.delta_gpu, 1, wz.delta_gpu, 1);
galaxy_detection/data/darknet/src/gru_layer.c:        s.input_gpu = l.state_gpu;
galaxy_detection/data/darknet/src/gru_layer.c:        s.delta_gpu = prev_delta_gpu;
galaxy_detection/data/darknet/src/gru_layer.c:        backward_connected_layer_gpu(wr, s);
galaxy_detection/data/darknet/src/gru_layer.c:        backward_connected_layer_gpu(wz, s);
galaxy_detection/data/darknet/src/gru_layer.c:        s.input_gpu = net.input_gpu;
galaxy_detection/data/darknet/src/gru_layer.c:        s.delta_gpu = net.delta_gpu;
galaxy_detection/data/darknet/src/gru_layer.c:        backward_connected_layer_gpu(uh, s);
galaxy_detection/data/darknet/src/gru_layer.c:        backward_connected_layer_gpu(ur, s);
galaxy_detection/data/darknet/src/gru_layer.c:        backward_connected_layer_gpu(uz, s);
galaxy_detection/data/darknet/src/gru_layer.c:        net.input_gpu -= l.inputs*l.batch;
galaxy_detection/data/darknet/src/gru_layer.c:        if(net.delta_gpu) net.delta_gpu -= l.inputs*l.batch;
galaxy_detection/data/darknet/src/gru_layer.c:        l.output_gpu -= l.outputs*l.batch;
galaxy_detection/data/darknet/src/gru_layer.c:        l.delta_gpu -= l.outputs*l.batch;
galaxy_detection/data/darknet/src/gru_layer.c:    copy_gpu(l.outputs*l.batch, end_state, 1, l.state_gpu, 1);
galaxy_detection/data/darknet/src/blas_kernels.cu:#include "cuda_runtime.h"
galaxy_detection/data/darknet/src/blas_kernels.cu:#include "cuda.h"
galaxy_detection/data/darknet/src/blas_kernels.cu:void scale_bias_gpu(float *output, float *biases, int batch, int n, int size)
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:void backward_scale_gpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:void add_bias_gpu(float *output, float *biases, int batch, int n, int size)
galaxy_detection/data/darknet/src/blas_kernels.cu:    add_bias_kernel<<<cuda_gridsize(num), BLOCK>>>(output, biases, batch, n, size);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size)
galaxy_detection/data/darknet/src/blas_kernels.cu:        backward_bias_conn_kernel<<<cuda_gridsize(n), BLOCK>>>(bias_updates, delta, batch, n);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:void dot_error_gpu(layer l)
galaxy_detection/data/darknet/src/blas_kernels.cu:    dot_kernel<<<cuda_gridsize(l.n*l.n), BLOCK>>>(l.output_gpu, l.dot, l.batch, l.n, l.out_w * l.out_h, l.delta_gpu);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void adam_gpu(int n, float *x, float *m, float *v, float B1, float B2, float rate, float eps, int t)
galaxy_detection/data/darknet/src/blas_kernels.cu:    adam_kernel<<<cuda_gridsize(n), BLOCK>>>(n, x, m, v, B1, B2, rate, eps, t);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t)
galaxy_detection/data/darknet/src/blas_kernels.cu:    scal_gpu(n, B1, m, 1);
galaxy_detection/data/darknet/src/blas_kernels.cu:    scal_gpu(n, B2, v, 1);
galaxy_detection/data/darknet/src/blas_kernels.cu:    axpy_gpu(n, -decay*batch, w, 1, d, 1);
galaxy_detection/data/darknet/src/blas_kernels.cu:    axpy_gpu(n, (1-B1), d, 1, m, 1);
galaxy_detection/data/darknet/src/blas_kernels.cu:    mul_gpu(n, d, 1, d, 1);
galaxy_detection/data/darknet/src/blas_kernels.cu:    axpy_gpu(n, (1-B2), d, 1, v, 1);
galaxy_detection/data/darknet/src/blas_kernels.cu:    adam_gpu(n, w, m, v, B1, B2, rate, eps, t);
galaxy_detection/data/darknet/src/blas_kernels.cu:    fill_gpu(n, 0, d, 1);
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void normalize_delta_gpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
galaxy_detection/data/darknet/src/blas_kernels.cu:    normalize_delta_kernel<<<cuda_gridsize(N), BLOCK>>>(N, x, mean, variance, mean_delta, variance_delta, batch, filters, spatial, delta);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
galaxy_detection/data/darknet/src/blas_kernels.cu:    mean_delta_kernel<<<cuda_gridsize(filters), BLOCK>>>(delta, variance, batch, filters, spatial, mean_delta);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void fast_mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void fast_variance_delta_gpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
galaxy_detection/data/darknet/src/blas_kernels.cu:    normalize_kernel<<<cuda_gridsize(N), BLOCK>>>(N, x, mean, variance, batch, filters, spatial);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void fast_mean_gpu(float *x, int batch, int filters, int spatial, float *mean)
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void fast_variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void mean_gpu(float *x, int batch, int filters, int spatial, float *mean)
galaxy_detection/data/darknet/src/blas_kernels.cu:    mean_kernel<<<cuda_gridsize(filters), BLOCK>>>(x, batch, filters, spatial, mean);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
galaxy_detection/data/darknet/src/blas_kernels.cu:    variance_kernel<<<cuda_gridsize(filters), BLOCK>>>(x, mean, batch, filters, spatial, variance);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY)
galaxy_detection/data/darknet/src/blas_kernels.cu:    axpy_gpu_offset(N, ALPHA, X, 0, INCX, Y, 0, INCY);
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void pow_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY)
galaxy_detection/data/darknet/src/blas_kernels.cu:    pow_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX, Y, INCY);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void axpy_gpu_offset(int N, float ALPHA, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
galaxy_detection/data/darknet/src/blas_kernels.cu:    axpy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, OFFX, INCX, Y, OFFY, INCY);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void copy_gpu(int N, float * X, int INCX, float * Y, int INCY)
galaxy_detection/data/darknet/src/blas_kernels.cu:    copy_gpu_offset(N, X, 0, INCX, Y, 0, INCY);
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void mul_gpu(int N, float * X, int INCX, float * Y, int INCY)
galaxy_detection/data/darknet/src/blas_kernels.cu:    mul_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, INCX, Y, INCY);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void copy_gpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
galaxy_detection/data/darknet/src/blas_kernels.cu:    copy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, OFFX, INCX, Y, OFFY, INCY);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void flatten_gpu(float *x, int spatial, int layers, int batch, int forward, float *out)
galaxy_detection/data/darknet/src/blas_kernels.cu:    flatten_kernel<<<cuda_gridsize(size), BLOCK>>>(size, x, spatial, layers, batch, forward, out);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void reorg_gpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
galaxy_detection/data/darknet/src/blas_kernels.cu:    reorg_kernel<<<cuda_gridsize(size), BLOCK>>>(size, x, w, h, c, batch, stride, forward, out);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void scale_mask_gpu(int N, float * X, float mask_num, float * mask, float scale)
galaxy_detection/data/darknet/src/blas_kernels.cu:    scale_mask_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, mask_num, mask, scale);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void mask_gpu(int N, float * X, float mask_num, float * mask)
galaxy_detection/data/darknet/src/blas_kernels.cu:    mask_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, mask_num, mask);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void const_gpu(int N, float ALPHA, float * X, int INCX)
galaxy_detection/data/darknet/src/blas_kernels.cu:    const_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void constrain_gpu(int N, float ALPHA, float * X, int INCX)
galaxy_detection/data/darknet/src/blas_kernels.cu:    constrain_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void add_gpu(int N, float ALPHA, float * X, int INCX)
galaxy_detection/data/darknet/src/blas_kernels.cu:    add_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void scal_gpu(int N, float ALPHA, float * X, int INCX)
galaxy_detection/data/darknet/src/blas_kernels.cu:    scal_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void supp_gpu(int N, float ALPHA, float * X, int INCX)
galaxy_detection/data/darknet/src/blas_kernels.cu:    supp_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void fill_gpu(int N, float ALPHA, float * X, int INCX)
galaxy_detection/data/darknet/src/blas_kernels.cu:    fill_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void shortcut_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out)
galaxy_detection/data/darknet/src/blas_kernels.cu:    shortcut_kernel<<<cuda_gridsize(size), BLOCK>>>(size, minw, minh, minc, stride, sample, batch, w1, h1, c1, add, w2, h2, c2, out);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void smooth_l1_gpu(int n, float *pred, float *truth, float *delta, float *error)
galaxy_detection/data/darknet/src/blas_kernels.cu:    smooth_l1_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void l2_gpu(int n, float *pred, float *truth, float *delta, float *error)
galaxy_detection/data/darknet/src/blas_kernels.cu:    l2_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void l1_gpu(int n, float *pred, float *truth, float *delta, float *error)
galaxy_detection/data/darknet/src/blas_kernels.cu:    l1_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void deinter_gpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
galaxy_detection/data/darknet/src/blas_kernels.cu:    deinter_kernel<<<cuda_gridsize((NX+NY)*B), BLOCK>>>(NX, X, NY, Y, B, OUT);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void inter_gpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
galaxy_detection/data/darknet/src/blas_kernels.cu:    inter_kernel<<<cuda_gridsize((NX+NY)*B), BLOCK>>>(NX, X, NY, Y, B, OUT);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void weighted_sum_gpu(float *a, float *b, float *s, int num, float *c)
galaxy_detection/data/darknet/src/blas_kernels.cu:    weighted_sum_kernel<<<cuda_gridsize(num), BLOCK>>>(num, a, b, s, c);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void weighted_delta_gpu(float *a, float *b, float *s, float *da, float *db, float *ds, int num, float *dc)
galaxy_detection/data/darknet/src/blas_kernels.cu:    weighted_delta_kernel<<<cuda_gridsize(num), BLOCK>>>(num, a, b, s, da, db, ds, dc);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void mult_add_into_gpu(int num, float *a, float *b, float *c)
galaxy_detection/data/darknet/src/blas_kernels.cu:    mult_add_into_kernel<<<cuda_gridsize(num), BLOCK>>>(num, a, b, c);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:    int *tree_groups_size = cuda_make_int_array(hier.group_size, hier.groups);
galaxy_detection/data/darknet/src/blas_kernels.cu:    int *tree_groups_offset = cuda_make_int_array(hier.group_offset, hier.groups);
galaxy_detection/data/darknet/src/blas_kernels.cu:        tree_groups_size = cuda_make_int_array(hier.group_size, hier.groups);
galaxy_detection/data/darknet/src/blas_kernels.cu:        tree_groups_offset = cuda_make_int_array(hier.group_offset, hier.groups);
galaxy_detection/data/darknet/src/blas_kernels.cu:    softmax_tree_kernel<<<cuda_gridsize(num), BLOCK>>>(input, spatial, batch, stride, temp, output, hier.groups, tree_groups_size, tree_groups_offset);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/blas_kernels.cu:    cuda_free((float *)tree_groups_size);
galaxy_detection/data/darknet/src/blas_kernels.cu:    cuda_free((float *)tree_groups_offset);
galaxy_detection/data/darknet/src/blas_kernels.cu:extern "C" void softmax_gpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
galaxy_detection/data/darknet/src/blas_kernels.cu:    softmax_kernel<<<cuda_gridsize(batch*groups), BLOCK>>>(input, n, batch, batch_offset, groups, group_offset, stride, temp, output);
galaxy_detection/data/darknet/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/reorg_layer.c:#include "cuda.h"
galaxy_detection/data/darknet/src/reorg_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/reorg_layer.c:    l.forward_gpu = forward_reorg_layer_gpu;
galaxy_detection/data/darknet/src/reorg_layer.c:    l.backward_gpu = backward_reorg_layer_gpu;
galaxy_detection/data/darknet/src/reorg_layer.c:    l.output_gpu  = cuda_make_array(l.output, output_size);
galaxy_detection/data/darknet/src/reorg_layer.c:    l.delta_gpu   = cuda_make_array(l.delta, output_size);
galaxy_detection/data/darknet/src/reorg_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/reorg_layer.c:    cuda_free(l->output_gpu);
galaxy_detection/data/darknet/src/reorg_layer.c:    cuda_free(l->delta_gpu);
galaxy_detection/data/darknet/src/reorg_layer.c:    l->output_gpu  = cuda_make_array(l->output, output_size);
galaxy_detection/data/darknet/src/reorg_layer.c:    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
galaxy_detection/data/darknet/src/reorg_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/reorg_layer.c:void forward_reorg_layer_gpu(layer l, network net)
galaxy_detection/data/darknet/src/reorg_layer.c:            flatten_gpu(net.input_gpu, l.w*l.h, l.c, l.batch, 0, l.output_gpu);
galaxy_detection/data/darknet/src/reorg_layer.c:            flatten_gpu(net.input_gpu, l.w*l.h, l.c, l.batch, 1, l.output_gpu);
galaxy_detection/data/darknet/src/reorg_layer.c:            copy_gpu(l.inputs, net.input_gpu + i*l.inputs, 1, l.output_gpu + i*l.outputs, 1);
galaxy_detection/data/darknet/src/reorg_layer.c:        reorg_gpu(net.input_gpu, l.w, l.h, l.c, l.batch, l.stride, 1, l.output_gpu);
galaxy_detection/data/darknet/src/reorg_layer.c:        reorg_gpu(net.input_gpu, l.w, l.h, l.c, l.batch, l.stride, 0, l.output_gpu);
galaxy_detection/data/darknet/src/reorg_layer.c:void backward_reorg_layer_gpu(layer l, network net)
galaxy_detection/data/darknet/src/reorg_layer.c:            flatten_gpu(l.delta_gpu, l.w*l.h, l.c, l.batch, 1, net.delta_gpu);
galaxy_detection/data/darknet/src/reorg_layer.c:            flatten_gpu(l.delta_gpu, l.w*l.h, l.c, l.batch, 0, net.delta_gpu);
galaxy_detection/data/darknet/src/reorg_layer.c:            copy_gpu(l.inputs, l.delta_gpu + i*l.outputs, 1, net.delta_gpu + i*l.inputs, 1);
galaxy_detection/data/darknet/src/reorg_layer.c:        reorg_gpu(l.delta_gpu, l.w, l.h, l.c, l.batch, l.stride, 0, net.delta_gpu);
galaxy_detection/data/darknet/src/reorg_layer.c:        reorg_gpu(l.delta_gpu, l.w, l.h, l.c, l.batch, l.stride, 1, net.delta_gpu);
galaxy_detection/data/darknet/src/avgpool_layer_kernels.cu:#include "cuda_runtime.h"
galaxy_detection/data/darknet/src/avgpool_layer_kernels.cu:#include "cuda.h"
galaxy_detection/data/darknet/src/avgpool_layer_kernels.cu:extern "C" void forward_avgpool_layer_gpu(avgpool_layer layer, network net)
galaxy_detection/data/darknet/src/avgpool_layer_kernels.cu:    forward_avgpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.w, layer.h, layer.c, net.input_gpu, layer.output_gpu);
galaxy_detection/data/darknet/src/avgpool_layer_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/avgpool_layer_kernels.cu:extern "C" void backward_avgpool_layer_gpu(avgpool_layer layer, network net)
galaxy_detection/data/darknet/src/avgpool_layer_kernels.cu:    backward_avgpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.w, layer.h, layer.c, net.delta_gpu, layer.delta_gpu);
galaxy_detection/data/darknet/src/avgpool_layer_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/activations.h:#include "cuda.h"
galaxy_detection/data/darknet/src/activations.h:#ifdef GPU
galaxy_detection/data/darknet/src/activations.h:void activate_array_gpu(float *x, int n, ACTIVATION a);
galaxy_detection/data/darknet/src/activations.h:void gradient_array_gpu(float *x, int n, ACTIVATION a, float *delta);
galaxy_detection/data/darknet/src/lstm_layer.h:#ifdef GPU
galaxy_detection/data/darknet/src/lstm_layer.h:void forward_lstm_layer_gpu(layer l, network net);
galaxy_detection/data/darknet/src/lstm_layer.h:void backward_lstm_layer_gpu(layer l, network net);
galaxy_detection/data/darknet/src/lstm_layer.h:void update_lstm_layer_gpu(layer l, update_args a); 
galaxy_detection/data/darknet/src/dropout_layer.h:#ifdef GPU
galaxy_detection/data/darknet/src/dropout_layer.h:void forward_dropout_layer_gpu(dropout_layer l, network net);
galaxy_detection/data/darknet/src/dropout_layer.h:void backward_dropout_layer_gpu(dropout_layer l, network net);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:#include "cuda_runtime.h"
galaxy_detection/data/darknet/src/convolutional_kernels.cu:#include "cuda.h"
galaxy_detection/data/darknet/src/convolutional_kernels.cu:void binarize_gpu(float *x, int n, float *binary)
galaxy_detection/data/darknet/src/convolutional_kernels.cu:    binarize_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, binary);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/convolutional_kernels.cu:void binarize_input_gpu(float *input, int n, int size, float *binary)
galaxy_detection/data/darknet/src/convolutional_kernels.cu:    binarize_input_kernel<<<cuda_gridsize(size), BLOCK>>>(input, n, size, binary);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/convolutional_kernels.cu:void binarize_weights_gpu(float *weights, int n, int size, float *binary)
galaxy_detection/data/darknet/src/convolutional_kernels.cu:    binarize_weights_kernel<<<cuda_gridsize(n), BLOCK>>>(weights, n, size, binary);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/convolutional_kernels.cu:void forward_convolutional_layer_gpu(convolutional_layer l, network net)
galaxy_detection/data/darknet/src/convolutional_kernels.cu:    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:        binarize_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:        binarize_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:        binarize_gpu(net.input_gpu, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:        net.input_gpu = l.binary_input_gpu;
galaxy_detection/data/darknet/src/convolutional_kernels.cu:                net.input_gpu,
galaxy_detection/data/darknet/src/convolutional_kernels.cu:                l.weights_gpu,
galaxy_detection/data/darknet/src/convolutional_kernels.cu:                l.output_gpu);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:            float *a = l.weights_gpu + j*l.nweights/l.groups;
galaxy_detection/data/darknet/src/convolutional_kernels.cu:            float *c = l.output_gpu + (i*l.groups + j)*n*m;
galaxy_detection/data/darknet/src/convolutional_kernels.cu:            im2col_gpu(net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w,
galaxy_detection/data/darknet/src/convolutional_kernels.cu:            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:        forward_batchnorm_layer_gpu(l, net);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:    //if(l.dot > 0) dot_error_gpu(l);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:    smooth_kernel<<<cuda_gridsize(n), BLOCK>>>(l.output_gpu, n, l.w, l.h, l.c, size, rate, l.delta_gpu);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/convolutional_kernels.cu:void backward_convolutional_layer_gpu(convolutional_layer l, network net)
galaxy_detection/data/darknet/src/convolutional_kernels.cu:    constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:        backward_batchnorm_layer_gpu(l, net);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:    float *original_input = net.input_gpu;
galaxy_detection/data/darknet/src/convolutional_kernels.cu:    if(l.xnor) net.input_gpu = l.binary_input_gpu;
galaxy_detection/data/darknet/src/convolutional_kernels.cu:            net.input_gpu,
galaxy_detection/data/darknet/src/convolutional_kernels.cu:            l.delta_gpu,
galaxy_detection/data/darknet/src/convolutional_kernels.cu:            l.weight_updates_gpu);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:    if(net.delta_gpu){
galaxy_detection/data/darknet/src/convolutional_kernels.cu:                l.weights_gpu,
galaxy_detection/data/darknet/src/convolutional_kernels.cu:                l.delta_gpu,
galaxy_detection/data/darknet/src/convolutional_kernels.cu:                net.delta_gpu);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:        if(l.xnor) gradient_array_gpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, net.delta_gpu);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:            float *a = l.delta_gpu + (i*l.groups + j)*m*k;
galaxy_detection/data/darknet/src/convolutional_kernels.cu:            float *c = l.weight_updates_gpu + j*l.nweights/l.groups;
galaxy_detection/data/darknet/src/convolutional_kernels.cu:            im2col_gpu(im, l.c/l.groups, l.h, l.w,
galaxy_detection/data/darknet/src/convolutional_kernels.cu:            gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:            if(net.delta_gpu){
galaxy_detection/data/darknet/src/convolutional_kernels.cu:                a = l.weights_gpu + j*l.nweights/l.groups;
galaxy_detection/data/darknet/src/convolutional_kernels.cu:                b = l.delta_gpu + (i*l.groups + j)*m*k;
galaxy_detection/data/darknet/src/convolutional_kernels.cu:                gemm_gpu(1,0,n,k,m,1,a,n,b,k,0,c,k);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:                col2im_gpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, 
galaxy_detection/data/darknet/src/convolutional_kernels.cu:                    l.pad, net.delta_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:            if(l.xnor) gradient_array_gpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, net.delta_gpu + i*l.c*l.h*l.w);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:    cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:    cuda_pull_array(l.biases_gpu, l.biases, l.n);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:        cuda_pull_array(l.scales_gpu, l.scales, l.n);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:    cuda_push_array(l.weights_gpu, l.weights, l.nweights);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:    cuda_push_array(l.biases_gpu, l.biases, l.n);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:        cuda_push_array(l.scales_gpu, l.scales, l.n);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:void update_convolutional_layer_gpu(layer l, update_args a)
galaxy_detection/data/darknet/src/convolutional_kernels.cu:        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.nweights, batch, a.t);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:        if(l.scales_gpu){
galaxy_detection/data/darknet/src/convolutional_kernels.cu:            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:        axpy_gpu(l.nweights, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:        axpy_gpu(l.nweights, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:        scal_gpu(l.nweights, momentum, l.weight_updates_gpu, 1);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:        axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:        scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:        if(l.scales_gpu){
galaxy_detection/data/darknet/src/convolutional_kernels.cu:            axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
galaxy_detection/data/darknet/src/convolutional_kernels.cu:            scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
galaxy_detection/data/darknet/src/cuda.c:int gpu_index = 0;
galaxy_detection/data/darknet/src/cuda.c:#ifdef GPU
galaxy_detection/data/darknet/src/cuda.c:#include "cuda.h"
galaxy_detection/data/darknet/src/cuda.c:void cuda_set_device(int n)
galaxy_detection/data/darknet/src/cuda.c:    gpu_index = n;
galaxy_detection/data/darknet/src/cuda.c:    cudaError_t status = cudaSetDevice(n);
galaxy_detection/data/darknet/src/cuda.c:int cuda_get_device()
galaxy_detection/data/darknet/src/cuda.c:    cudaError_t status = cudaGetDevice(&n);
galaxy_detection/data/darknet/src/cuda.c:void check_error(cudaError_t status)
galaxy_detection/data/darknet/src/cuda.c:    //cudaDeviceSynchronize();
galaxy_detection/data/darknet/src/cuda.c:    cudaError_t status2 = cudaGetLastError();
galaxy_detection/data/darknet/src/cuda.c:    if (status != cudaSuccess)
galaxy_detection/data/darknet/src/cuda.c:        const char *s = cudaGetErrorString(status);
galaxy_detection/data/darknet/src/cuda.c:        printf("CUDA Error: %s\n", s);
galaxy_detection/data/darknet/src/cuda.c:        snprintf(buffer, 256, "CUDA Error: %s", s);
galaxy_detection/data/darknet/src/cuda.c:    if (status2 != cudaSuccess)
galaxy_detection/data/darknet/src/cuda.c:        const char *s = cudaGetErrorString(status);
galaxy_detection/data/darknet/src/cuda.c:        printf("CUDA Error Prev: %s\n", s);
galaxy_detection/data/darknet/src/cuda.c:        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
galaxy_detection/data/darknet/src/cuda.c:dim3 cuda_gridsize(size_t n){
galaxy_detection/data/darknet/src/cuda.c:    int i = cuda_get_device();
galaxy_detection/data/darknet/src/cuda.c:    int i = cuda_get_device();
galaxy_detection/data/darknet/src/cuda.c:float *cuda_make_array(float *x, size_t n)
galaxy_detection/data/darknet/src/cuda.c:    float *x_gpu;
galaxy_detection/data/darknet/src/cuda.c:    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
galaxy_detection/data/darknet/src/cuda.c:        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
galaxy_detection/data/darknet/src/cuda.c:        fill_gpu(n, 0, x_gpu, 1);
galaxy_detection/data/darknet/src/cuda.c:    if(!x_gpu) error("Cuda malloc failed\n");
galaxy_detection/data/darknet/src/cuda.c:    return x_gpu;
galaxy_detection/data/darknet/src/cuda.c:void cuda_random(float *x_gpu, size_t n)
galaxy_detection/data/darknet/src/cuda.c:    int i = cuda_get_device();
galaxy_detection/data/darknet/src/cuda.c:    curandGenerateUniform(gen[i], x_gpu, n);
galaxy_detection/data/darknet/src/cuda.c:    check_error(cudaPeekAtLastError());
galaxy_detection/data/darknet/src/cuda.c:float cuda_compare(float *x_gpu, float *x, size_t n, char *s)
galaxy_detection/data/darknet/src/cuda.c:    cuda_pull_array(x_gpu, tmp, n);
galaxy_detection/data/darknet/src/cuda.c:int *cuda_make_int_array(int *x, size_t n)
galaxy_detection/data/darknet/src/cuda.c:    int *x_gpu;
galaxy_detection/data/darknet/src/cuda.c:    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
galaxy_detection/data/darknet/src/cuda.c:        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
galaxy_detection/data/darknet/src/cuda.c:    if(!x_gpu) error("Cuda malloc failed\n");
galaxy_detection/data/darknet/src/cuda.c:    return x_gpu;
galaxy_detection/data/darknet/src/cuda.c:void cuda_free(float *x_gpu)
galaxy_detection/data/darknet/src/cuda.c:    cudaError_t status = cudaFree(x_gpu);
galaxy_detection/data/darknet/src/cuda.c:void cuda_push_array(float *x_gpu, float *x, size_t n)
galaxy_detection/data/darknet/src/cuda.c:    cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
galaxy_detection/data/darknet/src/cuda.c:void cuda_pull_array(float *x_gpu, float *x, size_t n)
galaxy_detection/data/darknet/src/cuda.c:    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
galaxy_detection/data/darknet/src/cuda.c:float cuda_mag_array(float *x_gpu, size_t n)
galaxy_detection/data/darknet/src/cuda.c:    cuda_pull_array(x_gpu, temp, n);
galaxy_detection/data/darknet/src/route_layer.c:#include "cuda.h"
galaxy_detection/data/darknet/src/route_layer.c:    #ifdef GPU
galaxy_detection/data/darknet/src/route_layer.c:    l.forward_gpu = forward_route_layer_gpu;
galaxy_detection/data/darknet/src/route_layer.c:    l.backward_gpu = backward_route_layer_gpu;
galaxy_detection/data/darknet/src/route_layer.c:    l.delta_gpu =  cuda_make_array(l.delta, outputs*batch);
galaxy_detection/data/darknet/src/route_layer.c:    l.output_gpu = cuda_make_array(l.output, outputs*batch);
galaxy_detection/data/darknet/src/route_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/route_layer.c:    cuda_free(l->output_gpu);
galaxy_detection/data/darknet/src/route_layer.c:    cuda_free(l->delta_gpu);
galaxy_detection/data/darknet/src/route_layer.c:    l->output_gpu  = cuda_make_array(l->output, l->outputs*l->batch);
galaxy_detection/data/darknet/src/route_layer.c:    l->delta_gpu   = cuda_make_array(l->delta,  l->outputs*l->batch);
galaxy_detection/data/darknet/src/route_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/route_layer.c:void forward_route_layer_gpu(const route_layer l, network net)
galaxy_detection/data/darknet/src/route_layer.c:        float *input = net.layers[index].output_gpu;
galaxy_detection/data/darknet/src/route_layer.c:            copy_gpu(input_size, input + j*input_size, 1, l.output_gpu + offset + j*l.outputs, 1);
galaxy_detection/data/darknet/src/route_layer.c:void backward_route_layer_gpu(const route_layer l, network net)
galaxy_detection/data/darknet/src/route_layer.c:        float *delta = net.layers[index].delta_gpu;
galaxy_detection/data/darknet/src/route_layer.c:            axpy_gpu(input_size, 1, l.delta_gpu + offset + j*l.outputs, 1, delta + j*input_size, 1);
galaxy_detection/data/darknet/src/avgpool_layer.c:#include "cuda.h"
galaxy_detection/data/darknet/src/avgpool_layer.c:    #ifdef GPU
galaxy_detection/data/darknet/src/avgpool_layer.c:    l.forward_gpu = forward_avgpool_layer_gpu;
galaxy_detection/data/darknet/src/avgpool_layer.c:    l.backward_gpu = backward_avgpool_layer_gpu;
galaxy_detection/data/darknet/src/avgpool_layer.c:    l.output_gpu  = cuda_make_array(l.output, output_size);
galaxy_detection/data/darknet/src/avgpool_layer.c:    l.delta_gpu   = cuda_make_array(l.delta, output_size);
galaxy_detection/data/darknet/src/gru_layer.h:#ifdef GPU
galaxy_detection/data/darknet/src/gru_layer.h:void forward_gru_layer_gpu(layer l, network state);
galaxy_detection/data/darknet/src/gru_layer.h:void backward_gru_layer_gpu(layer l, network state);
galaxy_detection/data/darknet/src/gru_layer.h:void update_gru_layer_gpu(layer l, update_args a);
galaxy_detection/data/darknet/src/cuda.h:#ifndef CUDA_H
galaxy_detection/data/darknet/src/cuda.h:#define CUDA_H
galaxy_detection/data/darknet/src/cuda.h:#ifdef GPU
galaxy_detection/data/darknet/src/cuda.h:void check_error(cudaError_t status);
galaxy_detection/data/darknet/src/cuda.h:int *cuda_make_int_array(int *x, size_t n);
galaxy_detection/data/darknet/src/cuda.h:void cuda_random(float *x_gpu, size_t n);
galaxy_detection/data/darknet/src/cuda.h:float cuda_compare(float *x_gpu, float *x, size_t n, char *s);
galaxy_detection/data/darknet/src/cuda.h:dim3 cuda_gridsize(size_t n);
galaxy_detection/data/darknet/src/convolutional_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/convolutional_layer.c:    swap = l->weights_gpu;
galaxy_detection/data/darknet/src/convolutional_layer.c:    l->weights_gpu = l->binary_weights_gpu;
galaxy_detection/data/darknet/src/convolutional_layer.c:    l->binary_weights_gpu = swap;
galaxy_detection/data/darknet/src/convolutional_layer.c:    if(gpu_index >= 0){
galaxy_detection/data/darknet/src/convolutional_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/convolutional_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/convolutional_layer.c:    l.forward_gpu = forward_convolutional_layer_gpu;
galaxy_detection/data/darknet/src/convolutional_layer.c:    l.backward_gpu = backward_convolutional_layer_gpu;
galaxy_detection/data/darknet/src/convolutional_layer.c:    l.update_gpu = update_convolutional_layer_gpu;
galaxy_detection/data/darknet/src/convolutional_layer.c:    if(gpu_index >= 0){
galaxy_detection/data/darknet/src/convolutional_layer.c:            l.m_gpu = cuda_make_array(l.m, l.nweights);
galaxy_detection/data/darknet/src/convolutional_layer.c:            l.v_gpu = cuda_make_array(l.v, l.nweights);
galaxy_detection/data/darknet/src/convolutional_layer.c:            l.bias_m_gpu = cuda_make_array(l.bias_m, n);
galaxy_detection/data/darknet/src/convolutional_layer.c:            l.bias_v_gpu = cuda_make_array(l.bias_v, n);
galaxy_detection/data/darknet/src/convolutional_layer.c:            l.scale_m_gpu = cuda_make_array(l.scale_m, n);
galaxy_detection/data/darknet/src/convolutional_layer.c:            l.scale_v_gpu = cuda_make_array(l.scale_v, n);
galaxy_detection/data/darknet/src/convolutional_layer.c:        l.weights_gpu = cuda_make_array(l.weights, l.nweights);
galaxy_detection/data/darknet/src/convolutional_layer.c:        l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);
galaxy_detection/data/darknet/src/convolutional_layer.c:        l.biases_gpu = cuda_make_array(l.biases, n);
galaxy_detection/data/darknet/src/convolutional_layer.c:        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);
galaxy_detection/data/darknet/src/convolutional_layer.c:        l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
galaxy_detection/data/darknet/src/convolutional_layer.c:        l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
galaxy_detection/data/darknet/src/convolutional_layer.c:            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
galaxy_detection/data/darknet/src/convolutional_layer.c:            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
galaxy_detection/data/darknet/src/convolutional_layer.c:            l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
galaxy_detection/data/darknet/src/convolutional_layer.c:            l.mean_gpu = cuda_make_array(l.mean, n);
galaxy_detection/data/darknet/src/convolutional_layer.c:            l.variance_gpu = cuda_make_array(l.variance, n);
galaxy_detection/data/darknet/src/convolutional_layer.c:            l.rolling_mean_gpu = cuda_make_array(l.mean, n);
galaxy_detection/data/darknet/src/convolutional_layer.c:            l.rolling_variance_gpu = cuda_make_array(l.variance, n);
galaxy_detection/data/darknet/src/convolutional_layer.c:            l.mean_delta_gpu = cuda_make_array(l.mean, n);
galaxy_detection/data/darknet/src/convolutional_layer.c:            l.variance_delta_gpu = cuda_make_array(l.variance, n);
galaxy_detection/data/darknet/src/convolutional_layer.c:            l.scales_gpu = cuda_make_array(l.scales, n);
galaxy_detection/data/darknet/src/convolutional_layer.c:            l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);
galaxy_detection/data/darknet/src/convolutional_layer.c:            l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
galaxy_detection/data/darknet/src/convolutional_layer.c:            l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
galaxy_detection/data/darknet/src/convolutional_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/convolutional_layer.c:    cuda_free(l->delta_gpu);
galaxy_detection/data/darknet/src/convolutional_layer.c:    cuda_free(l->output_gpu);
galaxy_detection/data/darknet/src/convolutional_layer.c:    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
galaxy_detection/data/darknet/src/convolutional_layer.c:    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);
galaxy_detection/data/darknet/src/convolutional_layer.c:        cuda_free(l->x_gpu);
galaxy_detection/data/darknet/src/convolutional_layer.c:        cuda_free(l->x_norm_gpu);
galaxy_detection/data/darknet/src/convolutional_layer.c:        l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
galaxy_detection/data/darknet/src/convolutional_layer.c:        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
galaxy_detection/data/darknet/src/batchnorm_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/batchnorm_layer.c:    l.forward_gpu = forward_batchnorm_layer_gpu;
galaxy_detection/data/darknet/src/batchnorm_layer.c:    l.backward_gpu = backward_batchnorm_layer_gpu;
galaxy_detection/data/darknet/src/batchnorm_layer.c:    l.output_gpu =  cuda_make_array(l.output, h * w * c * batch);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    l.delta_gpu =   cuda_make_array(l.delta, h * w * c * batch);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    l.biases_gpu = cuda_make_array(l.biases, c);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    l.bias_updates_gpu = cuda_make_array(l.bias_updates, c);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    l.scales_gpu = cuda_make_array(l.scales, c);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    l.scale_updates_gpu = cuda_make_array(l.scale_updates, c);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    l.mean_gpu = cuda_make_array(l.mean, c);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    l.variance_gpu = cuda_make_array(l.variance, c);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    l.rolling_mean_gpu = cuda_make_array(l.mean, c);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    l.rolling_variance_gpu = cuda_make_array(l.variance, c);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    l.mean_delta_gpu = cuda_make_array(l.mean, c);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    l.variance_delta_gpu = cuda_make_array(l.variance, c);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    l.x_gpu = cuda_make_array(l.output, l.batch*l.outputs);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    l.x_norm_gpu = cuda_make_array(l.output, l.batch*l.outputs);
galaxy_detection/data/darknet/src/batchnorm_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/batchnorm_layer.c:    cuda_pull_array(l.scales_gpu, l.scales, l.c);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.c);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.c);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    cuda_push_array(l.scales_gpu, l.scales, l.c);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.c);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.c);
galaxy_detection/data/darknet/src/batchnorm_layer.c:void forward_batchnorm_layer_gpu(layer l, network net)
galaxy_detection/data/darknet/src/batchnorm_layer.c:    if(l.type == BATCHNORM) copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
galaxy_detection/data/darknet/src/batchnorm_layer.c:                l.x_gpu,
galaxy_detection/data/darknet/src/batchnorm_layer.c:                l.output_gpu,
galaxy_detection/data/darknet/src/batchnorm_layer.c:                l.scales_gpu,
galaxy_detection/data/darknet/src/batchnorm_layer.c:                l.biases_gpu,
galaxy_detection/data/darknet/src/batchnorm_layer.c:                l.rolling_mean_gpu,
galaxy_detection/data/darknet/src/batchnorm_layer.c:                l.rolling_variance_gpu,
galaxy_detection/data/darknet/src/batchnorm_layer.c:                l.mean_gpu,
galaxy_detection/data/darknet/src/batchnorm_layer.c:                l.variance_gpu);
galaxy_detection/data/darknet/src/batchnorm_layer.c:        fast_mean_gpu(l.output_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.mean_gpu);
galaxy_detection/data/darknet/src/batchnorm_layer.c:        fast_variance_gpu(l.output_gpu, l.mean_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.variance_gpu);
galaxy_detection/data/darknet/src/batchnorm_layer.c:        scal_gpu(l.out_c, .99, l.rolling_mean_gpu, 1);
galaxy_detection/data/darknet/src/batchnorm_layer.c:        axpy_gpu(l.out_c, .01, l.mean_gpu, 1, l.rolling_mean_gpu, 1);
galaxy_detection/data/darknet/src/batchnorm_layer.c:        scal_gpu(l.out_c, .99, l.rolling_variance_gpu, 1);
galaxy_detection/data/darknet/src/batchnorm_layer.c:        axpy_gpu(l.out_c, .01, l.variance_gpu, 1, l.rolling_variance_gpu, 1);
galaxy_detection/data/darknet/src/batchnorm_layer.c:        copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
galaxy_detection/data/darknet/src/batchnorm_layer.c:        normalize_gpu(l.output_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
galaxy_detection/data/darknet/src/batchnorm_layer.c:        copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.x_norm_gpu, 1);
galaxy_detection/data/darknet/src/batchnorm_layer.c:        scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
galaxy_detection/data/darknet/src/batchnorm_layer.c:        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
galaxy_detection/data/darknet/src/batchnorm_layer.c:        normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
galaxy_detection/data/darknet/src/batchnorm_layer.c:        scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
galaxy_detection/data/darknet/src/batchnorm_layer.c:        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
galaxy_detection/data/darknet/src/batchnorm_layer.c:void backward_batchnorm_layer_gpu(layer l, network net)
galaxy_detection/data/darknet/src/batchnorm_layer.c:        l.mean_gpu = l.rolling_mean_gpu;
galaxy_detection/data/darknet/src/batchnorm_layer.c:        l.variance_gpu = l.rolling_variance_gpu;
galaxy_detection/data/darknet/src/batchnorm_layer.c:            l.x_gpu,
galaxy_detection/data/darknet/src/batchnorm_layer.c:            l.delta_gpu,
galaxy_detection/data/darknet/src/batchnorm_layer.c:            l.x_norm_gpu,
galaxy_detection/data/darknet/src/batchnorm_layer.c:            l.scales_gpu,
galaxy_detection/data/darknet/src/batchnorm_layer.c:            l.scale_updates_gpu,
galaxy_detection/data/darknet/src/batchnorm_layer.c:            l.bias_updates_gpu,
galaxy_detection/data/darknet/src/batchnorm_layer.c:            l.mean_gpu,
galaxy_detection/data/darknet/src/batchnorm_layer.c:            l.variance_gpu);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    copy_gpu(l.outputs*l.batch, l.x_norm_gpu, 1, l.delta_gpu, 1);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    backward_scale_gpu(l.x_norm_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates_gpu);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    scale_bias_gpu(l.delta_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    fast_mean_delta_gpu(l.delta_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta_gpu);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    fast_variance_delta_gpu(l.x_gpu, l.delta_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta_gpu);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    normalize_delta_gpu(l.x_gpu, l.mean_gpu, l.variance_gpu, l.mean_delta_gpu, l.variance_delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.delta_gpu);
galaxy_detection/data/darknet/src/batchnorm_layer.c:    if(l.type == BATCHNORM) copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, net.delta_gpu, 1);
galaxy_detection/data/darknet/src/shortcut_layer.c:#include "cuda.h"
galaxy_detection/data/darknet/src/shortcut_layer.c:    #ifdef GPU
galaxy_detection/data/darknet/src/shortcut_layer.c:    l.forward_gpu = forward_shortcut_layer_gpu;
galaxy_detection/data/darknet/src/shortcut_layer.c:    l.backward_gpu = backward_shortcut_layer_gpu;
galaxy_detection/data/darknet/src/shortcut_layer.c:    l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
galaxy_detection/data/darknet/src/shortcut_layer.c:    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
galaxy_detection/data/darknet/src/shortcut_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/shortcut_layer.c:void forward_shortcut_layer_gpu(const layer l, network net)
galaxy_detection/data/darknet/src/shortcut_layer.c:    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
galaxy_detection/data/darknet/src/shortcut_layer.c:    shortcut_gpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output_gpu, l.out_w, l.out_h, l.out_c, l.output_gpu);
galaxy_detection/data/darknet/src/shortcut_layer.c:    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
galaxy_detection/data/darknet/src/shortcut_layer.c:void backward_shortcut_layer_gpu(const layer l, network net)
galaxy_detection/data/darknet/src/shortcut_layer.c:    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
galaxy_detection/data/darknet/src/shortcut_layer.c:    axpy_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1, net.delta_gpu, 1);
galaxy_detection/data/darknet/src/shortcut_layer.c:    shortcut_gpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta_gpu, l.w, l.h, l.c, net.layers[l.index].delta_gpu);
galaxy_detection/data/darknet/src/normalization_layer.h:#ifdef GPU
galaxy_detection/data/darknet/src/normalization_layer.h:void forward_normalization_layer_gpu(const layer layer, network net);
galaxy_detection/data/darknet/src/normalization_layer.h:void backward_normalization_layer_gpu(const layer layer, network net);
galaxy_detection/data/darknet/src/im2col_kernels.cu:#include "cuda_runtime.h"
galaxy_detection/data/darknet/src/im2col_kernels.cu:#include "cuda.h"
galaxy_detection/data/darknet/src/im2col_kernels.cu:__global__ void im2col_gpu_kernel(const int n, const float* data_im,
galaxy_detection/data/darknet/src/im2col_kernels.cu:void im2col_gpu(float *im,
galaxy_detection/data/darknet/src/im2col_kernels.cu:    im2col_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
galaxy_detection/data/darknet/src/convolutional_layer.h:#include "cuda.h"
galaxy_detection/data/darknet/src/convolutional_layer.h:#ifdef GPU
galaxy_detection/data/darknet/src/convolutional_layer.h:void forward_convolutional_layer_gpu(convolutional_layer layer, network net);
galaxy_detection/data/darknet/src/convolutional_layer.h:void backward_convolutional_layer_gpu(convolutional_layer layer, network net);
galaxy_detection/data/darknet/src/convolutional_layer.h:void update_convolutional_layer_gpu(convolutional_layer layer, update_args a);
galaxy_detection/data/darknet/src/convolutional_layer.h:void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
galaxy_detection/data/darknet/src/convolutional_layer.h:void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);
galaxy_detection/data/darknet/src/convolutional_layer.h:void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t);
galaxy_detection/data/darknet/src/connected_layer.h:#ifdef GPU
galaxy_detection/data/darknet/src/connected_layer.h:void forward_connected_layer_gpu(layer l, network net);
galaxy_detection/data/darknet/src/connected_layer.h:void backward_connected_layer_gpu(layer l, network net);
galaxy_detection/data/darknet/src/connected_layer.h:void update_connected_layer_gpu(layer l, update_args a);
galaxy_detection/data/darknet/src/im2col.h:#ifdef GPU
galaxy_detection/data/darknet/src/im2col.h:void im2col_gpu(float *im,
galaxy_detection/data/darknet/src/local_layer.h:#include "cuda.h"
galaxy_detection/data/darknet/src/local_layer.h:#ifdef GPU
galaxy_detection/data/darknet/src/local_layer.h:void forward_local_layer_gpu(local_layer layer, network net);
galaxy_detection/data/darknet/src/local_layer.h:void backward_local_layer_gpu(local_layer layer, network net);
galaxy_detection/data/darknet/src/local_layer.h:void update_local_layer_gpu(local_layer layer, update_args a);
galaxy_detection/data/darknet/src/deconvolutional_layer.h:#include "cuda.h"
galaxy_detection/data/darknet/src/deconvolutional_layer.h:#ifdef GPU
galaxy_detection/data/darknet/src/deconvolutional_layer.h:void forward_deconvolutional_layer_gpu(layer l, network net);
galaxy_detection/data/darknet/src/deconvolutional_layer.h:void backward_deconvolutional_layer_gpu(layer l, network net);
galaxy_detection/data/darknet/src/deconvolutional_layer.h:void update_deconvolutional_layer_gpu(layer l, update_args a);
galaxy_detection/data/darknet/src/detection_layer.h:#ifdef GPU
galaxy_detection/data/darknet/src/detection_layer.h:void forward_detection_layer_gpu(const detection_layer l, network net);
galaxy_detection/data/darknet/src/detection_layer.h:void backward_detection_layer_gpu(detection_layer l, network net);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/deconvolutional_layer.c:    l.forward_gpu = forward_deconvolutional_layer_gpu;
galaxy_detection/data/darknet/src/deconvolutional_layer.c:    l.backward_gpu = backward_deconvolutional_layer_gpu;
galaxy_detection/data/darknet/src/deconvolutional_layer.c:    l.update_gpu = update_deconvolutional_layer_gpu;
galaxy_detection/data/darknet/src/deconvolutional_layer.c:    if(gpu_index >= 0){
galaxy_detection/data/darknet/src/deconvolutional_layer.c:            l.m_gpu = cuda_make_array(l.m, c*n*size*size);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:            l.v_gpu = cuda_make_array(l.v, c*n*size*size);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:            l.bias_m_gpu = cuda_make_array(l.bias_m, n);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:            l.bias_v_gpu = cuda_make_array(l.bias_v, n);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:            l.scale_m_gpu = cuda_make_array(l.scale_m, n);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:            l.scale_v_gpu = cuda_make_array(l.scale_v, n);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:        l.weights_gpu = cuda_make_array(l.weights, c*n*size*size);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:        l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*n*size*size);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:        l.biases_gpu = cuda_make_array(l.biases, n);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:        l.delta_gpu = cuda_make_array(l.delta, l.batch*l.out_h*l.out_w*n);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:        l.output_gpu = cuda_make_array(l.output, l.batch*l.out_h*l.out_w*n);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:            l.mean_gpu = cuda_make_array(0, n);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:            l.variance_gpu = cuda_make_array(0, n);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:            l.rolling_mean_gpu = cuda_make_array(0, n);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:            l.rolling_variance_gpu = cuda_make_array(0, n);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:            l.mean_delta_gpu = cuda_make_array(0, n);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:            l.variance_delta_gpu = cuda_make_array(0, n);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:            l.scales_gpu = cuda_make_array(0, n);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:            l.scale_updates_gpu = cuda_make_array(0, n);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:            l.x_gpu = cuda_make_array(0, l.batch*l.out_h*l.out_w*n);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:            l.x_norm_gpu = cuda_make_array(0, l.batch*l.out_h*l.out_w*n);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/deconvolutional_layer.c:    cuda_free(l->delta_gpu);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:    cuda_free(l->output_gpu);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:        cuda_free(l->x_gpu);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:        cuda_free(l->x_norm_gpu);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:        l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
galaxy_detection/data/darknet/src/deconvolutional_layer.c:        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
galaxy_detection/data/darknet/src/cost_layer.c:#include "cuda.h"
galaxy_detection/data/darknet/src/cost_layer.c:    #ifdef GPU
galaxy_detection/data/darknet/src/cost_layer.c:    l.forward_gpu = forward_cost_layer_gpu;
galaxy_detection/data/darknet/src/cost_layer.c:    l.backward_gpu = backward_cost_layer_gpu;
galaxy_detection/data/darknet/src/cost_layer.c:    l.delta_gpu = cuda_make_array(l.output, inputs*batch);
galaxy_detection/data/darknet/src/cost_layer.c:    l.output_gpu = cuda_make_array(l.delta, inputs*batch);
galaxy_detection/data/darknet/src/cost_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/cost_layer.c:    cuda_free(l->delta_gpu);
galaxy_detection/data/darknet/src/cost_layer.c:    cuda_free(l->output_gpu);
galaxy_detection/data/darknet/src/cost_layer.c:    l->delta_gpu = cuda_make_array(l->delta, inputs*l->batch);
galaxy_detection/data/darknet/src/cost_layer.c:    l->output_gpu = cuda_make_array(l->output, inputs*l->batch);
galaxy_detection/data/darknet/src/cost_layer.c:#ifdef GPU
galaxy_detection/data/darknet/src/cost_layer.c:    cuda_pull_array(l.delta_gpu, l.delta, l.batch*l.inputs);
galaxy_detection/data/darknet/src/cost_layer.c:    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
galaxy_detection/data/darknet/src/cost_layer.c:void forward_cost_layer_gpu(cost_layer l, network net)
galaxy_detection/data/darknet/src/cost_layer.c:    if (!net.truth_gpu) return;
galaxy_detection/data/darknet/src/cost_layer.c:        scal_gpu(l.batch*l.inputs, (1-l.smooth), net.truth_gpu, 1);
galaxy_detection/data/darknet/src/cost_layer.c:        add_gpu(l.batch*l.inputs, l.smooth * 1./l.inputs, net.truth_gpu, 1);
galaxy_detection/data/darknet/src/cost_layer.c:        mask_gpu(l.batch*l.inputs, net.input_gpu, SECRET_NUM, net.truth_gpu);
galaxy_detection/data/darknet/src/cost_layer.c:        smooth_l1_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
galaxy_detection/data/darknet/src/cost_layer.c:        l1_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
galaxy_detection/data/darknet/src/cost_layer.c:        l2_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
galaxy_detection/data/darknet/src/cost_layer.c:        scale_mask_gpu(l.batch*l.inputs, l.delta_gpu, 0, net.truth_gpu, l.noobject_scale);
galaxy_detection/data/darknet/src/cost_layer.c:        scale_mask_gpu(l.batch*l.inputs, l.output_gpu, 0, net.truth_gpu, l.noobject_scale);
galaxy_detection/data/darknet/src/cost_layer.c:        cuda_pull_array(l.delta_gpu, l.delta, l.batch*l.inputs);
galaxy_detection/data/darknet/src/cost_layer.c:        supp_gpu(l.batch*l.inputs, thresh, l.delta_gpu, 1);
galaxy_detection/data/darknet/src/cost_layer.c:        supp_gpu(l.batch*l.inputs, l.thresh*1./l.inputs, l.delta_gpu, 1);
galaxy_detection/data/darknet/src/cost_layer.c:    cuda_pull_array(l.output_gpu, l.output, l.batch*l.inputs);
galaxy_detection/data/darknet/src/cost_layer.c:void backward_cost_layer_gpu(const cost_layer l, network net)
galaxy_detection/data/darknet/src/cost_layer.c:    axpy_gpu(l.batch*l.inputs, l.scale, l.delta_gpu, 1, net.delta_gpu, 1);
galaxy_detection/data/darknet/src/activation_layer.h:#ifdef GPU
galaxy_detection/data/darknet/src/activation_layer.h:void forward_activation_layer_gpu(layer l, network net);
galaxy_detection/data/darknet/src/activation_layer.h:void backward_activation_layer_gpu(layer l, network net);
galaxy_detection/data/darknet/src/reorg_layer.h:#include "cuda.h"
galaxy_detection/data/darknet/src/reorg_layer.h:#ifdef GPU
galaxy_detection/data/darknet/src/reorg_layer.h:void forward_reorg_layer_gpu(layer l, network net);
galaxy_detection/data/darknet/src/reorg_layer.h:void backward_reorg_layer_gpu(layer l, network net);
galaxy_detection/README.md:To begin the training using two GPU cards you can use a command like:
galaxy_detection/README.md:$ ./darknet detector train cfg/sdss.data cfg/yolo.cfg darknet19_448.conv.23 -gpus 0,1
galaxy_detection/README.md:$ ./darknet detector train cfg/sdss.data cfg/yolo.cfg result/yolo.backup -gpus 0,1
galaxy_detection/README.md:$ ./darknet detector recall cfg/sdss.data cfg/yolo.cfg result/yolo_400.weights -gpus 0,1
galaxy_detection/README.md:$ ./darknet detector *action *path/to/.data *path/to/.cfh *path/to/weights -gpus *gpus to use
installation_guide.md:The following steps have been tested with Ubuntu 16.04 and Python 2.7. The GPU used is Nvidia GTX1060. For every requisite for AstroCV, you can follow the respective installation guide, but the main steps will still be included here.
installation_guide.md:Check you have the mentioned versions of CUDA, OpenCV and cuDNN. Darknet can have issues with more recent versions of those libraries.
installation_guide.md:## Installing CUDA 8.0
installation_guide.md:Now we will install CUDA 8.0. Before we begin, check that you have NVIDIA drivers installed with
installation_guide.md:nvidia-smi
installation_guide.md:If you don't have the NVIDIA drivers, open a terminal and run
installation_guide.md:sudo apt-get purge nvidia* 
installation_guide.md:sudo apt-get install nvidia-384 ##Replace 384 for the version you want. 384 was the latest version up to January 2018.
installation_guide.md:Download the CUDA .dev files from [https://developer.nvidia.com/cuda-80-download-archive]. You might want to sign up with an account, since you'll need it later for CUDNN installation. From the terminal, go to the path where you downloaded the files and run
installation_guide.md:sudo dpkg -i cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64.deb
installation_guide.md:sudo apt-get install cuda
installation_guide.md:CUDA is installed. To verify that, run the following code:
installation_guide.md:If Ubuntu doesn't find CUDA, run the next file:
installation_guide.md:export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}" into ~/.bashrc
installation_guide.md:Download Runtime Library, Developer Library and Code Samples .deb files from [https://developer.nvidia.com/rdp/cudnn-download]. You want to download cuDNN 5.1 for CUDA 8.0.
installation_guide.md:sudo dpkg -i libcudnn5_5.1.10-1+cuda8.0_amd64.deb 
installation_guide.md:sudo dpkg -i libcudnn5-dev_5.1.10-1+cuda8.0_amd64.deb 
installation_guide.md:sudo dpkg -i libcudnn5-doc_5.1.10-1+cuda8.0_amd64.deb 
installation_guide.md:cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D FORCE_VTK=ON -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D WITH_CUBLAS=ON -D CUDA_NVCC_FLAGS="-D_FORCE_INLINES" -D WITH_GDAL=ON -D WITH_XINE=ON -D BUILD_EXAMPLES=ON ..
installation_guide.md:Set GPU=1, CUDNN=1, DEBUG=0 and OPENCV=0.
installation_guide.md:Replace NVCC = nvcc for NVCC = /usr/local/cuda-8.0/bin/nvcc
installation_guide.md:python setup_gpu.py build
installation_guide.md:sudo python setup_gpu.py install

```
