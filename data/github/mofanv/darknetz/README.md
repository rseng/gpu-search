# https://github.com/mofanv/darknetz

```console
host/Makefile:OBJ = tcp.transfer.o diffprivate.o gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o detection_layer.o route_layer.o upsample_layer.o box.o normalization_layer.o avgpool_layer.o layer.o local_layer.o shortcut_layer.o logistic_layer.o activation_layer.o rnn_layer.o gru_layer.o crnn_layer.o demo.o batchnorm_layer.o region_layer.o reorg_layer.o tree.o  lstm_layer.o l2norm_layer.o yolo_layer.o iseg_layer.o
host/Makefile_changed_bak:OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o detection_layer.o route_layer.o upsample_layer.o box.o normalization_layer.o avgpool_layer.o layer.o local_layer.o shortcut_layer.o logistic_layer.o activation_layer.o rnn_layer.o gru_layer.o crnn_layer.o demo.o batchnorm_layer.o region_layer.o reorg_layer.o tree.o  lstm_layer.o l2norm_layer.o yolo_layer.o iseg_layer.o image_opencv.o
host/include/darknet.h:#ifdef GPU
host/include/darknet.h:    #include "cuda_runtime.h"
host/include/darknet.h:extern int gpu_index;
host/include/darknet.h:    void (*forward_gpu)   (struct layer, struct network);
host/include/darknet.h:    void (*backward_gpu)  (struct layer, struct network);
host/include/darknet.h:    void (*update_gpu)    (struct layer, update_args);
host/include/darknet.h:#ifdef GPU
host/include/darknet.h:    int *indexes_gpu;
host/include/darknet.h:    float *z_gpu;
host/include/darknet.h:    float *r_gpu;
host/include/darknet.h:    float *h_gpu;
host/include/darknet.h:    float *temp_gpu;
host/include/darknet.h:    float *temp2_gpu;
host/include/darknet.h:    float *temp3_gpu;
host/include/darknet.h:    float *dh_gpu;
host/include/darknet.h:    float *hh_gpu;
host/include/darknet.h:    float *prev_cell_gpu;
host/include/darknet.h:    float *cell_gpu;
host/include/darknet.h:    float *f_gpu;
host/include/darknet.h:    float *i_gpu;
host/include/darknet.h:    float *g_gpu;
host/include/darknet.h:    float *o_gpu;
host/include/darknet.h:    float *c_gpu;
host/include/darknet.h:    float *dc_gpu; 
host/include/darknet.h:    float *m_gpu;
host/include/darknet.h:    float *v_gpu;
host/include/darknet.h:    float *bias_m_gpu;
host/include/darknet.h:    float *scale_m_gpu;
host/include/darknet.h:    float *bias_v_gpu;
host/include/darknet.h:    float *scale_v_gpu;
host/include/darknet.h:    float * combine_gpu;
host/include/darknet.h:    float * combine_delta_gpu;
host/include/darknet.h:    float * prev_state_gpu;
host/include/darknet.h:    float * forgot_state_gpu;
host/include/darknet.h:    float * forgot_delta_gpu;
host/include/darknet.h:    float * state_gpu;
host/include/darknet.h:    float * state_delta_gpu;
host/include/darknet.h:    float * gate_gpu;
host/include/darknet.h:    float * gate_delta_gpu;
host/include/darknet.h:    float * save_gpu;
host/include/darknet.h:    float * save_delta_gpu;
host/include/darknet.h:    float * concat_gpu;
host/include/darknet.h:    float * concat_delta_gpu;
host/include/darknet.h:    float * binary_input_gpu;
host/include/darknet.h:    float * binary_weights_gpu;
host/include/darknet.h:    float * mean_gpu;
host/include/darknet.h:    float * variance_gpu;
host/include/darknet.h:    float * rolling_mean_gpu;
host/include/darknet.h:    float * rolling_variance_gpu;
host/include/darknet.h:    float * variance_delta_gpu;
host/include/darknet.h:    float * mean_delta_gpu;
host/include/darknet.h:    float * x_gpu;
host/include/darknet.h:    float * x_norm_gpu;
host/include/darknet.h:    float * weights_gpu;
host/include/darknet.h:    float * weight_updates_gpu;
host/include/darknet.h:    float * weight_change_gpu;
host/include/darknet.h:    float * biases_gpu;
host/include/darknet.h:    float * bias_updates_gpu;
host/include/darknet.h:    float * bias_change_gpu;
host/include/darknet.h:    float * scales_gpu;
host/include/darknet.h:    float * scale_updates_gpu;
host/include/darknet.h:    float * scale_change_gpu;
host/include/darknet.h:    float * output_gpu;
host/include/darknet.h:    float * loss_gpu;
host/include/darknet.h:    float * delta_gpu;
host/include/darknet.h:    float * rand_gpu;
host/include/darknet.h:    float * squared_gpu;
host/include/darknet.h:    float * norms_gpu;
host/include/darknet.h:    int gpu_index;
host/include/darknet.h:#ifdef GPU
host/include/darknet.h:    float *input_gpu;
host/include/darknet.h:    float *truth_gpu;
host/include/darknet.h:    float *delta_gpu;
host/include/darknet.h:    float *output_gpu;
host/include/darknet.h:#ifdef GPU
host/include/darknet.h:void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY);
host/include/darknet.h:void fill_gpu(int N, float ALPHA, float * X, int INCX);
host/include/darknet.h:void scal_gpu(int N, float ALPHA, float * X, int INCX);
host/include/darknet.h:void copy_gpu(int N, float * X, int INCX, float * Y, int INCY);
host/include/darknet.h:void cuda_set_device(int n);
host/include/darknet.h:void cuda_free(float *x_gpu);
host/include/darknet.h:float *cuda_make_array(float *x, size_t n);
host/include/darknet.h:void cuda_pull_array(float *x_gpu, float *x, size_t n);
host/include/darknet.h:float cuda_mag_array(float *x_gpu, size_t n);
host/include/darknet.h:void cuda_push_array(float *x_gpu, float *x, size_t n);
host/include/darknet.h:void forward_network_gpu(network *net);
host/include/darknet.h:void backward_network_gpu(network *net);
host/include/darknet.h:void update_network_gpu(network *net);
host/include/darknet.h:void harmless_update_network_gpu(network *net);
host/examples/darknet.c:    gpu_index = -1;
host/examples/darknet.c:    gpu_index = -1;
host/examples/darknet.c:    gpu_index = -1;
host/examples/darknet.c:    gpu_index = -1;
host/examples/darknet.c:    gpu_index = -1;
host/examples/darknet.c:    gpu_index = -1;
host/examples/darknet.c:    gpu_index = -1;
host/examples/darknet.c:    gpu_index = -1;
host/examples/darknet.c:    gpu_index = -1;
host/examples/darknet.c:    gpu_index = -1;
host/examples/darknet.c:    gpu_index = -1;
host/examples/darknet.c:    gpu_index = -1;
host/examples/darknet.c:    gpu_index = find_int_arg(argc, argv, "-i", 0);
host/examples/darknet.c:    if(find_arg(argc, argv, "-nogpu")) {
host/examples/darknet.c:        gpu_index = -1;
host/examples/darknet.c:#ifndef GPU
host/examples/darknet.c:    gpu_index = -1;
host/examples/darknet.c:    if(gpu_index >= 0){
host/examples/darknet.c:        cuda_set_device(gpu_index);
host/examples/go.c:void train_go(char *cfgfile, char *weightfile, char *filename, int *gpus, int ngpus, int clear)
host/examples/go.c:    printf("%d\n", ngpus);
host/examples/go.c:    network **nets = calloc(ngpus, sizeof(network*));
host/examples/go.c:    for(i = 0; i < ngpus; ++i){
host/examples/go.c:#ifdef GPU
host/examples/go.c:        cuda_set_device(gpus[i]);
host/examples/go.c:        nets[i]->learning_rate *= ngpus;
host/examples/go.c:        data train = random_go_moves(m, net->batch*net->subdivisions*ngpus);
host/examples/go.c:#ifdef GPU
host/examples/go.c:        if(ngpus == 1){
host/examples/go.c:            loss = train_networks(nets, ngpus, train, 10);
host/examples/go.c:    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
host/examples/go.c:    int *gpus = 0;
host/examples/go.c:    int gpu = 0;
host/examples/go.c:    int ngpus = 0;
host/examples/go.c:    if(gpu_list){
host/examples/go.c:        printf("%s\n", gpu_list);
host/examples/go.c:        int len = strlen(gpu_list);
host/examples/go.c:        ngpus = 1;
host/examples/go.c:            if (gpu_list[i] == ',') ++ngpus;
host/examples/go.c:        gpus = calloc(ngpus, sizeof(int));
host/examples/go.c:        for(i = 0; i < ngpus; ++i){
host/examples/go.c:            gpus[i] = atoi(gpu_list);
host/examples/go.c:            gpu_list = strchr(gpu_list, ',')+1;
host/examples/go.c:        gpu = gpu_index;
host/examples/go.c:        gpus = &gpu;
host/examples/go.c:        ngpus = 1;
host/examples/go.c:    if(0==strcmp(argv[2], "train")) train_go(cfg, weights, c2, gpus, ngpus, clear);
host/examples/segmenter.c:void train_segmenter(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int display)
host/examples/segmenter.c:    printf("%d\n", ngpus);
host/examples/segmenter.c:    network **nets = calloc(ngpus, sizeof(network*));
host/examples/segmenter.c:    for(i = 0; i < ngpus; ++i){
host/examples/segmenter.c:#ifdef GPU
host/examples/segmenter.c:        cuda_set_device(gpus[i]);
host/examples/segmenter.c:        nets[i]->learning_rate *= ngpus;
host/examples/segmenter.c:    int imgs = net->batch * net->subdivisions * ngpus;
host/examples/segmenter.c:#ifdef GPU
host/examples/segmenter.c:        if(ngpus == 1){
host/examples/segmenter.c:            loss = train_networks(nets, ngpus, train, 4);
host/examples/segmenter.c:    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
host/examples/segmenter.c:    int *gpus = 0;
host/examples/segmenter.c:    int gpu = 0;
host/examples/segmenter.c:    int ngpus = 0;
host/examples/segmenter.c:    if(gpu_list){
host/examples/segmenter.c:        printf("%s\n", gpu_list);
host/examples/segmenter.c:        int len = strlen(gpu_list);
host/examples/segmenter.c:        ngpus = 1;
host/examples/segmenter.c:            if (gpu_list[i] == ',') ++ngpus;
host/examples/segmenter.c:        gpus = calloc(ngpus, sizeof(int));
host/examples/segmenter.c:        for(i = 0; i < ngpus; ++i){
host/examples/segmenter.c:            gpus[i] = atoi(gpu_list);
host/examples/segmenter.c:            gpu_list = strchr(gpu_list, ',')+1;
host/examples/segmenter.c:        gpu = gpu_index;
host/examples/segmenter.c:        gpus = &gpu;
host/examples/segmenter.c:        ngpus = 1;
host/examples/segmenter.c:    else if(0==strcmp(argv[2], "train")) train_segmenter(data, cfg, weights, gpus, ngpus, clear, display);
host/examples/rnn.c:        #ifdef GPU
host/examples/rnn.c:        cuda_pull_array(l.output_gpu, l.output, l.outputs);
host/examples/nightmare.c:#ifdef GPU
host/examples/nightmare.c:    net->delta_gpu = cuda_make_array(delta.data, im.w*im.h*im.c);
host/examples/nightmare.c:    forward_network_gpu(net);
host/examples/nightmare.c:    copy_gpu(last.outputs, last.output_gpu, 1, last.delta_gpu, 1);
host/examples/nightmare.c:    cuda_pull_array(last.delta_gpu, last.delta, last.outputs);
host/examples/nightmare.c:    cuda_push_array(last.delta_gpu, last.delta, last.outputs);
host/examples/nightmare.c:    backward_network_gpu(net);
host/examples/nightmare.c:    cuda_pull_array(net->delta_gpu, delta.data, im.w*im.h*im.c);
host/examples/nightmare.c:    cuda_free(net->delta_gpu);
host/examples/nightmare.c:    net->delta_gpu = 0;
host/examples/nightmare.c:#ifdef GPU
host/examples/nightmare.c:        cuda_push_array(net->input_gpu, recon.data, recon.w*recon.h*recon.c);
host/examples/nightmare.c:        //cuda_push_array(net->truth_gpu, features, net->truths);
host/examples/nightmare.c:        net->delta_gpu = cuda_make_array(delta.data, delta.w*delta.h*delta.c);
host/examples/nightmare.c:        forward_network_gpu(net);
host/examples/nightmare.c:        cuda_push_array(l.delta_gpu, features, l.outputs);
host/examples/nightmare.c:        axpy_gpu(l.outputs, -1, l.output_gpu, 1, l.delta_gpu, 1);
host/examples/nightmare.c:        backward_network_gpu(net);
host/examples/nightmare.c:        cuda_pull_array(net->delta_gpu, delta.data, delta.w*delta.h*delta.c);
host/examples/nightmare.c:        cuda_free(net->delta_gpu);
host/examples/detector.c:void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
host/examples/detector.c:    network **nets = calloc(ngpus, sizeof(network));
host/examples/detector.c:    for(i = 0; i < ngpus; ++i){
host/examples/detector.c:#ifdef GPU
host/examples/detector.c:        cuda_set_device(gpus[i]);
host/examples/detector.c:        nets[i]->learning_rate *= ngpus;
host/examples/detector.c:    int imgs = net->batch * net->subdivisions * ngpus;
host/examples/detector.c:            for(i = 0; i < ngpus; ++i){
host/examples/detector.c:#ifdef GPU
host/examples/detector.c:        if(ngpus == 1){
host/examples/detector.c:            loss = train_networks(nets, ngpus, train, 4);
host/examples/detector.c:#ifdef GPU
host/examples/detector.c:            if(ngpus != 1) sync_nets(nets, ngpus, 0);
host/examples/detector.c:#ifdef GPU
host/examples/detector.c:            if(ngpus != 1) sync_nets(nets, ngpus, 0);
host/examples/detector.c:#ifdef GPU
host/examples/detector.c:    if(ngpus != 1) sync_nets(nets, ngpus, 0);
host/examples/detector.c:    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
host/examples/detector.c:    int *gpus = 0;
host/examples/detector.c:    int gpu = 0;
host/examples/detector.c:    int ngpus = 0;
host/examples/detector.c:    if(gpu_list){
host/examples/detector.c:        printf("%s\n", gpu_list);
host/examples/detector.c:        int len = strlen(gpu_list);
host/examples/detector.c:        ngpus = 1;
host/examples/detector.c:            if (gpu_list[i] == ',') ++ngpus;
host/examples/detector.c:        gpus = calloc(ngpus, sizeof(int));
host/examples/detector.c:        for(i = 0; i < ngpus; ++i){
host/examples/detector.c:            gpus[i] = atoi(gpu_list);
host/examples/detector.c:            gpu_list = strchr(gpu_list, ',')+1;
host/examples/detector.c:        gpu = gpu_index;
host/examples/detector.c:        gpus = &gpu;
host/examples/detector.c:        ngpus = 1;
host/examples/detector.c:    else if(0==strcmp(argv[2], "train")) train_detector(datacfg, cfg, weights, gpus, ngpus, clear);
host/examples/classifier.c:void train_classifier(char *datacfg, char *cfgfile, char *weightfile_o, int *gpus, int ngpus, int clear, bool fl)
host/examples/classifier.c:        printf("%d\n", ngpus);
host/examples/classifier.c:        network **nets = calloc(ngpus, sizeof(network*));
host/examples/classifier.c:        for(i = 0; i < ngpus; ++i) {
host/examples/classifier.c:#ifdef GPU
host/examples/classifier.c:                cuda_set_device(gpus[i]);
host/examples/classifier.c:                nets[i]->learning_rate *= ngpus;
host/examples/classifier.c:        int imgs = net->batch * net->subdivisions * ngpus;
host/examples/classifier.c:                        for(i = 0; i < ngpus; ++i) {
host/examples/classifier.c:#ifdef GPU
host/examples/classifier.c:                if(ngpus == 1) {
host/examples/classifier.c:                        loss = train_networks(nets, ngpus, train, 4);
host/examples/classifier.c:#ifdef GPU
host/examples/classifier.c:                cuda_pull_array(l.output_gpu, l.output, l.outputs);
host/examples/classifier.c:        char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
host/examples/classifier.c:        int ngpus;
host/examples/classifier.c:        int *gpus = read_intlist(gpu_list, &ngpus, gpu_index);
host/examples/classifier.c:        else if(0==strcmp(argv[2], "train")) train_classifier(data, cfg, weights, gpus, ngpus, clear, false);
host/examples/classifier.c:        else if(0==strcmp(argv[2], "train_fl")) train_classifier(data, cfg, weights, gpus, ngpus, clear, true);
host/examples/regressor.c:void train_regressor(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
host/examples/regressor.c:    printf("%d\n", ngpus);
host/examples/regressor.c:    network **nets = calloc(ngpus, sizeof(network*));
host/examples/regressor.c:    for(i = 0; i < ngpus; ++i){
host/examples/regressor.c:#ifdef GPU
host/examples/regressor.c:        cuda_set_device(gpus[i]);
host/examples/regressor.c:        nets[i]->learning_rate *= ngpus;
host/examples/regressor.c:    int imgs = net->batch * net->subdivisions * ngpus;
host/examples/regressor.c:#ifdef GPU
host/examples/regressor.c:        if(ngpus == 1){
host/examples/regressor.c:            loss = train_networks(nets, ngpus, train, 4);
host/examples/regressor.c:    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
host/examples/regressor.c:    int *gpus = 0;
host/examples/regressor.c:    int gpu = 0;
host/examples/regressor.c:    int ngpus = 0;
host/examples/regressor.c:    if(gpu_list){
host/examples/regressor.c:        printf("%s\n", gpu_list);
host/examples/regressor.c:        int len = strlen(gpu_list);
host/examples/regressor.c:        ngpus = 1;
host/examples/regressor.c:            if (gpu_list[i] == ',') ++ngpus;
host/examples/regressor.c:        gpus = calloc(ngpus, sizeof(int));
host/examples/regressor.c:        for(i = 0; i < ngpus; ++i){
host/examples/regressor.c:            gpus[i] = atoi(gpu_list);
host/examples/regressor.c:            gpu_list = strchr(gpu_list, ',')+1;
host/examples/regressor.c:        gpu = gpu_index;
host/examples/regressor.c:        gpus = &gpu;
host/examples/regressor.c:        ngpus = 1;
host/examples/regressor.c:    else if(0==strcmp(argv[2], "train")) train_regressor(data, cfg, weights, gpus, ngpus, clear);
host/examples/lsd.c:#ifdef GPU
host/examples/lsd.c:    fill_gpu(ay_size, .9, anet->truth_gpu, 1);
host/examples/lsd.c:    anet->delta_gpu = cuda_make_array(0, ax_size);
host/examples/lsd.c:    gstate.input = cuda_make_array(0, gx_size);
host/examples/lsd.c:            cuda_push_array(fstate.input, X, x_size);
host/examples/lsd.c:            cuda_push_array(gstate.input, X, gx_size);
host/examples/lsd.c:            forward_network_gpu(fnet, fstate);
host/examples/lsd.c:            float *feats = fnet->layers[fnet->n - 2].output_gpu;
host/examples/lsd.c:            copy_gpu(y_size, feats, 1, fstate.truth, 1);
host/examples/lsd.c:            forward_network_gpu(gnet, gstate);
host/examples/lsd.c:            float *gen = gnet->layers[gnet->n-1].output_gpu;
host/examples/lsd.c:            copy_gpu(x_size, gen, 1, fstate.input, 1);
host/examples/lsd.c:            fill_gpu(x_size, 0, fstate.delta, 1);
host/examples/lsd.c:            forward_network_gpu(fnet, fstate);
host/examples/lsd.c:            backward_network_gpu(fnet, fstate);
host/examples/lsd.c:            fill_gpu(ax_size, 0, astate.delta, 1);
host/examples/lsd.c:            forward_network_gpu(anet, astate);
host/examples/lsd.c:            backward_network_gpu(anet, astate);
host/examples/lsd.c:            float *delta = imlayer.delta_gpu;
host/examples/lsd.c:            fill_gpu(x_size, 0, delta, 1);
host/examples/lsd.c:            scal_gpu(x_size, 100, astate.delta, 1);
host/examples/lsd.c:            scal_gpu(x_size, .001, fstate.delta, 1);
host/examples/lsd.c:            axpy_gpu(x_size, 1, fstate.delta, 1, delta, 1);
host/examples/lsd.c:            axpy_gpu(x_size, 1, astate.delta, 1, delta, 1);
host/examples/lsd.c:            //fill_gpu(x_size, 0, delta, 1);
host/examples/lsd.c:            //cuda_push_array(delta, X, x_size);
host/examples/lsd.c:            //axpy_gpu(x_size, -1, imlayer.output_gpu, 1, delta, 1);
host/examples/lsd.c:            //printf("pix error: %f\n", cuda_mag_array(delta, x_size));
host/examples/lsd.c:            printf("fea error: %f\n", cuda_mag_array(fstate.delta, x_size));
host/examples/lsd.c:            printf("adv error: %f\n", cuda_mag_array(astate.delta, x_size));
host/examples/lsd.c:            //axpy_gpu(x_size, 1, astate.delta, 1, delta, 1);
host/examples/lsd.c:            backward_network_gpu(gnet, gstate);
host/examples/lsd.c:            cuda_pull_array(imlayer.output_gpu, imlayer.output, imlayer.outputs*imlayer.batch);
host/examples/lsd.c:        harmless_update_network_gpu(anet);
host/examples/lsd.c:        update_network_gpu(gnet);
host/examples/lsd.c:#ifdef GPU
host/examples/lsd.c:    gstate.input = cuda_make_array(0, x_size);
host/examples/lsd.c:    gstate.truth = cuda_make_array(0, y_size);
host/examples/lsd.c:    float *imerror = cuda_make_array(0, imlayer.outputs);
host/examples/lsd.c:    float *ones_gpu = cuda_make_array(0, ay_size);
host/examples/lsd.c:    fill_gpu(ay_size, .9, ones_gpu, 1);
host/examples/lsd.c:            cuda_push_array(gstate.input, graypixs, x_size);
host/examples/lsd.c:            cuda_push_array(gstate.truth, pixs, y_size);
host/examples/lsd.c:            forward_network_gpu(net, gstate);
host/examples/lsd.c:            fill_gpu(imlayer.outputs, 0, imerror, 1);
host/examples/lsd.c:            astate.input = imlayer.output_gpu;
host/examples/lsd.c:            astate.truth = ones_gpu;
host/examples/lsd.c:            forward_network_gpu(anet, astate);
host/examples/lsd.c:            backward_network_gpu(anet, astate);
host/examples/lsd.c:            scal_gpu(imlayer.outputs, .1, net->layers[net->n-1].delta_gpu, 1);
host/examples/lsd.c:            backward_network_gpu(net, gstate);
host/examples/lsd.c:            scal_gpu(imlayer.outputs, 1000, imerror, 1);
host/examples/lsd.c:            printf("realness %f\n", cuda_mag_array(imerror, imlayer.outputs));
host/examples/lsd.c:            printf("features %f\n", cuda_mag_array(net->layers[net->n-1].delta_gpu, imlayer.outputs));
host/examples/lsd.c:            axpy_gpu(imlayer.outputs, 1, imerror, 1, imlayer.delta_gpu, 1);
host/examples/lsd.c:            cuda_pull_array(imlayer.output_gpu, imlayer.output, imlayer.outputs*imlayer.batch);
host/examples/lsd.c:        harmless_update_network_gpu(anet);
host/examples/lsd.c:        update_network_gpu(net);
host/examples/lsd.c:        update_network_gpu(anet);
host/examples/lsd.c:#ifdef GPU
host/examples/lsd.c:    float *imerror = cuda_make_array(0, y_size);
host/examples/lsd.c:            fill_gpu(imlayer.outputs*imlayer.batch, 0, imerror, 1);
host/examples/lsd.c:            anet->delta_gpu = imerror;
host/examples/lsd.c:            scal_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1);
host/examples/lsd.c:            scal_gpu(imlayer.outputs*imlayer.batch, 0, gnet->layers[gnet->n-1].delta_gpu, 1);
host/examples/lsd.c:            axpy_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1, gnet->layers[gnet->n-1].delta_gpu, 1);
host/examples/lsd.c:        harmless_update_network_gpu(anet);
host/examples/lsd.c:        update_network_gpu(gnet);
host/examples/lsd.c:#ifdef GPU
host/examples/lsd.c:    float *imerror = cuda_make_array(0, y_size);
host/examples/lsd.c:            //cuda_push_array(gnet->input_gpu, gnet->input, x_size);
host/examples/lsd.c:            //cuda_push_array(gnet->truth_gpu, gnet->truth, y_size);
host/examples/lsd.c:            fill_gpu(imlayer.outputs*imlayer.batch, 0, imerror, 1);
host/examples/lsd.c:            anet->delta_gpu = imerror;
host/examples/lsd.c:            scal_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1);
host/examples/lsd.c:            scal_gpu(imlayer.outputs*imlayer.batch, 0, gnet->layers[gnet->n-1].delta_gpu, 1);
host/examples/lsd.c:            //printf("realness %f\n", cuda_mag_array(imerror, imlayer.outputs*imlayer.batch));
host/examples/lsd.c:            //printf("features %f\n", cuda_mag_array(gnet->layers[gnet->n-1].delta_gpu, imlayer.outputs*imlayer.batch));
host/examples/lsd.c:            axpy_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1, gnet->layers[gnet->n-1].delta_gpu, 1);
host/examples/lsd.c:               cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
host/examples/lsd.c:        harmless_update_network_gpu(anet);
host/examples/lsd.c:        update_network_gpu(gnet);
host/examples/lsd.c:#ifdef GPU
host/examples/lsd.c:    float *imerror = cuda_make_array(0, imlayer.outputs*imlayer.batch);
host/examples/lsd.c:            cuda_push_array(net->input_gpu, graypixs, net->inputs*net->batch);
host/examples/lsd.c:            cuda_push_array(net->truth_gpu, pixs, net->truths*net->batch);
host/examples/lsd.c:            forward_network_gpu(net);
host/examples/lsd.c:            fill_gpu(imlayer.outputs*imlayer.batch, 0, imerror, 1);
host/examples/lsd.c:            copy_gpu(anet->inputs*anet->batch, imlayer.output_gpu, 1, anet->input_gpu, 1);
host/examples/lsd.c:            fill_gpu(anet->inputs*anet->batch, .95, anet->truth_gpu, 1);
host/examples/lsd.c:            anet->delta_gpu = imerror;
host/examples/lsd.c:            forward_network_gpu(anet);
host/examples/lsd.c:            backward_network_gpu(anet);
host/examples/lsd.c:            scal_gpu(imlayer.outputs*imlayer.batch, 1./100., net->layers[net->n-1].delta_gpu, 1);
host/examples/lsd.c:            scal_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1);
host/examples/lsd.c:            printf("realness %f\n", cuda_mag_array(imerror, imlayer.outputs*imlayer.batch));
host/examples/lsd.c:            printf("features %f\n", cuda_mag_array(net->layers[net->n-1].delta_gpu, imlayer.outputs*imlayer.batch));
host/examples/lsd.c:            axpy_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1, net->layers[net->n-1].delta_gpu, 1);
host/examples/lsd.c:            backward_network_gpu(net);
host/examples/lsd.c:        harmless_update_network_gpu(anet);
host/examples/lsd.c:        update_network_gpu(net);
host/examples/lsd.c:#ifdef GPU
host/examples/lsd.c:gstate.input = cuda_make_array(0, x_size);
host/examples/lsd.c:float *imerror = cuda_make_array(0, imlayer.outputs);
host/examples/lsd.c:float *ones_gpu = cuda_make_array(0, ay_size);
host/examples/lsd.c:fill_gpu(ay_size, 1, ones_gpu, 1);
host/examples/lsd.c:        cuda_push_array(gstate.input, X, x_size);
host/examples/lsd.c:        forward_network_gpu(net, gstate);
host/examples/lsd.c:        fill_gpu(imlayer.outputs, 0, imerror, 1);
host/examples/lsd.c:        astate.input = imlayer.output_gpu;
host/examples/lsd.c:        astate.truth = ones_gpu;
host/examples/lsd.c:        forward_network_gpu(anet, astate);
host/examples/lsd.c:        backward_network_gpu(anet, astate);
host/examples/lsd.c:        scal_gpu(imlayer.outputs, 1, imerror, 1);
host/examples/lsd.c:        axpy_gpu(imlayer.outputs, 1, imerror, 1, imlayer.delta_gpu, 1);
host/examples/lsd.c:        backward_network_gpu(net, gstate);
host/examples/lsd.c:        printf("features %f\n", cuda_mag_array(imlayer.delta_gpu, imlayer.outputs));
host/examples/lsd.c:        printf("realness %f\n", cuda_mag_array(imerror, imlayer.outputs));
host/examples/lsd.c:        cuda_pull_array(imlayer.output_gpu, imlayer.output, imlayer.outputs*imlayer.batch);
host/examples/lsd.c:    harmless_update_network_gpu(anet);
host/examples/lsd.c:    update_network_gpu(net);
host/examples/lsd.c:    update_network_gpu(anet);
host/examples/instance-segmenter.c:void train_isegmenter(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int display)
host/examples/instance-segmenter.c:    printf("%d\n", ngpus);
host/examples/instance-segmenter.c:    network **nets = calloc(ngpus, sizeof(network*));
host/examples/instance-segmenter.c:    for(i = 0; i < ngpus; ++i){
host/examples/instance-segmenter.c:#ifdef GPU
host/examples/instance-segmenter.c:        cuda_set_device(gpus[i]);
host/examples/instance-segmenter.c:        nets[i]->learning_rate *= ngpus;
host/examples/instance-segmenter.c:    int imgs = net->batch * net->subdivisions * ngpus;
host/examples/instance-segmenter.c:#ifdef GPU
host/examples/instance-segmenter.c:        if(ngpus == 1){
host/examples/instance-segmenter.c:            loss = train_networks(nets, ngpus, train, 4);
host/examples/instance-segmenter.c:    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
host/examples/instance-segmenter.c:    int *gpus = 0;
host/examples/instance-segmenter.c:    int gpu = 0;
host/examples/instance-segmenter.c:    int ngpus = 0;
host/examples/instance-segmenter.c:    if(gpu_list){
host/examples/instance-segmenter.c:        printf("%s\n", gpu_list);
host/examples/instance-segmenter.c:        int len = strlen(gpu_list);
host/examples/instance-segmenter.c:        ngpus = 1;
host/examples/instance-segmenter.c:            if (gpu_list[i] == ',') ++ngpus;
host/examples/instance-segmenter.c:        gpus = calloc(ngpus, sizeof(int));
host/examples/instance-segmenter.c:        for(i = 0; i < ngpus; ++i){
host/examples/instance-segmenter.c:            gpus[i] = atoi(gpu_list);
host/examples/instance-segmenter.c:            gpu_list = strchr(gpu_list, ',')+1;
host/examples/instance-segmenter.c:        gpu = gpu_index;
host/examples/instance-segmenter.c:        gpus = &gpu;
host/examples/instance-segmenter.c:        ngpus = 1;
host/examples/instance-segmenter.c:    else if(0==strcmp(argv[2], "train")) train_isegmenter(data, cfg, weights, gpus, ngpus, clear, display);
host/src/deconvolutional_kernels.cu:#include "cuda_runtime.h"
host/src/deconvolutional_kernels.cu:#include "cuda.h"
host/src/deconvolutional_kernels.cu:extern "C" void forward_deconvolutional_layer_gpu(layer l, network net)
host/src/deconvolutional_kernels.cu:    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
host/src/deconvolutional_kernels.cu:        float *a = l.weights_gpu;
host/src/deconvolutional_kernels.cu:        float *b = net.input_gpu + i*l.c*l.h*l.w;
host/src/deconvolutional_kernels.cu:        gemm_gpu(1,0,m,n,k,1,a,m,b,n,0,c,n);
host/src/deconvolutional_kernels.cu:        col2im_gpu(net.workspace, l.out_c, l.out_h, l.out_w, l.size, l.stride, l.pad, l.output_gpu+i*l.outputs);
host/src/deconvolutional_kernels.cu:        forward_batchnorm_layer_gpu(l, net);
host/src/deconvolutional_kernels.cu:        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
host/src/deconvolutional_kernels.cu:    activate_array_gpu(l.output_gpu, l.batch*l.n*l.out_w*l.out_h, l.activation);
host/src/deconvolutional_kernels.cu:extern "C" void backward_deconvolutional_layer_gpu(layer l, network net)
host/src/deconvolutional_kernels.cu:    //constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
host/src/deconvolutional_kernels.cu:    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
host/src/deconvolutional_kernels.cu:        backward_batchnorm_layer_gpu(l, net);
host/src/deconvolutional_kernels.cu:        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
host/src/deconvolutional_kernels.cu:    //if(net.delta_gpu) memset(net.delta_gpu, 0, l.batch*l.h*l.w*l.c*sizeof(float));
host/src/deconvolutional_kernels.cu:        float *a = net.input_gpu + i*m*k;
host/src/deconvolutional_kernels.cu:        float *c = l.weight_updates_gpu;
host/src/deconvolutional_kernels.cu:        im2col_gpu(l.delta_gpu + i*l.outputs, l.out_c, l.out_h, l.out_w, 
host/src/deconvolutional_kernels.cu:        gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);
host/src/deconvolutional_kernels.cu:        if(net.delta_gpu){
host/src/deconvolutional_kernels.cu:            float *a = l.weights_gpu;
host/src/deconvolutional_kernels.cu:            float *c = net.delta_gpu + i*n*m;
host/src/deconvolutional_kernels.cu:            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
host/src/deconvolutional_kernels.cu:    cuda_pull_array(l.weights_gpu, l.weights, l.c*l.n*l.size*l.size);
host/src/deconvolutional_kernels.cu:    cuda_pull_array(l.biases_gpu, l.biases, l.n);
host/src/deconvolutional_kernels.cu:    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.c*l.n*l.size*l.size);
host/src/deconvolutional_kernels.cu:    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
host/src/deconvolutional_kernels.cu:        cuda_pull_array(l.scales_gpu, l.scales, l.n);
host/src/deconvolutional_kernels.cu:        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
host/src/deconvolutional_kernels.cu:        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
host/src/deconvolutional_kernels.cu:    cuda_push_array(l.weights_gpu, l.weights, l.c*l.n*l.size*l.size);
host/src/deconvolutional_kernels.cu:    cuda_push_array(l.biases_gpu, l.biases, l.n);
host/src/deconvolutional_kernels.cu:    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.c*l.n*l.size*l.size);
host/src/deconvolutional_kernels.cu:    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
host/src/deconvolutional_kernels.cu:        cuda_push_array(l.scales_gpu, l.scales, l.n);
host/src/deconvolutional_kernels.cu:        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
host/src/deconvolutional_kernels.cu:        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
host/src/deconvolutional_kernels.cu:void update_deconvolutional_layer_gpu(layer l, update_args a)
host/src/deconvolutional_kernels.cu:        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.nweights, batch, a.t);
host/src/deconvolutional_kernels.cu:        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
host/src/deconvolutional_kernels.cu:        if(l.scales_gpu){
host/src/deconvolutional_kernels.cu:            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
host/src/deconvolutional_kernels.cu:        axpy_gpu(l.nweights, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
host/src/deconvolutional_kernels.cu:        axpy_gpu(l.nweights, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
host/src/deconvolutional_kernels.cu:        scal_gpu(l.nweights, momentum, l.weight_updates_gpu, 1);
host/src/deconvolutional_kernels.cu:        axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
host/src/deconvolutional_kernels.cu:        scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);
host/src/deconvolutional_kernels.cu:        if(l.scales_gpu){
host/src/deconvolutional_kernels.cu:            axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
host/src/deconvolutional_kernels.cu:            scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
host/src/maxpool_layer.h:#include "cuda.h"
host/src/maxpool_layer.h:#ifdef GPU
host/src/maxpool_layer.h:void forward_maxpool_layer_gpu(maxpool_layer l, network net);
host/src/maxpool_layer.h:void backward_maxpool_layer_gpu(maxpool_layer l, network net);
host/src/crop_layer.c:#include "cuda.h"
host/src/crop_layer.c:void backward_crop_layer_gpu(const crop_layer l, network net){}
host/src/crop_layer.c:    #ifdef GPU
host/src/crop_layer.c:    l.forward_gpu = forward_crop_layer_gpu;
host/src/crop_layer.c:    l.backward_gpu = backward_crop_layer_gpu;
host/src/crop_layer.c:    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
host/src/crop_layer.c:    l.rand_gpu   = cuda_make_array(0, l.batch*8);
host/src/crop_layer.c:    #ifdef GPU
host/src/crop_layer.c:    cuda_free(l->output_gpu);
host/src/crop_layer.c:    l->output_gpu = cuda_make_array(l->output, l->outputs*l->batch);
host/src/softmax_layer.c:#include "cuda.h"
host/src/softmax_layer.c:    #ifdef GPU
host/src/softmax_layer.c:    l.forward_gpu = forward_softmax_layer_gpu;
host/src/softmax_layer.c:    l.backward_gpu = backward_softmax_layer_gpu;
host/src/softmax_layer.c:    l.output_gpu = cuda_make_array(l.output, inputs*batch);
host/src/softmax_layer.c:    l.loss_gpu = cuda_make_array(l.loss, inputs*batch);
host/src/softmax_layer.c:    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
host/src/softmax_layer.c:#ifdef GPU
host/src/softmax_layer.c:    cuda_pull_array(layer.output_gpu, layer.output, layer.inputs*layer.batch);
host/src/softmax_layer.c:void forward_softmax_layer_gpu(const softmax_layer l, network net)
host/src/softmax_layer.c:        softmax_tree(net.input_gpu, 1, l.batch, l.inputs, l.temperature, l.output_gpu, *l.softmax_tree);
host/src/softmax_layer.c:            softmax_gpu(net.input_gpu + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output_gpu + count);
host/src/softmax_layer.c:            softmax_gpu(net.input_gpu, l.c, l.batch*l.c, l.inputs/l.c, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu);
host/src/softmax_layer.c:            softmax_gpu(net.input_gpu, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output_gpu);
host/src/softmax_layer.c:        softmax_x_ent_gpu(l.batch*l.inputs, l.output_gpu, net.truth_gpu, l.delta_gpu, l.loss_gpu);
host/src/softmax_layer.c:            mask_gpu(l.batch*l.inputs, l.delta_gpu, SECRET_NUM, net.truth_gpu, 0);
host/src/softmax_layer.c:            mask_gpu(l.batch*l.inputs, l.loss_gpu, SECRET_NUM, net.truth_gpu, 0);
host/src/softmax_layer.c:        cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
host/src/softmax_layer.c:void backward_softmax_layer_gpu(const softmax_layer layer, network net)
host/src/softmax_layer.c:    axpy_gpu(layer.batch*layer.inputs, 1, layer.delta_gpu, 1, net.delta_gpu, 1);
host/src/image.c:#include "cuda.h"
host/src/avgpool_layer.h:#include "cuda.h"
host/src/avgpool_layer.h:#ifdef GPU
host/src/avgpool_layer.h:void forward_avgpool_layer_gpu(avgpool_layer l, network net);
host/src/avgpool_layer.h:void backward_avgpool_layer_gpu(avgpool_layer l, network net);
host/src/upsample_layer.c:#include "cuda.h"
host/src/upsample_layer.c:    #ifdef GPU
host/src/upsample_layer.c:    l.forward_gpu = forward_upsample_layer_gpu;
host/src/upsample_layer.c:    l.backward_gpu = backward_upsample_layer_gpu;
host/src/upsample_layer.c:    l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
host/src/upsample_layer.c:    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
host/src/upsample_layer.c:#ifdef GPU
host/src/upsample_layer.c:    cuda_free(l->output_gpu);
host/src/upsample_layer.c:    cuda_free(l->delta_gpu);
host/src/upsample_layer.c:    l->output_gpu  = cuda_make_array(l->output, l->outputs*l->batch);
host/src/upsample_layer.c:    l->delta_gpu   = cuda_make_array(l->delta,  l->outputs*l->batch);
host/src/upsample_layer.c:#ifdef GPU
host/src/upsample_layer.c:void forward_upsample_layer_gpu(const layer l, network net)
host/src/upsample_layer.c:    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
host/src/upsample_layer.c:        upsample_gpu(l.output_gpu, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, l.scale, net.input_gpu);
host/src/upsample_layer.c:        upsample_gpu(net.input_gpu, l.w, l.h, l.c, l.batch, l.stride, 1, l.scale, l.output_gpu);
host/src/upsample_layer.c:void backward_upsample_layer_gpu(const layer l, network net)
host/src/upsample_layer.c:        upsample_gpu(l.delta_gpu, l.out_w, l.out_h, l.c, l.batch, l.stride, 1, l.scale, net.delta_gpu);
host/src/upsample_layer.c:        upsample_gpu(net.delta_gpu, l.w, l.h, l.c, l.batch, l.stride, 0, l.scale, l.delta_gpu);
host/src/gemm.h:#ifdef GPU
host/src/gemm.h:void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
host/src/gemm.h:              float *A_gpu, int lda,
host/src/gemm.h:              float *B_gpu, int ldb,
host/src/gemm.h:              float *C_gpu, int ldc);
host/src/gemm.h:void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
host/src/network.h:#ifdef GPU
host/src/shortcut_layer.h:#ifdef GPU
host/src/shortcut_layer.h:void forward_shortcut_layer_gpu(const layer l, network net);
host/src/shortcut_layer.h:void backward_shortcut_layer_gpu(const layer l, network net);
host/src/crop_layer_kernels.cu:#include "cuda_runtime.h"
host/src/crop_layer_kernels.cu:#include "cuda.h"
host/src/crop_layer_kernels.cu:extern "C" void forward_crop_layer_gpu(crop_layer layer, network net)
host/src/crop_layer_kernels.cu:    cuda_random(layer.rand_gpu, layer.batch*8);
host/src/crop_layer_kernels.cu:    levels_image_kernel<<<cuda_gridsize(size), BLOCK>>>(net.input_gpu, layer.rand_gpu, layer.batch, layer.w, layer.h, net.train, layer.saturation, layer.exposure, translate, scale, layer.shift);
host/src/crop_layer_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/crop_layer_kernels.cu:    forward_crop_layer_kernel<<<cuda_gridsize(size), BLOCK>>>(net.input_gpu, layer.rand_gpu, size, layer.c, layer.h, layer.w, layer.out_h, layer.out_w, net.train, layer.flip, radians, layer.output_gpu);
host/src/crop_layer_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/crop_layer_kernels.cu:       cuda_pull_array(layer.output_gpu, layer.output, size);
host/src/upsample_layer.h:#ifdef GPU
host/src/upsample_layer.h:void forward_upsample_layer_gpu(const layer l, network net);
host/src/upsample_layer.h:void backward_upsample_layer_gpu(const layer l, network net);
host/src/connected_layer.c:#include "cuda.h"
host/src/connected_layer.c:#ifdef GPU
host/src/connected_layer.c:    l.forward_gpu = forward_connected_layer_gpu;
host/src/connected_layer.c:    l.backward_gpu = backward_connected_layer_gpu;
host/src/connected_layer.c:    l.update_gpu = update_connected_layer_gpu;
host/src/connected_layer.c:    l.weights_gpu = cuda_make_array(l.weights, outputs*inputs);
host/src/connected_layer.c:    l.biases_gpu = cuda_make_array(l.biases, outputs);
host/src/connected_layer.c:    l.weight_updates_gpu = cuda_make_array(l.weight_updates, outputs*inputs);
host/src/connected_layer.c:    l.bias_updates_gpu = cuda_make_array(l.bias_updates, outputs);
host/src/connected_layer.c:    l.output_gpu = cuda_make_array(l.output, outputs*batch);
host/src/connected_layer.c:    l.delta_gpu = cuda_make_array(l.delta, outputs*batch);
host/src/connected_layer.c:        l.m_gpu =       cuda_make_array(0, inputs*outputs);
host/src/connected_layer.c:        l.v_gpu =       cuda_make_array(0, inputs*outputs);
host/src/connected_layer.c:        l.bias_m_gpu =  cuda_make_array(0, outputs);
host/src/connected_layer.c:        l.bias_v_gpu =  cuda_make_array(0, outputs);
host/src/connected_layer.c:        l.scale_m_gpu = cuda_make_array(0, outputs);
host/src/connected_layer.c:        l.scale_v_gpu = cuda_make_array(0, outputs);
host/src/connected_layer.c:        l.mean_gpu = cuda_make_array(l.mean, outputs);
host/src/connected_layer.c:        l.variance_gpu = cuda_make_array(l.variance, outputs);
host/src/connected_layer.c:        l.rolling_mean_gpu = cuda_make_array(l.mean, outputs);
host/src/connected_layer.c:        l.rolling_variance_gpu = cuda_make_array(l.variance, outputs);
host/src/connected_layer.c:        l.mean_delta_gpu = cuda_make_array(l.mean, outputs);
host/src/connected_layer.c:        l.variance_delta_gpu = cuda_make_array(l.variance, outputs);
host/src/connected_layer.c:        l.scales_gpu = cuda_make_array(l.scales, outputs);
host/src/connected_layer.c:        l.scale_updates_gpu = cuda_make_array(l.scale_updates, outputs);
host/src/connected_layer.c:        l.x_gpu = cuda_make_array(l.output, l.batch*outputs);
host/src/connected_layer.c:        l.x_norm_gpu = cuda_make_array(l.output, l.batch*outputs);
host/src/connected_layer.c:#ifdef GPU
host/src/connected_layer.c:    cuda_pull_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
host/src/connected_layer.c:    cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
host/src/connected_layer.c:    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
host/src/connected_layer.c:    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
host/src/connected_layer.c:        cuda_pull_array(l.scales_gpu, l.scales, l.outputs);
host/src/connected_layer.c:        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
host/src/connected_layer.c:        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
host/src/connected_layer.c:    cuda_push_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
host/src/connected_layer.c:    cuda_push_array(l.biases_gpu, l.biases, l.outputs);
host/src/connected_layer.c:    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
host/src/connected_layer.c:    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
host/src/connected_layer.c:        cuda_push_array(l.scales_gpu, l.scales, l.outputs);
host/src/connected_layer.c:        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
host/src/connected_layer.c:        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
host/src/connected_layer.c:void update_connected_layer_gpu(layer l, update_args a)
host/src/connected_layer.c:        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.inputs*l.outputs, batch, a.t);
host/src/connected_layer.c:        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
host/src/connected_layer.c:        if(l.scales_gpu){
host/src/connected_layer.c:            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
host/src/connected_layer.c:        axpy_gpu(l.outputs, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
host/src/connected_layer.c:        scal_gpu(l.outputs, momentum, l.bias_updates_gpu, 1);
host/src/connected_layer.c:            axpy_gpu(l.outputs, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
host/src/connected_layer.c:            scal_gpu(l.outputs, momentum, l.scale_updates_gpu, 1);
host/src/connected_layer.c:        axpy_gpu(l.inputs*l.outputs, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
host/src/connected_layer.c:        axpy_gpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
host/src/connected_layer.c:        scal_gpu(l.inputs*l.outputs, momentum, l.weight_updates_gpu, 1);
host/src/connected_layer.c:void forward_connected_layer_gpu(layer l, network net)
host/src/connected_layer.c:    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
host/src/connected_layer.c:    float * a = net.input_gpu;
host/src/connected_layer.c:    float * b = l.weights_gpu;
host/src/connected_layer.c:    float * c = l.output_gpu;
host/src/connected_layer.c:    gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);
host/src/connected_layer.c:        forward_batchnorm_layer_gpu(l, net);
host/src/connected_layer.c:        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.outputs, 1);
host/src/connected_layer.c:    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
host/src/connected_layer.c:void backward_connected_layer_gpu(layer l, network net)
host/src/connected_layer.c:    constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
host/src/connected_layer.c:    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
host/src/connected_layer.c:        backward_batchnorm_layer_gpu(l, net);
host/src/connected_layer.c:        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.outputs, 1);
host/src/connected_layer.c:    float * a = l.delta_gpu;
host/src/connected_layer.c:    float * b = net.input_gpu;
host/src/connected_layer.c:    float * c = l.weight_updates_gpu;
host/src/connected_layer.c:    gemm_gpu(1,0,m,n,k,1,a,m,b,n,1,c,n);
host/src/connected_layer.c:    a = l.delta_gpu;
host/src/connected_layer.c:    b = l.weights_gpu;
host/src/connected_layer.c:    c = net.delta_gpu;
host/src/connected_layer.c:    if(c) gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
host/src/batchnorm_layer.h:#ifdef GPU
host/src/batchnorm_layer.h:void forward_batchnorm_layer_gpu(layer l, network net);
host/src/batchnorm_layer.h:void backward_batchnorm_layer_gpu(layer l, network net);
host/src/route_layer.h:#ifdef GPU
host/src/route_layer.h:void forward_route_layer_gpu(const route_layer l, network net);
host/src/route_layer.h:void backward_route_layer_gpu(const route_layer l, network net);
host/src/l2norm_layer.h:#ifdef GPU
host/src/l2norm_layer.h:void forward_l2norm_layer_gpu(const layer l, network net);
host/src/l2norm_layer.h:void backward_l2norm_layer_gpu(const layer l, network net);
host/src/region_layer.c:#include "cuda.h"
host/src/region_layer.c:#ifdef GPU
host/src/region_layer.c:    l.forward_gpu = forward_region_layer_gpu;
host/src/region_layer.c:    l.backward_gpu = backward_region_layer_gpu;
host/src/region_layer.c:    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
host/src/region_layer.c:    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
host/src/region_layer.c:#ifdef GPU
host/src/region_layer.c:    cuda_free(l->delta_gpu);
host/src/region_layer.c:    cuda_free(l->output_gpu);
host/src/region_layer.c:    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
host/src/region_layer.c:    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
host/src/region_layer.c:#ifndef GPU
host/src/region_layer.c:#ifdef GPU
host/src/region_layer.c:void forward_region_layer_gpu(const layer l, network net)
host/src/region_layer.c:    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
host/src/region_layer.c:            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
host/src/region_layer.c:                activate_array_gpu(l.output_gpu + index, (l.coords - 4)*l.w*l.h, LOGISTIC);
host/src/region_layer.c:            if(!l.background) activate_array_gpu(l.output_gpu + index,   l.w*l.h, LOGISTIC);
host/src/region_layer.c:            if(!l.softmax && !l.softmax_tree) activate_array_gpu(l.output_gpu + index, l.classes*l.w*l.h, LOGISTIC);
host/src/region_layer.c:        softmax_tree(net.input_gpu + index, l.w*l.h, l.batch*l.n, l.inputs/l.n, 1, l.output_gpu + index, *l.softmax_tree);
host/src/region_layer.c:        softmax_gpu(net.input_gpu + index, l.classes + l.background, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu + index);
host/src/region_layer.c:        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
host/src/region_layer.c:    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
host/src/region_layer.c:    //cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
host/src/region_layer.c:    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
host/src/region_layer.c:void backward_region_layer_gpu(const layer l, network net)
host/src/region_layer.c:            gradient_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC, l.delta_gpu + index);
host/src/region_layer.c:                gradient_array_gpu(l.output_gpu + index, (l.coords - 4)*l.w*l.h, LOGISTIC, l.delta_gpu + index);
host/src/region_layer.c:            if(!l.background) gradient_array_gpu(l.output_gpu + index,   l.w*l.h, LOGISTIC, l.delta_gpu + index);
host/src/region_layer.c:    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
host/src/normalization_layer.c:    #ifdef GPU
host/src/normalization_layer.c:    layer.forward_gpu = forward_normalization_layer_gpu;
host/src/normalization_layer.c:    layer.backward_gpu = backward_normalization_layer_gpu;
host/src/normalization_layer.c:    layer.output_gpu =  cuda_make_array(layer.output, h * w * c * batch);
host/src/normalization_layer.c:    layer.delta_gpu =   cuda_make_array(layer.delta, h * w * c * batch);
host/src/normalization_layer.c:    layer.squared_gpu = cuda_make_array(layer.squared, h * w * c * batch);
host/src/normalization_layer.c:    layer.norms_gpu =   cuda_make_array(layer.norms, h * w * c * batch);
host/src/normalization_layer.c:#ifdef GPU
host/src/normalization_layer.c:    cuda_free(layer->output_gpu);
host/src/normalization_layer.c:    cuda_free(layer->delta_gpu); 
host/src/normalization_layer.c:    cuda_free(layer->squared_gpu); 
host/src/normalization_layer.c:    cuda_free(layer->norms_gpu);   
host/src/normalization_layer.c:    layer->output_gpu =  cuda_make_array(layer->output, h * w * c * batch);
host/src/normalization_layer.c:    layer->delta_gpu =   cuda_make_array(layer->delta, h * w * c * batch);
host/src/normalization_layer.c:    layer->squared_gpu = cuda_make_array(layer->squared, h * w * c * batch);
host/src/normalization_layer.c:    layer->norms_gpu =   cuda_make_array(layer->norms, h * w * c * batch);
host/src/normalization_layer.c:#ifdef GPU
host/src/normalization_layer.c:void forward_normalization_layer_gpu(const layer layer, network net)
host/src/normalization_layer.c:    scal_gpu(w*h*c*layer.batch, 0, layer.squared_gpu, 1);
host/src/normalization_layer.c:        float *squared = layer.squared_gpu + w*h*c*b;
host/src/normalization_layer.c:        float *norms   = layer.norms_gpu + w*h*c*b;
host/src/normalization_layer.c:        float *input   = net.input_gpu + w*h*c*b;
host/src/normalization_layer.c:        pow_gpu(w*h*c, 2, input, 1, squared, 1);
host/src/normalization_layer.c:        const_gpu(w*h, layer.kappa, norms, 1);
host/src/normalization_layer.c:            axpy_gpu(w*h, layer.alpha, squared + w*h*k, 1, norms, 1);
host/src/normalization_layer.c:            copy_gpu(w*h, norms + w*h*(k-1), 1, norms + w*h*k, 1);
host/src/normalization_layer.c:            if(prev >= 0)      axpy_gpu(w*h, -layer.alpha, squared + w*h*prev, 1, norms + w*h*k, 1);
host/src/normalization_layer.c:            if(next < layer.c) axpy_gpu(w*h,  layer.alpha, squared + w*h*next, 1, norms + w*h*k, 1);
host/src/normalization_layer.c:    pow_gpu(w*h*c*layer.batch, -layer.beta, layer.norms_gpu, 1, layer.output_gpu, 1);
host/src/normalization_layer.c:    mul_gpu(w*h*c*layer.batch, net.input_gpu, 1, layer.output_gpu, 1);
host/src/normalization_layer.c:void backward_normalization_layer_gpu(const layer layer, network net)
host/src/normalization_layer.c:    pow_gpu(w*h*c*layer.batch, -layer.beta, layer.norms_gpu, 1, net.delta_gpu, 1);
host/src/normalization_layer.c:    mul_gpu(w*h*c*layer.batch, layer.delta_gpu, 1, net.delta_gpu, 1);
host/src/gemm.c:#include "cuda.h"
host/src/gemm.c:#ifdef GPU
host/src/gemm.c:void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
host/src/gemm.c:              float *A_gpu, int lda,
host/src/gemm.c:              float *B_gpu, int ldb,
host/src/gemm.c:              float *C_gpu, int ldc)
host/src/gemm.c:    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
host/src/gemm.c:                                     (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
host/src/gemm.c:void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
host/src/gemm.c:        gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
host/src/gemm.c:void time_gpu(int TA, int TB, int m, int k, int n)
host/src/gemm.c:    float *a_cl = cuda_make_array(a, m*k);
host/src/gemm.c:    float *b_cl = cuda_make_array(b, k*n);
host/src/gemm.c:    float *c_cl = cuda_make_array(c, m*n);
host/src/gemm.c:        gemm_gpu(TA,TB,m,n,k,1,a_cl,lda,b_cl,ldb,1,c_cl,n);
host/src/gemm.c:        cudaThreadSynchronize();
host/src/gemm.c:    cuda_free(a_cl);
host/src/gemm.c:    cuda_free(b_cl);
host/src/gemm.c:    cuda_free(c_cl);
host/src/gemm.c:void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
host/src/gemm.c:    float *c_gpu = random_matrix(m,n);
host/src/gemm.c:    memset(c_gpu, 0, m*n*sizeof(float));
host/src/gemm.c:    gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c_gpu,n);
host/src/gemm.c:    //printf("GPU\n");
host/src/gemm.c:    //pm(m, n, c_gpu);
host/src/gemm.c:        //printf("%f %f\n", c[i], c_gpu[i]);
host/src/gemm.c:        sse += pow(c[i]-c_gpu[i], 2);
host/src/gemm.c:    free(c_gpu);
host/src/gemm.c:int test_gpu_blas()
host/src/gemm.c:     test_gpu_accuracy(0,0,10,576,75);
host/src/gemm.c:     test_gpu_accuracy(0,0,17,10,10);
host/src/gemm.c:     test_gpu_accuracy(1,0,17,10,10);
host/src/gemm.c:     test_gpu_accuracy(0,1,17,10,10);
host/src/gemm.c:     test_gpu_accuracy(1,1,17,10,10);
host/src/gemm.c:     test_gpu_accuracy(0,0,1000,10,100);
host/src/gemm.c:     test_gpu_accuracy(1,0,1000,10,100);
host/src/gemm.c:     test_gpu_accuracy(0,1,1000,10,100);
host/src/gemm.c:     test_gpu_accuracy(1,1,1000,10,100);
host/src/gemm.c:     test_gpu_accuracy(0,0,10,10,10);
host/src/gemm.c:     time_gpu(0,0,64,2916,363);
host/src/gemm.c:     time_gpu(0,0,64,2916,363);
host/src/gemm.c:     time_gpu(0,0,64,2916,363);
host/src/gemm.c:     time_gpu(0,0,192,729,1600);
host/src/gemm.c:     time_gpu(0,0,384,196,1728);
host/src/gemm.c:     time_gpu(0,0,256,196,3456);
host/src/gemm.c:     time_gpu(0,0,256,196,2304);
host/src/gemm.c:     time_gpu(0,0,128,4096,12544);
host/src/gemm.c:     time_gpu(0,0,128,4096,4096);
host/src/gemm.c:    time_gpu(0,0,64,75,12544);
host/src/gemm.c:    time_gpu(0,0,64,75,12544);
host/src/gemm.c:    time_gpu(0,0,64,75,12544);
host/src/gemm.c:    time_gpu(0,0,64,576,12544);
host/src/gemm.c:    time_gpu(0,0,256,2304,784);
host/src/gemm.c:    time_gpu(1,1,2304,256,784);
host/src/gemm.c:    time_gpu(0,0,512,4608,196);
host/src/gemm.c:    time_gpu(1,1,4608,512,196);
host/src/activation_layer.c:#include "cuda.h"
host/src/activation_layer.c:#ifdef GPU
host/src/activation_layer.c:    l.forward_gpu = forward_activation_layer_gpu;
host/src/activation_layer.c:    l.backward_gpu = backward_activation_layer_gpu;
host/src/activation_layer.c:    l.output_gpu = cuda_make_array(l.output, inputs*batch);
host/src/activation_layer.c:    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
host/src/activation_layer.c:#ifdef GPU
host/src/activation_layer.c:void forward_activation_layer_gpu(layer l, network net)
host/src/activation_layer.c:    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
host/src/activation_layer.c:    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
host/src/activation_layer.c:void backward_activation_layer_gpu(layer l, network net)
host/src/activation_layer.c:    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
host/src/activation_layer.c:    copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, net.delta_gpu, 1);
host/src/cost_layer.h:#ifdef GPU
host/src/cost_layer.h:void forward_cost_layer_gpu(cost_layer l, network net);
host/src/cost_layer.h:void backward_cost_layer_gpu(const cost_layer l, network net);
host/src/detection_layer.c:#include "cuda.h"
host/src/detection_layer.c:#ifdef GPU
host/src/detection_layer.c:    l.forward_gpu = forward_detection_layer_gpu;
host/src/detection_layer.c:    l.backward_gpu = backward_detection_layer_gpu;
host/src/detection_layer.c:    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
host/src/detection_layer.c:    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
host/src/detection_layer.c:#ifdef GPU
host/src/detection_layer.c:void forward_detection_layer_gpu(const detection_layer l, network net)
host/src/detection_layer.c:        copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
host/src/detection_layer.c:    cuda_pull_array(net.input_gpu, net.input, l.batch*l.inputs);
host/src/detection_layer.c:    cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
host/src/detection_layer.c:    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
host/src/detection_layer.c:void backward_detection_layer_gpu(detection_layer l, network net)
host/src/detection_layer.c:    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
host/src/detection_layer.c:    //copy_gpu(l.batch*l.inputs, l.delta_gpu, 1, net.delta_gpu, 1);
host/src/local_layer.c:#ifdef GPU
host/src/local_layer.c:    l.forward_gpu = forward_local_layer_gpu;
host/src/local_layer.c:    l.backward_gpu = backward_local_layer_gpu;
host/src/local_layer.c:    l.update_gpu = update_local_layer_gpu;
host/src/local_layer.c:    l.weights_gpu = cuda_make_array(l.weights, c*n*size*size*locations);
host/src/local_layer.c:    l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*n*size*size*locations);
host/src/local_layer.c:    l.biases_gpu = cuda_make_array(l.biases, l.outputs);
host/src/local_layer.c:    l.bias_updates_gpu = cuda_make_array(l.bias_updates, l.outputs);
host/src/local_layer.c:    l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
host/src/local_layer.c:    l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
host/src/local_layer.c:#ifdef GPU
host/src/local_layer.c:void forward_local_layer_gpu(const local_layer l, network net)
host/src/local_layer.c:        copy_gpu(l.outputs, l.biases_gpu, 1, l.output_gpu + i*l.outputs, 1);
host/src/local_layer.c:        float *input = net.input_gpu + i*l.w*l.h*l.c;
host/src/local_layer.c:        im2col_gpu(input, l.c, l.h, l.w, 
host/src/local_layer.c:        float *output = l.output_gpu + i*l.outputs;
host/src/local_layer.c:            float *a = l.weights_gpu + j*l.size*l.size*l.c*l.n;
host/src/local_layer.c:            gemm_gpu(0,0,m,n,k,1,a,k,b,locations,1,c,locations);
host/src/local_layer.c:    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
host/src/local_layer.c:void backward_local_layer_gpu(local_layer l, network net)
host/src/local_layer.c:    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
host/src/local_layer.c:        axpy_gpu(l.outputs, 1, l.delta_gpu + i*l.outputs, 1, l.bias_updates_gpu, 1);
host/src/local_layer.c:        float *input = net.input_gpu + i*l.w*l.h*l.c;
host/src/local_layer.c:        im2col_gpu(input, l.c, l.h, l.w, 
host/src/local_layer.c:            float *a = l.delta_gpu + i*l.outputs + j;
host/src/local_layer.c:            float *c = l.weight_updates_gpu + j*l.size*l.size*l.c*l.n;
host/src/local_layer.c:            gemm_gpu(0,1,m,n,k,1,a,locations,b,locations,1,c,n);
host/src/local_layer.c:        if(net.delta_gpu){
host/src/local_layer.c:                float *a = l.weights_gpu + j*l.size*l.size*l.c*l.n;
host/src/local_layer.c:                float *b = l.delta_gpu + i*l.outputs + j;
host/src/local_layer.c:                gemm_gpu(1,0,m,n,k,1,a,m,b,locations,0,c,locations);
host/src/local_layer.c:            col2im_gpu(net.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, net.delta_gpu+i*l.c*l.h*l.w);
host/src/local_layer.c:void update_local_layer_gpu(local_layer l, update_args a)
host/src/local_layer.c:    axpy_gpu(l.outputs, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
host/src/local_layer.c:    scal_gpu(l.outputs, momentum, l.bias_updates_gpu, 1);
host/src/local_layer.c:    axpy_gpu(size, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
host/src/local_layer.c:    axpy_gpu(size, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
host/src/local_layer.c:    scal_gpu(size, momentum, l.weight_updates_gpu, 1);
host/src/local_layer.c:    cuda_pull_array(l.weights_gpu, l.weights, size);
host/src/local_layer.c:    cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
host/src/local_layer.c:    cuda_push_array(l.weights_gpu, l.weights, size);
host/src/local_layer.c:    cuda_push_array(l.biases_gpu, l.biases, l.outputs);
host/src/col2im_kernels.cu:#include "cuda_runtime.h"
host/src/col2im_kernels.cu:#include "cuda.h"
host/src/col2im_kernels.cu:__global__ void col2im_gpu_kernel(const int n, const float* data_col,
host/src/col2im_kernels.cu:void col2im_gpu(float *data_col,
host/src/col2im_kernels.cu:    col2im_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
host/src/blas.h:void constrain_gpu(int N, float ALPHA, float * X, int INCX);
host/src/blas.h:int test_gpu_blas();
host/src/blas.h:#ifdef GPU
host/src/blas.h:#include "cuda.h"
host/src/blas.h:void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY);
host/src/blas.h:void axpy_gpu_offset(int N, float ALPHA, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY);
host/src/blas.h:void copy_gpu(int N, float * X, int INCX, float * Y, int INCY);
host/src/blas.h:void copy_gpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY);
host/src/blas.h:void add_gpu(int N, float ALPHA, float * X, int INCX);
host/src/blas.h:void supp_gpu(int N, float ALPHA, float * X, int INCX);
host/src/blas.h:void mask_gpu(int N, float * X, float mask_num, float * mask, float val);
host/src/blas.h:void scale_mask_gpu(int N, float * X, float mask_num, float * mask, float scale);
host/src/blas.h:void const_gpu(int N, float ALPHA, float *X, int INCX);
host/src/blas.h:void pow_gpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
host/src/blas.h:void mul_gpu(int N, float *X, int INCX, float *Y, int INCY);
host/src/blas.h:void mean_gpu(float *x, int batch, int filters, int spatial, float *mean);
host/src/blas.h:void variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
host/src/blas.h:void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
host/src/blas.h:void l2normalize_gpu(float *x, float *dx, int batch, int filters, int spatial);
host/src/blas.h:void normalize_delta_gpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta);
host/src/blas.h:void fast_mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
host/src/blas.h:void fast_variance_delta_gpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta);
host/src/blas.h:void fast_variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
host/src/blas.h:void fast_mean_gpu(float *x, int batch, int filters, int spatial, float *mean);
host/src/blas.h:void shortcut_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out);
host/src/blas.h:void scale_bias_gpu(float *output, float *biases, int batch, int n, int size);
host/src/blas.h:void backward_scale_gpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);
host/src/blas.h:void scale_bias_gpu(float *output, float *biases, int batch, int n, int size);
host/src/blas.h:void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
host/src/blas.h:void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);
host/src/blas.h:void logistic_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error);
host/src/blas.h:void softmax_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error);
host/src/blas.h:void smooth_l1_gpu(int n, float *pred, float *truth, float *delta, float *error);
host/src/blas.h:void l2_gpu(int n, float *pred, float *truth, float *delta, float *error);
host/src/blas.h:void l1_gpu(int n, float *pred, float *truth, float *delta, float *error);
host/src/blas.h:void wgan_gpu(int n, float *pred, float *truth, float *delta, float *error);
host/src/blas.h:void weighted_delta_gpu(float *a, float *b, float *s, float *da, float *db, float *ds, int num, float *dc);
host/src/blas.h:void weighted_sum_gpu(float *a, float *b, float *s, int num, float *c);
host/src/blas.h:void mult_add_into_gpu(int num, float *a, float *b, float *c);
host/src/blas.h:void inter_gpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
host/src/blas.h:void deinter_gpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
host/src/blas.h:void reorg_gpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out);
host/src/blas.h:void softmax_gpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);
host/src/blas.h:void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t);
host/src/blas.h:void adam_gpu(int n, float *x, float *m, float *v, float B1, float B2, float rate, float eps, int t);
host/src/blas.h:void flatten_gpu(float *x, int spatial, int layers, int batch, int forward, float *out);
host/src/blas.h:void upsample_gpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);
host/src/iseg_layer.c:#include "cuda.h"
host/src/iseg_layer.c:#ifdef GPU
host/src/iseg_layer.c:    l.forward_gpu = forward_iseg_layer_gpu;
host/src/iseg_layer.c:    l.backward_gpu = backward_iseg_layer_gpu;
host/src/iseg_layer.c:    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
host/src/iseg_layer.c:    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
host/src/iseg_layer.c:#ifdef GPU
host/src/iseg_layer.c:    cuda_free(l->delta_gpu);
host/src/iseg_layer.c:    cuda_free(l->output_gpu);
host/src/iseg_layer.c:    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
host/src/iseg_layer.c:    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
host/src/iseg_layer.c:#ifndef GPU
host/src/iseg_layer.c:            if(b == 0 && net.gpu_index == 0){
host/src/iseg_layer.c:#ifdef GPU
host/src/iseg_layer.c:void forward_iseg_layer_gpu(const layer l, network net)
host/src/iseg_layer.c:    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
host/src/iseg_layer.c:        activate_array_gpu(l.output_gpu + b*l.outputs, l.classes*l.w*l.h, LOGISTIC);
host/src/iseg_layer.c:        //if(l.extra) activate_array_gpu(l.output_gpu + b*l.outputs + l.classes*l.w*l.h, l.extra*l.w*l.h, LOGISTIC);
host/src/iseg_layer.c:    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
host/src/iseg_layer.c:    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
host/src/iseg_layer.c:void backward_iseg_layer_gpu(const layer l, network net)
host/src/iseg_layer.c:        //if(l.extra) gradient_array_gpu(l.output_gpu + b*l.outputs + l.classes*l.w*l.h, l.extra*l.w*l.h, LOGISTIC, l.delta_gpu + b*l.outputs + l.classes*l.w*l.h);
host/src/iseg_layer.c:    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
host/src/dropout_layer.c:#include "cuda.h"
host/src/dropout_layer.c:#ifdef GPU
host/src/dropout_layer.c:    l.forward_gpu = forward_dropout_layer_gpu;
host/src/dropout_layer.c:    l.backward_gpu = backward_dropout_layer_gpu;
host/src/dropout_layer.c:    l.rand_gpu = cuda_make_array(l.rand, inputs*batch);
host/src/dropout_layer.c:#ifdef GPU
host/src/dropout_layer.c:    cuda_free(l->rand_gpu);
host/src/dropout_layer.c:    l->rand_gpu = cuda_make_array(l->rand, inputs*l->batch);
host/src/data.c:#include "cuda.h"
host/src/layer.c:#include "cuda.h"
host/src/layer.c:#ifdef GPU
host/src/layer.c:        if(l.rand_gpu)             cuda_free(l.rand_gpu);
host/src/layer.c:#ifdef GPU
host/src/layer.c:    if(l.indexes_gpu)           cuda_free((float *)l.indexes_gpu);
host/src/layer.c:    if(l.z_gpu)                   cuda_free(l.z_gpu);
host/src/layer.c:    if(l.r_gpu)                   cuda_free(l.r_gpu);
host/src/layer.c:    if(l.h_gpu)                   cuda_free(l.h_gpu);
host/src/layer.c:    if(l.m_gpu)                   cuda_free(l.m_gpu);
host/src/layer.c:    if(l.v_gpu)                   cuda_free(l.v_gpu);
host/src/layer.c:    if(l.prev_state_gpu)          cuda_free(l.prev_state_gpu);
host/src/layer.c:    if(l.forgot_state_gpu)        cuda_free(l.forgot_state_gpu);
host/src/layer.c:    if(l.forgot_delta_gpu)        cuda_free(l.forgot_delta_gpu);
host/src/layer.c:    if(l.state_gpu)               cuda_free(l.state_gpu);
host/src/layer.c:    if(l.state_delta_gpu)         cuda_free(l.state_delta_gpu);
host/src/layer.c:    if(l.gate_gpu)                cuda_free(l.gate_gpu);
host/src/layer.c:    if(l.gate_delta_gpu)          cuda_free(l.gate_delta_gpu);
host/src/layer.c:    if(l.save_gpu)                cuda_free(l.save_gpu);
host/src/layer.c:    if(l.save_delta_gpu)          cuda_free(l.save_delta_gpu);
host/src/layer.c:    if(l.concat_gpu)              cuda_free(l.concat_gpu);
host/src/layer.c:    if(l.concat_delta_gpu)        cuda_free(l.concat_delta_gpu);
host/src/layer.c:    if(l.binary_input_gpu)        cuda_free(l.binary_input_gpu);
host/src/layer.c:    if(l.binary_weights_gpu)      cuda_free(l.binary_weights_gpu);
host/src/layer.c:    if(l.mean_gpu)                cuda_free(l.mean_gpu);
host/src/layer.c:    if(l.variance_gpu)            cuda_free(l.variance_gpu);
host/src/layer.c:    if(l.rolling_mean_gpu)        cuda_free(l.rolling_mean_gpu);
host/src/layer.c:    if(l.rolling_variance_gpu)    cuda_free(l.rolling_variance_gpu);
host/src/layer.c:    if(l.variance_delta_gpu)      cuda_free(l.variance_delta_gpu);
host/src/layer.c:    if(l.mean_delta_gpu)          cuda_free(l.mean_delta_gpu);
host/src/layer.c:    if(l.x_gpu)                   cuda_free(l.x_gpu);
host/src/layer.c:    if(l.x_norm_gpu)              cuda_free(l.x_norm_gpu);
host/src/layer.c:    if(l.weights_gpu)             cuda_free(l.weights_gpu);
host/src/layer.c:    if(l.weight_updates_gpu)      cuda_free(l.weight_updates_gpu);
host/src/layer.c:    if(l.biases_gpu)              cuda_free(l.biases_gpu);
host/src/layer.c:    if(l.bias_updates_gpu)        cuda_free(l.bias_updates_gpu);
host/src/layer.c:    if(l.scales_gpu)              cuda_free(l.scales_gpu);
host/src/layer.c:    if(l.scale_updates_gpu)       cuda_free(l.scale_updates_gpu);
host/src/layer.c:    if(l.output_gpu)              cuda_free(l.output_gpu);
host/src/layer.c:    if(l.delta_gpu)               cuda_free(l.delta_gpu);
host/src/layer.c:    if(l.rand_gpu)                cuda_free(l.rand_gpu);
host/src/layer.c:    if(l.squared_gpu)             cuda_free(l.squared_gpu);
host/src/layer.c:    if(l.norms_gpu)               cuda_free(l.norms_gpu);
host/src/parser.c:    net->gpu_index = gpu_index;
host/src/parser.c:#ifdef GPU
host/src/parser.c:            l.output_gpu = net->layers[count-1].output_gpu;
host/src/parser.c:            l.delta_gpu = net->layers[count-1].delta_gpu;
host/src/parser.c:#ifdef GPU
host/src/parser.c:    net->output_gpu = out.output_gpu;
host/src/parser.c:    net->input_gpu = cuda_make_array(net->input, net->inputs*net->batch);
host/src/parser.c:    net->truth_gpu = cuda_make_array(net->truth, net->truths*net->batch);
host/src/parser.c:#ifdef GPU
host/src/parser.c:        if(gpu_index >= 0){
host/src/parser.c:            net->workspace = cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
host/src/parser.c:#ifdef GPU
host/src/parser.c:    if(gpu_index >= 0){
host/src/parser.c:#ifdef GPU
host/src/parser.c:    if(gpu_index >= 0){
host/src/parser.c:#ifdef GPU
host/src/parser.c:    if(gpu_index >= 0){
host/src/parser.c:#ifdef GPU
host/src/parser.c:    if(gpu_index >= 0){
host/src/parser.c:   #ifdef GPU
host/src/parser.c:        if(gpu_index >= 0){
host/src/parser.c:#ifdef GPU
host/src/parser.c:    if(net->gpu_index >= 0){
host/src/parser.c:        cuda_set_device(net->gpu_index);
host/src/parser.c:#ifdef GPU
host/src/parser.c:    if(net->gpu_index >= 0){
host/src/parser.c:        cuda_set_device(net->gpu_index);
host/src/parser.c:#ifdef GPU
host/src/parser.c:    if(gpu_index >= 0){
host/src/parser.c:#ifdef GPU
host/src/parser.c:    if(gpu_index >= 0){
host/src/parser.c:#ifdef GPU
host/src/parser.c:    if(gpu_index >= 0){
host/src/parser.c:#ifdef GPU
host/src/parser.c:    if(gpu_index >= 0){
host/src/parser.c:   #ifdef GPU
host/src/parser.c:        if(gpu_index >= 0){
host/src/parser.c:#ifdef GPU
host/src/parser.c:    if(net->gpu_index >= 0){
host/src/parser.c:        cuda_set_device(net->gpu_index);
host/src/parser.c:#ifdef GPU
host/src/parser.c:    if(net->gpu_index >= 0){
host/src/parser.c:        cuda_set_device(net->gpu_index);
host/src/activation_kernels.cu:#include "cuda_runtime.h"
host/src/activation_kernels.cu:#include "cuda.h"
host/src/activation_kernels.cu:extern "C" void binary_gradient_array_gpu(float *x, float *dx, int n, int size, BINARY_ACTIVATION a, float *y) 
host/src/activation_kernels.cu:    binary_gradient_array_kernel<<<cuda_gridsize(n/2), BLOCK>>>(x, dx, n/2, size, a, y);
host/src/activation_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/activation_kernels.cu:extern "C" void binary_activate_array_gpu(float *x, int n, int size, BINARY_ACTIVATION a, float *y) 
host/src/activation_kernels.cu:    binary_activate_array_kernel<<<cuda_gridsize(n/2), BLOCK>>>(x, n/2, size, a, y);
host/src/activation_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/activation_kernels.cu:extern "C" void activate_array_gpu(float *x, int n, ACTIVATION a) 
host/src/activation_kernels.cu:    activate_array_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, a);
host/src/activation_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/activation_kernels.cu:extern "C" void gradient_array_gpu(float *x, int n, ACTIVATION a, float *delta) 
host/src/activation_kernels.cu:    gradient_array_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, a, delta);
host/src/activation_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/maxpool_layer.c:#include "cuda.h"
host/src/maxpool_layer.c:    #ifdef GPU
host/src/maxpool_layer.c:    l.forward_gpu = forward_maxpool_layer_gpu;
host/src/maxpool_layer.c:    l.backward_gpu = backward_maxpool_layer_gpu;
host/src/maxpool_layer.c:    l.indexes_gpu = cuda_make_int_array(0, output_size);
host/src/maxpool_layer.c:    l.output_gpu  = cuda_make_array(l.output, output_size);
host/src/maxpool_layer.c:    l.delta_gpu   = cuda_make_array(l.delta, output_size);
host/src/maxpool_layer.c:    #ifdef GPU
host/src/maxpool_layer.c:    cuda_free((float *)l->indexes_gpu);
host/src/maxpool_layer.c:    cuda_free(l->output_gpu);
host/src/maxpool_layer.c:    cuda_free(l->delta_gpu);
host/src/maxpool_layer.c:    l->indexes_gpu = cuda_make_int_array(0, output_size);
host/src/maxpool_layer.c:    l->output_gpu  = cuda_make_array(l->output, output_size);
host/src/maxpool_layer.c:    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
host/src/crop_layer.h:#ifdef GPU
host/src/crop_layer.h:void forward_crop_layer_gpu(crop_layer l, network net);
host/src/yolo_layer.h:#ifdef GPU
host/src/yolo_layer.h:void forward_yolo_layer_gpu(const layer l, network net);
host/src/yolo_layer.h:void backward_yolo_layer_gpu(layer l, network net);
host/src/dropout_layer_kernels.cu:#include "cuda_runtime.h"
host/src/dropout_layer_kernels.cu:#include "cuda.h"
host/src/dropout_layer_kernels.cu:void forward_dropout_layer_gpu(dropout_layer layer, network net)
host/src/dropout_layer_kernels.cu:    cuda_random(layer.rand_gpu, size);
host/src/dropout_layer_kernels.cu:    cuda_push_array(layer.rand_gpu, layer.rand, size);
host/src/dropout_layer_kernels.cu:    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(net.input_gpu, size, layer.rand_gpu, layer.probability, layer.scale);
host/src/dropout_layer_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/dropout_layer_kernels.cu:void backward_dropout_layer_gpu(dropout_layer layer, network net)
host/src/dropout_layer_kernels.cu:    if(!net.delta_gpu) return;
host/src/dropout_layer_kernels.cu:    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(net.delta_gpu, size, layer.rand_gpu, layer.probability, layer.scale);
host/src/dropout_layer_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/network.c:#ifdef GPU
host/src/network.c:        if(l.state_gpu){
host/src/network.c:            fill_gpu(l.outputs, 0, l.state_gpu + l.outputs*b, 1);
host/src/network.c:        if(l.h_gpu){
host/src/network.c:            fill_gpu(l.outputs, 0, l.h_gpu + l.outputs*b, 1);
host/src/network.c:#ifdef GPU
host/src/network.c:    if(netp->gpu_index >= 0){
host/src/network.c:        forward_network_gpu(netp);
host/src/network.c:#ifdef GPU
host/src/network.c:    if(netp->gpu_index >= 0){
host/src/network.c:        update_network_gpu(netp);
host/src/network.c:#ifdef GPU
host/src/network.c:    if(netp->gpu_index >= 0){
host/src/network.c:        backward_network_gpu(netp);
host/src/network.c:#ifdef GPU
host/src/network.c:    cuda_set_device(net->gpu_index);
host/src/network.c:    cuda_free(net->workspace);
host/src/network.c:#ifdef GPU
host/src/network.c:    if(gpu_index >= 0){
host/src/network.c:        cuda_free(net->input_gpu);
host/src/network.c:        cuda_free(net->truth_gpu);
host/src/network.c:        net->input_gpu = cuda_make_array(net->input, net->inputs*net->batch);
host/src/network.c:        net->truth_gpu = cuda_make_array(net->truth, net->truths*net->batch);
host/src/network.c:            net->workspace = cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
host/src/network.c:#ifdef GPU
host/src/network.c:    //cuda_pull_array(l.output_gpu, l.output, l.outputs);
host/src/network.c:#ifdef GPU
host/src/network.c:    if(net->input_gpu) cuda_free(net->input_gpu);
host/src/network.c:    if(net->truth_gpu) cuda_free(net->truth_gpu);
host/src/network.c:#ifdef GPU
host/src/network.c:void forward_network_gpu(network *netp)
host/src/network.c:    cuda_set_device(net.gpu_index);
host/src/network.c:    cuda_push_array(net.input_gpu, net.input, net.inputs*net.batch);
host/src/network.c:        cuda_push_array(net.truth_gpu, net.truth, net.truths*net.batch);
host/src/network.c:        if(l.delta_gpu){
host/src/network.c:            fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
host/src/network.c:        l.forward_gpu(l, net);
host/src/network.c:        net.input_gpu = l.output_gpu;
host/src/network.c:            net.truth_gpu = l.output_gpu;
host/src/network.c:void backward_network_gpu(network *netp)
host/src/network.c:    cuda_set_device(net.gpu_index);
host/src/network.c:            net.input_gpu = prev.output_gpu;
host/src/network.c:            net.delta_gpu = prev.delta_gpu;
host/src/network.c:        l.backward_gpu(l, net);
host/src/network.c:void update_network_gpu(network *netp)
host/src/network.c:    cuda_set_device(net.gpu_index);
host/src/network.c:        if(l.update_gpu){
host/src/network.c:            l.update_gpu(l, a);
host/src/network.c:void harmless_update_network_gpu(network *netp)
host/src/network.c:    cuda_set_device(net.gpu_index);
host/src/network.c:        if(l.weight_updates_gpu) fill_gpu(l.nweights, 0, l.weight_updates_gpu, 1);
host/src/network.c:        if(l.bias_updates_gpu) fill_gpu(l.nbiases, 0, l.bias_updates_gpu, 1);
host/src/network.c:        if(l.scale_updates_gpu) fill_gpu(l.nbiases, 0, l.scale_updates_gpu, 1);
host/src/network.c:    cuda_set_device(args.net->gpu_index);
host/src/network.c:        cuda_pull_array(l.biases_gpu, l.bias_updates, l.n);
host/src/network.c:        cuda_pull_array(l.weights_gpu, l.weight_updates, l.nweights);
host/src/network.c:        if(l.scales) cuda_pull_array(l.scales_gpu, l.scale_updates, l.n);
host/src/network.c:        cuda_pull_array(l.biases_gpu, l.bias_updates, l.outputs);
host/src/network.c:        cuda_pull_array(l.weights_gpu, l.weight_updates, l.outputs*l.inputs);
host/src/network.c:        cuda_push_array(l.biases_gpu, l.biases, l.n);
host/src/network.c:        cuda_push_array(l.weights_gpu, l.weights, l.nweights);
host/src/network.c:        if(l.scales) cuda_push_array(l.scales_gpu, l.scales, l.n);
host/src/network.c:        cuda_push_array(l.biases_gpu, l.biases, l.outputs);
host/src/network.c:        cuda_push_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
host/src/network.c:        cuda_push_array(l.biases_gpu, base.biases, l.n);
host/src/network.c:        cuda_push_array(l.weights_gpu, base.weights, l.nweights);
host/src/network.c:        if (base.scales) cuda_push_array(l.scales_gpu, base.scales, l.n);
host/src/network.c:        cuda_push_array(l.biases_gpu, base.biases, l.outputs);
host/src/network.c:        cuda_push_array(l.weights_gpu, base.weights, l.outputs*l.inputs);
host/src/network.c: cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
host/src/network.c: cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
host/src/network.c: if(l.scale_updates) cuda_pull_array(l.scale_updates_gpu, l.scale_updates, l.n);
host/src/network.c: cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
host/src/network.c: cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
host/src/network.c: cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
host/src/network.c: cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
host/src/network.c: if(l.scale_updates) cuda_push_array(l.scale_updates_gpu, l.scale_updates, l.n);
host/src/network.c: cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
host/src/network.c: cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
host/src/network.c: if(l.update_gpu){
host/src/network.c: l.update_gpu(l, update_batch, rate*l.learning_rate_scale, net.momentum, net.decay);
host/src/network.c: cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.n);
host/src/network.c: cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.nweights);
host/src/network.c: if(base.scale_updates) cuda_push_array(l.scale_updates_gpu, base.scale_updates, l.n);
host/src/network.c: cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.outputs);
host/src/network.c: cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.outputs*l.inputs);
host/src/network.c: cuda_set_device(nets[i].gpu_index);
host/src/network.c: cuda_set_device(nets[i].gpu_index);
host/src/network.c:        cuda_set_device(nets[i]->gpu_index);
host/src/network.c:        cuda_set_device(nets[i]->gpu_index);
host/src/network.c:    //cudaDeviceSynchronize();
host/src/network.c:    //cudaDeviceSynchronize();
host/src/network.c:    cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
host/src/utils.c:int *read_intlist(char *gpu_list, int *ngpus, int d)
host/src/utils.c:    int *gpus = 0;
host/src/utils.c:    if(gpu_list){
host/src/utils.c:        int len = strlen(gpu_list);
host/src/utils.c:        *ngpus = 1;
host/src/utils.c:            if (gpu_list[i] == ',') ++*ngpus;
host/src/utils.c:        gpus = calloc(*ngpus, sizeof(int));
host/src/utils.c:        for(i = 0; i < *ngpus; ++i){
host/src/utils.c:            gpus[i] = atoi(gpu_list);
host/src/utils.c:            gpu_list = strchr(gpu_list, ',')+1;
host/src/utils.c:        gpus = calloc(1, sizeof(float));
host/src/utils.c:        *gpus = d;
host/src/utils.c:        *ngpus = 1;
host/src/utils.c:    return gpus;
host/src/crnn_layer.h:#ifdef GPU
host/src/crnn_layer.h:void forward_crnn_layer_gpu(layer l, network net);
host/src/crnn_layer.h:void backward_crnn_layer_gpu(layer l, network net);
host/src/crnn_layer.h:void update_crnn_layer_gpu(layer l, update_args a);
host/src/col2im.h:#ifdef GPU
host/src/col2im.h:void col2im_gpu(float *data_col,
host/src/maxpool_layer_kernels.cu:#include "cuda_runtime.h"
host/src/maxpool_layer_kernels.cu:#include "cuda.h"
host/src/maxpool_layer_kernels.cu:extern "C" void forward_maxpool_layer_gpu(maxpool_layer layer, network net)
host/src/maxpool_layer_kernels.cu:    forward_maxpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.h, layer.w, layer.c, layer.stride, layer.size, layer.pad, net.input_gpu, layer.output_gpu, layer.indexes_gpu);
host/src/maxpool_layer_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/maxpool_layer_kernels.cu:extern "C" void backward_maxpool_layer_gpu(maxpool_layer layer, network net)
host/src/maxpool_layer_kernels.cu:    backward_maxpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.h, layer.w, layer.c, layer.stride, layer.size, layer.pad, layer.delta_gpu, net.delta_gpu, layer.indexes_gpu);
host/src/maxpool_layer_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/rnn_layer.c:#include "cuda.h"
host/src/rnn_layer.c:#ifdef GPU
host/src/rnn_layer.c:    l->output_gpu += num;
host/src/rnn_layer.c:    l->delta_gpu += num;
host/src/rnn_layer.c:    l->x_gpu += num;
host/src/rnn_layer.c:    l->x_norm_gpu += num;
host/src/rnn_layer.c:#ifdef GPU
host/src/rnn_layer.c:    l.forward_gpu = forward_rnn_layer_gpu;
host/src/rnn_layer.c:    l.backward_gpu = backward_rnn_layer_gpu;
host/src/rnn_layer.c:    l.update_gpu = update_rnn_layer_gpu;
host/src/rnn_layer.c:    l.state_gpu = cuda_make_array(0, batch*outputs);
host/src/rnn_layer.c:    l.prev_state_gpu = cuda_make_array(0, batch*outputs);
host/src/rnn_layer.c:    l.output_gpu = l.output_layer->output_gpu;
host/src/rnn_layer.c:    l.delta_gpu = l.output_layer->delta_gpu;
host/src/rnn_layer.c:#ifdef GPU
host/src/rnn_layer.c:void update_rnn_layer_gpu(layer l, update_args a)
host/src/rnn_layer.c:    update_connected_layer_gpu(*(l.input_layer),  a);
host/src/rnn_layer.c:    update_connected_layer_gpu(*(l.self_layer),   a);
host/src/rnn_layer.c:    update_connected_layer_gpu(*(l.output_layer), a);
host/src/rnn_layer.c:void forward_rnn_layer_gpu(layer l, network net)
host/src/rnn_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, output_layer.delta_gpu, 1);
host/src/rnn_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, self_layer.delta_gpu, 1);
host/src/rnn_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, input_layer.delta_gpu, 1);
host/src/rnn_layer.c:        fill_gpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1);
host/src/rnn_layer.c:        copy_gpu(l.outputs*l.batch, l.state_gpu, 1, l.prev_state_gpu, 1);
host/src/rnn_layer.c:        s.input_gpu = net.input_gpu;
host/src/rnn_layer.c:        forward_connected_layer_gpu(input_layer, s);
host/src/rnn_layer.c:        s.input_gpu = l.state_gpu;
host/src/rnn_layer.c:        forward_connected_layer_gpu(self_layer, s);
host/src/rnn_layer.c:        fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
host/src/rnn_layer.c:        axpy_gpu(l.outputs * l.batch, 1, input_layer.output_gpu, 1, l.state_gpu, 1);
host/src/rnn_layer.c:        axpy_gpu(l.outputs * l.batch, 1, self_layer.output_gpu, 1, l.state_gpu, 1);
host/src/rnn_layer.c:        s.input_gpu = l.state_gpu;
host/src/rnn_layer.c:        forward_connected_layer_gpu(output_layer, s);
host/src/rnn_layer.c:        net.input_gpu += l.inputs*l.batch;
host/src/rnn_layer.c:void backward_rnn_layer_gpu(layer l, network net)
host/src/rnn_layer.c:    float *last_input = input_layer.output_gpu;
host/src/rnn_layer.c:    float *last_self = self_layer.output_gpu;
host/src/rnn_layer.c:        fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
host/src/rnn_layer.c:        axpy_gpu(l.outputs * l.batch, 1, input_layer.output_gpu, 1, l.state_gpu, 1);
host/src/rnn_layer.c:        axpy_gpu(l.outputs * l.batch, 1, self_layer.output_gpu, 1, l.state_gpu, 1);
host/src/rnn_layer.c:        s.input_gpu = l.state_gpu;
host/src/rnn_layer.c:        s.delta_gpu = self_layer.delta_gpu;
host/src/rnn_layer.c:        backward_connected_layer_gpu(output_layer, s);
host/src/rnn_layer.c:            fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
host/src/rnn_layer.c:            axpy_gpu(l.outputs * l.batch, 1, input_layer.output_gpu - l.outputs*l.batch, 1, l.state_gpu, 1);
host/src/rnn_layer.c:            axpy_gpu(l.outputs * l.batch, 1, self_layer.output_gpu - l.outputs*l.batch, 1, l.state_gpu, 1);
host/src/rnn_layer.c:            copy_gpu(l.outputs*l.batch, l.prev_state_gpu, 1, l.state_gpu, 1);
host/src/rnn_layer.c:        copy_gpu(l.outputs*l.batch, self_layer.delta_gpu, 1, input_layer.delta_gpu, 1);
host/src/rnn_layer.c:        s.input_gpu = l.state_gpu;
host/src/rnn_layer.c:        s.delta_gpu = (i > 0) ? self_layer.delta_gpu - l.outputs*l.batch : 0;
host/src/rnn_layer.c:        if (i == 0) s.delta_gpu = 0;
host/src/rnn_layer.c:        backward_connected_layer_gpu(self_layer, s);
host/src/rnn_layer.c:        s.input_gpu = net.input_gpu + i*l.inputs*l.batch;
host/src/rnn_layer.c:        if(net.delta_gpu) s.delta_gpu = net.delta_gpu + i*l.inputs*l.batch;
host/src/rnn_layer.c:        else s.delta_gpu = 0;
host/src/rnn_layer.c:        backward_connected_layer_gpu(input_layer, s);
host/src/rnn_layer.c:    fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
host/src/rnn_layer.c:    axpy_gpu(l.outputs * l.batch, 1, last_input, 1, l.state_gpu, 1);
host/src/rnn_layer.c:    axpy_gpu(l.outputs * l.batch, 1, last_self, 1, l.state_gpu, 1);
host/src/rnn_layer.h:#ifdef GPU
host/src/rnn_layer.h:void forward_rnn_layer_gpu(layer l, network net);
host/src/rnn_layer.h:void backward_rnn_layer_gpu(layer l, network net);
host/src/rnn_layer.h:void update_rnn_layer_gpu(layer l, update_args a);
host/src/logistic_layer.h:#ifdef GPU
host/src/logistic_layer.h:void forward_logistic_layer_gpu(const layer l, network net);
host/src/logistic_layer.h:void backward_logistic_layer_gpu(const layer l, network net);
host/src/region_layer.h:#ifdef GPU
host/src/region_layer.h:void forward_region_layer_gpu(const layer l, network net);
host/src/region_layer.h:void backward_region_layer_gpu(layer l, network net);
host/src/softmax_layer.h:#ifdef GPU
host/src/softmax_layer.h:void forward_softmax_layer_gpu(const softmax_layer l, network net);
host/src/softmax_layer.h:void backward_softmax_layer_gpu(const softmax_layer l, network net);
host/src/lstm_layer.c:#include "cuda.h"
host/src/lstm_layer.c:#ifdef GPU
host/src/lstm_layer.c:    l->output_gpu += num;
host/src/lstm_layer.c:    l->delta_gpu += num;
host/src/lstm_layer.c:    l->x_gpu += num;
host/src/lstm_layer.c:    l->x_norm_gpu += num;
host/src/lstm_layer.c:#ifdef GPU
host/src/lstm_layer.c:    l.forward_gpu = forward_lstm_layer_gpu;
host/src/lstm_layer.c:    l.backward_gpu = backward_lstm_layer_gpu;
host/src/lstm_layer.c:    l.update_gpu = update_lstm_layer_gpu;
host/src/lstm_layer.c:    l.output_gpu = cuda_make_array(0, batch*outputs*steps);
host/src/lstm_layer.c:    l.delta_gpu = cuda_make_array(0, batch*l.outputs*steps);
host/src/lstm_layer.c:    l.prev_state_gpu = cuda_make_array(0, batch*outputs);
host/src/lstm_layer.c:    l.prev_cell_gpu = cuda_make_array(0, batch*outputs);
host/src/lstm_layer.c:    l.cell_gpu = cuda_make_array(0, batch*outputs*steps);
host/src/lstm_layer.c:    l.f_gpu = cuda_make_array(0, batch*outputs);
host/src/lstm_layer.c:    l.i_gpu = cuda_make_array(0, batch*outputs);
host/src/lstm_layer.c:    l.g_gpu = cuda_make_array(0, batch*outputs);
host/src/lstm_layer.c:    l.o_gpu = cuda_make_array(0, batch*outputs);
host/src/lstm_layer.c:    l.c_gpu = cuda_make_array(0, batch*outputs);
host/src/lstm_layer.c:    l.h_gpu = cuda_make_array(0, batch*outputs);
host/src/lstm_layer.c:    l.temp_gpu =  cuda_make_array(0, batch*outputs);
host/src/lstm_layer.c:    l.temp2_gpu = cuda_make_array(0, batch*outputs);
host/src/lstm_layer.c:    l.temp3_gpu = cuda_make_array(0, batch*outputs);
host/src/lstm_layer.c:    l.dc_gpu = cuda_make_array(0, batch*outputs);
host/src/lstm_layer.c:    l.dh_gpu = cuda_make_array(0, batch*outputs);
host/src/lstm_layer.c:#ifdef GPU
host/src/lstm_layer.c:void update_lstm_layer_gpu(layer l, update_args a)
host/src/lstm_layer.c:    update_connected_layer_gpu(*(l.wf), a);
host/src/lstm_layer.c:    update_connected_layer_gpu(*(l.wi), a);
host/src/lstm_layer.c:    update_connected_layer_gpu(*(l.wg), a);
host/src/lstm_layer.c:    update_connected_layer_gpu(*(l.wo), a);
host/src/lstm_layer.c:    update_connected_layer_gpu(*(l.uf), a);
host/src/lstm_layer.c:    update_connected_layer_gpu(*(l.ui), a);
host/src/lstm_layer.c:    update_connected_layer_gpu(*(l.ug), a);
host/src/lstm_layer.c:    update_connected_layer_gpu(*(l.uo), a);
host/src/lstm_layer.c:void forward_lstm_layer_gpu(layer l, network state)
host/src/lstm_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, wf.delta_gpu, 1);
host/src/lstm_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, wi.delta_gpu, 1);
host/src/lstm_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, wg.delta_gpu, 1);
host/src/lstm_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, wo.delta_gpu, 1);
host/src/lstm_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, uf.delta_gpu, 1);
host/src/lstm_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, ui.delta_gpu, 1);
host/src/lstm_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, ug.delta_gpu, 1);
host/src/lstm_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, uo.delta_gpu, 1);
host/src/lstm_layer.c:        fill_gpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1);
host/src/lstm_layer.c:        s.input_gpu = l.h_gpu;
host/src/lstm_layer.c:        forward_connected_layer_gpu(wf, s);							
host/src/lstm_layer.c:        forward_connected_layer_gpu(wi, s);							
host/src/lstm_layer.c:        forward_connected_layer_gpu(wg, s);							
host/src/lstm_layer.c:        forward_connected_layer_gpu(wo, s);							
host/src/lstm_layer.c:        s.input_gpu = state.input_gpu;
host/src/lstm_layer.c:        forward_connected_layer_gpu(uf, s);							
host/src/lstm_layer.c:        forward_connected_layer_gpu(ui, s);							
host/src/lstm_layer.c:        forward_connected_layer_gpu(ug, s);							
host/src/lstm_layer.c:        forward_connected_layer_gpu(uo, s);							
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, wf.output_gpu, 1, l.f_gpu, 1);
host/src/lstm_layer.c:        axpy_gpu(l.outputs*l.batch, 1, uf.output_gpu, 1, l.f_gpu, 1);
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, wi.output_gpu, 1, l.i_gpu, 1);	
host/src/lstm_layer.c:        axpy_gpu(l.outputs*l.batch, 1, ui.output_gpu, 1, l.i_gpu, 1);	
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, wg.output_gpu, 1, l.g_gpu, 1);	
host/src/lstm_layer.c:        axpy_gpu(l.outputs*l.batch, 1, ug.output_gpu, 1, l.g_gpu, 1);	
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, wo.output_gpu, 1, l.o_gpu, 1);	
host/src/lstm_layer.c:        axpy_gpu(l.outputs*l.batch, 1, uo.output_gpu, 1, l.o_gpu, 1);	
host/src/lstm_layer.c:        activate_array_gpu(l.f_gpu, l.outputs*l.batch, LOGISTIC);		
host/src/lstm_layer.c:        activate_array_gpu(l.i_gpu, l.outputs*l.batch, LOGISTIC);		
host/src/lstm_layer.c:        activate_array_gpu(l.g_gpu, l.outputs*l.batch, TANH);			
host/src/lstm_layer.c:        activate_array_gpu(l.o_gpu, l.outputs*l.batch, LOGISTIC);		
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.i_gpu, 1, l.temp_gpu, 1);		
host/src/lstm_layer.c:        mul_gpu(l.outputs*l.batch, l.g_gpu, 1, l.temp_gpu, 1);		
host/src/lstm_layer.c:        mul_gpu(l.outputs*l.batch, l.f_gpu, 1, l.c_gpu, 1);			
host/src/lstm_layer.c:        axpy_gpu(l.outputs*l.batch, 1, l.temp_gpu, 1, l.c_gpu, 1);	
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.h_gpu, 1);			
host/src/lstm_layer.c:        activate_array_gpu(l.h_gpu, l.outputs*l.batch, TANH);		
host/src/lstm_layer.c:        mul_gpu(l.outputs*l.batch, l.o_gpu, 1, l.h_gpu, 1);	
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.cell_gpu, 1);		
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.h_gpu, 1, l.output_gpu, 1);
host/src/lstm_layer.c:        state.input_gpu += l.inputs*l.batch;
host/src/lstm_layer.c:        l.output_gpu    += l.outputs*l.batch;
host/src/lstm_layer.c:        l.cell_gpu      += l.outputs*l.batch;
host/src/lstm_layer.c:void backward_lstm_layer_gpu(layer l, network state)
host/src/lstm_layer.c:    state.input_gpu += l.inputs*l.batch*(l.steps - 1);
host/src/lstm_layer.c:    if (state.delta_gpu) state.delta_gpu += l.inputs*l.batch*(l.steps - 1);
host/src/lstm_layer.c:    l.output_gpu += l.outputs*l.batch*(l.steps - 1);
host/src/lstm_layer.c:    l.cell_gpu += l.outputs*l.batch*(l.steps - 1);
host/src/lstm_layer.c:    l.delta_gpu += l.outputs*l.batch*(l.steps - 1);
host/src/lstm_layer.c:        if (i != 0) copy_gpu(l.outputs*l.batch, l.cell_gpu - l.outputs*l.batch, 1, l.prev_cell_gpu, 1);
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.cell_gpu, 1, l.c_gpu, 1);
host/src/lstm_layer.c:        if (i != 0) copy_gpu(l.outputs*l.batch, l.output_gpu - l.outputs*l.batch, 1, l.prev_state_gpu, 1);
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.h_gpu, 1);
host/src/lstm_layer.c:        l.dh_gpu = (i == 0) ? 0 : l.delta_gpu - l.outputs*l.batch;
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, wf.output_gpu, 1, l.f_gpu, 1);			
host/src/lstm_layer.c:        axpy_gpu(l.outputs*l.batch, 1, uf.output_gpu, 1, l.f_gpu, 1);			
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, wi.output_gpu, 1, l.i_gpu, 1);			
host/src/lstm_layer.c:        axpy_gpu(l.outputs*l.batch, 1, ui.output_gpu, 1, l.i_gpu, 1);			
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, wg.output_gpu, 1, l.g_gpu, 1);			
host/src/lstm_layer.c:        axpy_gpu(l.outputs*l.batch, 1, ug.output_gpu, 1, l.g_gpu, 1);			
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, wo.output_gpu, 1, l.o_gpu, 1);			
host/src/lstm_layer.c:        axpy_gpu(l.outputs*l.batch, 1, uo.output_gpu, 1, l.o_gpu, 1);			
host/src/lstm_layer.c:        activate_array_gpu(l.f_gpu, l.outputs*l.batch, LOGISTIC);			
host/src/lstm_layer.c:        activate_array_gpu(l.i_gpu, l.outputs*l.batch, LOGISTIC);		
host/src/lstm_layer.c:        activate_array_gpu(l.g_gpu, l.outputs*l.batch, TANH);			
host/src/lstm_layer.c:        activate_array_gpu(l.o_gpu, l.outputs*l.batch, LOGISTIC);		
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, l.temp3_gpu, 1);		
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.temp_gpu, 1);			
host/src/lstm_layer.c:        activate_array_gpu(l.temp_gpu, l.outputs*l.batch, TANH);			
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp3_gpu, 1, l.temp2_gpu, 1);		
host/src/lstm_layer.c:        mul_gpu(l.outputs*l.batch, l.o_gpu, 1, l.temp2_gpu, 1);			
host/src/lstm_layer.c:        gradient_array_gpu(l.temp_gpu, l.outputs*l.batch, TANH, l.temp2_gpu);
host/src/lstm_layer.c:        axpy_gpu(l.outputs*l.batch, 1, l.dc_gpu, 1, l.temp2_gpu, 1);		
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.temp_gpu, 1);			
host/src/lstm_layer.c:        activate_array_gpu(l.temp_gpu, l.outputs*l.batch, TANH);			
host/src/lstm_layer.c:        mul_gpu(l.outputs*l.batch, l.temp3_gpu, 1, l.temp_gpu, 1);		
host/src/lstm_layer.c:        gradient_array_gpu(l.o_gpu, l.outputs*l.batch, LOGISTIC, l.temp_gpu);
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wo.delta_gpu, 1);
host/src/lstm_layer.c:        s.input_gpu = l.prev_state_gpu;
host/src/lstm_layer.c:        s.delta_gpu = l.dh_gpu;															
host/src/lstm_layer.c:        backward_connected_layer_gpu(wo, s);	
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, uo.delta_gpu, 1);
host/src/lstm_layer.c:        s.input_gpu = state.input_gpu;
host/src/lstm_layer.c:        s.delta_gpu = state.delta_gpu;
host/src/lstm_layer.c:        backward_connected_layer_gpu(uo, s);									
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);			
host/src/lstm_layer.c:        mul_gpu(l.outputs*l.batch, l.i_gpu, 1, l.temp_gpu, 1);				
host/src/lstm_layer.c:        gradient_array_gpu(l.g_gpu, l.outputs*l.batch, TANH, l.temp_gpu);		
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wg.delta_gpu, 1);
host/src/lstm_layer.c:        s.input_gpu = l.prev_state_gpu;
host/src/lstm_layer.c:        s.delta_gpu = l.dh_gpu;														
host/src/lstm_layer.c:        backward_connected_layer_gpu(wg, s);	
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, ug.delta_gpu, 1);
host/src/lstm_layer.c:        s.input_gpu = state.input_gpu;
host/src/lstm_layer.c:        s.delta_gpu = state.delta_gpu;
host/src/lstm_layer.c:        backward_connected_layer_gpu(ug, s);																
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);			
host/src/lstm_layer.c:        mul_gpu(l.outputs*l.batch, l.g_gpu, 1, l.temp_gpu, 1);				
host/src/lstm_layer.c:        gradient_array_gpu(l.i_gpu, l.outputs*l.batch, LOGISTIC, l.temp_gpu);	
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wi.delta_gpu, 1);
host/src/lstm_layer.c:        s.input_gpu = l.prev_state_gpu;
host/src/lstm_layer.c:        s.delta_gpu = l.dh_gpu;
host/src/lstm_layer.c:        backward_connected_layer_gpu(wi, s);						
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, ui.delta_gpu, 1);
host/src/lstm_layer.c:        s.input_gpu = state.input_gpu;
host/src/lstm_layer.c:        s.delta_gpu = state.delta_gpu;
host/src/lstm_layer.c:        backward_connected_layer_gpu(ui, s);									
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);		
host/src/lstm_layer.c:        mul_gpu(l.outputs*l.batch, l.prev_cell_gpu, 1, l.temp_gpu, 1);
host/src/lstm_layer.c:        gradient_array_gpu(l.f_gpu, l.outputs*l.batch, LOGISTIC, l.temp_gpu);
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wf.delta_gpu, 1);
host/src/lstm_layer.c:        s.input_gpu = l.prev_state_gpu;
host/src/lstm_layer.c:        s.delta_gpu = l.dh_gpu;
host/src/lstm_layer.c:        backward_connected_layer_gpu(wf, s);						
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, uf.delta_gpu, 1);
host/src/lstm_layer.c:        s.input_gpu = state.input_gpu;
host/src/lstm_layer.c:        s.delta_gpu = state.delta_gpu;
host/src/lstm_layer.c:        backward_connected_layer_gpu(uf, s);									
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);			
host/src/lstm_layer.c:        mul_gpu(l.outputs*l.batch, l.f_gpu, 1, l.temp_gpu, 1);				
host/src/lstm_layer.c:        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, l.dc_gpu, 1);				
host/src/lstm_layer.c:        state.input_gpu -= l.inputs*l.batch;
host/src/lstm_layer.c:        if (state.delta_gpu) state.delta_gpu -= l.inputs*l.batch;
host/src/lstm_layer.c:        l.output_gpu -= l.outputs*l.batch;
host/src/lstm_layer.c:        l.cell_gpu -= l.outputs*l.batch;
host/src/lstm_layer.c:        l.delta_gpu -= l.outputs*l.batch;
host/src/iseg_layer.h:#ifdef GPU
host/src/iseg_layer.h:void forward_iseg_layer_gpu(const layer l, network net);
host/src/iseg_layer.h:void backward_iseg_layer_gpu(layer l, network net);
host/src/crnn_layer.c:#include "cuda.h"
host/src/crnn_layer.c:#ifdef GPU
host/src/crnn_layer.c:    l->output_gpu += num;
host/src/crnn_layer.c:    l->delta_gpu += num;
host/src/crnn_layer.c:    l->x_gpu += num;
host/src/crnn_layer.c:    l->x_norm_gpu += num;
host/src/crnn_layer.c:#ifdef GPU
host/src/crnn_layer.c:    l.forward_gpu = forward_crnn_layer_gpu;
host/src/crnn_layer.c:    l.backward_gpu = backward_crnn_layer_gpu;
host/src/crnn_layer.c:    l.update_gpu = update_crnn_layer_gpu;
host/src/crnn_layer.c:    l.state_gpu = cuda_make_array(l.state, l.hidden*batch*(steps+1));
host/src/crnn_layer.c:    l.output_gpu = l.output_layer->output_gpu;
host/src/crnn_layer.c:    l.delta_gpu = l.output_layer->delta_gpu;
host/src/crnn_layer.c:#ifdef GPU
host/src/crnn_layer.c:void update_crnn_layer_gpu(layer l, update_args a)
host/src/crnn_layer.c:    update_convolutional_layer_gpu(*(l.input_layer),  a);
host/src/crnn_layer.c:    update_convolutional_layer_gpu(*(l.self_layer),   a);
host/src/crnn_layer.c:    update_convolutional_layer_gpu(*(l.output_layer), a);
host/src/crnn_layer.c:void forward_crnn_layer_gpu(layer l, network net)
host/src/crnn_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, output_layer.delta_gpu, 1);
host/src/crnn_layer.c:    fill_gpu(l.hidden * l.batch * l.steps, 0, self_layer.delta_gpu, 1);
host/src/crnn_layer.c:    fill_gpu(l.hidden * l.batch * l.steps, 0, input_layer.delta_gpu, 1);
host/src/crnn_layer.c:    if(net.train) fill_gpu(l.hidden * l.batch, 0, l.state_gpu, 1);
host/src/crnn_layer.c:        s.input_gpu = net.input_gpu;
host/src/crnn_layer.c:        forward_convolutional_layer_gpu(input_layer, s);
host/src/crnn_layer.c:        s.input_gpu = l.state_gpu;
host/src/crnn_layer.c:        forward_convolutional_layer_gpu(self_layer, s);
host/src/crnn_layer.c:        float *old_state = l.state_gpu;
host/src/crnn_layer.c:        if(net.train) l.state_gpu += l.hidden*l.batch;
host/src/crnn_layer.c:            copy_gpu(l.hidden * l.batch, old_state, 1, l.state_gpu, 1);
host/src/crnn_layer.c:            fill_gpu(l.hidden * l.batch, 0, l.state_gpu, 1);
host/src/crnn_layer.c:        axpy_gpu(l.hidden * l.batch, 1, input_layer.output_gpu, 1, l.state_gpu, 1);
host/src/crnn_layer.c:        axpy_gpu(l.hidden * l.batch, 1, self_layer.output_gpu, 1, l.state_gpu, 1);
host/src/crnn_layer.c:        s.input_gpu = l.state_gpu;
host/src/crnn_layer.c:        forward_convolutional_layer_gpu(output_layer, s);
host/src/crnn_layer.c:        net.input_gpu += l.inputs*l.batch;
host/src/crnn_layer.c:void backward_crnn_layer_gpu(layer l, network net)
host/src/crnn_layer.c:    l.state_gpu += l.hidden*l.batch*l.steps;
host/src/crnn_layer.c:        copy_gpu(l.hidden * l.batch, input_layer.output_gpu, 1, l.state_gpu, 1);
host/src/crnn_layer.c:        axpy_gpu(l.hidden * l.batch, 1, self_layer.output_gpu, 1, l.state_gpu, 1);
host/src/crnn_layer.c:        s.input_gpu = l.state_gpu;
host/src/crnn_layer.c:        s.delta_gpu = self_layer.delta_gpu;
host/src/crnn_layer.c:        backward_convolutional_layer_gpu(output_layer, s);
host/src/crnn_layer.c:        l.state_gpu -= l.hidden*l.batch;
host/src/crnn_layer.c:        s.input_gpu = l.state_gpu;
host/src/crnn_layer.c:        s.delta_gpu = self_layer.delta_gpu - l.hidden*l.batch;
host/src/crnn_layer.c:        if (i == 0) s.delta_gpu = 0;
host/src/crnn_layer.c:        backward_convolutional_layer_gpu(self_layer, s);
host/src/crnn_layer.c:        copy_gpu(l.hidden*l.batch, self_layer.delta_gpu, 1, input_layer.delta_gpu, 1);
host/src/crnn_layer.c:        if (i > 0 && l.shortcut) axpy_gpu(l.hidden*l.batch, 1, self_layer.delta_gpu, 1, self_layer.delta_gpu - l.hidden*l.batch, 1);
host/src/crnn_layer.c:        s.input_gpu = net.input_gpu + i*l.inputs*l.batch;
host/src/crnn_layer.c:        if(net.delta_gpu) s.delta_gpu = net.delta_gpu + i*l.inputs*l.batch;
host/src/crnn_layer.c:        else s.delta_gpu = 0;
host/src/crnn_layer.c:        backward_convolutional_layer_gpu(input_layer, s);
host/src/gru_layer.c:#include "cuda.h"
host/src/gru_layer.c:#ifdef GPU
host/src/gru_layer.c:    l->output_gpu += num;
host/src/gru_layer.c:    l->delta_gpu += num;
host/src/gru_layer.c:    l->x_gpu += num;
host/src/gru_layer.c:    l->x_norm_gpu += num;
host/src/gru_layer.c:#ifdef GPU
host/src/gru_layer.c:    l.forward_gpu = forward_gru_layer_gpu;
host/src/gru_layer.c:    l.backward_gpu = backward_gru_layer_gpu;
host/src/gru_layer.c:    l.update_gpu = update_gru_layer_gpu;
host/src/gru_layer.c:    l.forgot_state_gpu = cuda_make_array(0, batch*outputs);
host/src/gru_layer.c:    l.forgot_delta_gpu = cuda_make_array(0, batch*outputs);
host/src/gru_layer.c:    l.prev_state_gpu = cuda_make_array(0, batch*outputs);
host/src/gru_layer.c:    l.state_gpu = cuda_make_array(0, batch*outputs);
host/src/gru_layer.c:    l.output_gpu = cuda_make_array(0, batch*outputs*steps);
host/src/gru_layer.c:    l.delta_gpu = cuda_make_array(0, batch*outputs*steps);
host/src/gru_layer.c:    l.r_gpu = cuda_make_array(0, batch*outputs);
host/src/gru_layer.c:    l.z_gpu = cuda_make_array(0, batch*outputs);
host/src/gru_layer.c:    l.h_gpu = cuda_make_array(0, batch*outputs);
host/src/gru_layer.c:#ifdef GPU
host/src/gru_layer.c:void update_gru_layer_gpu(layer l, update_args a)
host/src/gru_layer.c:    update_connected_layer_gpu(*(l.ur), a);
host/src/gru_layer.c:    update_connected_layer_gpu(*(l.uz), a);
host/src/gru_layer.c:    update_connected_layer_gpu(*(l.uh), a);
host/src/gru_layer.c:    update_connected_layer_gpu(*(l.wr), a);
host/src/gru_layer.c:    update_connected_layer_gpu(*(l.wz), a);
host/src/gru_layer.c:    update_connected_layer_gpu(*(l.wh), a);
host/src/gru_layer.c:void forward_gru_layer_gpu(layer l, network net)
host/src/gru_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, uz.delta_gpu, 1);
host/src/gru_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, ur.delta_gpu, 1);
host/src/gru_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, uh.delta_gpu, 1);
host/src/gru_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, wz.delta_gpu, 1);
host/src/gru_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, wr.delta_gpu, 1);
host/src/gru_layer.c:    fill_gpu(l.outputs * l.batch * l.steps, 0, wh.delta_gpu, 1);
host/src/gru_layer.c:        fill_gpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1);
host/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, l.state_gpu, 1, l.prev_state_gpu, 1);
host/src/gru_layer.c:        s.input_gpu = l.state_gpu;
host/src/gru_layer.c:        forward_connected_layer_gpu(wz, s);
host/src/gru_layer.c:        forward_connected_layer_gpu(wr, s);
host/src/gru_layer.c:        s.input_gpu = net.input_gpu;
host/src/gru_layer.c:        forward_connected_layer_gpu(uz, s);
host/src/gru_layer.c:        forward_connected_layer_gpu(ur, s);
host/src/gru_layer.c:        forward_connected_layer_gpu(uh, s);
host/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, uz.output_gpu, 1, l.z_gpu, 1);
host/src/gru_layer.c:        axpy_gpu(l.outputs*l.batch, 1, wz.output_gpu, 1, l.z_gpu, 1);
host/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, ur.output_gpu, 1, l.r_gpu, 1);
host/src/gru_layer.c:        axpy_gpu(l.outputs*l.batch, 1, wr.output_gpu, 1, l.r_gpu, 1);
host/src/gru_layer.c:        activate_array_gpu(l.z_gpu, l.outputs*l.batch, LOGISTIC);
host/src/gru_layer.c:        activate_array_gpu(l.r_gpu, l.outputs*l.batch, LOGISTIC);
host/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, l.state_gpu, 1, l.forgot_state_gpu, 1);
host/src/gru_layer.c:        mul_gpu(l.outputs*l.batch, l.r_gpu, 1, l.forgot_state_gpu, 1);
host/src/gru_layer.c:        s.input_gpu = l.forgot_state_gpu;
host/src/gru_layer.c:        forward_connected_layer_gpu(wh, s);
host/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, uh.output_gpu, 1, l.h_gpu, 1);
host/src/gru_layer.c:        axpy_gpu(l.outputs*l.batch, 1, wh.output_gpu, 1, l.h_gpu, 1);
host/src/gru_layer.c:            activate_array_gpu(l.h_gpu, l.outputs*l.batch, TANH);
host/src/gru_layer.c:            activate_array_gpu(l.h_gpu, l.outputs*l.batch, LOGISTIC);
host/src/gru_layer.c:        weighted_sum_gpu(l.state_gpu, l.h_gpu, l.z_gpu, l.outputs*l.batch, l.output_gpu);
host/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.state_gpu, 1);
host/src/gru_layer.c:        net.input_gpu += l.inputs*l.batch;
host/src/gru_layer.c:        l.output_gpu += l.outputs*l.batch;
host/src/gru_layer.c:void backward_gru_layer_gpu(layer l, network net)
host/src/gru_layer.c:    net.input_gpu += l.inputs*l.batch*(l.steps-1);
host/src/gru_layer.c:    if(net.delta_gpu) net.delta_gpu += l.inputs*l.batch*(l.steps-1);
host/src/gru_layer.c:    l.output_gpu += l.outputs*l.batch*(l.steps-1);
host/src/gru_layer.c:    l.delta_gpu += l.outputs*l.batch*(l.steps-1);
host/src/gru_layer.c:    float *end_state = l.output_gpu;
host/src/gru_layer.c:        if(i != 0) copy_gpu(l.outputs*l.batch, l.output_gpu - l.outputs*l.batch, 1, l.state_gpu, 1);
host/src/gru_layer.c:        else copy_gpu(l.outputs*l.batch, l.prev_state_gpu, 1, l.state_gpu, 1);
host/src/gru_layer.c:        float *prev_delta_gpu = (i == 0) ? 0 : l.delta_gpu - l.outputs*l.batch;
host/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, uz.output_gpu, 1, l.z_gpu, 1);
host/src/gru_layer.c:        axpy_gpu(l.outputs*l.batch, 1, wz.output_gpu, 1, l.z_gpu, 1);
host/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, ur.output_gpu, 1, l.r_gpu, 1);
host/src/gru_layer.c:        axpy_gpu(l.outputs*l.batch, 1, wr.output_gpu, 1, l.r_gpu, 1);
host/src/gru_layer.c:        activate_array_gpu(l.z_gpu, l.outputs*l.batch, LOGISTIC);
host/src/gru_layer.c:        activate_array_gpu(l.r_gpu, l.outputs*l.batch, LOGISTIC);
host/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, uh.output_gpu, 1, l.h_gpu, 1);
host/src/gru_layer.c:        axpy_gpu(l.outputs*l.batch, 1, wh.output_gpu, 1, l.h_gpu, 1);
host/src/gru_layer.c:            activate_array_gpu(l.h_gpu, l.outputs*l.batch, TANH);
host/src/gru_layer.c:            activate_array_gpu(l.h_gpu, l.outputs*l.batch, LOGISTIC);
host/src/gru_layer.c:        weighted_delta_gpu(l.state_gpu, l.h_gpu, l.z_gpu, prev_delta_gpu, uh.delta_gpu, uz.delta_gpu, l.outputs*l.batch, l.delta_gpu);
host/src/gru_layer.c:            gradient_array_gpu(l.h_gpu, l.outputs*l.batch, TANH, uh.delta_gpu);
host/src/gru_layer.c:            gradient_array_gpu(l.h_gpu, l.outputs*l.batch, LOGISTIC, uh.delta_gpu);
host/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, uh.delta_gpu, 1, wh.delta_gpu, 1);
host/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, l.state_gpu, 1, l.forgot_state_gpu, 1);
host/src/gru_layer.c:        mul_gpu(l.outputs*l.batch, l.r_gpu, 1, l.forgot_state_gpu, 1);
host/src/gru_layer.c:        fill_gpu(l.outputs*l.batch, 0, l.forgot_delta_gpu, 1);
host/src/gru_layer.c:        s.input_gpu = l.forgot_state_gpu;
host/src/gru_layer.c:        s.delta_gpu = l.forgot_delta_gpu;
host/src/gru_layer.c:        backward_connected_layer_gpu(wh, s);
host/src/gru_layer.c:        if(prev_delta_gpu) mult_add_into_gpu(l.outputs*l.batch, l.forgot_delta_gpu, l.r_gpu, prev_delta_gpu);
host/src/gru_layer.c:        mult_add_into_gpu(l.outputs*l.batch, l.forgot_delta_gpu, l.state_gpu, ur.delta_gpu);
host/src/gru_layer.c:        gradient_array_gpu(l.r_gpu, l.outputs*l.batch, LOGISTIC, ur.delta_gpu);
host/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, ur.delta_gpu, 1, wr.delta_gpu, 1);
host/src/gru_layer.c:        gradient_array_gpu(l.z_gpu, l.outputs*l.batch, LOGISTIC, uz.delta_gpu);
host/src/gru_layer.c:        copy_gpu(l.outputs*l.batch, uz.delta_gpu, 1, wz.delta_gpu, 1);
host/src/gru_layer.c:        s.input_gpu = l.state_gpu;
host/src/gru_layer.c:        s.delta_gpu = prev_delta_gpu;
host/src/gru_layer.c:        backward_connected_layer_gpu(wr, s);
host/src/gru_layer.c:        backward_connected_layer_gpu(wz, s);
host/src/gru_layer.c:        s.input_gpu = net.input_gpu;
host/src/gru_layer.c:        s.delta_gpu = net.delta_gpu;
host/src/gru_layer.c:        backward_connected_layer_gpu(uh, s);
host/src/gru_layer.c:        backward_connected_layer_gpu(ur, s);
host/src/gru_layer.c:        backward_connected_layer_gpu(uz, s);
host/src/gru_layer.c:        net.input_gpu -= l.inputs*l.batch;
host/src/gru_layer.c:        if(net.delta_gpu) net.delta_gpu -= l.inputs*l.batch;
host/src/gru_layer.c:        l.output_gpu -= l.outputs*l.batch;
host/src/gru_layer.c:        l.delta_gpu -= l.outputs*l.batch;
host/src/gru_layer.c:    copy_gpu(l.outputs*l.batch, end_state, 1, l.state_gpu, 1);
host/src/yolo_layer.c:#include "cuda.h"
host/src/yolo_layer.c:#ifdef GPU
host/src/yolo_layer.c:    l.forward_gpu = forward_yolo_layer_gpu;
host/src/yolo_layer.c:    l.backward_gpu = backward_yolo_layer_gpu;
host/src/yolo_layer.c:    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
host/src/yolo_layer.c:    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
host/src/yolo_layer.c:#ifdef GPU
host/src/yolo_layer.c:    cuda_free(l->delta_gpu);
host/src/yolo_layer.c:    cuda_free(l->output_gpu);
host/src/yolo_layer.c:    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
host/src/yolo_layer.c:    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
host/src/yolo_layer.c:#ifndef GPU
host/src/yolo_layer.c:#ifdef GPU
host/src/yolo_layer.c:void forward_yolo_layer_gpu(const layer l, network net)
host/src/yolo_layer.c:    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
host/src/yolo_layer.c:            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
host/src/yolo_layer.c:            activate_array_gpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC);
host/src/yolo_layer.c:        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
host/src/yolo_layer.c:    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
host/src/yolo_layer.c:    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
host/src/yolo_layer.c:void backward_yolo_layer_gpu(const layer l, network net)
host/src/yolo_layer.c:    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
host/src/blas_kernels.cu:#include "cuda_runtime.h"
host/src/blas_kernels.cu:#include "cuda.h"
host/src/blas_kernels.cu:void scale_bias_gpu(float *output, float *biases, int batch, int n, int size)
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:void backward_scale_gpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:void add_bias_gpu(float *output, float *biases, int batch, int n, int size)
host/src/blas_kernels.cu:    add_bias_kernel<<<cuda_gridsize(num), BLOCK>>>(output, biases, batch, n, size);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size)
host/src/blas_kernels.cu:        backward_bias_conn_kernel<<<cuda_gridsize(n), BLOCK>>>(bias_updates, delta, batch, n);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:void dot_error_gpu(layer l)
host/src/blas_kernels.cu:    dot_kernel<<<cuda_gridsize(l.n*l.n), BLOCK>>>(l.output_gpu, l.dot, l.batch, l.n, l.out_w * l.out_h, l.delta_gpu);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void adam_gpu(int n, float *x, float *m, float *v, float B1, float B2, float rate, float eps, int t)
host/src/blas_kernels.cu:    adam_kernel<<<cuda_gridsize(n), BLOCK>>>(n, x, m, v, B1, B2, rate, eps, t);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t)
host/src/blas_kernels.cu:    scal_gpu(n, B1, m, 1);
host/src/blas_kernels.cu:    scal_gpu(n, B2, v, 1);
host/src/blas_kernels.cu:    axpy_gpu(n, -decay*batch, w, 1, d, 1);
host/src/blas_kernels.cu:    axpy_gpu(n, (1-B1), d, 1, m, 1);
host/src/blas_kernels.cu:    mul_gpu(n, d, 1, d, 1);
host/src/blas_kernels.cu:    axpy_gpu(n, (1-B2), d, 1, v, 1);
host/src/blas_kernels.cu:    adam_gpu(n, w, m, v, B1, B2, rate, eps, t);
host/src/blas_kernels.cu:    fill_gpu(n, 0, d, 1);
host/src/blas_kernels.cu:extern "C" void normalize_delta_gpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
host/src/blas_kernels.cu:    normalize_delta_kernel<<<cuda_gridsize(N), BLOCK>>>(N, x, mean, variance, mean_delta, variance_delta, batch, filters, spatial, delta);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
host/src/blas_kernels.cu:    mean_delta_kernel<<<cuda_gridsize(filters), BLOCK>>>(delta, variance, batch, filters, spatial, mean_delta);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void fast_mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void fast_variance_delta_gpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
host/src/blas_kernels.cu:    normalize_kernel<<<cuda_gridsize(N), BLOCK>>>(N, x, mean, variance, batch, filters, spatial);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void l2normalize_gpu(float *x, float *dx, int batch, int filters, int spatial)
host/src/blas_kernels.cu:    l2norm_kernel<<<cuda_gridsize(N), BLOCK>>>(N, x, dx, batch, filters, spatial);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void fast_mean_gpu(float *x, int batch, int filters, int spatial, float *mean)
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void fast_variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void mean_gpu(float *x, int batch, int filters, int spatial, float *mean)
host/src/blas_kernels.cu:    mean_kernel<<<cuda_gridsize(filters), BLOCK>>>(x, batch, filters, spatial, mean);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
host/src/blas_kernels.cu:    variance_kernel<<<cuda_gridsize(filters), BLOCK>>>(x, mean, batch, filters, spatial, variance);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY)
host/src/blas_kernels.cu:    axpy_gpu_offset(N, ALPHA, X, 0, INCX, Y, 0, INCY);
host/src/blas_kernels.cu:extern "C" void pow_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY)
host/src/blas_kernels.cu:    pow_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX, Y, INCY);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void axpy_gpu_offset(int N, float ALPHA, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
host/src/blas_kernels.cu:    axpy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, OFFX, INCX, Y, OFFY, INCY);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void copy_gpu(int N, float * X, int INCX, float * Y, int INCY)
host/src/blas_kernels.cu:    copy_gpu_offset(N, X, 0, INCX, Y, 0, INCY);
host/src/blas_kernels.cu:extern "C" void mul_gpu(int N, float * X, int INCX, float * Y, int INCY)
host/src/blas_kernels.cu:    mul_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, INCX, Y, INCY);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void copy_gpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
host/src/blas_kernels.cu:    copy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, OFFX, INCX, Y, OFFY, INCY);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void flatten_gpu(float *x, int spatial, int layers, int batch, int forward, float *out)
host/src/blas_kernels.cu:    flatten_kernel<<<cuda_gridsize(size), BLOCK>>>(size, x, spatial, layers, batch, forward, out);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void reorg_gpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
host/src/blas_kernels.cu:    reorg_kernel<<<cuda_gridsize(size), BLOCK>>>(size, x, w, h, c, batch, stride, forward, out);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void mask_gpu(int N, float * X, float mask_num, float * mask, float val)
host/src/blas_kernels.cu:    mask_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, mask_num, mask, val);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void scale_mask_gpu(int N, float * X, float mask_num, float * mask, float scale)
host/src/blas_kernels.cu:    scale_mask_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, mask_num, mask, scale);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void const_gpu(int N, float ALPHA, float * X, int INCX)
host/src/blas_kernels.cu:    const_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void constrain_gpu(int N, float ALPHA, float * X, int INCX)
host/src/blas_kernels.cu:    constrain_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void add_gpu(int N, float ALPHA, float * X, int INCX)
host/src/blas_kernels.cu:    add_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void scal_gpu(int N, float ALPHA, float * X, int INCX)
host/src/blas_kernels.cu:    scal_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void supp_gpu(int N, float ALPHA, float * X, int INCX)
host/src/blas_kernels.cu:    supp_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void fill_gpu(int N, float ALPHA, float * X, int INCX)
host/src/blas_kernels.cu:    fill_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void shortcut_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out)
host/src/blas_kernels.cu:    shortcut_kernel<<<cuda_gridsize(size), BLOCK>>>(size, minw, minh, minc, stride, sample, batch, w1, h1, c1, add, w2, h2, c2, s1, s2, out);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void smooth_l1_gpu(int n, float *pred, float *truth, float *delta, float *error)
host/src/blas_kernels.cu:    smooth_l1_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void softmax_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error)
host/src/blas_kernels.cu:    softmax_x_ent_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void logistic_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error)
host/src/blas_kernels.cu:    logistic_x_ent_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void l2_gpu(int n, float *pred, float *truth, float *delta, float *error)
host/src/blas_kernels.cu:    l2_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void l1_gpu(int n, float *pred, float *truth, float *delta, float *error)
host/src/blas_kernels.cu:    l1_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void wgan_gpu(int n, float *pred, float *truth, float *delta, float *error)
host/src/blas_kernels.cu:    wgan_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void deinter_gpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
host/src/blas_kernels.cu:    deinter_kernel<<<cuda_gridsize((NX+NY)*B), BLOCK>>>(NX, X, NY, Y, B, OUT);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void inter_gpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
host/src/blas_kernels.cu:    inter_kernel<<<cuda_gridsize((NX+NY)*B), BLOCK>>>(NX, X, NY, Y, B, OUT);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void weighted_sum_gpu(float *a, float *b, float *s, int num, float *c)
host/src/blas_kernels.cu:    weighted_sum_kernel<<<cuda_gridsize(num), BLOCK>>>(num, a, b, s, c);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void weighted_delta_gpu(float *a, float *b, float *s, float *da, float *db, float *ds, int num, float *dc)
host/src/blas_kernels.cu:    weighted_delta_kernel<<<cuda_gridsize(num), BLOCK>>>(num, a, b, s, da, db, ds, dc);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void mult_add_into_gpu(int num, float *a, float *b, float *c)
host/src/blas_kernels.cu:    mult_add_into_kernel<<<cuda_gridsize(num), BLOCK>>>(num, a, b, c);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:    int *tree_groups_size = cuda_make_int_array(hier.group_size, hier.groups);
host/src/blas_kernels.cu:    int *tree_groups_offset = cuda_make_int_array(hier.group_offset, hier.groups);
host/src/blas_kernels.cu:       tree_groups_size = cuda_make_int_array(hier.group_size, hier.groups);
host/src/blas_kernels.cu:       tree_groups_offset = cuda_make_int_array(hier.group_offset, hier.groups);
host/src/blas_kernels.cu:    softmax_tree_kernel<<<cuda_gridsize(num), BLOCK>>>(input, spatial, batch, stride, temp, output, hier.groups, tree_groups_size, tree_groups_offset);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:    cuda_free((float *)tree_groups_size);
host/src/blas_kernels.cu:    cuda_free((float *)tree_groups_offset);
host/src/blas_kernels.cu:extern "C" void softmax_gpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
host/src/blas_kernels.cu:    softmax_kernel<<<cuda_gridsize(batch*groups), BLOCK>>>(input, n, batch, batch_offset, groups, group_offset, stride, temp, output);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/blas_kernels.cu:extern "C" void upsample_gpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
host/src/blas_kernels.cu:    upsample_kernel<<<cuda_gridsize(size), BLOCK>>>(size, in, w, h, c, batch, stride, forward, scale, out);
host/src/blas_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/reorg_layer.c:#include "cuda.h"
host/src/reorg_layer.c:#ifdef GPU
host/src/reorg_layer.c:    l.forward_gpu = forward_reorg_layer_gpu;
host/src/reorg_layer.c:    l.backward_gpu = backward_reorg_layer_gpu;
host/src/reorg_layer.c:    l.output_gpu  = cuda_make_array(l.output, output_size);
host/src/reorg_layer.c:    l.delta_gpu   = cuda_make_array(l.delta, output_size);
host/src/reorg_layer.c:#ifdef GPU
host/src/reorg_layer.c:    cuda_free(l->output_gpu);
host/src/reorg_layer.c:    cuda_free(l->delta_gpu);
host/src/reorg_layer.c:    l->output_gpu  = cuda_make_array(l->output, output_size);
host/src/reorg_layer.c:    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
host/src/reorg_layer.c:#ifdef GPU
host/src/reorg_layer.c:void forward_reorg_layer_gpu(layer l, network net)
host/src/reorg_layer.c:            flatten_gpu(net.input_gpu, l.w*l.h, l.c, l.batch, 0, l.output_gpu);
host/src/reorg_layer.c:            flatten_gpu(net.input_gpu, l.w*l.h, l.c, l.batch, 1, l.output_gpu);
host/src/reorg_layer.c:            copy_gpu(l.inputs, net.input_gpu + i*l.inputs, 1, l.output_gpu + i*l.outputs, 1);
host/src/reorg_layer.c:        reorg_gpu(net.input_gpu, l.w, l.h, l.c, l.batch, l.stride, 1, l.output_gpu);
host/src/reorg_layer.c:        reorg_gpu(net.input_gpu, l.w, l.h, l.c, l.batch, l.stride, 0, l.output_gpu);
host/src/reorg_layer.c:void backward_reorg_layer_gpu(layer l, network net)
host/src/reorg_layer.c:            flatten_gpu(l.delta_gpu, l.w*l.h, l.c, l.batch, 1, net.delta_gpu);
host/src/reorg_layer.c:            flatten_gpu(l.delta_gpu, l.w*l.h, l.c, l.batch, 0, net.delta_gpu);
host/src/reorg_layer.c:            copy_gpu(l.inputs, l.delta_gpu + i*l.outputs, 1, net.delta_gpu + i*l.inputs, 1);
host/src/reorg_layer.c:        reorg_gpu(l.delta_gpu, l.w, l.h, l.c, l.batch, l.stride, 0, net.delta_gpu);
host/src/reorg_layer.c:        reorg_gpu(l.delta_gpu, l.w, l.h, l.c, l.batch, l.stride, 1, net.delta_gpu);
host/src/l2norm_layer.c:#include "cuda.h"
host/src/l2norm_layer.c:    #ifdef GPU
host/src/l2norm_layer.c:    l.forward_gpu = forward_l2norm_layer_gpu;
host/src/l2norm_layer.c:    l.backward_gpu = backward_l2norm_layer_gpu;
host/src/l2norm_layer.c:    l.output_gpu = cuda_make_array(l.output, inputs*batch); 
host/src/l2norm_layer.c:    l.scales_gpu = cuda_make_array(l.output, inputs*batch); 
host/src/l2norm_layer.c:    l.delta_gpu = cuda_make_array(l.delta, inputs*batch); 
host/src/l2norm_layer.c:#ifdef GPU
host/src/l2norm_layer.c:void forward_l2norm_layer_gpu(const layer l, network net)
host/src/l2norm_layer.c:    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
host/src/l2norm_layer.c:    l2normalize_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_w*l.out_h);
host/src/l2norm_layer.c:void backward_l2norm_layer_gpu(const layer l, network net)
host/src/l2norm_layer.c:    axpy_gpu(l.batch*l.inputs, 1, l.scales_gpu, 1, l.delta_gpu, 1);
host/src/l2norm_layer.c:    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
host/src/avgpool_layer_kernels.cu:#include "cuda_runtime.h"
host/src/avgpool_layer_kernels.cu:#include "cuda.h"
host/src/avgpool_layer_kernels.cu:extern "C" void forward_avgpool_layer_gpu(avgpool_layer layer, network net)
host/src/avgpool_layer_kernels.cu:    forward_avgpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.w, layer.h, layer.c, net.input_gpu, layer.output_gpu);
host/src/avgpool_layer_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/avgpool_layer_kernels.cu:extern "C" void backward_avgpool_layer_gpu(avgpool_layer layer, network net)
host/src/avgpool_layer_kernels.cu:    backward_avgpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.w, layer.h, layer.c, net.delta_gpu, layer.delta_gpu);
host/src/avgpool_layer_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/activations.h:#include "cuda.h"
host/src/activations.h:#ifdef GPU
host/src/activations.h:void activate_array_gpu(float *x, int n, ACTIVATION a);
host/src/activations.h:void gradient_array_gpu(float *x, int n, ACTIVATION a, float *delta);
host/src/lstm_layer.h:#ifdef GPU
host/src/lstm_layer.h:void forward_lstm_layer_gpu(layer l, network net);
host/src/lstm_layer.h:void backward_lstm_layer_gpu(layer l, network net);
host/src/lstm_layer.h:void update_lstm_layer_gpu(layer l, update_args a); 
host/src/dropout_layer.h:#ifdef GPU
host/src/dropout_layer.h:void forward_dropout_layer_gpu(dropout_layer l, network net);
host/src/dropout_layer.h:void backward_dropout_layer_gpu(dropout_layer l, network net);
host/src/convolutional_kernels.cu:#include "cuda_runtime.h"
host/src/convolutional_kernels.cu:#include "cuda.h"
host/src/convolutional_kernels.cu:void binarize_gpu(float *x, int n, float *binary)
host/src/convolutional_kernels.cu:    binarize_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, binary);
host/src/convolutional_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/convolutional_kernels.cu:void binarize_input_gpu(float *input, int n, int size, float *binary)
host/src/convolutional_kernels.cu:    binarize_input_kernel<<<cuda_gridsize(size), BLOCK>>>(input, n, size, binary);
host/src/convolutional_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/convolutional_kernels.cu:void binarize_weights_gpu(float *weights, int n, int size, float *binary)
host/src/convolutional_kernels.cu:    binarize_weights_kernel<<<cuda_gridsize(n), BLOCK>>>(weights, n, size, binary);
host/src/convolutional_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/convolutional_kernels.cu:void forward_convolutional_layer_gpu(convolutional_layer l, network net)
host/src/convolutional_kernels.cu:    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
host/src/convolutional_kernels.cu:        binarize_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu);
host/src/convolutional_kernels.cu:        binarize_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu);
host/src/convolutional_kernels.cu:        binarize_gpu(net.input_gpu, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
host/src/convolutional_kernels.cu:        net.input_gpu = l.binary_input_gpu;
host/src/convolutional_kernels.cu:                net.input_gpu,
host/src/convolutional_kernels.cu:                l.weights_gpu,
host/src/convolutional_kernels.cu:                l.output_gpu);
host/src/convolutional_kernels.cu:            float *a = l.weights_gpu + j*l.nweights/l.groups;
host/src/convolutional_kernels.cu:            float *c = l.output_gpu + (i*l.groups + j)*n*m;
host/src/convolutional_kernels.cu:            float *im = net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
host/src/convolutional_kernels.cu:                im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
host/src/convolutional_kernels.cu:            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
host/src/convolutional_kernels.cu:        forward_batchnorm_layer_gpu(l, net);
host/src/convolutional_kernels.cu:        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
host/src/convolutional_kernels.cu:    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
host/src/convolutional_kernels.cu:    //if(l.dot > 0) dot_error_gpu(l);
host/src/convolutional_kernels.cu:    smooth_kernel<<<cuda_gridsize(n), BLOCK>>>(l.output_gpu, n, l.w, l.h, l.c, size, rate, l.delta_gpu);
host/src/convolutional_kernels.cu:    check_error(cudaPeekAtLastError());
host/src/convolutional_kernels.cu:void backward_convolutional_layer_gpu(convolutional_layer l, network net)
host/src/convolutional_kernels.cu:    //constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
host/src/convolutional_kernels.cu:    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
host/src/convolutional_kernels.cu:        backward_batchnorm_layer_gpu(l, net);
host/src/convolutional_kernels.cu:        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
host/src/convolutional_kernels.cu:    float *original_input = net.input_gpu;
host/src/convolutional_kernels.cu:    if(l.xnor) net.input_gpu = l.binary_input_gpu;
host/src/convolutional_kernels.cu:            net.input_gpu,
host/src/convolutional_kernels.cu:            l.delta_gpu,
host/src/convolutional_kernels.cu:            l.weight_updates_gpu);
host/src/convolutional_kernels.cu:    if(net.delta_gpu){
host/src/convolutional_kernels.cu:                l.weights_gpu,
host/src/convolutional_kernels.cu:                l.delta_gpu,
host/src/convolutional_kernels.cu:                net.delta_gpu);
host/src/convolutional_kernels.cu:        if(l.xnor) gradient_array_gpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, net.delta_gpu);
host/src/convolutional_kernels.cu:            float *a = l.delta_gpu + (i*l.groups + j)*m*k;
host/src/convolutional_kernels.cu:            float *c = l.weight_updates_gpu + j*l.nweights/l.groups;
host/src/convolutional_kernels.cu:            float *im  = net.input_gpu+(i*l.groups + j)*l.c/l.groups*l.h*l.w;
host/src/convolutional_kernels.cu:            float *imd = net.delta_gpu+(i*l.groups + j)*l.c/l.groups*l.h*l.w;
host/src/convolutional_kernels.cu:            im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
host/src/convolutional_kernels.cu:            gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);
host/src/convolutional_kernels.cu:            if (net.delta_gpu) {
host/src/convolutional_kernels.cu:                a = l.weights_gpu + j*l.nweights/l.groups;
host/src/convolutional_kernels.cu:                b = l.delta_gpu + (i*l.groups + j)*m*k;
host/src/convolutional_kernels.cu:                gemm_gpu(1,0,n,k,m,1,a,n,b,k,0,c,k);
host/src/convolutional_kernels.cu:                    col2im_gpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
host/src/convolutional_kernels.cu:            if(l.xnor) gradient_array_gpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, net.delta_gpu + i*l.c*l.h*l.w);
host/src/convolutional_kernels.cu:    cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
host/src/convolutional_kernels.cu:    cuda_pull_array(l.biases_gpu, l.biases, l.n);
host/src/convolutional_kernels.cu:    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
host/src/convolutional_kernels.cu:    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
host/src/convolutional_kernels.cu:        cuda_pull_array(l.scales_gpu, l.scales, l.n);
host/src/convolutional_kernels.cu:        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
host/src/convolutional_kernels.cu:        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
host/src/convolutional_kernels.cu:    cuda_push_array(l.weights_gpu, l.weights, l.nweights);
host/src/convolutional_kernels.cu:    cuda_push_array(l.biases_gpu, l.biases, l.n);
host/src/convolutional_kernels.cu:    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
host/src/convolutional_kernels.cu:    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
host/src/convolutional_kernels.cu:        cuda_push_array(l.scales_gpu, l.scales, l.n);
host/src/convolutional_kernels.cu:        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
host/src/convolutional_kernels.cu:        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
host/src/convolutional_kernels.cu:void update_convolutional_layer_gpu(layer l, update_args a)
host/src/convolutional_kernels.cu:        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.nweights, batch, a.t);
host/src/convolutional_kernels.cu:        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
host/src/convolutional_kernels.cu:        if(l.scales_gpu){
host/src/convolutional_kernels.cu:            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
host/src/convolutional_kernels.cu:        axpy_gpu(l.nweights, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
host/src/convolutional_kernels.cu:        axpy_gpu(l.nweights, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
host/src/convolutional_kernels.cu:        scal_gpu(l.nweights, momentum, l.weight_updates_gpu, 1);
host/src/convolutional_kernels.cu:        axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
host/src/convolutional_kernels.cu:        scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);
host/src/convolutional_kernels.cu:        if(l.scales_gpu){
host/src/convolutional_kernels.cu:            axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
host/src/convolutional_kernels.cu:            scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
host/src/convolutional_kernels.cu:        constrain_gpu(l.nweights, l.clip, l.weights_gpu, 1);
host/src/cuda.c:int gpu_index = 0;
host/src/cuda.c:#ifdef GPU
host/src/cuda.c:#include "cuda.h"
host/src/cuda.c:void cuda_set_device(int n)
host/src/cuda.c:    gpu_index = n;
host/src/cuda.c:    cudaError_t status = cudaSetDevice(n);
host/src/cuda.c:int cuda_get_device()
host/src/cuda.c:    cudaError_t status = cudaGetDevice(&n);
host/src/cuda.c:void check_error(cudaError_t status)
host/src/cuda.c:    //cudaDeviceSynchronize();
host/src/cuda.c:    cudaError_t status2 = cudaGetLastError();
host/src/cuda.c:    if (status != cudaSuccess)
host/src/cuda.c:        const char *s = cudaGetErrorString(status);
host/src/cuda.c:        printf("CUDA Error: %s\n", s);
host/src/cuda.c:        snprintf(buffer, 256, "CUDA Error: %s", s);
host/src/cuda.c:    if (status2 != cudaSuccess)
host/src/cuda.c:        const char *s = cudaGetErrorString(status);
host/src/cuda.c:        printf("CUDA Error Prev: %s\n", s);
host/src/cuda.c:        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
host/src/cuda.c:dim3 cuda_gridsize(size_t n){
host/src/cuda.c:    int i = cuda_get_device();
host/src/cuda.c:    int i = cuda_get_device();
host/src/cuda.c:float *cuda_make_array(float *x, size_t n)
host/src/cuda.c:    float *x_gpu;
host/src/cuda.c:    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
host/src/cuda.c:        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
host/src/cuda.c:        fill_gpu(n, 0, x_gpu, 1);
host/src/cuda.c:    if(!x_gpu) error("Cuda malloc failed\n");
host/src/cuda.c:    return x_gpu;
host/src/cuda.c:void cuda_random(float *x_gpu, size_t n)
host/src/cuda.c:    int i = cuda_get_device();
host/src/cuda.c:    curandGenerateUniform(gen[i], x_gpu, n);
host/src/cuda.c:    check_error(cudaPeekAtLastError());
host/src/cuda.c:float cuda_compare(float *x_gpu, float *x, size_t n, char *s)
host/src/cuda.c:    cuda_pull_array(x_gpu, tmp, n);
host/src/cuda.c:int *cuda_make_int_array(int *x, size_t n)
host/src/cuda.c:    int *x_gpu;
host/src/cuda.c:    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
host/src/cuda.c:        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
host/src/cuda.c:    if(!x_gpu) error("Cuda malloc failed\n");
host/src/cuda.c:    return x_gpu;
host/src/cuda.c:void cuda_free(float *x_gpu)
host/src/cuda.c:    cudaError_t status = cudaFree(x_gpu);
host/src/cuda.c:void cuda_push_array(float *x_gpu, float *x, size_t n)
host/src/cuda.c:    cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
host/src/cuda.c:void cuda_pull_array(float *x_gpu, float *x, size_t n)
host/src/cuda.c:    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
host/src/cuda.c:float cuda_mag_array(float *x_gpu, size_t n)
host/src/cuda.c:    cuda_pull_array(x_gpu, temp, n);
host/src/cuda.c:void cuda_set_device(int n){}
host/src/route_layer.c:#include "cuda.h"
host/src/route_layer.c:    #ifdef GPU
host/src/route_layer.c:    l.forward_gpu = forward_route_layer_gpu;
host/src/route_layer.c:    l.backward_gpu = backward_route_layer_gpu;
host/src/route_layer.c:    l.delta_gpu =  cuda_make_array(l.delta, outputs*batch);
host/src/route_layer.c:    l.output_gpu = cuda_make_array(l.output, outputs*batch);
host/src/route_layer.c:#ifdef GPU
host/src/route_layer.c:    cuda_free(l->output_gpu);
host/src/route_layer.c:    cuda_free(l->delta_gpu);
host/src/route_layer.c:    l->output_gpu  = cuda_make_array(l->output, l->outputs*l->batch);
host/src/route_layer.c:    l->delta_gpu   = cuda_make_array(l->delta,  l->outputs*l->batch);
host/src/route_layer.c:#ifdef GPU
host/src/route_layer.c:void forward_route_layer_gpu(const route_layer l, network net)
host/src/route_layer.c:        float *input = net.layers[index].output_gpu;
host/src/route_layer.c:            copy_gpu(input_size, input + j*input_size, 1, l.output_gpu + offset + j*l.outputs, 1);
host/src/route_layer.c:void backward_route_layer_gpu(const route_layer l, network net)
host/src/route_layer.c:        float *delta = net.layers[index].delta_gpu;
host/src/route_layer.c:            axpy_gpu(input_size, 1, l.delta_gpu + offset + j*l.outputs, 1, delta + j*input_size, 1);
host/src/avgpool_layer.c:#include "cuda.h"
host/src/avgpool_layer.c:    #ifdef GPU
host/src/avgpool_layer.c:    l.forward_gpu = forward_avgpool_layer_gpu;
host/src/avgpool_layer.c:    l.backward_gpu = backward_avgpool_layer_gpu;
host/src/avgpool_layer.c:    l.output_gpu  = cuda_make_array(l.output, output_size);
host/src/avgpool_layer.c:    l.delta_gpu   = cuda_make_array(l.delta, output_size);
host/src/gru_layer.h:#ifdef GPU
host/src/gru_layer.h:void forward_gru_layer_gpu(layer l, network state);
host/src/gru_layer.h:void backward_gru_layer_gpu(layer l, network state);
host/src/gru_layer.h:void update_gru_layer_gpu(layer l, update_args a);
host/src/cuda.h:#ifndef CUDA_H
host/src/cuda.h:#define CUDA_H
host/src/cuda.h:#ifdef GPU
host/src/cuda.h:void check_error(cudaError_t status);
host/src/cuda.h:int *cuda_make_int_array(int *x, size_t n);
host/src/cuda.h:void cuda_random(float *x_gpu, size_t n);
host/src/cuda.h:float cuda_compare(float *x_gpu, float *x, size_t n, char *s);
host/src/cuda.h:dim3 cuda_gridsize(size_t n);
host/src/convolutional_layer.c:#ifdef GPU
host/src/convolutional_layer.c:    swap = l->weights_gpu;
host/src/convolutional_layer.c:    l->weights_gpu = l->binary_weights_gpu;
host/src/convolutional_layer.c:    l->binary_weights_gpu = swap;
host/src/convolutional_layer.c:    if(gpu_index >= 0){
host/src/convolutional_layer.c:#ifdef GPU
host/src/convolutional_layer.c:#ifdef GPU
host/src/convolutional_layer.c:    l.forward_gpu = forward_convolutional_layer_gpu;
host/src/convolutional_layer.c:    l.backward_gpu = backward_convolutional_layer_gpu;
host/src/convolutional_layer.c:    l.update_gpu = update_convolutional_layer_gpu;
host/src/convolutional_layer.c:    if(gpu_index >= 0){
host/src/convolutional_layer.c:            l.m_gpu = cuda_make_array(l.m, l.nweights);
host/src/convolutional_layer.c:            l.v_gpu = cuda_make_array(l.v, l.nweights);
host/src/convolutional_layer.c:            l.bias_m_gpu = cuda_make_array(l.bias_m, n);
host/src/convolutional_layer.c:            l.bias_v_gpu = cuda_make_array(l.bias_v, n);
host/src/convolutional_layer.c:            l.scale_m_gpu = cuda_make_array(l.scale_m, n);
host/src/convolutional_layer.c:            l.scale_v_gpu = cuda_make_array(l.scale_v, n);
host/src/convolutional_layer.c:        l.weights_gpu = cuda_make_array(l.weights, l.nweights);
host/src/convolutional_layer.c:        l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);
host/src/convolutional_layer.c:        l.biases_gpu = cuda_make_array(l.biases, n);
host/src/convolutional_layer.c:        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);
host/src/convolutional_layer.c:        l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
host/src/convolutional_layer.c:        l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
host/src/convolutional_layer.c:            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
host/src/convolutional_layer.c:            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
host/src/convolutional_layer.c:            l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
host/src/convolutional_layer.c:            l.mean_gpu = cuda_make_array(l.mean, n);
host/src/convolutional_layer.c:            l.variance_gpu = cuda_make_array(l.variance, n);
host/src/convolutional_layer.c:            l.rolling_mean_gpu = cuda_make_array(l.mean, n);
host/src/convolutional_layer.c:            l.rolling_variance_gpu = cuda_make_array(l.variance, n);
host/src/convolutional_layer.c:            l.mean_delta_gpu = cuda_make_array(l.mean, n);
host/src/convolutional_layer.c:            l.variance_delta_gpu = cuda_make_array(l.variance, n);
host/src/convolutional_layer.c:            l.scales_gpu = cuda_make_array(l.scales, n);
host/src/convolutional_layer.c:            l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);
host/src/convolutional_layer.c:            l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
host/src/convolutional_layer.c:            l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
host/src/convolutional_layer.c:#ifdef GPU
host/src/convolutional_layer.c:    cuda_free(l->delta_gpu);
host/src/convolutional_layer.c:    cuda_free(l->output_gpu);
host/src/convolutional_layer.c:    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
host/src/convolutional_layer.c:    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);
host/src/convolutional_layer.c:        cuda_free(l->x_gpu);
host/src/convolutional_layer.c:        cuda_free(l->x_norm_gpu);
host/src/convolutional_layer.c:        l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
host/src/convolutional_layer.c:        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
host/src/batchnorm_layer.c:#ifdef GPU
host/src/batchnorm_layer.c:    l.forward_gpu = forward_batchnorm_layer_gpu;
host/src/batchnorm_layer.c:    l.backward_gpu = backward_batchnorm_layer_gpu;
host/src/batchnorm_layer.c:    l.output_gpu =  cuda_make_array(l.output, h * w * c * batch);
host/src/batchnorm_layer.c:    l.delta_gpu =   cuda_make_array(l.delta, h * w * c * batch);
host/src/batchnorm_layer.c:    l.biases_gpu = cuda_make_array(l.biases, c);
host/src/batchnorm_layer.c:    l.bias_updates_gpu = cuda_make_array(l.bias_updates, c);
host/src/batchnorm_layer.c:    l.scales_gpu = cuda_make_array(l.scales, c);
host/src/batchnorm_layer.c:    l.scale_updates_gpu = cuda_make_array(l.scale_updates, c);
host/src/batchnorm_layer.c:    l.mean_gpu = cuda_make_array(l.mean, c);
host/src/batchnorm_layer.c:    l.variance_gpu = cuda_make_array(l.variance, c);
host/src/batchnorm_layer.c:    l.rolling_mean_gpu = cuda_make_array(l.mean, c);
host/src/batchnorm_layer.c:    l.rolling_variance_gpu = cuda_make_array(l.variance, c);
host/src/batchnorm_layer.c:    l.mean_delta_gpu = cuda_make_array(l.mean, c);
host/src/batchnorm_layer.c:    l.variance_delta_gpu = cuda_make_array(l.variance, c);
host/src/batchnorm_layer.c:    l.x_gpu = cuda_make_array(l.output, l.batch*l.outputs);
host/src/batchnorm_layer.c:    l.x_norm_gpu = cuda_make_array(l.output, l.batch*l.outputs);
host/src/batchnorm_layer.c:#ifdef GPU
host/src/batchnorm_layer.c:    cuda_pull_array(l.scales_gpu, l.scales, l.c);
host/src/batchnorm_layer.c:    cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.c);
host/src/batchnorm_layer.c:    cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.c);
host/src/batchnorm_layer.c:    cuda_push_array(l.scales_gpu, l.scales, l.c);
host/src/batchnorm_layer.c:    cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.c);
host/src/batchnorm_layer.c:    cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.c);
host/src/batchnorm_layer.c:void forward_batchnorm_layer_gpu(layer l, network net)
host/src/batchnorm_layer.c:    if(l.type == BATCHNORM) copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
host/src/batchnorm_layer.c:    copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
host/src/batchnorm_layer.c:                l.x_gpu,
host/src/batchnorm_layer.c:                l.output_gpu,
host/src/batchnorm_layer.c:                l.scales_gpu,
host/src/batchnorm_layer.c:                l.biases_gpu,
host/src/batchnorm_layer.c:                l.rolling_mean_gpu,
host/src/batchnorm_layer.c:                l.rolling_variance_gpu,
host/src/batchnorm_layer.c:                l.mean_gpu,
host/src/batchnorm_layer.c:                l.variance_gpu);
host/src/batchnorm_layer.c:        fast_mean_gpu(l.output_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.mean_gpu);
host/src/batchnorm_layer.c:        fast_variance_gpu(l.output_gpu, l.mean_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.variance_gpu);
host/src/batchnorm_layer.c:        scal_gpu(l.out_c, .99, l.rolling_mean_gpu, 1);
host/src/batchnorm_layer.c:        axpy_gpu(l.out_c, .01, l.mean_gpu, 1, l.rolling_mean_gpu, 1);
host/src/batchnorm_layer.c:        scal_gpu(l.out_c, .99, l.rolling_variance_gpu, 1);
host/src/batchnorm_layer.c:        axpy_gpu(l.out_c, .01, l.variance_gpu, 1, l.rolling_variance_gpu, 1);
host/src/batchnorm_layer.c:        copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
host/src/batchnorm_layer.c:        normalize_gpu(l.output_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
host/src/batchnorm_layer.c:        copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.x_norm_gpu, 1);
host/src/batchnorm_layer.c:        scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
host/src/batchnorm_layer.c:        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
host/src/batchnorm_layer.c:        normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
host/src/batchnorm_layer.c:        scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
host/src/batchnorm_layer.c:        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
host/src/batchnorm_layer.c:void backward_batchnorm_layer_gpu(layer l, network net)
host/src/batchnorm_layer.c:        l.mean_gpu = l.rolling_mean_gpu;
host/src/batchnorm_layer.c:        l.variance_gpu = l.rolling_variance_gpu;
host/src/batchnorm_layer.c:            l.x_gpu,
host/src/batchnorm_layer.c:            l.delta_gpu,
host/src/batchnorm_layer.c:            l.x_norm_gpu,
host/src/batchnorm_layer.c:            l.scales_gpu,
host/src/batchnorm_layer.c:            l.scale_updates_gpu,
host/src/batchnorm_layer.c:            l.bias_updates_gpu,
host/src/batchnorm_layer.c:            l.mean_gpu,
host/src/batchnorm_layer.c:            l.variance_gpu);
host/src/batchnorm_layer.c:    copy_gpu(l.outputs*l.batch, l.x_norm_gpu, 1, l.delta_gpu, 1);
host/src/batchnorm_layer.c:    backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h);
host/src/batchnorm_layer.c:    backward_scale_gpu(l.x_norm_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates_gpu);
host/src/batchnorm_layer.c:    scale_bias_gpu(l.delta_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
host/src/batchnorm_layer.c:    fast_mean_delta_gpu(l.delta_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta_gpu);
host/src/batchnorm_layer.c:    fast_variance_delta_gpu(l.x_gpu, l.delta_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta_gpu);
host/src/batchnorm_layer.c:    normalize_delta_gpu(l.x_gpu, l.mean_gpu, l.variance_gpu, l.mean_delta_gpu, l.variance_delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.delta_gpu);
host/src/batchnorm_layer.c:    if(l.type == BATCHNORM) copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, net.delta_gpu, 1);
host/src/shortcut_layer.c:#include "cuda.h"
host/src/shortcut_layer.c:    #ifdef GPU
host/src/shortcut_layer.c:    l.forward_gpu = forward_shortcut_layer_gpu;
host/src/shortcut_layer.c:    l.backward_gpu = backward_shortcut_layer_gpu;
host/src/shortcut_layer.c:    l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
host/src/shortcut_layer.c:    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
host/src/shortcut_layer.c:#ifdef GPU
host/src/shortcut_layer.c:    cuda_free(l->output_gpu);
host/src/shortcut_layer.c:    cuda_free(l->delta_gpu);
host/src/shortcut_layer.c:    l->output_gpu  = cuda_make_array(l->output, l->outputs*l->batch);
host/src/shortcut_layer.c:    l->delta_gpu   = cuda_make_array(l->delta,  l->outputs*l->batch);
host/src/shortcut_layer.c:#ifdef GPU
host/src/shortcut_layer.c:void forward_shortcut_layer_gpu(const layer l, network net)
host/src/shortcut_layer.c:    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
host/src/shortcut_layer.c:    shortcut_gpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output_gpu, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output_gpu);
host/src/shortcut_layer.c:    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
host/src/shortcut_layer.c:void backward_shortcut_layer_gpu(const layer l, network net)
host/src/shortcut_layer.c:    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
host/src/shortcut_layer.c:    axpy_gpu(l.outputs*l.batch, l.alpha, l.delta_gpu, 1, net.delta_gpu, 1);
host/src/shortcut_layer.c:    shortcut_gpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta_gpu, l.w, l.h, l.c, 1, l.beta, net.layers[l.index].delta_gpu);
host/src/normalization_layer.h:#ifdef GPU
host/src/normalization_layer.h:void forward_normalization_layer_gpu(const layer layer, network net);
host/src/normalization_layer.h:void backward_normalization_layer_gpu(const layer layer, network net);
host/src/im2col_kernels.cu:#include "cuda_runtime.h"
host/src/im2col_kernels.cu:#include "cuda.h"
host/src/im2col_kernels.cu:__global__ void im2col_gpu_kernel(const int n, const float* data_im,
host/src/im2col_kernels.cu:void im2col_gpu(float *im,
host/src/im2col_kernels.cu:    im2col_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
host/src/convolutional_layer.h:#include "cuda.h"
host/src/convolutional_layer.h:#ifdef GPU
host/src/convolutional_layer.h:void forward_convolutional_layer_gpu(convolutional_layer layer, network net);
host/src/convolutional_layer.h:void backward_convolutional_layer_gpu(convolutional_layer layer, network net);
host/src/convolutional_layer.h:void update_convolutional_layer_gpu(convolutional_layer layer, update_args a);
host/src/convolutional_layer.h:void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
host/src/convolutional_layer.h:void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);
host/src/convolutional_layer.h:void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t);
host/src/connected_layer.h:#ifdef GPU
host/src/connected_layer.h:void forward_connected_layer_gpu(layer l, network net);
host/src/connected_layer.h:void backward_connected_layer_gpu(layer l, network net);
host/src/connected_layer.h:void update_connected_layer_gpu(layer l, update_args a);
host/src/im2col.h:#ifdef GPU
host/src/im2col.h:void im2col_gpu(float *im,
host/src/local_layer.h:#include "cuda.h"
host/src/local_layer.h:#ifdef GPU
host/src/local_layer.h:void forward_local_layer_gpu(local_layer layer, network net);
host/src/local_layer.h:void backward_local_layer_gpu(local_layer layer, network net);
host/src/local_layer.h:void update_local_layer_gpu(local_layer layer, update_args a);
host/src/logistic_layer.c:#include "cuda.h"
host/src/logistic_layer.c:    #ifdef GPU
host/src/logistic_layer.c:    l.forward_gpu = forward_logistic_layer_gpu;
host/src/logistic_layer.c:    l.backward_gpu = backward_logistic_layer_gpu;
host/src/logistic_layer.c:    l.output_gpu = cuda_make_array(l.output, inputs*batch); 
host/src/logistic_layer.c:    l.loss_gpu = cuda_make_array(l.loss, inputs*batch); 
host/src/logistic_layer.c:    l.delta_gpu = cuda_make_array(l.delta, inputs*batch); 
host/src/logistic_layer.c:#ifdef GPU
host/src/logistic_layer.c:void forward_logistic_layer_gpu(const layer l, network net)
host/src/logistic_layer.c:    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
host/src/logistic_layer.c:    activate_array_gpu(l.output_gpu, l.outputs*l.batch, LOGISTIC);
host/src/logistic_layer.c:        logistic_x_ent_gpu(l.batch*l.inputs, l.output_gpu, net.truth_gpu, l.delta_gpu, l.loss_gpu);
host/src/logistic_layer.c:        cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
host/src/logistic_layer.c:void backward_logistic_layer_gpu(const layer l, network net)
host/src/logistic_layer.c:    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
host/src/deconvolutional_layer.h:#include "cuda.h"
host/src/deconvolutional_layer.h:#ifdef GPU
host/src/deconvolutional_layer.h:void forward_deconvolutional_layer_gpu(layer l, network net);
host/src/deconvolutional_layer.h:void backward_deconvolutional_layer_gpu(layer l, network net);
host/src/deconvolutional_layer.h:void update_deconvolutional_layer_gpu(layer l, update_args a);
host/src/detection_layer.h:#ifdef GPU
host/src/detection_layer.h:void forward_detection_layer_gpu(const detection_layer l, network net);
host/src/detection_layer.h:void backward_detection_layer_gpu(detection_layer l, network net);
host/src/deconvolutional_layer.c:#ifdef GPU
host/src/deconvolutional_layer.c:    l.forward_gpu = forward_deconvolutional_layer_gpu;
host/src/deconvolutional_layer.c:    l.backward_gpu = backward_deconvolutional_layer_gpu;
host/src/deconvolutional_layer.c:    l.update_gpu = update_deconvolutional_layer_gpu;
host/src/deconvolutional_layer.c:    if(gpu_index >= 0){
host/src/deconvolutional_layer.c:            l.m_gpu = cuda_make_array(l.m, c*n*size*size);
host/src/deconvolutional_layer.c:            l.v_gpu = cuda_make_array(l.v, c*n*size*size);
host/src/deconvolutional_layer.c:            l.bias_m_gpu = cuda_make_array(l.bias_m, n);
host/src/deconvolutional_layer.c:            l.bias_v_gpu = cuda_make_array(l.bias_v, n);
host/src/deconvolutional_layer.c:            l.scale_m_gpu = cuda_make_array(l.scale_m, n);
host/src/deconvolutional_layer.c:            l.scale_v_gpu = cuda_make_array(l.scale_v, n);
host/src/deconvolutional_layer.c:        l.weights_gpu = cuda_make_array(l.weights, c*n*size*size);
host/src/deconvolutional_layer.c:        l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*n*size*size);
host/src/deconvolutional_layer.c:        l.biases_gpu = cuda_make_array(l.biases, n);
host/src/deconvolutional_layer.c:        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);
host/src/deconvolutional_layer.c:        l.delta_gpu = cuda_make_array(l.delta, l.batch*l.out_h*l.out_w*n);
host/src/deconvolutional_layer.c:        l.output_gpu = cuda_make_array(l.output, l.batch*l.out_h*l.out_w*n);
host/src/deconvolutional_layer.c:            l.mean_gpu = cuda_make_array(0, n);
host/src/deconvolutional_layer.c:            l.variance_gpu = cuda_make_array(0, n);
host/src/deconvolutional_layer.c:            l.rolling_mean_gpu = cuda_make_array(0, n);
host/src/deconvolutional_layer.c:            l.rolling_variance_gpu = cuda_make_array(0, n);
host/src/deconvolutional_layer.c:            l.mean_delta_gpu = cuda_make_array(0, n);
host/src/deconvolutional_layer.c:            l.variance_delta_gpu = cuda_make_array(0, n);
host/src/deconvolutional_layer.c:            l.scales_gpu = cuda_make_array(l.scales, n);
host/src/deconvolutional_layer.c:            l.scale_updates_gpu = cuda_make_array(0, n);
host/src/deconvolutional_layer.c:            l.x_gpu = cuda_make_array(0, l.batch*l.out_h*l.out_w*n);
host/src/deconvolutional_layer.c:            l.x_norm_gpu = cuda_make_array(0, l.batch*l.out_h*l.out_w*n);
host/src/deconvolutional_layer.c:#ifdef GPU
host/src/deconvolutional_layer.c:    cuda_free(l->delta_gpu);
host/src/deconvolutional_layer.c:    cuda_free(l->output_gpu);
host/src/deconvolutional_layer.c:    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
host/src/deconvolutional_layer.c:    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);
host/src/deconvolutional_layer.c:        cuda_free(l->x_gpu);
host/src/deconvolutional_layer.c:        cuda_free(l->x_norm_gpu);
host/src/deconvolutional_layer.c:        l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
host/src/deconvolutional_layer.c:        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
host/src/cost_layer.c:#include "cuda.h"
host/src/cost_layer.c:    #ifdef GPU
host/src/cost_layer.c:    l.forward_gpu = forward_cost_layer_gpu;
host/src/cost_layer.c:    l.backward_gpu = backward_cost_layer_gpu;
host/src/cost_layer.c:    l.delta_gpu = cuda_make_array(l.output, inputs*batch);
host/src/cost_layer.c:    l.output_gpu = cuda_make_array(l.delta, inputs*batch);
host/src/cost_layer.c:#ifdef GPU
host/src/cost_layer.c:    cuda_free(l->delta_gpu);
host/src/cost_layer.c:    cuda_free(l->output_gpu);
host/src/cost_layer.c:    l->delta_gpu = cuda_make_array(l->delta, inputs*l->batch);
host/src/cost_layer.c:    l->output_gpu = cuda_make_array(l->output, inputs*l->batch);
host/src/cost_layer.c:#ifdef GPU
host/src/cost_layer.c:    cuda_pull_array(l.delta_gpu, l.delta, l.batch*l.inputs);
host/src/cost_layer.c:    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
host/src/cost_layer.c:void forward_cost_layer_gpu(cost_layer l, network net)
host/src/cost_layer.c:        scal_gpu(l.batch*l.inputs, (1-l.smooth), net.truth_gpu, 1);
host/src/cost_layer.c:        add_gpu(l.batch*l.inputs, l.smooth * 1./l.inputs, net.truth_gpu, 1);
host/src/cost_layer.c:        smooth_l1_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
host/src/cost_layer.c:        l1_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
host/src/cost_layer.c:        wgan_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
host/src/cost_layer.c:        l2_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
host/src/cost_layer.c:        scale_mask_gpu(l.batch*l.inputs, l.delta_gpu, 0, net.truth_gpu, l.noobject_scale);
host/src/cost_layer.c:        scale_mask_gpu(l.batch*l.inputs, l.output_gpu, 0, net.truth_gpu, l.noobject_scale);
host/src/cost_layer.c:        mask_gpu(l.batch*l.inputs, net.delta_gpu, SECRET_NUM, net.truth_gpu, 0);
host/src/cost_layer.c:        cuda_pull_array(l.delta_gpu, l.delta, l.batch*l.inputs);
host/src/cost_layer.c:        supp_gpu(l.batch*l.inputs, thresh, l.delta_gpu, 1);
host/src/cost_layer.c:        supp_gpu(l.batch*l.inputs, l.thresh*1./l.inputs, l.delta_gpu, 1);
host/src/cost_layer.c:    cuda_pull_array(l.output_gpu, l.output, l.batch*l.inputs);
host/src/cost_layer.c:void backward_cost_layer_gpu(const cost_layer l, network net)
host/src/cost_layer.c:    axpy_gpu(l.batch*l.inputs, l.scale, l.delta_gpu, 1, net.delta_gpu, 1);
host/src/activation_layer.h:#ifdef GPU
host/src/activation_layer.h:void forward_activation_layer_gpu(layer l, network net);
host/src/activation_layer.h:void backward_activation_layer_gpu(layer l, network net);
host/src/reorg_layer.h:#include "cuda.h"
host/src/reorg_layer.h:#ifdef GPU
host/src/reorg_layer.h:void forward_reorg_layer_gpu(layer l, network net);
host/src/reorg_layer.h:void backward_reorg_layer_gpu(layer l, network net);
ta/include/darknet_TA.h:extern int gpu_index;
ta/include/darknet_TA.h:    int gpu_index;

```
