# https://github.com/JungleComputing/rocket

```console
phylogenetics/src/main/cpp/native.cu: * Memory is aligned 256-byte segments for good performance on the GPU.
phylogenetics/src/main/cpp/native.cu:    cudaError_t err = cub::DeviceReduce::Sum(
phylogenetics/src/main/cpp/native.cu:    if (err != cudaSuccess) return err;
phylogenetics/src/main/cpp/native.cu:    if (err != cudaSuccess) return err;
phylogenetics/src/main/cpp/native.cu:    if (err != cudaSuccess) return err;
phylogenetics/src/main/cpp/native.cu:cudaError_t count_kmers(
phylogenetics/src/main/cpp/native.cu:        cudaStream_t stream,
phylogenetics/src/main/cpp/native.cu:    auto exec = thrust::cuda::par.on(stream);
phylogenetics/src/main/cpp/native.cu:    return cudaSuccess;
phylogenetics/src/main/cpp/native.cu:cudaError_t build_composition_vector(
phylogenetics/src/main/cpp/native.cu:    cudaStream_t stream,
phylogenetics/src/main/cpp/native.cu:    cudaError_t err = cudaSuccess;
phylogenetics/src/main/cpp/native.cu:    auto exec = thrust::cuda::par.on(stream);
phylogenetics/src/main/cpp/native.cu:        return cudaErrorUnknown;
phylogenetics/src/main/cpp/native.cu:    if (err != cudaSuccess) return err;
phylogenetics/src/main/cpp/native.cu:    if (err != cudaSuccess) return err;
phylogenetics/src/main/cpp/native.cu:    if (err != cudaSuccess) return err;
phylogenetics/src/main/cpp/native.cu:    if (err != cudaSuccess) return err;
phylogenetics/src/main/cpp/native.cu:    if (err != cudaSuccess) return err;
phylogenetics/src/main/cpp/native.cu:    if (err != cudaSuccess) return err;
phylogenetics/src/main/cpp/native.cu:    err = cudaMemcpyAsync(
phylogenetics/src/main/cpp/native.cu:            cudaMemcpyDeviceToHost,
phylogenetics/src/main/cpp/native.cu:    if (err != cudaSuccess) return err;
phylogenetics/src/main/cpp/native.cu:    err = cudaStreamSynchronize(stream);
phylogenetics/src/main/cpp/native.cu:    if (err != cudaSuccess) return err;
phylogenetics/src/main/cpp/native.cu:        return cudaErrorUnknown;
phylogenetics/src/main/cpp/native.cu:    if (err != cudaSuccess) return err;
phylogenetics/src/main/cpp/native.cu:    err = cudaStreamSynchronize(stream);
phylogenetics/src/main/cpp/native.cu:    if (err != cudaSuccess) return err;
phylogenetics/src/main/cpp/native.cu:    return cudaSuccess;
phylogenetics/src/main/cpp/native.cu:            //printf("GPU found %d %d (%d == %d): %f * %f == %f\n",
phylogenetics/src/main/cpp/native.cu:cudaError_t calculate_cosine_similarity(
phylogenetics/src/main/cpp/native.cu:    cudaStream_t stream,
phylogenetics/src/main/cpp/native.cu:    cudaError_t err = cudaSuccess;
phylogenetics/src/main/cpp/native.cu:            thrust::cuda::par.on(stream),
phylogenetics/src/main/cpp/native.cu:    err = cudaGetLastError();
phylogenetics/src/main/cpp/native.cu:    if (err != cudaSuccess) return err;
phylogenetics/src/main/cpp/native.cu:    if (err != cudaSuccess) return err;
phylogenetics/src/main/cpp/native.cu:    err = cudaStreamSynchronize(stream);
phylogenetics/src/main/cpp/native.cu:    if (err != cudaSuccess) return err;
phylogenetics/src/main/cpp/native.cu:    return cudaSuccess;
phylogenetics/src/main/cpp/native.cu:        cudaError_t err = build_composition_vector<A>( \
phylogenetics/src/main/cpp/native.cu:                (cudaStream_t) stream, \
phylogenetics/src/main/cpp/native.cu:    return cudaErrorUnknown;
phylogenetics/src/main/cpp/native.cu:    cudaError_t err = calculate_cosine_similarity(
phylogenetics/src/main/cpp/native.cu:            (cudaStream_t) stream,
phylogenetics/src/main/cpp/native.cu:    cudaStream_t stream;
phylogenetics/src/main/cpp/native.cu:    cudaEvent_t event_before;
phylogenetics/src/main/cpp/native.cu:    cudaEvent_t event_after;
phylogenetics/src/main/cpp/native.cu:    cudaStreamCreate(&stream);
phylogenetics/src/main/cpp/native.cu:    cudaEventCreate(&event_before);
phylogenetics/src/main/cpp/native.cu:    cudaEventCreate(&event_after);
phylogenetics/src/main/cpp/native.cu:    cudaEventRecord(event_before, stream);
phylogenetics/src/main/cpp/native.cu:    cudaError_t err = calculate_cosine_similarity(
phylogenetics/src/main/cpp/native.cu:    cudaEventRecord(event_after, stream);
phylogenetics/src/main/cpp/native.cu:    cudaStreamSynchronize(stream);
phylogenetics/src/main/cpp/native.cu:    cudaEventElapsedTime(&elapsed, event_before, event_after);
phylogenetics/src/main/cpp/native.cu:    cudaStreamDestroy(stream);
phylogenetics/src/main/cpp/native.cu:    cudaEventDestroy(event_before);
phylogenetics/src/main/cpp/native.cu:    cudaEventDestroy(event_after);
build.gradle:    // jcuda dependencies
build.gradle:    compile (group: 'org.jcuda', name: 'jcuda', version: '10.0.0',){
build.gradle:    compile (group: 'org.jcuda', name: 'jcufft', version: '10.0.0',){
build.gradle:    compile group: 'org.jcuda', name: 'jcuda-natives', classifier: classifier, version: '10.0.0'
build.gradle:    compile group: 'org.jcuda', name: 'jcufft-natives', classifier: classifier, version: '10.0.0'
README.md:Rocket targets heterogeneous distributed platforms. In practice, this means any platform consisting of multiple nodes each equipped with at least one CUDA-enabled GPU.
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilterTest.java:import nl.esciencecenter.rocket.cubaapi.CudaMemFloat;
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilterTest.java:	public void applyGPUTest() {
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilterTest.java:		float[] pixelsGPU = Util.from2DTo1D(HEIGHT, WIDTH, pixels);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilterTest.java:		CudaMemFloat dmem = context.allocFloats(HEIGHT * WIDTH);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilterTest.java:		dmem.copyFromHost(pixelsGPU);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilterTest.java:		filter.getFastNoiseFilter().applyGPU(dmem);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilterTest.java:		dmem.copyToHost(pixelsGPU);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilterTest.java:		//applyGPU CPU and GPU result
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilterTest.java:				pixelsGPU);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilterTest.java:import nl.esciencecenter.rocket.cubaapi.CudaMemByte;
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilterTest.java:import nl.esciencecenter.rocket.cubaapi.CudaMemFloat;
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilterTest.java:	public void applyGPUTest() {
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilterTest.java:		byte[] rgbImageGPU = new byte[HEIGHT * WIDTH * 3];
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilterTest.java:					rgbImageGPU[(y * WIDTH + x) * 3 + i] = rgbImage[y][x][i];
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilterTest.java:		float[] pixelsGPU = new float[HEIGHT * WIDTH];
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilterTest.java:		CudaMemByte dinput = context.allocBytes(HEIGHT * WIDTH * 3);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilterTest.java:		CudaMemFloat doutput = context.allocFloats(HEIGHT * WIDTH);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilterTest.java:		dinput.copyFromHost(rgbImageGPU);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilterTest.java:		filter.getGrayscaleFilter().applyGPU(dinput, doutput);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilterTest.java:		doutput.copyToHost(pixelsGPU);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilterTest.java:		//applyGPU CPU and GPU result
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilterTest.java:				pixelsGPU);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilterTest.java:import nl.esciencecenter.rocket.cubaapi.CudaMemFloat;
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilterTest.java:    public void applyGPUTest() {
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilterTest.java:        float[] outputGPU = new float[HEIGHT * 2 * (WIDTH / 2 + 1)];
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilterTest.java:        float[] pixelsGPU = Util.from2DTo1D(HEIGHT, WIDTH, pixels);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilterTest.java:        CudaMemFloat dinput = context.allocFloats(pixelsGPU.length);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilterTest.java:        CudaMemFloat doutput = context.allocFloats(outputGPU.length);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilterTest.java:        dinput.copyFromHost(pixelsGPU);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilterTest.java:        filter.getSpectralFilter().applyGPU(dinput, doutput);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilterTest.java:        doutput.copyToHost(outputGPU);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilterTest.java:        //applyGPU CPU and GPU result
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilterTest.java:                outputGPU);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilterTest.java:import nl.esciencecenter.rocket.cubaapi.CudaMemFloat;
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilterTest.java:	public void applyGPUTest() {
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilterTest.java:		float[] pixelsGPU = Util.from2DTo1D(HEIGHT, WIDTH, pixels);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilterTest.java:		CudaMemFloat dmem = context.allocFloats(HEIGHT * WIDTH);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilterTest.java:		dmem.copyFromHost(pixelsGPU);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilterTest.java:		filter.getWienerFilter().applyGPU(dmem);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilterTest.java:		dmem.copyToHost(pixelsGPU);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilterTest.java:		//applyGPU CPU and GPU result
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilterTest.java:				pixelsGPU);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/AbstractFilterTest.java:import nl.esciencecenter.rocket.cubaapi.CudaContext;
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/AbstractFilterTest.java:import nl.esciencecenter.rocket.cubaapi.CudaDevice;
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/AbstractFilterTest.java:    protected CudaContext context;
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/AbstractFilterTest.java:        context = CudaDevice.getBestDevice().createContext();
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilterTest.java:import nl.esciencecenter.rocket.cubaapi.CudaMemFloat;
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilterTest.java:	public void applyGPUTest() {
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilterTest.java:		float[] pixelsGPU = Util.from2DTo1D(HEIGHT, WIDTH, pixels);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilterTest.java:		CudaMemFloat dmem = context.allocFloats(HEIGHT * WIDTH);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilterTest.java:		dmem.copyFromHost(pixelsGPU);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilterTest.java:		filter.getZeroMeanTotalFilter().applyGPU(dmem);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilterTest.java:		dmem.copyToHost(pixelsGPU);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilterTest.java:		//applyGPU CPU and GPU result
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilterTest.java:				pixelsGPU);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilterTest.java:import nl.esciencecenter.rocket.cubaapi.CudaMemByte;
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilterTest.java:import nl.esciencecenter.rocket.cubaapi.CudaMemFloat;
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilterTest.java:    public void applyGPUTest() {
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilterTest.java:        byte[] rgbImageGPU = new byte[HEIGHT * WIDTH * 3];
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilterTest.java:                    rgbImageGPU[(y * WIDTH + x) * 3 + i] = rgbImage[y][x][i];
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilterTest.java:        float[] outputGPU = new float[HEIGHT * 2 * (WIDTH / 2 + 1)];
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilterTest.java:        CudaMemByte dinput = context.allocBytes(rgbImageGPU.length);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilterTest.java:        CudaMemFloat doutput = context.allocFloats(outputGPU.length);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilterTest.java:        dinput.copyFromHost(rgbImageGPU);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilterTest.java:        filter.applyGPU(dinput, doutput);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilterTest.java:        doutput.copyToHost(outputGPU);
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilterTest.java:        //applyGPU CPU and GPU result
src/test/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilterTest.java:                outputGPU);
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/ParticleRegistrationFunctionTest.java:import nl.esciencecenter.rocket.cubaapi.CudaContext;
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/ParticleRegistrationFunctionTest.java:import nl.esciencecenter.rocket.cubaapi.CudaDevice;
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/ParticleRegistrationFunctionTest.java:import nl.esciencecenter.rocket.cubaapi.CudaMemDouble;
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/ParticleRegistrationFunctionTest.java:    protected CudaContext context;
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/ParticleRegistrationFunctionTest.java:        context = CudaDevice.getBestDevice().createContext();
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/ParticleRegistrationFunctionTest.java:    public void correlateGPUTest() {
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/ParticleRegistrationFunctionTest.java:        // Alloc GPU memory
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/ParticleRegistrationFunctionTest.java:        CudaMemDouble da = context.allocDoubles(a.length);
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/ParticleRegistrationFunctionTest.java:        CudaMemDouble db = context.allocDoubles(b.length);
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/ParticleRegistrationFunctionTest.java:        CudaMemDouble doutput = context.allocDoubles(3);
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/ParticleRegistrationFunctionTest.java:        // Run on GPU
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/ParticleRegistrationFunctionTest.java:        //registrationFunction.correlateGPU(
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/ParticleRegistrationFunctionTest.java:        // Free GPU memory
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransformTest.java:import nl.esciencecenter.rocket.cubaapi.CudaContext;
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransformTest.java:import nl.esciencecenter.rocket.cubaapi.CudaDevice;
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransformTest.java:import nl.esciencecenter.rocket.cubaapi.CudaMemDouble;
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransformTest.java:    protected CudaContext context;
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransformTest.java:        context = CudaDevice.getBestDevice().createContext();
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransformTest.java:    public void applyGPUTest() {
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransformTest.java:        double[] gradientGPU = new double[3];
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransformTest.java:        double[] crossTermGPU = new double[1];
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransformTest.java:        // Alloc GPU memory
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransformTest.java:        CudaMemDouble da = context.allocDoubles(a.length);
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransformTest.java:        CudaMemDouble db = context.allocDoubles(b.length);
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransformTest.java:        CudaMemDouble dgradient = context.allocDoubles(gradientCPU.length);
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransformTest.java:        CudaMemDouble dcrossTerm = context.allocDoubles(1);
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransformTest.java:        // Run on GPU
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransformTest.java:        gaussTransform.applyGPU(da, db, tx, ty, theta, scale, dgradient, dcrossTerm);
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransformTest.java:        dgradient.copyToHost(gradientGPU);
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransformTest.java:        dcrossTerm.copyToHost(crossTermGPU);
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransformTest.java:        // Free GPU memory
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransformTest.java:        assertEquals(crossTermCPU, crossTermGPU[0], 1e-9);
src/test/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransformTest.java:        assertArrayEquals(gradientCPU, gradientGPU, 1e-9);
src/test/java/nl/esciencecenter/radio_correlator/kernels/CorrelatorTest.java:import nl.esciencecenter.rocket.cubaapi.CudaContext;
src/test/java/nl/esciencecenter/radio_correlator/kernels/CorrelatorTest.java:import nl.esciencecenter.rocket.cubaapi.CudaDevice;
src/test/java/nl/esciencecenter/radio_correlator/kernels/CorrelatorTest.java:import nl.esciencecenter.rocket.cubaapi.CudaMemFloat;
src/test/java/nl/esciencecenter/radio_correlator/kernels/CorrelatorTest.java:    protected CudaContext context;
src/test/java/nl/esciencecenter/radio_correlator/kernels/CorrelatorTest.java:        context = CudaDevice.getBestDevice().createContext();
src/test/java/nl/esciencecenter/radio_correlator/kernels/CorrelatorTest.java:    public void compareGPUTest() {
src/test/java/nl/esciencecenter/radio_correlator/kernels/CorrelatorTest.java:        float[] resultGPU = new float[resultCPU.length];
src/test/java/nl/esciencecenter/radio_correlator/kernels/CorrelatorTest.java:        // Alloc GPU memory
src/test/java/nl/esciencecenter/radio_correlator/kernels/CorrelatorTest.java:        CudaMemFloat dleft = context.allocFloats(left);
src/test/java/nl/esciencecenter/radio_correlator/kernels/CorrelatorTest.java:        CudaMemFloat dright = context.allocFloats(right);
src/test/java/nl/esciencecenter/radio_correlator/kernels/CorrelatorTest.java:        CudaMemFloat dresult = context.allocFloats(resultGPU.length);
src/test/java/nl/esciencecenter/radio_correlator/kernels/CorrelatorTest.java:        correlator.compareGPU(dleft, dright, dresult);
src/test/java/nl/esciencecenter/radio_correlator/kernels/CorrelatorTest.java:        dresult.copyToHost(resultGPU);
src/test/java/nl/esciencecenter/radio_correlator/kernels/CorrelatorTest.java:        // Free GPU memory
src/test/java/nl/esciencecenter/radio_correlator/kernels/CorrelatorTest.java:        assertArrayEquals(resultCPU, resultGPU, 0.001f);
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:import nl.esciencecenter.rocket.cubaapi.CudaContext;
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:import nl.esciencecenter.rocket.cubaapi.CudaDevice;
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:import nl.esciencecenter.rocket.cubaapi.CudaMemByte;
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:import nl.esciencecenter.rocket.cubaapi.CudaMemDouble;
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:import nl.esciencecenter.rocket.cubaapi.CudaMemFloat;
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:import nl.esciencecenter.rocket.cubaapi.CudaMemInt;
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:    CudaContext context;
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:        context = CudaDevice.getBestDevice().createContext();
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:    public int buildVectorGPU(String string, String alphabet, int k, int[] keys, float[] values) throws Throwable {
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:        CudaMemByte d_string = context.allocBytes(string.getBytes("ASCII"));
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:        CudaMemInt d_keys = context.allocInts(maxVectorSize);
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:        CudaMemFloat d_values = context.allocFloats(maxVectorSize);
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:        int outSize = kernels.buildVectorGPU(d_string, d_keys, d_values);
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:        int[] gpuKeys = new int[maxVectorSize];
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:        float[] gpuValues = new float[maxVectorSize];
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:        int sizeGPU = buildVectorGPU(input.toString(), alphabet, k, gpuKeys, gpuValues);
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:        Assert.assertEquals("composition vector sizes do not match from CPU and GPU",
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:                size, sizeGPU);
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:        Assert.assertArrayEquals("composition vector keys do not match from CPU and GPU",
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:                Arrays.copyOf(cpuKeys, size), Arrays.copyOf(gpuKeys, size));
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:        Assert.assertArrayEquals("composition vector values do not match from CPU and GPU",
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:                Arrays.copyOf(cpuValues, size), Arrays.copyOf(gpuValues, size), 1e-6f);
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:    public double compareVectorGPU(int[] leftKeys, float[] leftValues, int[] rightKeys, float[] rightValues) throws Throwable {
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:        CudaMemInt d_leftKeys = context.allocInts(leftKeys);
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:        CudaMemFloat d_leftValues = context.allocFloats(leftValues);
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:        CudaMemInt d_rightKeys = context.allocInts(rightKeys);
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:        CudaMemFloat d_rightValues = context.allocFloats(rightValues);
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:        CudaMemDouble d_output = context.allocDoubles(1);
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:        kernels.compareVectorsGPU(d_leftKeys, d_leftValues, leftSize, d_rightKeys, d_rightValues, rightSize, d_output);
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:        double gpu = compareVectorGPU(leftKeys, leftValues, rightKeys, rightValues);
src/test/java/nl/esciencecenter/phylogenetics_analysis/BindingsTest.java:        Assert.assertEquals(cpu, gpu, 1e-5);
src/main/java/nl/esciencecenter/common_source_identification/CommonSourceIdentificationContext.java:import jcuda.CudaException;
src/main/java/nl/esciencecenter/common_source_identification/CommonSourceIdentificationContext.java:import jcuda.Sizeof;
src/main/java/nl/esciencecenter/common_source_identification/CommonSourceIdentificationContext.java:import nl.esciencecenter.rocket.cubaapi.CudaContext;
src/main/java/nl/esciencecenter/common_source_identification/CommonSourceIdentificationContext.java:import nl.esciencecenter.rocket.cubaapi.CudaMem;
src/main/java/nl/esciencecenter/common_source_identification/CommonSourceIdentificationContext.java:    public CommonSourceIdentificationContext(CudaContext context, Dimension dim, ComparisonStrategy comp) throws CudaException, IOException {
src/main/java/nl/esciencecenter/common_source_identification/CommonSourceIdentificationContext.java:    public long preprocessInputGPU(ImageIdentifier id, CudaMem input, CudaMem output) {
src/main/java/nl/esciencecenter/common_source_identification/CommonSourceIdentificationContext.java:        filter.applyGPU(input, output);
src/main/java/nl/esciencecenter/common_source_identification/CommonSourceIdentificationContext.java:    public long correlateGPU(ImageIdentifier left, CudaMem leftMem, ImageIdentifier right, CudaMem rightMem, CudaMem output) {
src/main/java/nl/esciencecenter/common_source_identification/CommonSourceIdentificationContext.java:        pce.applyGPU(leftMem, rightMem, output);
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:import jcuda.*;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:import jcuda.runtime.cudaStream_t;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:import jcuda.jcufft.*;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:	protected CudaContext _context;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:	protected CudaStream _stream;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:	//handles to CUDA kernels
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:	protected CudaFunction _tocomplex;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:	protected CudaFunction _toreal;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:	protected CudaFunction _computeSquaredMagnitudes;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:	protected CudaFunction _computeVarianceEstimates;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:	protected CudaFunction _computeVarianceZeroMean;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:	protected CudaFunction _sumFloats;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:	protected CudaFunction _scaleWithVariances;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:	protected CudaFunction _normalizeToReal;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:	protected CudaFunction _normalizeComplex;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:	protected CudaMemFloat _d_comp;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:	protected CudaMemFloat _d_sqmag;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:	protected CudaMemFloat _d_varest;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:	protected CudaMemFloat _d_variance;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:	 * @param context - the CudaContext as created by the factory
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:	 * @param stream - the CudaStream as created by the factory
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:	 * @param module - the CudaModule containing the kernels compiled by the factory
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:	public WienerFilter(int h, int w, CudaContext context, CudaStream stream, CudaModule module) {
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:		//setup CUDA functions
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:		//allocate local variables in GPU memory
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:			throw new CudaException(cufftResult.stringFor(res));
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:		res = JCufft.cufftSetStream(_planc2c, new cudaStream_t(_stream.cuStream()));
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:			throw new CudaException(cufftResult.stringFor(res));
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:	 * Applies the Wiener Filter to the input pattern already in GPU memory
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:	public void applyGPU(CudaMemFloat input) {
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:		//construct parameter lists for the CUDA kernels
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:		//applyGPU complex to complex forward Fourier transform
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/WienerFilter.java:	 * Cleans up GPU memory and destroys FFT plan
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilter.java:import jcuda.*;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilter.java:	private CudaContext _context;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilter.java:	protected CudaStream _stream;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilter.java:	//handles to CUDA kernels
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilter.java:	private CudaFunction _grayscale;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilter.java:	 * @param context - the CudaContext as created by the factory
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilter.java:	 * @param stream - the CudaStream as created by the factory
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilter.java:	 * @param module - the CudaModule containing the kernels compiled by the factory
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilter.java:	public GrayscaleFilter (int h, int w, CudaContext context, CudaStream stream, CudaModule module) {
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilter.java:		//setup cuda function
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilter.java:	 * Convert the image into a grayscaled image stored as an 1D float array on the GPU.
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilter.java:	 * The output is left in GPU memory for further processing.
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilter.java:	public void applyGPU(CudaMem image, CudaMemFloat output) {
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilter.java:		//call GPU kernel to convert the color values to grayscaled float values
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/GrayscaleFilter.java:	 * cleans up allocated GPU memory
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/grayscalefilter.cu: * This file contains the CUDA kernel for converting an image into
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilter.java:import jcuda.*;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilter.java:	protected CudaContext _context;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilter.java:	protected CudaStream _stream;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilter.java:	//handles to CUDA kernels
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilter.java:	protected CudaFunction _computeMeanVertically;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilter.java:	protected CudaFunction _computeMeanHorizontally;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilter.java:	 * @param context - the CudaContext as created by the factory
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilter.java:	 * @param stream - the CudaStream as created by the factory
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilter.java:	 * @param module - the CudaModule containing the kernels compiled by the factory
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilter.java:	public ZeroMeanTotalFilter (int h, int w, CudaContext context, CudaStream stream, CudaModule module) {
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilter.java:		// Setup cuda functions
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilter.java:	 * Applies the Zero Mean Total filter on the GPU.
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilter.java:	 * The input image is already in GPU memory and the output is also left
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilter.java:	 * on the GPU.
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilter.java:	public void applyGPU(CudaMemFloat input) {
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilter.java:		//applyGPU zero mean filter vertically
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilter.java:		//applyGPU the horizontal filter again to the transposed values
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/ZeroMeanTotalFilter.java:     * cleans up GPU memory
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/fastnoisefilter.cu: * This file contains the CUDA kernels for extracting a PRNU pattern from a
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:import nl.esciencecenter.rocket.cubaapi.CudaContext;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:import nl.esciencecenter.rocket.cubaapi.CudaMem;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:import nl.esciencecenter.rocket.cubaapi.CudaMemByte;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:import nl.esciencecenter.rocket.cubaapi.CudaMemFloat;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:import nl.esciencecenter.rocket.cubaapi.CudaModule;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:import nl.esciencecenter.rocket.cubaapi.CudaStream;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:import jcuda.*;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java: * PRNUFilter is created for a specific image size. The CUDA source files
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java: * allocate GPU memory, etc.
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:	protected CudaMemFloat d_image;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:	private CudaModule[] modules;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:	protected CudaStream stream;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:	 * This constructor creates a CUDA stream for this filter, and
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:	 * @param context - CudaContext object as created by PRNUFilterFactory
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:	public PRNUFilter(int height, int width, CudaContext context, boolean applySpectralFilter, String... compileArgs) throws CudaException, IOException {
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:		modules = new CudaModule[filenames.length];
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:		//setup GPU memory
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:		stream = new CudaStream();
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:	public void applyGPU(CudaMem input, CudaMem output) {
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:		applyGPU(input.asBytes(), output.asFloats());
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:	public void applyGPU(CudaMemByte rgbImage, CudaMemFloat output) {
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:			grayscaleFilter.applyGPU(rgbImage, d_image);
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:			fastNoiseFilter.applyGPU(d_image);
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:			zeroMeanTotalFilter.applyGPU(d_image);
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:			wienerFilter.applyGPU(d_image);
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:				spectralFilter.applyGPU(d_image, output);
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:	 * cleans up allocated GPU memory and other resources
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/PRNUFilter.java:		//free GPU memory
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilter.java:import jcuda.*;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilter.java:	protected CudaContext _context;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilter.java:	protected CudaStream _stream;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilter.java:	//handles to CUDA kernels
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilter.java:	protected CudaFunction _normalized_gradient;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilter.java:	protected CudaFunction _gradient;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilter.java:	protected CudaMemFloat _d_dxs;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilter.java:	protected CudaMemFloat _d_dys;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilter.java:	 * @param context - the CudaContext as created by the factory
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilter.java:	 * @param stream - the CudaStream as created by the factory
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilter.java:	 * @param module - the CudaModule containing the kernels compiled by the factory
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilter.java:	public FastNoiseFilter (int h, int w, CudaContext context, CudaStream stream, CudaModule module) {
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilter.java:		//setup cuda functions
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilter.java:		// Allocate the CUDA buffers for this kernel
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilter.java:	 * This method applies the FastNoise Filter on the GPU.
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilter.java:	 * The input is already in GPU memory.
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilter.java:	public void applyGPU(CudaMemFloat input) {
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/FastNoiseFilter.java:	 * cleans up GPU memory
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilter.java:import jcuda.CudaException;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilter.java:import jcuda.jcufft.JCufft;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilter.java:import jcuda.jcufft.cufftHandle;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilter.java:import jcuda.jcufft.cufftResult;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilter.java:import jcuda.jcufft.cufftType;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilter.java:import jcuda.runtime.cudaStream_t;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilter.java:import nl.esciencecenter.rocket.cubaapi.CudaContext;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilter.java:import nl.esciencecenter.rocket.cubaapi.CudaMemFloat;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilter.java:import nl.esciencecenter.rocket.cubaapi.CudaStream;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilter.java:    protected CudaContext _context;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilter.java:    protected CudaStream _stream;
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilter.java:    public SpectralFilter(int h, int w, CudaContext context, CudaStream stream) {
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilter.java:            throw new CudaException(cufftResult.stringFor(res));
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilter.java:        res = JCufft.cufftSetStream(_planr2c, new cudaStream_t(_stream.cuStream()));
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilter.java:            throw new CudaException(cufftResult.stringFor(res));
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/SpectralFilter.java:    public void applyGPU(CudaMemFloat input, CudaMemFloat d_output) {
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/zeromeantotalfilter.cu: * This file contains CUDA kernels for applying a zero-mean total
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/wienerfilter.cu: * This file contains CUDA kernels for applying a Wiener filter to a 
src/main/java/nl/esciencecenter/common_source_identification/kernels/filter/wienerfilter.cu: * Simple CUDA Helper function to reduce the output of a
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:import jcuda.*;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:import jcuda.runtime.JCuda;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:import jcuda.driver.*;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java: * This class is performs a Normalized Cross Correlation on the GPU
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:    private static int reducing_thread_blocks = 1024; //optimally this equals the number of SMs in the GPU
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:    //cuda handles
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:    protected CudaModule _module;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:    protected CudaContext _context;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:    protected CudaStream _stream;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:    //handles to CUDA kernels
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:    protected CudaFunction _computeSums;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:    protected CudaFunction _computeNCC;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:    // CUDA memory
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:    protected CudaMemDouble _partialXX;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:    protected CudaMemDouble _partialX;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:    protected CudaMemDouble _partialYY;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:    protected CudaMemDouble _partialY;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:    protected CudaMemDouble _partialXY;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:     * Constructor for the Normalized Cross Correlation GPU implementation
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:     * @param context   - the CudaContext as created by the factory
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:    public NormalizedCrossCorrelation(int h, int w, CudaContext context, String... compileArgs)
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:            throws CudaException, IOException {
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:        JCudaDriver.setExceptionsEnabled(true);
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:        _stream = new CudaStream();
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:        //setup CUDA functions
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:        JCudaDriver.setExceptionsEnabled(true);
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:    public void applyGPU(CudaMem left, CudaMem right, CudaMem result) {
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/NormalizedCrossCorrelation.java:     * Cleans up GPU memory 
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/peaktocorrelationenergy.cu: * This file contains CUDA kernels for comparing two PRNU noise patterns
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/peaktocorrelationenergy.cu: * Simple CUDA Helper function to reduce the output of a
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/peaktocorrelationenergy.cu: * Simple CUDA helper functions to reduce the output of a reducing kernel with multiple
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/normalizedcrosscorrelation.cu: * This file contains CUDA kernels for comparing two PRNU noise patterns
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/normalizedcrosscorrelation.cu: * Simple CUDA Helper function to reduce the output of a
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PatternComparator.java:import jcuda.Pointer;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PatternComparator.java:import nl.esciencecenter.rocket.cubaapi.CudaMem;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PatternComparator.java:   public void applyGPU(CudaMem left, CudaMem right, CudaMem result);
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:import jcuda.CudaException;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:import jcuda.Sizeof;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:import jcuda.runtime.cudaStream_t;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:import jcuda.driver.*;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:import jcuda.jcufft.*;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:import jcuda.Pointer;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java: * Class for comparing PRNU patterns using Peak to Correlation Energy ratio on the GPU
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    protected CudaModule _module;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    protected CudaContext _context;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    protected CudaStream _stream;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    protected CudaStream _stream2;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    protected CudaMemFloat _d_input;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    protected CudaEvent _event;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    //handles to CUDA kernels
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    protected CudaFunction _computeEnergy;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    protected CudaFunction _sumDoubles;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    protected CudaFunction _computeCrossCorr;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    protected CudaFunction _findPeak;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    protected CudaFunction _maxlocFloats;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    protected CudaFunction _computePCE;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    protected CudaMemFloat _d_inputx;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    protected CudaMemFloat _d_inputy;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    protected CudaMemFloat _d_x;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    protected CudaMemFloat _d_y;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    protected CudaMemFloat _d_c;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    protected CudaMemInt _d_peakIndex;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    protected CudaMemFloat _d_peakValues;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    protected CudaMemFloat _d_peakValue;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    protected CudaMemDouble _d_energy;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    protected CudaMemDouble _d_pce;
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:     * @param context - the CudaContext as created by the factory
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    public PeakToCorrelationEnergy(int h, int w, CudaContext context, boolean usePeak,
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:                                   String... compileArgs) throws CudaException, IOException {
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:        JCudaDriver.setExceptionsEnabled(true);
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:        _stream = new CudaStream();
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:        _stream2 = new CudaStream();
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:        _event = new CudaEvent();
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:        //setup CUDA functions
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:        //System.out.println("detected " + num_sm + " SMs on GPU");
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:        //JCuda.cudaMemGetInfo(free, total);
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:        //System.out.println("Before allocations in PCE free GPU mem: " + free[0]/1024/1024 + " MB total: " + total[0]/1024/1024 + " MB ");
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:        //allocate local variables in GPU memory
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:        //JCuda.cudaMemGetInfo(free, total);
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:        //System.out.println("After allocations in PCE free GPU mem: " + free[0]/1024/1024 + " MB total: " + total[0]/1024/1024 + " MB ");
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:            throw new CudaException(cufftResult.stringFor(res));
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:        res = JCufft.cufftSetStream(_plan1, new cudaStream_t(_stream.cuStream()));
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:            throw new CudaException(cufftResult.stringFor(res));
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:        //construct parameter lists for the CUDA kernels
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    public void compare(CudaMemFloat d_x, CudaMemFloat d_y, CudaMemDouble result) {
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:    public void applyGPU(CudaMem left, CudaMem right, CudaMem result) {
src/main/java/nl/esciencecenter/common_source_identification/kernels/compare/PeakToCorrelationEnergy.java:     * Cleans up GPU memory and destroys FFT plan
src/main/java/nl/esciencecenter/microscopy_particle_registration/ParticleRegistrationContext.java:import jcuda.Sizeof;
src/main/java/nl/esciencecenter/microscopy_particle_registration/ParticleRegistrationContext.java:import nl.esciencecenter.rocket.cubaapi.CudaContext;
src/main/java/nl/esciencecenter/microscopy_particle_registration/ParticleRegistrationContext.java:import nl.esciencecenter.rocket.cubaapi.CudaMem;
src/main/java/nl/esciencecenter/microscopy_particle_registration/ParticleRegistrationContext.java:import nl.esciencecenter.rocket.cubaapi.CudaMemDouble;
src/main/java/nl/esciencecenter/microscopy_particle_registration/ParticleRegistrationContext.java:import nl.esciencecenter.rocket.cubaapi.CudaStream;
src/main/java/nl/esciencecenter/microscopy_particle_registration/ParticleRegistrationContext.java:    private CudaStream stream;
src/main/java/nl/esciencecenter/microscopy_particle_registration/ParticleRegistrationContext.java:    public ParticleRegistrationContext(CudaContext context, PairFitting pairFitting, ExpDist expDist, int maxSize) throws IOException {
src/main/java/nl/esciencecenter/microscopy_particle_registration/ParticleRegistrationContext.java:    public long preprocessInputGPU(ParticleIdentifier s, CudaMem input, CudaMem output) {
src/main/java/nl/esciencecenter/microscopy_particle_registration/ParticleRegistrationContext.java:    public long correlateGPU(
src/main/java/nl/esciencecenter/microscopy_particle_registration/ParticleRegistrationContext.java:            CudaMem leftMem,
src/main/java/nl/esciencecenter/microscopy_particle_registration/ParticleRegistrationContext.java:            CudaMem rightMem,
src/main/java/nl/esciencecenter/microscopy_particle_registration/ParticleRegistrationContext.java:            CudaMem outputMem
src/main/java/nl/esciencecenter/microscopy_particle_registration/ParticleRegistrationContext.java:        CudaMemDouble model = leftMem.asDoubles().slice(0, 2 * m);
src/main/java/nl/esciencecenter/microscopy_particle_registration/ParticleRegistrationContext.java:        CudaMemDouble scene = rightMem.asDoubles().slice(0, 2 * n);
src/main/java/nl/esciencecenter/microscopy_particle_registration/ParticleRegistrationContext.java:        CudaMemDouble modelSigmas = leftMem.asDoubles().slice(2 * m, m);
src/main/java/nl/esciencecenter/microscopy_particle_registration/ParticleRegistrationContext.java:        CudaMemDouble sceneSigmas = rightMem.asDoubles().slice(2 * n, n);
src/main/java/nl/esciencecenter/microscopy_particle_registration/ParticleRegistrationContext.java:            pairFitting.applyGPU(
src/main/java/nl/esciencecenter/microscopy_particle_registration/ParticleRegistrationContext.java:            double score = expDist.applyGPU(
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/transformrigid2d/TransformRigid2D.java:import jcuda.CudaException;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/transformrigid2d/TransformRigid2D.java:import nl.esciencecenter.rocket.cubaapi.CudaContext;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/transformrigid2d/TransformRigid2D.java:import nl.esciencecenter.rocket.cubaapi.CudaFunction;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/transformrigid2d/TransformRigid2D.java:import nl.esciencecenter.rocket.cubaapi.CudaMemDouble;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/transformrigid2d/TransformRigid2D.java:import nl.esciencecenter.rocket.cubaapi.CudaModule;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/transformrigid2d/TransformRigid2D.java:import nl.esciencecenter.rocket.cubaapi.CudaStream;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/transformrigid2d/TransformRigid2D.java:    //cuda handles
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/transformrigid2d/TransformRigid2D.java:    protected CudaModule _module;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/transformrigid2d/TransformRigid2D.java:    protected CudaFunction _transformRigid;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/transformrigid2d/TransformRigid2D.java:    public TransformRigid2D(CudaContext context) throws CudaException, IOException {
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/transformrigid2d/TransformRigid2D.java:    public void applyGPU(CudaStream stream, CudaMemDouble input, CudaMemDouble output,
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/resamplecloud2d/ResampleCloud2D.java:import nl.esciencecenter.rocket.cubaapi.CudaMemDouble;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/resamplecloud2d/ResampleCloud2D.java:    public void applyGPU(
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/resamplecloud2d/ResampleCloud2D.java:            CudaMemDouble pos,
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/resamplecloud2d/ResampleCloud2D.java:            CudaMemDouble sigma,
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/resamplecloud2d/ResampleCloud2D.java:            CudaMemDouble resampledPos,
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/resamplecloud2d/ResampleCloud2D.java:            CudaMemDouble resampledSigma,
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:import jcuda.CudaException;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:import jcuda.Sizeof;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:import nl.esciencecenter.rocket.cubaapi.CudaContext;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:import nl.esciencecenter.rocket.cubaapi.CudaFunction;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:import nl.esciencecenter.rocket.cubaapi.CudaMemDouble;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:import nl.esciencecenter.rocket.cubaapi.CudaModule;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:import nl.esciencecenter.rocket.cubaapi.CudaPinned;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:import nl.esciencecenter.rocket.cubaapi.CudaStream;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:    //cuda handles
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:    protected CudaContext _context;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:    protected CudaStream _stream;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:    protected CudaModule _module;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:    protected CudaFunction _expDist;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:    protected CudaFunction _reduceCrossTerm;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:    protected CudaMemDouble _d_transformedModel;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:    protected CudaMemDouble _d_crossTerm;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:    protected CudaPinned _h_crossTerm;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:    public ExpDist(CudaContext context, int maxSize) throws CudaException, IOException {
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:    public double applyGPU(
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:            CudaMemDouble a,
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:            CudaMemDouble b,
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:            CudaMemDouble scaleA,
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:            CudaMemDouble scaleB
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:        _transformKernel.applyGPU(
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:                _h_crossTerm.asCudaMem(),
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/expdist/ExpDist.java:     * Cleans up GPU memory
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/PairFitting.java:import jcuda.CudaException;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/PairFitting.java:import jcuda.Sizeof;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/PairFitting.java:import nl.esciencecenter.rocket.cubaapi.CudaContext;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/PairFitting.java:import nl.esciencecenter.rocket.cubaapi.CudaMemDouble;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/PairFitting.java:import nl.esciencecenter.rocket.cubaapi.CudaPinned;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/PairFitting.java:    private CudaPinned gradient;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/PairFitting.java:    private CudaPinned crossTerm;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/PairFitting.java:    public PairFitting(CudaContext context, double scale, double tolerance, int maxIterations, int maxSize) throws IOException, CudaException {
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/PairFitting.java:            CudaMemDouble model,
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/PairFitting.java:            CudaMemDouble scene
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/PairFitting.java:        gaussTransform.applyGPU(
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/PairFitting.java:                gradient.asCudaMem().asDoubles(),
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/PairFitting.java:                crossTerm.asCudaMem().asDoubles());
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/PairFitting.java:    public double applyGPU(
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/PairFitting.java:            CudaMemDouble model,
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/PairFitting.java:            CudaMemDouble scene,
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransform.java:import jcuda.CudaException;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransform.java:import nl.esciencecenter.rocket.cubaapi.CudaContext;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransform.java:import nl.esciencecenter.rocket.cubaapi.CudaFunction;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransform.java:import nl.esciencecenter.rocket.cubaapi.CudaMemDouble;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransform.java:import nl.esciencecenter.rocket.cubaapi.CudaModule;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransform.java:import nl.esciencecenter.rocket.cubaapi.CudaStream;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransform.java:    //cuda handles
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransform.java:    protected CudaContext _context;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransform.java:    protected CudaStream _stream;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransform.java:    protected CudaModule _module;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransform.java:    protected CudaFunction _gaussTransform;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransform.java:    protected CudaFunction _reduceTerms;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransform.java:    protected CudaMemDouble _d_crossTermPartials;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransform.java:    protected CudaMemDouble _d_gradientPartials;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransform.java:    protected CudaMemDouble _d_transformedModel;
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransform.java:    public GaussTransform(CudaContext context, TransformRigid2D transformKernel, int maxSize) throws CudaException, IOException {
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransform.java:    public void applyGPU(
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransform.java:            CudaMemDouble model,
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransform.java:            CudaMemDouble scene,
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransform.java:            CudaMemDouble gradient,
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransform.java:            CudaMemDouble crossTerm
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransform.java:        _transformKernel.applyGPU(
src/main/java/nl/esciencecenter/microscopy_particle_registration/kernels/gausstransform/GaussTransform.java:     * Cleans up GPU memory
src/main/java/nl/esciencecenter/radio_correlator/CorrelatorContext.java:import jcuda.CudaException;
src/main/java/nl/esciencecenter/radio_correlator/CorrelatorContext.java:import jcuda.Sizeof;
src/main/java/nl/esciencecenter/radio_correlator/CorrelatorContext.java:import nl.esciencecenter.rocket.cubaapi.CudaContext;
src/main/java/nl/esciencecenter/radio_correlator/CorrelatorContext.java:import nl.esciencecenter.rocket.cubaapi.CudaMem;
src/main/java/nl/esciencecenter/radio_correlator/CorrelatorContext.java:import nl.esciencecenter.rocket.cubaapi.CudaStream;
src/main/java/nl/esciencecenter/radio_correlator/CorrelatorContext.java:    private CudaContext context;
src/main/java/nl/esciencecenter/radio_correlator/CorrelatorContext.java:    private CudaStream stream;
src/main/java/nl/esciencecenter/radio_correlator/CorrelatorContext.java:    CorrelatorContext(CudaContext context, int numChannels, int numTimes) throws IOException, CudaException {
src/main/java/nl/esciencecenter/radio_correlator/CorrelatorContext.java:    public long preprocessInputGPU(StationIdentifier key, CudaMem input, CudaMem output) {
src/main/java/nl/esciencecenter/radio_correlator/CorrelatorContext.java:    public long correlateGPU(
src/main/java/nl/esciencecenter/radio_correlator/CorrelatorContext.java:            StationIdentifier leftKey, CudaMem leftMem,
src/main/java/nl/esciencecenter/radio_correlator/CorrelatorContext.java:            StationIdentifier rightKey, CudaMem rightMem,
src/main/java/nl/esciencecenter/radio_correlator/CorrelatorContext.java:            CudaMem output
src/main/java/nl/esciencecenter/radio_correlator/CorrelatorContext.java:        correlator.compareGPU(
src/main/java/nl/esciencecenter/radio_correlator/kernels/Correlator.java:import jcuda.CudaException;
src/main/java/nl/esciencecenter/radio_correlator/kernels/Correlator.java:import nl.esciencecenter.rocket.cubaapi.CudaContext;
src/main/java/nl/esciencecenter/radio_correlator/kernels/Correlator.java:import nl.esciencecenter.rocket.cubaapi.CudaFunction;
src/main/java/nl/esciencecenter/radio_correlator/kernels/Correlator.java:import nl.esciencecenter.rocket.cubaapi.CudaMemFloat;
src/main/java/nl/esciencecenter/radio_correlator/kernels/Correlator.java:import nl.esciencecenter.rocket.cubaapi.CudaModule;
src/main/java/nl/esciencecenter/radio_correlator/kernels/Correlator.java:import nl.esciencecenter.rocket.cubaapi.CudaStream;
src/main/java/nl/esciencecenter/radio_correlator/kernels/Correlator.java:    private CudaStream stream;
src/main/java/nl/esciencecenter/radio_correlator/kernels/Correlator.java:    private CudaContext context;
src/main/java/nl/esciencecenter/radio_correlator/kernels/Correlator.java:    private CudaModule module;
src/main/java/nl/esciencecenter/radio_correlator/kernels/Correlator.java:    private CudaFunction correlate;
src/main/java/nl/esciencecenter/radio_correlator/kernels/Correlator.java:    public Correlator(CudaContext context, int numChannels, int numTimes) throws IOException, CudaException {
src/main/java/nl/esciencecenter/radio_correlator/kernels/Correlator.java:                getClass().getResource("gpu_correlator_1x1.cu"),
src/main/java/nl/esciencecenter/radio_correlator/kernels/Correlator.java:    public void compareGPU(CudaMemFloat left, CudaMemFloat right, CudaMemFloat result) {
src/main/java/nl/esciencecenter/rocket/cache/HostCache.java:import jcuda.CudaException;
src/main/java/nl/esciencecenter/rocket/cache/HostCache.java:import nl.esciencecenter.rocket.cubaapi.CudaContext;
src/main/java/nl/esciencecenter/rocket/cache/HostCache.java:import nl.esciencecenter.rocket.cubaapi.CudaPinned;
src/main/java/nl/esciencecenter/rocket/cache/HostCache.java:import static jcuda.driver.JCudaDriver.CU_MEMHOSTALLOC_PORTABLE;
src/main/java/nl/esciencecenter/rocket/cache/HostCache.java:public class HostCache extends AbstractCache<CudaPinned, Long> {
src/main/java/nl/esciencecenter/rocket/cache/HostCache.java:    private ArrayDeque<CudaPinned> buffers;
src/main/java/nl/esciencecenter/rocket/cache/HostCache.java:    public HostCache(CudaContext ctx, long totalSize, long bufferSize) throws CudaException {
src/main/java/nl/esciencecenter/rocket/cache/HostCache.java:            for (CudaPinned buffer: buffers) {
src/main/java/nl/esciencecenter/rocket/cache/HostCache.java:    protected Optional<CudaPinned> createBuffer(String key) {
src/main/java/nl/esciencecenter/rocket/cache/HostCache.java:    protected void destroyBuffer(CudaPinned buffer) {
src/main/java/nl/esciencecenter/rocket/cache/DeviceCache.java:import jcuda.CudaException;
src/main/java/nl/esciencecenter/rocket/cache/DeviceCache.java:import nl.esciencecenter.rocket.cubaapi.CudaContext;
src/main/java/nl/esciencecenter/rocket/cache/DeviceCache.java:import nl.esciencecenter.rocket.cubaapi.CudaMemByte;
src/main/java/nl/esciencecenter/rocket/cache/DeviceCache.java:public class DeviceCache extends AbstractCache<CudaMemByte, Long> {
src/main/java/nl/esciencecenter/rocket/cache/DeviceCache.java:    private ArrayDeque<CudaMemByte> buffers;
src/main/java/nl/esciencecenter/rocket/cache/DeviceCache.java:    public DeviceCache(CudaContext ctx, long totalSize, long bufferSize) throws CudaException {
src/main/java/nl/esciencecenter/rocket/cache/DeviceCache.java:        } catch (CudaException e) {
src/main/java/nl/esciencecenter/rocket/cache/DeviceCache.java:            for (CudaMemByte buffer: buffers) {
src/main/java/nl/esciencecenter/rocket/cache/DeviceCache.java:    protected Optional<CudaMemByte> createBuffer(String key) {
src/main/java/nl/esciencecenter/rocket/cache/DeviceCache.java:    protected void destroyBuffer(CudaMemByte buffer) {
src/main/java/nl/esciencecenter/rocket/RocketLauncherArgs.java:    @Parameter(names="--devices", description="Ordinals of CUDA devices to use (for example \"0,2,3\"). If empty, all available devices are used.", arity=1)
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:import static jcuda.driver.CUresult.CUDA_SUCCESS;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:import static jcuda.driver.JCudaDriver.cuDeviceGet;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:import static jcuda.driver.JCudaDriver.cuDeviceGetCount;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:import static jcuda.driver.JCudaDriver.cuDeviceGetName;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:import static jcuda.driver.JCudaDriver.cuInit;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:import jcuda.driver.CUdevice;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:import jcuda.driver.CUdevice_attribute;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:import jcuda.driver.CUdevprop;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:import jcuda.driver.JCudaDriver;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:import jcuda.runtime.JCuda;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:import jcuda.runtime.cudaDeviceProp;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:public final class CudaDevice {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:    CudaDevice(final int ordinal) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:    public CudaContext createContext() {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:        return new CudaContext(_device, this);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:        JCuda.cudaDeviceSetSharedMemConfig(config);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:    public static CudaDevice[] getDevices() {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:        final CudaDevice[] devices = new CudaDevice[count[0]];
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:            devices[i] = new CudaDevice(i);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:    public static CudaDevice getBestDevice() {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:        //selecting a GPU based on the largest number of SMs in all GPUs
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:            cudaDeviceProp prop = new cudaDeviceProp();
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:            JCuda.cudaGetDeviceProperties(prop, i);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:        return new CudaDevice(max_ind);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:        JCudaDriver.setExceptionsEnabled(true);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:        JCudaDriver.cuDeviceGetAttribute(pi, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, _device);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:        JCudaDriver.cuDeviceComputeCapability(major, minor, _device);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:        JCudaDriver.cuDeviceTotalMem(amountOfBytes, _device);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:        JCudaDriver.cuDeviceGetProperties(prop, _device);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaDevice.java:        if (cuDeviceGetName(name, name.length, _device) == CUDA_SUCCESS) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemLong.java:import jcuda.Pointer;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemLong.java:import jcuda.Sizeof;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemLong.java:import jcuda.driver.CUdeviceptr;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemLong.java:public final class CudaMemLong extends CudaMem {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemLong.java:    CudaMemLong(CUdeviceptr ptr, long elementCount) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemLong.java:    public void copyFromHostAsync(final long[] src, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemLong.java:    public void copyToHostAsync(final long[] dst, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemLong.java:    public void copyToDevice(final CudaMemLong mem) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemLong.java:    public void copyFromDevice(final CudaMemLong mem) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemLong.java:    public void copyToDeviceAsync(final CudaMemLong mem, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemLong.java:    public void copyFromDeviceAsync(final CudaMemLong mem, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemLong.java:    public CudaMemLong slice(long offset) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemLong.java:    public CudaMemLong slice(long offset, long length) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:import static jcuda.driver.JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:import static jcuda.driver.JCudaDriver.CU_MEMHOSTALLOC_PORTABLE;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:import static jcuda.driver.JCudaDriver.cuMemFreeHost;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:import static jcuda.driver.JCudaDriver.cuMemHostAlloc;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:import static jcuda.driver.JCudaDriver.cuMemHostGetDevicePointer;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:import static jcuda.driver.JCudaDriver.cuMemcpyDtoHAsync;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:import static jcuda.driver.JCudaDriver.cuMemcpyHtoDAsync;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:import jcuda.Pointer;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:import jcuda.driver.CUdeviceptr;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:public class CudaPinned {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:    CudaPinned(Pointer hostptr, long bytesize, int flags) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:    public CudaMem asCudaMem() {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:        return new CudaMem(asDevicePointer(), _sizeInByte);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:    public void copyToDevice(final CudaMem dstDev) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:    public void copyToDevice(final CudaMem dstDev, final long byteCount) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:    public void copyFromDevice(final CudaMem srcDev) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:    public void copyFromDevice(final CudaMem srcDev, final long byteCount) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:    public void copyToDeviceAsync(final CudaMem srcDev, final long byteCount, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:    public void copyToDeviceAsync(final CUdeviceptr srcDev, final long byteCount, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:    public void copyFromDeviceAsync(final CudaMem dstDev, final long byteCount, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:    public void copyFromDeviceAsync(final CUdeviceptr dstDev, final long byteCount, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:    public CudaPinned sliceBytes(long offset, long length) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaPinned.java:        return new CudaPinned(_hostptr.withByteOffset(offset), length, _flags);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaFunction.java:import static jcuda.driver.JCudaDriver.cuLaunchKernel;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaFunction.java:import jcuda.NativePointerObject;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaFunction.java:import jcuda.Pointer;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaFunction.java:import jcuda.driver.CUfunction;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaFunction.java:public final class CudaFunction {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaFunction.java:	CudaFunction(final CUfunction function) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaFunction.java:	public void launch(CudaStream stream, Pointer parameters) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaFunction.java:	public void launch(CudaStream stream, Object ...args) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaFunction.java:			} else if (arg instanceof CudaMem) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaFunction.java:				p = Pointer.to(((CudaMem)arg).asDevicePointer());
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaFunction.java:				throw new IllegalArgumentException("invalid CudaFunction launch argument: " +
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaEvent.java:import static jcuda.driver.JCudaDriver.cuEventCreate;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaEvent.java:import static jcuda.driver.JCudaDriver.cuEventDestroy;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaEvent.java:import static jcuda.driver.JCudaDriver.cuEventRecord;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaEvent.java:import static jcuda.driver.JCudaDriver.cuEventSynchronize;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaEvent.java:import jcuda.driver.CUevent;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaEvent.java:import jcuda.driver.CUevent_flags;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaEvent.java:public class CudaEvent {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaEvent.java:    public CudaEvent() {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaEvent.java:    public void record(final CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaModule.java:import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaModule.java:import jcuda.CudaException;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaModule.java:import jcuda.driver.CUfunction;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaModule.java:import jcuda.driver.CUmodule;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaModule.java:import jcuda.driver.JCudaDriver;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaModule.java:public final class CudaModule {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaModule.java:    CudaModule(final String cuSource, final String[] nvccOptions) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaModule.java:            throw new CudaException("Failed to compile CUDA source", e);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaModule.java:        JCudaDriver.cuModuleLoadData(_module, _cubinData);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaModule.java:        final File cuFile = File.createTempFile("jcuda", ".cu");
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaModule.java:        final File cubinFile = File.createTempFile("jcuda", ".cubin");
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaModule.java:    public CudaFunction getFunction(final String name) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaModule.java:        return new CudaFunction(function);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaModule.java:    	JCudaDriver.cuModuleUnload(_module);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemInt.java:import jcuda.Pointer;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemInt.java:import jcuda.Sizeof;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemInt.java:import jcuda.driver.CUdeviceptr;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemInt.java:public final class CudaMemInt extends CudaMem {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemInt.java:    CudaMemInt(CUdeviceptr ptr, long elementCount) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemInt.java:    public void copyFromHostAsync(final int[] src, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemInt.java:    public void copyToHostAsync(final int[] dst, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemInt.java:    public void copyToDevice(final CudaMemInt mem) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemInt.java:    public void copyFromDevice(final CudaMemInt mem) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemInt.java:    public void copyToDeviceAsync(final CudaMemInt mem, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemInt.java:    public void copyFromDeviceAsync(final CudaMemInt mem, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemInt.java:    public void fillAsync(int val, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemInt.java:    public CudaMemInt slice(long offset) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemInt.java:    public CudaMemInt slice(long offset, long length) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:import static jcuda.driver.JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:import static jcuda.driver.JCudaDriver.CU_MEMHOSTALLOC_PORTABLE;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:import static jcuda.driver.JCudaDriver.cuCtxCreate;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:import static jcuda.driver.JCudaDriver.cuCtxDestroy;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:import static jcuda.driver.JCudaDriver.cuCtxPopCurrent;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:import static jcuda.driver.JCudaDriver.cuCtxPushCurrent;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:import static jcuda.driver.JCudaDriver.cuGetErrorName;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:import static jcuda.driver.JCudaDriver.cuGetErrorString;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:import static jcuda.driver.JCudaDriver.cuMemAlloc;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:import static jcuda.driver.JCudaDriver.cuMemHostAlloc;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:import static jcuda.runtime.cudaError.cudaSuccess;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:import jcuda.CudaException;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:import jcuda.Pointer;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:import jcuda.driver.CUcontext;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:import jcuda.driver.CUdevice;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:import jcuda.driver.CUdeviceptr;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:import jcuda.driver.JCudaDriver;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:import jcuda.runtime.cudaError;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:public final class CudaContext {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:    CudaDevice device;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:    CudaContext(final CUdevice device, CudaDevice cudaDev) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:        this.device = cudaDev;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:    public CudaStream createStream() {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:        return withSupplier(() -> new CudaStream());
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:    public CudaEvent createEvent() {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:        return withSupplier(() -> new CudaEvent());
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:    public CudaModule compileModule(URL url, String... options) throws IOException {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:    public CudaModule compileModule(String source, String... options) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:        //compile the CUDA code to run on the GPU
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:    public CudaModule loadModule(final String cuSource, final String... nvccOptions) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:        return new CudaModule(cuSource, nvccOptions);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:    public CudaDevice getDevice() {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:            JCudaDriver.cuMemGetInfo(free, new long[1]);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:            JCudaDriver.cuMemGetInfo(new long[1], total);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:    public CudaPinned allocHostBytes(final long elementCount) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:    public CudaPinned allocHostBytes(final long elementCount, int flags) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:            return new CudaPinned(ptr, elementCount, flags);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:    public CudaMemByte allocBytes(final long elementCount) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:            return new CudaMemByte(ptr, elementCount);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:    public CudaMemByte allocBytes(final byte[] data) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:        final CudaMemByte mem = allocBytes(data.length);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:    public CudaMemInt allocInts(final long elementCount) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:        return allocBytes(elementCount * CudaMemInt.ELEMENT_SIZE).asInts();
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:    public CudaMemInt allocInts(final int[] data) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:        final CudaMemInt mem = allocInts(data.length);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:    public CudaMemFloat allocFloats(final long elementCount) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:        return allocBytes(elementCount * CudaMemFloat.ELEMENT_SIZE).asFloats();
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:    public CudaMemFloat allocFloats(final float[] data) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:        final CudaMemFloat mem = allocFloats(data.length);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:    public CudaMemDouble allocDoubles(final long elementCount) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:        return allocBytes(elementCount * CudaMemDouble.ELEMENT_SIZE).asDoubles();
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:    public CudaMemLong allocLongs(final long elementCount) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaContext.java:        return allocBytes(elementCount * CudaMemLong.ELEMENT_SIZE).asLongs();
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:import static jcuda.driver.JCudaDriver.cuMemFree;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:import static jcuda.driver.JCudaDriver.cuMemcpyDtoD;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:import static jcuda.driver.JCudaDriver.cuMemcpyDtoDAsync;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:import static jcuda.driver.JCudaDriver.cuMemcpyDtoHAsync;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:import static jcuda.driver.JCudaDriver.cuMemcpyHtoDAsync;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:import static jcuda.driver.JCudaDriver.cuMemsetD32;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:import static jcuda.driver.JCudaDriver.cuMemsetD32Async;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:import jcuda.Pointer;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:import jcuda.Sizeof;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:import jcuda.driver.CUdeviceptr;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:public class CudaMem {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:    CudaMem(CUdeviceptr ptr, final long size) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:    public CudaMemFloat asFloats() {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:        if (sizeInBytes() %  CudaMemFloat.ELEMENT_SIZE != 0) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:        return new CudaMemFloat(_deviceptr, _sizeInBytes / CudaMemFloat.ELEMENT_SIZE);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:    public CudaMemDouble asDoubles() {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:        if (sizeInBytes() %  CudaMemDouble.ELEMENT_SIZE != 0) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:        return new CudaMemDouble(_deviceptr, _sizeInBytes / CudaMemDouble.ELEMENT_SIZE);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:    public CudaMemInt asInts() {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:        if (sizeInBytes() %  CudaMemInt.ELEMENT_SIZE != 0) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:        return new CudaMemInt(_deviceptr, _sizeInBytes / CudaMemInt.ELEMENT_SIZE);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:    public CudaMemByte asBytes() {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:        if (sizeInBytes() %  CudaMemByte.ELEMENT_SIZE != 0) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:        return new CudaMemByte(_deviceptr, _sizeInBytes / CudaMemByte.ELEMENT_SIZE);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:    public CudaMemLong asLongs() {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:        if (sizeInBytes() %  CudaMemLong.ELEMENT_SIZE != 0) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:        return new CudaMemLong(_deviceptr, _sizeInBytes / CudaMemLong.ELEMENT_SIZE);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:    public void copyFromHostAsync(final Pointer srcHost, final long byteCount, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:    public void copyToHostAsync(final Pointer dstHost, final long byteCount, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:    public void copyToDeviceAsync(final CUdeviceptr dstDev, long byteCount, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:    public void copyFromDeviceAsync(final CUdeviceptr srcDev, long byteCount, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:    protected void memsetD32Async(final int ui, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:    public CudaMem sliceBytes(long offset, long length) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMem.java:        return new CudaMem(_deviceptr.withByteOffset(offset), length);
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemDouble.java:import jcuda.Pointer;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemDouble.java:import jcuda.Sizeof;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemDouble.java:import jcuda.driver.CUdeviceptr;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemDouble.java:public final class CudaMemDouble extends CudaMem {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemDouble.java:    CudaMemDouble(CUdeviceptr ptr, long elementCount) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemDouble.java:    public void copyFromHostAsync(final double[] src, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemDouble.java:    public void copyToHostAsync(final double[] dst, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemDouble.java:    public void copyToDevice(final CudaMemDouble mem) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemDouble.java:    public void copyFromDevice(final CudaMemDouble mem) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemDouble.java:    public void copyToDeviceAsync(final CudaMemDouble mem, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemDouble.java:    public void copyFromDeviceAsync(final CudaMemDouble mem, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemDouble.java:    public CudaMemDouble slice(long offset) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemDouble.java:    public CudaMemDouble slice(long offset, long length) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemByte.java:import jcuda.Pointer;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemByte.java:import jcuda.Sizeof;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemByte.java:import jcuda.driver.CUdeviceptr;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemByte.java:public final class CudaMemByte extends CudaMem {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemByte.java:    CudaMemByte(CUdeviceptr ptr, long elementCount) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemByte.java:    public void copyFromHostAsync(final byte[] src, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemByte.java:    public void copyToHostAsync(final byte[] dst, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemByte.java:    public void copyToDevice(final CudaMemByte mem) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemByte.java:    public void copyFromDevice(final CudaMemByte mem) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemByte.java:    public void copyToDeviceAsync(final CudaMemByte mem, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemByte.java:    public void copyFromDeviceAsync(final CudaMemByte mem, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemByte.java:    public CudaMemByte slice(long offset) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemByte.java:    public CudaMemByte slice(long offset, long length) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaStream.java:import static jcuda.driver.JCudaDriver.cuStreamAddCallback;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaStream.java:import static jcuda.driver.JCudaDriver.cuStreamCreate;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaStream.java:import static jcuda.driver.JCudaDriver.cuStreamDestroy;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaStream.java:import static jcuda.driver.JCudaDriver.cuStreamSynchronize;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaStream.java:import static jcuda.driver.JCudaDriver.cuStreamWaitEvent;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaStream.java:import jcuda.driver.CUstream;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaStream.java:import jcuda.driver.CUstreamCallback;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaStream.java:import jcuda.driver.CUstream_flags;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaStream.java:public final class CudaStream {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaStream.java:    public CudaStream() {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaStream.java:    public void waitEvent(CudaEvent event) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemFloat.java:import jcuda.Pointer;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemFloat.java:import jcuda.Sizeof;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemFloat.java:import jcuda.driver.CUdeviceptr;
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemFloat.java:public final class CudaMemFloat extends CudaMem {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemFloat.java:    CudaMemFloat(CUdeviceptr ptr, long elementCount) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemFloat.java:    public void copyFromHostAsync(final float[] src, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemFloat.java:    public void copyToHostAsync(final float[] dst, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemFloat.java:    public void copyToDevice(final CudaMemFloat mem) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemFloat.java:    public void copyFromDevice(final CudaMemFloat mem) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemFloat.java:    public void copyToDeviceAsync(final CudaMemFloat mem, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemFloat.java:    public void copyFromDeviceAsync(final CudaMemFloat mem, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemFloat.java:    public void fillAsync(float val, CudaStream stream) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemFloat.java:    public CudaMemFloat slice(long offset) {
src/main/java/nl/esciencecenter/rocket/cubaapi/CudaMemFloat.java:    public CudaMemFloat slice(long offset, long length) {
src/main/java/nl/esciencecenter/rocket/util/NodeInfo.java:import nl.esciencecenter.rocket.cubaapi.CudaContext;
src/main/java/nl/esciencecenter/rocket/util/NodeInfo.java:        public DeviceInfo(CudaContext context) {
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:import nl.esciencecenter.rocket.cubaapi.CudaContext;
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:import nl.esciencecenter.rocket.cubaapi.CudaDevice;
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:import nl.esciencecenter.rocket.cubaapi.CudaMemByte;
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:import nl.esciencecenter.rocket.cubaapi.CudaPinned;
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:        public ApplicationContext<K, R> create(CudaContext ctx) throws Throwable;
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:    private void runBenchmark(NodeInfo.DeviceInfo info, CudaContext context, ApplicationContext fun, List<Tuple<K, K>> corrs) {
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:        CudaPinned scratchHost = context.allocHostBytes(bufferSize);
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:        CudaMemByte scratchDev = context.allocBytes(bufferSize);
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:        CudaMemByte patternLeft = context.allocBytes(bufferSize);
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:        CudaMemByte patternRight = context.allocBytes(bufferSize);
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:        CudaMemByte result = context.allocBytes(fun.getMaxOutputSize());
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:                    info.preprocessingTime += benchmark(() -> sizes[0] = fun.preprocessInputGPU(
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:                    info.preprocessingTime += benchmark(() -> sizes[1] = fun.preprocessInputGPU(
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:                    info.correlationTime += benchmark(() -> fun.correlateGPU(
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:        // Prepare CUDA
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:        CudaDevice[] devices = parseDeviceList();
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:        CudaContext[] contexts = new CudaContext[n];
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:            throw new IllegalStateException("no CUDA capable devices detected");
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:            logger.info("creating CUDA context");
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:            CudaContext ctx = devices[i].createContext();
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:            for (CudaContext ctx: contexts) {
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:    private CudaDevice[] parseDeviceList() {
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:        CudaDevice[] allDevices = CudaDevice.getDevices();
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:                    throw new RuntimeException("failed to parse " + part + " as CUDA device ordinal.");
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:                            + " CUDA devices");
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:        CudaDevice[] devices = new CudaDevice[ordinals.size()];
src/main/java/nl/esciencecenter/rocket/RocketLauncher.java:            CudaContext[] contexts,
src/main/java/nl/esciencecenter/rocket/scheduler/ApplicationContext.java:import nl.esciencecenter.rocket.cubaapi.CudaMem;
src/main/java/nl/esciencecenter/rocket/scheduler/ApplicationContext.java:     * Returns the maximum buffer size required to store the output of parseFiles and the input of preprocessInputGPU
src/main/java/nl/esciencecenter/rocket/scheduler/ApplicationContext.java:     * Returns the maximum output buffer size required for the output of correlateGPU. This size is used to
src/main/java/nl/esciencecenter/rocket/scheduler/ApplicationContext.java:     * bytes that will be transferred to the GPU. This is useful for variable-length inputs where only the first bytes
src/main/java/nl/esciencecenter/rocket/scheduler/ApplicationContext.java:     * Performs preprocessing of the input data on the GPU. Data should be read from and written to the given buffer.
src/main/java/nl/esciencecenter/rocket/scheduler/ApplicationContext.java:    public long preprocessInputGPU(K key, CudaMem input, CudaMem output);
src/main/java/nl/esciencecenter/rocket/scheduler/ApplicationContext.java:     * Calculates the correlation of two keys on the GPU. The output data must be written to the given output buffer.
src/main/java/nl/esciencecenter/rocket/scheduler/ApplicationContext.java:    public long correlateGPU(K leftKey, CudaMem left, K rightKey, CudaMem right, CudaMem output);
src/main/java/nl/esciencecenter/rocket/scheduler/ApplicationContext.java:     * Process the result of a correlation (i.e., the result of correlateGPU) and return an object. This object can
src/main/java/nl/esciencecenter/rocket/scheduler/ApplicationContext.java:     * @param output The output data of correlateGPU
src/main/java/nl/esciencecenter/rocket/scheduler/HostWorker.java:import nl.esciencecenter.rocket.cubaapi.CudaPinned;
src/main/java/nl/esciencecenter/rocket/scheduler/HostWorker.java:        public Future<Void> call(CudaPinned src);
src/main/java/nl/esciencecenter/rocket/scheduler/HostWorker.java:        public Future<Long> call(CudaPinned dst);
src/main/java/nl/esciencecenter/rocket/scheduler/DeviceWorker.java:import nl.esciencecenter.rocket.cubaapi.CudaContext;
src/main/java/nl/esciencecenter/rocket/scheduler/DeviceWorker.java:import nl.esciencecenter.rocket.cubaapi.CudaPinned;
src/main/java/nl/esciencecenter/rocket/scheduler/DeviceWorker.java:import nl.esciencecenter.rocket.cubaapi.CudaMem;
src/main/java/nl/esciencecenter/rocket/scheduler/DeviceWorker.java:import nl.esciencecenter.rocket.cubaapi.CudaMemByte;
src/main/java/nl/esciencecenter/rocket/scheduler/DeviceWorker.java:import nl.esciencecenter.rocket.cubaapi.CudaStream;
src/main/java/nl/esciencecenter/rocket/scheduler/DeviceWorker.java:        CudaMemByte dmem;
src/main/java/nl/esciencecenter/rocket/scheduler/DeviceWorker.java:        CudaPinned hmem;
src/main/java/nl/esciencecenter/rocket/scheduler/DeviceWorker.java:        private ScratchMem(CudaContext ctx, long size) {
src/main/java/nl/esciencecenter/rocket/scheduler/DeviceWorker.java:    private CudaContext context;
src/main/java/nl/esciencecenter/rocket/scheduler/DeviceWorker.java:    private ThreadPoolExecutor gpuExecutor;
src/main/java/nl/esciencecenter/rocket/scheduler/DeviceWorker.java:    private CudaStream h2dStream;
src/main/java/nl/esciencecenter/rocket/scheduler/DeviceWorker.java:    private CudaStream d2hStream;
src/main/java/nl/esciencecenter/rocket/scheduler/DeviceWorker.java:            CudaContext ctx,
src/main/java/nl/esciencecenter/rocket/scheduler/DeviceWorker.java:        PriorityBlockingQueue<Runnable> gpuQueue = new PriorityBlockingQueue<>();
src/main/java/nl/esciencecenter/rocket/scheduler/DeviceWorker.java:        this.gpuExecutor = newFixedThreadPool(name + "-execute", 1, gpuQueue);
src/main/java/nl/esciencecenter/rocket/scheduler/DeviceWorker.java:    private Future<Long> loadInput(K f, CudaMem devMem) {
src/main/java/nl/esciencecenter/rocket/scheduler/DeviceWorker.java:                        () -> applicationContext.preprocessInputGPU(
src/main/java/nl/esciencecenter/rocket/scheduler/DeviceWorker.java:    private Future<Long> loadInputFromHostCacheAsync(K f, CudaMem dmem) {
src/main/java/nl/esciencecenter/rocket/scheduler/DeviceWorker.java:                // Submit correlation to GPU
src/main/java/nl/esciencecenter/rocket/scheduler/DeviceWorker.java:                        () -> applicationContext.correlateGPU(
src/main/java/nl/esciencecenter/rocket/scheduler/DeviceWorker.java:    private Future<Void> submitH2D(CudaPinned srcHost, CudaMem dstDev, long size) {
src/main/java/nl/esciencecenter/rocket/scheduler/DeviceWorker.java:    private Future<Void> submitD2H(CudaMem srcDev, CudaPinned dstHost, long size) {
src/main/java/nl/esciencecenter/rocket/scheduler/DeviceWorker.java:        gpuExecutor.execute(new PriorityTask(priority, wrapper));
src/main/java/nl/esciencecenter/rocket/scheduler/DeviceWorker.java:        this.gpuExecutor.shutdown();
src/main/java/nl/esciencecenter/phylogenetics_analysis/PointerHack.java:import jcuda.NativePointerObject;
src/main/java/nl/esciencecenter/phylogenetics_analysis/PointerHack.java:import nl.esciencecenter.rocket.cubaapi.CudaMem;
src/main/java/nl/esciencecenter/phylogenetics_analysis/PointerHack.java:import nl.esciencecenter.rocket.cubaapi.CudaStream;
src/main/java/nl/esciencecenter/phylogenetics_analysis/PointerHack.java:    static class JCUDAPointerProxy extends jcuda.Pointer {
src/main/java/nl/esciencecenter/phylogenetics_analysis/PointerHack.java:        JCUDAPointerProxy(jcuda.Pointer p) {
src/main/java/nl/esciencecenter/phylogenetics_analysis/PointerHack.java:    static public Pointer ptrOf(CudaMem mem) {
src/main/java/nl/esciencecenter/phylogenetics_analysis/PointerHack.java:    static public Pointer ptrOf(CudaStream stream) {
src/main/java/nl/esciencecenter/phylogenetics_analysis/PointerHack.java:        if (obj instanceof jcuda.Pointer) {
src/main/java/nl/esciencecenter/phylogenetics_analysis/PointerHack.java:            ptr += new JCUDAPointerProxy((jcuda.Pointer) obj).byteOffset();
src/main/java/nl/esciencecenter/phylogenetics_analysis/Main.java:import nl.esciencecenter.rocket.cubaapi.CudaContext;
src/main/java/nl/esciencecenter/phylogenetics_analysis/Main.java:import nl.esciencecenter.rocket.cubaapi.CudaDevice;
src/main/java/nl/esciencecenter/phylogenetics_analysis/Main.java:import nl.esciencecenter.rocket.cubaapi.CudaMemByte;
src/main/java/nl/esciencecenter/phylogenetics_analysis/Main.java:import nl.esciencecenter.rocket.cubaapi.CudaMemInt;
src/main/java/nl/esciencecenter/phylogenetics_analysis/Main.java:import nl.esciencecenter.rocket.cubaapi.CudaPinned;
src/main/java/nl/esciencecenter/phylogenetics_analysis/Main.java:        CudaContext context = CudaDevice.getBestDevice().createContext();
src/main/java/nl/esciencecenter/phylogenetics_analysis/Main.java:        CudaPinned inputHost = context.allocHostBytes(maxInputSize);
src/main/java/nl/esciencecenter/phylogenetics_analysis/Main.java:        CudaMemByte inputDev = context.allocBytes(maxInputSize);
src/main/java/nl/esciencecenter/phylogenetics_analysis/Main.java:        CudaMemInt outputKeys = context.allocInts(maxVectorSize);
src/main/java/nl/esciencecenter/phylogenetics_analysis/Main.java:        CudaMemInt outputValues = context.allocInts(maxVectorSize);
src/main/java/nl/esciencecenter/phylogenetics_analysis/Main.java:                long n = kernels.buildVectorGPU(inputDev.slice(0, size), outputKeys, outputValues);
src/main/java/nl/esciencecenter/phylogenetics_analysis/SequenceAnalysisContext.java:import jcuda.Sizeof;
src/main/java/nl/esciencecenter/phylogenetics_analysis/SequenceAnalysisContext.java:import nl.esciencecenter.rocket.cubaapi.CudaContext;
src/main/java/nl/esciencecenter/phylogenetics_analysis/SequenceAnalysisContext.java:import nl.esciencecenter.rocket.cubaapi.CudaMem;
src/main/java/nl/esciencecenter/phylogenetics_analysis/SequenceAnalysisContext.java:import nl.esciencecenter.rocket.cubaapi.CudaMemByte;
src/main/java/nl/esciencecenter/phylogenetics_analysis/SequenceAnalysisContext.java:import nl.esciencecenter.rocket.cubaapi.CudaMemDouble;
src/main/java/nl/esciencecenter/phylogenetics_analysis/SequenceAnalysisContext.java:    private CudaMemByte scratch;
src/main/java/nl/esciencecenter/phylogenetics_analysis/SequenceAnalysisContext.java:    public SequenceAnalysisContext(CudaContext context, String alphabet, int k, int maxVectorSize, int maxInputSize) {
src/main/java/nl/esciencecenter/phylogenetics_analysis/SequenceAnalysisContext.java:    public long preprocessInputGPU(SequenceIdentifier key, CudaMem inputBuf, CudaMem outputBuf) {
src/main/java/nl/esciencecenter/phylogenetics_analysis/SequenceAnalysisContext.java:        CudaMemByte input = inputBuf.asBytes();
src/main/java/nl/esciencecenter/phylogenetics_analysis/SequenceAnalysisContext.java:        CudaMemByte output = outputBuf.asBytes();
src/main/java/nl/esciencecenter/phylogenetics_analysis/SequenceAnalysisContext.java:        CudaMemByte outputKeys = output.slice(0, maxVectorSize * KEY_SIZE);
src/main/java/nl/esciencecenter/phylogenetics_analysis/SequenceAnalysisContext.java:        CudaMemByte outputValues = output.slice(maxVectorSize * KEY_SIZE, maxVectorSize * VALUE_SIZE);
src/main/java/nl/esciencecenter/phylogenetics_analysis/SequenceAnalysisContext.java:        int size = kernel.buildVectorGPU(
src/main/java/nl/esciencecenter/phylogenetics_analysis/SequenceAnalysisContext.java:            CudaMemByte src = output.slice((maxVectorSize + offset) * KEY_SIZE, chunk * KEY_SIZE);
src/main/java/nl/esciencecenter/phylogenetics_analysis/SequenceAnalysisContext.java:            CudaMemByte dst = output.slice((size + offset) * KEY_SIZE, chunk * KEY_SIZE);
src/main/java/nl/esciencecenter/phylogenetics_analysis/SequenceAnalysisContext.java:    public long correlateGPU(SequenceIdentifier leftKey, CudaMem left, SequenceIdentifier rightKey, CudaMem right, CudaMem output) {
src/main/java/nl/esciencecenter/phylogenetics_analysis/SequenceAnalysisContext.java:        CudaMemByte leftKeys = left
src/main/java/nl/esciencecenter/phylogenetics_analysis/SequenceAnalysisContext.java:        CudaMemByte leftValues = left
src/main/java/nl/esciencecenter/phylogenetics_analysis/SequenceAnalysisContext.java:        CudaMemByte rightKeys = right
src/main/java/nl/esciencecenter/phylogenetics_analysis/SequenceAnalysisContext.java:        CudaMemByte rightValues = right
src/main/java/nl/esciencecenter/phylogenetics_analysis/SequenceAnalysisContext.java:        CudaMemDouble result = output.asDoubles();
src/main/java/nl/esciencecenter/phylogenetics_analysis/SequenceAnalysisContext.java:        kernel.compareVectorsGPU(leftKeys, leftValues, leftSize, rightKeys, rightValues, rightSize, result);
src/main/java/nl/esciencecenter/phylogenetics_analysis/kernels/CompositionVectorKernels.java:import jcuda.CudaException;
src/main/java/nl/esciencecenter/phylogenetics_analysis/kernels/CompositionVectorKernels.java:import jcuda.Sizeof;
src/main/java/nl/esciencecenter/phylogenetics_analysis/kernels/CompositionVectorKernels.java:import jcuda.runtime.cudaError;
src/main/java/nl/esciencecenter/phylogenetics_analysis/kernels/CompositionVectorKernels.java:import nl.esciencecenter.rocket.cubaapi.CudaContext;
src/main/java/nl/esciencecenter/phylogenetics_analysis/kernels/CompositionVectorKernels.java:import nl.esciencecenter.rocket.cubaapi.CudaMem;
src/main/java/nl/esciencecenter/phylogenetics_analysis/kernels/CompositionVectorKernels.java:import nl.esciencecenter.rocket.cubaapi.CudaMemByte;
src/main/java/nl/esciencecenter/phylogenetics_analysis/kernels/CompositionVectorKernels.java:import nl.esciencecenter.rocket.cubaapi.CudaMemDouble;
src/main/java/nl/esciencecenter/phylogenetics_analysis/kernels/CompositionVectorKernels.java:import nl.esciencecenter.rocket.cubaapi.CudaStream;
src/main/java/nl/esciencecenter/phylogenetics_analysis/kernels/CompositionVectorKernels.java:    private CudaContext context;
src/main/java/nl/esciencecenter/phylogenetics_analysis/kernels/CompositionVectorKernels.java:    private CudaStream stream;
src/main/java/nl/esciencecenter/phylogenetics_analysis/kernels/CompositionVectorKernels.java:    private CudaMemByte scratch;
src/main/java/nl/esciencecenter/phylogenetics_analysis/kernels/CompositionVectorKernels.java:    public CompositionVectorKernels(CudaContext context, String alphabet, int k, int maxVectorSize, int maxInputSize) {
src/main/java/nl/esciencecenter/phylogenetics_analysis/kernels/CompositionVectorKernels.java:    public int buildVectorGPU(CudaMemByte input, CudaMem outputKeys, CudaMem outputValues) {
src/main/java/nl/esciencecenter/phylogenetics_analysis/kernels/CompositionVectorKernels.java:        if (err != cudaError.cudaSuccess) {
src/main/java/nl/esciencecenter/phylogenetics_analysis/kernels/CompositionVectorKernels.java:            throw new CudaException(cudaError.stringFor(err));
src/main/java/nl/esciencecenter/phylogenetics_analysis/kernels/CompositionVectorKernels.java:    public void compareVectorsGPU(
src/main/java/nl/esciencecenter/phylogenetics_analysis/kernels/CompositionVectorKernels.java:            CudaMem leftKeys, CudaMem leftValues, int leftSize,
src/main/java/nl/esciencecenter/phylogenetics_analysis/kernels/CompositionVectorKernels.java:            CudaMem rightKeys, CudaMem rightValues, int rightSize,
src/main/java/nl/esciencecenter/phylogenetics_analysis/kernels/CompositionVectorKernels.java:            CudaMemDouble result) {
src/main/java/nl/esciencecenter/phylogenetics_analysis/kernels/CompositionVectorKernels.java:        if (err != cudaError.cudaSuccess) {
src/main/java/nl/esciencecenter/phylogenetics_analysis/kernels/CompositionVectorKernels.java:            throw new CudaException(cudaError.stringFor(err));
src/main/java/nl/esciencecenter/phylogenetics_analysis/kernels/CompositionVectorKernels.java:     * Cleans up GPU memory
src/main/java/nl/esciencecenter/phylogenetics_analysis/Bindings.java:import jcuda.Sizeof;
src/main/java/nl/esciencecenter/phylogenetics_analysis/Bindings.java:import nl.esciencecenter.rocket.cubaapi.CudaContext;
src/main/java/nl/esciencecenter/phylogenetics_analysis/Bindings.java:import nl.esciencecenter.rocket.cubaapi.CudaDevice;
src/main/java/nl/esciencecenter/phylogenetics_analysis/Bindings.java:import nl.esciencecenter.rocket.cubaapi.CudaMem;

```
