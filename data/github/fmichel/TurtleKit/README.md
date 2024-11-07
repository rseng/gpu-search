# https://github.com/fmichel/TurtleKit

```console
resource/VERSION:	- upgrading to JCuda 6.5
resource/JCuda_License.txt:JCuda - Java bindings for NVIDIA CUDA
resource/JCuda_License.txt:Copyright (c) 2008-2013 Marco Hutter - http://www.jcuda.org
build.xml:		<include name="jcuda-0.9.0.jar" />
build.xml:		<include name="jcuda-natives-0.9.0-linux-x86_64.jar" />
build.xml:		<include name="jcudaUtils-0.0.4.jar" />
build.xml:		<!-- remove jcuda docs and jfreechart maven files -->
build.xml:		<zipfileset dir="${build.dir}" includes="VERSION,${doc.html},build.xml,JCuda_License.txt" prefix="${docs.dir}" />
build.xml:		<zipfileset file="${resource.dir}/JCuda_License.txt" prefix="${docs.dir}" />
demos/turtlekit/mle/MLEEnvironment.java:import turtlekit.cuda.CudaGPUGradientsPhero;
demos/turtlekit/mle/MLEEnvironment.java:import turtlekit.cuda.CudaPheromone;
demos/turtlekit/mle/MLEEnvironment.java://	    return createCudaPheromone(name, evaporationPercentage, diffusionPercentage);
demos/turtlekit/mle/MLEEnvironment.java://			((CudaPheromoneV3) p).updateV3();
demos/turtlekit/mle/MLEEnvironment.java:	protected Pheromone createCudaPheromone(String name, int evaporationPercentage, int diffusionPercentage){
demos/turtlekit/mle/MLEEnvironment.java:		if(GPU_GRADIENTS && ! name.contains("PRE"))
demos/turtlekit/mle/MLEEnvironment.java:			return new CudaGPUGradientsPhero(name, getWidth(), getHeight(), evaporationPercentage, diffusionPercentage);
demos/turtlekit/mle/MLEEnvironment.java:		return new CudaPheromone(name, getWidth(),	getHeight(), evaporationPercentage, diffusionPercentage);
demos/turtlekit/mle/Particule.java:	// String[] args2 = {"128","10","--GPU_gradients"};
demos/turtlekit/mle/Particule.java:	// String[] args2 = {"256","10","--GPU_gradients"};
demos/turtlekit/mle/Particule.java:	String[] args2 = { "200", "5", "--GPU_gradients" };
demos/turtlekit/mle/Particule.java:	// String[] args2 = {"512","10","--GPU_gradients"};
demos/turtlekit/mle/Particule.java:	// ,"--GPU_gradients"
demos/turtlekit/mle/Particule.java:		, Option.cuda.toString()
demos/turtlekit/mle/Particule.java:	//// ,Option.noCuda.toString()
demos/turtlekit/mle/Particule.java:	// ,Option.environmentClass.toString(),MLEEnvGPUDiffusionAndGradients.class.getName()
demos/turtlekit/mle/MLEScheduler.java:import jcuda.utils.Timer;
demos/turtlekit/mle/MLE_512_Cuda.xml:		cuda="true"
demos/turtlekit/mle/MLELauncher.java:	// setMadkitProperty("GPU_gradients", "true");
demos/turtlekit/mle/MLELauncher.java:	executeThisLauncher("--popDensity", "100", Option.cuda.toString());
demos/turtlekit/toys/PheroEmmiter.java://		System.err.println(pheromone instanceof CudaPheromone);
demos/turtlekit/toys/PheroEmmiter.java:				,Option.cuda.toString()
demos/turtlekit/toys/UFO.java:		Option.cuda.toString());
demos/turtlekit/toys/Runaway.java://				,Option.noCuda.toString()
demos/turtlekit/toys/Homogeneization.java:import turtlekit.cuda.CudaPheromone;
demos/turtlekit/toys/Homogeneization.java:		System.err.println(pheromone instanceof CudaPheromone);
demos/turtlekit/toys/Homogeneization.java:				,Option.cuda.toString()
demos/turtlekit/preypredator/Prey.java:	// ,Option.cuda.toString()
demos/turtlekit/preypredator/Predator.java:	// ,Option.cuda.toString()
src/turtlekit/kernel/TKScheduler.java:import jcuda.utils.Timer;
src/turtlekit/kernel/TKScheduler.java:	// if (isMadkitPropertyTrue(TurtleKit.Option.cuda)) {
src/turtlekit/kernel/TKScheduler.java:	// CudaEngine.stop();
src/turtlekit/kernel/TKScheduler.java:	// logger.fine("cuda freed");
src/turtlekit/kernel/TKScheduler.java:	// CudaEngine.stop();
src/turtlekit/kernel/TKLauncher.java:import static turtlekit.kernel.TurtleKit.Option.cuda;
src/turtlekit/kernel/TKLauncher.java:import turtlekit.cuda.CudaEngine;
src/turtlekit/kernel/TKLauncher.java:	if (isMadkitPropertyTrue(TurtleKit.Option.cuda) && !CudaEngine.init(getMadkitProperty(LevelOption.turtleKitLogLevel))) {
src/turtlekit/kernel/TKLauncher.java:	    setMadkitProperty(TurtleKit.Option.cuda, "false");
src/turtlekit/kernel/TKLauncher.java:	executeThisAgent(1, false, Option.configFile.toString(), "turtlekit/kernel/turtlekit.properties", cuda.toString(), turtles.toString(), Turtle.class.getName());
src/turtlekit/kernel/turtlekit.properties:cuda=false
src/turtlekit/kernel/TKEnvironment.java:import turtlekit.cuda.CudaEngine;
src/turtlekit/kernel/TKEnvironment.java:import turtlekit.cuda.CudaGPUGradientsPhero;
src/turtlekit/kernel/TKEnvironment.java:import turtlekit.cuda.CudaPheromone;
src/turtlekit/kernel/TKEnvironment.java:import turtlekit.cuda.CudaPheromoneWithBlock;
src/turtlekit/kernel/TKEnvironment.java:import turtlekit.cuda.GPUSobelGradientsPhero;
src/turtlekit/kernel/TKEnvironment.java:    private boolean cudaOn;
src/turtlekit/kernel/TKEnvironment.java:    protected boolean GPU_GRADIENTS = false;
src/turtlekit/kernel/TKEnvironment.java:    private boolean synchronizeGPU = true;
src/turtlekit/kernel/TKEnvironment.java:    public boolean isSynchronizeGPU() {
src/turtlekit/kernel/TKEnvironment.java:	return synchronizeGPU;
src/turtlekit/kernel/TKEnvironment.java:    public void setSynchronizeGPU(boolean synchronizeGPU) {
src/turtlekit/kernel/TKEnvironment.java:	this.synchronizeGPU = synchronizeGPU;
src/turtlekit/kernel/TKEnvironment.java:	GPU_GRADIENTS = Boolean.parseBoolean(getMadkitProperty("GPU_gradients"));
src/turtlekit/kernel/TKEnvironment.java:	cudaOn = isMadkitPropertyTrue(TurtleKit.Option.cuda);
src/turtlekit/kernel/TKEnvironment.java:	// logger.info("----------------------CUDA ON "+isCudaOn());
src/turtlekit/kernel/TKEnvironment.java:	// logger.info("----------------------GPU_GRADIENTS "+GPU_GRADIENTS);
src/turtlekit/kernel/TKEnvironment.java:	if (cudaOn) {
src/turtlekit/kernel/TKEnvironment.java:	    // Turtle.generator = new GPU_PRNG(12134); //TODO
src/turtlekit/kernel/TKEnvironment.java:	    if (i instanceof CudaPheromone) {
src/turtlekit/kernel/TKEnvironment.java:		((CudaPheromone) i).freeMemory();
src/turtlekit/kernel/TKEnvironment.java:	if (cudaOn) {
src/turtlekit/kernel/TKEnvironment.java:	    CudaEngine.cuCtxSynchronizeAll();
src/turtlekit/kernel/TKEnvironment.java:		if (isCudaOn() && synchronizeGPU) {
src/turtlekit/kernel/TKEnvironment.java:		    CudaEngine.cuCtxSynchronizeAll();
src/turtlekit/kernel/TKEnvironment.java:    public Pheromone<Float> getCudaPheromoneWithBlock(String name, int evaporationPercentage, int diffusionPercentage) {
src/turtlekit/kernel/TKEnvironment.java:		    phero = new CudaPheromoneWithBlock(name, width, height, evaporationPercentage / 100f, diffusionPercentage / 100f);
src/turtlekit/kernel/TKEnvironment.java:		    if (cudaOn && CudaEngine.isCudaAvailable()) {
src/turtlekit/kernel/TKEnvironment.java:			// phero = new CudaPheromone(name, width, height, evaporationPercentage, diffusionPercentage);
src/turtlekit/kernel/TKEnvironment.java:			phero = createCudaPheromone(name, evaporationPercentage, diffusionPercentage);
src/turtlekit/kernel/TKEnvironment.java:		    if (cudaOn && CudaEngine.isCudaAvailable()) {
src/turtlekit/kernel/TKEnvironment.java:			phero = new GPUSobelGradientsPhero(name, width, height, evaporationPercentage, diffusionPercentage);
src/turtlekit/kernel/TKEnvironment.java:    protected Pheromone<Float> createCudaPheromone(String name, float evaporationPercentage, float diffusionPercentage) {
src/turtlekit/kernel/TKEnvironment.java:	if (GPU_GRADIENTS)
src/turtlekit/kernel/TKEnvironment.java:	    return new CudaGPUGradientsPhero(name, getWidth(), getHeight(), evaporationPercentage, diffusionPercentage);
src/turtlekit/kernel/TKEnvironment.java:	return new CudaPheromone(name, getWidth(), getHeight(), evaporationPercentage, diffusionPercentage);
src/turtlekit/kernel/TKEnvironment.java:     * @return the cudaOn
src/turtlekit/kernel/TKEnvironment.java:    public boolean isCudaOn() {
src/turtlekit/kernel/TKEnvironment.java:	return cudaOn;
src/turtlekit/kernel/TKEnvironment.java:	this.synchronizeGPU = synchronizedEnvironment;
src/turtlekit/kernel/TurtleKit.java:		 * If set to <code>true</code>, TurtleKit will try to use Nvidia Cuda for 
src/turtlekit/kernel/TurtleKit.java:		 * This requires the Cuda toolkit to be installed on the host.
src/turtlekit/kernel/TurtleKit.java:		 * See {@link https://developer.nvidia.com/cuda-downloads} for more
src/turtlekit/kernel/TurtleKit.java:		cuda,
src/turtlekit/gui/CudaMenu.java:import static turtlekit.kernel.TurtleKit.Option.cuda;
src/turtlekit/gui/CudaMenu.java:public class CudaMenu extends JMenu {
src/turtlekit/gui/CudaMenu.java:	public CudaMenu(final TKEnvironment agent) {
src/turtlekit/gui/CudaMenu.java:		super("Cuda");
src/turtlekit/gui/CudaMenu.java:		final JCheckBoxMenuItem item = new JCheckBoxMenuItem("Synchronized CPU / GPU");
src/turtlekit/gui/CudaMenu.java:		setToolTipText("only available when Cuda is in use");
src/turtlekit/gui/CudaMenu.java:		setEnabled(agent.isMadkitPropertyTrue(cuda));
src/turtlekit/pheromone/AbstractPheromoneGrid.java:	 * Only one GPU kernel is called.
src/turtlekit/viewer/AbstractGridViewer.java:import turtlekit.gui.CudaMenu;
src/turtlekit/viewer/AbstractGridViewer.java:		jMenuBar.add(new CudaMenu(getCurrentEnvironment()),7);
src/turtlekit/cuda/GPUSobelGradientsPhero.java:package turtlekit.cuda;
src/turtlekit/cuda/GPUSobelGradientsPhero.java:public class GPUSobelGradientsPhero extends CudaPheromone {
src/turtlekit/cuda/GPUSobelGradientsPhero.java:    private CudaKernel sobel, sobel2;
src/turtlekit/cuda/GPUSobelGradientsPhero.java:    private CudaIntBuffer cudaIntBuffer;
src/turtlekit/cuda/GPUSobelGradientsPhero.java:    public GPUSobelGradientsPhero(String name, int width, int height, final float evapCoeff, final float diffCoeff) {
src/turtlekit/cuda/GPUSobelGradientsPhero.java://	sobel = getCudaKernel("SOBEL", "/turtlekit/cuda/kernels/SobelGradient_2D.cu", getKernelConfiguration());
src/turtlekit/cuda/GPUSobelGradientsPhero.java:	sobel2 = createKernel("DIFFUSION_UPDATE_AND_SOBEL_THEN_EVAPORATION", "/turtlekit/cuda/kernels/SobelGradient_2D.cu");
src/turtlekit/cuda/GPUSobelGradientsPhero.java:	cudaIntBuffer = new CudaIntBuffer(this);
src/turtlekit/cuda/GPUSobelGradientsPhero.java:		cudaIntBuffer.getDataPpointer());
src/turtlekit/cuda/GPUSobelGradientsPhero.java:	return cudaIntBuffer.get(get1DIndex(i, j));
src/turtlekit/cuda/GPUSobelGradientsPhero.java:	cudaIntBuffer.freeMemory();
src/turtlekit/cuda/GPU_PRNG.java:package turtlekit.cuda;
src/turtlekit/cuda/GPU_PRNG.java:import static jcuda.runtime.JCuda.cudaFree;
src/turtlekit/cuda/GPU_PRNG.java:import static jcuda.runtime.JCuda.cudaMalloc;
src/turtlekit/cuda/GPU_PRNG.java:import static jcuda.runtime.JCuda.cudaMemcpy;
src/turtlekit/cuda/GPU_PRNG.java:import jcuda.Pointer;
src/turtlekit/cuda/GPU_PRNG.java:import jcuda.Sizeof;
src/turtlekit/cuda/GPU_PRNG.java:import jcuda.jcurand.JCurand;
src/turtlekit/cuda/GPU_PRNG.java:import jcuda.jcurand.curandGenerator;
src/turtlekit/cuda/GPU_PRNG.java:import jcuda.jcurand.curandRngType;
src/turtlekit/cuda/GPU_PRNG.java:import jcuda.runtime.cudaMemcpyKind;
src/turtlekit/cuda/GPU_PRNG.java:public class GPU_PRNG extends Random {
src/turtlekit/cuda/GPU_PRNG.java:	public GPU_PRNG() {
src/turtlekit/cuda/GPU_PRNG.java:	public GPU_PRNG(long seed) {
src/turtlekit/cuda/GPU_PRNG.java://		CudaEngine cuda = CudaEngine.getCudaEngine(this);
src/turtlekit/cuda/GPU_PRNG.java:		cudaMalloc(deviceData, n * Sizeof.FLOAT);
src/turtlekit/cuda/GPU_PRNG.java:		cudaMemcpy(Pointer.to(hostData), deviceData, n * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost); 
src/turtlekit/cuda/GPU_PRNG.java:		cudaFree(deviceData);
src/turtlekit/cuda/GPU_PRNG.java:		GPU_PRNG name = new GPU_PRNG();
src/turtlekit/cuda/CudaUnifiedBuffer.java:package turtlekit.cuda;
src/turtlekit/cuda/CudaUnifiedBuffer.java:import jcuda.Pointer;
src/turtlekit/cuda/CudaUnifiedBuffer.java:import jcuda.driver.CUdeviceptr;
src/turtlekit/cuda/CudaUnifiedBuffer.java:public abstract class CudaUnifiedBuffer {
src/turtlekit/cuda/CudaUnifiedBuffer.java:    private CudaObject cudaObject;
src/turtlekit/cuda/CudaUnifiedBuffer.java:    public CudaUnifiedBuffer(CudaObject co) {
src/turtlekit/cuda/CudaUnifiedBuffer.java:	this.cudaObject = co;
src/turtlekit/cuda/CudaUnifiedBuffer.java:	cudaObject.getCudaEngine().freeCudaMemory(pinnedMemory);
src/turtlekit/cuda/CudaUnifiedBuffer.java:	cudaObject.getCudaEngine().freeCudaMemory(devicePtr);
src/turtlekit/cuda/CudaUnifiedBuffer.java:     * @return the cudaObject
src/turtlekit/cuda/CudaUnifiedBuffer.java:    public CudaObject getCudaObject() {
src/turtlekit/cuda/CudaUnifiedBuffer.java:        return cudaObject;
src/turtlekit/cuda/CudaGPUGradientsPhero.java:package turtlekit.cuda;
src/turtlekit/cuda/CudaGPUGradientsPhero.java:import jcuda.Pointer;
src/turtlekit/cuda/CudaGPUGradientsPhero.java:import jcuda.driver.CUdeviceptr;
src/turtlekit/cuda/CudaGPUGradientsPhero.java:public class CudaGPUGradientsPhero extends CudaPheromone{
src/turtlekit/cuda/CudaGPUGradientsPhero.java:	private CudaKernel diffusionUpdateAndEvaporationAndFieldMaxDirKernel;
src/turtlekit/cuda/CudaGPUGradientsPhero.java:	public CudaGPUGradientsPhero(String name, int width, int height, final float evapCoeff, final float diffCoeff) {
src/turtlekit/cuda/CudaGPUGradientsPhero.java:		diffusionUpdateAndEvaporationAndFieldMaxDirKernel = createKernel("DIFFUSION_UPDATE_THEN_EVAPORATION_THEN_FIELDMAXDIRV2", "/turtlekit/cuda/kernels/DiffusionEvaporationGradients_2D.cu");
src/turtlekit/cuda/CudaGPUGradientsPhero.java:	 * Only one GPU kernel is called.
src/turtlekit/cuda/CudaGPUGradientsPhero.java://			cuda.submit(fieldMinDirComputation);
src/turtlekit/cuda/CudaGPUGradientsPhero.java:		freeCudaMemory(maxPinnedMemory);
src/turtlekit/cuda/CudaGPUGradientsPhero.java:		freeCudaMemory(fieldMaxDirPtr);
src/turtlekit/cuda/CudaKernel.java:package turtlekit.cuda;
src/turtlekit/cuda/CudaKernel.java:import jcuda.Pointer;
src/turtlekit/cuda/CudaKernel.java:import jcuda.driver.CUfunction;
src/turtlekit/cuda/CudaKernel.java:import jcuda.driver.JCudaDriver;
src/turtlekit/cuda/CudaKernel.java:public class CudaKernel {
src/turtlekit/cuda/CudaKernel.java:	private CudaEngine myCe;
src/turtlekit/cuda/CudaKernel.java:	CudaKernel(CUfunction function, CudaEngine ce, String kernelFunctionName, String dotCuSourceFilePath, KernelConfiguration kc) {
src/turtlekit/cuda/CudaKernel.java:				JCudaDriver.cuLaunchKernel(myFonction, //TODO cach
src/turtlekit/cuda/CudaKernel.java://		myJob = () -> JCudaDriver.cuLaunchKernel(myFonction, 
src/turtlekit/cuda/CudaKernel.java:			final Path path = Paths.get(CudaEngine.ioTmpDir, f.getName());
src/turtlekit/cuda/CudaKernel.java://		try(InputStream is = CudaEngine.class.getResourceAsStreéam(dotCuSourceFilePath)){
src/turtlekit/cuda/CudaKernel.java:		CudaObject name = new CudaObject() {
src/turtlekit/cuda/CudaKernel.java:		CudaEngine.init(Level.ALL.toString());
src/turtlekit/cuda/CudaKernel.java:		CudaEngine ce = CudaEngine.getCudaEngine(name);
src/turtlekit/cuda/kernels/package-info.java:package turtlekit.cuda.kernels;
src/turtlekit/cuda/CudaObject.java:package turtlekit.cuda;
src/turtlekit/cuda/CudaObject.java:import jcuda.Pointer;
src/turtlekit/cuda/CudaObject.java:import jcuda.driver.CUdeviceptr;
src/turtlekit/cuda/CudaObject.java:public interface CudaObject {
src/turtlekit/cuda/CudaObject.java:    default CudaEngine getCudaEngine() {
src/turtlekit/cuda/CudaObject.java:	return CudaEngine.getCudaEngine(this);
src/turtlekit/cuda/CudaObject.java:     * Shortcut for <code>getCudaEngine().createNewKernelConfiguration(getWidth(), getHeight())</code>
src/turtlekit/cuda/CudaObject.java:     * @return a new kernel configuration according to the CudaObject dimensions
src/turtlekit/cuda/CudaObject.java:	return getCudaEngine().createNewKernelConfiguration(getWidth(), getHeight());
src/turtlekit/cuda/CudaObject.java:     * see {@link CudaEngine#createNewKernelConfiguration(int, int)} for creating a default one.
src/turtlekit/cuda/CudaObject.java:     * @return the default kernel configuration of this {@link CudaObject}
src/turtlekit/cuda/CudaObject.java:     * <code>return getCudaEngine().createKernel(kernelFunctionName, cuSourceFilePath, getKernelConfiguration())</code>
src/turtlekit/cuda/CudaObject.java:    default CudaKernel createKernel(final String kernelFunctionName, final String cuSourceFilePath) {
src/turtlekit/cuda/CudaObject.java:	return getCudaEngine().createKernel(kernelFunctionName, cuSourceFilePath, getKernelConfiguration());
src/turtlekit/cuda/CudaObject.java:	return getCudaEngine().createDeviceDataGrid(getWidth(), getHeight(), dataType);
src/turtlekit/cuda/CudaObject.java:	return getCudaEngine().getUnifiedBufferBetweenPointer(hostData, deviceData, dataType, getWidth(), getHeight());
src/turtlekit/cuda/CudaObject.java:    default void freeCudaMemory(Pointer p) {
src/turtlekit/cuda/CudaObject.java:	getCudaEngine().freeCudaMemory(p);
src/turtlekit/cuda/CudaPheromoneWithBlock.java:package turtlekit.cuda;
src/turtlekit/cuda/CudaPheromoneWithBlock.java:public class CudaPheromoneWithBlock extends CudaPheromone{
src/turtlekit/cuda/CudaPheromoneWithBlock.java:	public CudaPheromoneWithBlock(String name, int width, int height, float evapPercentage, float diffPercentage) {
src/turtlekit/cuda/CudaPheromoneWithBlock.java:	public CudaPheromoneWithBlock(String name, int width, int height, final int evapPercentage, final int diffPercentage) {
src/turtlekit/cuda/CudaPheromoneWithBlock.java://		kernelConfiguration.setStreamID(cudaEngine.getNewCudaStream());
src/turtlekit/cuda/CudaPheromoneWithBlock.java:		diffusionToTmpKernel = createKernel("DIFFUSION_TO_TMP", "/turtlekit/cuda/kernels/DiffusionEvaporationWithBlock_2D.cu");
src/turtlekit/cuda/CudaPheromoneWithBlock.java:		diffusionUpdateKernel = createKernel("DIFFUSION_UPDATE", "/turtlekit/cuda/kernels/DiffusionEvaporationWithBlock_2D.cu");
src/turtlekit/cuda/CudaPheromoneWithBlock.java:		diffusionUpdateThenEvaporationKernel = createKernel("DIFFUSION_UPDATE_THEN_EVAPORATION", "/turtlekit/cuda/kernels/DiffusionEvaporationWithBlock_2D.cu");
src/turtlekit/cuda/CudaPheromoneWithBlock.java:		evaporationKernel = createKernel("EVAPORATION", "/turtlekit/cuda/kernels/Evaporation_2D.cu");
src/turtlekit/cuda/CudaIntBuffer.java:package turtlekit.cuda;
src/turtlekit/cuda/CudaIntBuffer.java:public class CudaIntBuffer extends CudaUnifiedBuffer {
src/turtlekit/cuda/CudaIntBuffer.java:    public CudaIntBuffer(CudaObject co) {
src/turtlekit/cuda/CudaIntBuffer.java:	values = (IntBuffer) co.getCudaEngine().getUnifiedBufferBetweenPointer(getPinnedMemory(), getDevicePtr(), Integer.class, co.getWidth(), co.getHeight());
src/turtlekit/cuda/CudaAverageField.java:package turtlekit.cuda;
src/turtlekit/cuda/CudaAverageField.java:import jcuda.Pointer;
src/turtlekit/cuda/CudaAverageField.java:import jcuda.driver.CUdeviceptr;
src/turtlekit/cuda/CudaAverageField.java:public class CudaAverageField extends DataGrid<Integer> implements CudaObject {
src/turtlekit/cuda/CudaAverageField.java:	private CudaIntBuffer values;
src/turtlekit/cuda/CudaAverageField.java:	CudaKernel averageComputation;
src/turtlekit/cuda/CudaAverageField.java:	CudaKernel computeAverageToTmpGrid;
src/turtlekit/cuda/CudaAverageField.java:	public CudaAverageField(String name, int width, int height, int depth) {
src/turtlekit/cuda/CudaAverageField.java:		values = new CudaIntBuffer(this);
src/turtlekit/cuda/CudaAverageField.java:		averageComputation = createKernel("AVERAGE_DEPTH_1D_V2", "/turtlekit/cuda/kernels/Average_2D.cu");
src/turtlekit/cuda/CudaFloatBuffer.java:package turtlekit.cuda;
src/turtlekit/cuda/CudaFloatBuffer.java:public class CudaFloatBuffer extends CudaUnifiedBuffer {
src/turtlekit/cuda/CudaFloatBuffer.java:    public CudaFloatBuffer(CudaObject co) {
src/turtlekit/cuda/CudaFloatBuffer.java:	values = (FloatBuffer) co.getCudaEngine().getUnifiedBufferBetweenPointer(getPinnedMemory(), getDevicePtr(), Float.class, co.getWidth(), co.getHeight());
src/turtlekit/cuda/CudaPheromone.java:package turtlekit.cuda;
src/turtlekit/cuda/CudaPheromone.java:import jcuda.Pointer;
src/turtlekit/cuda/CudaPheromone.java:import jcuda.driver.CUdeviceptr;
src/turtlekit/cuda/CudaPheromone.java:public class CudaPheromone extends AbstractPheromoneGrid<Float> implements CudaObject,Pheromone<Float>{
src/turtlekit/cuda/CudaPheromone.java:	private CudaFloatBuffer values;
src/turtlekit/cuda/CudaPheromone.java:	protected CudaKernel diffusionToTmpKernel;
src/turtlekit/cuda/CudaPheromone.java:	protected CudaKernel diffusionUpdateKernel;
src/turtlekit/cuda/CudaPheromone.java:	protected CudaKernel diffusionUpdateThenEvaporationKernel;
src/turtlekit/cuda/CudaPheromone.java:	protected CudaKernel evaporationKernel;
src/turtlekit/cuda/CudaPheromone.java:	public CudaPheromone(String name, int width, int height, final int evapPercentage,
src/turtlekit/cuda/CudaPheromone.java:	public CudaPheromone(String name, int width, int height, final float evapPercentage,
src/turtlekit/cuda/CudaPheromone.java:		values = new CudaFloatBuffer(this);
src/turtlekit/cuda/CudaPheromone.java:		diffusionToTmpKernel = createKernel("DIFFUSION_TO_TMP", "/turtlekit/cuda/kernels/Diffusion_2D.cu");
src/turtlekit/cuda/CudaPheromone.java:		diffusionUpdateKernel = createKernel("DIFFUSION_UPDATE", "/turtlekit/cuda/kernels/Diffusion_2D.cu");
src/turtlekit/cuda/CudaPheromone.java:		diffusionUpdateThenEvaporationKernel = createKernel("DIFFUSION_UPDATE_THEN_EVAPORATION", "/turtlekit/cuda/kernels/DiffusionEvaporation_2D.cu");
src/turtlekit/cuda/CudaPheromone.java:		evaporationKernel = createKernel("EVAPORATION", "/turtlekit/cuda/kernels/Evaporation_2D.cu");
src/turtlekit/cuda/CudaPheromone.java:		freeCudaMemory(tmpDeviceDataGrid);
src/turtlekit/cuda/CudaPheromone.java:	public CudaFloatBuffer getValues() {
src/turtlekit/cuda/KernelConfiguration.java:package turtlekit.cuda;
src/turtlekit/cuda/KernelConfiguration.java:import jcuda.driver.CUstream;
src/turtlekit/cuda/KernelConfiguration.java: * A kernel configuration defines the dimensions of the cuda grid and blocks to be used, as well as a stream ID
src/turtlekit/cuda/CudaEngine.java:package turtlekit.cuda;
src/turtlekit/cuda/CudaEngine.java:import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
src/turtlekit/cuda/CudaEngine.java:import static jcuda.driver.JCudaDriver.cuMemAlloc;
src/turtlekit/cuda/CudaEngine.java:import static jcuda.driver.JCudaDriver.cuMemFree;
src/turtlekit/cuda/CudaEngine.java:import static jcuda.driver.JCudaDriver.cuMemFreeHost;
src/turtlekit/cuda/CudaEngine.java:import jcuda.CudaException;
src/turtlekit/cuda/CudaEngine.java:import jcuda.Pointer;
src/turtlekit/cuda/CudaEngine.java:import jcuda.driver.CUcontext;
src/turtlekit/cuda/CudaEngine.java:import jcuda.driver.CUdevice;
src/turtlekit/cuda/CudaEngine.java:import jcuda.driver.CUdeviceptr;
src/turtlekit/cuda/CudaEngine.java:import jcuda.driver.CUfunction;
src/turtlekit/cuda/CudaEngine.java:import jcuda.driver.CUmodule;
src/turtlekit/cuda/CudaEngine.java:import jcuda.driver.CUstream;
src/turtlekit/cuda/CudaEngine.java:import jcuda.driver.CUstream_flags;
src/turtlekit/cuda/CudaEngine.java:import jcuda.driver.JCudaDriver;
src/turtlekit/cuda/CudaEngine.java:import jcuda.utils.KernelLauncher;
src/turtlekit/cuda/CudaEngine.java:public class CudaEngine {
src/turtlekit/cuda/CudaEngine.java:    final static Map<Integer, CudaEngine> cudaEngines = new HashMap<>();
src/turtlekit/cuda/CudaEngine.java:	logger = Logger.getLogger(CudaEngine.class.getSimpleName());
src/turtlekit/cuda/CudaEngine.java:	synchronized (cudaEngines) {
src/turtlekit/cuda/CudaEngine.java:	    logger.finer("---------Initializing Cuda----------------");
src/turtlekit/cuda/CudaEngine.java:		JCudaDriver.setExceptionsEnabled(true);
src/turtlekit/cuda/CudaEngine.java:		JCudaDriver.cuInit(0);
src/turtlekit/cuda/CudaEngine.java:		JCudaDriver.cuDeviceGetCount(deviceCountArray);
src/turtlekit/cuda/CudaEngine.java:		logger.finer("Found " + availableDevicesNb + " GPU devices");
src/turtlekit/cuda/CudaEngine.java:			    cudaEngines.put(index, new CudaEngine(index));
src/turtlekit/cuda/CudaEngine.java:	    catch(InterruptedException | ExecutionException | CudaException | UnsatisfiedLinkError e) {
src/turtlekit/cuda/CudaEngine.java:		logger.log(Level.FINER, e, () -> "---------Cannot initialize Cuda !!! ----------------");
src/turtlekit/cuda/CudaEngine.java:		    CudaEngine.stop();
src/turtlekit/cuda/CudaEngine.java:	    logger.fine("---------Cuda Initialized----------------");
src/turtlekit/cuda/CudaEngine.java:    private static AtomicInteger cudaObjectID = new AtomicInteger(0);
src/turtlekit/cuda/CudaEngine.java:    private static Map<CudaObject, CudaEngine> engineBinds = new HashMap<>();
src/turtlekit/cuda/CudaEngine.java:    private List<CudaObject> cudaObjects = new ArrayList<CudaObject>();
src/turtlekit/cuda/CudaEngine.java:    private CudaEngine(final int deviceId) {
src/turtlekit/cuda/CudaEngine.java:	}); // mandatory: Only one cuda thread per context
src/turtlekit/cuda/CudaEngine.java:		    JCudaDriver.cuDeviceGet(device, deviceId);
src/turtlekit/cuda/CudaEngine.java:		    JCudaDriver.cuDeviceGetAttribute(array, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
src/turtlekit/cuda/CudaEngine.java:		    JCudaDriver.cuCtxCreate(context, 0, device);
src/turtlekit/cuda/CudaEngine.java:    public static boolean isCudaAvailable() {
src/turtlekit/cuda/CudaEngine.java:	return "cudaEngine device #" + Id;
src/turtlekit/cuda/CudaEngine.java:     * Creates a new kernel configuration with a new Cuda stream ID according to 2D data size.
src/turtlekit/cuda/CudaEngine.java:	return new KernelConfiguration(gridSizeX, gridSizeY, maxThreads, maxThreads, getNewCudaStream());
src/turtlekit/cuda/CudaEngine.java:     * Creates a new Cuda kernel using a configuration and a kernel name 
src/turtlekit/cuda/CudaEngine.java:     * which could be found in a Cuda source file.
src/turtlekit/cuda/CudaEngine.java:     * @return a new Cuda Kernel
src/turtlekit/cuda/CudaEngine.java:    public CudaKernel createKernel(final String kernelFunctionName, final String cuSourceFilePath, final KernelConfiguration kc) {
src/turtlekit/cuda/CudaEngine.java:		return new CudaKernel(function, CudaEngine.this, kernelFunctionName, cuSourceFilePath, kc);
src/turtlekit/cuda/CudaEngine.java:		JCudaDriver.cuMemHostAlloc(hostData, size, JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP);
src/turtlekit/cuda/CudaEngine.java:		JCudaDriver.cuMemHostGetDevicePointer(deviceData, hostData, 0);
src/turtlekit/cuda/CudaEngine.java:    public static CudaEngine getCudaEngine(CudaObject co) {
src/turtlekit/cuda/CudaEngine.java:	synchronized (cudaEngines) {
src/turtlekit/cuda/CudaEngine.java:	    if (!isCudaAvailable())
src/turtlekit/cuda/CudaEngine.java:		throw new CudaException("No cuda device found");
src/turtlekit/cuda/CudaEngine.java:		    final int pheroID = cudaObjectID.incrementAndGet();
src/turtlekit/cuda/CudaEngine.java:		    final CudaEngine ce = cudaEngines.get(pheroID % availableDevicesNb);
src/turtlekit/cuda/CudaEngine.java:		    // final CudaEngine ce = cudaEngines.get(0);
src/turtlekit/cuda/CudaEngine.java:		    ce.cudaObjects.add(co);
src/turtlekit/cuda/CudaEngine.java:		    logger.finer(co + "ID " + pheroID + " getting cuda engine Id " + ce.Id);
src/turtlekit/cuda/CudaEngine.java://	JCudaDriver.cuMemHostAlloc(pinnedMemory, size, JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP);
src/turtlekit/cuda/CudaEngine.java://	JCudaDriver.cuMemHostGetDevicePointer(devicePtr, pinnedMemory, 0);
src/turtlekit/cuda/CudaEngine.java://    public static CudaIntBuffer getUnifiedIntBuffer(Pointer pinnedMemory, CUdeviceptr devicePtr, int size) {
src/turtlekit/cuda/CudaEngine.java://	JCudaDriver.cuMemHostAlloc(pinnedMemory, size, JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP);
src/turtlekit/cuda/CudaEngine.java://	JCudaDriver.cuMemHostGetDevicePointer(devicePtr, pinnedMemory, 0);
src/turtlekit/cuda/CudaEngine.java://	JCudaDriver.cuMemHostAlloc(pinnedMemory, size, JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP);
src/turtlekit/cuda/CudaEngine.java://	JCudaDriver.cuMemHostGetDevicePointer(devicePtr, pinnedMemory, 0);
src/turtlekit/cuda/CudaEngine.java://	JCudaDriver.cuMemHostAlloc(pinnedMemory, size, JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP);
src/turtlekit/cuda/CudaEngine.java://	JCudaDriver.cuMemHostGetDevicePointer(devicePtr, pinnedMemory, 0);
src/turtlekit/cuda/CudaEngine.java:	synchronized (cudaEngines) {
src/turtlekit/cuda/CudaEngine.java:	    for (Iterator<CudaEngine> iterator = cudaEngines.values().iterator(); iterator.hasNext();) {
src/turtlekit/cuda/CudaEngine.java:	    // for (CudaEngine ce : cudaEngines.values()) {
src/turtlekit/cuda/CudaEngine.java:	for (CudaEngine ce : cudaEngines.values()) {
src/turtlekit/cuda/CudaEngine.java:		for (CudaObject co : cudaObjects) {
src/turtlekit/cuda/CudaEngine.java:		JCudaDriver.cuCtxDestroy(context);
src/turtlekit/cuda/CudaEngine.java:	    System.err.println("cuda device " + Id + " freed ? " + exe.awaitTermination(10, TimeUnit.SECONDS));
src/turtlekit/cuda/CudaEngine.java:    public CUstream getNewCudaStream() {
src/turtlekit/cuda/CudaEngine.java:		final CUstream cudaStream = new CUstream();
src/turtlekit/cuda/CudaEngine.java:		JCudaDriver.cuStreamCreate(cudaStream, CUstream_flags.CU_STREAM_NON_BLOCKING);
src/turtlekit/cuda/CudaEngine.java:		return cudaStream;
src/turtlekit/cuda/CudaEngine.java:	KernelLauncher.setCompilerPath("/opt/cuda/bin/");// FIXME
src/turtlekit/cuda/CudaEngine.java:	try (final InputStream is = CudaEngine.class.getResourceAsStream(dotCuSourceFilePath)) {
src/turtlekit/cuda/CudaEngine.java:	    final URL resource = CudaEngine.class.getResource(dotCuSourceFilePath);
src/turtlekit/cuda/CudaEngine.java:	    final Path path = Paths.get(CudaEngine.ioTmpDir, fileName);
src/turtlekit/cuda/CudaEngine.java:	    KernelLauncher.create(cuFile, kernelFunctionName, rebuildNeeded, "--use_fast_math", "--prec-div=false");// ,"--gpu-architecture=sm_20");
src/turtlekit/cuda/CudaEngine.java:	    JCudaDriver.cuModuleLoad(myModule, cuFile.substring(0, cuFile.lastIndexOf('.')) + ".ptx");
src/turtlekit/cuda/CudaEngine.java:	    JCudaDriver.cuModuleGetFunction(function, myModule, kernelFunctionName);
src/turtlekit/cuda/CudaEngine.java:	// try(InputStream is = CudaEngine.class.getResourceAsStreéam(dotCuSourceFilePath)){
src/turtlekit/cuda/CudaEngine.java:	for (CudaEngine ce : cudaEngines.values()) {
src/turtlekit/cuda/CudaEngine.java:	    exe.submit(() -> JCudaDriver.cuCtxSynchronize()).get();
src/turtlekit/cuda/CudaEngine.java:    public void freeCudaMemory(Pointer p) {
src/turtlekit/cuda/CudaEngine.java:    public void freeCudaMemory(CUdeviceptr p) {
src/turtlekit/cuda/CudaEngine.java:     * Implements a little test that instantiates the CudaEngine and then cleans up
src/turtlekit/cuda/CudaEngine.java:	CudaEngine cudaEngine = new CudaEngine(0);
src/turtlekit/cuda/CudaEngine.java:	// KernelConfiguration kernelConfiguration = cudaEngine.getDefaultKernelConfiguration(100, 100);
src/turtlekit/cuda/CudaEngine.java:	// cudaEngine.getKernel("EVAPORATION", "/turtlekit/cuda/kernels/Evaporation_2D.cu", kernelConfiguration);

```
