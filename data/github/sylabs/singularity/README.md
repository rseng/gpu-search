# https://github.com/sylabs/singularity

```console
internal/pkg/util/gpu/nvidia_test.go:package gpu
internal/pkg/util/gpu/nvidia_test.go:				"NVIDIA_VISIBLE_DEVICES=all",
internal/pkg/util/gpu/nvidia_test.go:				"NVIDIA_MIG_CONFIG_DEVICES=all",
internal/pkg/util/gpu/nvidia_test.go:				"NVIDIA_MIG_MONITOR_DEVICES=all",
internal/pkg/util/gpu/nvidia_test.go:				"NVIDIA_DRIVER_CAPABILITIES=compute",
internal/pkg/util/gpu/nvidia_test.go:				"NVIDIA_DRIVER_CAPABILITIES=compute,compat32,graphics,utility,video,display",
internal/pkg/util/gpu/nvidia_test.go:				"NVIDIA_DRIVER_CAPABILITIES=notacap",
internal/pkg/util/gpu/nvidia_test.go:				"NVIDIA_REQUIRE_CUDA=cuda>=9.0",
internal/pkg/util/gpu/nvidia_test.go:				"--require=cuda>=9.0",
internal/pkg/util/gpu/nvidia_test.go:				"NVIDIA_REQUIRE_BRAND=brand=GRID",
internal/pkg/util/gpu/nvidia_test.go:				"NVIDIA_REQUIRE_CUDA=cuda>=9.0",
internal/pkg/util/gpu/nvidia_test.go:				"--require=cuda>=9.0",
internal/pkg/util/gpu/nvidia_test.go:				"NVIDIA_REQUIRE_BRAND=brand=GRID",
internal/pkg/util/gpu/nvidia_test.go:				"NVIDIA_REQUIRE_CUDA=cuda>=9.0",
internal/pkg/util/gpu/nvidia_test.go:				"NVIDIA_DISABLE_REQUIRE=1",
internal/pkg/util/gpu/paths_test.go:package gpu
internal/pkg/util/gpu/paths_test.go:func Test_gpuliblist(t *testing.T) {
internal/pkg/util/gpu/paths_test.go:	gotLibs, err := gpuliblist(testLibFile)
internal/pkg/util/gpu/paths_test.go:		t.Errorf("gpuliblist() error = %v", err)
internal/pkg/util/gpu/paths_test.go:		t.Error("gpuliblist() gave no results")
internal/pkg/util/gpu/paths_test.go:		t.Errorf("gpuliblist() gave unexpected results, got: %v expected: %v", gotLibs, testLibList)
internal/pkg/util/gpu/paths.go:package gpu
internal/pkg/util/gpu/paths.go:// gpuliblist returns libraries/binaries listed in a gpu lib list config file, typically
internal/pkg/util/gpu/paths.go:func gpuliblist(configFilePath string) ([]string, error) {
internal/pkg/util/gpu/paths.go:func paths(gpuFileList []string) ([]string, []string, error) {
internal/pkg/util/gpu/paths.go:	for _, file := range gpuFileList {
internal/pkg/util/gpu/paths.go:	// returned by nvidia-container-cli OR the nvliblist.conf file contents
internal/pkg/util/gpu/paths.go:	// libnvidia-ml.so.1 (libc6,x86-64) => /usr/lib64/nvidia/libnvidia-ml.so.1
internal/pkg/util/gpu/paths.go:			// libName is the "libnvidia-ml.so.1" (from the above example)
internal/pkg/util/gpu/paths.go:			// libPath is the "/usr/lib64/nvidia/libnvidia-ml.so.1" (from the above example)
internal/pkg/util/gpu/rocm.go:package gpu
internal/pkg/util/gpu/rocm.go:// RocmPaths returns a list of rocm libraries/binaries that should be
internal/pkg/util/gpu/rocm.go:// mounted into the container in order to use AMD GPUs
internal/pkg/util/gpu/rocm.go:func RocmPaths(configFilePath string) ([]string, []string, error) {
internal/pkg/util/gpu/rocm.go:	rocmFiles, err := gpuliblist(configFilePath)
internal/pkg/util/gpu/rocm.go:	return paths(rocmFiles)
internal/pkg/util/gpu/rocm.go:// RocmDevices returns a list of /dev entries required for ROCm functionality.
internal/pkg/util/gpu/rocm.go:func RocmDevices() ([]string, error) {
internal/pkg/util/gpu/rocm.go:	// Use same paths as ROCm Docker container documentation.
internal/pkg/util/gpu/rocm.go:	// Must bind in all GPU DRI devices, and /dev/kfd device.
internal/pkg/util/gpu/nvidia_legacy.go:package gpu
internal/pkg/util/gpu/nvidia_legacy.go:// NvidiaPaths returns a list of Nvidia libraries/binaries that should be
internal/pkg/util/gpu/nvidia_legacy.go:// mounted into the container in order to use Nvidia GPUs
internal/pkg/util/gpu/nvidia_legacy.go:func NvidiaPaths(configFilePath string) ([]string, []string, error) {
internal/pkg/util/gpu/nvidia_legacy.go:	nvidiaFiles, err := gpuliblist(configFilePath)
internal/pkg/util/gpu/nvidia_legacy.go:	return paths(nvidiaFiles)
internal/pkg/util/gpu/nvidia_legacy.go:// NvidiaIpcsPath returns a list of nvidia driver ipcs.
internal/pkg/util/gpu/nvidia_legacy.go:func NvidiaIpcsPath() ([]string, error) {
internal/pkg/util/gpu/nvidia_legacy.go:	const persistencedSocket = "/var/run/nvidia-persistenced/socket"
internal/pkg/util/gpu/nvidia_legacy.go:	var nvidiaFiles []string
internal/pkg/util/gpu/nvidia_legacy.go:	nvidiaFiles = append(nvidiaFiles, persistencedSocket)
internal/pkg/util/gpu/nvidia_legacy.go:	return nvidiaFiles, nil
internal/pkg/util/gpu/nvidia_legacy.go:// NvidiaDevices return list of all non-GPU nvidia devices present on host. If withGPU
internal/pkg/util/gpu/nvidia_legacy.go:// is true all GPUs are included in the resulting list as well.
internal/pkg/util/gpu/nvidia_legacy.go:func NvidiaDevices(withGPU bool) ([]string, error) {
internal/pkg/util/gpu/nvidia_legacy.go:	nvidiaGlob := "/dev/nvidia*"
internal/pkg/util/gpu/nvidia_legacy.go:	if !withGPU {
internal/pkg/util/gpu/nvidia_legacy.go:		nvidiaGlob = "/dev/nvidia[^0-9]*"
internal/pkg/util/gpu/nvidia_legacy.go:	devs, err := filepath.Glob(nvidiaGlob)
internal/pkg/util/gpu/nvidia_legacy.go:		return nil, fmt.Errorf("could not list nvidia devices: %v", err)
internal/pkg/util/gpu/nvidia.go:package gpu
internal/pkg/util/gpu/nvidia.go:	errNvCCLIInsecure   = errors.New("nvidia-container-cli is not owned by root user")
internal/pkg/util/gpu/nvidia.go:// nVDriverCapabilities is the set of driver capabilities supported by nvidia-container-cli.
internal/pkg/util/gpu/nvidia.go:// See: https://github.com/nvidia/nvidia-container-runtime#nvidia_driver_capabilities
internal/pkg/util/gpu/nvidia.go:// nVDriverDefaultCapabilities is the default set of nvidia-container-cli driver capabilities.
internal/pkg/util/gpu/nvidia.go:// It is used if NVIDIA_DRIVER_CAPABILITIES is not set.
internal/pkg/util/gpu/nvidia.go:// See: https://github.com/nvidia/nvidia-container-runtime#nvidia_driver_capabilities
internal/pkg/util/gpu/nvidia.go:// nVCLIAmbientCaps is the ambient capability set required by nvidia-container-cli.
internal/pkg/util/gpu/nvidia.go:// NVCLIConfigure calls out to the nvidia-container-cli configure operation.
internal/pkg/util/gpu/nvidia.go:// This sets up the GPU with the container. Note that the ability to set a
internal/pkg/util/gpu/nvidia.go:// error if the bounding set does not include NvidiaContainerCLIAmbientCaps.
internal/pkg/util/gpu/nvidia.go:// any privilege escalation when calling out to `nvidia-container-cli`.
internal/pkg/util/gpu/nvidia.go:// exec `nvidia-container-cli` as root via SysProcAttr, having first ensured
internal/pkg/util/gpu/nvidia.go:func NVCLIConfigure(nvidiaEnv []string, rootfs string, userNS bool) error {
internal/pkg/util/gpu/nvidia.go:	nvCCLIPath, err := bin.FindBin("nvidia-container-cli")
internal/pkg/util/gpu/nvidia.go:	// If we will run nvidia-container-cli as the host root user then ensure
internal/pkg/util/gpu/nvidia.go:	// Translate the passed in NVIDIA_ env vars to option flags
internal/pkg/util/gpu/nvidia.go:	flags, err := NVCLIEnvToFlags(nvidiaEnv)
internal/pkg/util/gpu/nvidia.go:	// If we will run nvidia-container-cli as the host root user then ensure
internal/pkg/util/gpu/nvidia.go:	// or nvidia-container-cli will fail.
internal/pkg/util/gpu/nvidia.go:	sylog.Debugf("nvidia-container-cli binary: %q args: %q", nvCCLIPath, nccArgs)
internal/pkg/util/gpu/nvidia.go:	// nvidia-container-cli requires a default sensible PATH to work correctly.
internal/pkg/util/gpu/nvidia.go:	// We need to run nvidia-container-cli as host root when there is no user
internal/pkg/util/gpu/nvidia.go:		sylog.Debugf("Running nvidia-container-cli as uid=0 gid=0")
internal/pkg/util/gpu/nvidia.go:		sylog.Debugf("Running nvidia-container-cli in user namespace")
internal/pkg/util/gpu/nvidia.go:		return fmt.Errorf("nvidia-container-cli failed with %v: %s", err, stdoutStderr)
internal/pkg/util/gpu/nvidia.go:// NVCLIEnvToFlags reads the passed in NVIDIA_ environment variables supported
internal/pkg/util/gpu/nvidia.go:// by nvidia-container-runtime and converts them to flags for
internal/pkg/util/gpu/nvidia.go:// nvidia-container-cli. See:
internal/pkg/util/gpu/nvidia.go:// https://github.com/nvidia/nvidia-container-runtime#environment-variables-oci-spec
internal/pkg/util/gpu/nvidia.go:func NVCLIEnvToFlags(nvidiaEnv []string) (flags []string, err error) {
internal/pkg/util/gpu/nvidia.go:	for _, e := range nvidiaEnv {
internal/pkg/util/gpu/nvidia.go:		if pair[0] == "NVIDIA_VISIBLE_DEVICES" && pair[1] != "" {
internal/pkg/util/gpu/nvidia.go:		if pair[0] == "NVIDIA_MIG_CONFIG_DEVICES" && pair[1] != "" {
internal/pkg/util/gpu/nvidia.go:		if pair[0] == "NVIDIA_MIG_MONITOR_DEVICES" && pair[1] != "" {
internal/pkg/util/gpu/nvidia.go:		if pair[0] == "NVIDIA_DRIVER_CAPABILITIES" && pair[1] != "" {
internal/pkg/util/gpu/nvidia.go:					return nil, fmt.Errorf("unknown NVIDIA_DRIVER_CAPABILITIES value: %s", cap)
internal/pkg/util/gpu/nvidia.go:		// One --require flag for each NVIDIA_REQUIRE_* environment
internal/pkg/util/gpu/nvidia.go:		// https://github.com/nvidia/nvidia-container-runtime#nvidia_require_
internal/pkg/util/gpu/nvidia.go:		if strings.HasPrefix(pair[0], "NVIDIA_REQUIRE_") {
internal/pkg/util/gpu/nvidia.go:		if pair[0] == "NVIDIA_DISABLE_REQUIRE" {
internal/pkg/util/bin/bin.go:	// cryptsetup & nvidia-container-cli paths must be explicitly specified
internal/pkg/util/bin/bin.go:	// ldconfig is invoked by nvidia-container-cli, so must be trusted also.
internal/pkg/util/bin/bin.go:	case "cryptsetup", "ldconfig", "nvidia-container-cli":
internal/pkg/util/bin/bin_test.go:			name:          "nvidia-container-cli valid",
internal/pkg/util/bin/bin_test.go:			bin:           "nvidia-container-cli",
internal/pkg/util/bin/bin_test.go:			buildcfg:      buildcfg.NVIDIA_CONTAINER_CLI_PATH,
internal/pkg/util/bin/bin_test.go:			configKey:     "nvidia-container-cli path",
internal/pkg/util/bin/bin_test.go:			configVal:     buildcfg.NVIDIA_CONTAINER_CLI_PATH,
internal/pkg/util/bin/bin_test.go:			expectPath:    buildcfg.NVIDIA_CONTAINER_CLI_PATH,
internal/pkg/util/bin/bin_test.go:			name:          "nvidia-container-cli invalid",
internal/pkg/util/bin/bin_test.go:			bin:           "nvidia-container-cli",
internal/pkg/util/bin/bin_test.go:			buildcfg:      buildcfg.NVIDIA_CONTAINER_CLI_PATH,
internal/pkg/util/bin/bin_test.go:			configKey:     "nvidia-container-cli path",
internal/pkg/util/bin/bin_test.go:			name:          "nvidia-container-cli empty",
internal/pkg/util/bin/bin_test.go:			bin:           "nvidia-container-cli",
internal/pkg/util/bin/bin_test.go:			buildcfg:      buildcfg.NVIDIA_CONTAINER_CLI_PATH,
internal/pkg/util/bin/bin_test.go:			configKey:     "nvidia-container-cli path",
internal/pkg/util/bin/bin_singularity.go:	case "nvidia-container-cli":
internal/pkg/util/bin/bin_singularity.go:		path = cfg.NvidiaContainerCliPath
internal/pkg/test/tool/require/require.go:// Nvidia checks that an NVIDIA stack is available
internal/pkg/test/tool/require/require.go:func Nvidia(t *testing.T) {
internal/pkg/test/tool/require/require.go:	nvsmi, err := exec.LookPath("nvidia-smi")
internal/pkg/test/tool/require/require.go:		t.Skipf("nvidia-smi not found on PATH: %v", err)
internal/pkg/test/tool/require/require.go:		t.Skipf("nvidia-smi failed to run: %v", err)
internal/pkg/test/tool/require/require.go:// NvCCLI checks that nvidia-container-cli is available
internal/pkg/test/tool/require/require.go:	_, err := exec.LookPath("nvidia-container-cli")
internal/pkg/test/tool/require/require.go:		t.Skipf("nvidia-container-cli not found on PATH: %v", err)
internal/pkg/test/tool/require/require.go:// Rocm checks that a Rocm stack is available
internal/pkg/test/tool/require/require.go:func Rocm(t *testing.T) {
internal/pkg/test/tool/require/require.go:	rocminfo, err := exec.LookPath("rocminfo")
internal/pkg/test/tool/require/require.go:		t.Skipf("rocminfo not found on PATH: %v", err)
internal/pkg/test/tool/require/require.go:	cmd := exec.Command(rocminfo)
internal/pkg/test/tool/require/require.go:		t.Skipf("rocminfo failed to run: %v - %v", err, string(output))
internal/pkg/runtime/launcher/native/launcher_linux.go:	"github.com/sylabs/singularity/v4/internal/pkg/util/gpu"
internal/pkg/runtime/launcher/native/launcher_linux.go:	// GPU configuration may add library bind to /.singularity.d/libs.
internal/pkg/runtime/launcher/native/launcher_linux.go:	// Note: --nvccli may implicitly add --writable-tmpfs, so handle that *after* GPUs.
internal/pkg/runtime/launcher/native/launcher_linux.go:	if err := l.SetGPUConfig(); err != nil {
internal/pkg/runtime/launcher/native/launcher_linux.go:		// We must fatal on error, as we are checking for correct ownership of nvidia-container-cli,
internal/pkg/runtime/launcher/native/launcher_linux.go:		sylog.Fatalf("While setting GPU configuration: %s", err)
internal/pkg/runtime/launcher/native/launcher_linux.go:// SetGPUConfig sets up EngineConfig entries for NV / ROCm usage, if requested.
internal/pkg/runtime/launcher/native/launcher_linux.go:func (l *Launcher) SetGPUConfig() error {
internal/pkg/runtime/launcher/native/launcher_linux.go:	if l.engineConfig.File.AlwaysUseNv && !l.cfg.NoNvidia {
internal/pkg/runtime/launcher/native/launcher_linux.go:		l.cfg.Nvidia = true
internal/pkg/runtime/launcher/native/launcher_linux.go:	if l.engineConfig.File.AlwaysUseRocm && !l.cfg.NoRocm {
internal/pkg/runtime/launcher/native/launcher_linux.go:		l.cfg.Rocm = true
internal/pkg/runtime/launcher/native/launcher_linux.go:		sylog.Verbosef("'always use rocm = yes' found in singularity.conf")
internal/pkg/runtime/launcher/native/launcher_linux.go:	if l.cfg.Nvidia && l.cfg.Rocm {
internal/pkg/runtime/launcher/native/launcher_linux.go:		sylog.Warningf("--nv and --rocm cannot be used together. Only --nv will be applied.")
internal/pkg/runtime/launcher/native/launcher_linux.go:	if l.cfg.Nvidia {
internal/pkg/runtime/launcher/native/launcher_linux.go:		// TODO: In privileged fakeroot mode we don't have the correct namespace context to run nvidia-container-cli
internal/pkg/runtime/launcher/native/launcher_linux.go:	if l.cfg.Rocm {
internal/pkg/runtime/launcher/native/launcher_linux.go:		return l.setRocmConfig()
internal/pkg/runtime/launcher/native/launcher_linux.go:// setNvCCLIConfig sets up EngineConfig entries for NVIDIA GPU configuration via nvidia-container-cli.
internal/pkg/runtime/launcher/native/launcher_linux.go:	sylog.Debugf("Using nvidia-container-cli for GPU setup")
internal/pkg/runtime/launcher/native/launcher_linux.go:	if os.Getenv("NVIDIA_VISIBLE_DEVICES") == "" {
internal/pkg/runtime/launcher/native/launcher_linux.go:			// When we use --contain we don't mount the NV devices by default in the nvidia-container-cli flow,
internal/pkg/runtime/launcher/native/launcher_linux.go:			// they must be mounted via specifying with`NVIDIA_VISIBLE_DEVICES`. This differs from the legacy
internal/pkg/runtime/launcher/native/launcher_linux.go:			// flow which mounts all GPU devices, always... so warn the user.
internal/pkg/runtime/launcher/native/launcher_linux.go:			sylog.Warningf("When using nvidia-container-cli with --contain NVIDIA_VISIBLE_DEVICES must be set or no GPUs will be available in container.")
internal/pkg/runtime/launcher/native/launcher_linux.go:			// In non-contained mode set NVIDIA_VISIBLE_DEVICES="all" by default, so MIGs are available.
internal/pkg/runtime/launcher/native/launcher_linux.go:			// Otherwise there is a difference vs legacy GPU binding. See Issue #471.
internal/pkg/runtime/launcher/native/launcher_linux.go:			sylog.Infof("Setting 'NVIDIA_VISIBLE_DEVICES=all' to emulate legacy GPU binding.")
internal/pkg/runtime/launcher/native/launcher_linux.go:			os.Setenv("NVIDIA_VISIBLE_DEVICES", "all")
internal/pkg/runtime/launcher/native/launcher_linux.go:	// Pass NVIDIA_ env vars that will be converted to nvidia-container-cli options
internal/pkg/runtime/launcher/native/launcher_linux.go:		if strings.HasPrefix(e, "NVIDIA_") {
internal/pkg/runtime/launcher/native/launcher_linux.go:		return fmt.Errorf("nvidia-container-cli requires --writable with user namespace/fakeroot")
internal/pkg/runtime/launcher/native/launcher_linux.go:		sylog.Infof("Setting --writable-tmpfs (required by nvidia-container-cli)")
internal/pkg/runtime/launcher/native/launcher_linux.go:// setNvLegacyConfig sets up EngineConfig entries for NVIDIA GPU configuration via direct binds of configured bins/libs.
internal/pkg/runtime/launcher/native/launcher_linux.go:	sylog.Debugf("Using legacy binds for nv GPU setup")
internal/pkg/runtime/launcher/native/launcher_linux.go:	gpuConfFile := filepath.Join(buildcfg.SINGULARITY_CONFDIR, "nvliblist.conf")
internal/pkg/runtime/launcher/native/launcher_linux.go:	ipcs, err := gpu.NvidiaIpcsPath()
internal/pkg/runtime/launcher/native/launcher_linux.go:	libs, bins, err := gpu.NvidiaPaths(gpuConfFile)
internal/pkg/runtime/launcher/native/launcher_linux.go:	l.setGPUBinds(libs, bins, ipcs, "nv")
internal/pkg/runtime/launcher/native/launcher_linux.go:// setRocmConfig sets up EngineConfig entries for ROCm GPU configuration via direct binds of configured bins/libs.
internal/pkg/runtime/launcher/native/launcher_linux.go:func (l *Launcher) setRocmConfig() error {
internal/pkg/runtime/launcher/native/launcher_linux.go:	sylog.Debugf("Using rocm GPU setup")
internal/pkg/runtime/launcher/native/launcher_linux.go:	l.engineConfig.SetRocm(true)
internal/pkg/runtime/launcher/native/launcher_linux.go:	gpuConfFile := filepath.Join(buildcfg.SINGULARITY_CONFDIR, "rocmliblist.conf")
internal/pkg/runtime/launcher/native/launcher_linux.go:	libs, bins, err := gpu.RocmPaths(gpuConfFile)
internal/pkg/runtime/launcher/native/launcher_linux.go:		sylog.Warningf("While finding ROCm bind points: %v", err)
internal/pkg/runtime/launcher/native/launcher_linux.go:	l.setGPUBinds(libs, bins, []string{}, "nv")
internal/pkg/runtime/launcher/native/launcher_linux.go:// setGPUBinds sets EngineConfig entries to bind the provided list of libs, bins, ipc files.
internal/pkg/runtime/launcher/native/launcher_linux.go:func (l *Launcher) setGPUBinds(libs, bins, ipcs []string, gpuPlatform string) {
internal/pkg/runtime/launcher/native/launcher_linux.go:		sylog.Warningf("Could not find any %s files on this host!", gpuPlatform)
internal/pkg/runtime/launcher/native/launcher_linux.go:			sylog.Warningf("%s files may not be bound with --writable", gpuPlatform)
internal/pkg/runtime/launcher/native/launcher_linux.go:		sylog.Warningf("Could not find any %s libraries on this host!", gpuPlatform)
internal/pkg/runtime/launcher/oci/mounts_linux.go:	"github.com/sylabs/singularity/v4/internal/pkg/util/gpu"
internal/pkg/runtime/launcher/oci/mounts_linux.go:	if err := l.addProcMount(mounts); err != nil {
internal/pkg/runtime/launcher/oci/mounts_linux.go:	if (l.cfg.Rocm || l.singularityConf.AlwaysUseRocm) && !l.cfg.NoRocm {
internal/pkg/runtime/launcher/oci/mounts_linux.go:		if err := l.addRocmMounts(mounts); err != nil {
internal/pkg/runtime/launcher/oci/mounts_linux.go:			return nil, fmt.Errorf("while configuring ROCm mount(s): %w", err)
internal/pkg/runtime/launcher/oci/mounts_linux.go:	if (l.cfg.Nvidia || l.singularityConf.AlwaysUseNv) && !l.cfg.NoNvidia {
internal/pkg/runtime/launcher/oci/mounts_linux.go:		if err := l.addNvidiaMounts(mounts); err != nil {
internal/pkg/runtime/launcher/oci/mounts_linux.go:			return nil, fmt.Errorf("while configuring Nvidia mount(s): %w", err)
internal/pkg/runtime/launcher/oci/mounts_linux.go:// addProcMount adds the /proc tree in the container.
internal/pkg/runtime/launcher/oci/mounts_linux.go:func (l *Launcher) addProcMount(mounts *[]specs.Mount) error {
internal/pkg/runtime/launcher/oci/mounts_linux.go:func (l *Launcher) addRocmMounts(mounts *[]specs.Mount) error {
internal/pkg/runtime/launcher/oci/mounts_linux.go:	gpuConfFile := filepath.Join(buildcfg.SINGULARITY_CONFDIR, "rocmliblist.conf")
internal/pkg/runtime/launcher/oci/mounts_linux.go:	libs, bins, err := gpu.RocmPaths(gpuConfFile)
internal/pkg/runtime/launcher/oci/mounts_linux.go:		sylog.Warningf("While finding ROCm bind points: %v", err)
internal/pkg/runtime/launcher/oci/mounts_linux.go:		sylog.Warningf("Could not find any ROCm libraries on this host!")
internal/pkg/runtime/launcher/oci/mounts_linux.go:	devs, err := gpu.RocmDevices()
internal/pkg/runtime/launcher/oci/mounts_linux.go:		sylog.Warningf("While finding ROCm devices: %v", err)
internal/pkg/runtime/launcher/oci/mounts_linux.go:		sylog.Warningf("Could not find any ROCm devices on this host!")
internal/pkg/runtime/launcher/oci/mounts_linux.go:func (l *Launcher) addNvidiaMounts(mounts *[]specs.Mount) error {
internal/pkg/runtime/launcher/oci/mounts_linux.go:	gpuConfFile := filepath.Join(buildcfg.SINGULARITY_CONFDIR, "nvliblist.conf")
internal/pkg/runtime/launcher/oci/mounts_linux.go:	libs, bins, err := gpu.NvidiaPaths(gpuConfFile)
internal/pkg/runtime/launcher/oci/mounts_linux.go:		sylog.Warningf("While finding NVIDIA bind points: %v", err)
internal/pkg/runtime/launcher/oci/mounts_linux.go:		sylog.Warningf("Could not find any NVIDIA libraries on this host!")
internal/pkg/runtime/launcher/oci/mounts_linux.go:	ipcs, err := gpu.NvidiaIpcsPath()
internal/pkg/runtime/launcher/oci/mounts_linux.go:		sylog.Warningf("While finding NVIDIA IPCs: %v", err)
internal/pkg/runtime/launcher/oci/mounts_linux.go:	devs, err := gpu.NvidiaDevices(true)
internal/pkg/runtime/launcher/oci/mounts_linux.go:		sylog.Warningf("While finding NVIDIA devices: %v", err)
internal/pkg/runtime/launcher/oci/mounts_linux.go:		sylog.Warningf("Could not find any NVIDIA devices on this host!")
internal/pkg/runtime/launcher/options.go:	// Nvidia enables NVIDIA GPU support.
internal/pkg/runtime/launcher/options.go:	Nvidia bool
internal/pkg/runtime/launcher/options.go:	// NcCCLI sets NVIDIA GPU support to use the nvidia-container-cli.
internal/pkg/runtime/launcher/options.go:	// NoNvidia disables NVIDIA GPU support when set default in singularity.conf.
internal/pkg/runtime/launcher/options.go:	NoNvidia bool
internal/pkg/runtime/launcher/options.go:	// Rocm enables Rocm GPU support.
internal/pkg/runtime/launcher/options.go:	Rocm bool
internal/pkg/runtime/launcher/options.go:	// NoRocm disable Rocm GPU support when set default in singularity.conf.
internal/pkg/runtime/launcher/options.go:	NoRocm bool
internal/pkg/runtime/launcher/options.go:// OptNvidia enables NVIDIA GPU support.
internal/pkg/runtime/launcher/options.go:// nvccli sets whether to use the nvidia-container-runtime (true), or legacy bind mounts (false).
internal/pkg/runtime/launcher/options.go:func OptNvidia(nv bool, nvccli bool) Option {
internal/pkg/runtime/launcher/options.go:		lo.Nvidia = nv || nvccli
internal/pkg/runtime/launcher/options.go:// OptNoNvidia disables NVIDIA GPU support, even if enabled via singularity.conf.
internal/pkg/runtime/launcher/options.go:func OptNoNvidia(b bool) Option {
internal/pkg/runtime/launcher/options.go:		lo.NoNvidia = b
internal/pkg/runtime/launcher/options.go:// OptRocm enable Rocm GPU support.
internal/pkg/runtime/launcher/options.go:func OptRocm(b bool) Option {
internal/pkg/runtime/launcher/options.go:		lo.Rocm = b
internal/pkg/runtime/launcher/options.go:// OptNoRocm disables Rocm GPU support, even if enabled via singularity.conf.
internal/pkg/runtime/launcher/options.go:func OptNoRocm(b bool) Option {
internal/pkg/runtime/launcher/options.go:		lo.NoRocm = b
internal/pkg/runtime/engine/config/starter/starter_linux.go:// nvidia-container-cli
internal/pkg/runtime/engine/singularity/prepare_linux.go:	// nvidia-container-cli requires additional caps in the starter bounding set.
internal/pkg/runtime/engine/singularity/container_linux.go:	"github.com/sylabs/singularity/v4/internal/pkg/util/gpu"
internal/pkg/runtime/engine/singularity/container_linux.go:		// If a container has a CUDA install in it then nvidia-container-cli will bind mount
internal/pkg/runtime/engine/singularity/container_linux.go:		// from <session_dir>/final/usr/local/cuda/compat into the main container lib dir.
internal/pkg/runtime/engine/singularity/container_linux.go:		// handling, so make it private here just before calling nvidia-container-cli.
internal/pkg/runtime/engine/singularity/container_linux.go:		sylog.Debugf("nvidia-container-cli")
internal/pkg/runtime/engine/singularity/container_linux.go:		// If we are not inside a user namespace then the NVCCLI call must exec nvidia-container-cli
internal/pkg/runtime/engine/singularity/container_linux.go:			devs, err := gpu.NvidiaDevices(true)
internal/pkg/runtime/engine/singularity/container_linux.go:				return fmt.Errorf("failed to get nvidia devices: %v", err)
internal/pkg/runtime/engine/singularity/container_linux.go:		if c.engine.EngineConfig.GetRocm() {
internal/pkg/runtime/engine/singularity/container_linux.go:			devs, err := gpu.RocmDevices()
internal/pkg/runtime/engine/singularity/container_linux.go:				return fmt.Errorf("failed to get rocm devices: %v", err)
internal/pkg/runtime/engine/singularity/rpc/client/client.go:// NvCCLI will call nvidia-container-cli to configure GPU(s) for the container.
internal/pkg/runtime/engine/singularity/rpc/server/server_linux.go:	"github.com/sylabs/singularity/v4/internal/pkg/util/gpu"
internal/pkg/runtime/engine/singularity/rpc/server/server_linux.go:// NvCCLI will call nvidia-container-cli to configure GPU(s) for the container.
internal/pkg/runtime/engine/singularity/rpc/server/server_linux.go:	// nvidia-container-cli successfully as root.
internal/pkg/runtime/engine/singularity/rpc/server/server_linux.go:	return gpu.NVCLIConfigure(arguments.Flags, arguments.RootFsPath, arguments.UserNS)
internal/pkg/cgroups/manager_linux.go:	procMgr := m.cgroup
internal/pkg/cgroups/manager_linux.go:		procMgr, err = lcmanager.New(lcConfig)
internal/pkg/cgroups/manager_linux.go:	return procMgr.Apply(pid)
internal/pkg/build/stage.go:		cmd.Env = currentEnvNoSingularity([]string{"DEBUG", "NV", "NVCCLI", "ROCM", "BINDPATH", "MOUNT", "PROOT"})
internal/pkg/build/stage.go:		cmd.Env = currentEnvNoSingularity([]string{"DEBUG", "NV", "NVCCLI", "ROCM", "BINDPATH", "MOUNT", "WRITABLE_TMPFS", "PROOT"})
internal/pkg/build/build.go:	// nvidia-container-cli path / ldconfig path will be needed by %post/%test
internal/pkg/build/build.go:	config.NvidiaContainerCliPath = sysConfig.NvidiaContainerCliPath
internal/pkg/image/unpacker/squashfs_singularity.go:		"--no-rocm",
mlocal/checks/project-post.chk:config_add_def NVIDIALIBS_FILE SINGULARITY_CONFDIR \"/nvliblist.conf\"
mlocal/checks/project-post.chk:   printf " checking: nvidia-container-cli... "
mlocal/checks/project-post.chk:   nvidia_container_cli_path=`command -v nvidia-container-cli || true`
mlocal/checks/project-post.chk:   if test -z "${nvidia_container_cli_path}" ; then
mlocal/checks/project-post.chk:      echo "yes (${nvidia_container_cli_path})"
mlocal/checks/project-post.chk:config_add_def NVIDIA_CONTAINER_CLI_PATH \"${nvidia_container_cli_path}\"
mlocal/frags/build_runtime.mk:# nvidia liblist config file
mlocal/frags/build_runtime.mk:nvidia_liblist := $(SOURCEDIR)/etc/nvliblist.conf
mlocal/frags/build_runtime.mk:nvidia_liblist_INSTALL := $(DESTDIR)$(SYSCONFDIR)/singularity/nvliblist.conf
mlocal/frags/build_runtime.mk:$(nvidia_liblist_INSTALL): $(nvidia_liblist)
mlocal/frags/build_runtime.mk:INSTALLFILES += $(nvidia_liblist_INSTALL)
mlocal/frags/build_runtime.mk:# rocm liblist config file
mlocal/frags/build_runtime.mk:rocm_liblist := $(SOURCEDIR)/etc/rocmliblist.conf
mlocal/frags/build_runtime.mk: rocm_liblist_INSTALL := $(DESTDIR)$(SYSCONFDIR)/singularity/rocmliblist.conf
mlocal/frags/build_runtime.mk:$(rocm_liblist_INSTALL): $(rocm_liblist)
mlocal/frags/build_runtime.mk:INSTALLFILES += $(rocm_liblist_INSTALL)
pkg/util/singularityconf/config.go:	UseNvCCLI               bool     `default:"no" authorized:"yes,no" directive:"use nvidia-container-cli"`
pkg/util/singularityconf/config.go:	AlwaysUseRocm           bool     `default:"no" authorized:"yes,no" directive:"always use rocm"`
pkg/util/singularityconf/config.go:	NvidiaContainerCliPath  string   `directive:"nvidia-container-cli path"`
pkg/util/singularityconf/config.go:# should be executed implicitly with the --nv option (useful for GPU only 
pkg/util/singularityconf/config.go:# USE NVIDIA-NVIDIA-CONTAINER-CLI ${TYPE}: [BOOL]
pkg/util/singularityconf/config.go:# If set to yes, Singularity will attempt to use nvidia-container-cli to setup
pkg/util/singularityconf/config.go:# GPUs within a container when the --nv flag is enabled.
pkg/util/singularityconf/config.go:use nvidia-container-cli = {{ if eq .UseNvCCLI true }}yes{{ else }}no{{ end }}
pkg/util/singularityconf/config.go:# ALWAYS USE ROCM ${TYPE}: [BOOL]
pkg/util/singularityconf/config.go:# should be executed implicitly with the --rocm option (useful for GPU only
pkg/util/singularityconf/config.go:always use rocm = {{ if eq .AlwaysUseRocm true }}yes{{ else }}no{{ end }}
pkg/util/singularityconf/config.go:# Path to the ldconfig executable, used to find GPU libraries.
pkg/util/singularityconf/config.go:# NVIDIA-CONTAINER-CLI PATH: [STRING]
pkg/util/singularityconf/config.go:# Path to the nvidia-container-cli executable, used to find GPU libraries.
pkg/util/singularityconf/config.go:# nvidia-container-cli path =
pkg/util/singularityconf/config.go:{{ if ne .NvidiaContainerCliPath "" }}nvidia-container-cli path = {{ .NvidiaContainerCliPath }}{{ end }}
pkg/runtime/engine/singularity/config/config.go:	Rocm                  bool              `json:"rocm,omitempty"`
pkg/runtime/engine/singularity/config/config.go:// SetNvLegacy sets nvLegacy flag to bind cuda libraries into containee.JSON.
pkg/runtime/engine/singularity/config/config.go:// SetNvCCLI sets nvcontainer flag to use nvidia-container-cli for CUDA setup
pkg/runtime/engine/singularity/config/config.go:// SetNVCCLIEnv sets env vars holding options for nvidia-container-cli GPU setup
pkg/runtime/engine/singularity/config/config.go:// GetNVCCLIEnv returns env vars holding options for nvidia-container-cli GPU setup
pkg/runtime/engine/singularity/config/config.go:// SetRocm sets rocm flag to bind rocm libraries into containee.JSON.
pkg/runtime/engine/singularity/config/config.go:func (e *EngineConfig) SetRocm(rocm bool) {
pkg/runtime/engine/singularity/config/config.go:	e.JSON.Rocm = rocm
pkg/runtime/engine/singularity/config/config.go:// GetRocm returns if rocm flag is set or not.
pkg/runtime/engine/singularity/config/config.go:func (e *EngineConfig) GetRocm() bool {
pkg/runtime/engine/singularity/config/config.go:	return e.JSON.Rocm
test/keys/pgp-private.asc:SF8U0SMA5mEfFGOmpkZbABEBAAEAD/40144feNt5cIkfEuDE46gPUeLU66CXEUHJ
test/keys/pgp-private.asc:qZzjyD8jiMp4bvY2TAW+urWTDe/QzJeKIwgA/oN614cYMn7mNCCLhhcpuONU0a82
CHANGELOG.md:- Added `libnvidia-nvvm` to `nvliblist.conf`. Newer NVIDIA Drivers (known with
CHANGELOG.md:  >= 525.85.05) require this lib to compile OpenCL programs against NVIDIA GPUs,
CHANGELOG.md:  i.e. `libnvidia-opencl` depends on `libnvidia-nvvm`.
CHANGELOG.md:- Added the upcoming NVIDIA driver library `libnvidia-gpucomp.so` to the
CHANGELOG.md:  list of libraries to add to NVIDIA GPU-enabled containers.
CHANGELOG.md:- Ensure consistent binding of libraries under `--nv/--rocm` when duplicate
CHANGELOG.md:    - `--rocm` to bind ROCm GPU libraries and devices into the container.
CHANGELOG.md:    - `--nv` to bind Nvidia driver / basic CUDA libraries and devices into the
CHANGELOG.md:- In `--rocm` mode, the whole of `/dev/dri` is now bound into the container when
CHANGELOG.md:  required for later ROCm versions.
CHANGELOG.md:- Support nvidia-container-cli v1.8.0 and above, via fix to capability set.
CHANGELOG.md:  legacy GPU binding behaviour.
CHANGELOG.md:- Paths for `cryptsetup`, `go`, `ldconfig`, `mksquashfs`, `nvidia-container-cli`,
CHANGELOG.md:- When calling `ldconfig` to find GPU libraries, singularity will *not* fall back
CHANGELOG.md:  find GPU libraries.
CHANGELOG.md:- `--nv` will not call `nvidia-container-cli` to find host libraries, unless
CHANGELOG.md:  the new experimental GPU setup flow that employs `nvidia-container-cli`
CHANGELOG.md:  for all GPU related operations is enabled (see below).
CHANGELOG.md:- If a container is run with `--nvcli` and `--contain`, only GPU devices
CHANGELOG.md:  specified via the `NVIDIA_VISIBLE_DEVICES` environment variable will be
CHANGELOG.md:  exposed within the container. Use `NVIDIA_VISIBLE_DEVICES=all` to access all
CHANGELOG.md:  GPUs inside a container run with `--nvccli`.
CHANGELOG.md:- The experimental `--nvccli` flag will use `nvidia-container-cli` to setup the
CHANGELOG.md:  container for Nvidia GPU operation. SingularityCE will not bind GPU libraries
CHANGELOG.md:  itself. Environment variables that are used with Nvidia's `docker-nvidia`
CHANGELOG.md:  runtime to configure GPU visibility / driver capabilities & requirements are
CHANGELOG.md:  default, the `compute` and `utility` GPU capabilities are configured. The `use
CHANGELOG.md:  nvidia-container-cli` option in `singularity.conf` can be set to `yes` to
CHANGELOG.md:  always use `nvidia-container-cli` when supported. Note that in a setuid
CHANGELOG.md:  install, `nvidia-container-cli` will be run as root with required ambient
CHANGELOG.md:- The `build` command now honors `--nv`, `--rocm`, and `--bind` flags,
CHANGELOG.md:  permitting builds that require GPU access or files bound in from the host.
CHANGELOG.md:- The `--nv` flag for NVIDIA GPU support will not resolve libraries reported by
CHANGELOG.md:  `nvidia-container-cli` via the ld cache. Will instead respect absolute paths
CHANGELOG.md:  GPU libraries. Fixes problems on systems using Nix / Guix.
CHANGELOG.md:- Ensure `/dev/kfd` is bound into container for ROCm when `--rocm` is used with
CHANGELOG.md:- Fix LD_LIBRARY_PATH environment override regression with `--nv/--rocm`.
CHANGELOG.md:- Bind additional CUDA 10.2 libs when using the `--nv` option without
CHANGELOG.md:  `nvidia-container-cli`.
CHANGELOG.md:- Fix an NVIDIA persistenced socket bind error with `--writable`.
CHANGELOG.md:- New support for AMD GPUs via `--rocm` option added to bind ROCm devices and
CHANGELOG.md:  - Binds NVIDIA persistenced socket when `--nv` is invoked
README.md:- Integration over isolation by default. Easily make use of GPUs, high speed
CONTRIBUTORS.md:- Adam Simpson <asimpson@nvidia.com>, <adambsimpson@gmail.com>
CONTRIBUTORS.md:- Daniel Dadap <ddadap@nvidia.com>
CONTRIBUTORS.md:- Felix Abecassis <fabecassis@nvidia.com>
CONTRIBUTORS.md:- Evan Lezar <elezar@nvidia.com>, <evanlezar@gmail.com>
etc/seccomp-profiles/default.json:				"rt_sigprocmask",
etc/rocmliblist.conf:# ROCMLIBLIST.CONF
etc/rocmliblist.conf:# This configuration file determines which ROCm libraries to search for on the
etc/rocmliblist.conf:# host system when the --rocm option is invoked.  You can edit it if you have
etc/rocmliblist.conf:# will be mounted into the container when the --rocm option is passed.
etc/rocmliblist.conf:rocm-smi
etc/rocmliblist.conf:rocminfo
etc/rocmliblist.conf:libdrm_amdgpu.so
etc/rocmliblist.conf:libOpenCL.so
etc/conf/testdata/test_default.tmpl:# should be executed implicitly with the --nv option (useful for GPU only 
etc/conf/testdata/test_3.out.correct:# should be executed implicitly with the --nv option (useful for GPU only 
etc/conf/testdata/test_1.out.correct:# should be executed implicitly with the --nv option (useful for GPU only 
etc/conf/testdata/test_3.in:# should be executed implicitly with the --nv option (useful for GPU only 
etc/conf/testdata/test_2.out.correct:# should be executed implicitly with the --nv option (useful for GPU only 
etc/conf/testdata/test_2.in:# should be executed implicitly with the --nv option (useful for GPU only 
etc/conf/gen.go:	c.NvidiaContainerCliPath = buildcfg.NVIDIA_CONTAINER_CLI_PATH
etc/nvliblist.conf:# This configuration file determines which NVIDIA libraries to search for on 
etc/nvliblist.conf:nvidia-smi
etc/nvliblist.conf:nvidia-debugdump
etc/nvliblist.conf:nvidia-persistenced
etc/nvliblist.conf:nvidia-cuda-mps-control
etc/nvliblist.conf:nvidia-cuda-mps-server
etc/nvliblist.conf:libcuda.so
etc/nvliblist.conf:libEGL_nvidia.so
etc/nvliblist.conf:libGLESv1_CM_nvidia.so
etc/nvliblist.conf:libGLESv2_nvidia.so
etc/nvliblist.conf:libGLX_nvidia.so
etc/nvliblist.conf:libnvidia-cbl.so
etc/nvliblist.conf:libnvidia-cfg.so
etc/nvliblist.conf:libnvidia-compiler.so
etc/nvliblist.conf:libnvidia-eglcore.so
etc/nvliblist.conf:libnvidia-egl-wayland.so
etc/nvliblist.conf:libnvidia-encode.so
etc/nvliblist.conf:libnvidia-fatbinaryloader.so
etc/nvliblist.conf:libnvidia-fbc.so
etc/nvliblist.conf:libnvidia-glcore.so
etc/nvliblist.conf:libnvidia-glsi.so
etc/nvliblist.conf:libnvidia-glvkspirv.so
etc/nvliblist.conf:libnvidia-gpucomp.so
etc/nvliblist.conf:libnvidia-gtk2.so
etc/nvliblist.conf:libnvidia-gtk3.so
etc/nvliblist.conf:libnvidia-ifr.so
etc/nvliblist.conf:libnvidia-ml.so
etc/nvliblist.conf:libnvidia-nvvm.so
etc/nvliblist.conf:libnvidia-opencl.so
etc/nvliblist.conf:libnvidia-opticalflow.so
etc/nvliblist.conf:libnvidia-ptxjitcompiler.so
etc/nvliblist.conf:libnvidia-rtcore.so
etc/nvliblist.conf:libnvidia-tls.so
etc/nvliblist.conf:libnvidia-wfb.so
etc/nvliblist.conf:libOpenCL.so
etc/nvliblist.conf:libvdpau_nvidia.so
etc/nvliblist.conf:nvidia_drv.so
e2e/internal/e2e/config.go:	c.NvidiaContainerCliPath = buildcfg.NVIDIA_CONTAINER_CLI_PATH
e2e/suite.go:	"github.com/sylabs/singularity/v4/e2e/gpu"
e2e/suite.go:	"GPU":            gpu.E2ETests,
e2e/data/data.go:	// Basic test that we can run the bound in `nvidia-smi` which *should* be on the PATH
e2e/gpu/gpu.go:package gpu
e2e/gpu/gpu.go:func (c ctx) testNvidiaLegacy(t *testing.T) {
e2e/gpu/gpu.go:	require.Nvidia(t)
e2e/gpu/gpu.go:	// Use Ubuntu 20.04 as this is a recent distro officially supported by Nvidia CUDA.
e2e/gpu/gpu.go:	imageFile, err := fs.MakeTmpFile("", "test-nvidia-legacy-", 0o755)
e2e/gpu/gpu.go:	// Basic test that we can run the bound in `nvidia-smi` which *should* be on the PATH
e2e/gpu/gpu.go:			args:    []string{"--nv", imagePath, "nvidia-smi"},
e2e/gpu/gpu.go:			args:    []string{"--contain", "--nv", imagePath, "nvidia-smi"},
e2e/gpu/gpu.go:			args:    []string{"--nv", imagePath, "nvidia-smi"},
e2e/gpu/gpu.go:			args:    []string{"--nv", imagePath, "nvidia-smi"},
e2e/gpu/gpu.go:			args:    []string{"--nv", imagePath, "nvidia-smi"},
e2e/gpu/gpu.go:func (c ctx) ociTestNvidiaLegacy(t *testing.T) {
e2e/gpu/gpu.go:	require.Nvidia(t)
e2e/gpu/gpu.go:	// Basic test that we can run the bound in `nvidia-smi` which *should* be on the PATH
e2e/gpu/gpu.go:			args:    []string{"--nv", imageURL, "nvidia-smi"},
e2e/gpu/gpu.go:			args:    []string{"--nv", imageURL, "nvidia-smi"},
e2e/gpu/gpu.go:			args:    []string{"--nv", imageURL, "nvidia-smi"},
e2e/gpu/gpu.go:	require.Nvidia(t)
e2e/gpu/gpu.go:	// Use Ubuntu 20.04 as this is a recent distro officially supported by Nvidia CUDA.
e2e/gpu/gpu.go:	// Basic test that we can run the bound in `nvidia-smi` which *should* be on the PATH
e2e/gpu/gpu.go:			args:       []string{"--nv", "--nvccli", imagePath, "nvidia-smi"},
e2e/gpu/gpu.go:			// With --contain, we should only see NVIDIA_VISIBLE_DEVICES configured GPUs
e2e/gpu/gpu.go:			args:        []string{"--contain", "--nv", "--nvccli", imagePath, "nvidia-smi"},
e2e/gpu/gpu.go:			args:       []string{"--contain", "--nv", "--nvccli", imagePath, "nvidia-smi"},
e2e/gpu/gpu.go:			env:        []string{"NVIDIA_VISIBLE_DEVICES=all"},
e2e/gpu/gpu.go:			// If we only request compute, not utility, then nvidia-smi should not be present
e2e/gpu/gpu.go:			args:        []string{"--nv", "--nvccli", imagePath, "nvidia-smi"},
e2e/gpu/gpu.go:			env:         []string{"NVIDIA_DRIVER_CAPABILITIES=compute"},
e2e/gpu/gpu.go:			expectMatch: e2e.ExpectError(e2e.ContainMatch, "\"nvidia-smi\": executable file not found in $PATH"),
e2e/gpu/gpu.go:			// Require CUDA version >8 should be fine!
e2e/gpu/gpu.go:			args:       []string{"--nv", "--nvccli", imagePath, "nvidia-smi"},
e2e/gpu/gpu.go:			env:        []string{"NVIDIA_REQUIRE_CUDA=cuda>8"},
e2e/gpu/gpu.go:			// Require CUDA version >999 should not be satisfied
e2e/gpu/gpu.go:			args:        []string{"--nv", "--nvccli", imagePath, "nvidia-smi"},
e2e/gpu/gpu.go:			env:         []string{"NVIDIA_REQUIRE_CUDA=cuda>999"},
e2e/gpu/gpu.go:			expectMatch: e2e.ExpectError(e2e.ContainMatch, "requirement error: unsatisfied condition: cuda>99"),
e2e/gpu/gpu.go:			args:    []string{"--nv", "--nvccli", "--writable", imagePath, "nvidia-smi"},
e2e/gpu/gpu.go:			args:    []string{"--nv", "--nvccli", imagePath, "nvidia-smi"},
e2e/gpu/gpu.go:func (c ctx) testRocm(t *testing.T) {
e2e/gpu/gpu.go:	require.Rocm(t)
e2e/gpu/gpu.go:	// rocminfo now needs lsmod - do a brittle bind in for simplicity.
e2e/gpu/gpu.go:	// Use Ubuntu 22.04 as this is the most recent distro officially supported by ROCm.
e2e/gpu/gpu.go:	imageFile, err := fs.MakeTmpFile("", "test-rocm-", 0o755)
e2e/gpu/gpu.go:	// Basic test that we can run the bound in `rocminfo` which *should* be on the PATH
e2e/gpu/gpu.go:			args:    []string{"-B", lsmod, "--rocm", imagePath, "rocminfo"},
e2e/gpu/gpu.go:			args:    []string{"-B", lsmod, "--contain", "--rocm", imagePath, "rocminfo"},
e2e/gpu/gpu.go:			args:    []string{"-B", lsmod, "--rocm", imagePath, "rocminfo"},
e2e/gpu/gpu.go:			args:    []string{"-B", lsmod, "--rocm", imagePath, "rocminfo"},
e2e/gpu/gpu.go:			args:    []string{"-B", lsmod, "--rocm", imagePath, "rocminfo"},
e2e/gpu/gpu.go:func (c ctx) ociTestRocm(t *testing.T) {
e2e/gpu/gpu.go:	require.Rocm(t)
e2e/gpu/gpu.go:	// rocminfo now needs lsmod - do a brittle bind in for simplicity.
e2e/gpu/gpu.go:	// Use Ubuntu 22.04 as this is the most recent distro officially supported by ROCm.
e2e/gpu/gpu.go:	// Basic test that we can run the bound in `rocminfo` which *should* be on the PATH
e2e/gpu/gpu.go:			args:    []string{"-B", lsmod, "--rocm", imageURL, "rocminfo"},
e2e/gpu/gpu.go:			args:    []string{"-B", lsmod, "--rocm", imageURL, "rocminfo"},
e2e/gpu/gpu.go:			args:    []string{"-B", lsmod, "--rocm", imageURL, "rocminfo"},
e2e/gpu/gpu.go:func (c ctx) testBuildNvidiaLegacy(t *testing.T) {
e2e/gpu/gpu.go:	require.Nvidia(t)
e2e/gpu/gpu.go:	nvsmi, _ := exec.LookPath("nvidia-smi")
e2e/gpu/gpu.go:	// Use Ubuntu 20.04 as this is the most recent distro officially supported by Nvidia CUDA.
e2e/gpu/gpu.go:	tmpdir, cleanup := e2e.MakeTempDir(t, c.env.TestDir, "build-nvidia-legacy", "build with nvidia")
e2e/gpu/gpu.go:	// Basic test that we can run the bound in `rocminfo` which *should* be on the PATH
e2e/gpu/gpu.go:	rawDef := fmt.Sprintf(buildDefinition, sourceImage, nvsmi, "", "nvidia-smi")
e2e/gpu/gpu.go:	require.Nvidia(t)
e2e/gpu/gpu.go:	nvsmi, _ := exec.LookPath("nvidia-smi")
e2e/gpu/gpu.go:	// Use Ubuntu 20.04 as this is the most recent distro officially supported by Nvidia CUDA.
e2e/gpu/gpu.go:	// Basic test that we can run the bound in `rocminfo` which *should* be on the PATH
e2e/gpu/gpu.go:	rawDef := fmt.Sprintf(buildDefinition, sourceImage, nvsmi, "", "nvidia-smi")
e2e/gpu/gpu.go:func (c ctx) testBuildRocm(t *testing.T) {
e2e/gpu/gpu.go:	require.Rocm(t)
e2e/gpu/gpu.go:	// rocminfo now needs lsmod - do a brittle bind in for simplicity.
e2e/gpu/gpu.go:	rocmInfo, _ := exec.LookPath("rocminfo")
e2e/gpu/gpu.go:	// Use Ubuntu 22.04 as this is the most recent distro officially supported by ROCm.
e2e/gpu/gpu.go:	tmpdir, cleanup := e2e.MakeTempDir(t, c.env.TestDir, "build-rocm-image", "build with rocm")
e2e/gpu/gpu.go:	// Basic test that we can run the bound in `rocminfo` which *should* be on the PATH
e2e/gpu/gpu.go:		setRocmFlag bool
e2e/gpu/gpu.go:			name:        "WithRocmRoot",
e2e/gpu/gpu.go:			setRocmFlag: true,
e2e/gpu/gpu.go:			name:        "WithRocmFakeroot",
e2e/gpu/gpu.go:			setRocmFlag: true,
e2e/gpu/gpu.go:			name:        "WithoutRocmRoot",
e2e/gpu/gpu.go:			setRocmFlag: false,
e2e/gpu/gpu.go:			name:        "WithoutRocmFakeroot",
e2e/gpu/gpu.go:			setRocmFlag: false,
e2e/gpu/gpu.go:	rawDef := fmt.Sprintf(buildDefinition, sourceImage, rocmInfo, lsmod, "rocminfo")
e2e/gpu/gpu.go:		if tt.setRocmFlag {
e2e/gpu/gpu.go:			args = append(args, "--rocm")
e2e/gpu/gpu.go:		"nvidia":       c.testNvidiaLegacy,
e2e/gpu/gpu.go:		"rocm":         c.testRocm,
e2e/gpu/gpu.go:		"build nvidia": c.testBuildNvidiaLegacy,
e2e/gpu/gpu.go:		"build rocm":   c.testBuildRocm,
e2e/gpu/gpu.go:		"oci nvidia": c.ociTestNvidiaLegacy,
e2e/gpu/gpu.go:		"oci rocm":   c.ociTestRocm,
cmd/internal/cli/singularity.go:	cmdArgs := []string{"exec", "--contain", "--no-home", "--no-nv", "--no-rocm", abspath}
cmd/internal/cli/build.go:	nvidia          bool
cmd/internal/cli/build.go:	rocm            bool
cmd/internal/cli/build.go:	Value:        &buildArgs.nvidia,
cmd/internal/cli/build.go:	Usage:        "inject host Nvidia libraries during build for post and test sections (not supported with remote build)",
cmd/internal/cli/build.go:	Usage:        "use nvidia-container-cli for GPU setup (experimental)",
cmd/internal/cli/build.go:// --rocm
cmd/internal/cli/build.go:var buildRocmFlag = cmdline.Flag{
cmd/internal/cli/build.go:	ID:           "rocmFlag",
cmd/internal/cli/build.go:	Value:        &buildArgs.rocm,
cmd/internal/cli/build.go:	Name:         "rocm",
cmd/internal/cli/build.go:	Usage:        "inject host Rocm libraries during build for post and test sections (not supported with remote build)",
cmd/internal/cli/build.go:	EnvKeys:      []string{"ROCM"},
cmd/internal/cli/build.go:		cmdManager.RegisterFlagForCmd(&buildRocmFlag, buildCmd)
cmd/internal/cli/actions.go:		launcher.OptNvidia(nvidia, nvCCLI),
cmd/internal/cli/actions.go:		launcher.OptNoNvidia(noNvidia),
cmd/internal/cli/actions.go:		launcher.OptRocm(rocm),
cmd/internal/cli/actions.go:		launcher.OptNoRocm(noRocm),
cmd/internal/cli/build_linux.go:	if buildArgs.nvidia {
cmd/internal/cli/build_linux.go:	if buildArgs.rocm {
cmd/internal/cli/build_linux.go:			sylog.Fatalf("--rocm option is not supported for remote build")
cmd/internal/cli/build_linux.go:			sylog.Fatalf("--rocm option is not supported for OCI builds from Dockerfiles")
cmd/internal/cli/build_linux.go:		os.Setenv("SINGULARITY_ROCM", "1")
cmd/internal/cli/action_flags.go:	nvidia          bool
cmd/internal/cli/action_flags.go:	rocm            bool
cmd/internal/cli/action_flags.go:	noNvidia        bool
cmd/internal/cli/action_flags.go:	noRocm          bool
cmd/internal/cli/action_flags.go:var actionNvidiaFlag = cmdline.Flag{
cmd/internal/cli/action_flags.go:	ID:           "actionNvidiaFlag",
cmd/internal/cli/action_flags.go:	Value:        &nvidia,
cmd/internal/cli/action_flags.go:	Usage:        "enable Nvidia support",
cmd/internal/cli/action_flags.go:	Usage:        "use nvidia-container-cli for GPU setup (experimental)",
cmd/internal/cli/action_flags.go:// --rocm flag to automatically bind
cmd/internal/cli/action_flags.go:var actionRocmFlag = cmdline.Flag{
cmd/internal/cli/action_flags.go:	ID:           "actionRocmFlag",
cmd/internal/cli/action_flags.go:	Value:        &rocm,
cmd/internal/cli/action_flags.go:	Name:         "rocm",
cmd/internal/cli/action_flags.go:	Usage:        "enable experimental Rocm support",
cmd/internal/cli/action_flags.go:	EnvKeys:      []string{"ROCM"},
cmd/internal/cli/action_flags.go:// hidden flag to disable nvidia bindings when 'always use nv = yes'
cmd/internal/cli/action_flags.go:var actionNoNvidiaFlag = cmdline.Flag{
cmd/internal/cli/action_flags.go:	ID:           "actionNoNvidiaFlag",
cmd/internal/cli/action_flags.go:	Value:        &noNvidia,
cmd/internal/cli/action_flags.go:// hidden flag to disable rocm bindings when 'always use rocm = yes'
cmd/internal/cli/action_flags.go:var actionNoRocmFlag = cmdline.Flag{
cmd/internal/cli/action_flags.go:	ID:           "actionNoRocmFlag",
cmd/internal/cli/action_flags.go:	Value:        &noRocm,
cmd/internal/cli/action_flags.go:	Name:         "no-rocm",
cmd/internal/cli/action_flags.go:	EnvKeys:      []string{"ROCM_OFF", "NO_ROCM"},
cmd/internal/cli/action_flags.go:		cmdManager.RegisterFlagForCmd(&actionNoNvidiaFlag, actionsInstanceCmd...)
cmd/internal/cli/action_flags.go:		cmdManager.RegisterFlagForCmd(&actionNoRocmFlag, actionsInstanceCmd...)
cmd/internal/cli/action_flags.go:		cmdManager.RegisterFlagForCmd(&actionNvidiaFlag, actionsInstanceCmd...)
cmd/internal/cli/action_flags.go:		cmdManager.RegisterFlagForCmd(&actionRocmFlag, actionsInstanceCmd...)
cmd/starter/c/include/starter.h:    /* bounding capability set will include caps needed by nvidia-container-cli */
cmd/starter/c/starter.c:    /* required by nvidia-container-cli */
cmd/starter/c/starter.c:        debugf("Enabling bounding capabilities for nvidia-container-cli\n");
cmd/starter/c/starter.c:    if ( sigprocmask(SIG_SETMASK, &mask, NULL) == -1 ) {
cmd/starter/c/starter.c:            if ( sigprocmask(SIG_SETMASK, &usrmask, NULL) == -1 ) {
cmd/starter/c/starter.c:            if ( sigprocmask(SIG_UNBLOCK, &usrmask, NULL) == -1 ) {
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:#   * cuda-linux64-rel-8.0.44-21122537.run  (* see below)
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:#   * NVIDIA-Linux-x86_64-375.20.run        (* see below)
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:#   * cudnn-8.0-linux-x64-v5.1.tgz          (https://developer.nvidia.com/cudnn)
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:# * The cuda-linux64 and NVIDIA-Linux files can be obtained by downloading the
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:# NVIDIA CUDA local runfile `cuda_8.0.44_linux.run` from:
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:#   https://developer.nvidia.com/cuda-downloads
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:#   sh cuda_8.0.44_linux.run --extract=<absolute/path/to/bootstrap/directory>
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:# IF YOUR HPC SYSTEM IS USING A DIFFERENT VERSION OF CUDA AND/OR NVIDIA DRIVERS
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:# cuda_8.0.44_linux.run returns driver version 367.48.
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:# a GPU, comment out the final test.
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:    chk_nvidia_uvm=$(grep nvidia_uvm /proc/modules)
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:    if [ -z "$chk_nvidia_uvm" ]; then
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:        echo "Problem detected on the host: the Linux kernel module nvidia_uvm is not loaded"
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:    NV_CUDA_FILE=cuda-linux64-rel-8.0.44-21122537.run
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:    NV_DRIVER_FILE=NVIDIA-Linux-x86_64-${NV_DRIVER_VERSION}.run
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:    echo "Unpacking NVIDIA driver into container..."
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:    mv NVIDIA-Linux-x86_64-${NV_DRIVER_VERSION} NVIDIA-Linux-x86_64
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:    cd NVIDIA-Linux-x86_64/
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:    ln -v -s libnvidia-ml.so.$NV_DRIVER_VERSION libnvidia-ml.so.1
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:    ln -v -s libcuda.so.$NV_DRIVER_VERSION libcuda.so.1
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:    echo "Running NVIDIA CUDA installer..."
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:    sh $NV_CUDA_FILE -noprompt -nosymlink -prefix=${SINGULARITY_ROOTFS}/usr/local/cuda-8.0
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:    ln -r -s ${SINGULARITY_ROOTFS}/usr/local/cuda-8.0 ${SINGULARITY_ROOTFS}/usr/local/cuda
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:    echo "Adding NVIDIA PATHs to /environment..."
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:    NV_DRIVER_PATH=/usr/local/NVIDIA-Linux-x86_64
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$NV_DRIVER_PATH:\$LD_LIBRARY_PATH
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:    # Ubuntu/Linux 64-bit, GPU enabled, Python 2.7 (Requires CUDA toolkit 8.0 and CuDNN v5)
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:    export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp27-none-linux_x86_64.whl
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/NVIDIA-Linux-x86_64:$LD_LIBRARY_PATH
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:    # Runs in less than 30 minutes on low-end CPU; in less than 2 minutes on GPU
examples/legacy/2.2/contrib/ubuntu16-tensorflow-0.12.1-gpu.def:    # Comment the following line if building the container inside a VM with no access to a GPU

```
