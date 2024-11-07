# https://github.com/containers/podman

```console
RELEASE_NOTES.md:- The `--gpus` option to `podman create` and `podman run` is now compatible with Nvidia GPUs ([#21156](https://github.com/containers/podman/issues/21156)).
RELEASE_NOTES.md:- The `podman kube generate` and `podman kube play` commands now support `securityContext.procMount: Unmasked` ([#19881](https://github.com/containers/podman/issues/19881)).
RELEASE_NOTES.md:- A new option, `--gpus`, has been added to `podman create` and `podman run` as a no-op for better compatibility with Docker. If the nvidia-container-runtime package is installed, GPUs should be automatically added to containers without using the flag.
vendor/github.com/klauspost/cpuid/v2/detect_arm64.go:		c.VendorString = "NVIDIA Corporation"
vendor/github.com/klauspost/cpuid/v2/detect_arm64.go:		c.VendorID = NVIDIA
vendor/github.com/klauspost/cpuid/v2/featureid_string.go:const _FeatureID_name = "firstIDADXAESNIAMD3DNOWAMD3DNOWEXTAMXBF16AMXFP16AMXINT8AMXTILEAPX_FAVXAVX10AVX10_128AVX10_256AVX10_512AVX2AVX512BF16AVX512BITALGAVX512BWAVX512CDAVX512DQAVX512ERAVX512FAVX512FP16AVX512IFMAAVX512PFAVX512VBMIAVX512VBMI2AVX512VLAVX512VNNIAVX512VP2INTERSECTAVX512VPOPCNTDQAVXIFMAAVXNECONVERTAVXSLOWAVXVNNIAVXVNNIINT8AVXVNNIINT16BHI_CTRLBMI1BMI2CETIBTCETSSCLDEMOTECLMULCLZEROCMOVCMPCCXADDCMPSB_SCADBS_SHORTCMPXCHG8CPBOOSTCPPCCX16EFER_LMSLE_UNSENQCMDERMSF16CFLUSH_L1DFMA3FMA4FP128FP256FSRMFXSRFXSROPTGFNIHLEHRESETHTTHWAHYBRID_CPUHYPERVISORIA32_ARCH_CAPIA32_CORE_CAPIBPBIBPB_BRTYPEIBRSIBRS_PREFERREDIBRS_PROVIDES_SMPIBSIBSBRNTRGTIBSFETCHSAMIBSFFVIBSOPCNTIBSOPCNTEXTIBSOPSAMIBSRDWROPCNTIBSRIPINVALIDCHKIBS_FETCH_CTLXIBS_OPDATA4IBS_OPFUSEIBS_PREVENTHOSTIBS_ZEN4IDPRED_CTRLINT_WBINVDINVLPGBKEYLOCKERKEYLOCKERWLAHFLAMLBRVIRTLZCNTMCAOVERFLOWMCDT_NOMCOMMITMD_CLEARMMXMMXEXTMOVBEMOVDIR64BMOVDIRIMOVSB_ZLMOVUMPXMSRIRCMSRLISTMSR_PAGEFLUSHNRIPSNXOSXSAVEPCONFIGPOPCNTPPINPREFETCHIPSFDRDPRURDRANDRDSEEDRDTSCPRRSBA_CTRLRTMRTM_ALWAYS_ABORTSBPBSERIALIZESEVSEV_64BITSEV_ALTERNATIVESEV_DEBUGSWAPSEV_ESSEV_RESTRICTEDSEV_SNPSGXSGXLCSHASMESME_COHERENTSPEC_CTRL_SSBDSRBDS_CTRLSRSO_MSR_FIXSRSO_NOSRSO_USER_KERNEL_NOSSESSE2SSE3SSE4SSE42SSE4ASSSE3STIBPSTIBP_ALWAYSONSTOSB_SHORTSUCCORSVMSVMDASVMFBASIDSVMLSVMNPSVMPFSVMPFTSYSCALLSYSEETBMTDX_GUESTTLB_FLUSH_NESTEDTMETOPEXTTSCRATEMSRTSXLDTRKVAESVMCBCLEANVMPLVMSA_REGPROTVMXVPCLMULQDQVTEWAITPKGWBNOINVDWRMSRNSX87XGETBV1XOPXSAVEXSAVECXSAVEOPTXSAVESAESARMARMCPUIDASIMDASIMDDPASIMDHPASIMDRDMATOMICSCRC32DCPOPEVTSTRMFCMAFPFPHPGPAJSCVTLRCPCPMULLSHA1SHA2SHA3SHA512SM3SM4SVElastID"
vendor/github.com/klauspost/cpuid/v2/featureid_string.go:	_ = x[NVIDIA-22]
vendor/github.com/klauspost/cpuid/v2/featureid_string.go:const _Vendor_name = "VendorUnknownIntelAMDVIATransmetaNSCKVMMSVMVMwareXenHVMBhyveHygonSiSRDCAmpereARMBroadcomCaviumDECFujitsuInfineonMotorolaNVIDIAAMCCQualcommMarvelllastVendor"
vendor/github.com/klauspost/cpuid/v2/cpuid.go:	NVIDIA
vendor/github.com/checkpoint-restore/checkpointctl/lib/metadata.go:	AmdgpuPagesPrefix = "amdgpu-pages-"
vendor/github.com/google/pprof/profile/legacy_profile.go:	procMapsRE  = regexp.MustCompile(`^` + cHexRange + cPerm + cSpaceHex + hexPair + spaceDigits + cSpaceString)
vendor/github.com/google/pprof/profile/legacy_profile.go:// ParseProcMaps parses a memory map in the format of /proc/self/maps.
vendor/github.com/google/pprof/profile/legacy_profile.go:func ParseProcMaps(rd io.Reader) ([]*Mapping, error) {
vendor/github.com/google/pprof/profile/legacy_profile.go:	return parseProcMapsFromScanner(s)
vendor/github.com/google/pprof/profile/legacy_profile.go:func parseProcMapsFromScanner(s *bufio.Scanner) ([]*Mapping, error) {
vendor/github.com/google/pprof/profile/legacy_profile.go:	mapping, err := parseProcMapsFromScanner(s)
vendor/github.com/google/pprof/profile/legacy_profile.go:	if me := procMapsRE.FindStringSubmatch(l); len(me) == 6 {
vendor/github.com/vishvananda/netlink/virtio.go:	VIRTIO_ID_GPU            = 16 // virtio GPU
vendor/github.com/crc-org/vfkit/pkg/config/json.go:	vfGpu          vmComponentKind = "virtiogpu"
vendor/github.com/crc-org/vfkit/pkg/config/json.go:	case vfGpu:
vendor/github.com/crc-org/vfkit/pkg/config/json.go:		var newDevice VirtioGPU
vendor/github.com/crc-org/vfkit/pkg/config/json.go:func (dev *VirtioGPU) MarshalJSON() ([]byte, error) {
vendor/github.com/crc-org/vfkit/pkg/config/json.go:		VirtioGPU
vendor/github.com/crc-org/vfkit/pkg/config/json.go:		jsonKind:  kind(vfGpu),
vendor/github.com/crc-org/vfkit/pkg/config/json.go:		VirtioGPU: *dev,
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:	// Options for VirtioGPUResolution
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:	VirtioGPUResolutionWidth  = "width"
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:	VirtioGPUResolutionHeight = "height"
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:	// Default VirtioGPU Resolution
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:	defaultVirtioGPUResolutionWidth  = 800
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:	defaultVirtioGPUResolutionHeight = 600
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:type VirtioGPUResolution struct {
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:// VirtioGPU configures a GPU device, such as the host computer's display
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:type VirtioGPU struct {
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:	VirtioGPUResolution
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:	case "virtio-gpu":
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:		dev = &VirtioGPU{}
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:// VirtioGPUNew creates a new gpu device for the virtual machine.
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:func VirtioGPUNew() (VirtioDevice, error) {
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:	return &VirtioGPU{
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:		VirtioGPUResolution: VirtioGPUResolution{
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:			Width:  defaultVirtioGPUResolutionWidth,
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:			Height: defaultVirtioGPUResolutionHeight,
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:func (dev *VirtioGPU) validate() error {
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:		return fmt.Errorf("invalid dimensions for virtio-gpu device resolution: %dx%d", dev.Width, dev.Height)
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:func (dev *VirtioGPU) ToCmdLine() ([]string, error) {
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:	return []string{"--device", fmt.Sprintf("virtio-gpu,width=%d,height=%d", dev.Width, dev.Height)}, nil
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:func (dev *VirtioGPU) FromOptions(options []option) error {
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:		case VirtioGPUResolutionHeight:
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:				return fmt.Errorf(fmt.Sprintf("Invalid value for virtio-gpu %s: %s", option.key, option.value))
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:		case VirtioGPUResolutionWidth:
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:				return fmt.Errorf(fmt.Sprintf("Invalid value for virtio-gpu %s: %s", option.key, option.value))
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:			return fmt.Errorf("unknown option for virtio-gpu devices: %s", option.key)
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:		dev.Width = defaultVirtioGPUResolutionWidth
vendor/github.com/crc-org/vfkit/pkg/config/virtio.go:		dev.Height = defaultVirtioGPUResolutionHeight
vendor/github.com/crc-org/vfkit/pkg/config/config.go:func (vm *VirtualMachine) VirtioGPUDevices() []*VirtioGPU {
vendor/github.com/crc-org/vfkit/pkg/config/config.go:	gpuDevs := []*VirtioGPU{}
vendor/github.com/crc-org/vfkit/pkg/config/config.go:		if gpuDev, isVirtioGPU := dev.(*VirtioGPU); isVirtioGPU {
vendor/github.com/crc-org/vfkit/pkg/config/config.go:			gpuDevs = append(gpuDevs, gpuDev)
vendor/github.com/crc-org/vfkit/pkg/config/config.go:	return gpuDevs
vendor/github.com/power-devops/perfstat/helpers.go:	l.OpenClose = int64(n.open_close)
vendor/github.com/power-devops/perfstat/types_lvm.go:	OpenClose              int64  /* LVM_QLVOPEN, etc. (see lvm.h) */
vendor/github.com/ugorji/go/codec/fast-path.go.tmpl:			v = append(v, {{ zerocmd .Elem }})
vendor/github.com/ugorji/go/codec/gen.go:	funcs["zerocmd"] = genInternalZeroValue
vendor/github.com/ugorji/go/codec/gen.go:	funcs["nonzerocmd"] = genInternalNonZeroValue
vendor/github.com/ugorji/go/codec/mammoth-test.go.tmpl:    for _, v := range [][]{{ .Elem }}{ nil, {}, { {{ nonzerocmd .Elem }}, {{ zerocmd .Elem }}, {{ zerocmd .Elem }}, {{ nonzerocmd .Elem }} } } {
vendor/github.com/ugorji/go/codec/mammoth-test.go.tmpl:    for _, v := range []map[{{ .MapKey }}]{{ .Elem }}{ nil, {}, { {{ nonzerocmd .MapKey }}:{{ zerocmd .Elem }} {{if ne "bool" .MapKey}}, {{ nonzerocmd .MapKey }}:{{ nonzerocmd .Elem }} {{end}} } } {
vendor/github.com/davecgh/go-spew/spew/bypass.go:	flagPublic := *flagField(&vA)
vendor/github.com/davecgh/go-spew/spew/bypass.go:	flagRO = flagPublic ^ flagWithRO
vendor/github.com/cyphar/filepath-securejoin/procfs_linux.go:func newPrivateProcMount() (*os.File, error) {
vendor/github.com/cyphar/filepath-securejoin/procfs_linux.go:func clonePrivateProcMount() (_ *os.File, Err error) {
vendor/github.com/cyphar/filepath-securejoin/procfs_linux.go:	procRoot, err := newPrivateProcMount()
vendor/github.com/cyphar/filepath-securejoin/procfs_linux.go:		procRoot, err = clonePrivateProcMount()
vendor/github.com/shirou/gopsutil/v4/internal/common/common.go:func HostProcMountInfoWithContext(ctx context.Context, combineWith ...string) string {
vendor/github.com/shirou/gopsutil/v4/cpu/cpu_linux.go:					c.VendorID = "NVIDIA"
vendor/github.com/shirou/gopsutil/v4/common/env.go:	HostProcMountinfo EnvKeyType = "HOST_PROC_MOUNTINFO"
vendor/github.com/Microsoft/hcsshim/internal/hcs/schema2/device.go:	GPUMirror        DeviceType = "GpuMirror"
vendor/github.com/docker/docker/api/types/container/hostconfig.go:// Used by GPU device drivers.
vendor/github.com/docker/docker/api/types/container/hostconfig.go:	Capabilities [][]string        // An OR list of AND lists of device capabilities (e.g. "gpu")
vendor/github.com/docker/docker/api/types/swarm/task.go:// "Kind" is used to describe the Kind of a resource (e.g: "GPU", "FPGA", "SSD", ...)
vendor/github.com/docker/docker/api/types/swarm/task.go:// Value is used to identify the resource (GPU="UUID-1", FPGA="/dev/sdb5", ...)
vendor/github.com/docker/docker/api/types/swarm/task.go:// "Kind" is used to describe the Kind of a resource (e.g: "GPU", "FPGA", "SSD", ...)
vendor/github.com/docker/docker/api/types/swarm/network.go:	PublishMode PortConfigPublishMode `json:",omitempty"`
vendor/github.com/docker/docker/api/types/swarm/network.go:// PortConfigPublishMode represents the mode in which the port is to
vendor/github.com/docker/docker/api/types/swarm/network.go:type PortConfigPublishMode string
vendor/github.com/docker/docker/api/types/swarm/network.go:	// PortConfigPublishModeIngress is used for ports published
vendor/github.com/docker/docker/api/types/swarm/network.go:	PortConfigPublishModeIngress PortConfigPublishMode = "ingress"
vendor/github.com/docker/docker/api/types/swarm/network.go:	// PortConfigPublishModeHost is used for ports published
vendor/github.com/docker/docker/api/types/swarm/network.go:	PortConfigPublishModeHost PortConfigPublishMode = "host"
vendor/github.com/docker/docker/api/swagger.yaml:        example: "nvidia"
vendor/github.com/docker/docker/api/swagger.yaml:          - "GPU-fef8089b-4820-abfc-e83e-94318197576e"
vendor/github.com/docker/docker/api/swagger.yaml:          # gpu AND nvidia AND compute
vendor/github.com/docker/docker/api/swagger.yaml:          - ["gpu", "nvidia", "compute"]
vendor/github.com/docker/docker/api/swagger.yaml:      String resources (e.g, `GPU=UUID1`).
vendor/github.com/docker/docker/api/swagger.yaml:          Kind: "GPU"
vendor/github.com/docker/docker/api/swagger.yaml:          Kind: "GPU"
vendor/github.com/docker/docker/api/swagger.yaml:            Kind: "GPU"
vendor/github.com/docker/docker/api/swagger.yaml:            Kind: "GPU"
vendor/github.com/docker/docker/api/swagger.yaml:                  - Driver: "nvidia"
vendor/github.com/docker/docker/api/swagger.yaml:                    DeviceIDs": ["0", "1", "GPU-fef8089b-4820-abfc-e83e-94318197576e"]
vendor/github.com/docker/docker/api/swagger.yaml:                    Capabilities: [["gpu", "nvidia", "compute"]]
vendor/github.com/docker/docker/api/swagger.yaml:                  - Driver: "nvidia"
vendor/github.com/docker/docker/api/swagger.yaml:                    DeviceIDs": ["0", "1", "GPU-fef8089b-4820-abfc-e83e-94318197576e"]
vendor/github.com/docker/docker/api/swagger.yaml:                    Capabilities: [["gpu", "nvidia", "compute"]]
vendor/github.com/docker/docker/AUTHORS:Evan Lezar <elezar@nvidia.com>
vendor/github.com/docker/docker/AUTHORS:Felix Abecassis <fabecassis@nvidia.com>
vendor/github.com/docker/docker/AUTHORS:Mageee <fangpuyi@foxmail.com>
vendor/github.com/docker/docker/AUTHORS:Renaud Gaubert <rgaubert@nvidia.com>
vendor/github.com/containers/libhvee/pkg/hypervctl/summary.go:	SummaryRequestAllocatedGPU                     = 8
vendor/github.com/containers/libhvee/pkg/hypervctl/summary.go:		SummaryRequestAllocatedGPU,
vendor/github.com/containers/libhvee/pkg/hypervctl/summary.go:	AllocatedGPU                    string
vendor/github.com/containers/ocicrypt/config/constructors.go:func EncryptWithGpg(gpgRecipients [][]byte, gpgPubRingFile []byte) (CryptoConfig, error) {
vendor/github.com/containers/ocicrypt/config/constructors.go:		"gpg-pubkeyringfile": {gpgPubRingFile},
vendor/github.com/containers/ocicrypt/gpg.go:	// ReadGPGPubRingFile gets the byte sequence of the gpg public keyring
vendor/github.com/containers/ocicrypt/gpg.go:	ReadGPGPubRingFile() ([]byte, error)
vendor/github.com/containers/ocicrypt/gpg.go:// ReadGPGPubRingFile reads the GPG public key ring file
vendor/github.com/containers/ocicrypt/gpg.go:func (gc *gpgv2Client) ReadGPGPubRingFile() ([]byte, error) {
vendor/github.com/containers/ocicrypt/gpg.go:// ReadGPGPubRingFile reads the GPG public key ring file
vendor/github.com/containers/ocicrypt/gpg.go:func (gc *gpgv1Client) ReadGPGPubRingFile() ([]byte, error) {
vendor/github.com/containers/ocicrypt/helpers/parse_helpers.go:			gpgPubRingFile, err := gpgClient.ReadGPGPubRingFile()
vendor/github.com/containers/ocicrypt/helpers/parse_helpers.go:			gpgCc, err := encconfig.EncryptWithGpg(gpgRecipients, gpgPubRingFile)
vendor/github.com/containers/buildah/pkg/cli/common.go:	SbomImgPurlOutput   string
vendor/github.com/containers/buildah/pkg/cli/common.go:	fs.StringVar(&flags.SbomImgPurlOutput, "sbom-image-purl-output", "", "add scan results to image as `path`")
vendor/github.com/containers/buildah/run_linux.go:		procMount := specs.Mount{
vendor/github.com/containers/buildah/run_linux.go:		mounts = addOrReplaceMount(mounts, procMount)
vendor/github.com/containers/buildah/docker/AUTHORS:Felix Abecassis <fabecassis@nvidia.com>
vendor/github.com/containers/common/pkg/config/config.go:	// for GPU passthrough.
vendor/github.com/containers/common/pkg/config/containers.conf:#              for sharing GPU with the machine.
vendor/github.com/containers/common/pkg/seccomp/seccomp.json:				"rt_sigprocmask",
vendor/github.com/containers/common/pkg/seccomp/seccomp.json:				"sigprocmask",
vendor/github.com/containers/common/pkg/seccomp/default_linux.go:				"rt_sigprocmask",
vendor/github.com/containers/common/pkg/seccomp/default_linux.go:				"sigprocmask",
vendor/github.com/opencontainers/runtime-tools/generate/seccomp/seccomp_default.go:				"rt_sigprocmask",
vendor/golang.org/x/text/internal/language/tables.go:	"MQTQMRRTMSSRMTLTMUUSMVDVMWWIMXEXMYYSMZOZNAAMNCCLNEERNFFKNGGANHHBNIICNLLD" +
vendor/golang.org/x/net/html/entity.go:	"cudarrl;":                         '\U00002938',
vendor/golang.org/x/net/html/entity.go:	"cudarrr;":                         '\U00002935',
vendor/golang.org/x/net/http2/frame.go:	FlagPushPromiseEndHeaders Flags = 0x4
vendor/golang.org/x/net/http2/frame.go:	FlagPushPromisePadded     Flags = 0x8
vendor/golang.org/x/net/http2/frame.go:		FlagPushPromiseEndHeaders: "END_HEADERS",
vendor/golang.org/x/net/http2/frame.go:		FlagPushPromisePadded:     "PADDED",
vendor/golang.org/x/net/http2/frame.go:	return f.FrameHeader.Flags.Has(FlagPushPromiseEndHeaders)
vendor/golang.org/x/net/http2/frame.go:	if fh.Flags.Has(FlagPushPromisePadded) {
vendor/golang.org/x/net/http2/frame.go:		flags |= FlagPushPromisePadded
vendor/golang.org/x/net/http2/frame.go:		flags |= FlagPushPromiseEndHeaders
vendor/golang.org/x/sys/windows/zerrors_windows.go:	ERROR_GRAPHICS_GPU_EXCEPTION_ON_DEVICE                                    Handle        = 0xC0262200
vendor/golang.org/x/sys/windows/zerrors_windows.go:	STATUS_GRAPHICS_GPU_EXCEPTION_ON_DEVICE                                   NTStatus      = 0xC01E0200
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	procMakeAbsoluteSD                                       = modadvapi32.NewProc("MakeAbsoluteSD")
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	procMakeSelfRelativeSD                                   = modadvapi32.NewProc("MakeSelfRelativeSD")
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	procMapViewOfFile                                        = modkernel32.NewProc("MapViewOfFile")
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	procModule32FirstW                                       = modkernel32.NewProc("Module32FirstW")
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	procModule32NextW                                        = modkernel32.NewProc("Module32NextW")
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	procMoveFileExW                                          = modkernel32.NewProc("MoveFileExW")
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	procMoveFileW                                            = modkernel32.NewProc("MoveFileW")
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	procMultiByteToWideChar                                  = modkernel32.NewProc("MultiByteToWideChar")
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	procMessageBoxW                                          = moduser32.NewProc("MessageBoxW")
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	r1, _, e1 := syscall.Syscall12(procMakeAbsoluteSD.Addr(), 11, uintptr(unsafe.Pointer(selfRelativeSD)), uintptr(unsafe.Pointer(absoluteSD)), uintptr(unsafe.Pointer(absoluteSDSize)), uintptr(unsafe.Pointer(dacl)), uintptr(unsafe.Pointer(daclSize)), uintptr(unsafe.Pointer(sacl)), uintptr(unsafe.Pointer(saclSize)), uintptr(unsafe.Pointer(owner)), uintptr(unsafe.Pointer(ownerSize)), uintptr(unsafe.Pointer(group)), uintptr(unsafe.Pointer(groupSize)), 0)
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	r1, _, e1 := syscall.Syscall(procMakeSelfRelativeSD.Addr(), 3, uintptr(unsafe.Pointer(absoluteSD)), uintptr(unsafe.Pointer(selfRelativeSD)), uintptr(unsafe.Pointer(selfRelativeSDSize)))
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	r0, _, e1 := syscall.Syscall6(procMapViewOfFile.Addr(), 5, uintptr(handle), uintptr(access), uintptr(offsetHigh), uintptr(offsetLow), uintptr(length), 0)
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	r1, _, e1 := syscall.Syscall(procModule32FirstW.Addr(), 2, uintptr(snapshot), uintptr(unsafe.Pointer(moduleEntry)), 0)
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	r1, _, e1 := syscall.Syscall(procModule32NextW.Addr(), 2, uintptr(snapshot), uintptr(unsafe.Pointer(moduleEntry)), 0)
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	r1, _, e1 := syscall.Syscall(procMoveFileExW.Addr(), 3, uintptr(unsafe.Pointer(from)), uintptr(unsafe.Pointer(to)), uintptr(flags))
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	r1, _, e1 := syscall.Syscall(procMoveFileW.Addr(), 2, uintptr(unsafe.Pointer(from)), uintptr(unsafe.Pointer(to)), 0)
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	r0, _, e1 := syscall.Syscall6(procMultiByteToWideChar.Addr(), 6, uintptr(codePage), uintptr(dwFlags), uintptr(unsafe.Pointer(str)), uintptr(nstr), uintptr(unsafe.Pointer(wchar)), uintptr(nwchar))
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	r0, _, e1 := syscall.Syscall6(procMessageBoxW.Addr(), 4, uintptr(hwnd), uintptr(unsafe.Pointer(text)), uintptr(unsafe.Pointer(caption)), uintptr(boxtype), 0, 0)
vendor/golang.org/x/sys/unix/zsysnum_freebsd_386.go:	SYS_SIGPROCMASK              = 340 // { int sigprocmask(int how, const sigset_t *set, sigset_t *oset); }
vendor/golang.org/x/sys/unix/zsysnum_freebsd_arm64.go:	SYS_SIGPROCMASK              = 340 // { int sigprocmask(int how, const sigset_t *set, sigset_t *oset); }
vendor/golang.org/x/sys/unix/zsysnum_linux_ppc64le.go:	SYS_SIGPROCMASK             = 126
vendor/golang.org/x/sys/unix/zsysnum_linux_ppc64le.go:	SYS_RT_SIGPROCMASK          = 174
vendor/golang.org/x/sys/unix/zsysnum_linux_mips64.go:	SYS_RT_SIGPROCMASK          = 5014
vendor/golang.org/x/sys/unix/zsysnum_linux_ppc.go:	SYS_SIGPROCMASK                  = 126
vendor/golang.org/x/sys/unix/zsysnum_linux_ppc.go:	SYS_RT_SIGPROCMASK               = 174
vendor/golang.org/x/sys/unix/zsysnum_linux_mips.go:	SYS_SIGPROCMASK                  = 4126
vendor/golang.org/x/sys/unix/zsysnum_linux_mips.go:	SYS_RT_SIGPROCMASK               = 4195
vendor/golang.org/x/sys/unix/zsysnum_openbsd_amd64.go:	SYS_SIGPROCMASK    = 48  // { int sys_sigprocmask(int how, sigset_t mask); }
vendor/golang.org/x/sys/unix/zsysnum_linux_riscv64.go:	SYS_RT_SIGPROCMASK          = 135
vendor/golang.org/x/sys/unix/zsysnum_openbsd_arm.go:	SYS_SIGPROCMASK    = 48  // { int sys_sigprocmask(int how, sigset_t mask); }
vendor/golang.org/x/sys/unix/zsysnum_openbsd_riscv64.go:	SYS_SIGPROCMASK    = 48  // { int sys_sigprocmask(int how, sigset_t mask); }
vendor/golang.org/x/sys/unix/zsysnum_linux_mipsle.go:	SYS_SIGPROCMASK                  = 4126
vendor/golang.org/x/sys/unix/zsysnum_linux_mipsle.go:	SYS_RT_SIGPROCMASK               = 4195
vendor/golang.org/x/sys/unix/zsysnum_freebsd_amd64.go:	SYS_SIGPROCMASK              = 340 // { int sigprocmask(int how, const sigset_t *set, sigset_t *oset); }
vendor/golang.org/x/sys/unix/zsysnum_openbsd_386.go:	SYS_SIGPROCMASK    = 48  // { int sys_sigprocmask(int how, sigset_t mask); }
vendor/golang.org/x/sys/unix/zsysnum_zos_s390x.go:	SYS_SIGPROCMASK                     = 0x1C5 // 453
vendor/golang.org/x/sys/unix/zsysnum_linux_loong64.go:	SYS_RT_SIGPROCMASK          = 135
vendor/golang.org/x/sys/unix/zsysnum_openbsd_mips64.go:	SYS_SIGPROCMASK    = 48  // { int sys_sigprocmask(int how, sigset_t mask); }
vendor/golang.org/x/sys/unix/zsysnum_darwin_amd64.go:	SYS_SIGPROCMASK                    = 48
vendor/golang.org/x/sys/unix/zsysnum_linux_amd64.go:	SYS_RT_SIGPROCMASK          = 14
vendor/golang.org/x/sys/unix/zsyscall_linux.go:func rtSigprocmask(how int, set *Sigset_t, oldset *Sigset_t, sigsetsize uintptr) (err error) {
vendor/golang.org/x/sys/unix/zsyscall_linux.go:	_, _, e1 := RawSyscall6(SYS_RT_SIGPROCMASK, uintptr(how), uintptr(unsafe.Pointer(set)), uintptr(unsafe.Pointer(oldset)), uintptr(sigsetsize), 0, 0)
vendor/golang.org/x/sys/unix/zsysnum_linux_386.go:	SYS_SIGPROCMASK                  = 126
vendor/golang.org/x/sys/unix/zsysnum_linux_386.go:	SYS_RT_SIGPROCMASK               = 175
vendor/golang.org/x/sys/unix/zsysnum_freebsd_arm.go:	SYS_SIGPROCMASK              = 340 // { int sigprocmask(int how, const sigset_t *set, sigset_t *oset); }
vendor/golang.org/x/sys/unix/zsysnum_linux_ppc64.go:	SYS_SIGPROCMASK             = 126
vendor/golang.org/x/sys/unix/zsysnum_linux_ppc64.go:	SYS_RT_SIGPROCMASK          = 174
vendor/golang.org/x/sys/unix/zsysnum_linux_arm64.go:	SYS_RT_SIGPROCMASK          = 135
vendor/golang.org/x/sys/unix/zsysnum_linux_mips64le.go:	SYS_RT_SIGPROCMASK          = 5014
vendor/golang.org/x/sys/unix/zerrors_zos_s390x.go:	SO_IGNOREINCOMINGPUSH           = 0x1
vendor/golang.org/x/sys/unix/zsysnum_openbsd_arm64.go:	SYS_SIGPROCMASK    = 48  // { int sys_sigprocmask(int how, sigset_t mask); }
vendor/golang.org/x/sys/unix/syscall_linux.go://sysnb	rtSigprocmask(how int, set *Sigset_t, oldset *Sigset_t, sigsetsize uintptr) (err error) = SYS_RT_SIGPROCMASK
vendor/golang.org/x/sys/unix/syscall_linux.go:	return rtSigprocmask(how, set, oldset, _C__NSIG/8)
vendor/golang.org/x/sys/unix/zsysnum_darwin_arm64.go:	SYS_SIGPROCMASK                    = 48
vendor/golang.org/x/sys/unix/zsysnum_freebsd_riscv64.go:	SYS_SIGPROCMASK              = 340 // { int sigprocmask(int how, const sigset_t *set, sigset_t *oset); }
vendor/golang.org/x/sys/unix/zsysnum_linux_sparc64.go:	SYS_RT_SIGPROCMASK          = 103
vendor/golang.org/x/sys/unix/zsysnum_linux_sparc64.go:	SYS_SIGPROCMASK             = 220
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMadvise libc_madvise
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMkdir libc_mkdir
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMkdirat libc_mkdirat
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMkfifo libc_mkfifo
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMkfifoat libc_mkfifoat
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMknod libc_mknod
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMknodat libc_mknodat
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMlock libc_mlock
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMlockall libc_mlockall
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMprotect libc_mprotect
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMsync libc_msync
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMunlock libc_munlock
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMunlockall libc_munlockall
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procmmap libc_mmap
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procmunmap libc_munmap
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMadvise,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMkdir,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMkdirat,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMkfifo,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMkfifoat,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMknod,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMknodat,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMlock,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMlockall,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMprotect,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMsync,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMunlock,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMunlockall,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procmmap,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procmunmap,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMadvise)), 3, uintptr(unsafe.Pointer(_p0)), uintptr(len(b)), uintptr(advice), 0, 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMkdir)), 2, uintptr(unsafe.Pointer(_p0)), uintptr(mode), 0, 0, 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMkdirat)), 3, uintptr(dirfd), uintptr(unsafe.Pointer(_p0)), uintptr(mode), 0, 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMkfifo)), 2, uintptr(unsafe.Pointer(_p0)), uintptr(mode), 0, 0, 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMkfifoat)), 3, uintptr(dirfd), uintptr(unsafe.Pointer(_p0)), uintptr(mode), 0, 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMknod)), 3, uintptr(unsafe.Pointer(_p0)), uintptr(mode), uintptr(dev), 0, 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMknodat)), 4, uintptr(dirfd), uintptr(unsafe.Pointer(_p0)), uintptr(mode), uintptr(dev), 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMlock)), 2, uintptr(unsafe.Pointer(_p0)), uintptr(len(b)), 0, 0, 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMlockall)), 1, uintptr(flags), 0, 0, 0, 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMprotect)), 3, uintptr(unsafe.Pointer(_p0)), uintptr(len(b)), uintptr(prot), 0, 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMsync)), 3, uintptr(unsafe.Pointer(_p0)), uintptr(len(b)), uintptr(flags), 0, 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMunlock)), 2, uintptr(unsafe.Pointer(_p0)), uintptr(len(b)), 0, 0, 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMunlockall)), 0, 0, 0, 0, 0, 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	r0, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procmmap)), 6, uintptr(addr), uintptr(length), uintptr(prot), uintptr(flag), uintptr(fd), uintptr(pos))
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procmunmap)), 2, uintptr(addr), uintptr(length), 0, 0, 0, 0)
vendor/golang.org/x/sys/unix/zsysnum_openbsd_ppc64.go:	SYS_SIGPROCMASK    = 48  // { int sys_sigprocmask(int how, sigset_t mask); }
vendor/golang.org/x/sys/unix/zsysnum_linux_s390x.go:	SYS_SIGPROCMASK             = 126
vendor/golang.org/x/sys/unix/zsysnum_linux_s390x.go:	SYS_RT_SIGPROCMASK          = 175
vendor/golang.org/x/sys/unix/zsysnum_dragonfly_amd64.go:	SYS_SIGPROCMASK            = 340 // { int sigprocmask(int how, const sigset_t *set, sigset_t *oset); }
vendor/golang.org/x/sys/unix/zsysnum_linux_arm.go:	SYS_SIGPROCMASK                  = 126
vendor/golang.org/x/sys/unix/zsysnum_linux_arm.go:	SYS_RT_SIGPROCMASK               = 175
vendor/go.opentelemetry.io/otel/semconv/v1.24.0/metric.go:	// MessagingPublishDuration is the metric conforming to the
vendor/go.opentelemetry.io/otel/semconv/v1.24.0/metric.go:	MessagingPublishDurationName        = "messaging.publish.duration"
vendor/go.opentelemetry.io/otel/semconv/v1.24.0/metric.go:	MessagingPublishDurationUnit        = "s"
vendor/go.opentelemetry.io/otel/semconv/v1.24.0/metric.go:	MessagingPublishDurationDescription = "Measures the duration of publish operation."
vendor/go.opentelemetry.io/otel/semconv/v1.24.0/metric.go:	// MessagingPublishMessages is the metric conforming to the
vendor/go.opentelemetry.io/otel/semconv/v1.24.0/metric.go:	MessagingPublishMessagesName        = "messaging.publish.messages"
vendor/go.opentelemetry.io/otel/semconv/v1.24.0/metric.go:	MessagingPublishMessagesUnit        = "{message}"
vendor/go.opentelemetry.io/otel/semconv/v1.24.0/metric.go:	MessagingPublishMessagesDescription = "Measures the number of published messages."
docs/source/markdown/podman-pod-clone.1.md.in:@@option gpus
docs/source/markdown/podman-pod-create.1.md.in:@@option gpus
docs/source/markdown/podman-run.1.md.in:@@option gpus
docs/source/markdown/podman-run.1.md.in:$ podman run --mount type=glob,src=/usr/lib64/libnvidia\*,ro=true -i -t fedora /bin/bash
docs/source/markdown/podman-create.1.md.in:@@option gpus
docs/source/markdown/podman-create.1.md.in:$ podman create --mount type=glob,src=/usr/lib64/libnvidia\*,ro -i -t fedora /bin/bash
docs/source/markdown/options/gpus.md:#### **--gpus**=*ENTRY*
docs/source/markdown/options/gpus.md:GPU devices to add to the container ('all' to pass all GPUs) Currently only
docs/source/markdown/options/gpus.md:Nvidia devices are supported.
docs/source/locale/ja/LC_MESSAGES/markdown.po:msgid "securityContext\\.procMount"
docs/kubernetes_support.md:| securityContext\.procMount                          | ✅      |
pkg/k8s.io/api/core/v1/types.go:	// procMount denotes the type of proc mount to use for the containers.
pkg/k8s.io/api/core/v1/types.go:	// The default is DefaultProcMount which uses the container runtime defaults for
pkg/k8s.io/api/core/v1/types.go:	// This requires the ProcMountType feature flag to be enabled.
pkg/k8s.io/api/core/v1/types.go:	ProcMount *ProcMountType `json:"procMount,omitempty"`
pkg/k8s.io/api/core/v1/types.go:type ProcMountType string
pkg/k8s.io/api/core/v1/types.go:	// DefaultProcMount uses the container runtime defaults for readonly and masked
pkg/k8s.io/api/core/v1/types.go:	DefaultProcMount ProcMountType = "Default"
pkg/k8s.io/api/core/v1/types.go:	// UnmaskedProcMount bypasses the default masking behavior of the container
pkg/k8s.io/api/core/v1/types.go:	UnmaskedProcMount ProcMountType = "Unmasked"
pkg/rootless/rootless_linux.c:  if (sigprocmask (SIG_BLOCK, &sigset, &oldsigset) < 0)
pkg/rootless/rootless_linux.c:  if (sigprocmask (SIG_SETMASK, &oldsigset, NULL) < 0)
pkg/rootless/rootless_linux.c:  if (sigprocmask (SIG_BLOCK, &sigset, &oldsigset) < 0)
pkg/rootless/rootless_linux.c:  if (sigprocmask (SIG_SETMASK, &oldsigset, NULL) < 0)
pkg/specgen/generate/oci_linux.go:		procMount := spec.Mount{
pkg/specgen/generate/oci_linux.go:		g.AddMount(procMount)
pkg/specgen/generate/kube/kube.go:	if securityContext.ProcMount != nil && *securityContext.ProcMount == v1.UnmaskedProcMount {
pkg/specgenutil/specgen.go:	for _, gpu := range c.GPUs {
pkg/specgenutil/specgen.go:		devices = append(devices, "nvidia.com/gpu="+gpu)
pkg/machine/apple/vfkit.go:	gpu, err := vfConfig.VirtioGPUNew()
pkg/machine/apple/vfkit.go:	return append(devices, gpu, mouse, kb), nil
pkg/domain/entities/pods.go:	GPUs                 []string
test/tools/go.sum:dmitri.shuralyov.com/gpu/mtl v0.0.0-20190408044501-666a987793e9/go.mod h1:H6x//7gZCb22OMCxBHrMx7a5I7Hp++hsVxbQ4BYO7hU=
test/tools/go.sum:github.com/imdario/mergo v0.3.11/go.mod h1:jmQim1M+e3UYxmgPu/WyfjB3N3VflVyUjjjwH0dnCYA=
test/tools/go.sum:github.com/imdario/mergo v0.3.12/go.mod h1:jmQim1M+e3UYxmgPu/WyfjB3N3VflVyUjjjwH0dnCYA=
test/tools/vendor/github.com/google/pprof/profile/legacy_profile.go:	procMapsRE  = regexp.MustCompile(`^` + cHexRange + cPerm + cSpaceHex + hexPair + spaceDigits + cSpaceString)
test/tools/vendor/github.com/google/pprof/profile/legacy_profile.go:// ParseProcMaps parses a memory map in the format of /proc/self/maps.
test/tools/vendor/github.com/google/pprof/profile/legacy_profile.go:func ParseProcMaps(rd io.Reader) ([]*Mapping, error) {
test/tools/vendor/github.com/google/pprof/profile/legacy_profile.go:	return parseProcMapsFromScanner(s)
test/tools/vendor/github.com/google/pprof/profile/legacy_profile.go:func parseProcMapsFromScanner(s *bufio.Scanner) ([]*Mapping, error) {
test/tools/vendor/github.com/google/pprof/profile/legacy_profile.go:	mapping, err := parseProcMapsFromScanner(s)
test/tools/vendor/github.com/google/pprof/profile/legacy_profile.go:	if me := procMapsRE.FindStringSubmatch(l); len(me) == 6 {
test/tools/vendor/github.com/russross/blackfriday/v2/entities.go:	"&cudarrl;":                         true,
test/tools/vendor/github.com/russross/blackfriday/v2/entities.go:	"&cudarrr;":                         true,
test/tools/vendor/golang.org/x/tools/internal/stdlib/manifest.go:		{"EM_AMDGPU", Const, 11},
test/tools/vendor/golang.org/x/tools/internal/stdlib/manifest.go:		{"EM_CUDA", Const, 11},
test/tools/vendor/golang.org/x/tools/internal/stdlib/manifest.go:		{"SYS_RT_SIGPROCMASK", Const, 0},
test/tools/vendor/golang.org/x/tools/internal/stdlib/manifest.go:		{"SYS_SIGPROCMASK", Const, 0},
test/tools/vendor/golang.org/x/tools/internal/pkgbits/sync.go:	SyncCloseScope
test/tools/vendor/golang.org/x/tools/internal/pkgbits/sync.go:	SyncCloseAnotherScope
test/tools/vendor/golang.org/x/tools/internal/pkgbits/syncmarker_string.go:	_ = x[SyncCloseScope-45]
test/tools/vendor/golang.org/x/tools/internal/pkgbits/syncmarker_string.go:	_ = x[SyncCloseAnotherScope-46]
test/tools/vendor/golang.org/x/sys/windows/zerrors_windows.go:	ERROR_GRAPHICS_GPU_EXCEPTION_ON_DEVICE                                    Handle        = 0xC0262200
test/tools/vendor/golang.org/x/sys/windows/zerrors_windows.go:	STATUS_GRAPHICS_GPU_EXCEPTION_ON_DEVICE                                   NTStatus      = 0xC01E0200
test/tools/vendor/golang.org/x/sys/windows/zsyscall_windows.go:	procMakeAbsoluteSD                                       = modadvapi32.NewProc("MakeAbsoluteSD")
test/tools/vendor/golang.org/x/sys/windows/zsyscall_windows.go:	procMakeSelfRelativeSD                                   = modadvapi32.NewProc("MakeSelfRelativeSD")
test/tools/vendor/golang.org/x/sys/windows/zsyscall_windows.go:	procMapViewOfFile                                        = modkernel32.NewProc("MapViewOfFile")
test/tools/vendor/golang.org/x/sys/windows/zsyscall_windows.go:	procModule32FirstW                                       = modkernel32.NewProc("Module32FirstW")
test/tools/vendor/golang.org/x/sys/windows/zsyscall_windows.go:	procModule32NextW                                        = modkernel32.NewProc("Module32NextW")
test/tools/vendor/golang.org/x/sys/windows/zsyscall_windows.go:	procMoveFileExW                                          = modkernel32.NewProc("MoveFileExW")
test/tools/vendor/golang.org/x/sys/windows/zsyscall_windows.go:	procMoveFileW                                            = modkernel32.NewProc("MoveFileW")
test/tools/vendor/golang.org/x/sys/windows/zsyscall_windows.go:	procMultiByteToWideChar                                  = modkernel32.NewProc("MultiByteToWideChar")
test/tools/vendor/golang.org/x/sys/windows/zsyscall_windows.go:	procMessageBoxW                                          = moduser32.NewProc("MessageBoxW")
test/tools/vendor/golang.org/x/sys/windows/zsyscall_windows.go:	r1, _, e1 := syscall.Syscall12(procMakeAbsoluteSD.Addr(), 11, uintptr(unsafe.Pointer(selfRelativeSD)), uintptr(unsafe.Pointer(absoluteSD)), uintptr(unsafe.Pointer(absoluteSDSize)), uintptr(unsafe.Pointer(dacl)), uintptr(unsafe.Pointer(daclSize)), uintptr(unsafe.Pointer(sacl)), uintptr(unsafe.Pointer(saclSize)), uintptr(unsafe.Pointer(owner)), uintptr(unsafe.Pointer(ownerSize)), uintptr(unsafe.Pointer(group)), uintptr(unsafe.Pointer(groupSize)), 0)
test/tools/vendor/golang.org/x/sys/windows/zsyscall_windows.go:	r1, _, e1 := syscall.Syscall(procMakeSelfRelativeSD.Addr(), 3, uintptr(unsafe.Pointer(absoluteSD)), uintptr(unsafe.Pointer(selfRelativeSD)), uintptr(unsafe.Pointer(selfRelativeSDSize)))
test/tools/vendor/golang.org/x/sys/windows/zsyscall_windows.go:	r0, _, e1 := syscall.Syscall6(procMapViewOfFile.Addr(), 5, uintptr(handle), uintptr(access), uintptr(offsetHigh), uintptr(offsetLow), uintptr(length), 0)
test/tools/vendor/golang.org/x/sys/windows/zsyscall_windows.go:	r1, _, e1 := syscall.Syscall(procModule32FirstW.Addr(), 2, uintptr(snapshot), uintptr(unsafe.Pointer(moduleEntry)), 0)
test/tools/vendor/golang.org/x/sys/windows/zsyscall_windows.go:	r1, _, e1 := syscall.Syscall(procModule32NextW.Addr(), 2, uintptr(snapshot), uintptr(unsafe.Pointer(moduleEntry)), 0)
test/tools/vendor/golang.org/x/sys/windows/zsyscall_windows.go:	r1, _, e1 := syscall.Syscall(procMoveFileExW.Addr(), 3, uintptr(unsafe.Pointer(from)), uintptr(unsafe.Pointer(to)), uintptr(flags))
test/tools/vendor/golang.org/x/sys/windows/zsyscall_windows.go:	r1, _, e1 := syscall.Syscall(procMoveFileW.Addr(), 2, uintptr(unsafe.Pointer(from)), uintptr(unsafe.Pointer(to)), 0)
test/tools/vendor/golang.org/x/sys/windows/zsyscall_windows.go:	r0, _, e1 := syscall.Syscall6(procMultiByteToWideChar.Addr(), 6, uintptr(codePage), uintptr(dwFlags), uintptr(unsafe.Pointer(str)), uintptr(nstr), uintptr(unsafe.Pointer(wchar)), uintptr(nwchar))
test/tools/vendor/golang.org/x/sys/windows/zsyscall_windows.go:	r0, _, e1 := syscall.Syscall6(procMessageBoxW.Addr(), 4, uintptr(hwnd), uintptr(unsafe.Pointer(text)), uintptr(unsafe.Pointer(caption)), uintptr(boxtype), 0, 0)
test/tools/vendor/golang.org/x/sys/unix/zsysnum_freebsd_386.go:	SYS_SIGPROCMASK              = 340 // { int sigprocmask(int how, const sigset_t *set, sigset_t *oset); }
test/tools/vendor/golang.org/x/sys/unix/zsysnum_freebsd_arm64.go:	SYS_SIGPROCMASK              = 340 // { int sigprocmask(int how, const sigset_t *set, sigset_t *oset); }
test/tools/vendor/golang.org/x/sys/unix/zsysnum_linux_ppc64le.go:	SYS_SIGPROCMASK             = 126
test/tools/vendor/golang.org/x/sys/unix/zsysnum_linux_ppc64le.go:	SYS_RT_SIGPROCMASK          = 174
test/tools/vendor/golang.org/x/sys/unix/zsysnum_linux_mips64.go:	SYS_RT_SIGPROCMASK          = 5014
test/tools/vendor/golang.org/x/sys/unix/zsysnum_linux_ppc.go:	SYS_SIGPROCMASK                  = 126
test/tools/vendor/golang.org/x/sys/unix/zsysnum_linux_ppc.go:	SYS_RT_SIGPROCMASK               = 174
test/tools/vendor/golang.org/x/sys/unix/zsysnum_linux_mips.go:	SYS_SIGPROCMASK                  = 4126
test/tools/vendor/golang.org/x/sys/unix/zsysnum_linux_mips.go:	SYS_RT_SIGPROCMASK               = 4195
test/tools/vendor/golang.org/x/sys/unix/zsysnum_openbsd_amd64.go:	SYS_SIGPROCMASK    = 48  // { int sys_sigprocmask(int how, sigset_t mask); }
test/tools/vendor/golang.org/x/sys/unix/zsysnum_linux_riscv64.go:	SYS_RT_SIGPROCMASK          = 135
test/tools/vendor/golang.org/x/sys/unix/zsysnum_openbsd_arm.go:	SYS_SIGPROCMASK    = 48  // { int sys_sigprocmask(int how, sigset_t mask); }
test/tools/vendor/golang.org/x/sys/unix/zsysnum_openbsd_riscv64.go:	SYS_SIGPROCMASK    = 48  // { int sys_sigprocmask(int how, sigset_t mask); }
test/tools/vendor/golang.org/x/sys/unix/zsysnum_linux_mipsle.go:	SYS_SIGPROCMASK                  = 4126
test/tools/vendor/golang.org/x/sys/unix/zsysnum_linux_mipsle.go:	SYS_RT_SIGPROCMASK               = 4195
test/tools/vendor/golang.org/x/sys/unix/zsysnum_freebsd_amd64.go:	SYS_SIGPROCMASK              = 340 // { int sigprocmask(int how, const sigset_t *set, sigset_t *oset); }
test/tools/vendor/golang.org/x/sys/unix/zsysnum_openbsd_386.go:	SYS_SIGPROCMASK    = 48  // { int sys_sigprocmask(int how, sigset_t mask); }
test/tools/vendor/golang.org/x/sys/unix/zsysnum_zos_s390x.go:	SYS_SIGPROCMASK                     = 0x1C5 // 453
test/tools/vendor/golang.org/x/sys/unix/zsysnum_linux_loong64.go:	SYS_RT_SIGPROCMASK          = 135
test/tools/vendor/golang.org/x/sys/unix/zsysnum_openbsd_mips64.go:	SYS_SIGPROCMASK    = 48  // { int sys_sigprocmask(int how, sigset_t mask); }
test/tools/vendor/golang.org/x/sys/unix/zsysnum_darwin_amd64.go:	SYS_SIGPROCMASK                    = 48
test/tools/vendor/golang.org/x/sys/unix/zsysnum_linux_amd64.go:	SYS_RT_SIGPROCMASK          = 14
test/tools/vendor/golang.org/x/sys/unix/zsyscall_linux.go:func rtSigprocmask(how int, set *Sigset_t, oldset *Sigset_t, sigsetsize uintptr) (err error) {
test/tools/vendor/golang.org/x/sys/unix/zsyscall_linux.go:	_, _, e1 := RawSyscall6(SYS_RT_SIGPROCMASK, uintptr(how), uintptr(unsafe.Pointer(set)), uintptr(unsafe.Pointer(oldset)), uintptr(sigsetsize), 0, 0)
test/tools/vendor/golang.org/x/sys/unix/zsysnum_linux_386.go:	SYS_SIGPROCMASK                  = 126
test/tools/vendor/golang.org/x/sys/unix/zsysnum_linux_386.go:	SYS_RT_SIGPROCMASK               = 175
test/tools/vendor/golang.org/x/sys/unix/zsysnum_freebsd_arm.go:	SYS_SIGPROCMASK              = 340 // { int sigprocmask(int how, const sigset_t *set, sigset_t *oset); }
test/tools/vendor/golang.org/x/sys/unix/zsysnum_linux_ppc64.go:	SYS_SIGPROCMASK             = 126
test/tools/vendor/golang.org/x/sys/unix/zsysnum_linux_ppc64.go:	SYS_RT_SIGPROCMASK          = 174
test/tools/vendor/golang.org/x/sys/unix/zsysnum_linux_arm64.go:	SYS_RT_SIGPROCMASK          = 135
test/tools/vendor/golang.org/x/sys/unix/zsysnum_linux_mips64le.go:	SYS_RT_SIGPROCMASK          = 5014
test/tools/vendor/golang.org/x/sys/unix/zerrors_zos_s390x.go:	SO_IGNOREINCOMINGPUSH           = 0x1
test/tools/vendor/golang.org/x/sys/unix/zsysnum_openbsd_arm64.go:	SYS_SIGPROCMASK    = 48  // { int sys_sigprocmask(int how, sigset_t mask); }
test/tools/vendor/golang.org/x/sys/unix/syscall_linux.go://sysnb	rtSigprocmask(how int, set *Sigset_t, oldset *Sigset_t, sigsetsize uintptr) (err error) = SYS_RT_SIGPROCMASK
test/tools/vendor/golang.org/x/sys/unix/syscall_linux.go:	return rtSigprocmask(how, set, oldset, _C__NSIG/8)
test/tools/vendor/golang.org/x/sys/unix/zsysnum_darwin_arm64.go:	SYS_SIGPROCMASK                    = 48
test/tools/vendor/golang.org/x/sys/unix/zsysnum_freebsd_riscv64.go:	SYS_SIGPROCMASK              = 340 // { int sigprocmask(int how, const sigset_t *set, sigset_t *oset); }
test/tools/vendor/golang.org/x/sys/unix/zsysnum_linux_sparc64.go:	SYS_RT_SIGPROCMASK          = 103
test/tools/vendor/golang.org/x/sys/unix/zsysnum_linux_sparc64.go:	SYS_SIGPROCMASK             = 220
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMadvise libc_madvise
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMkdir libc_mkdir
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMkdirat libc_mkdirat
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMkfifo libc_mkfifo
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMkfifoat libc_mkfifoat
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMknod libc_mknod
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMknodat libc_mknodat
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMlock libc_mlock
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMlockall libc_mlockall
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMprotect libc_mprotect
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMsync libc_msync
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMunlock libc_munlock
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMunlockall libc_munlockall
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procmmap libc_mmap
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procmunmap libc_munmap
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMadvise,
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMkdir,
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMkdirat,
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMkfifo,
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMkfifoat,
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMknod,
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMknodat,
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMlock,
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMlockall,
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMprotect,
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMsync,
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMunlock,
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMunlockall,
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procmmap,
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procmunmap,
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMadvise)), 3, uintptr(unsafe.Pointer(_p0)), uintptr(len(b)), uintptr(advice), 0, 0, 0)
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMkdir)), 2, uintptr(unsafe.Pointer(_p0)), uintptr(mode), 0, 0, 0, 0)
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMkdirat)), 3, uintptr(dirfd), uintptr(unsafe.Pointer(_p0)), uintptr(mode), 0, 0, 0)
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMkfifo)), 2, uintptr(unsafe.Pointer(_p0)), uintptr(mode), 0, 0, 0, 0)
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMkfifoat)), 3, uintptr(dirfd), uintptr(unsafe.Pointer(_p0)), uintptr(mode), 0, 0, 0)
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMknod)), 3, uintptr(unsafe.Pointer(_p0)), uintptr(mode), uintptr(dev), 0, 0, 0)
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMknodat)), 4, uintptr(dirfd), uintptr(unsafe.Pointer(_p0)), uintptr(mode), uintptr(dev), 0, 0)
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMlock)), 2, uintptr(unsafe.Pointer(_p0)), uintptr(len(b)), 0, 0, 0, 0)
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMlockall)), 1, uintptr(flags), 0, 0, 0, 0, 0)
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMprotect)), 3, uintptr(unsafe.Pointer(_p0)), uintptr(len(b)), uintptr(prot), 0, 0, 0)
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMsync)), 3, uintptr(unsafe.Pointer(_p0)), uintptr(len(b)), uintptr(flags), 0, 0, 0)
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMunlock)), 2, uintptr(unsafe.Pointer(_p0)), uintptr(len(b)), 0, 0, 0, 0)
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMunlockall)), 0, 0, 0, 0, 0, 0, 0)
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	r0, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procmmap)), 6, uintptr(addr), uintptr(length), uintptr(prot), uintptr(flag), uintptr(fd), uintptr(pos))
test/tools/vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procmunmap)), 2, uintptr(addr), uintptr(length), 0, 0, 0, 0)
test/tools/vendor/golang.org/x/sys/unix/zsysnum_openbsd_ppc64.go:	SYS_SIGPROCMASK    = 48  // { int sys_sigprocmask(int how, sigset_t mask); }
test/tools/vendor/golang.org/x/sys/unix/zsysnum_linux_s390x.go:	SYS_SIGPROCMASK             = 126
test/tools/vendor/golang.org/x/sys/unix/zsysnum_linux_s390x.go:	SYS_RT_SIGPROCMASK          = 175
test/tools/vendor/golang.org/x/sys/unix/zsysnum_dragonfly_amd64.go:	SYS_SIGPROCMASK            = 340 // { int sigprocmask(int how, const sigset_t *set, sigset_t *oset); }
test/tools/vendor/golang.org/x/sys/unix/zsysnum_linux_arm.go:	SYS_SIGPROCMASK                  = 126
test/tools/vendor/golang.org/x/sys/unix/zsysnum_linux_arm.go:	SYS_RT_SIGPROCMASK               = 175
test/system/710-kube.bats:      assert "$(< $KUBE)" =~ "procMount: Unmasked" "Generated kube yaml should have procMount unmasked"
cmd/podman/common/create.go:		gpuFlagName := "gpus"
cmd/podman/common/create.go:		createFlags.StringSliceVar(&cf.GPUs, gpuFlagName, []string{}, "GPU devices to add to the container ('all' to pass all GPUs)")
cmd/podman/common/create.go:		_ = cmd.RegisterFlagCompletionFunc(gpuFlagName, completion.AutocompleteNone)
libpod/oci_conmon_exec_common.go:		if pipes.syncPipe != nil && !pipes.syncClosed {
libpod/oci_conmon_exec_common.go:			pipes.syncClosed = true
libpod/oci_conmon_exec_common.go:		if pipes.syncPipe != nil && !pipes.syncClosed {
libpod/oci_conmon_exec_common.go:			pipes.syncClosed = true
libpod/oci_conmon_exec_common.go:	syncClosed   bool
libpod/oci_conmon_exec_common.go:	if p.syncPipe != nil && !p.syncClosed {
libpod/oci_conmon_exec_common.go:		p.syncClosed = true
libpod/kube.go:		unmask := v1.UnmaskedProcMount
libpod/kube.go:		sc.ProcMount = &unmask

```
