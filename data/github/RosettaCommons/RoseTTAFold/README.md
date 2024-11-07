# https://github.com/RosettaCommons/RoseTTAFold

```console
network/SE3_network.py:    @torch.cuda.amp.autocast(enabled=False)
network/SE3_network.py:    @torch.cuda.amp.autocast(enabled=False)
network/predict_e2e.py:        if torch.cuda.is_available() and (not use_cpu):
network/predict_e2e.py:            self.device = torch.device("cuda")
network/predict_e2e.py:                        with torch.cuda.amp.autocast():
network/predict_e2e.py:                torch.cuda.empty_cache()
network/predict_e2e.py:                with torch.cuda.amp.autocast():
network/predict_e2e.py:                with torch.cuda.amp.autocast():
network/equivariant_attention/from_se3cnn/representations.py:    device = 'cuda'
network/InitStrGenerator.py:    #@torch.cuda.amp.autocast(enabled=True)
network/Attention_module_w_str.py:    @torch.cuda.amp.autocast(enabled=False)
network/predict_complex.py:        if torch.cuda.is_available() and (not use_cpu):
network/predict_complex.py:            self.device = torch.device("cuda")
network/predict_complex.py:                        with torch.cuda.amp.autocast():
network/predict_complex.py:                with torch.cuda.amp.autocast():
network/predict_complex.py:                with torch.cuda.amp.autocast():
network/Refine_module.py:    @torch.cuda.amp.autocast(enabled=False)
network/predict_pyRosetta.py:        if torch.cuda.is_available() and (not use_cpu):
network/predict_pyRosetta.py:            self.device = torch.device("cuda")
network/predict_pyRosetta.py:                        with torch.cuda.amp.autocast():
folding-linux.yml:  - tensorflow-gpu=1.14
README.md:#   If your NVIDIA driver compatible with cuda11
README.md:#   If not (but compatible with cuda10)
README.md:The modeling pipeline provided here (run_pyrosetta_ver.sh/run_e2e_ver.sh) is a kind of guidelines to show how RoseTTAFold works. For more efficient use of computing resources, you might want to modify the provided bash script to submit separate jobs with proper dependencies for each of steps (more cpus/memory for hhblits/hhsearch, using gpus only for running the networks, etc). 
DAN-msa/pyErrorPred/model.py:        # Allow gpu memory growth to combat error.
DAN-msa/pyErrorPred/model.py:        config.gpu_options.allow_growth=True
DAN-msa/ErrorPredictorMSA.py:    parser.add_argument("--gpu",
DAN-msa/ErrorPredictorMSA.py:                        help="gpu device to use (default gpu0)")
DAN-msa/ErrorPredictorMSA.py:    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
network_2track/predict_msa.py:        if torch.cuda.is_available() and (not use_cpu):
network_2track/predict_msa.py:            self.device = torch.device("cuda")
example/end-to-end/log/network.stdout:SE(3) iteration 0 tensor(0.5254, device='cuda:0', dtype=torch.float16) 0 tensor(0., device='cuda:0') 0
example/end-to-end/log/network.stdout:SE(3) iteration 1 tensor(0.6665, device='cuda:0', dtype=torch.float16) tensor(0.5254, device='cuda:0', dtype=torch.float16) tensor(0.5254, device='cuda:0', dtype=torch.float16) 0
example/end-to-end/log/network.stdout:SE(3) iteration 2 tensor(0.7100, device='cuda:0', dtype=torch.float16) tensor(0.6665, device='cuda:0', dtype=torch.float16) tensor(0.6665, device='cuda:0', dtype=torch.float16) 0
example/end-to-end/log/network.stdout:SE(3) iteration 3 tensor(0.7163, device='cuda:0', dtype=torch.float16) tensor(0.7100, device='cuda:0', dtype=torch.float16) tensor(0.7100, device='cuda:0', dtype=torch.float16) 0
example/end-to-end/log/network.stdout:SE(3) iteration 4 tensor(0.7173, device='cuda:0', dtype=torch.float16) tensor(0.7163, device='cuda:0', dtype=torch.float16) tensor(0.7163, device='cuda:0', dtype=torch.float16) 0
example/end-to-end/log/network.stdout:SE(3) iteration 5 tensor(0.7144, device='cuda:0', dtype=torch.float16) tensor(0.7173, device='cuda:0', dtype=torch.float16) tensor(0.7173, device='cuda:0', dtype=torch.float16) 0
example/end-to-end/log/network.stdout:SE(3) iteration 6 tensor(0.7168, device='cuda:0', dtype=torch.float16) tensor(0.7144, device='cuda:0', dtype=torch.float16) tensor(0.7173, device='cuda:0', dtype=torch.float16) 1
example/end-to-end/log/network.stdout:SE(3) iteration 7 tensor(0.7153, device='cuda:0', dtype=torch.float16) tensor(0.7168, device='cuda:0', dtype=torch.float16) tensor(0.7173, device='cuda:0', dtype=torch.float16) 0
example/end-to-end/log/network.stdout:SE(3) iteration 8 tensor(0.7144, device='cuda:0', dtype=torch.float16) tensor(0.7153, device='cuda:0', dtype=torch.float16) tensor(0.7173, device='cuda:0', dtype=torch.float16) 1
example/end-to-end/log/network.stdout:SE(3) iteration 9 tensor(0.7163, device='cuda:0', dtype=torch.float16) tensor(0.7144, device='cuda:0', dtype=torch.float16) tensor(0.7173, device='cuda:0', dtype=torch.float16) 2
example/end-to-end/log/network.stdout:SE(3) iteration 10 tensor(0.7163, device='cuda:0', dtype=torch.float16) tensor(0.7163, device='cuda:0', dtype=torch.float16) tensor(0.7173, device='cuda:0', dtype=torch.float16) 0
example/end-to-end/log/network.stdout:SE(3) iteration 11 tensor(0.7163, device='cuda:0', dtype=torch.float16) tensor(0.7163, device='cuda:0', dtype=torch.float16) tensor(0.7173, device='cuda:0', dtype=torch.float16) 1
example/end-to-end/log/network.stdout:SE(3) iteration 12 tensor(0.7173, device='cuda:0', dtype=torch.float16) tensor(0.7163, device='cuda:0', dtype=torch.float16) tensor(0.7173, device='cuda:0', dtype=torch.float16) 2
example/end-to-end/log/network.stdout:SE(3) iteration 13 tensor(0.7168, device='cuda:0', dtype=torch.float16) tensor(0.7173, device='cuda:0', dtype=torch.float16) tensor(0.7173, device='cuda:0', dtype=torch.float16) 0
example/end-to-end/log/network.stdout:SE(3) iteration 14 tensor(0.7158, device='cuda:0', dtype=torch.float16) tensor(0.7168, device='cuda:0', dtype=torch.float16) tensor(0.7173, device='cuda:0', dtype=torch.float16) 1
example/end-to-end/log/network.stdout:SE(3) iteration 15 tensor(0.7153, device='cuda:0', dtype=torch.float16) tensor(0.7158, device='cuda:0', dtype=torch.float16) tensor(0.7173, device='cuda:0', dtype=torch.float16) 2
example/end-to-end/log/network.stdout:SE(3) iteration 16 tensor(0.7139, device='cuda:0', dtype=torch.float16) tensor(0.7153, device='cuda:0', dtype=torch.float16) tensor(0.7173, device='cuda:0', dtype=torch.float16) 3
example/end-to-end/log/network.stdout:SE(3) iteration 17 tensor(0.7173, device='cuda:0', dtype=torch.float16) tensor(0.7139, device='cuda:0', dtype=torch.float16) tensor(0.7173, device='cuda:0', dtype=torch.float16) 4
example/end-to-end/log/network.stdout:SE(3) iteration 18 tensor(0.7153, device='cuda:0', dtype=torch.float16) tensor(0.7173, device='cuda:0', dtype=torch.float16) tensor(0.7173, device='cuda:0', dtype=torch.float16) 0
example/end-to-end/log/network.stdout:SE(3) iteration 19 tensor(0.7163, device='cuda:0', dtype=torch.float16) tensor(0.7153, device='cuda:0', dtype=torch.float16) tensor(0.7173, device='cuda:0', dtype=torch.float16) 1
example/end-to-end/log/network.stdout:SE(3) iteration 20 tensor(0.7183, device='cuda:0', dtype=torch.float16) tensor(0.7163, device='cuda:0', dtype=torch.float16) tensor(0.7173, device='cuda:0', dtype=torch.float16) 0
example/end-to-end/log/network.stdout:SE(3) iteration 21 tensor(0.7153, device='cuda:0', dtype=torch.float16) tensor(0.7183, device='cuda:0', dtype=torch.float16) tensor(0.7183, device='cuda:0', dtype=torch.float16) 0
example/end-to-end/log/network.stdout:SE(3) iteration 22 tensor(0.7148, device='cuda:0', dtype=torch.float16) tensor(0.7153, device='cuda:0', dtype=torch.float16) tensor(0.7183, device='cuda:0', dtype=torch.float16) 1
example/end-to-end/log/network.stdout:SE(3) iteration 23 tensor(0.7163, device='cuda:0', dtype=torch.float16) tensor(0.7148, device='cuda:0', dtype=torch.float16) tensor(0.7183, device='cuda:0', dtype=torch.float16) 2
example/end-to-end/log/network.stdout:SE(3) iteration 24 tensor(0.7158, device='cuda:0', dtype=torch.float16) tensor(0.7163, device='cuda:0', dtype=torch.float16) tensor(0.7183, device='cuda:0', dtype=torch.float16) 0
example/end-to-end/log/network.stdout:SE(3) iteration 25 tensor(0.7139, device='cuda:0', dtype=torch.float16) tensor(0.7158, device='cuda:0', dtype=torch.float16) tensor(0.7183, device='cuda:0', dtype=torch.float16) 1
example/end-to-end/log/network.stdout:SE(3) iteration 26 tensor(0.7148, device='cuda:0', dtype=torch.float16) tensor(0.7139, device='cuda:0', dtype=torch.float16) tensor(0.7183, device='cuda:0', dtype=torch.float16) 2
example/end-to-end/log/network.stdout:SE(3) iteration 27 tensor(0.7158, device='cuda:0', dtype=torch.float16) tensor(0.7148, device='cuda:0', dtype=torch.float16) tensor(0.7183, device='cuda:0', dtype=torch.float16) 0
example/end-to-end/log/network.stdout:SE(3) iteration 28 tensor(0.7153, device='cuda:0', dtype=torch.float16) tensor(0.7158, device='cuda:0', dtype=torch.float16) tensor(0.7183, device='cuda:0', dtype=torch.float16) 0
example/end-to-end/log/network.stdout:SE(3) iteration 29 tensor(0.7153, device='cuda:0', dtype=torch.float16) tensor(0.7153, device='cuda:0', dtype=torch.float16) tensor(0.7183, device='cuda:0', dtype=torch.float16) 1
example/end-to-end/log/network.stdout:SE(3) iteration 30 tensor(0.7168, device='cuda:0', dtype=torch.float16) tensor(0.7153, device='cuda:0', dtype=torch.float16) tensor(0.7183, device='cuda:0', dtype=torch.float16) 2
example/end-to-end/log/network.stdout:SE(3) iteration 31 tensor(0.7173, device='cuda:0', dtype=torch.float16) tensor(0.7168, device='cuda:0', dtype=torch.float16) tensor(0.7183, device='cuda:0', dtype=torch.float16) 0
example/end-to-end/log/network.stdout:SE(3) iteration 32 tensor(0.7178, device='cuda:0', dtype=torch.float16) tensor(0.7173, device='cuda:0', dtype=torch.float16) tensor(0.7183, device='cuda:0', dtype=torch.float16) 0
example/end-to-end/log/network.stdout:SE(3) iteration 33 tensor(0.7153, device='cuda:0', dtype=torch.float16) tensor(0.7178, device='cuda:0', dtype=torch.float16) tensor(0.7183, device='cuda:0', dtype=torch.float16) 0
example/end-to-end/log/network.stdout:SE(3) iteration 34 tensor(0.7163, device='cuda:0', dtype=torch.float16) tensor(0.7153, device='cuda:0', dtype=torch.float16) tensor(0.7183, device='cuda:0', dtype=torch.float16) 1
example/end-to-end/log/network.stdout:SE(3) iteration 35 tensor(0.7163, device='cuda:0', dtype=torch.float16) tensor(0.7163, device='cuda:0', dtype=torch.float16) tensor(0.7183, device='cuda:0', dtype=torch.float16) 0
example/end-to-end/log/network.stdout:SE(3) iteration 36 tensor(0.7163, device='cuda:0', dtype=torch.float16) tensor(0.7163, device='cuda:0', dtype=torch.float16) tensor(0.7183, device='cuda:0', dtype=torch.float16) 1
example/end-to-end/log/network.stdout:SE(3) iteration 37 tensor(0.7168, device='cuda:0', dtype=torch.float16) tensor(0.7163, device='cuda:0', dtype=torch.float16) tensor(0.7183, device='cuda:0', dtype=torch.float16) 2
example/end-to-end/log/network.stdout:SE(3) iteration 38 tensor(0.7178, device='cuda:0', dtype=torch.float16) tensor(0.7168, device='cuda:0', dtype=torch.float16) tensor(0.7183, device='cuda:0', dtype=torch.float16) 0
example/end-to-end/log/network.stdout:SE(3) iteration 39 tensor(0.7163, device='cuda:0', dtype=torch.float16) tensor(0.7178, device='cuda:0', dtype=torch.float16) tensor(0.7183, device='cuda:0', dtype=torch.float16) 0
example/end-to-end/log/network.stdout:SE(3) iteration 40 tensor(0.7153, device='cuda:0', dtype=torch.float16) tensor(0.7163, device='cuda:0', dtype=torch.float16) tensor(0.7183, device='cuda:0', dtype=torch.float16) 1
example/end-to-end/log/network.stdout:SE(3) iteration 41 tensor(0.7168, device='cuda:0', dtype=torch.float16) tensor(0.7153, device='cuda:0', dtype=torch.float16) tensor(0.7183, device='cuda:0', dtype=torch.float16) 2
example/complex_modeling/subunit1.a3m:>tr|H5TA63|H5TA63_9ALTE Uncharacterized protein OS=Glaciecola punicea ACAM 611 GN=GPUN_1059 PE=4 SV=1
example/complex_modeling/subunit1.a3m:>tr|A0A0N0GPU4|A0A0N0GPU4_9NEIS Putative GTP cyclohydrolase 1 type 2 OS=Amantichitinum ursilacus GN=ybgI PE=4 SV=1
example/complex_modeling/subunit1.a3m:>UniRef100_A0A1G5QPA3 Sensor histidine kinase inhibitor, KipI family n=1 Tax=Arthrobacter sp. UNCCL28 TaxID=1502752 RepID=A0A1G5QPA3_9MICC
example/complex_modeling/subunit1.a3m:>UniRef100_A0A1K1R276 Inhibitor of KinA n=1 Tax=Paenibacillus sp. UNCCL117 TaxID=1502764 RepID=A0A1K1R276_9BACL
example/complex_modeling/subunit1.a3m:------------------------LLRGRLDLLAENLDAepGL---------LHRrYAELD-FGSTGARRLKLrgrpcstwPRSTGMsmppnccliveRmx---------------------------------------------------------------------------------------------------------------------------------------------
example/complex_modeling/subunit1.a3m:>tr|A0A1K1R276|A0A1K1R276_9BACL Inhibitor of KinA OS=Paenibacillus sp. UNCCL117 GN=SAMN02799630_05483 PE=4 SV=1
example/complex_modeling/subunit1.a3m:>tr|A0A0P9GPU5|A0A0P9GPU5_9GAMM Kinase A inhibitor OS=Pseudoalteromonas sp. P1-11 GN=kipI PE=4 SV=1
example/complex_modeling/subunit1.a3m:>tr|A0A1I1J6E2|A0A1I1J6E2_9BACI Inhibitor of KinA OS=Bacillus sp. UNCCL81 GN=SAMN02799633_00798 PE=4 SV=1
example/complex_modeling/paired.a3m:>A0A0N0GPU4_A0A0N0GPU4
example/complex_modeling/subunit2.a3m:>tr|H5TA63|H5TA63_9ALTE Uncharacterized protein OS=Glaciecola punicea ACAM 611 GN=GPUN_1059 PE=4 SV=1
example/complex_modeling/subunit2.a3m:>tr|A0A0N0GPU4|A0A0N0GPU4_9NEIS Putative GTP cyclohydrolase 1 type 2 OS=Amantichitinum ursilacus GN=ybgI PE=4 SV=1
example/complex_modeling/subunit2.a3m:MLNILRAGIFTTVQDLGRSGYRQLGVSQTGALDAPALRIGNLLVG---ndeNAAGLEITLGQFSTEFNRAGWIALTGAGCHAELDGKPLWTGWRYAVKPGQVLTMKTptr---GMRSYLTVSGGIDV-pEVLGSRSTDLKTGFGGLSGRPLRDGDHLSVSNNCCLatvfvpCRDQNTR-----ssvlkGGT----PS-GGHHGSSVRRAIVWDIVCTVMPX-----------------------------------------------------------------------------------------------------------
example/complex_modeling/subunit2.a3m:>UniRef100_A0A1I1J5G2 Biotin-dependent carboxylase uncharacterized domain-containing protein n=1 Tax=Bacillus sp. UNCCL81 TaxID=1502755 RepID=A0A1I1J5G2_9BACI
example/complex_modeling/subunit2.a3m:>UniRef100_A0A1K1R284 Antagonist of KipI n=1 Tax=Paenibacillus sp. UNCCL117 TaxID=1502764 RepID=A0A1K1R284_9BACL
example/complex_modeling/subunit2.a3m:HLLIEASTPLCLLQDRGRFGVRHLGVTQGGALDWVSMSWANHLLGNPLDASVVEIALGGLTLVAQENCCLALAGADLGAQIDGQALVPWRSFVLNKGQRLQFTQPLLGARAYLAAPGGFDAPKVLGSSACVVREELGGLDglGKPLPKGASLSYSGAA-VPPrelSAAQVPDFHAKAPLRVVLGAQIGAFSGQSLFDAFNSTWTLDSRGDRMGIRLLGPALTYQ-GQPMISEGIPLGAIQVPPDGQPIVLLNDRQTIGGYPRLGALTPLSLARLAQCLPGTKVRLAPTVQDSAHRQQVEFMRKFA----------
example/complex_modeling/subunit2.a3m:>tr|A0A1K1R284|A0A1K1R284_9BACL Antagonist of KipI OS=Paenibacillus sp. UNCCL117 GN=SAMN02799630_05484 PE=4 SV=1
example/complex_modeling/subunit2.a3m:>UniRef100_A0A1C6TXS0 Biotin-dependent carboxylase uncharacterized domain-containing protein n=1 Tax=Micromonospora yangpuensis TaxID=683228 RepID=A0A1C6TXS0_9ACTN
example/complex_modeling/subunit2.a3m:>UniRef100_UPI000628FA5A biotin-dependent carboxyltransferase family protein n=1 Tax=Streptomyces yangpuensis TaxID=1648182 RepID=UPI000628FA5A
example/complex_modeling/subunit2.a3m:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------RFIRGSDVNCCLP-vAIDIFSTGSFFVSPDSDRMGVRLDGLRLERs-NETDLVSEAVAPGTVQVPPSGKPILLLNDCQTIGGYPKLAHVIAVDLPIAAELRPGDIVRFREVSLADAHRAlnerE----RDLEQFRRgiES----
example/complex_modeling/subunit2.a3m:>UniRef100_A0A1G5QGZ9 Allophanate hydrolase subunit 2 n=1 Tax=Arthrobacter sp. UNCCL28 TaxID=1502752 RepID=A0A1G5QGZ9_9MICC
example/complex_modeling/subunit2.a3m:>tr|A0A1K1R276|A0A1K1R276_9BACL Inhibitor of KinA OS=Paenibacillus sp. UNCCL117 GN=SAMN02799630_05483 PE=4 SV=1
RoseTTAFold-linux-cu101.yml:  - nvidia
RoseTTAFold-linux-cu101.yml:  - cudatoolkit=10.2.89=h8f6ccaa_8
RoseTTAFold-linux-cu101.yml:  - pytorch=1.8.1=py3.8_cuda10.2_cudnn7.6.5_0
RoseTTAFold-linux.yml:  - nvidia
RoseTTAFold-linux.yml:  - cudatoolkit=11.1.74=h6bb024c_0
RoseTTAFold-linux.yml:  - pytorch=1.9.0=py3.8_cuda11.1_cudnn8.0.5_0

```
