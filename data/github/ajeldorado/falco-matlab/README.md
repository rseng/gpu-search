# https://github.com/ajeldorado/falco-matlab

```console
config/EXAMPLE_defaults_WFIRST_FOHLC.m:mp.useGPU = false;
config/EXAMPLE_defaults_WFIRST_LC.m:mp.useGPU = false;
config/EXAMPLE_defaults_SVC_chromatic.m:mp.useGPU = false;
config/EXAMPLE_defaults_LUVOIRB_VC_design.m:mp.useGPU = false;
config/EXAMPLE_defaults_WFIRST_HLC_BMC_WFE.m:mp.useGPU = false;
config/EXAMPLE_defaults_try_running_FALCO.m:mp.useGPU = false;
config/EXAMPLE_defaults_LUVOIRA_APLC_smallFPM.m:mp.useGPU = false;
config/EXAMPLE_defaults_DST_FLC_design.m:mp.useGPU = false;
config/EXAMPLE_defaults_LUVOIRB_VC_design_fibers.m:mp.useGPU = false;
config/EXAMPLE_defaults_WFIRST_PhaseB_SPC_Spec_simple.m:mp.useGPU = false;
config/EXAMPLE_defaults_LUVOIRA_APLC_largeFPM.m:mp.useGPU = false;
config/EXAMPLE_defaults_design_HLC.m:mp.useGPU = false;
config/EXAMPLE_defaults_MSWC.m:mp.useGPU = false;
config/EXAMPLE_defaults_VC_simple.m:mp.useGPU = false;
config/EXAMPLE_defaults_SCC.m:mp.useGPU = false;
config/EXAMPLE_defaults_LUVOIRA_APLC_mediumFPM.m:mp.useGPU = false;
config/EXAMPLE_defaults_DST_LC_design.m:mp.useGPU = false;
models/model_Jacobian_no_FPM.m:if mp.useGPU
models/model_Jacobian_no_FPM.m:    pupil = gpuArray(pupil);
models/model_Jacobian_no_FPM.m:    Ein = gpuArray(Ein);
models/model_Jacobian_no_FPM.m:    if any(mp.dm_ind == 1); DM1surf = gpuArray(DM1surf); end
models/model_Jacobian_no_FPM.m:    if any(mp.dm_ind == 2); DM2surf = gpuArray(DM2surf); end
models/model_Jacobian_no_FPM.m:if mp.useGPU
models/model_full.m:%% Undo GPU variables if they exist
models/model_full.m:if(mp.useGPU)
models/model_Jacobian_VC.m:if mp.useGPU
models/model_Jacobian_VC.m:    pupil = gpuArray(pupil);
models/model_Jacobian_VC.m:    Ein = gpuArray(Ein);
models/model_Jacobian_VC.m:        DM1surf = gpuArray(DM1surf);
models/model_Jacobian_VC.m:            if mp.useGPU; dEbox = gpuArray(dEbox); end
models/model_Jacobian_VC.m:                if isa(dEP2boxEff, 'gpuArray')
models/model_Jacobian_VC.m:                    EP2eff = gpuArray.zeros(mp.dm1.compact.NdmPad);
models/model_Jacobian_VC.m:                if mp.useGPU
models/model_Jacobian_VC.m:            if mp.useGPU
models/model_Jacobian_VC.m:                dEbox = gpuArray(dEbox);
models/model_Jacobian_VC.m:                if isa(dEP2boxEff, 'gpuArray')
models/model_Jacobian_VC.m:                    EP2eff = gpuArray.zeros(mp.dm2.compact.NdmPad);
models/model_Jacobian_VC.m:                if(mp.useGPU)
models/model_Jacobian_lyot.m:if(mp.useGPU)
models/model_Jacobian_lyot.m:    pupil = gpuArray(pupil);
models/model_Jacobian_lyot.m:    Ein = gpuArray(Ein);
models/model_Jacobian_lyot.m:if mp.useGPU
models/model_Jacobian_lyot.m:    if any(mp.dm_ind == 1); DM1surf = gpuArray(DM1surf); end
models/model_Jacobian_lyot.m:    if any(mp.dm_ind == 2); DM2surf = gpuArray(DM2surf); end
models/model_Jacobian_lyot.m:                    if mp.useGPU; EP4noFPM = gpuArray(EP4noFPM);end
models/model_Jacobian_lyot.m:                if mp.useGPU
models/model_Jacobian_lyot.m:                    if mp.useGPU; EP4noFPM = gpuArray(EP4noFPM);end
models/model_Jacobian_lyot.m:                if(mp.useGPU)
models/model_Jacobian_lyot.m:            if mp.useGPU; EFend = gather(EFend); end
models/model_Jacobian_lyot.m:            if mp.useGPU; EFend = gather(EFend); end
models/model_Jacobian_lyot.m:if(mp.useGPU)
models/model_full_Fourier.m:if(mp.useGPU)
models/model_full_Fourier.m:    pupil = gpuArray(pupil);
models/model_full_Fourier.m:    Ein = gpuArray(Ein);
models/model_full_Fourier.m:    if(any(mp.dm_ind==1)); DM1surf = gpuArray(DM1surf); end
models/model_full_Fourier.m:    if(any(mp.dm_ind==2)); DM2surf = gpuArray(DM2surf); end
models/model_full_Fourier.m:%             EP4 = propcustom_mft_PtoFtoP_multispot(EP3, fpm, mp.P1.full.Nbeam/2, inVal, outVal, mp.useGPU,'spotDiamVec',mp.F3.VortexSpotDiamVec * (phaseScaleFac),'spotAmpVec',mp.F3.VortexSpotAmpVec,'spotPhaseVec',mp.F3.VortexSpotPhaseVec/phaseScaleFac);
models/model_full_Fourier.m:            EP4 = propcustom_mft_PtoFtoP_multispot(EP3, FPMcoarse, FPMfine, mp.P1.full.Nbeam/2, inVal, outVal, mp.useGPU);
models/model_full_Fourier.m:            EP4 = propcustom_mft_PtoFtoP(EP3, fpm, mp.P1.full.Nbeam/2, inVal, outVal, mp.useGPU, spotDiam, spotOffsets);
models/model_full_Fourier.m:if mp.useGPU; Eout = gather(Eout); end
models/model_compact_general.m:if mp.useGPU
models/model_compact_general.m:    pupil = gpuArray(pupil);
models/model_compact_general.m:    Ein = gpuArray(Ein);
models/model_compact_general.m:    if any(mp.dm_ind == 1); DM1surf = gpuArray(DM1surf); end
models/model_compact_general.m:    if any(mp.dm_ind == 2); DM2surf = gpuArray(DM2surf); end
models/model_compact_general.m:                EP4 = propcustom_mft_PtoFtoP_multispot(EP3, FPMcoarse, FPMfine, mp.P1.compact.Nbeam/2, inVal, outVal, mp.useGPU);
models/model_compact_general.m:                EP4 = propcustom_mft_PtoFtoP(EP3, fpm, mp.P1.compact.Nbeam/2, inVal, outVal, mp.useGPU, spotDiam, spotOffsets);
models/model_compact_general.m:if mp.useGPU
models/model_ZWFS.m:if(mp.useGPU)
models/model_ZWFS.m:    pupil = gpuArray(pupil);
models/model_ZWFS.m:    Ein = gpuArray(Ein);
models/model_ZWFS.m:    if(any(mp.dm_ind==1)); DM1surf = gpuArray(DM1surf); end
models/model_ZWFS.m:if(mp.useGPU)
old/models/model_full_scale.m:if(mp.useGPU)
old/models/model_full_scale.m:    pupil = gpuArray(pupil);
old/models/model_full_scale.m:    Ein = gpuArray(Ein);
old/models/model_full_scale.m:    if(any(mp.dm_ind==1)); DM1surf = gpuArray(DM1surf); end
old/models/model_full_scale.m:    if(any(mp.dm_ind==2)); DM2surf = gpuArray(DM2surf); end
old/models/model_full_scale.m:%         EP4 = propcustom_mft_Pup2Vortex2Pup( EP3, charge, mp.P1.full.Nbeam/2, 0.3, 5, mp.useGPU );  %--MFTs
old/models/model_full_scale.m:if(mp.useGPU); Eout = gather(Eout); end
old/models/model_compact_scale.m:if mp.useGPU
old/models/model_compact_scale.m:    pupil = gpuArray(pupil);
old/models/model_compact_scale.m:    Ein = gpuArray(Ein);
old/models/model_compact_scale.m:    if(any(mp.dm_ind==1)); DM1surf = gpuArray(DM1surf); end
old/models/model_compact_scale.m:if mp.useGPU
old/models/model_Jacobian_HLC_scale.m:if(mp.useGPU)
old/models/model_Jacobian_HLC_scale.m:    pupil = gpuArray(pupil);
old/models/model_Jacobian_HLC_scale.m:    Ein = gpuArray(Ein);
old/models/model_Jacobian_HLC_scale.m:    if(any(mp.dm_ind==1)); DM1surf = gpuArray(DM1surf); end
old/models/model_Jacobian_HLC_scale.m:            if mp.useGPU; EP4noFPM = gpuArray(EP4noFPM);end
old/models/model_Jacobian_HLC_scale.m:            if mp.useGPU; EFend = gather(EFend); end
old/models/model_Jacobian_HLC_scale.m:            if mp.useGPU; EP4noFPM = gpuArray(EP4noFPM); end
old/models/model_Jacobian_HLC_scale.m:            if mp.useGPU; EFend = gather(EFend); end
old/models/model_Jacobian_HLC_scale.m:if mp.useGPU;  Gmode = gather(Gmode);  end
old/Roddier/model_Jacobian_Roddier.m:if(mp.useGPU)
old/Roddier/model_Jacobian_Roddier.m:    pupil = gpuArray(pupil);
old/Roddier/model_Jacobian_Roddier.m:    Ein = gpuArray(Ein);
old/Roddier/model_Jacobian_Roddier.m:    if(any(mp.dm_ind==1)); DM1surf = gpuArray(DM1surf); end
old/Roddier/model_Jacobian_Roddier.m:            if(mp.useGPU);dEP2box = gather(dEP2box);end
old/Roddier/model_Jacobian_Roddier.m:            if(mp.useGPU);EFend = gather(EFend) ;end
old/Roddier/model_Jacobian_Roddier.m:            if(mp.useGPU);dEP2box = gather(dEP2box);end
old/Roddier/model_Jacobian_Roddier.m:            if(mp.useGPU);EFend = gather(EFend) ;end
old/Roddier/model_Jacobian_Roddier.m:if(mp.useGPU)
old/setup/falco_gen_chosen_apodizer.m:                mp.P3.full.mask = falco_gen_tradApodizer(mp.P1.full.mask,mp.P1.full.Nbeam,mp.F3.Rin,(1+mp.fracBW/2)*mp.F3.Rout,mp.useGPU);
old/EHLC/model_compact_EHLC.m:if(mp.useGPU)
old/EHLC/model_compact_EHLC.m:    pupil = gpuArray(pupil);
old/EHLC/model_compact_EHLC.m:    Ein = gpuArray(Ein);
old/EHLC/model_compact_EHLC.m:    if(any(mp.dm_ind==1)); DM1surf = gpuArray(DM1surf); end
old/EHLC/model_compact_EHLC.m:if(mp.useGPU)
old/EHLC/model_Jacobian_EHLC.m:if(mp.useGPU)
old/EHLC/model_Jacobian_EHLC.m:    pupil = gpuArray(pupil);
old/EHLC/model_Jacobian_EHLC.m:    Ein = gpuArray(Ein);
old/EHLC/model_Jacobian_EHLC.m:    if(any(mp.dm_ind==1)); DM1surf = gpuArray(DM1surf); end
old/EHLC/model_Jacobian_EHLC.m:if(mp.useGPU)
testing/tests_long/@ConfigurationFLC/ConfigurationFLC.m:mp.useGPU = false;
testing/tests_long/@ConfigurationLC/ConfigurationLC.m:mp.useGPU = false;
testing/tests_long/@ConfigurationVortex/ConfigurationVortex.m:mp.useGPU = false;
testing/tests_long/@ConfigurationMSWC/ConfigurationMSWC.m:mp.useGPU = false;
lib/propcustom/propcustom_mft_PtoFtoP.m:% function OUT = propcustom_mft_PtoFtoP(IN, FPM, charge, apRad, inVal, outVal, useGPU ) 
lib/propcustom/propcustom_mft_PtoFtoP.m:% useGPU: boolean flag whether to use a GPU to speed up calculations
lib/propcustom/propcustom_mft_PtoFtoP.m:function OUT = propcustom_mft_PtoFtoP(IN, FPM, apRad, inVal, outVal, useGPU, varargin)
lib/propcustom/propcustom_mft_PtoFtoP.m:    if useGPU
lib/propcustom/propcustom_mft_PtoFtoP.m:        IN = gpuArray(IN);
lib/propcustom/propcustom_mft_PtoFtoP.m:        x = gpuArray(x);
lib/propcustom/propcustom_mft_PtoFtoP.m:        u1 = gpuArray(u1);
lib/propcustom/propcustom_mft_PtoFtoP.m:        u2 = gpuArray(u2);
lib/propcustom/propcustom_mft_PtoFtoP.m:        windowMASK1 = gpuArray(windowMASK1);
lib/propcustom/propcustom_mft_PtoFtoP.m:        windowMASK2 = gpuArray(windowMASK2);
lib/propcustom/propcustom_mft_PtoFtoP.m:    if useGPU
lib/propcustom/propcustom_mft_PtoFtoP_multispot.m:% function OUT = propcustom_mft_PtoFtoP(IN, FPM, charge, apRad, inVal, outVal, useGPU ) 
lib/propcustom/propcustom_mft_PtoFtoP_multispot.m:% useGPU: boolean flag whether to use a GPU to speed up calculations
lib/propcustom/propcustom_mft_PtoFtoP_multispot.m:function OUT = propcustom_mft_PtoFtoP_multispot(IN, FPMcoarse, FPMfine, apRad, inVal, outVal, useGPU, varargin)
lib/propcustom/propcustom_mft_PtoFtoP_multispot.m:    if useGPU
lib/propcustom/propcustom_mft_PtoFtoP_multispot.m:        IN = gpuArray(IN);
lib/propcustom/propcustom_mft_PtoFtoP_multispot.m:        x = gpuArray(x);
lib/propcustom/propcustom_mft_PtoFtoP_multispot.m:        u1 = gpuArray(u1);
lib/propcustom/propcustom_mft_PtoFtoP_multispot.m:        u2 = gpuArray(u2);
lib/propcustom/propcustom_mft_PtoFtoP_multispot.m:        windowMASK1 = gpuArray(windowMASK1);
lib/propcustom/propcustom_mft_PtoFtoP_multispot.m:        windowMASK2 = gpuArray(windowMASK2);
lib/propcustom/propcustom_mft_PtoFtoP_multispot.m:    if useGPU
lib/propcustom/propcustom_mft_Pup2Vortex2Pup.m:% function OUT = propcustom_mft_Pup2Vortex2Pup( IN, charge, apRad,  inVal, outVal, useGPU, diamSpotLamD) 
lib/propcustom/propcustom_mft_Pup2Vortex2Pup.m:function OUT = propcustom_mft_Pup2Vortex2Pup(IN, charge, apRad, inVal, outVal, useGPU, varargin)
lib/propcustom/propcustom_mft_Pup2Vortex2Pup.m:    if(useGPU)
lib/propcustom/propcustom_mft_Pup2Vortex2Pup.m:        IN = gpuArray(IN);
lib/propcustom/propcustom_mft_Pup2Vortex2Pup.m:        x = gpuArray(x);
lib/propcustom/propcustom_mft_Pup2Vortex2Pup.m:        u1 = gpuArray(u1);
lib/propcustom/propcustom_mft_Pup2Vortex2Pup.m:        u2 = gpuArray(u2);
lib/propcustom/propcustom_mft_Pup2Vortex2Pup.m:        windowMASK1 = gpuArray(windowMASK1);
lib/propcustom/propcustom_mft_Pup2Vortex2Pup.m:        windowMASK2 = gpuArray(windowMASK2);
lib/propcustom/propcustom_mft_Pup2Vortex2Pup.m:    if(useGPU)
lib/masks/falco_gen_azimuthal_phase_mask.m:% function OUT = propcustom_mft_PtoFtoP(IN, FPM, charge, apRad, inVal, outVal, useGPU ) 
lib/masks/falco_gen_tradApodizer.m:%  useGPU: (logical) Run FFTs on GPUs
lib/masks/falco_gen_tradApodizer.m:function APOD = falco_gen_tradApodizer(PUPIL,apDiaSamps,effIWA,effOWA,useGPU)
lib/masks/falco_gen_tradApodizer.m:    if(useGPU)
lib/masks/falco_gen_tradApodizer.m:        PUPIL = gpuArray(PUPIL);
lib/masks/falco_gen_tradApodizer.m:    if(useGPU)
lib/utils/padOrCropEven.m:	if(isa(Ain,'gpuArray'))
lib/utils/padOrCropEven.m:        Aout = gpuArray.ones(Ndes);
lib/utils/padOrCropEven.m:            if(~isa(Ain,'gpuArray'))
demo/demo_vortex_propagation.m:useGPU = false;
demo/demo_vortex_propagation.m:Epup2 = propcustom_mft_Pup2Vortex2Pup(pad_crop(Epup1, Nout), charge, Nbeam/2, 0.3, 5, useGPU, diamSpotLamD);  %--MFTs
roman/EXAMPLE_config_Roman_CGI_HLC_NFOV_Band1.m:mp.useGPU = false;
roman/EXAMPLE_config_Roman_CGI_SPC_Multistar_Band4.m:mp.useGPU = false;
roman/EXAMPLE_config_Roman_CGI_HLC_NFOV_Band2.m:mp.useGPU = false;
roman/EXAMPLE_config_Roman_CGI_SPC_WFOV_Band4.m:mp.useGPU = false;
roman/EXAMPLE_config_Roman_CGI_SPC_Bowtie_Band3.m:mp.useGPU = false;
roman/EXAMPLE_config_Roman_CGI_SPC_RotatedBowtie_Band2.m:mp.useGPU = false;
roman/EXAMPLE_config_Roman_CGI_SPC_Bowtie_Band2.m:mp.useGPU = false;
roman/EXAMPLE_config_Roman_CGI_HLC_NFOV_Band4.m:mp.useGPU = false;
roman/EXAMPLE_config_Roman_CGI_HLC_NFOV_Band3.m:mp.useGPU = false;
roman/EXAMPLE_config_Roman_CGI_SPC_Multistar_Band1.m:mp.useGPU = false;
roman/EXAMPLE_config_Roman_CGI_SPC_WFOV_Band1.m:mp.useGPU = false;
roman/EXAMPLE_config_Roman_CGI_SPC_RotatedBowtie_Band3.m:mp.useGPU = false;
wfirst_modeling/EXAMPLE_defaults_WFIRST_PhaseB_PROPER_HLC.m:mp.useGPU = false;
wfirst_modeling/EXAMPLE_defaults_WFIRST_PhaseB_PROPER_SPC_Spec.m:mp.useGPU = false;
wfirst_modeling/EXAMPLE_defaults_WFIRST_PhaseB_PROPER_SPC_WFOV.m:mp.useGPU = false;

```
