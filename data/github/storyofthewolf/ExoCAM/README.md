# https://github.com/storyofthewolf/ExoCAM

```console
cesm1.2.1/configs/cam_aqua_fv/SourceMods/src.drv/ccsm_comp_mod.F90:   type (seq_timemgr_type), SAVE :: seq_SyncClock ! array of all clocks & alarm
cesm1.2.1/configs/cam_aqua_fv/SourceMods/src.drv/ccsm_comp_mod.F90:   call seq_timemgr_clockInit(seq_SyncClock,nlfilename,read_restart,rest_file,mpicom_gloid, &
cesm1.2.1/configs/cam_aqua_fv/SourceMods/src.drv/ccsm_comp_mod.F90:       call seq_timemgr_clockPrint(seq_SyncClock)
cesm1.2.1/configs/cam_aqua_fv/SourceMods/src.drv/ccsm_comp_mod.F90:      call seq_timemgr_clockAdvance( seq_SyncClock)
cesm1.2.1/configs/cam_aqua_fv/SourceMods/src.drv/ccsm_comp_mod.F90:         call seq_rest_write(EClock_d,seq_SyncClock)
cesm1.2.1/configs/cam_mixed_fv/SourceMods/src.drv/ccsm_comp_mod.F90:   type (seq_timemgr_type), SAVE :: seq_SyncClock ! array of all clocks & alarm
cesm1.2.1/configs/cam_mixed_fv/SourceMods/src.drv/ccsm_comp_mod.F90:   call seq_timemgr_clockInit(seq_SyncClock,nlfilename,read_restart,rest_file,mpicom_gloid, &
cesm1.2.1/configs/cam_mixed_fv/SourceMods/src.drv/ccsm_comp_mod.F90:       call seq_timemgr_clockPrint(seq_SyncClock)
cesm1.2.1/configs/cam_mixed_fv/SourceMods/src.drv/ccsm_comp_mod.F90:      call seq_timemgr_clockAdvance( seq_SyncClock)
cesm1.2.1/configs/cam_mixed_fv/SourceMods/src.drv/ccsm_comp_mod.F90:         call seq_rest_write(EClock_d,seq_SyncClock)
cesm1.2.1/configs/experimental/co2condense/SourceMods/src.drv/ccsm_comp_mod.F90:   type (seq_timemgr_type), SAVE :: seq_SyncClock ! array of all clocks & alarm
cesm1.2.1/configs/experimental/co2condense/SourceMods/src.drv/ccsm_comp_mod.F90:   call seq_timemgr_clockInit(seq_SyncClock,nlfilename,read_restart,rest_file,mpicom_gloid, &
cesm1.2.1/configs/experimental/co2condense/SourceMods/src.drv/ccsm_comp_mod.F90:       call seq_timemgr_clockPrint(seq_SyncClock)
cesm1.2.1/configs/experimental/co2condense/SourceMods/src.drv/ccsm_comp_mod.F90:      call seq_timemgr_clockAdvance( seq_SyncClock)
cesm1.2.1/configs/experimental/co2condense/SourceMods/src.drv/ccsm_comp_mod.F90:         call seq_rest_write(EClock_d,seq_SyncClock)
cesm1.2.1/configs/cam_land_fv/SourceMods/src.drv/ccsm_comp_mod.F90:   type (seq_timemgr_type), SAVE :: seq_SyncClock ! array of all clocks & alarm
cesm1.2.1/configs/cam_land_fv/SourceMods/src.drv/ccsm_comp_mod.F90:   call seq_timemgr_clockInit(seq_SyncClock,nlfilename,read_restart,rest_file,mpicom_gloid, &
cesm1.2.1/configs/cam_land_fv/SourceMods/src.drv/ccsm_comp_mod.F90:       call seq_timemgr_clockPrint(seq_SyncClock)
cesm1.2.1/configs/cam_land_fv/SourceMods/src.drv/ccsm_comp_mod.F90:      call seq_timemgr_clockAdvance( seq_SyncClock)
cesm1.2.1/configs/cam_land_fv/SourceMods/src.drv/ccsm_comp_mod.F90:         call seq_rest_write(EClock_d,seq_SyncClock)
cesm1.2.1/configs/cam_aqua_se/SourceMods/src.drv/ccsm_comp_mod.F90:   type (seq_timemgr_type), SAVE :: seq_SyncClock ! array of all clocks & alarm
cesm1.2.1/configs/cam_aqua_se/SourceMods/src.drv/ccsm_comp_mod.F90:   call seq_timemgr_clockInit(seq_SyncClock,nlfilename,read_restart,rest_file,mpicom_gloid, &
cesm1.2.1/configs/cam_aqua_se/SourceMods/src.drv/ccsm_comp_mod.F90:       call seq_timemgr_clockPrint(seq_SyncClock)
cesm1.2.1/configs/cam_aqua_se/SourceMods/src.drv/ccsm_comp_mod.F90:      call seq_timemgr_clockAdvance( seq_SyncClock)
cesm1.2.1/configs/cam_aqua_se/SourceMods/src.drv/ccsm_comp_mod.F90:         call seq_rest_write(EClock_d,seq_SyncClock)
cesm1.2.1/configs/circumbinary/SourceMods/src.drv/ccsm_comp_mod.F90:   type (seq_timemgr_type), SAVE :: seq_SyncClock ! array of all clocks & alarm
cesm1.2.1/configs/circumbinary/SourceMods/src.drv/ccsm_comp_mod.F90:   call seq_timemgr_clockInit(seq_SyncClock,nlfilename,read_restart,rest_file,mpicom_gloid, &
cesm1.2.1/configs/circumbinary/SourceMods/src.drv/ccsm_comp_mod.F90:       call seq_timemgr_clockPrint(seq_SyncClock)
cesm1.2.1/configs/circumbinary/SourceMods/src.drv/ccsm_comp_mod.F90:      call seq_timemgr_clockAdvance( seq_SyncClock)
cesm1.2.1/configs/circumbinary/SourceMods/src.drv/ccsm_comp_mod.F90:         call seq_rest_write(EClock_d,seq_SyncClock)

```
