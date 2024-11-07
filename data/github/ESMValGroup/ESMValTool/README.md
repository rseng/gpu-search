# https://github.com/ESMValGroup/ESMValTool

```console
esmvaltool/diag_scripts/ipcc_ar5/tsline.ncl:      procmod = tmp1
esmvaltool/diag_scripts/ipcc_ar5/tsline.ncl:      procmod = (tmp1 + tmp2) / 2.
esmvaltool/diag_scripts/ipcc_ar5/tsline.ncl:      procmod = area_operations(A0_timavg, -90., 90., 0., 360., \
esmvaltool/diag_scripts/ipcc_ar5/tsline.ncl:      tmp = runave_Wrap(procmod, 12 * diag_script_info@runave, 0)
esmvaltool/diag_scripts/ipcc_ar5/tsline.ncl:      delete(procmod)
esmvaltool/diag_scripts/ipcc_ar5/tsline.ncl:      procmod = tmp
esmvaltool/diag_scripts/ipcc_ar5/tsline.ncl:      date = cd_calendar(procmod&time, -1)
esmvaltool/diag_scripts/ipcc_ar5/tsline.ncl:      date = procmod&year
esmvaltool/diag_scripts/ipcc_ar5/tsline.ncl:    model_arr(imod, idx1:idx2) = (/procmod/)
esmvaltool/diag_scripts/ipcc_ar5/tsline.ncl:      copy_VarAtts(procmod, model_arr)
esmvaltool/diag_scripts/ipcc_ar5/tsline.ncl:    delete(procmod)
esmvaltool/diag_scripts/ipcc_ar6/precip_anom.ncl:        procmod = area_operations(A0_timavg, -90., 90., 0., 360., \
esmvaltool/diag_scripts/ipcc_ar6/precip_anom.ncl:        date = procmod&year
esmvaltool/diag_scripts/ipcc_ar6/precip_anom.ncl:        model_arr(imod, idx1:idx2) = (/procmod/)
esmvaltool/diag_scripts/ipcc_ar6/precip_anom.ncl:          copy_VarAtts(procmod, model_arr)
esmvaltool/diag_scripts/ipcc_ar6/precip_anom.ncl:        delete(procmod)
esmvaltool/diag_scripts/ipcc_ar6/zonal_st_dev.ncl:    procmod = dim_avg_Wrap(A0)
esmvaltool/diag_scripts/ipcc_ar6/zonal_st_dev.ncl:    A0_timavg = time_operations(procmod, -1, -1, "average", "yearly", True)
esmvaltool/diag_scripts/ipcc_ar6/zonal_st_dev.ncl:    delete(procmod)
esmvaltool/diag_scripts/ipcc_ar6/tas_anom_damip.ncl:        procmod = area_operations(A0_timavg, -90., 90., 0., 360., \
esmvaltool/diag_scripts/ipcc_ar6/tas_anom_damip.ncl:        date = procmod&year
esmvaltool/diag_scripts/ipcc_ar6/tas_anom_damip.ncl:        model_arr(imod, idx1:idx2) = (/procmod/)
esmvaltool/diag_scripts/ipcc_ar6/tas_anom_damip.ncl:          copy_VarAtts(procmod, model_arr)
esmvaltool/diag_scripts/ipcc_ar6/tas_anom_damip.ncl:        delete(procmod)
esmvaltool/diag_scripts/ipcc_ar6/tas_anom.ncl:    procmod := diag_tas
esmvaltool/diag_scripts/ipcc_ar6/tas_anom.ncl:      procmod := diag
esmvaltool/diag_scripts/ipcc_ar6/tas_anom.ncl:    date = procmod&year
esmvaltool/diag_scripts/ipcc_ar6/tas_anom.ncl:    model_arr(imod, idx1:idx2) = (/procmod/)
esmvaltool/diag_scripts/ipcc_ar6/tas_anom.ncl:    delete(procmod)
esmvaltool/diag_scripts/bock20jgr/tsline.ncl:      procmod = tmp1
esmvaltool/diag_scripts/bock20jgr/tsline.ncl:      procmod = (tmp1 + tmp2) / 2.
esmvaltool/diag_scripts/bock20jgr/tsline.ncl:      procmod = area_operations(A0_timavg, -90., 90., 0., 360., \
esmvaltool/diag_scripts/bock20jgr/tsline.ncl:    if (.not.isdefined("procmod")) then
esmvaltool/diag_scripts/bock20jgr/tsline.ncl:      date = cd_calendar(procmod&time, -1)
esmvaltool/diag_scripts/bock20jgr/tsline.ncl:      date = procmod&year
esmvaltool/diag_scripts/bock20jgr/tsline.ncl:    model_arr(imod, idx1:idx2) = (/procmod/)
esmvaltool/diag_scripts/bock20jgr/tsline.ncl:      copy_VarAtts(procmod, model_arr)
esmvaltool/diag_scripts/bock20jgr/tsline.ncl:    delete(procmod)
esmvaltool/diag_scripts/carbon_ec/carbon_tsline.ncl:      procmod = tmp1
esmvaltool/diag_scripts/carbon_ec/carbon_tsline.ncl:      procmod = (tmp1 + tmp2) / 2.
esmvaltool/diag_scripts/carbon_ec/carbon_tsline.ncl:      procmod = area_operations(A0_timavg, diag_script_info@ts_minlat, \
esmvaltool/diag_scripts/carbon_ec/carbon_tsline.ncl:      date = cd_calendar(procmod&time, -1)
esmvaltool/diag_scripts/carbon_ec/carbon_tsline.ncl:      date = procmod&year
esmvaltool/diag_scripts/carbon_ec/carbon_tsline.ncl:      model_arr(imod, :dimsizes(procmod)-1) = (/procmod/)
esmvaltool/diag_scripts/carbon_ec/carbon_tsline.ncl:      xmax = date(dimsizes(procmod)-1)
esmvaltool/diag_scripts/carbon_ec/carbon_tsline.ncl:      model_arr(imod, idx1:idx2) = (/procmod/)
esmvaltool/diag_scripts/carbon_ec/carbon_tsline.ncl:      copy_VarAtts(procmod, model_arr)
esmvaltool/diag_scripts/carbon_ec/carbon_tsline.ncl:    delete(procmod)
conda-linux-64.lock:https://conda.anaconda.org/conda-forge/linux-64/_py-xgboost-mutex-2.0-gpu_0.tar.bz2#7702188077361f43a4d61e64c694f850
conda-linux-64.lock:https://conda.anaconda.org/conda-forge/noarch/cuda-version-11.8-h70ddcb2_3.conda#670f0e1593b8c1d84f57ad5fe5256799
conda-linux-64.lock:https://conda.anaconda.org/conda-forge/linux-64/nccl-2.23.4.1-h03a54cd_2.conda#a08604ac3f9c3dbd128bb24e089dee5f
conda-linux-64.lock:https://conda.anaconda.org/conda-forge/linux-64/libxgboost-2.1.2-cuda118_h09a87be_0.conda#d59c3f95f80071f24ebce434494ead0a
conda-linux-64.lock:https://conda.anaconda.org/conda-forge/noarch/py-xgboost-2.1.2-cuda118_pyh40095f8_0.conda#aa5881b02bd9555a7b06c709aa33bd20
conda-linux-64.lock:https://conda.anaconda.org/conda-forge/noarch/xgboost-2.1.2-cuda118_pyh256f914_0.conda#2dcf3e60ef65fd4cb95048f2491f6a89

```
