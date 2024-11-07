# https://github.com/EoRImaging/FHD

```console
data_download.py:#Script that downloads GPU box files, runs cotter, updates the mwa_qc database with the uvfits
data_download.py:#locations, and deletes the GPU box files. It performs a check for existing GPU box files
data_download.py:	#Check to see if GPU box files already exist, define a preferred node if they do:
data_download.py:		gpu_loc_node = find_gpubox(obsid, save_directories[i], all_nodes)
data_download.py:		if not gpu_loc_node:
data_download.py:			node_preferred.append(gpu_loc_node)
data_download.py:			#Download the files (a uvfits or gpuboxes depending on uvfits_download_check)
data_download.py:			#If metafits does not exist in the same location as the gpubox files, set up logic to create it
data_download.py:			#Run cotter if gpubox files were downloaded
data_download.py:#Module that searches for saved GPU box files
data_download.py:def find_gpubox(obsid, save_directory, all_nodes):
data_download.py:	for gpu_loc_node in all_nodes:
data_download.py:		gpu_loc_path = gpu_loc_node + save_directory
data_download.py:		if os.path.isdir(gpu_loc_path + obsid): #checks to see if the directory exists
data_download.py:			directory_contents = os.listdir(gpu_loc_path + obsid)
data_download.py:			gpubox00 = 0
data_download.py:			gpubox01 = 0
data_download.py:					gpubox00 += 1
data_download.py:					gpubox01 += 1
data_download.py:			if gpubox00 >= 24 and (gpubox01 >= 24 or gpubox01 == 0) and flags >= 1 and metafits >= 1:
data_download.py:				print "GPU box files for obsid " + obsid + " located in " + gpu_loc_path
data_download.py:				#if gpubox00 != 24 or gpubox01 != 24 or flags != 1 or metafits != 1:
data_download.py:			     	#	print "WARNING: Directory contains extra GPU box files."
data_download.py:				return gpu_loc_node   
data_download.py:#checks if the downloads were successful, and deletes the gpubox files
data_download.py:	#Check that all gpubox files were successfully downloaded
data_download.py:			gpubox00 = 0
data_download.py:			gpubox01 = 0
data_download.py:					gpubox00 += 1
data_download.py:					gpubox01 += 1
data_download.py:			if gpubox00 < 24 or (gpubox01 < 24 and gpubox01 != 0) or flags < 1 or metafits_ppds < 1 or metafits < 1:
data_download.py:	#Delete the gpubox files
data_download.py:	delete_gpubox(obs_chunk_finished,save_path_finished)
data_download.py:	#4,1 was the same as 4,0 but for running on compressed gpubox files
data_download.py:	gpubox_path = [save_paths[i] + obs_chunk[i] + '/' + obs_chunk[i] for i in range(len(obs_chunk))]
data_download.py:		'gpubox_path=(0 ' +  " ".join(gpubox_path) + ')\n' +  \
data_download.py:		'ls ${gpubox_path[$SGE_TASK_ID]} > /dev/null\n' + \
data_download.py:		'if [ "$(ls -l ${gpubox_path[$SGE_TASK_ID]}*gpubox*_00.fits | wc -l)" -ne "24" ] ; then exit ; fi\n')	#Check to make sure gpubox files exist before queuing up 
data_download.py:		' -mem 25 -m ${metafits_path[$SGE_TASK_ID]} -o ${uvfits_path[$SGE_TASK_ID]} ${gpubox_path[$SGE_TASK_ID]}*gpubox*.fits')
data_download.py:	#Run cotter with the correct arguments, the path to the metafits, the uvfits output path, and the gpubox file paths
data_download.py:#Module for deleting gpubox files after the uvfits creation. Will check to see if a uvfits file
data_download.py:def delete_gpubox(obs_chunk,save_paths):
data_download.py:	gpubox_flag = False
data_download.py:			print "WARNING: obsid not defined in delete_gpubox. Gpubox files not deleted"
data_download.py:			print "WARNING: save_path not defined in delete_gpubox. Gpubox files not deleted"
data_download.py:		#If the uvfits file does not exist with the gpubox files, do not delete the gpubox files
data_download.py:			print "WARNING: uvfits file does not exist in the directory with the gpubox files. Gpubox files not deleted"
data_download.py:					gpubox_flag = True
data_download.py:		#If the gpubox files do not exist, exit module
data_download.py:		if not gpubox_flag:
data_download.py:			print "WARNING: there are not gpubox files to delete in " + save_path + " for obsid " + obsid
data_download.py:			print "Gpubox files in " + save_path + " for obsid " + obsid + " have been deleted."
fhd_core/deconvolution/fhd_wrap.pro:    map_fn_arr=map_fn_arr,GPU_enable=GPU_enable,image_uv_arr=image_uv_arr,weights_arr=weights_arr,$
fhd_core/obsolete/visibility_grid_GPU.pro:FUNCTION visibility_grid_GPU,visibility_array,vis_weights,obs,psf,params,weights=weights,$
fhd_core/obsolete/visibility_grid_GPU.pro:    timing=timing,polarization=polarization,mapfn_recalculate=mapfn_recalculate,silent=silent,GPU_enable=GPU_enable    
fhd_core/obsolete/visibility_grid_GPU.pro:        gpu_box_matrix=gpuPutArr(box_matrix)
fhd_core/obsolete/visibility_grid_GPU.pro:        box_arr_map=gpuGetArr(gpumatrix_multiply(gpu_box_matrix,gpu_box_matrix,/atranspose))
fhd_core/obsolete/visibility_grid_GPU.pro:        gpufree,gpu_box_matrix
fhd_core/gridding/visibility_grid.pro:    GPU_enable=GPU_enable,complex_flag=complex_flag,fi_use=fi_use,bi_use=bi_use,$
fhd_core/setup_metadata/fhd_setup.pro:;IF N_Elements(GPU_enable) EQ 0 THEN GPU_enable=0
fhd_core/setup_metadata/fhd_setup.pro:;IF Keyword_Set(GPU_enable) THEN BEGIN
fhd_core/setup_metadata/fhd_setup.pro:;    Defsysv,'GPU',exist=gpuvar_exist
fhd_core/setup_metadata/fhd_setup.pro:;    IF gpuvar_exist eq 0 THEN GPUinit
fhd_core/setup_metadata/fhd_setup.pro:;    IF !GPU.mode NE 1 THEN GPU_enable=0
dictionary.md:**uvfits_subversion**: the subversion number of the uvfits. Here are the available uvfits versions, ordered by version number and subversion number: 3,3 was used to test compressed fits; 3,4 was a rerun of 3,1 with a newer version of cotter before that version was recorded; 4,0 went back to old settings for an industrial run; 4,1 was the same as 4,0 but for running on compressed gpubox files; 5,0 was a test to phase all obs to zenith (phasing needs to be added per obs currently); 5,1 incorperates flag files and runs cotter without the bandpass applied, with all the other default settings.<br />

```
