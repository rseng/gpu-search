# https://github.com/brunettsp/myosothes

```console
README.md:2. In terminal, type the following command, replacing the arguments with their values : `./automate_analysis.sh path_to_qupath_bin path_to_conda_env_bin image_name path_to_project_dir path_to_scripts tile_size overlap_size path_to_mask_storage_directory use_gpu model_name_with_path segm_channel diameter` 
scripts/Perform_analysis/ManualVersion/runCellpose_fragmented.py:use_gpu = True
scripts/Perform_analysis/ManualVersion/runCellpose_fragmented.py:model = models.CellposeModel(gpu=use_gpu, model_type=model)   
scripts/Perform_analysis/AutomatedVersion/remove_overlap_auto.py:        print(f"Usage: {sys.argv[0]} in_path out_path use_gpu model segm_channel diameter")
scripts/Perform_analysis/AutomatedVersion/runCellpose_fragmented_auto.py:    print(f"Usage: {sys.argv[0]} in_path out_path use_gpu model segm_channel diameter")
scripts/Perform_analysis/AutomatedVersion/runCellpose_fragmented_auto.py:    use_gpu = True
scripts/Perform_analysis/AutomatedVersion/runCellpose_fragmented_auto.py:    use_gpu = False
scripts/Perform_analysis/AutomatedVersion/runCellpose_fragmented_auto.py:    print("Invalid argument for use_gpu statement.")
scripts/Perform_analysis/AutomatedVersion/runCellpose_fragmented_auto.py:model = models.CellposeModel(gpu=use_gpu, model_type=model)   
scripts/Perform_analysis/AutomatedVersion/automate_analysis.sh:    echo "usage: $0 path_to_qupath_bin path_to_conda_env_bin image_name path_to_project_dir path_to_scripts tile_size overlap_size path_to_mask_storage_directory use_gpu model_name_with_path segm_channel diameter"
scripts/Perform_analysis/AutomatedVersion/automate_analysis.sh:use_gpu=$9
scripts/Perform_analysis/AutomatedVersion/automate_analysis.sh:$path_to_conda_env_bin/python $path_to_scripts/runCellpose_fragmented_auto.py $path_to_project_dir/tiles/$(echo $image_name | cut -d'.' -f1) $path_to_mask_storage_directory $use_gpu $model_name_with_path $segm_channel $diameter

```
