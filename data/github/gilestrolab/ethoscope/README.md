# https://github.com/gilestrolab/ethoscope

```console
accessories/install_ethoscope_debian.sh:    echo 'gpu_mem=256' >> "$BOOTCFG"
scripts/post-arch-device.sh:echo 'gpu_mem=256' >>  /boot/config.txt
scripts/gui_configurators/ethoscope_configurator.py:        f.write("gpu_mem=256\n")
src/ethoscope/utils/io.py:        #cmd = "SET GLOBAL innodb_file_format=Barracuda"

```
