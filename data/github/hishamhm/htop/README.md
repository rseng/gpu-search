# https://github.com/hishamhm/htop

```console
linux/LinuxProcessList.c:#ifndef PROCMEMINFOFILE
linux/LinuxProcessList.c:#define PROCMEMINFOFILE PROCDIR "/meminfo"
linux/LinuxProcessList.c:   FILE* file = fopen(PROCMEMINFOFILE, "r");
linux/LinuxProcessList.c:      CRT_fatalError("Cannot open " PROCMEMINFOFILE);
linux/LinuxProcessList.h:#ifndef PROCMEMINFOFILE
linux/LinuxProcessList.h:#define PROCMEMINFOFILE PROCDIR "/meminfo"

```
