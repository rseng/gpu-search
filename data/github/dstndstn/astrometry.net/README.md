# https://github.com/dstndstn/astrometry.net

```console
include/astrometry/plotstuff.h:    bl* cairocmds;
plot/plotstuff.c:struct cairocmd {
plot/plotstuff.c:typedef struct cairocmd cairocmd_t;
plot/plotstuff.c:    args->cairocmds = bl_new(256, sizeof(cairocmd_t));
plot/plotstuff.c:static void cairocmd_init(cairocmd_t* cmd) {
plot/plotstuff.c:    memset(cmd, 0, sizeof(cairocmd_t));
plot/plotstuff.c:static void cairocmd_clear(cairocmd_t* cmd) {
plot/plotstuff.c:static void add_cmd(plot_args_t* pargs, cairocmd_t* cmd) {
plot/plotstuff.c:    bl_append(pargs->cairocmds, cmd);
plot/plotstuff.c:static void set_cmd_args(plot_args_t* pargs, cairocmd_t* cmd) {
plot/plotstuff.c:    cairocmd_t cmd;
plot/plotstuff.c:    cairocmd_init(&cmd);
plot/plotstuff.c:    cairocmd_t cmd;
plot/plotstuff.c:    cairocmd_init(&cmd);
plot/plotstuff.c:    cairocmd_t cmd;
plot/plotstuff.c:    cairocmd_init(&cmd);
plot/plotstuff.c:    logverb("Plotting %zu stacked plot commands.\n", bl_size(pargs->cairocmds));
plot/plotstuff.c:        for (i=0; i<bl_size(pargs->cairocmds); i++) {
plot/plotstuff.c:            cairocmd_t* cmd = bl_access(pargs->cairocmds, i);
plot/plotstuff.c:    for (i=0; i<bl_size(pargs->cairocmds); i++) {
plot/plotstuff.c:        cairocmd_t* cmd = bl_access(pargs->cairocmds, i);
plot/plotstuff.c:        cairocmd_clear(cmd);
plot/plotstuff.c:    bl_remove_all(pargs->cairocmds);
plot/plotstuff.c:    bl_free(pargs->cairocmds);

```
