# https://github.com/rdicosmo/parmap

```console
tests/simplescalemapfold.ml:let scale_test iter nprocmin nprocmax =
tests/simplescalemapfold.ml:  Printf.eprintf "Testing scalability with %d iterations on %d*2 to %d*2 cores\n" iter nprocmin nprocmax;
tests/simplescalemapfold.ml:  for i = nprocmin to nprocmax do
tests/utils.ml:let scale_test ?(init=(fun _ -> ())) ?(inorder=true) ?(step=1) ?chunksize ?(keeporder=false) compute sequence iter nprocmin nprocmax =
tests/utils.ml:  Printf.eprintf "Testing scalability with %d iterations on %d to %d cores, step %d\n%!" iter nprocmin nprocmax step;
tests/utils.ml:  for incr = 0 to (nprocmax-nprocmin)/step do
tests/utils.ml:    let i = nprocmin + incr in
tests/utils.ml:let array_scale_test ?(init= (fun _ -> ())) ?(inorder=true) ?(step=1) ?chunksize ?(keeporder=false) compute a iter nprocmin nprocmax =
tests/utils.ml:  Printf.eprintf "Testing scalability with %d iterations on %d to %d cores, step %d\n" iter nprocmin nprocmax step;
tests/utils.ml:  for incr = 0 to (nprocmax-nprocmin)/step do
tests/utils.ml:    let i = nprocmin + incr in
tests/utils.ml:let array_float_scale_test ?(init= (fun _ -> ())) ?(inorder=true) ?(step=1) ?chunksize compute a iter nprocmin nprocmax =
tests/utils.ml:  Printf.eprintf "Testing scalability with %d iterations on %d to %d cores, step %d\n" iter nprocmin nprocmax step;
tests/utils.ml:  for incr = 0 to (nprocmax-nprocmin)/step do
tests/utils.ml:    let i = nprocmin + incr in
tests/simplescalefold.ml:let scale_test iter nprocmin nprocmax =
tests/simplescalefold.ml:  Printf.eprintf "Testing scalability with %d iterations on %d*2 to %d*2 cores\n" iter nprocmin nprocmax;
tests/simplescalefold.ml:  for i = nprocmin to nprocmax do

```
