# https://github.com/pyccel/pyccel

```console
.dict_custom.txt:NVIDIA
install_scripts/hook.py:                'lapack', 'mpi', 'openacc', 'openmp']
docs/compiler.md:-   **NVIDIA** : `nvc` / `nvfort`
docs/compiler.md:In addition, for each accelerator (`mpi`/`openmp`/`openacc`/`python`) that you will use the JSON file must define the following:
docs/quickstart.md:In most cases this is written in a statically compiled language like Fortran/C/C++, and it uses SIMD vectorisation, parallel multi-threading, MPI parallelisation, GPU offloading, etc.
docs/quickstart.md:We are also working on supporting [MPI](https://en.wikipedia.org/wiki/Open_MPI), [LAPACK](https://en.wikipedia.org/wiki/LAPACK)/[BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms), and [OpenACC](https://en.wikipedia.org/wiki/OpenACC).
docs/quickstart.md:In the future we plan to support GPU programming with [CUDA](https://en.wikipedia.org/wiki/CUDA) and [task-based parallelism](https://en.wikipedia.org/wiki/Task_parallelism).
developer_docs/wrapper_stage.md:Optional arguments are passed as C pointers. An if/else block then determines whether the pointer is assigned or not. This can be quite lengthy, however it is unavoidable for compilation with Intel, or NVIDIA. It is also unavoidable for arrays as it is important not to index an array (to access the strides) which is not present.
CHANGELOG.md:-   #929 : Allow optional variables when compiling with Intel or NVIDIA.
tests/internal/test_internal.py:# TODO test if compiler exists before execute_pyccelning mpi, openacc
tests/internal/test_internal.py:#@pytest.mark.parametrize("f", get_files_from_folder('openacc'))
tests/internal/test_internal.py:#def test_openacc():
tests/internal/test_internal.py:#    execute_pyccel(f, compiler='pgfortran', accelerator='openacc')
tests/internal/test_internal.py:#    print('***  TESTING INTERNAL/OPENACC ***')
tests/internal/test_internal.py:#    for f in get_files_from_folder('openacc'):
tests/internal/test_internal.py:#        test_openacc(f)
tests/internal/scripts/openacc/ex1.py:from pyccel.stdlib.internal.openacc import acc_get_device_type
tests/internal/scripts/openacc/ex1.py:from pyccel.stdlib.internal.openacc import acc_get_num_devices
tests/internal/scripts/openacc/ex1.py:    print(' number of available OpenACC devices :', dev_num)
tests/internal/scripts/openacc/ex1.py:    print(' type of available OpenACC devices   :', dev_kind)
tests/parser/test_openacc.py:from pyccel.parser.syntax.openacc import parse
tests/external/test_external.py:# TODO test if compiler exists before running mpi, openacc
tests/run_tests.bat:python -m pytest ../tests/parser/test_openacc.py
tests/macro/test_macro.py:# TODO test if compiler exists before execute_pyccelning mpi, openacc
tests/macro/test_macro.py:#@pytest.mark.parametrize("f", get_files_from_folder('openacc'))
tests/macro/test_macro.py:#def test_openacc():
tests/macro/test_macro.py:#    execute_pyccel(f, compiler='pgfortran', accelerator='openacc')
tests/macro/test_macro.py:#    print('***  TESTING MACRO/OPENACC ***')
tests/macro/test_macro.py:#    for f in get_files_from_folder('openacc'):
tests/macro/test_macro.py:#        test_openacc(f)
tests/epyccel/test_epyccel_openmp.py:@pytest.mark.skip("Compiling is not fully managed for GPU commands. See #798")
tests/epyccel/test_epyccel_openmp.py:@pytest.mark.skip("Compiling is not fully managed for GPU commands. See #798")
tests/epyccel/test_epyccel_openmp.py:@pytest.mark.skip("Compiling is not fully managed for GPU commands. See #798")
tests/epyccel/test_epyccel_openmp.py:@pytest.mark.skip("Compiling is not fully managed for GPU commands. See #798")
tests/run_tests_py3.sh:#python3 "$SCRIPT_DIR"/parser/test_openacc.py
pyccel/parser/semantic.py:            # in some cases (blas, lapack and openacc level-0)
pyccel/parser/syntax/openacc.py:class Openacc(object):
pyccel/parser/syntax/openacc.py:    """Class for Openacc syntax."""
pyccel/parser/syntax/openacc.py:        Constructor for Openacc.
pyccel/parser/syntax/openacc.py:class OpenaccStmt(AccBasic):
pyccel/parser/syntax/openacc.py:        super(OpenaccStmt, self).__init__(**kwargs)
pyccel/parser/syntax/openacc.py:            print("> OpenaccStmt: expr")
pyccel/parser/syntax/openacc.py:acc_classes = [Openacc, OpenaccStmt] + acc_directives + acc_clauses
pyccel/parser/syntax/openacc.py:grammar = join(this_folder, '../grammar/openacc.tx')
pyccel/parser/syntax/openacc.py:    """ Parse openacc pragmas
pyccel/parser/syntax/openacc.py:        if isinstance(stmt, OpenaccStmt):
pyccel/parser/syntactic.py:from pyccel.parser.syntax.openacc import parse as acc_parse
pyccel/parser/syntactic.py:                    errors.report(f"Invalid OpenACC header. {e.message}",
pyccel/parser/grammar/pyccel.tx:import openacc
pyccel/parser/grammar/pyccel.tx:  | 'openacc'
pyccel/parser/grammar/openacc.tx:// The following grammar is compatible with OpenACC 2.5
pyccel/parser/grammar/openacc.tx:Openacc:
pyccel/parser/grammar/openacc.tx:  statements*=OpenaccStmt
pyccel/parser/grammar/openacc.tx:OpenaccStmt: '#$' 'acc' stmt=AccConstructOrDirective;
pyccel/stdlib/internal/openacc.pyh:# pyccel header for OpenACC.
pyccel/stdlib/internal/openacc.pyh:# OpenACC directives and Constructs are handled by the parser (see openacc.tx) and are parts of the Pyccel language.
pyccel/stdlib/internal/openacc.pyh:# We only list here what can not be described in the openacc grammar.
pyccel/stdlib/internal/openacc.pyh:#$ header metavar module_name='openacc'
pyccel/compilers/default_compilers.py:              'openacc': {
pyccel/compilers/default_compilers.py:              'openacc': {
pyccel/compilers/default_compilers.py:              'openacc': {
pyccel/compilers/default_compilers.py:              'openacc': {
pyccel/compilers/default_compilers.py:              'family': 'nvidia',
pyccel/compilers/default_compilers.py:            'openacc': {
pyccel/compilers/default_compilers.py:            'openacc': {
pyccel/compilers/default_compilers.py:            'openacc': {
pyccel/compilers/default_compilers.py:            'openacc': {
pyccel/compilers/default_compilers.py:            'family': 'nvidia',
pyccel/compilers/default_compilers.py:                       ('nvidia', 'c') : nvc_info,
pyccel/compilers/default_compilers.py:                       ('nvidia', 'fortran') : nvfort_info}
pyccel/compilers/default_compilers.py:vendors = ('GNU','intel','PGI','nvidia')
pyccel/codegen/pipeline.py:        Tool used to accelerate the code (e.g., OpenMP, OpenACC).
pyccel/codegen/compiling/basic.py:        Tool used to accelerate the code (e.g. openmp openacc).
pyccel/codegen/compiling/basic.py:        openmp, openacc, python.
pyccel/codegen/utilities.py:        Tool used to accelerate the code (e.g. openmp openacc).
pyccel/codegen/printing/fcode.py:    #                   OpenACC statements
pyccel/codegen/printing/fcode.py:        # ... TODO adapt get_statement to have continuation with OpenACC
pyccel/codegen/printing/fcode.py:        # ... TODO adapt get_statement to have continuation with OpenACC
pyccel/commands/console.py:def pyccel(files=None, mpi=None, openmp=None, openacc=None, output_dir=None, compiler=None):
pyccel/commands/console.py:    options to specify the files to be translated, enable support for MPI, OpenMP, and OpenACC, specify the output
pyccel/commands/console.py:    openacc : bool, optional
pyccel/commands/console.py:        Enable OpenACC support, by default None.
pyccel/commands/console.py:    group.add_argument('--openacc', action='store_true', \
pyccel/commands/console.py:                       help='uses openacc')
pyccel/commands/console.py:    if not openacc:
pyccel/commands/console.py:        openacc = args.openacc
pyccel/commands/console.py:    if openacc:
pyccel/commands/console.py:        accelerators.append("openacc")
pyccel/commands/epyccel.py:        (currently supported: 'mpi', 'openmp', 'openacc').
pyccel/commands/pyccel_init.py:            'lapack.pyh', 'mpi.pyh', 'openacc.pyh', 'openmp.pyh']

```
