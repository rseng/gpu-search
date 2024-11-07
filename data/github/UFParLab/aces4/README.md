# https://github.com/UFParLab/aces4

```console
config.h.cmake:/* Defined if CUDA & CUBLAS are present */
config.h.cmake:#cmakedefine HAVE_CUDA
configure.ac:# CUDA
configure.ac:# largely adapted from http://wili.cc/blog/cuda-m4.html
configure.ac:AC_ARG_WITH(cuda, AC_HELP_STRING([--with-cuda=PATH],[Enable CUDA]), 
configure.ac:            [cuda_enable="yes"; cuda_prefix=$withval], 
configure.ac:            [cuda_enable="no"])
configure.ac:if test "x$cuda_enable" == "xyes"; then
configure.ac:    if test "x$cuda_prefix" != "xyes"; then
configure.ac:        CUDA_CFLAGS="-I$cuda_prefix/include"
configure.ac:        CFLAGS="$CUDA_CFLAGS $CFLAGS"
configure.ac:        CUDA_LDFLAGS="-L$cuda_prefix/lib64"
configure.ac:        LDFLAGS="$CUDA_LDFLAGS $LDFLAGS"
configure.ac:        NVCCLDFLAGS=" -L${cuda_prefix}/lib64 -lcudart -lcublas"
configure.ac:        NVCCCPPFLAGS=" -I${cuda_prefix}/include -arch=sm_13"
configure.ac:        NVCCLDFLAGS=" -lcudart -lcublas "
configure.ac:    AC_CHECK_HEADER([cuda.h], [], AC_MSG_FAILURE([Couldnt find cuda.h]), [#include <cuda.h>])
configure.ac:    AC_CHECK_LIB([cudart], [cudaDeviceReset], [AC_MSG_NOTICE([Founc cudart])], AC_MSG_FAILURE([Couldnt find libcudart]))
configure.ac:    AC_DEFINE(HAVE_CUDA, 1, [Defined if CUDA & CUBLAS are present])
configure.ac:    AC_MSG_NOTICE([Using CUDA])
configure.ac:AM_CONDITIONAL([WORKING_NVCC], [test "x$cuda_enable" = "xyes"])
test/test_mpi_simple.cpp:///* This test does contraction on GPU is available and implemented */
test/test_basic_sial.cpp:/* This test does contraction on GPU is available and implemented */
test/test_simple.cpp_old:TEST(Sial,gpu_contraction_small){
test/test_simple.cpp_old:	std::string job("gpu_contraction_small_test");
test/test_simple.cpp_old:TEST(Sial,gpu_sum_op){
test/test_simple.cpp_old:	std::string job("gpu_sum_op_test");
test/test_simple.cpp_old:// Tests whether all GPU ops compile
test/test_simple.cpp_old:// We may want to test is data is available on GPU after gpu_put
test/test_simple.cpp_old:// and on host after gpu_get
test/test_simple.cpp_old:TEST(Sial,gpu_ops){
test/test_simple.cpp_old:	std::string job("gpu_ops");
test/test_simple.cpp_old:TEST(Sial,gpu_contract_to_scalar){
test/test_simple.cpp_old:	std::string job("gpu_contract_to_scalar");
test/test_simple.cpp_old:TEST(Sial,gpu_transpose_tmp){
test/test_simple.cpp_old:	std::string job("gpu_transpose_tmp");
test/test_simple.cpp_old:TEST(Sial,gpu_self_multiply_op){
test/test_simple.cpp_old:	std::string job("gpu_self_multiply_test");
test/test_simple.cpp_old:TEST(Sial,gpu_contraction_predefined){
test/test_simple.cpp_old:	std::string job("gpu_contraction_predefined_test");
test/test_simple.cpp:/* This test does contraction on GPU is available and implemented */
.ycm_extra_conf.py:'-I', './src/sip/cuda',
tool/gtksourceview/share/gtksourceview-3.0/language-specs/sialx.lang:      <keyword>gpu_on</keyword>
tool/gtksourceview/share/gtksourceview-3.0/language-specs/sialx.lang:      <keyword>gpu_off</keyword>
tool/gtksourceview/share/gtksourceview-3.0/language-specs/sialx.lang:      <keyword>gpu_allocate</keyword>
tool/gtksourceview/share/gtksourceview-3.0/language-specs/sialx.lang:      <keyword>gpu_free</keyword>
tool/gtksourceview/share/gtksourceview-3.0/language-specs/sialx.lang:      <keyword>gpu_put</keyword>
tool/gtksourceview/share/gtksourceview-3.0/language-specs/sialx.lang:      <keyword>gpu_get</keyword>
tool/vim/syntax/sial.vim:syn keyword sialSuperInstruction allocate deallocate create delete put get prepare request collective destroy create delete println print print_index print_scalar execute gpu_on gpu_allocate gpu_free gpu_put gpu_get gpu_off set_persistent restore_persistent
CMakeLists.txt:# HAVE_CUDA Options - Whether to use CUDA accelerated super instructions
CMakeLists.txt:#option(HAVE_CUDA "Whether to use CUDA accelerated super instructions" OFF)
CMakeLists.txt:## Check if CUDA available on system
CMakeLists.txt:#if (HAVE_CUDA)
CMakeLists.txt:#    find_package(CUDA REQUIRED)
CMakeLists.txt:# CUDA - Conditional compile for CUDA super instructions
CMakeLists.txt:if (HAVE_CUDA AND CUDA_FOUND)
CMakeLists.txt:    set(ACES_CUDA_FILES
CMakeLists.txt:        src/sip/cuda/gpu_super_instructions.cu
CMakeLists.txt:        src/sip/cuda/cuda_check.h)
CMakeLists.txt:    set(ACES_CUDA_FILES "")
CMakeLists.txt:    src/sip/cuda/gpu_super_instructions.h;
CMakeLists.txt:    ${CMAKE_SOURCE_DIR}/src/sip/cuda;
CMakeLists.txt:# CUDA Super instructions
CMakeLists.txt:if (CUDA_FOUND)
CMakeLists.txt:    set(CUDA_HOST_COMPILATION_CPP ON)
CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${INCLUDE_DIRS})
CMakeLists.txt:    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_13) 
CMakeLists.txt:    CUDA_ADD_LIBRARY(cudasuperinstructions ${ACES_CUDA_FILES})
CMakeLists.txt:    CUDA_ADD_CUBLAS_TO_TARGET(cudasuperinstructions)
CMakeLists.txt:    set(TOLINK_LIBRARIES ${TOLINK_LIBRARIES} cudasuperinstructions)
CMakeLists.txt:if(HAVE_CUDA)
CMakeLists.txt:	add_dependencies(aces4 tensordil superinstructions cudasuperinstructions)
CMakeLists.txt:	add_dependencies(print_siptables tensordil superinstructions cudasuperinstructions)
CMakeLists.txt:	add_dependencies(print_array_info tensordil superinstructions cudasuperinstructions)
CMakeLists.txt:	add_dependencies(print_init_file tensordil superinstructions cudasuperinstructions)
CMakeLists.txt:	add_dependencies(print_worker_checkpoint tensordil superinstructions cudasuperinstructions)
CMakeLists.txt:    src/sialx/test/gpu_contraction_small_test.sialx;
CMakeLists.txt:    src/sialx/test/gpu_ops.sialx;
CMakeLists.txt:    src/sialx/test/gpu_sum_op_test.sialx;
CMakeLists.txt:    src/sialx/test/gpu_contract_to_scalar.sialx;
CMakeLists.txt:    src/sialx/test/gpu_transpose_tmp.sialx;
CMakeLists.txt:    src/sialx/test/gpu_self_multiply_test.sialx;
CMakeLists.txt:    src/sialx/test/gpu_contraction_predefined_test.sialx;
Makefile.am:./src/sip/cuda/gpu_super_instructions.h\
Makefile.am:# CUDA
Makefile.am:    ACES_SOURCEFILES += ./src/sip/cuda/gpu_super_instructions.cu ./src/sip/cuda/cuda_check.h
Makefile.am:-I${top_srcdir}/src/sip/cuda\
Makefile.am:./src/sialx/test/gpu_contraction_small_test.siox\
Makefile.am:./src/sialx/test/gpu_ops.siox\
Makefile.am:./src/sialx/test/gpu_sum_op_test.siox\
Makefile.am:./src/sialx/test/gpu_contract_to_scalar.siox\
Makefile.am:./src/sialx/test/gpu_transpose_tmp.siox\
Makefile.am:./src/sialx/test/gpu_self_multiply_test.siox\
Makefile.am:./src/sialx/test/gpu_contraction_predefined_test.siox\
Makefile.am:#./src/sialx/gpu_contraction_small_test.sialx\
Makefile.am:#./src/sialx/gpu_ops.sialx\
Makefile.am:#./src/sialx/gpu_sum_op_test.sialx\
Makefile.am:#./src/sialx/gpu_contract_to_scalar.sialx\
Makefile.am:#./src/sialx/gpu_transpose_tmp.sialx\
Makefile.am:#./src/sialx/gpu_self_multiply_test.sialx\
Makefile.am:#./src/sialx/gpu_contraction_predefined_test.sialx\
Makefile.am:# CUDA Compile Rule
.cproject:									<entry excluding="*.h" flags="VALUE_WORKSPACE_PATH" kind="outputPath" name="CUDA"/>
.cproject:									<listOptionValue builtIn="false" value="&quot;${workspace_loc:/aces4/src/sip/cuda}&quot;"/>
.cproject:									<listOptionValue builtIn="false" value="&quot;${workspace_loc:/aces4/src/sip/cuda}&quot;"/>
.cproject:									<entry excluding="*.h" flags="VALUE_WORKSPACE_PATH" kind="outputPath" name="CUDA"/>
.cproject:									<listOptionValue builtIn="false" value="&quot;${workspace_loc:/aces4/src/sip/cuda}&quot;"/>
.cproject:									<listOptionValue builtIn="false" value="&quot;${workspace_loc:/aces4/src/sip/cuda}&quot;"/>
.cproject:									<entry excluding="*.h" flags="VALUE_WORKSPACE_PATH" kind="outputPath" name="CUDA"/>
.cproject:									<listOptionValue builtIn="false" value="&quot;${workspace_loc:/aces4/src/sip/cuda}&quot;"/>
.cproject:									<listOptionValue builtIn="false" value="&quot;${workspace_loc:/aces4/src/sip/cuda}&quot;"/>
.cproject:		<configuration configurationName="Build (CUDA)">
.cproject:		<configuration configurationName="CUDA">
src/sip/worker/interpreter.h:	 * Records whether or not we are executing in a section of code where gpu_on has been invoked
src/sip/worker/interpreter.h:	 * TODO  is nesting of gpu sections allowed?  If so, this needs to be an int.  If not, need check.
src/sip/worker/interpreter.h:	bool gpu_enabled_;
src/sip/worker/interpreter.h:	// GPU
src/sip/worker/interpreter.h:	sip::Block::BlockPtr get_gpu_block(char intent, sip::BlockId&,
src/sip/worker/interpreter.h:	sip::Block::BlockPtr get_gpu_block(char intent, sip::BlockSelector&,
src/sip/worker/interpreter.h:	sip::Block::BlockPtr get_gpu_block(char intent, bool contiguous_allowed =
src/sip/worker/interpreter.h:	sip::Block::BlockPtr get_gpu_block_from_selector_stack(char intent,
src/sip/worker/interpreter.cpp:// For CUDA Super Instructions
src/sip/worker/interpreter.cpp:#ifdef HAVE_CUDA
src/sip/worker/interpreter.cpp:#include "gpu_super_instructions.h"
src/sip/worker/interpreter.cpp:	gpu_enabled_ = false;
src/sip/worker/interpreter.cpp:#ifdef HAVE_CUDA
src/sip/worker/interpreter.cpp:	_init_gpu(&devid, &rank);
src/sip/worker/interpreter.cpp:			//#ifdef HAVE_CUDA
src/sip/worker/interpreter.cpp:			//		if (gpu_enabled_) {
src/sip/worker/interpreter.cpp:			//			rhs_block = get_gpu_block('r', rhs_selector, rhs_blockid);
src/sip/worker/interpreter.cpp:			//			lhs_block = get_gpu_block('w', lhs_selector, lhs_blockid);
src/sip/worker/interpreter.cpp:			//#ifdef HAVE_CUDA
src/sip/worker/interpreter.cpp:			//				if (gpu_enabled_) {
src/sip/worker/interpreter.cpp:			//					lhs_block->gpu_copy_data(rhs_block);
src/sip/worker/interpreter.cpp:			//#ifdef HAVE_CUDA
src/sip/worker/interpreter.cpp:			//				if (gpu_enabled_) {
src/sip/worker/interpreter.cpp:			//					_gpu_permute(lhs_block->get_gpu_data(), lhs_rank, lhs_block->shape().segment_sizes_, lhs_selector.index_ids_,
src/sip/worker/interpreter.cpp:			//							rhs_block->get_gpu_data(), lhs_rank, rhs_block->shape().segment_sizes_ , rhs_selector.index_ids_);
src/sip/worker/interpreter.cpp:			//#ifdef HAVE_CUDA
src/sip/worker/interpreter.cpp:			//			if (gpu_enabled_) {   //FIXME.  This looks OK, but need to double check
src/sip/worker/interpreter.cpp:			//				sip::Block::BlockPtr g_lhs_block = get_gpu_block('w');
src/sip/worker/interpreter.cpp:			//				g_lhs_block->gpu_fill(scalar_value(rhs));
src/sip/worker/interpreter.cpp:		case gpu_on_op: {
src/sip/worker/interpreter.cpp:		case gpu_off_op: {
src/sip/worker/interpreter.cpp:			/* FIX ME:  other gpu instruction omitted for now */
src/sip/worker/interpreter.cpp://old CUDA STUFF MOVED TO BOTTOM OF FILE
src/sip/worker/interpreter.cpp:////#ifdef HAVE_CUDA
src/sip/worker/interpreter.cpp://	if (gpu_enabled_) {
src/sip/worker/interpreter.cpp://		std::cout<<"Contraction on the GPU at line "<<line_number()<<std::endl;
src/sip/worker/interpreter.cpp://		sip::Block::BlockPtr g_rblock = get_gpu_block('r');
src/sip/worker/interpreter.cpp://		sip::Block::BlockPtr g_lblock = get_gpu_block('r');
src/sip/worker/interpreter.cpp://			//g_dblock->gpu_data_ = _gpu_allocate(1);
src/sip/worker/interpreter.cpp://			g_dblock->allocate_gpu_data();
src/sip/worker/interpreter.cpp://			g_dblock = get_gpu_block('w');
src/sip/worker/interpreter.cpp://		_gpu_contract(g_dblock->get_gpu_data(), drank, g_dblock->shape().segment_sizes_, g_dselected_index_ids,
src/sip/worker/interpreter.cpp://				g_lblock->get_gpu_data(), lrank, g_lblock->shape().segment_sizes_, lselector.index_ids_,
src/sip/worker/interpreter.cpp://				g_rblock->get_gpu_data(), rrank, g_rblock->shape().segment_sizes_, rselector.index_ids_);
src/sip/worker/interpreter.cpp://			_gpu_device_to_host(&h_dbl, g_dblock->get_gpu_data(), 1);
src/sip/worker/interpreter.cpp:	//#ifdef HAVE_CUDA
src/sip/worker/interpreter.cpp:	//		if (gpu_enabled_) {
src/sip/worker/interpreter.cpp:	//			data_manager_.block_manager_.lazy_gpu_read_on_host(cblock);
src/sip/worker/interpreter.cpp:	//		// TODO FIXME GPU ?????????????????
src/sip/worker/interpreter.cpp:#ifdef HAVE_CUDA
src/sip/worker/interpreter.cpp:		if (gpu_enabled_) {
src/sip/worker/interpreter.cpp:			// Read from GPU back into host to do write back.
src/sip/worker/interpreter.cpp:			data_manager_.block_manager_.lazy_gpu_read_on_host(cblock);
src/sip/worker/interpreter.cpp://#ifdef HAVE_CUDA
src/sip/worker/interpreter.cpp://			if (gpu_enabled_) {
src/sip/worker/interpreter.cpp://				sip::Block::BlockPtr g_lhs_block = get_gpu_block('w');
src/sip/worker/interpreter.cpp://				g_lhs_block->gpu_fill(scalar_value(rhs));
src/sip/worker/interpreter.cpp://#ifdef HAVE_CUDA
src/sip/worker/interpreter.cpp://		if (gpu_enabled_) {
src/sip/worker/interpreter.cpp://			rhs_block = get_gpu_block('r', rhs_selector, rhs_blockid);
src/sip/worker/interpreter.cpp://			lhs_block = get_gpu_block('w', lhs_selector, lhs_blockid);
src/sip/worker/interpreter.cpp://#ifdef HAVE_CUDA
src/sip/worker/interpreter.cpp://				if (gpu_enabled_) {
src/sip/worker/interpreter.cpp://					lhs_block->gpu_copy_data(rhs_block);
src/sip/worker/interpreter.cpp://#ifdef HAVE_CUDA
src/sip/worker/interpreter.cpp://				if (gpu_enabled_) {
src/sip/worker/interpreter.cpp://					_gpu_permute(lhs_block->get_gpu_data(), lhs_rank, lhs_block->shape().segment_sizes_, lhs_selector.index_ids_,
src/sip/worker/interpreter.cpp://							rhs_block->get_gpu_data(), lhs_rank, rhs_block->shape().segment_sizes_ , rhs_selector.index_ids_);
src/sip/worker/interpreter.cpp://#ifdef HAVE_CUDA
src/sip/worker/interpreter.cpp://	if (gpu_enabled_) {
src/sip/worker/interpreter.cpp://		std::cout<<"Contraction on the GPU at line "<<line_number()<<std::endl;
src/sip/worker/interpreter.cpp://		sip::Block::BlockPtr g_rblock = get_gpu_block('r');
src/sip/worker/interpreter.cpp://		sip::Block::BlockPtr g_lblock = get_gpu_block('r');
src/sip/worker/interpreter.cpp://			//g_dblock->gpu_data_ = _gpu_allocate(1);
src/sip/worker/interpreter.cpp://			g_dblock->allocate_gpu_data();
src/sip/worker/interpreter.cpp://			g_dblock = get_gpu_block('w');
src/sip/worker/interpreter.cpp://		_gpu_contract(g_dblock->get_gpu_data(), drank, g_dblock->shape().segment_sizes_, g_dselected_index_ids,
src/sip/worker/interpreter.cpp://				g_lblock->get_gpu_data(), lrank, g_lblock->shape().segment_sizes_, lselector.index_ids_,
src/sip/worker/interpreter.cpp://				g_rblock->get_gpu_data(), rrank, g_rblock->shape().segment_sizes_, rselector.index_ids_);
src/sip/worker/interpreter.cpp://			_gpu_device_to_host(&h_dbl, g_dblock->get_gpu_data(), 1);
src/sip/worker/interpreter.cpp://#ifdef HAVE_CUDA
src/sip/worker/interpreter.cpp://		if (gpu_enabled_) {
src/sip/worker/interpreter.cpp://			sip::Block::BlockPtr dblock = get_gpu_block('u', did);
src/sip/worker/interpreter.cpp://			dblock->gpu_scale(lval);
src/sip/worker/interpreter.cpp://	//#ifdef HAVE_CUDA
src/sip/worker/interpreter.cpp://	//		if (gpu_enabled_) {
src/sip/worker/interpreter.cpp://	//			data_manager_.block_manager_.lazy_gpu_read_on_host(cblock);
src/sip/worker/interpreter.cpp://	//		// TODO FIXME GPU ?????????????????
src/sip/worker/interpreter.cpp://#ifdef HAVE_CUDA
src/sip/worker/interpreter.cpp://		if (gpu_enabled_) {
src/sip/worker/interpreter.cpp://			// Read from GPU back into host to do write back.
src/sip/worker/interpreter.cpp://			data_manager_.block_manager_.lazy_gpu_read_on_host(cblock);
src/sip/worker/interpreter.cpp://#ifdef HAVE_CUDA
src/sip/worker/interpreter.cpp://sip::Block::BlockPtr Interpreter::get_gpu_block(char intent, sip::BlockId& id, bool contiguous_allowed) {
src/sip/worker/interpreter.cpp://	return get_gpu_block(intent, selector, id, contiguous_allowed);
src/sip/worker/interpreter.cpp://sip::Block::BlockPtr Interpreter::get_gpu_block(char intent, bool contiguous_allowed) {
src/sip/worker/interpreter.cpp://	return get_gpu_block(intent, block_id, contiguous_allowed);
src/sip/worker/interpreter.cpp://sip::Block::BlockPtr Interpreter::get_gpu_block(char intent, sip::BlockSelector& selector, sip::BlockId& id,
src/sip/worker/interpreter.cpp://				data_manager_.block_manager_.lazy_gpu_read_on_device(block);
src/sip/worker/interpreter.cpp://				block = data_manager_.block_manager_.get_gpu_block_for_reading(id);
src/sip/worker/interpreter.cpp://				data_manager_.block_manager_.lazy_gpu_write_on_device(block, id, shape);
src/sip/worker/interpreter.cpp://				block = data_manager_.block_manager_.get_gpu_block_for_writing(id, is_scope_extent);
src/sip/worker/interpreter.cpp://				data_manager_.block_manager_.lazy_gpu_update_on_device(block);
src/sip/worker/interpreter.cpp://				block = data_manager_.block_manager_.get_gpu_block_for_updating(id);
src/sip/worker/interpreter.cpp://sip::Block::BlockPtr Interpreter::get_gpu_block_from_selector_stack(char intent,
src/sip/worker/interpreter.cpp://	sip::check(selector.rank_ != 0, "Contiguous arrays not supported for GPU", line_number());
src/sip/worker/interpreter.cpp://			block = data_manager_.block_manager_.get_gpu_block_for_reading(id);
src/sip/worker/interpreter.cpp://			block = data_manager_.block_manager_.get_gpu_block_for_writing(id,
src/sip/worker/interpreter.cpp://			block = data_manager_.block_manager_.get_gpu_block_for_updating(id); //w and u same for contig
src/sip/worker/interpreter.cpp://#ifdef HAVE_CUDA
src/sip/worker/interpreter.cpp://	if (gpu_enabled_) {
src/sip/worker/interpreter.cpp://		std::cout<<"Contraction on the GPU at line "<<line_number()<<std::endl;
src/sip/worker/interpreter.cpp://		sip::Block::BlockPtr g_rblock = get_gpu_block('r');
src/sip/worker/interpreter.cpp://		sip::Block::BlockPtr g_lblock = get_gpu_block('r');
src/sip/worker/interpreter.cpp://			//g_dblock->gpu_data_ = _gpu_allocate(1);
src/sip/worker/interpreter.cpp://			g_dblock->allocate_gpu_data();
src/sip/worker/interpreter.cpp://			g_dblock = get_gpu_block('w');
src/sip/worker/interpreter.cpp://		_gpu_contract(g_dblock->get_gpu_data(), drank, g_dblock->shape().segment_sizes_, g_dselected_index_ids,
src/sip/worker/interpreter.cpp://				g_lblock->get_gpu_data(), lrank, g_lblock->shape().segment_sizes_, lselector.index_ids_,
src/sip/worker/interpreter.cpp://				g_rblock->get_gpu_data(), rrank, g_rblock->shape().segment_sizes_, rselector.index_ids_);
src/sip/worker/interpreter.cpp://			_gpu_device_to_host(&h_dbl, g_dblock->get_gpu_data(), 1);
src/sip/dynamic_data/block_manager.h: * should suffice for standard aces.  Needs to be enhanced to support GPUs.
src/sip/dynamic_data/block_manager.h:#ifdef HAVE_CUDA
src/sip/dynamic_data/block_manager.h:	// GPU
src/sip/dynamic_data/block_manager.h:	Block::BlockPtr get_gpu_block_for_writing(const BlockId& id, bool is_scope_extent=false);
src/sip/dynamic_data/block_manager.h:	Block::BlockPtr get_gpu_block_for_updating(const BlockId& id);
src/sip/dynamic_data/block_manager.h:	Block::BlockPtr get_gpu_block_for_reading(const BlockId& id);
src/sip/dynamic_data/block_manager.h:	Block::BlockPtr get_gpu_block_for_accumulate(const BlockId& id, bool is_scope_extent=false);
src/sip/dynamic_data/block_manager.h:	void lazy_gpu_read_on_device(const Block::BlockPtr& blk);
src/sip/dynamic_data/block_manager.h:	void lazy_gpu_write_on_device(Block::BlockPtr& blk, const BlockId &id, const BlockShape& shape);
src/sip/dynamic_data/block_manager.h:	void lazy_gpu_update_on_device(const Block::BlockPtr& blk);
src/sip/dynamic_data/block_manager.h:	void lazy_gpu_read_on_host(const Block::BlockPtr& blk);
src/sip/dynamic_data/block_manager.h:	void lazy_gpu_write_on_host(Block::BlockPtr& blk, const BlockId &id, const BlockShape& shape);
src/sip/dynamic_data/block_manager.h:	void lazy_gpu_update_on_host(const Block::BlockPtr& blk);
src/sip/dynamic_data/block_manager.h:	 * Creates and returns a new block on the gpu with the given shape and records it
src/sip/dynamic_data/block_manager.h:	Block::BlockPtr create_gpu_block(const BlockId&, const BlockShape& shape);
src/sip/dynamic_data/block.cpp:#include "gpu_super_instructions.h"
src/sip/dynamic_data/block.cpp:	gpu_data_ = NULL;
src/sip/dynamic_data/block.cpp:	status_[Block::onGPU] = false;
src/sip/dynamic_data/block.cpp:	status_[Block::dirtyOnGPU] = false;
src/sip/dynamic_data/block.cpp:	gpu_data_ = NULL;
src/sip/dynamic_data/block.cpp:	status_[Block::onGPU] = false;
src/sip/dynamic_data/block.cpp:	status_[Block::dirtyOnGPU] = false;
src/sip/dynamic_data/block.cpp:	gpu_data_ = NULL;
src/sip/dynamic_data/block.cpp:	status_[Block::onGPU] = false;
src/sip/dynamic_data/block.cpp:	status_[Block::dirtyOnGPU] = false;
src/sip/dynamic_data/block.cpp:#ifdef HAVE_CUDA
src/sip/dynamic_data/block.cpp:	if (gpu_data_){
src/sip/dynamic_data/block.cpp:		std::cout<<"Now freeing gpu_data"<<std::endl;
src/sip/dynamic_data/block.cpp:		_gpu_free(gpu_data_);
src/sip/dynamic_data/block.cpp:		gpu_data_ = NULL;
src/sip/dynamic_data/block.cpp:#endif //HAVE_CUDA
src/sip/dynamic_data/block.cpp://	status_[Block::onGPU] = false;
src/sip/dynamic_data/block.cpp://	status_[Block::dirtyOnGPU] = false;
src/sip/dynamic_data/block.cpp:/**						GPU Specific methods						**/
src/sip/dynamic_data/block.cpp:#ifdef HAVE_CUDA
src/sip/dynamic_data/block.cpp:Block::BlockPtr Block::new_gpu_block(BlockShape shape){
src/sip/dynamic_data/block.cpp:	block_ptr->gpu_data_ = _gpu_allocate(block_ptr->size());
src/sip/dynamic_data/block.cpp:	block_ptr->status_[Block::onGPU] = true;
src/sip/dynamic_data/block.cpp:	block_ptr->status_[Block::dirtyOnGPU] = false;
src/sip/dynamic_data/block.cpp:Block::dataPtr Block::gpu_copy_data(BlockPtr source_block){
src/sip/dynamic_data/block.cpp:	dataPtr target = get_gpu_data();
src/sip/dynamic_data/block.cpp:	dataPtr source = source_block->get_gpu_data();
src/sip/dynamic_data/block.cpp:	_gpu_device_to_device(target, source, n);
src/sip/dynamic_data/block.cpp:void Block::free_gpu_data(){
src/sip/dynamic_data/block.cpp:	if (gpu_data_){
src/sip/dynamic_data/block.cpp:		_gpu_free(gpu_data_);
src/sip/dynamic_data/block.cpp:	gpu_data_ = NULL;
src/sip/dynamic_data/block.cpp:	status_[Block::onGPU] = false;
src/sip/dynamic_data/block.cpp:	status_[Block::dirtyOnGPU] = false;
src/sip/dynamic_data/block.cpp:void Block::allocate_gpu_data(){
src/sip/dynamic_data/block.cpp:	WARN(gpu_data_ == NULL, "Potentially causing a memory leak on GPU");
src/sip/dynamic_data/block.cpp:	gpu_data_ = _gpu_allocate(size_);
src/sip/dynamic_data/block.cpp:	status_[Block::onGPU] = true;
src/sip/dynamic_data/block.cpp:	status_[Block::dirtyOnGPU] = false;
src/sip/dynamic_data/block.cpp:Block::dataPtr Block::get_gpu_data(){
src/sip/dynamic_data/block.cpp:	return gpu_data_;
src/sip/dynamic_data/block.cpp:Block::dataPtr Block::gpu_fill(double value){
src/sip/dynamic_data/block.cpp:	_gpu_double_memset(gpu_data_, value, size_);
src/sip/dynamic_data/block.cpp:	return gpu_data_;
src/sip/dynamic_data/block.cpp:Block::dataPtr Block::gpu_scale(double value){
src/sip/dynamic_data/block.cpp:	_gpu_selfmultiply(gpu_data_, value, size_);
src/sip/dynamic_data/block.cpp:	return gpu_data_;
src/sip/dynamic_data/block.h:	// GPU specific super instructions.
src/sip/dynamic_data/block.h:#ifdef HAVE_CUDA
src/sip/dynamic_data/block.h:	 * Factory to get a block with memory allocated on the GPU.
src/sip/dynamic_data/block.h:	static Block::BlockPtr new_gpu_block(BlockShape shape);
src/sip/dynamic_data/block.h:    dataPtr get_gpu_data();
src/sip/dynamic_data/block.h:    dataPtr gpu_fill(double value);
src/sip/dynamic_data/block.h:    dataPtr gpu_scale(double value);
src/sip/dynamic_data/block.h:    dataPtr gpu_copy_data(BlockPtr source);
src/sip/dynamic_data/block.h:    dataPtr gpu_transpose_copy(BlockPtr source, int rank, permute_t&);
src/sip/dynamic_data/block.h:    dataPtr gpu_accumulate_data(BlockPtr source);
src/sip/dynamic_data/block.h:    bool is_on_gpu()			{return status_[Block::onGPU];}
src/sip/dynamic_data/block.h:    bool is_dirty_on_gpu()		{return status_[Block::dirtyOnGPU];}
src/sip/dynamic_data/block.h:    bool is_dirty_on_all()		{return status_[Block::dirtyOnHost] && status_[Block::dirtyOnGPU];}
src/sip/dynamic_data/block.h:    void set_on_gpu()			{status_[Block::onGPU] = true;}
src/sip/dynamic_data/block.h:    void set_dirty_on_gpu()		{status_[Block::dirtyOnGPU] = true;}
src/sip/dynamic_data/block.h:    void unset_on_gpu()			{status_[Block::onGPU] = false;}
src/sip/dynamic_data/block.h:    void unset_dirty_on_gpu()	{status_[Block::dirtyOnGPU] = false;}
src/sip/dynamic_data/block.h:     * Frees up gpu data and sets appropriate flags.
src/sip/dynamic_data/block.h:    void free_gpu_data();
src/sip/dynamic_data/block.h:     * Allocate memory on gpu for data.
src/sip/dynamic_data/block.h:    void allocate_gpu_data();
src/sip/dynamic_data/block.h:	dataPtr gpu_data_;
src/sip/dynamic_data/block.h:		onGPU			= 1,	// Block is on device (GPU)
src/sip/dynamic_data/block.h:		dirtyOnGPU 	    = 3		// Block dirty on device (GPU)
src/sip/dynamic_data/block.h:	// TODO Figure out what to do with the GPU pointer.
src/sip/dynamic_data/block_manager.cpp:#include "gpu_super_instructions.h"
src/sip/dynamic_data/block_manager.cpp:#ifdef HAVE_CUDA
src/sip/dynamic_data/block_manager.cpp:	// Lazy copying of data from gpu to host if needed.
src/sip/dynamic_data/block_manager.cpp:	lazy_gpu_write_on_host(blk, id, shape);
src/sip/dynamic_data/block_manager.cpp://#ifdef HAVE_CUDA
src/sip/dynamic_data/block_manager.cpp://	// Lazy copying of data from gpu to host if needed.
src/sip/dynamic_data/block_manager.cpp://	lazy_gpu_write_on_host(blk, id, shape);
src/sip/dynamic_data/block_manager.cpp:	//#ifdef HAVE_CUDA
src/sip/dynamic_data/block_manager.cpp:	//	// Lazy copying of data from gpu to host if needed.
src/sip/dynamic_data/block_manager.cpp:	//	lazy_gpu_read_on_host(blk);
src/sip/dynamic_data/block_manager.cpp:	//#endif //HAVE_CUDA
src/sip/dynamic_data/block_manager.cpp:#ifdef HAVE_CUDA
src/sip/dynamic_data/block_manager.cpp:	// Lazy copying of data from gpu to host if needed.
src/sip/dynamic_data/block_manager.cpp:	lazy_gpu_update_on_host(blk);
src/sip/dynamic_data/block_manager.cpp:#ifdef HAVE_CUDA
src/sip/dynamic_data/block_manager.cpp:/**						GPU Specific methods						**/
src/sip/dynamic_data/block_manager.cpp:Block::BlockPtr BlockManager::get_gpu_block_for_writing(const BlockId& id, bool is_scope_extent) {
src/sip/dynamic_data/block_manager.cpp:		blk = create_gpu_block(id, shape);
src/sip/dynamic_data/block_manager.cpp:	// Lazy copying of data from host to gpu if needed.
src/sip/dynamic_data/block_manager.cpp:	lazy_gpu_write_on_device(blk, id, shape);
src/sip/dynamic_data/block_manager.cpp:	//blk->gpu_fill(0);
src/sip/dynamic_data/block_manager.cpp:Block::BlockPtr BlockManager::get_gpu_block_for_updating(const BlockId& id) {
src/sip/dynamic_data/block_manager.cpp:	// Lazy copying of data from host to gpu if needed.
src/sip/dynamic_data/block_manager.cpp:	lazy_gpu_update_on_device(blk);
src/sip/dynamic_data/block_manager.cpp:Block::BlockPtr BlockManager::get_gpu_block_for_reading(const BlockId& id) {
src/sip/dynamic_data/block_manager.cpp:	CHECK(blk != NULL, "attempting to read non-existent gpu block", current_line());
src/sip/dynamic_data/block_manager.cpp:	// Lazy copying of data from host to gpu if needed.
src/sip/dynamic_data/block_manager.cpp:	lazy_gpu_read_on_device(blk);
src/sip/dynamic_data/block_manager.cpp:Block::BlockPtr BlockManager::get_gpu_block_for_accumulate(const BlockId& id,
src/sip/dynamic_data/block_manager.cpp:	return get_gpu_block_for_writing(id, is_scope_extent);
src/sip/dynamic_data/block_manager.cpp:Block::BlockPtr BlockManager::create_gpu_block(const BlockId& block_id,
src/sip/dynamic_data/block_manager.cpp:	Block::BlockPtr block_ptr = Block::new_gpu_block(shape);
src/sip/dynamic_data/block_manager.cpp:void BlockManager::lazy_gpu_read_on_device(const Block::BlockPtr& blk) {
src/sip/dynamic_data/block_manager.cpp:	if (!blk->is_on_gpu() && !blk->is_on_host()) {
src/sip/dynamic_data/block_manager.cpp:		fail("block allocated neither on host or gpu", current_line());
src/sip/dynamic_data/block_manager.cpp:	} else if (!blk->is_on_gpu()) {
src/sip/dynamic_data/block_manager.cpp:		blk->allocate_gpu_data();
src/sip/dynamic_data/block_manager.cpp:		_gpu_host_to_device(blk->get_data(), blk->get_gpu_data(), blk->size());
src/sip/dynamic_data/block_manager.cpp:		_gpu_host_to_device(blk->get_data(), blk->get_gpu_data(), blk->size());
src/sip/dynamic_data/block_manager.cpp:		fail("block dirty on host & gpu !", current_line());
src/sip/dynamic_data/block_manager.cpp:	blk->set_on_gpu();
src/sip/dynamic_data/block_manager.cpp:void BlockManager::lazy_gpu_write_on_device(Block::BlockPtr& blk, const BlockId &id, const BlockShape& shape) {
src/sip/dynamic_data/block_manager.cpp:	if (!blk->is_on_gpu() && !blk->is_on_host()) {
src/sip/dynamic_data/block_manager.cpp:		blk = create_gpu_block(id, shape);
src/sip/dynamic_data/block_manager.cpp:	} else if (!blk->is_on_gpu()) {
src/sip/dynamic_data/block_manager.cpp:		blk->allocate_gpu_data();
src/sip/dynamic_data/block_manager.cpp:		_gpu_host_to_device(blk->get_data(), blk->get_gpu_data(), blk->size());
src/sip/dynamic_data/block_manager.cpp:		_gpu_host_to_device(blk->get_data(), blk->get_gpu_data(), blk->size());
src/sip/dynamic_data/block_manager.cpp:		fail("block dirty on host & gpu !", current_line());
src/sip/dynamic_data/block_manager.cpp:	blk->set_on_gpu();
src/sip/dynamic_data/block_manager.cpp:	blk->set_dirty_on_gpu();
src/sip/dynamic_data/block_manager.cpp:void BlockManager::lazy_gpu_update_on_device(const Block::BlockPtr& blk) {
src/sip/dynamic_data/block_manager.cpp:	if (!blk->is_on_gpu() && !blk->is_on_host()) {
src/sip/dynamic_data/block_manager.cpp:		fail("block allocated neither on host or gpu", current_line());
src/sip/dynamic_data/block_manager.cpp:	} else if (!blk->is_on_gpu()) {
src/sip/dynamic_data/block_manager.cpp:		blk->allocate_gpu_data();
src/sip/dynamic_data/block_manager.cpp:		_gpu_host_to_device(blk->get_data(), blk->get_gpu_data(), blk->size());
src/sip/dynamic_data/block_manager.cpp:		_gpu_host_to_device(blk->get_data(), blk->get_gpu_data(), blk->size());
src/sip/dynamic_data/block_manager.cpp:		fail("block dirty on host & gpu !", current_line());
src/sip/dynamic_data/block_manager.cpp:	blk->set_on_gpu();
src/sip/dynamic_data/block_manager.cpp:	blk->set_dirty_on_gpu();
src/sip/dynamic_data/block_manager.cpp:void BlockManager::lazy_gpu_read_on_host(const Block::BlockPtr& blk) {
src/sip/dynamic_data/block_manager.cpp:	if (!blk->is_on_gpu() && !blk->is_on_host()) {
src/sip/dynamic_data/block_manager.cpp:		fail("block allocated neither on host or gpu", current_line());
src/sip/dynamic_data/block_manager.cpp:		_gpu_device_to_host(blk->get_data(), blk->get_gpu_data(), blk->size());
src/sip/dynamic_data/block_manager.cpp:	} else if (blk->is_dirty_on_gpu()) {
src/sip/dynamic_data/block_manager.cpp:		_gpu_device_to_host(blk->get_data(), blk->get_gpu_data(), blk->size());
src/sip/dynamic_data/block_manager.cpp:		fail("block dirty on host & gpu !", current_line());
src/sip/dynamic_data/block_manager.cpp:	blk->unset_dirty_on_gpu();
src/sip/dynamic_data/block_manager.cpp:void BlockManager::lazy_gpu_write_on_host(Block::BlockPtr& blk, const BlockId &id, const BlockShape& shape) {
src/sip/dynamic_data/block_manager.cpp:	if (!blk->is_on_gpu() && !blk->is_on_host()) {
src/sip/dynamic_data/block_manager.cpp:		_gpu_device_to_host(blk->get_data(), blk->get_gpu_data(), blk->size());
src/sip/dynamic_data/block_manager.cpp:	} else if (blk->is_dirty_on_gpu()) {
src/sip/dynamic_data/block_manager.cpp:		_gpu_device_to_host(blk->get_data(), blk->get_gpu_data(), blk->size());
src/sip/dynamic_data/block_manager.cpp:		fail("block dirty on host & gpu !", current_line());
src/sip/dynamic_data/block_manager.cpp:	blk->unset_dirty_on_gpu();
src/sip/dynamic_data/block_manager.cpp:void BlockManager::lazy_gpu_update_on_host(const Block::BlockPtr& blk) {
src/sip/dynamic_data/block_manager.cpp:	if (!blk->is_on_gpu() && !blk->is_on_host()) {
src/sip/dynamic_data/block_manager.cpp:		fail("block allocated neither on host or gpu", current_line());
src/sip/dynamic_data/block_manager.cpp:		_gpu_device_to_host(blk->get_data(), blk->get_gpu_data(), blk->size());
src/sip/dynamic_data/block_manager.cpp:	} else if (blk->is_dirty_on_gpu()) {
src/sip/dynamic_data/block_manager.cpp:		_gpu_device_to_host(blk->get_data(), blk->get_gpu_data(), blk->size());
src/sip/dynamic_data/block_manager.cpp:		fail("block dirty on host & gpu !", current_line());
src/sip/dynamic_data/block_manager.cpp:	blk->unset_dirty_on_gpu();
src/sip/static_data/opcode.h:SIPOP(gpu_on_op,189,"gpu_on",false)\
src/sip/static_data/opcode.h:SIPOP(gpu_off_op,190,"gpu_off",false)\
src/sip/static_data/opcode.h:SIPOP(gpu_allocate_op,191,"gpu_allocate",false)\
src/sip/static_data/opcode.h:SIPOP(gpu_free_op,192,"gpu_free",false)\
src/sip/static_data/opcode.h:SIPOP(gpu_put_op,193,"gpu_put",false)\
src/sip/static_data/opcode.h:SIPOP(gpu_get_op,194,"gpu_get",false)\
src/sip/static_data/opcode.h:SIPOP(gpu_get_int_op,195,"gpu_get_int",false)\
src/sip/static_data/opcode.h:SIPOP(gpu_put_int_op,196,"gpu_put_int",false)\
src/sip/cuda/cuda_check.h:// From https://bitbucket.org/seanmauch/stlib/src/77b6f65e53c8/src/cuda/check.h?at=default
src/sip/cuda/cuda_check.h:  \file cuda/check.h
src/sip/cuda/cuda_check.h:  \brief Check CUDA error codes.
src/sip/cuda/cuda_check.h:#if !defined(__cuda_check_h__)
src/sip/cuda/cuda_check.h:#define __cuda_check_h__
src/sip/cuda/cuda_check.h:#ifdef __CUDA_ARCH__
src/sip/cuda/cuda_check.h:#define CUDA_CHECK(err) (err)
src/sip/cuda/cuda_check.h:#include <cuda_runtime_api.h>
src/sip/cuda/cuda_check.h:\page cudaCheck Check CUDA error codes.
src/sip/cuda/cuda_check.h:Check CUDA error codes with cudaCheck() or the CUDA_CHECK macro.
src/sip/cuda/cuda_check.h://! Check the CUDA error code.
src/sip/cuda/cuda_check.h:cudaCheck(cudaError_t err, const char *file, const int line) {
src/sip/cuda/cuda_check.h:   if (err != cudaSuccess) {
src/sip/cuda/cuda_check.h:      std::cout << cudaGetErrorString(err) << " in " << file << " at line "
src/sip/cuda/cuda_check.h://! Check the CUDA error code.
src/sip/cuda/cuda_check.h:#define CUDA_CHECK(err) (cudaCheck(err, __FILE__, __LINE__))
src/sip/cuda/cuda_check.h:// http://stackoverflow.com/questions/13041399/equivalent-of-cudageterrorstring-for-cublas    
src/sip/cuda/gpu_super_instructions.h: * gpu_super_instructions.h
src/sip/cuda/gpu_super_instructions.h:#ifndef GPU_SUPER_INSTRUCTIONS_H_
src/sip/cuda/gpu_super_instructions.h:#define GPU_SUPER_INSTRUCTIONS_H_
src/sip/cuda/gpu_super_instructions.h: * Initialization routine for CUDA
src/sip/cuda/gpu_super_instructions.h:void _init_gpu(int* devid, int* myRank);
src/sip/cuda/gpu_super_instructions.h:void _gpu_axpy(double* y, double* x, const double alpha, const int numElems);
src/sip/cuda/gpu_super_instructions.h:void _gpu_double_memset(double * g_addr, double value, const int numElems);
src/sip/cuda/gpu_super_instructions.h:void _gpu_selfmultiply(double* x, const double alpha, const int numElems);
src/sip/cuda/gpu_super_instructions.h: * @param y         address of block y on GPU
src/sip/cuda/gpu_super_instructions.h: * @param x1        address of block x1 on GPU
src/sip/cuda/gpu_super_instructions.h: * @param x2        address of block x2 on GPU
src/sip/cuda/gpu_super_instructions.h:void _gpu_contract(double* y, const int ny, const int* yDims, const int* yInds,
src/sip/cuda/gpu_super_instructions.h: * @param y         address of block y on GPU
src/sip/cuda/gpu_super_instructions.h: * @param x        address of block x1 on GPU
src/sip/cuda/gpu_super_instructions.h:void _gpu_permute(double* y, const int ny, const int* yDims, const int* yInds,
src/sip/cuda/gpu_super_instructions.h: * bytes on GPU from contents of h_addr there
src/sip/cuda/gpu_super_instructions.h: * @param [in]  g_adr address of block on gpu
src/sip/cuda/gpu_super_instructions.h:void _gpu_host_to_device(double* c_addr, double* g_addr, const int numElems);
src/sip/cuda/gpu_super_instructions.h: * Mallocs a block of size numElems bytes on GPU
src/sip/cuda/gpu_super_instructions.h:double* _gpu_allocate(const int numElems);
src/sip/cuda/gpu_super_instructions.h: * Copies block on GPU back to CPU
src/sip/cuda/gpu_super_instructions.h: * @param [in]  g_adr address of block on gpu
src/sip/cuda/gpu_super_instructions.h:void _gpu_device_to_host(double* c_addr, double* g_addr, const int numElems);
src/sip/cuda/gpu_super_instructions.h:void _gpu_device_to_device(double* dst, double *src, const int numElems);
src/sip/cuda/gpu_super_instructions.h: * Frees block on the GPU
src/sip/cuda/gpu_super_instructions.h:void _gpu_free(double* g_addr);
src/sip/cuda/gpu_super_instructions.h:#endif /* GPU_SUPER_INSTRUCTIONS_H_ */
src/sip/cuda/gpu_super_instructions.cu:#ifndef __GPU_SUPER_INSTRUCTIONS_CU__
src/sip/cuda/gpu_super_instructions.cu:#define __GPU_SUPER_INSTRUCTIONS_CU__
src/sip/cuda/gpu_super_instructions.cu:#include "gpu_super_instructions.h"
src/sip/cuda/gpu_super_instructions.cu:#include <cuda_check.h>
src/sip/cuda/gpu_super_instructions.cu:#include <cuda_runtime.h>
src/sip/cuda/gpu_super_instructions.cu:void __gpu_contract_helper(double* y, const int ny, const int* yDims, const int* yInds,
src/sip/cuda/gpu_super_instructions.cu:void __gpu_permute_helper(double* y, const int ny, const int* yDims, const int* yInds, double* x1,
src/sip/cuda/gpu_super_instructions.cu:void __gpu_matplus_helper(double* p_y, double* p_x1, double* p_x2, const int numElems,
src/sip/cuda/gpu_super_instructions.cu:void printGPUArray(double*, int);
src/sip/cuda/gpu_super_instructions.cu: * Initialization routine for CUDA
src/sip/cuda/gpu_super_instructions.cu:void _init_gpu(int* devid, int* myRank) {
src/sip/cuda/gpu_super_instructions.cu:	cudaDeviceProp deviceProp;
src/sip/cuda/gpu_super_instructions.cu:	cudaError_t err = cudaGetDeviceCount(&devCnt);
src/sip/cuda/gpu_super_instructions.cu:	//CUDA_CHECK(cudaGetDevice(&myDevice));
src/sip/cuda/gpu_super_instructions.cu:	if (err == cudaSuccess) {
src/sip/cuda/gpu_super_instructions.cu:		//    printf ("Task %d : cudaGetDevice did not return a device (device id = %d)\n", *myRank,myDevice);
src/sip/cuda/gpu_super_instructions.cu:		CUDA_CHECK(cudaSetDevice(myDevice));
src/sip/cuda/gpu_super_instructions.cu:		printf("Task %d set device %d out of %d GPUs\n", *myRank, myDevice,
src/sip/cuda/gpu_super_instructions.cu:		CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, myDevice));
src/sip/cuda/gpu_super_instructions.cu:		printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
src/sip/cuda/gpu_super_instructions.cu:		printf("Task %d not using GPUs, error returned :%s\n", *myRank,
src/sip/cuda/gpu_super_instructions.cu:				cudaGetErrorString(err));
src/sip/cuda/gpu_super_instructions.cu: * Any cleanup that needs to be done on the GPU
src/sip/cuda/gpu_super_instructions.cu:void _finalize_gpu() {
src/sip/cuda/gpu_super_instructions.cu://void _gpu_matplus(double* y, double* x1, double* x2, int numElems) {
src/sip/cuda/gpu_super_instructions.cu://	__gpu_matplus_helper(y, x1, x2, numElems, alpha);
src/sip/cuda/gpu_super_instructions.cu://void _gpu_matminus(double* y, double* x1, double* x2, int numElems) {
src/sip/cuda/gpu_super_instructions.cu://	__gpu_matplus_helper(y, x1, x2, numElems, alpha);
src/sip/cuda/gpu_super_instructions.cu:void _gpu_selfmultiply(double* p_x, const double alpha, const int numElems) {
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaDeviceSynchronize());
src/sip/cuda/gpu_super_instructions.cu: * @param y         address of block y on GPU
src/sip/cuda/gpu_super_instructions.cu: * @param x1        address of block x1 on GPU
src/sip/cuda/gpu_super_instructions.cu: * @param x2        address of block x2 on GPU
src/sip/cuda/gpu_super_instructions.cu:void _gpu_contract(double* y, const int ny, const int* yDims, const int* yInds,
src/sip/cuda/gpu_super_instructions.cu://	cudaPointerAttributes ptrAttr;
src/sip/cuda/gpu_super_instructions.cu://	CUDA_CHECK(cudaPointerGetAttributes(&ptrAttr, y));
src/sip/cuda/gpu_super_instructions.cu://	assert(ptrAttr.memoryType == cudaMemoryTypeDevice);
src/sip/cuda/gpu_super_instructions.cu://	CUDA_CHECK(cudaPointerGetAttributes(&ptrAttr, x1));
src/sip/cuda/gpu_super_instructions.cu://	assert(ptrAttr.memoryType == cudaMemoryTypeDevice);
src/sip/cuda/gpu_super_instructions.cu://	CUDA_CHECK(cudaPointerGetAttributes(&ptrAttr, x2));
src/sip/cuda/gpu_super_instructions.cu://	assert (ptrAttr.memoryType == cudaMemoryTypeDevice);
src/sip/cuda/gpu_super_instructions.cu://printGPUArray(x1, 10);
src/sip/cuda/gpu_super_instructions.cu://printGPUArray(x2, 10);
src/sip/cuda/gpu_super_instructions.cu:	__gpu_contract_helper(y, ny, yDims, yInds, x1, n1, x1Dims, x1Inds, x2, n2,
src/sip/cuda/gpu_super_instructions.cu: * @param y         address of block y on GPU
src/sip/cuda/gpu_super_instructions.cu: * @param x        address of block x1 on GPU
src/sip/cuda/gpu_super_instructions.cu:void _gpu_permute(double* y, const int ny, const int* yDims, const int* yInds,
src/sip/cuda/gpu_super_instructions.cu:	__gpu_permute_helper(y, ny, yDims, yInds, x, nx, xDims, xInds);
src/sip/cuda/gpu_super_instructions.cu: * Mallocs a block of size numElems bytes on GPU
src/sip/cuda/gpu_super_instructions.cu:double* _gpu_allocate(const int numElems) {
src/sip/cuda/gpu_super_instructions.cu://	std::cout<< "_gpu_allocate called from "<<current_line()<<std::endl;
src/sip/cuda/gpu_super_instructions.cu:	double *gpuAddr = NULL;
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaMalloc((void**)&gpuAddr, (numElems) * sizeof(double)));
src/sip/cuda/gpu_super_instructions.cu:	//CUDA_CHECK(cudaMemcpy(gpuAddr, h_addr, (*numElems)*sizeof(double), cudaMemcpyHostToDevice));
src/sip/cuda/gpu_super_instructions.cu:	//*g_addr = gpuAddr;
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaMemset((void*)gpuAddr, 0,(numElems) * sizeof(double)));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaDeviceSynchronize());
src/sip/cuda/gpu_super_instructions.cu:	assert (gpuAddr != NULL);
src/sip/cuda/gpu_super_instructions.cu://std::cout<<"Allocated on the GPU ..."<<gpuAddr<<std::endl;
src/sip/cuda/gpu_super_instructions.cu://	cudaPointerAttributes ptrAttr;
src/sip/cuda/gpu_super_instructions.cu://	CUDA_CHECK(cudaPointerGetAttributes(&ptrAttr, (void*)gpuAddr));
src/sip/cuda/gpu_super_instructions.cu://	assert(ptrAttr.memoryType == cudaMemoryTypeDevice);
src/sip/cuda/gpu_super_instructions.cu://	CUDA_CHECK(cudaDeviceSynchronize());
src/sip/cuda/gpu_super_instructions.cu:	//printf("load_temp : gpuAddr=%u, *g_addr=%u\n", gpuAddr, *g_addr);
src/sip/cuda/gpu_super_instructions.cu:	return gpuAddr;
src/sip/cuda/gpu_super_instructions.cu: * Copies block on GPU back to CPU
src/sip/cuda/gpu_super_instructions.cu: * @param [in]  g_adr address of block on gpu
src/sip/cuda/gpu_super_instructions.cu:void _gpu_device_to_host(double* h_addr, double* g_addr, int numElems) {
src/sip/cuda/gpu_super_instructions.cu:	double *gpuAddr = g_addr;
src/sip/cuda/gpu_super_instructions.cu://	cudaPointerAttributes ptrAttr;
src/sip/cuda/gpu_super_instructions.cu://	CUDA_CHECK(cudaPointerGetAttributes(&ptrAttr, g_addr));
src/sip/cuda/gpu_super_instructions.cu://	assert(ptrAttr.memoryType == cudaMemoryTypeDevice);
src/sip/cuda/gpu_super_instructions.cu:	//printf("unload : gpuAddr=%u, *g_addr=%u\n", gpuAddr, *g_addr);
src/sip/cuda/gpu_super_instructions.cu:	//printf("\nunloading gpuaddr :\n");
src/sip/cuda/gpu_super_instructions.cu:	//printArray(gpuAddr, 10);
src/sip/cuda/gpu_super_instructions.cu:	//CUDA_CHECK(cudaMemcpy(gpuAddr, h_addr, (*numElems)*sizeof(double), cudaMemcpyHostToDevice));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(
src/sip/cuda/gpu_super_instructions.cu:			cudaMemcpy(h_addr, gpuAddr, (numElems) * sizeof(double),
src/sip/cuda/gpu_super_instructions.cu:					cudaMemcpyDeviceToHost));
src/sip/cuda/gpu_super_instructions.cu:	//CUDA_CHECK(cudaFree(gpuAddr)); //$$$$$ TODO GET RID OF THIS $$$$$
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaDeviceSynchronize());
src/sip/cuda/gpu_super_instructions.cu: * bytes from Host to GPU
src/sip/cuda/gpu_super_instructions.cu: * @param [in]  g_adr address of block on gpu
src/sip/cuda/gpu_super_instructions.cu:void _gpu_host_to_device(double* h_addr, double* g_addr, const int numElems) {
src/sip/cuda/gpu_super_instructions.cu://	cudaPointerAttributes ptrAttr;
src/sip/cuda/gpu_super_instructions.cu://	CUDA_CHECK(cudaPointerGetAttributes(&ptrAttr, g_addr));
src/sip/cuda/gpu_super_instructions.cu://	assert(ptrAttr.memoryType == cudaMemoryTypeDevice);
src/sip/cuda/gpu_super_instructions.cu:	//double *gpuAddr = NULL;
src/sip/cuda/gpu_super_instructions.cu:	//CUDA_CHECK(cudaMalloc(&gpuAddr, (numElems)*sizeof(double)));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaMemcpy(g_addr, h_addr, (numElems) * sizeof(double), cudaMemcpyHostToDevice));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaDeviceSynchronize());
src/sip/cuda/gpu_super_instructions.cu:	//*g_addr = gpuAddr;
src/sip/cuda/gpu_super_instructions.cu:	//printf("load_input : gpuAddr=%u, *g_addr=%u\n", gpuAddr, *g_addr);
src/sip/cuda/gpu_super_instructions.cu://printGPUArray(g_addr, 10);
src/sip/cuda/gpu_super_instructions.cu:void _gpu_device_to_device(double* dst, double *src, const int numElems){
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaMemcpy(dst, src, (numElems) * sizeof(double), cudaMemcpyDeviceToDevice));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaDeviceSynchronize());
src/sip/cuda/gpu_super_instructions.cu: * Frees block on the GPU
src/sip/cuda/gpu_super_instructions.cu:void _gpu_free(double* g_addr) {
src/sip/cuda/gpu_super_instructions.cu:	//double *gpuAddr = g_addr;
src/sip/cuda/gpu_super_instructions.cu://	cudaPointerAttributes ptrAttr;
src/sip/cuda/gpu_super_instructions.cu://	CUDA_CHECK(cudaPointerGetAttributes(&ptrAttr, g_addr));
src/sip/cuda/gpu_super_instructions.cu://	assert(ptrAttr.memoryType == cudaMemoryTypeDevice);
src/sip/cuda/gpu_super_instructions.cu:	//printf("unload : gpuAddr=%u, *g_addr=%u\n", gpuAddr, *g_addr);
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaFree(g_addr));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaDeviceSynchronize());
src/sip/cuda/gpu_super_instructions.cu:void _gpu_double_memset(double * g_addr, double value, int numElems){
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaGetLastError());
src/sip/cuda/gpu_super_instructions.cu:void printGPUArray(double* g_addr, int size) {
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaMemcpy(h_addr, g_addr, size * sizeof(double), cudaMemcpyDeviceToHost));
src/sip/cuda/gpu_super_instructions.cu:void __gpu_contract_helper(double* p_y, const int ny, const int* yDims, const int* yInds,
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaMalloc(&scratch1, SCRATCH_BUFFER_SIZE_MB * 1024 * 1024));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaMalloc(&scratch2, SCRATCH_BUFFER_SIZE_MB * 1024 * 1024));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaMalloc(&scratch3, SCRATCH_BUFFER_SIZE_MB * 1024 * 1024));
src/sip/cuda/gpu_super_instructions.cu:    //std::cout<<"Allocating " << BUFF_SIZE << " Memory on GPU for Scratch Blocks" <<std::endl;
src/sip/cuda/gpu_super_instructions.cu:	//CUDA_CHECK(cudaMalloc(&scratch1, BUFF_SIZE));
src/sip/cuda/gpu_super_instructions.cu:	//CUDA_CHECK(cudaMalloc(&scratch2, BUFF_SIZE));
src/sip/cuda/gpu_super_instructions.cu:	//CUDA_CHECK(cudaMalloc(&scratch3, BUFF_SIZE));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaMemcpyToSymbol(dimsDev, x1Dims, sizeof(int) * n1));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaMemcpyToSymbol(stepsDev, steps, sizeof(int) * n1));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(
src/sip/cuda/gpu_super_instructions.cu:			cudaMemcpy(scratch3, x1, size * sizeof(double),
src/sip/cuda/gpu_super_instructions.cu:					cudaMemcpyDeviceToDevice));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaDeviceSynchronize());
src/sip/cuda/gpu_super_instructions.cu:    	CUDA_CHECK(cudaGetLastError());
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaDeviceSynchronize());
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaGetLastError());
src/sip/cuda/gpu_super_instructions.cu://printGPUArray(scratch1, 10);
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaMemcpyToSymbol(dimsDev, x2Dims, sizeof(int) * n2));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaMemcpyToSymbol(stepsDev, steps, sizeof(int) * n2));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(
src/sip/cuda/gpu_super_instructions.cu:			cudaMemcpy(scratch3, x2, size * sizeof(double),
src/sip/cuda/gpu_super_instructions.cu:					cudaMemcpyDeviceToDevice));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaDeviceSynchronize());
src/sip/cuda/gpu_super_instructions.cu:    	CUDA_CHECK(cudaGetLastError());
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaDeviceSynchronize());
src/sip/cuda/gpu_super_instructions.cu://printGPUArray(scratch2, 10);
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaDeviceSynchronize());
src/sip/cuda/gpu_super_instructions.cu://printGPUArray(scratch3, 10);
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaDeviceSynchronize());
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaDeviceSynchronize());
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaGetLastError());
src/sip/cuda/gpu_super_instructions.cu:	// reorder y from scratch3 to scratch1 and copy back from GPU
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaMemcpyToSymbol(dimsDev, yDims, sizeof(int) * ny));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaMemcpyToSymbol(stepsDev, steps, sizeof(int) * ny));
src/sip/cuda/gpu_super_instructions.cu:    	CUDA_CHECK(cudaGetLastError());
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaDeviceSynchronize());
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(
src/sip/cuda/gpu_super_instructions.cu:			cudaMemcpy(y, scratch1, size * sizeof(double),
src/sip/cuda/gpu_super_instructions.cu:					cudaMemcpyDeviceToDevice));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaFree(scratch1));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaFree(scratch2));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaFree(scratch3));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaDeviceSynchronize());
src/sip/cuda/gpu_super_instructions.cu:void __gpu_permute_helper(double* p_y, const int ny, const int* yDims, const int* yInds,
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaMalloc(&scratch1, SCRATCH_BUFFER_SIZE_MB * 1024 * 1024)); 
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaMalloc(&scratch3, SCRATCH_BUFFER_SIZE_MB * 1024 * 1024)); 
src/sip/cuda/gpu_super_instructions.cu:	//CUDA_CHECK(cudaMalloc(&scratch1, BUFF_SIZE));
src/sip/cuda/gpu_super_instructions.cu:	//CUDA_CHECK(cudaMalloc(&scratch3, BUFF_SIZE));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaMemcpyToSymbol(dimsDev, x1Dims, sizeof(int) * n1));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaMemcpyToSymbol(stepsDev, steps, sizeof(int) * n1));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaMemcpy(scratch3, x1, size * sizeof(double), cudaMemcpyDeviceToDevice));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaDeviceSynchronize());
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaMemcpyToSymbol(dimsDev, yDims, sizeof(int) * ny));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaMemcpyToSymbol(stepsDev, steps, sizeof(int) * ny));
src/sip/cuda/gpu_super_instructions.cu:    	CUDA_CHECK(cudaGetLastError());
src/sip/cuda/gpu_super_instructions.cu:	// copy scratch1 into y  and copy back from GPU
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaMemcpy(y, scratch1, size * sizeof(double), cudaMemcpyDeviceToDevice));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaFree(scratch1));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaFree(scratch3));
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaDeviceSynchronize());
src/sip/cuda/gpu_super_instructions.cu:void _gpu_axpy(double *p_y, double *p_x, const double alpha, const int numElems){
src/sip/cuda/gpu_super_instructions.cu:	CUDA_CHECK(cudaDeviceSynchronize());
src/sip/cuda/gpu_super_instructions.cu:// * @param y         address of array y on GPU
src/sip/cuda/gpu_super_instructions.cu:// * @param x1        address of array x1 on GPU
src/sip/cuda/gpu_super_instructions.cu:// * @param x2        address of array x2 on GPU
src/sip/cuda/gpu_super_instructions.cu://void __gpu_matplus_helper(double* p_y, double* p_x1, double* p_x2, int numElems,
src/sip/cuda/gpu_super_instructions.cu://		CUDA_CHECK(
src/sip/cuda/gpu_super_instructions.cu://				cudaMemcpy(y, x2, (numElems) * sizeof(double),
src/sip/cuda/gpu_super_instructions.cu://						cudaMemcpyDeviceToDevice));
src/sip/cuda/gpu_super_instructions.cu://	CUDA_CHECK(cudaDeviceSynchronize());
src/sip/cuda/gpu_super_instructions.cu:#endif // __GPU_SUPER_INSTRUCTIONS_CU__
src/sip/super_instructions/special_instructions.cpp: * 	2. add a statement to the init_procmap method to add it to the procmap_. This is done as follows
src/sip/super_instructions/special_instructions.cpp: * 	    (*procmap)["sialname"] = (fp)&cname;
src/sip/super_instructions/special_instructions.cpp://    CHECK(procmap_.empty(), "attempting to initialize non-empty procmap");
src/sip/super_instructions/special_instructions.cpp:	init_procmap();
src/sip/super_instructions/special_instructions.cpp:	if (procmap_.empty()) init_procmap();
src/sip/super_instructions/special_instructions.cpp://	void(*func)(int* ierr) = procmap_.at(name);
src/sip/super_instructions/special_instructions.cpp:		std::map<std::string, fp0>::iterator it = procmap_.find(name);
src/sip/super_instructions/special_instructions.cpp:		if (it == procmap_.end()){
src/sip/super_instructions/special_instructions.cpp://	procmap_.clear();
src/sip/super_instructions/special_instructions.cpp:void SpecialInstructionManager::init_procmap(){
src/sip/super_instructions/special_instructions.cpp://	procmap_["dadd"] = (fp0)&dadd;
src/sip/super_instructions/special_instructions.cpp://	procmap_["dsub"] = (fp0)&dsub;
src/sip/super_instructions/special_instructions.cpp:	procmap_["print_something"] = (fp0)&print_something;
src/sip/super_instructions/special_instructions.cpp:	procmap_["fill_block_sequential"]= (fp0)&fill_block_sequential;
src/sip/super_instructions/special_instructions.cpp:	procmap_["fill_block_cyclic"]= (fp0)&fill_block_cyclic;
src/sip/super_instructions/special_instructions.cpp://	procmap_["test_print_block"]=(fp0)&test_print_block;
src/sip/super_instructions/special_instructions.cpp:	procmap_["print_block"]=(fp0)&print_block;
src/sip/super_instructions/special_instructions.cpp:	procmap_["write_block_to_file"]=(fp0)&write_block_to_file;
src/sip/super_instructions/special_instructions.cpp:	procmap_["read_block_from_file"]=(fp0)&read_block_from_file;
src/sip/super_instructions/special_instructions.cpp:	procmap_["read_block_from_text_file"]=(fp0)&read_block_from_text_file;
src/sip/super_instructions/special_instructions.cpp:	procmap_["print_static_array"]=(fp0)&print_static_array;
src/sip/super_instructions/special_instructions.cpp:	procmap_["list_block_map"]=(fp0)&list_block_map;
src/sip/super_instructions/special_instructions.cpp:	procmap_["compute_aabb_batch"]=(fp0)&compute_aabb_batch;
src/sip/super_instructions/special_instructions.cpp:	procmap_["get_my_rank"]=(fp0)&get_my_rank;
src/sip/super_instructions/special_instructions.cpp:	procmap_["return_sval"]=(fp0)&return_sval;
src/sip/super_instructions/special_instructions.cpp:	procmap_["check_dconf"]=(fp0)&check_dconf;
src/sip/super_instructions/special_instructions.cpp:	procmap_["compute_diis"]=(fp0)&compute_diis;
src/sip/super_instructions/special_instructions.cpp:	procmap_["return_h1"]=(fp0)&return_h1;
src/sip/super_instructions/special_instructions.cpp:	procmap_["return_ovl"]=(fp0)&return_ovl;
src/sip/super_instructions/special_instructions.cpp:	procmap_["return_1el_ecpints"]=(fp0)&return_1el_ecpints;
src/sip/super_instructions/special_instructions.cpp:	procmap_["scf_atom_lowmem"]=(fp0)&scf_atom_lowmem;
src/sip/super_instructions/special_instructions.cpp:	procmap_["place_scratch"]=(fp0)&place_scratch;
src/sip/super_instructions/special_instructions.cpp:	procmap_["return_pairs"]=(fp0)&return_pairs;
src/sip/super_instructions/special_instructions.cpp:	procmap_["compute_pair_nn"]=(fp0)&compute_pair_nn;
src/sip/super_instructions/special_instructions.cpp:	procmap_["return_h1frag"]=(fp0)&return_h1frag;
src/sip/super_instructions/special_instructions.cpp:	procmap_["compute_int_scratchmem"]=(fp0)&compute_int_scratchmem;
src/sip/super_instructions/special_instructions.cpp:	procmap_["compute_int_scratchmem_lowmem"]=(fp0)&compute_int_scratchmem_lowmem;
src/sip/super_instructions/special_instructions.cpp:	procmap_["energy_denominator_rhf"]=(fp0)&energy_denominator_rhf;
src/sip/super_instructions/special_instructions.cpp:	procmap_["energy_numerator_rhf"]=(fp0)&energy_numerator_rhf;
src/sip/super_instructions/special_instructions.cpp:	procmap_["return_vpq"]=(fp0)&return_vpq;
src/sip/super_instructions/special_instructions.cpp:	procmap_["return_diagonal"]=(fp0)&return_diagonal;
src/sip/super_instructions/special_instructions.cpp:	procmap_["anti_symm_o"]=(fp0)&anti_symm_o;
src/sip/super_instructions/special_instructions.cpp:	procmap_["anti_symm_v"]=(fp0)&anti_symm_v;
src/sip/super_instructions/special_instructions.cpp:	procmap_["eigen_calc_sqr_inv"]=(fp0)&eigen_calc_sqr_inv;
src/sip/super_instructions/special_instructions.cpp:	procmap_["eigen_calc"]=(fp0)&eigen_calc;
src/sip/super_instructions/special_instructions.cpp:	procmap_["gen_eigen_calc"]=(fp0)&gen_eigen_calc;
src/sip/super_instructions/special_instructions.cpp:    procmap_["set_flags2"]=(fp0)&set_flags2;
src/sip/super_instructions/special_instructions.cpp:    procmap_["compute_ubatch2"]=(fp0)&compute_ubatch2;
src/sip/super_instructions/special_instructions.cpp:    procmap_["get_scratch_array_dummy"]=(fp0)&get_scratch_array_dummy;
src/sip/super_instructions/special_instructions.cpp:    procmap_["get_and_print_int_array_dummy"]=(fp0)&get_and_print_int_array_dummy;
src/sip/super_instructions/special_instructions.cpp:    procmap_["get_and_print_scalar_array_dummy"]=(fp0)&get_and_print_scalar_array_dummy;
src/sip/super_instructions/special_instructions.cpp:    procmap_["get_first_block_element"]=(fp0)&get_first_block_element;
src/sip/super_instructions/special_instructions.cpp:    procmap_["compute_ubatch1"]=(fp0)&compute_ubatch1;
src/sip/super_instructions/special_instructions.cpp:    procmap_["compute_ubatch3"]=(fp0)&compute_ubatch3;
src/sip/super_instructions/special_instructions.cpp:    procmap_["compute_ubatch4"]=(fp0)&compute_ubatch4;
src/sip/super_instructions/special_instructions.cpp:    procmap_["compute_ubatch6"]=(fp0)&compute_ubatch6;
src/sip/super_instructions/special_instructions.cpp:    procmap_["compute_ubatch7"]=(fp0)&compute_ubatch7;
src/sip/super_instructions/special_instructions.cpp:    procmap_["compute_ubatch8"]=(fp0)&compute_ubatch8;
src/sip/super_instructions/special_instructions.cpp:    procmap_["compute_integral_batch"]=(fp0)&compute_integral_batch;
src/sip/super_instructions/special_instructions.cpp:    procmap_["compute_xyz_batch"]=(fp0)&compute_xyz_batch;
src/sip/super_instructions/special_instructions.cpp:    procmap_["compute_dipole_integrals"]=(fp0)&compute_dipole_integrals;
src/sip/super_instructions/special_instructions.cpp:    procmap_["aoladder_contraction"]=(fp0)&aoladder_contraction;
src/sip/super_instructions/special_instructions.cpp:    procmap_["compute_nn_repulsion"]=(fp0)&compute_nn_repulsion;
src/sip/super_instructions/special_instructions.cpp:    procmap_["drop_core_in_sip"]=(fp0)&drop_core_in_sip;
src/sip/super_instructions/special_instructions.cpp:    procmap_["set_frag"]=(fp0)&set_frag;
src/sip/super_instructions/special_instructions.cpp:    procmap_["frag_index_range"]=(fp0)&frag_index_range;
src/sip/super_instructions/special_instructions.cpp:    procmap_["stripi"]=(fp0)&stripi;
src/sip/super_instructions/special_instructions.cpp:    procmap_["set_ijk_aaa"]=(fp0)&set_ijk_aaa;
src/sip/super_instructions/special_instructions.cpp:    procmap_["set_ijk_aab"]=(fp0)&set_ijk_aab;
src/sip/super_instructions/special_instructions.cpp:    procmap_["swap_blocks"]=(fp0)&swap_blocks;
src/sip/super_instructions/special_instructions.cpp:    procmap_["cis_unit_guess"]=(fp0)&cis_unit_guess;
src/sip/super_instructions/special_instructions.cpp:    procmap_["invert_diagonal"]=(fp0)&invert_diagonal;
src/sip/super_instructions/special_instructions.cpp:    procmap_["invert_diagonal_asym"]=(fp0)&invert_diagonal_asym;
src/sip/super_instructions/special_instructions.cpp:    procmap_["energy_ty_denominator_rhf"]=(fp0)&energy_ty_denominator_rhf;
src/sip/super_instructions/special_instructions.cpp:    procmap_["return_diagonal_elements"]=(fp0)&return_diagonal_elements;
src/sip/super_instructions/special_instructions.cpp:    procmap_["enable_debug_print"]=(fp0)&enable_debug_print;
src/sip/super_instructions/special_instructions.cpp:    procmap_["disable_debug_print"]=(fp0)&disable_debug_print;
src/sip/super_instructions/special_instructions.cpp:    procmap_["enable_all_rank_print"]=(fp0)&enable_all_rank_print;
src/sip/super_instructions/special_instructions.cpp:    procmap_["disable_all_rank_print"]=(fp0)&disable_all_rank_print;
src/sip/super_instructions/special_instructions.cpp:    procmap_["get_and_print_mpi_rank"]=(fp0)&get_and_print_mpi_rank;
src/sip/super_instructions/special_instructions.cpp:    procmap_["one_arg_no_op"]=(fp0)&one_arg_no_op;
src/sip/super_instructions/special_instructions.cpp:    procmap_["list_blocks_with_number"]=(fp0)&list_blocks_with_number;
src/sip/super_instructions/special_instructions.cpp:    procmap_["a4_get_init_occupation"]=(fp0)&a4_get_init_occupation;
src/sip/super_instructions/special_instructions.cpp:    procmap_["a4_david_damp_factor"]=(fp0)&a4_david_damp_factor;
src/sip/super_instructions/special_instructions.cpp:    procmap_["a4_return_occupation"]=(fp0)&a4_return_occupation;
src/sip/super_instructions/special_instructions.cpp:    procmap_["a4_scf_atom"]=(fp0)&a4_scf_atom;
src/sip/super_instructions/special_instructions.cpp:    procmap_["a4_dscale"]=(fp0)&a4_dscale;
src/sip/super_instructions/special_instructions.cpp:    procmap_["print_block_and_index"]=(fp0)&print_block_and_index;
src/sip/super_instructions/special_instructions.cpp:    procmap_["form_diagonal_unit_matrix"]=(fp0)&form_diagonal_unit_matrix;
src/sip/super_instructions/special_instructions.cpp:    procmap_["check_block_number_calculation"]=(fp0)&check_block_number_calculation;
src/sip/super_instructions/special_instructions.cpp:    procmap_["moi_nn_repulsion"]=(fp0)&moi_nn_repulsion;
src/sip/super_instructions/special_instructions.cpp:    procmap_["return_h1_moi"]=(fp0)&return_h1_moi;
src/sip/super_instructions/special_instructions.cpp:    procmap_["return_ovl_moi"]=(fp0)&return_ovl_moi;
src/sip/super_instructions/special_instructions.cpp:    procmap_["remove_diagonal"]=(fp0)&remove_diagonal;
src/sip/super_instructions/special_instructions.h:	/**Initializes procmap.  Called by the constructor. */
src/sip/super_instructions/special_instructions.h:	void init_procmap();
src/sip/super_instructions/special_instructions.h:	 * It releases resources, such as the procmap that are no longer needed.
src/sip/super_instructions/special_instructions.h:	std::map<std::string, fp0> procmap_;
src/sip/super_instructions/CMakeLists.txt:    ${CMAKE_SOURCE_DIR}/src/sip/cuda)
src/sip/super_instructions/Makefile.am:-I${top_srcdir}/src/sip/cuda
src/sialx/test/gpu_contraction_predefined_test.sialx: sial gpu_contraction_predefined
src/sialx/test/gpu_contraction_predefined_test.sialx:			gpu_on    	
src/sialx/test/gpu_contraction_predefined_test.sialx:	     	gpu_off 
src/sialx/test/gpu_contraction_predefined_test.sialx:endsial gpu_contraction_predefined
src/sialx/test/gpu_transpose_tmp.sialx:				gpu_on	
src/sialx/test/gpu_transpose_tmp.sialx:				gpu_off
src/sialx/test/gpu_sum_op_test.sialx:		gpu_on
src/sialx/test/gpu_sum_op_test.sialx:		gpu_off
src/sialx/test/contraction_small_test2.sialx:					#gpu_on
src/sialx/test/contraction_small_test2.sialx:                        gpu_on
src/sialx/test/contraction_small_test2.sialx:                        gpu_off
src/sialx/test/contraction_small_test2.sialx:					#gpu_off 
src/sialx/test/gpu_ops.sialx:sial gpu_ops
src/sialx/test/gpu_ops.sialx:	gpu_on
src/sialx/test/gpu_ops.sialx:			gpu_allocate a[i, j]
src/sialx/test/gpu_ops.sialx:			gpu_allocate b[i,j,k, l]
src/sialx/test/gpu_ops.sialx:			gpu_put b[i,j,k, l]
src/sialx/test/gpu_ops.sialx:			gpu_get b[i,j,k, l]
src/sialx/test/gpu_ops.sialx:			gpu_free b[i,j,k, l]
src/sialx/test/gpu_ops.sialx:			gpu_free a[i, j]
src/sialx/test/gpu_ops.sialx:	gpu_off
src/sialx/test/gpu_ops.sialx:endsial gpu_ops
src/sialx/test/gpu_self_multiply_test.sialx:sial gpu_self_multiply_test
src/sialx/test/gpu_self_multiply_test.sialx:		gpu_on
src/sialx/test/gpu_self_multiply_test.sialx:		gpu_off	
src/sialx/test/gpu_self_multiply_test.sialx:println "end of gpu_self_multiply_test"
src/sialx/test/gpu_self_multiply_test.sialx:endsial gpu_self_multiply_test
src/sialx/test/gpu_contraction_small_test.sialx: sial gpu_contraction_small
src/sialx/test/gpu_contraction_small_test.sialx:					gpu_on
src/sialx/test/gpu_contraction_small_test.sialx:					gpu_put a[i, j, k, l]
src/sialx/test/gpu_contraction_small_test.sialx:					gpu_put b[j, k]
src/sialx/test/gpu_contraction_small_test.sialx:					gpu_allocate c[i, l]
src/sialx/test/gpu_contraction_small_test.sialx:					gpu_get c[i, l]
src/sialx/test/gpu_contraction_small_test.sialx:					gpu_free a[i, j, k, l]
src/sialx/test/gpu_contraction_small_test.sialx:					gpu_free b[j, k]
src/sialx/test/gpu_contraction_small_test.sialx:					gpu_off
src/sialx/test/gpu_contraction_small_test.sialx:endsial gpu_contraction_small
src/sialx/test/gpu_contract_to_scalar.sialx:gpu_on
src/sialx/test/gpu_contract_to_scalar.sialx:gpu_off
src/sialx/qm/cc/rccsd_rhf.sialx:     PROC AOLADDER_GPU   
src/sialx/qm/cc/rccsd_rhf.sialx:     ENDPROC AOLADDER_GPU   
src/sialx/qm/cc/rccsd_rhf.sialx:        #CALL AOLADDER_GPU 
src/sialx/qm/cc/mp2_rhf_disc.sialx:#                                 GPU                                  #
src/sialx/qm/cc/mp2_rhf_disc.sialx:    gpu_on
src/sialx/qm/cc/mp2_rhf_disc.sialx:#                                 GPU                                  #
src/sialx/qm/cc/mp2_rhf_disc.sialx:    gpu_off
src/sialx/qm/depreciated/scf_mp2_rhf.sialxXXX:#                                 GPU                                  #
src/sialx/qm/depreciated/scf_mp2_rhf.sialxXXX:    gpu_on
src/sialx/qm/depreciated/scf_mp2_rhf.sialxXXX:#                                 GPU                                  #
src/sialx/qm/depreciated/scf_mp2_rhf.sialxXXX:    gpu_off

```
