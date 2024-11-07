# https://github.com/stan-dev/stan

```console
Jenkinsfile:def skipOpenCL = false
Jenkinsfile:        OPENCL_DEVICE_ID_CPU = 0
Jenkinsfile:        OPENCL_DEVICE_ID_GPU = 0
Jenkinsfile:        OPENCL_PLATFORM_ID = 1
Jenkinsfile:        OPENCL_PLATFORM_ID_CPU = 0
Jenkinsfile:        OPENCL_PLATFORM_ID_GPU = 0
Jenkinsfile:                    image 'stanorg/ci:gpu'
Jenkinsfile:                    image 'stanorg/ci:gpu'
Jenkinsfile:                    image 'stanorg/ci:gpu'
Jenkinsfile:                    def openCLPaths = ['src/stan/model/indexing'].join(" ")
Jenkinsfile:                    skipOpenCL = utils.verifyChanges(openCLPaths)
Jenkinsfile:                            image 'stanorg/ci:gpu'
Jenkinsfile:                            args '--pull always --gpus 1'
Jenkinsfile:                            echo STAN_OPENCL=true > make/local
Jenkinsfile:                            echo OPENCL_PLATFORM_ID=${OPENCL_PLATFORM_ID_GPU} >> make/local
Jenkinsfile:                            echo OPENCL_DEVICE_ID=${OPENCL_DEVICE_ID_GPU} >> make/local
Jenkinsfile:                            image 'stanorg/ci:gpu'
RELEASE-NOTES.txt: - Fixed a bug in the stan opencl assign tests (#3219)
RELEASE-NOTES.txt:- Added a hard copy of the event vector for OpenCL before making a copy to go from a tbb concurrent vector to a standard vector. (#3217)
RELEASE-NOTES.txt:Stan now comes with GPU support for certain functions; see the Math library
src/test/unit/model/indexing/util_cl.hpp:#ifdef STAN_OPENCL
src/test/unit/model/indexing/util_cl.hpp: * Convert an index to a type usable with OpenCL overloads.
src/test/unit/model/indexing/util_cl.hpp: * @return OpenCL index
src/test/unit/model/indexing/util_cl.hpp:T opencl_index(T i) {
src/test/unit/model/indexing/util_cl.hpp:stan::math::matrix_cl<int> opencl_index(const stan::model::index_multi& i) {
src/test/unit/model/indexing/assign_cl_test.cpp:#ifdef STAN_OPENCL
src/test/unit/model/indexing/assign_cl_test.cpp:TEST(ModelIndexing, assign_opencl_vector_1d) {
src/test/unit/model/indexing/assign_cl_test.cpp:        auto index_cl = opencl_index(index);
src/test/unit/model/indexing/assign_cl_test.cpp:TEST(ModelIndexing, assign_opencl_matrix_1d) {
src/test/unit/model/indexing/assign_cl_test.cpp:        auto index_cl = opencl_index(index);
src/test/unit/model/indexing/assign_cl_test.cpp:TEST(ModelIndexing, assign_opencl_matrix_2d) {
src/test/unit/model/indexing/assign_cl_test.cpp:              auto index1_cl = opencl_index(index1);
src/test/unit/model/indexing/assign_cl_test.cpp:              auto index2_cl = opencl_index(index2);
src/test/unit/model/indexing/rvalue_cl_test.cpp:#ifdef STAN_OPENCL
src/test/unit/model/indexing/rvalue_cl_test.cpp:TEST(ModelIndexing, rvalue_opencl_vector_1d) {
src/test/unit/model/indexing/rvalue_cl_test.cpp:                                           m_cl, "", opencl_index(ind1))));
src/test/unit/model/indexing/rvalue_cl_test.cpp:                                             m_i_cl, "", opencl_index(ind1))));
src/test/unit/model/indexing/rvalue_cl_test.cpp:            = from_matrix_cl_nonscalar(rvalue(m_v_cl, "", opencl_index(ind1)));
src/test/unit/model/indexing/rvalue_cl_test.cpp:TEST(ModelIndexing, rvalue_opencl_matrix_1d) {
src/test/unit/model/indexing/rvalue_cl_test.cpp:                                           m_cl, "", opencl_index(ind1))));
src/test/unit/model/indexing/rvalue_cl_test.cpp:                                             m_i_cl, "", opencl_index(ind1))));
src/test/unit/model/indexing/rvalue_cl_test.cpp:            = from_matrix_cl_nonscalar(rvalue(m_v_cl, "", opencl_index(ind1)));
src/test/unit/model/indexing/rvalue_cl_test.cpp:TEST(ModelIndexing, rvalue_opencl_matrix_2d) {
src/test/unit/model/indexing/rvalue_cl_test.cpp:                            m_cl, "", opencl_index(ind1), opencl_index(ind2))));
src/test/unit/model/indexing/rvalue_cl_test.cpp:                      m_i_cl, "", opencl_index(ind1), opencl_index(ind2))));
src/test/unit/model/indexing/rvalue_cl_test.cpp:                  rvalue(m_v_cl, "", opencl_index(ind1), opencl_index(ind2)));
src/test/unit/model/indexing/rvalue_cl_test.cpp:TEST(ModelIndexing, rvalue_opencl_matrix_2d_errors) {
src/test/unit/model/indexing/rvalue_cl_test.cpp:                  rvalue(m_cl, "", opencl_index(ind), opencl_index(ind_err)),
src/test/unit/model/indexing/rvalue_cl_test.cpp:                  rvalue(m_cl, "", opencl_index(ind_err), opencl_index(ind)),
src/test/unit/model/indexing/rvalue_cl_test.cpp:                  rvalue(m_v_cl, "", opencl_index(ind), opencl_index(ind_err)),
src/test/unit/model/indexing/rvalue_cl_test.cpp:                  rvalue(m_v_cl, "", opencl_index(ind_err), opencl_index(ind)),
src/test/unit/model/indexing/rvalue_cl_test.cpp:TEST(ModelIndexing, rvalue_opencl_matrix_1d_errors) {
src/test/unit/model/indexing/rvalue_cl_test.cpp:        EXPECT_THROW(rvalue(m_cl, "", opencl_index(ind1)), std::out_of_range);
src/test/unit/model/indexing/rvalue_cl_test.cpp:        EXPECT_THROW(rvalue(m_v_cl, "", opencl_index(ind1)), std::out_of_range);
src/test/unit/model/indexing/rvalue_cl_test.cpp:TEST(ModelIndexing, rvalue_opencl_vector_1d_errors) {
src/test/unit/model/indexing/rvalue_cl_test.cpp:        EXPECT_THROW(rvalue(m_cl, "", opencl_index(ind1)), std::out_of_range);
src/test/unit/model/indexing/rvalue_cl_test.cpp:        EXPECT_THROW(rvalue(m_v_cl, "", opencl_index(ind1)), std::out_of_range);
src/stan/model/indexing.hpp:#ifdef STAN_OPENCL
src/stan/model/indexing/rvalue_cl.hpp:#ifdef STAN_OPENCL
src/stan/model/indexing/rvalue_cl.hpp:#include <stan/math/opencl/rev.hpp>
src/stan/model/indexing/rvalue_cl.hpp:#include <stan/math/opencl/indexing_rev.hpp>
src/stan/model/indexing/rvalue_cl.hpp:    cl::CommandQueue& queue = stan::math::opencl_context.queue();
src/stan/model/indexing/rvalue_cl.hpp:    stan::math::check_opencl_error(m.str().c_str(), e);
src/stan/model/indexing/rvalue_cl.hpp:    cl::CommandQueue& queue = stan::math::opencl_context.queue();
src/stan/model/indexing/rvalue_cl.hpp:    stan::math::check_opencl_error(m.str().c_str(), e);
src/stan/model/indexing/rvalue_index_size.hpp:#ifdef STAN_OPENCL
src/stan/model/indexing/rvalue_index_size.hpp:#include <stan/math/opencl/indexing_rev.hpp>
src/stan/model/indexing/rvalue_index_size.hpp:#ifdef STAN_OPENCL
src/stan/model/indexing/assign_cl.hpp:#ifdef STAN_OPENCL
src/stan/model/indexing/assign_cl.hpp:#include <stan/math/opencl/rev.hpp>
src/stan/model/indexing/assign_cl.hpp:#include <stan/math/opencl/indexing_rev.hpp>
src/stan/model/indexing/assign_cl.hpp:#ifdef STAN_OPENCL

```
