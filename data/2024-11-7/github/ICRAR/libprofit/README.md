# https://github.com/ICRAR/libprofit

```console
docs/changelog.rst:* A bug in the OpenCL implementation of the radial profiles
docs/changelog.rst:* When using OpenCL,
docs/changelog.rst:  by the OpenCL convolver,
docs/changelog.rst:* Fixed compilation of ``brokenexponential`` OpenCL kernel in platforms where it
docs/changelog.rst:* OpenCL kernel cache working for some platforms/devices that was not
docs/changelog.rst:* Fixed ``double`` detection support for OpenCL devices regardless of the
docs/changelog.rst:  supported OpenCL version.
docs/changelog.rst:* New on-disk OpenCL kernel cache. This speeds up the creation of OpenCL
docs/Doxyfile:PREDEFINED             = PROFIT_FFTW PROFIT_OPENCL PROFIT_OPENMP PROFIT_API=
docs/api/library.rst:.. doxygenfunction:: profit::has_opencl()
docs/api/library.rst:.. doxygenfunction:: profit::opencl_version_major()
docs/api/library.rst:.. doxygenfunction:: profit::opencl_version_minor()
docs/api/exceptions.rst:.. doxygenclass:: profit::opencl_error
docs/getting.rst:* An `OpenCL <https://www.khronos.org/opencl/>`_ installation
docs/getting.rst:* ``LIBPROFIT_NO_OPENCL``: disable OpenCL support
docs/convolution.rst:* :enumerator:`OPENCL` is a brute-force convolver
docs/convolution.rst:  implemented in OpenCL.
tests/test_opencl.h: * OpenCL-related tests
tests/test_opencl.h:class OpenCLParameters {
tests/test_opencl.h:	OpenCLParameters() {
tests/test_opencl.h:		if (!has_opencl()) {
tests/test_opencl.h:		if (const char *cl_spec = std::getenv("LIBPROFIT_OPENCL_TESTSPEC")) {
tests/test_opencl.h:		// Look preferably for an OpenCL device that has double support
tests/test_opencl.h:			auto platforms = get_opencl_info();
tests/test_opencl.h:		} catch (const opencl_error &) {
tests/test_opencl.h:			opencl_env = get_opencl_environment(plat_idx, dev_idx, use_double, false);
tests/test_opencl.h:		} catch (const opencl_error &) {
tests/test_opencl.h:		if( const char *tol = std::getenv("LIBPROFIT_OPENCL_TOLERANCE") ) {
tests/test_opencl.h:	OpenCLEnvPtr opencl_env;
tests/test_opencl.h:class TestOpenCLTimes : public CxxTest::TestSuite {
tests/test_opencl.h:	void test_opencl_command_times_value_initialization()
tests/test_opencl.h:		OpenCL_command_times t {};
tests/test_opencl.h:	void test_opencl_command_times_plus()
tests/test_opencl.h:		OpenCL_command_times t {1, 2, 3};
tests/test_opencl.h:		OpenCL_command_times t2 {4, 5, 6};
tests/test_opencl.h:	void test_opencl_times_value_initialization()
tests/test_opencl.h:		OpenCL_times t{};
tests/test_opencl.h:class TestOpenCL : public CxxTest::TestSuite {
tests/test_opencl.h:	std::unique_ptr<OpenCLParameters> openCLParameters;
tests/test_opencl.h:	void _check_opencl_support() {
tests/test_opencl.h:		if( !has_opencl() ) {
tests/test_opencl.h:			TS_SKIP("No OpenCL support found, cannot run this test");
tests/test_opencl.h:	void _pixels_within_tolerance(const Image &original_im, const Image &opencl_im,
tests/test_opencl.h:		double opencl = opencl_im[i];
tests/test_opencl.h:		auto diff = std::abs(original - opencl);
tests/test_opencl.h:			denomin = opencl;
tests/test_opencl.h:		msg << original << " v/s " << opencl;
tests/test_opencl.h:		if( !openCLParameters->opencl_env ) {
tests/test_opencl.h:			TS_SKIP("No OpenCL environment found to run OpenCL tests");
tests/test_opencl.h:		// evaluate normally first, and then using the OpenCL environment,
tests/test_opencl.h:		m.set_opencl_env(openCLParameters->opencl_env);
tests/test_opencl.h:		auto opencl_produced = m.evaluate();
tests/test_opencl.h:			_pixels_within_tolerance(original, opencl_produced, i, openCLParameters->tolerance);
tests/test_opencl.h:		openCLParameters = std::unique_ptr<OpenCLParameters>(new OpenCLParameters());
tests/test_opencl.h:		openCLParameters.reset(nullptr);
tests/test_opencl.h:	void test_no_opencl() {
tests/test_opencl.h:		if (has_opencl()) {
tests/test_opencl.h:			TS_SKIP("OpenCL support is available, skipping test");
tests/test_opencl.h:		TSM_ASSERT("OpenCl environment found, but none expected", get_opencl_info().empty());
tests/test_opencl.h:		TSM_ASSERT_EQUALS("Got OpenCL environment, but none expected", nullptr, get_opencl_environment(0, 0, false, false));
tests/test_opencl.h:	void test_opencldiff_brokenexp() {
tests/test_opencl.h:		_check_opencl_support();
tests/test_opencl.h:	void test_opencldiff_coresersic() {
tests/test_opencl.h:		_check_opencl_support();
tests/test_opencl.h:	void test_opencldiff_ferrer() {
tests/test_opencl.h:		_check_opencl_support();
tests/test_opencl.h:	void test_opencldiff_king() {
tests/test_opencl.h:		_check_opencl_support();
tests/test_opencl.h:	void test_opencldiff_moffat() {
tests/test_opencl.h:		_check_opencl_support();
tests/test_opencl.h:	void test_opencldiff_sersic() {
tests/test_opencl.h:		_check_opencl_support();
tests/test_opencl.h:		_check_opencl_support();
tests/test_opencl.h:		if( !openCLParameters->opencl_env ) {
tests/test_opencl.h:			TS_SKIP("No OpenCL environment found to run OpenCL tests");
tests/test_opencl.h:		prefs.opencl_env = openCLParameters->opencl_env;
tests/test_opencl.h:		_check_convolver(create_convolver(ConvolverType::OPENCL, prefs));
tests/test_opencl.h:	void test_opencl_addition()
tests/test_opencl.h:		_check_opencl_support();
tests/test_opencl.h:		m.set_opencl_env(openCLParameters->opencl_env);
tests/test_convolver.h:		if (has_opencl() && !get_opencl_info().empty()) {
tests/test_convolver.h:			prefs.opencl_env = get_opencl_environment(0, 0, false, false);
tests/test_convolver.h:			_test_psf_bigger_than_image(ConvolverType::OPENCL, prefs);
tests/CMakeLists.txt:set(LIBPROFIT_TEST_NAMES convolver fft image library model opencl profile psf radial sersic sky utils)
tests/test_model.h:		if (has_opencl() && !get_opencl_info().empty()) {
tests/test_model.h:			prefs.opencl_env = get_opencl_environment(0, 0, false, false);
tests/test_model.h:			convolvers.emplace_back(create_convolver(ConvolverType::OPENCL, prefs));
include/profit/brokenexponential.h:#ifdef PROFIT_OPENCL
include/profit/brokenexponential.h:#endif /* PROFIT_OPENCL */
include/profit/opencl_impl.h: * Internal header file for OpenCL functionality
include/profit/opencl_impl.h:#ifndef PROFIT_OPENCL_IMPL_H
include/profit/opencl_impl.h:#define PROFIT_OPENCL_IMPL_H
include/profit/opencl_impl.h:#include "profit/opencl.h"
include/profit/opencl_impl.h:#ifdef PROFIT_OPENCL
include/profit/opencl_impl.h:/* Quickly fail for OpenCL < 1.1 */
include/profit/opencl_impl.h:# if !defined(PROFIT_OPENCL_MAJOR) || !defined(PROFIT_OPENCL_MINOR)
include/profit/opencl_impl.h:#  error "No OpenCL version specified"
include/profit/opencl_impl.h:# elif PROFIT_OPENCL_MAJOR < 1 || (PROFIT_OPENCL_MAJOR == 1 && PROFIT_OPENCL_MINOR < 1 )
include/profit/opencl_impl.h:#  error "libprofit requires at minimum OpenCL >= 1.1"
include/profit/opencl_impl.h:/* MacOS 10.14 (Mojave) started deprecating OpenCL entirely */
include/profit/opencl_impl.h:/* Define the target OpenCL version based on the given major/minor version */
include/profit/opencl_impl.h:# define CL_HPP_TARGET_OPENCL_VERSION  MAKE_VERSION(PROFIT_OPENCL_MAJOR, PROFIT_OPENCL_MINOR)
include/profit/opencl_impl.h:# define CL_TARGET_OPENCL_VERSION  MAKE_VERSION(PROFIT_OPENCL_MAJOR, PROFIT_OPENCL_MINOR)
include/profit/opencl_impl.h:# define CL_HPP_MINIMUM_OPENCL_VERSION PROFIT_OPENCL_TARGET_VERSION
include/profit/opencl_impl.h: * given event as an OpenCL_comand_times structure.
include/profit/opencl_impl.h:OpenCL_command_times cl_cmd_times(const cl::Event &evt);
include/profit/opencl_impl.h:class OpenCLEnvImpl;
include/profit/opencl_impl.h:typedef std::shared_ptr<OpenCLEnvImpl> OpenCLEnvImplPtr;
include/profit/opencl_impl.h: * An OpenCL environment
include/profit/opencl_impl.h:class OpenCLEnvImpl : public OpenCLEnv {
include/profit/opencl_impl.h:	OpenCLEnvImpl(cl::Device device, cl_ver_t version, cl::Context context,
include/profit/opencl_impl.h:	static OpenCLEnvImplPtr fromOpenCLEnvPtr(const OpenCLEnvPtr &ptr) {
include/profit/opencl_impl.h:		return std::static_pointer_cast<OpenCLEnvImpl>(ptr);
include/profit/opencl_impl.h:	// Implementing OpenCL_env's interface
include/profit/opencl_impl.h:	// Implementing OpenCL_env's interface
include/profit/opencl_impl.h:	// Implementing OpenCL_env's interface
include/profit/opencl_impl.h:	 * Returns the amount of memory, in bytes, that each OpenCL Compute Unit
include/profit/opencl_impl.h:	 * by this OpenCL environment.
include/profit/opencl_impl.h:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/opencl_impl.h:	/** The device to be used throughout OpenCL operations */
include/profit/opencl_impl.h:	/** The OpenCL supported by the platform this device belongs to */
include/profit/opencl_impl.h:	/** The OpenCL context used throughout the OpenCL operations */
include/profit/opencl_impl.h:#endif /* PROFIT_OPENCL */
include/profit/opencl_impl.h:#endif /* PROFIT_OPENCL_IMPL_H */
include/profit/opencl.h: * User-facing header file for OpenCL functionality
include/profit/opencl.h:#ifndef PROFIT_OPENCL_H
include/profit/opencl.h:#define PROFIT_OPENCL_H
include/profit/opencl.h: * A datatype for storing an OpenCL version.
include/profit/opencl.h: * It should have the form major*100 + minor*10 (e.g., 120 for OpenCL 1.2)
include/profit/opencl.h: * A structure holding two times associated with OpenCL commands:
include/profit/opencl.h:class PROFIT_API OpenCL_command_times {
include/profit/opencl.h:	OpenCL_command_times &operator+=(const OpenCL_command_times &other);
include/profit/opencl.h:	const OpenCL_command_times operator+(const OpenCL_command_times &other) const;
include/profit/opencl.h: * A structure holding a number of OpenCL command times (filling, writing,
include/profit/opencl.h: * kernel and reading) plus other OpenCL-related times.
include/profit/opencl.h:struct PROFIT_API OpenCL_times {
include/profit/opencl.h:	OpenCL_command_times writing_times;
include/profit/opencl.h:	OpenCL_command_times reading_times;
include/profit/opencl.h:	OpenCL_command_times filling_times;
include/profit/opencl.h:	OpenCL_command_times kernel_times;
include/profit/opencl.h: * An OpenCL environment.
include/profit/opencl.h:class PROFIT_API OpenCLEnv {
include/profit/opencl.h:	virtual ~OpenCLEnv() {};
include/profit/opencl.h:	 * Returns the maximum OpenCL version supported by the underlying device.
include/profit/opencl.h:	 * Returns the name of the OpenCL platform of this environment.
include/profit/opencl.h:	 * @return The name of the OpenCL platform
include/profit/opencl.h:	 * Returns the name of the OpenCL device of this environment.
include/profit/opencl.h:	 * @return The name of the OpenCL device
include/profit/opencl.h:/// Handy typedef for shared pointers to OpenCL_env objects
include/profit/opencl.h:typedef std::shared_ptr<OpenCLEnv> OpenCLEnvPtr;
include/profit/opencl.h: * A structure holding information about a specific OpenCL device
include/profit/opencl.h:typedef struct PROFIT_API _OpenCL_dev_info {
include/profit/opencl.h:	/** The OpenCL version supported by this device */
include/profit/opencl.h:} OpenCL_dev_info;
include/profit/opencl.h: * An structure holding information about a specific OpenCL platform.
include/profit/opencl.h:typedef struct PROFIT_API _OpenCL_plat_info {
include/profit/opencl.h:	/** The supported OpenCL version */
include/profit/opencl.h:	cl_ver_t supported_opencl_version;
include/profit/opencl.h:	std::map<int, OpenCL_dev_info> dev_info;
include/profit/opencl.h:} OpenCL_plat_info;
include/profit/opencl.h: * Queries the system about the OpenCL supported platforms and devices and returns
include/profit/opencl.h: *         OpenCL platforms found on this system.
include/profit/opencl.h:PROFIT_API std::map<int, OpenCL_plat_info> get_opencl_info();
include/profit/opencl.h: * Prepares an OpenCL working space for using with libprofit.
include/profit/opencl.h: * the libprofit OpenCL kernel sources to be used against it, and set up a queue
include/profit/opencl.h: * @param enable_profiling Whether OpenCL profiling capabilities should be
include/profit/opencl.h: *        turned on in the OpenCL Queue created within this envinronment.
include/profit/opencl.h: * @return A pointer to a OpenCL_env structure, which contains the whole set of
include/profit/opencl.h:PROFIT_API OpenCLEnvPtr get_opencl_environment(
include/profit/opencl.h:#endif /* PROFIT_OPENCL_H */
include/profit/ferrer.h:#ifdef PROFIT_OPENCL
include/profit/ferrer.h:#endif /* PROFIT_OPENCL */
include/profit/exceptions.h: * Exception class thrown when an error occurs while dealing with OpenCL.
include/profit/exceptions.h:class PROFIT_API opencl_error : public exception
include/profit/exceptions.h:	explicit opencl_error(const std::string &what);
include/profit/exceptions.h:	~opencl_error() throw();
include/profit/config.h.in:/** Whether libprofit contains OpenCL support */
include/profit/config.h.in:#cmakedefine PROFIT_OPENCL
include/profit/config.h.in: * If OpenCL support is present, the major OpenCL version supported by
include/profit/config.h.in:#define PROFIT_OPENCL_MAJOR @PROFIT_OPENCL_MAJOR@
include/profit/config.h.in: * If OpenCL support is present, the minor OpenCL version supported by
include/profit/config.h.in:#define PROFIT_OPENCL_MINOR @PROFIT_OPENCL_MINOR@
include/profit/config.h.in: * If OpenCL support is present, the target OpenCL version supported by
include/profit/config.h.in:#define PROFIT_OPENCL_TARGET_VERSION @PROFIT_OPENCL_TARGET_VERSION@
include/profit/king.h:#ifdef PROFIT_OPENCL
include/profit/king.h:#endif /* PROFIT_OPENCL */
include/profit/model.h:#include "profit/opencl.h"
include/profit/model.h:	void set_opencl_env(const OpenCLEnvPtr &opencl_env) {
include/profit/model.h:		this->opencl_env = opencl_env;
include/profit/model.h:	OpenCLEnvPtr get_opencl_env() const {
include/profit/model.h:		return opencl_env;
include/profit/model.h:	OpenCLEnvPtr opencl_env;
include/profit/radial.h:#include "profit/opencl_impl.h"
include/profit/radial.h:	/// Whether the CPU evaluation method should be used, even if an OpenCL
include/profit/radial.h:	/// environment has been given (and libprofit has been compiled with OpenCL support)
include/profit/radial.h:#ifdef PROFIT_OPENCL
include/profit/radial.h:	 * Indicates whether this profile supports OpenCL evaluation or not
include/profit/radial.h:	 * (i.e., implements the required OpenCL kernels)
include/profit/radial.h:	 * @return Whether this profile supports OpenCL evaluation. The default
include/profit/radial.h:	virtual bool supports_opencl() const;
include/profit/radial.h:	/* Evaluates this radial profile using an OpenCL kernel and floating type FT */
include/profit/radial.h:	void evaluate_opencl(Image &image, const Mask &mask, const PixelScale &scale, OpenCLEnvImplPtr &env);
include/profit/radial.h:#endif /* PROFIT_OPENCL */
include/profit/coresersic.h:#ifdef PROFIT_OPENCL
include/profit/coresersic.h:#endif /* PROFIT_OPENCL */
include/profit/profile.h:#include "profit/opencl.h"
include/profit/profile.h:	OpenCL_times cl_times;
include/profit/profile.h:	OpenCL_times cl_times;
include/profit/profit.h:#include "profit/opencl.h"
include/profit/moffat.h:#ifdef PROFIT_OPENCL
include/profit/moffat.h:#endif /* PROFIT_OPENCL */
include/profit/convolve.h:#include "profit/opencl.h"
include/profit/convolve.h:	/// @copydoc OpenCLConvolver
include/profit/convolve.h:	OPENCL,
include/profit/convolve.h:		opencl_env(),
include/profit/convolve.h:	    OpenCLEnvPtr opencl_env, effort_t effort, bool reuse_krn_fft,
include/profit/convolve.h:		opencl_env(opencl_env),
include/profit/convolve.h:	/// A pointer to an OpenCL environment. Used by the OPENCL convolvers.
include/profit/convolve.h:	OpenCLEnvPtr opencl_env;
include/profit/sersic.h:#ifdef PROFIT_OPENCL
include/profit/sersic.h:#endif /* PROFIT_OPENCL */
include/profit/convolver_impl.h:#include "profit/opencl_impl.h"
include/profit/convolver_impl.h:#ifdef PROFIT_OPENCL
include/profit/convolver_impl.h: * A brute-force convolver that is implemented using OpenCL
include/profit/convolver_impl.h: * Depending on the floating-point support found at runtime in the given OpenCL
include/profit/convolver_impl.h:class OpenCLConvolver : public Convolver {
include/profit/convolver_impl.h:	explicit OpenCLConvolver(OpenCLEnvImplPtr opencl_env);
include/profit/convolver_impl.h:	OpenCLEnvImplPtr env;
include/profit/convolver_impl.h:	// returns the extra OpenCL-imposed padding
include/profit/convolver_impl.h: * Like OpenCLConvolver, but uses a local memory cache
include/profit/convolver_impl.h:class OpenCLLocalConvolver : public Convolver {
include/profit/convolver_impl.h:	explicit OpenCLLocalConvolver(OpenCLEnvImplPtr opencl_env);
include/profit/convolver_impl.h:	OpenCLEnvImplPtr env;
include/profit/convolver_impl.h:#endif // PROFIT_OPENCL
include/profit/cl/cl2.hpp: *   \brief C++ bindings for OpenCL 1.0 (rev 48), OpenCL 1.1 (rev 33),
include/profit/cl/cl2.hpp: *       OpenCL 1.2 (rev 15) and OpenCL 2.0 (rev 29)
include/profit/cl/cl2.hpp: *   Derived from the OpenCL 1.x C++ bindings written by
include/profit/cl/cl2.hpp: *       http://khronosgroup.github.io/OpenCL-CLHPP/
include/profit/cl/cl2.hpp: *       https://github.com/KhronosGroup/OpenCL-CLHPP/releases
include/profit/cl/cl2.hpp: *       https://github.com/KhronosGroup/OpenCL-CLHPP
include/profit/cl/cl2.hpp: * reasonable to define C++ bindings for OpenCL.
include/profit/cl/cl2.hpp: * fixes in the new header as well as additional OpenCL 2.0 features.
include/profit/cl/cl2.hpp: * Due to the evolution of the underlying OpenCL API the 2.0 C++ bindings
include/profit/cl/cl2.hpp: * and the range of valid underlying OpenCL runtime versions supported.
include/profit/cl/cl2.hpp: * The combination of preprocessor macros CL_HPP_TARGET_OPENCL_VERSION and 
include/profit/cl/cl2.hpp: * CL_HPP_MINIMUM_OPENCL_VERSION control this range. These are three digit
include/profit/cl/cl2.hpp: * decimal values representing OpenCL runime versions. The default for 
include/profit/cl/cl2.hpp: * the target is 200, representing OpenCL 2.0 and the minimum is also 
include/profit/cl/cl2.hpp: * The OpenCL 1.x versions of the C++ bindings included a size_t wrapper
include/profit/cl/cl2.hpp: * In OpenCL 2.0 OpenCL C is not entirely backward compatibility with 
include/profit/cl/cl2.hpp: * earlier versions. As a result a flag must be passed to the OpenCL C
include/profit/cl/cl2.hpp: * compiled to request OpenCL 2.0 compilation of kernels with 1.2 as
include/profit/cl/cl2.hpp: * For those cases the compilation defaults to OpenCL C 2.0.
include/profit/cl/cl2.hpp: * - CL_HPP_TARGET_OPENCL_VERSION
include/profit/cl/cl2.hpp: *   Defines the target OpenCL runtime version to build the header
include/profit/cl/cl2.hpp: *   against. Defaults to 200, representing OpenCL 2.0.
include/profit/cl/cl2.hpp: *   Enables device fission for OpenCL 1.2 platforms.
include/profit/cl/cl2.hpp: *   Default to OpenCL C 1.2 compilation rather than OpenCL C 2.0
include/profit/cl/cl2.hpp:    #define CL_HPP_TARGET_OPENCL_VERSION 200
include/profit/cl/cl2.hpp:            if (platver.find("OpenCL 2.") != std::string::npos) {
include/profit/cl/cl2.hpp:            std::cout << "No OpenCL 2.0 platform found.";
include/profit/cl/cl2.hpp:#if !defined(CL_HPP_TARGET_OPENCL_VERSION)
include/profit/cl/cl2.hpp:# pragma message("cl2.hpp: CL_HPP_TARGET_OPENCL_VERSION is not defined. It will default to 200 (OpenCL 2.0)")
include/profit/cl/cl2.hpp:# define CL_HPP_TARGET_OPENCL_VERSION 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION != 100 && CL_HPP_TARGET_OPENCL_VERSION != 110 && CL_HPP_TARGET_OPENCL_VERSION != 120 && CL_HPP_TARGET_OPENCL_VERSION != 200
include/profit/cl/cl2.hpp:# pragma message("cl2.hpp: CL_HPP_TARGET_OPENCL_VERSION is not a valid value (100, 110, 120 or 200). It will be set to 200")
include/profit/cl/cl2.hpp:# undef CL_HPP_TARGET_OPENCL_VERSION
include/profit/cl/cl2.hpp:# define CL_HPP_TARGET_OPENCL_VERSION 200
include/profit/cl/cl2.hpp:#if !defined(CL_HPP_MINIMUM_OPENCL_VERSION)
include/profit/cl/cl2.hpp:# define CL_HPP_MINIMUM_OPENCL_VERSION 200
include/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION != 100 && CL_HPP_MINIMUM_OPENCL_VERSION != 110 && CL_HPP_MINIMUM_OPENCL_VERSION != 120 && CL_HPP_MINIMUM_OPENCL_VERSION != 200
include/profit/cl/cl2.hpp:# pragma message("cl2.hpp: CL_HPP_MINIMUM_OPENCL_VERSION is not a valid value (100, 110, 120 or 200). It will be set to 100")
include/profit/cl/cl2.hpp:# undef CL_HPP_MINIMUM_OPENCL_VERSION
include/profit/cl/cl2.hpp:# define CL_HPP_MINIMUM_OPENCL_VERSION 100
include/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION > CL_HPP_TARGET_OPENCL_VERSION
include/profit/cl/cl2.hpp:# error "CL_HPP_MINIMUM_OPENCL_VERSION must not be greater than CL_HPP_TARGET_OPENCL_VERSION"
include/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 100 && !defined(CL_USE_DEPRECATED_OPENCL_1_0_APIS)
include/profit/cl/cl2.hpp:# define CL_USE_DEPRECATED_OPENCL_1_0_APIS
include/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 110 && !defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
include/profit/cl/cl2.hpp:# define CL_USE_DEPRECATED_OPENCL_1_1_APIS
include/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 120 && !defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
include/profit/cl/cl2.hpp:# define CL_USE_DEPRECATED_OPENCL_1_2_APIS
include/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 200 && !defined(CL_USE_DEPRECATED_OPENCL_2_0_APIS)
include/profit/cl/cl2.hpp:# define CL_USE_DEPRECATED_OPENCL_2_0_APIS
include/profit/cl/cl2.hpp:#include <OpenCL/opencl.h>
include/profit/cl/cl2.hpp:#include <CL/opencl.h>
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:        *  OpenCL C calls that require arrays of size_t values, whose
include/profit/cl/cl2.hpp: * \brief The OpenCL C++ bindings are defined within this namespace.
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
include/profit/cl/cl2.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
include/profit/cl/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
include/profit/cl/cl2.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:    F(cl_device_info, CL_DEVICE_OPENCL_C_VERSION, string) \
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
include/profit/cl/cl2.hpp:// Flags deprecated in OpenCL 2.0
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION > 100 && CL_HPP_MINIMUM_OPENCL_VERSION < 200 && CL_HPP_TARGET_OPENCL_VERSION < 200
include/profit/cl/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 110
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION > 110 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
include/profit/cl/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION > 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
include/profit/cl/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
include/profit/cl/cl2.hpp:#ifdef CL_DEVICE_GPU_OVERLAP_NV
include/profit/cl/cl2.hpp:CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_GPU_OVERLAP_NV, cl_bool)
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp: * OpenCL 1.2 devices do have retain/release.
include/profit/cl/cl2.hpp:#else // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp: * OpenCL 1.1 devices do not have retain/release.
include/profit/cl/cl2.hpp:#endif // ! (CL_HPP_TARGET_OPENCL_VERSION >= 120)
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
include/profit/cl/cl2.hpp:#else // CL_HPP_MINIMUM_OPENCL_VERSION < 120
include/profit/cl/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:     *  \param devices returns a vector of OpenCL D3D10 devices found. The cl::Device
include/profit/cl/cl2.hpp:     *  values returned in devices can be used to identify a specific OpenCL
include/profit/cl/cl2.hpp:     *  The application can query specific capabilities of the OpenCL device(s)
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
include/profit/cl/cl2.hpp: * Unload the OpenCL compiler.
include/profit/cl/cl2.hpp: * \note Deprecated for OpenCL 1.2. Use Platform::unloadCompiler instead.
include/profit/cl/cl2.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
include/profit/cl/cl2.hpp:/*! \brief Class interface for creating OpenCL buffers from ID3D10Buffer's.
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 110
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
include/profit/cl/cl2.hpp:            useCreateImage = (version >= 0x10002); // OpenCL 1.2 or above
include/profit/cl/cl2.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
include/profit/cl/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 || defined(CL_HPP_USE_CL_IMAGE2D_FROM_BUFFER_KHR)
include/profit/cl/cl2.hpp:#endif //#if CL_HPP_TARGET_OPENCL_VERSION >= 200 || defined(CL_HPP_USE_CL_IMAGE2D_FROM_BUFFER_KHR)
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:    *              The channel order may differ as described in the OpenCL 
include/profit/cl/cl2.hpp:#endif //#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
include/profit/cl/cl2.hpp: *  \note Deprecated for OpenCL 1.2. Please use ImageGL instead.
include/profit/cl/cl2.hpp:#endif // CL_USE_DEPRECATED_OPENCL_1_1_APIS
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
include/profit/cl/cl2.hpp:            useCreateImage = (version >= 0x10002); // OpenCL 1.2 or above
include/profit/cl/cl2.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#endif  // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
include/profit/cl/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
include/profit/cl/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
include/profit/cl/cl2.hpp:#endif // CL_USE_DEPRECATED_OPENCL_1_1_APIS
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp: * was performed by OpenCL anyway.
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:     * \param context A valid OpenCL context in which to construct the program.
include/profit/cl/cl2.hpp:     * \param devices A vector of OpenCL device objects for which the program will be created.
include/profit/cl/cl2.hpp:     *   CL_INVALID_DEVICE if OpenCL devices listed in devices are not in the list of devices associated with context.
include/profit/cl/cl2.hpp:     *   CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required by the OpenCL implementation on the host.
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
include/profit/cl/cl2.hpp:                useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
include/profit/cl/cl2.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
include/profit/cl/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
include/profit/cl/cl2.hpp:               useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
include/profit/cl/cl2.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
include/profit/cl/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
include/profit/cl/cl2.hpp:            useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
include/profit/cl/cl2.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
include/profit/cl/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
include/profit/cl/cl2.hpp:            useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
include/profit/cl/cl2.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
include/profit/cl/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
include/profit/cl/cl2.hpp:            useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
include/profit/cl/cl2.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
include/profit/cl/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
include/profit/cl/cl2.hpp:            useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
include/profit/cl/cl2.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
include/profit/cl/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#else // CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:     *     The pattern type must be an accepted OpenCL data type.
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:     * Enqueues a command that will release a coarse-grained SVM buffer back to the OpenCL runtime.
include/profit/cl/cl2.hpp:     * Enqueues a command that will release a coarse-grained SVM buffer back to the OpenCL runtime.
include/profit/cl/cl2.hpp:     * Enqueues a command that will release a coarse-grained SVM buffer back to the OpenCL runtime.
include/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
include/profit/cl/cl2.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
include/profit/cl/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
include/profit/cl/cl2.hpp:#endif // defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
include/profit/cl/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
include/profit/cl/cl2.hpp:#endif // CL_USE_DEPRECATED_OPENCL_1_1_APIS
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp: * SVM buffer back to the OpenCL runtime.
include/profit/cl/cl2.hpp: * SVM buffer back to the OpenCL runtime.
include/profit/cl/cl2.hpp: * SVM buffer back to the OpenCL runtime.
include/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
include/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
include/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/profit/library.h:/// Returns whether libprofit was compiled with OpenCL support
include/profit/library.h:/// @return Whether libprofit was compiled with OpenCL support
include/profit/library.h:PROFIT_API bool has_opencl();
include/profit/library.h:/// If OpenCL is supported, returns the major portion of the highest OpenCL
include/profit/library.h:/// compiled against a platform supporting OpenCL 2.1, this method returns 2.
include/profit/library.h:/// If OpenCL is not supported, the result is undefined.
include/profit/library.h:/// @return The major highest OpenCL platform version that libprofit can work
include/profit/library.h:PROFIT_API unsigned short opencl_version_major();
include/profit/library.h:/// If OpenCL is supported, returns the minor portion of the highest OpenCL
include/profit/library.h:/// compiled against a platform supporting OpenCL 1.2, this method returns 2.
include/profit/library.h:/// If OpenCL is not supported, the result is undefined.
include/profit/library.h:PROFIT_API unsigned short opencl_version_minor();
CMakeLists.txt:option(LIBPROFIT_NO_OPENCL "Don't attempt to include OpenCL support in libprofit" OFF)
CMakeLists.txt:#  * OpenCL
CMakeLists.txt:macro(find_opencl)
CMakeLists.txt:	find_package(OpenCL)
CMakeLists.txt:	if( OpenCL_FOUND )
CMakeLists.txt:		if ("${OpenCL_VERSION_STRING}" STREQUAL "")
CMakeLists.txt:			message("-- OpenCL found but no version reported. Compiling without OpenCL support")
CMakeLists.txt:			include_directories(${OpenCL_INCLUDE_DIRS})
CMakeLists.txt:			set(PROFIT_OPENCL ON)
CMakeLists.txt:			set(PROFIT_OPENCL_MAJOR ${OpenCL_VERSION_MAJOR})
CMakeLists.txt:			set(PROFIT_OPENCL_MINOR ${OpenCL_VERSION_MINOR})
CMakeLists.txt:			# We usually set our target version to be OpenCL 1.1,
CMakeLists.txt:			set(PROFIT_OPENCL_TARGET_VERSION 110)
CMakeLists.txt:					set(PROFIT_OPENCL_TARGET_VERSION 120)
CMakeLists.txt:			# shipping with OpenCL versions above 2.0, which is the
CMakeLists.txt:			if (${PROFIT_OPENCL_MAJOR} EQUAL 2 AND NOT ${PROFIT_OPENCL_MINOR} EQUAL 0)
CMakeLists.txt:				set(PROFIT_OPENCL_MINOR 0)
CMakeLists.txt:			set(profit_LIBS ${profit_LIBS} ${OpenCL_LIBRARIES})
CMakeLists.txt:# Check if there's OpenCL (users might opt out)
CMakeLists.txt:if( NOT LIBPROFIT_NO_OPENCL )
CMakeLists.txt:	find_opencl()
CMakeLists.txt:# Generate header files with the OpenCL kernel sources out of each individual
CMakeLists.txt:if (PROFIT_OPENCL)
CMakeLists.txt:set(OPENCL_KERNEL_HEADERS "")
CMakeLists.txt:			         -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateOpenCLHeader.cmake
CMakeLists.txt:			list(APPEND OPENCL_KERNEL_HEADERS ${KRN_HEADER_FNAME})
CMakeLists.txt:   src/opencl.cpp
CMakeLists.txt:add_library(profit ${LIB_TYPE} ${PROFIT_SRC} ${OPENCL_KERNEL_HEADERS})
CMakeLists.txt:        include/profit/opencl.h
CMakeLists.txt:		if (PROFIT_OPENCL)
CMakeLists.txt:			add_test("cli-opencl-list" profit-cli -c)
cmake/GenerateOpenCLHeader.cmake:# Generates OpenCL .h files out from the corresponding .cl
cmake/GenerateOpenCLHeader.cmake: * C++-compatible OpenCL kernel source code from ${KRN_NAME}
.travis.yml:      packages: [g++-4.6, libfftw3-dev, libgsl0-dev, opencl-headers, cxxtest]
.travis.yml:      packages: [g++-4.7, libfftw3-dev, libgsl0-dev, opencl-headers, cxxtest]
.travis.yml:      packages: [g++-4.9, libfftw3-dev, libgsl0-dev, opencl-headers, cxxtest]
.travis.yml:      packages: [g++-5, libfftw3-dev, libgsl0-dev, opencl-headers, cxxtest]
.travis.yml:      packages: [g++-6, libfftw3-dev, libgsl0-dev, opencl-headers, cxxtest]
.travis.yml:      packages: [libfftw3-dev, libgsl0-dev, opencl-headers, cxxtest]
src/exceptions.cpp:opencl_error::opencl_error(const std::string &what_arg) :
src/exceptions.cpp:opencl_error::~opencl_error() throw () {
src/king.cpp:#ifdef PROFIT_OPENCL
src/king.cpp:#endif /* PROFIT_OPENCL */
src/radial.cpp:#include "profit/opencl.h"
src/radial.cpp:#ifndef PROFIT_OPENCL
src/radial.cpp:	 * We fallback to the CPU implementation if no OpenCL context has been
src/radial.cpp:	 * given, or if there is no OpenCL kernel implementing the profile
src/radial.cpp:	auto env = OpenCLEnvImpl::fromOpenCLEnvPtr(model.get_opencl_env());
src/radial.cpp:	if( force_cpu || !env || !supports_opencl() ) {
src/radial.cpp:			evaluate_opencl<double>(image, mask, scale, env);
src/radial.cpp:			evaluate_opencl<float>(image, mask, scale, env);
src/radial.cpp:		os << "OpenCL error: " << e.what() << ". OpenCL error code: " << e.err();
src/radial.cpp:		throw opencl_error(os.str());
src/radial.cpp:#endif /* PROFIT_OPENCL */
src/radial.cpp:#ifdef PROFIT_OPENCL
src/radial.cpp:void RadialProfile::evaluate_opencl(Image &image, const Mask & /*mask*/, const PixelScale &scale, OpenCLEnvImplPtr &env) {
src/radial.cpp:	OpenCL_times cl_times0 {};
src/radial.cpp:	OpenCL_times ss_cl_times {};
src/radial.cpp:	system_clock::time_point t0, t_kprep, t_opencl, t_loopstart, t_loopend, t_imgtrans;
src/radial.cpp:	// OpenCL 1.2 allows to do this; otherwise the work has to be done in the kernel
src/radial.cpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/radial.cpp:#endif /* CL_HPP_TARGET_OPENCL_VERSION >= 120 */
src/radial.cpp:	t_opencl = system_clock::now();
src/radial.cpp:	stats->final_image += to_nsecs(system_clock::now() - t_opencl);
src/radial.cpp:	/* These are the OpenCL-related timings so far */
src/radial.cpp:	cl_times0.total = to_nsecs(t_opencl - t_kprep);
src/radial.cpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/radial.cpp:#endif /* CL_HPP_TARGET_OPENCL_VERSION >= 120 */
src/radial.cpp:		system_clock::time_point t0, t_newsamples, t_trans_h2k, t_kprep, t_opencl, t_trans_k2h;
src/radial.cpp:			t_opencl = system_clock::now();
src/radial.cpp:			stats->subsampling.final_transform += to_nsecs(t_trans_k2h - t_opencl);
src/radial.cpp:			ss_cl_times.total += to_nsecs(t_opencl - t_trans_h2k);
src/radial.cpp:	stats->subsampling.pre_subsampling = to_nsecs(t_loopstart - t_opencl);
src/radial.cpp:bool RadialProfile::supports_opencl() const {
src/radial.cpp:#endif /* PROFIT_OPENCL */
src/radial.cpp:#ifdef PROFIT_OPENCL
src/radial.cpp:#endif /* PROFIT_OPENCL */
src/convolve.cpp:#ifdef PROFIT_OPENCL
src/convolve.cpp:OpenCLConvolver::OpenCLConvolver(OpenCLEnvImplPtr opencl_env) :
src/convolve.cpp:	env(std::move(opencl_env))
src/convolve.cpp:		throw invalid_parameter("Empty OpenCL environment given to OpenCLConvolver");
src/convolve.cpp:Image OpenCLConvolver::convolve_impl(const Image &src, const Image &krn, const Mask &mask, bool crop, Point &offset_out)
src/convolve.cpp:		os << "OpenCL error while convolving: " << e.what() << ". OpenCL error code: " << e.err();
src/convolve.cpp:		throw opencl_error(os.str());
src/convolve.cpp:Dimensions OpenCLConvolver::cl_padding(const Dimensions &src_dims) const
src/convolve.cpp:PointPair OpenCLConvolver::padding(const Dimensions &src_dims, const Dimensions &/*krn_dims*/) const
src/convolve.cpp:Image OpenCLConvolver::_convolve(const Image &src, const Image &krn, const Mask &mask, bool crop, Point &offset_out) {
src/convolve.cpp:Image OpenCLConvolver::_clpadded_convolve(const Image &src, const Image &krn, const Image &orig_src) {
src/convolve.cpp:OpenCLLocalConvolver::OpenCLLocalConvolver(OpenCLEnvImplPtr opencl_env) :
src/convolve.cpp:	env(std::move(opencl_env))
src/convolve.cpp:		throw invalid_parameter("Empty OpenCL environment given to OpenCLLocalConvolver");
src/convolve.cpp:Image OpenCLLocalConvolver::convolve_impl(const Image &src, const Image &krn, const Mask &mask, bool crop, Point &offset_out)
src/convolve.cpp:		os << "OpenCL error while convolving: " << e.what() << ". OpenCL error code: " << e.err();
src/convolve.cpp:		throw opencl_error(os.str());
src/convolve.cpp:Image OpenCLLocalConvolver::_convolve(const Image &src, const Image &krn, const Mask &mask, bool crop, Point &offset_out) {
src/convolve.cpp:Image OpenCLLocalConvolver::_clpadded_convolve(const Image &src, const Image &krn, const Image &orig_src) {
src/convolve.cpp:		os << "Not enough local memory available for OpenCL local 2D convolution. ";
src/convolve.cpp:		throw opencl_error(os.str());
src/convolve.cpp:#endif // PROFIT_OPENCL
src/convolve.cpp:#ifdef PROFIT_OPENCL
src/convolve.cpp:		case OPENCL:
src/convolve.cpp:			return std::make_shared<OpenCLConvolver>(OpenCLEnvImpl::fromOpenCLEnvPtr(prefs.opencl_env));
src/convolve.cpp:#endif // PROFIT_OPENCL
src/convolve.cpp:#ifdef PROFIT_OPENCL
src/convolve.cpp:	else if (type == "opencl") {
src/convolve.cpp:		return create_convolver(OPENCL, prefs);
src/convolve.cpp:#endif // PROFIT_OPENCL
src/profit-cli.cpp:	os << "OpenCL support: ";
src/profit-cli.cpp:	if (has_opencl()) {
src/profit-cli.cpp:		os << "Yes (up to " << opencl_version_major() << "." << opencl_version_minor() << ")" << endl;
src/profit-cli.cpp:  -C <p,d>  Use OpenCL with platform p, device d, and double support (0|1)
src/profit-cli.cpp:  -c        Display OpenCL information about devices and platforms
src/profit-cli.cpp: * opencl: An OpenCL-based brute-force convolver
src/profit-cli.cpp:void print_opencl_info(std::ostream &out) {
src/profit-cli.cpp:	const auto info = get_opencl_info();
src/profit-cli.cpp:		out << "OpenCL information" << endl;
src/profit-cli.cpp:			out << "  OpenCL version : " << clver{plat_info.supported_opencl_version} << endl;
src/profit-cli.cpp:				out << "    OpenCL version : " << clver {std::get<1>(device_info).cl_version} << endl;
src/profit-cli.cpp:		out << "No OpenCL installation found" << endl;
src/profit-cli.cpp:void print_cl_command_times(std::ostream &os, const std::string &prefix, const OpenCL_command_times &t, const std::string &action)
src/profit-cli.cpp:void print_cl_stats(std::ostream &os, const std::string &prefix0, bool opencl_120, const OpenCL_times &stats) {
src/profit-cli.cpp:	cl_ops_os << "OpenCL operations (" << stats.nwork_items << " work items)";
src/profit-cli.cpp:	if( opencl_120 ) {
src/profit-cli.cpp:		auto opencl_env = m.get_opencl_env();
src/profit-cli.cpp:		if( rprofile_stats && opencl_env ) {
src/profit-cli.cpp:			bool opencl_120 = opencl_env->get_version() >= 120;
src/profit-cli.cpp:			print_cl_stats(os, prefix0, opencl_120, rprofile_stats->cl_times);
src/profit-cli.cpp:			print_cl_stats(os, prefix1, opencl_120, rprofile_stats->subsampling.cl_times);
src/profit-cli.cpp:	bool use_opencl = false;
src/profit-cli.cpp:				print_opencl_info(cout);
src/profit-cli.cpp:				if (!has_opencl()) {
src/profit-cli.cpp:					throw invalid_cmdline("libprofit was compiled without OpenCL support, but support was requested. See -V for details");
src/profit-cli.cpp:				use_opencl = true;
src/profit-cli.cpp:	/* Get an OpenCL environment */
src/profit-cli.cpp:	if( use_opencl ) {
src/profit-cli.cpp:		auto opencl_env = get_opencl_environment(clplat_idx, cldev_idx, use_double, show_stats);
src/profit-cli.cpp:		m.set_opencl_env(opencl_env);
src/profit-cli.cpp:		convolver_prefs.opencl_env = opencl_env;
src/profit-cli.cpp:		auto opencl_duration = chrono::duration_cast<chrono::milliseconds>(end-start).count();
src/profit-cli.cpp:		cout << "OpenCL environment (platform=" <<
src/profit-cli.cpp:		        opencl_env->get_platform_name() << ", device=" <<
src/profit-cli.cpp:		        opencl_env->get_device_name() << ", version=" <<
src/profit-cli.cpp:		        clver{opencl_env->get_version()} <<
src/profit-cli.cpp:		        ") created in " << opencl_duration << " [ms]" << std::endl;
src/profit-cli.cpp:	catch (const profit::opencl_error &e) {
src/profit-cli.cpp:		cerr << "Error in OpenCL operation: " << e.what() << std::endl;
src/opencl.cpp: * OpenCL utility methods for libprofit
src/opencl.cpp:#include "profit/opencl_impl.h"
src/opencl.cpp:#ifdef PROFIT_OPENCL
src/opencl.cpp:#endif // PROFIT_OPENCL
src/opencl.cpp:OpenCL_command_times &OpenCL_command_times::operator+=(const OpenCL_command_times &other) {
src/opencl.cpp:const OpenCL_command_times OpenCL_command_times::operator+(const OpenCL_command_times &other) const {
src/opencl.cpp:	OpenCL_command_times t1 = *this;
src/opencl.cpp:// Simple implementation of public methods for non-OpenCL builds
src/opencl.cpp:#ifndef PROFIT_OPENCL
src/opencl.cpp:std::map<int, OpenCL_plat_info> get_opencl_info() {
src/opencl.cpp:	return std::map<int, OpenCL_plat_info>();
src/opencl.cpp:OpenCLEnvPtr get_opencl_environment(unsigned int platform_idx, unsigned int device_idx, bool use_double, bool enable_profiling)
src/opencl.cpp:// Functions to read the duration of OpenCL events (queue->submit and start->end)
src/opencl.cpp:OpenCL_command_times cl_cmd_times(const cl::Event &evt) {
src/opencl.cpp:static cl_ver_t get_opencl_version(const std::string &version)
src/opencl.cpp:	// Version string should be of type "OpenCL<space><major_version.minor_version><space><platform-specific information>"
src/opencl.cpp:	if( version.find("OpenCL ") != 0) {
src/opencl.cpp:		throw opencl_error(std::string("OpenCL version string doesn't start with 'OpenCL ': ") + version);
src/opencl.cpp:	auto opencl_version = version.substr(7, next_space);
src/opencl.cpp:	auto dot_idx = opencl_version.find(".");
src/opencl.cpp:	if( dot_idx == opencl_version.npos ) {
src/opencl.cpp:		throw opencl_error("OpenCL version doesn't contain a dot: " + opencl_version);
src/opencl.cpp:	auto major = stoui(opencl_version.substr(0, dot_idx));
src/opencl.cpp:	auto minor = stoui(opencl_version.substr(dot_idx+1, opencl_version.npos));
src/opencl.cpp:static cl_ver_t get_opencl_version(const cl::Platform &platform) {
src/opencl.cpp:	return get_opencl_version(platform.getInfo<CL_PLATFORM_VERSION>());
src/opencl.cpp:static cl_ver_t get_opencl_version(const cl::Device &device) {
src/opencl.cpp:	return get_opencl_version(device.getInfo<CL_DEVICE_VERSION>());
src/opencl.cpp:	if (get_opencl_version(device) < 120) {
src/opencl.cpp:std::map<int, OpenCL_plat_info> _get_opencl_info() {
src/opencl.cpp:	std::map<int, OpenCL_plat_info> pinfo;
src/opencl.cpp:		std::map<int, OpenCL_dev_info> dinfo;
src/opencl.cpp:			dinfo[didx++] = OpenCL_dev_info{
src/opencl.cpp:				get_opencl_version(device),
src/opencl.cpp:		pinfo[pidx++] = OpenCL_plat_info{name, get_opencl_version(platform), dinfo};
src/opencl.cpp:std::map<int, OpenCL_plat_info> get_opencl_info() {
src/opencl.cpp:		return _get_opencl_info();
src/opencl.cpp:		os << "OpenCL error: " << e.what() << ". OpenCL error code: " << e.err();
src/opencl.cpp:		throw opencl_error(os.str());
src/opencl.cpp:	auto plat_part = valid_fname(plat.getInfo<CL_PLATFORM_NAME>()) + "_" + std::to_string(get_opencl_version(plat));
src/opencl.cpp:	auto dev_part = valid_fname(device.getInfo<CL_DEVICE_NAME>()) + "_" + std::to_string(get_opencl_version(device));
src/opencl.cpp:	auto the_dir = create_dirs(get_profit_home(), {std::string("opencl_cache"), plat_part});
src/opencl.cpp:		throw opencl_error("Error building program: " + program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
src/opencl.cpp:			throw opencl_error(os.str());
src/opencl.cpp:		throw opencl_error("Error while getting OpenCL platforms");
src/opencl.cpp:		throw opencl_error("No platforms found. Check OpenCL installation");
src/opencl.cpp:		ss << "OpenCL platform index " << platform_idx << " must be < " << all_platforms.size();
src/opencl.cpp:		throw opencl_error("No devices found. Check OpenCL installation");
src/opencl.cpp:		ss << "OpenCL device index " << device_idx << " must be < " << all_devices.size();
src/opencl.cpp:		throw opencl_error("Double precision requested but not supported by device");
src/opencl.cpp:OpenCLEnvPtr _get_opencl_environment(unsigned int platform_idx, unsigned int device_idx, bool use_double, bool enable_profiling) {
src/opencl.cpp:	return std::make_shared<OpenCLEnvImpl>(device, get_opencl_version(device), context, queue, program, use_double, enable_profiling);
src/opencl.cpp:OpenCLEnvPtr get_opencl_environment(unsigned int platform_idx, unsigned int device_idx, bool use_double, bool enable_profiling) {
src/opencl.cpp:		return _get_opencl_environment(platform_idx, device_idx, use_double, enable_profiling);
src/opencl.cpp:		os << "OpenCL error: " << e.what() << ". OpenCL error code: " << e.err();
src/opencl.cpp:		throw opencl_error(os.str());
src/opencl.cpp:unsigned long OpenCLEnvImpl::max_local_memory() {
src/opencl.cpp:unsigned int OpenCLEnvImpl::compute_units() {
src/opencl.cpp:cl::Event OpenCLEnvImpl::queue_write(const cl::Buffer &buffer, const void *data, const std::vector<cl::Event>* wait_evts) {
src/opencl.cpp:cl::Event OpenCLEnvImpl::queue_kernel(const cl::Kernel &kernel, const cl::NDRange global, const std::vector<cl::Event>* wait_evts, const cl::NDRange &local) {
src/opencl.cpp:cl::Event OpenCLEnvImpl::queue_read(const cl::Buffer &buffer, void *data, const std::vector<cl::Event>* wait_evts) {
src/opencl.cpp:cl::Kernel OpenCLEnvImpl::get_kernel(const std::string &name) {
src/opencl.cpp:#endif /* PROFIT_OPENCL */
src/moffat.cpp:#ifdef PROFIT_OPENCL
src/moffat.cpp:#endif /* PROFIT_OPENCL */
src/brokenexponential.cpp:#ifdef PROFIT_OPENCL
src/brokenexponential.cpp:#endif /* PROFIT_OPENCL */
src/library.cpp:bool has_opencl()
src/library.cpp:#ifdef PROFIT_OPENCL
src/library.cpp:#endif // PROFIT_OPENCL
src/library.cpp:unsigned short opencl_version_major()
src/library.cpp:#ifdef PROFIT_OPENCL
src/library.cpp:	return PROFIT_OPENCL_MAJOR;
src/library.cpp:#endif // PROFIT_OPENCL
src/library.cpp:unsigned short opencl_version_minor()
src/library.cpp:#ifdef PROFIT_OPENCL
src/library.cpp:	return PROFIT_OPENCL_MINOR;
src/library.cpp:#endif // PROFIT_OPENCL
src/library.cpp:#ifdef PROFIT_OPENCL
src/library.cpp:	auto opencl_cache = profit_home + "/opencl_cache";
src/library.cpp:	if (dir_exists(opencl_cache)) {
src/library.cpp:		recursive_remove(opencl_cache);
src/coresersic.cpp:#ifdef PROFIT_OPENCL
src/coresersic.cpp:#endif /* PROFIT_OPENCL */
src/ferrer.cpp:#ifdef PROFIT_OPENCL
src/ferrer.cpp:#endif /* PROFIT_OPENCL */
src/cl/brokenexponential-double.cl: * Double-precision Broken Exponential profile OpenCL kernel implementation for libprofit
src/cl/brokenexponential-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/brokenexponential-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/brokenexponential-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/brokenexponential-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/brokenexponential-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/brokenexponential-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/convolve-float.cl: * float 2D convolution OpenCL implementation for libprofit
src/cl/ferrer-float.cl: * Single-precision Ferrer profile OpenCL kernel implementation for libprofit
src/cl/ferrer-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/ferrer-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/ferrer-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/ferrer-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/ferrer-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/ferrer-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/moffat-double.cl: * Double-precision Moffat profile OpenCL kernel implementation for libprofit
src/cl/moffat-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/moffat-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/moffat-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/moffat-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/moffat-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/moffat-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/moffat-float.cl: * Single-precision Moffat profile OpenCL kernel implementation for libprofit
src/cl/moffat-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/moffat-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/moffat-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/moffat-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/moffat-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/moffat-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/common-double.cl: * Common double-precision OpenCL routines for libprofit
src/cl/common-double.cl:#if __OPENCL_C_VERSION__ < 120
src/cl/common-double.cl:#pragma OPENCL EXTENSION cl_khr_fp64: enable
src/cl/common-float.cl: * Common single-precision OpenCL routines for libprofit
src/cl/sersic-float.cl: * Single-precision Sersic profile OpenCL kernel implementation for libprofit
src/cl/sersic-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/sersic-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/sersic-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/sersic-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/sersic-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/sersic-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/king-float.cl: * Single-precision King profile OpenCL kernel implementation for libprofit
src/cl/king-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/king-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/king-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/king-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/king-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/king-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/sersic-double.cl: * Double-precision Sersic profile OpenCL kernel implementation for libprofit
src/cl/sersic-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/sersic-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/sersic-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/sersic-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/sersic-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/sersic-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/ferrer-double.cl: * Double-precision Ferrer profile OpenCL kernel implementation for libprofit
src/cl/ferrer-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/ferrer-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/ferrer-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/ferrer-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/ferrer-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/ferrer-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/king-double.cl: * Double-precision King profile OpenCL kernel implementation for libprofit
src/cl/king-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/king-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/king-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/king-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/king-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/king-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/convolve-double.cl: * double 2D convolution OpenCL implementation for libprofit
src/cl/brokenexponential-float.cl: * Single-precision Broken Exponential profile OpenCL kernel implementation for libprofit
src/cl/brokenexponential-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/brokenexponential-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/brokenexponential-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/brokenexponential-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/brokenexponential-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/brokenexponential-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/coresersic-float.cl: * Single-precision Core-Sersic profile OpenCL kernel implementation for libprofit
src/cl/coresersic-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/coresersic-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/coresersic-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/coresersic-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/coresersic-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/coresersic-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/coresersic-double.cl: * Double-precision Core-Sersic profile OpenCL kernel implementation for libprofit
src/cl/coresersic-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/coresersic-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/coresersic-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/coresersic-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/cl/coresersic-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/cl/coresersic-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/sersic.cpp:#include "profit/opencl.h"
src/sersic.cpp:#ifdef PROFIT_OPENCL
src/sersic.cpp:#endif /* PROFIT_OPENCL */
src/model.cpp:	opencl_env(),
src/model.cpp:	opencl_env(),

```
