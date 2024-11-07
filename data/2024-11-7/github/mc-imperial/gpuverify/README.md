# https://github.com/mc-imperial/gpuverify

```console
getversion.py:""" This module is responsible for trying to determine the GPUVerify version"""
getversion.py:GPUVerifyDirectory = os.path.abspath( os.path.dirname(__file__))
getversion.py:GPUVerifyDeployVersionFile = os.path.join(GPUVerifyDirectory, '.gvdeployversion')
getversion.py:GPUVerifyRevisionErrorMessage = 'Error getting version information'
getversion.py:                                  repoTuple('vcgen', path=GPUVerifyDirectory),
getversion.py:                                  repoTuple('local-revision', path=GPUVerifyDirectory, getLocalRev=True) # GPUVerifyRise4Fun depends on this
getversion.py:    return GPUVerifyRevisionErrorMessage + " : " + str(e)
getversion.py:  If not it will look for a file GPUVerifyDeployVersionFile and if
getversion.py:  vs="GPUVerify:"
getversion.py:  gitPath = os.path.join(GPUVerifyDirectory, '.git')
getversion.py:    errorMessage = "Error Could not read version from file " + GPUVerifyDeployVersionFile + "\n"
getversion.py:    if os.path.isfile(GPUVerifyDeployVersionFile):
getversion.py:      with open(GPUVerifyDeployVersionFile,'r') as f:
gvtester.py:from GPUVerifyScript.error_codes import ErrorCodes
gvtester.py:GPUVerifyExecutable=sys.path[0] + os.sep + "GPUVerify.py"
gvtester.py:class GPUVerifyErrorCodes(ErrorCodes):
gvtester.py:        potential exit codes of GPUVerify.
gvtester.py:GPUVerifyErrorCodes.static_init()
gvtester.py:GPUVerifyTesterErrorCodes=enum('SUCCESS', 'FILE_SEARCH_ERROR','KERNEL_PARSE_ERROR', 'TEST_FAILED', 'FILE_OPEN_ERROR', 'GENERAL_ERROR')
gvtester.py:class GPUVerifyTestKernel(object):
gvtester.py:            Initialise CUDA/OpenCL GPUVerify test.
gvtester.py:            timeAsCSV           : Get CSV timing information from GPUVerify
gvtester.py:            additionalOptions   : A list of additional command line options to pass to GPUVerify
gvtester.py:            .expectedReturnCode : The expected return code of GPUVerify
gvtester.py:            .gpuverifyCmdArgs   : A list of command line arguments to pass to GPUVerify
gvtester.py:                    self.expectedReturnCode=GPUVerifyErrorCodes.SUCCESS
gvtester.py:                    if xfailCodeAsString in [ t[1] for t in GPUVerifyErrorCodes.getValidxfailCodes() ]:
gvtester.py:                        self.expectedReturnCode=getattr(GPUVerifyErrorCodes,xfailCodeAsString)
gvtester.py:                        raise KernelParseError(1,self.path, "\"" + xfailCodeAsString + "\" is not a valid error code for expected fail; valid codes are " + (", ".join([code[1] for code in GPUVerifyErrorCodes.getValidxfailCodes()])) + ".")
gvtester.py:            #Grab command line args to pass to GPUVerify
gvtester.py:                raise KernelParseError(2,self.path,"Second Line should start with \"//\" and then optionally space seperate arguments to pass to GPUVerify")
gvtester.py:            self.gpuverifyCmdArgs = cmdArgs[2:].strip().split() #Split on spaces
gvtester.py:            for index in range(0,len(self.gpuverifyCmdArgs)):
gvtester.py:                if self.gpuverifyCmdArgs[index].find('$') != -1:
gvtester.py:                    template=string.Template(self.gpuverifyCmdArgs[index])
gvtester.py:                    logging.debug('Performing command line argument substitution on:' + self.gpuverifyCmdArgs[index])
gvtester.py:                    self.gpuverifyCmdArgs[index]=template.substitute(cmdArgsSubstitution)
gvtester.py:                    logging.debug('Substitution complete, result:' + self.gpuverifyCmdArgs[index])
gvtester.py:            self.gpuverifyReturnCode=""
gvtester.py:              self.gpuverifyCmdArgs.append("--time-as-csv=" + self.path)
gvtester.py:            #Add additional GPUVerify command line args
gvtester.py:              self.gpuverifyCmdArgs.extend(additionalOptions)
gvtester.py:        """ Executes GPUVerify on this test's kernel
gvtester.py:            .gpuverifyReturnCode : GPUVerify's actual return code (doesn't include REGEX_MISMATCH_ERROR)
gvtester.py:        cmdLine=[sys.executable, GPUVerifyExecutable] \
gvtester.py:            + self.gpuverifyCmdArgs + [self.path]
gvtester.py:            logging.error("Received keyboard interrupt. Attempting to kill GPUVerify process")
gvtester.py:        #Record the true return code of GPUVerify
gvtester.py:          self.gpuverifyReturnCode=processInstance.returncode
gvtester.py:          logging.debug("GPUVerify return code:" + GPUVerifyErrorCodes.errorCodeToString[self.gpuverifyReturnCode])
gvtester.py:        if self.gpuverifyReturnCode == self.expectedReturnCode:
gvtester.py:            self.returnedCode=GPUVerifyErrorCodes.REGEX_MISMATCH_ERROR
gvtester.py:            logging.error(threadStr + self.path + " FAILED with " + GPUVerifyErrorCodes.errorCodeToString[self.returnedCode] +
gvtester.py:                         " expected " + GPUVerifyErrorCodes.errorCodeToString[self.expectedReturnCode])
gvtester.py:                         ("pass" if self.expectedReturnCode == GPUVerifyErrorCodes.SUCCESS else "xfail") + ")")
gvtester.py:        testString="GPUVerifyTestKernel:\nFull Path:{0}\nExpected exit code:{1}\nCmdArgs: {2}\n".format(
gvtester.py:              GPUVerifyErrorCodes.errorCodeToString[self.expectedReturnCode],
gvtester.py:              self.gpuverifyCmdArgs,
gvtester.py:          testString+= "Actual result:" + GPUVerifyErrorCodes.errorCodeToString[self.returnedCode] + "\n"
gvtester.py:          testString+= "GPUVerify return code:" + GPUVerifyErrorCodes.errorCodeToString[self.gpuverifyReturnCode] + "\n"
gvtester.py:class GPUVerifyTesterError(Exception):
gvtester.py:class KernelParseError(GPUVerifyTesterError):
gvtester.py:class CanonicalisationError(GPUVerifyTesterError):
gvtester.py:            for errorCode in GPUVerifyErrorCodes.getValidxfailCodes():
gvtester.py:            sys.exit(GPUVerifyTesterErrorCodes.SUCCESS)
gvtester.py:        sys.exit(GPUVerifyTesterErrorCodes.FILE_OPEN_ERROR)
gvtester.py:        sys.exit(GPUVerifyTesterErrorCodes.FILE_OPEN_ERROR)
gvtester.py:        sys.exit(GPUVerifyTesterErrorCodes.SUCCESS)
gvtester.py:                sys.exit(GPUVerifyTesterErrorCodes.FILE_OPEN_ERROR)
gvtester.py:            sys.exit(GPUVerifyTesterErrorCodes.GENERAL_ERROR)
gvtester.py:        if result in [-1, 0]: sys.exit(GPUVerifyTesterErrorCodes.SUCCESS)
gvtester.py:        Iterates through a list of GPUVerifyTestKernel objects and prints out a summary
gvtester.py:    OpenCLCounter=0
gvtester.py:    CUDACounter=0
gvtester.py:            OpenCLCounter += 1
gvtester.py:            CUDACounter += 1
gvtester.py:                if test.returnedCode == GPUVerifyErrorCodes.SUCCESS:
gvtester.py:    print('# OpenCL kernels:{0}'.format(OpenCLCounter))
gvtester.py:    print('# CUDA kernels:{0}'.format(CUDACounter))
gvtester.py:    self.cudaCount = 0
gvtester.py:    self.openCLCount = 0
gvtester.py:      sys.exit(GPUVerifyTesterErrorCodes.FILE_SEARCH_ERROR)
gvtester.py:      self.cudaCount+=1
gvtester.py:      self.openCLCount+=1
gvtester.py:            logging.debug("Found CUDA kernel:\"{}\"".format(f))
gvtester.py:            logging.debug("Found OpenCL kernel:\"{}\"".format(f))
gvtester.py:          sys.exit(GPUVerifyErrorCodes.CONFIGURATION_ERROR)
gvtester.py:    logging.info("Found    {0} OpenCL kernels, {1} CUDA kernels and {2} miscellaneous tests".format(counters.openCLCount, counters.cudaCount, counters.miscCount))
gvtester.py:    logging.info("Ignoring {0} OpenCL kernels, {1} CUDA kernels and {2} miscellaneous tests".format(ignoredCounters.openCLCount, ignoredCounters.cudaCount, ignoredCounters.miscCount))
gvtester.py:    logging.info("Running  {0} OpenCL kernels, {1} CUDA kernels and {2} miscellaneous tests".format(
gvtester.py:      counters.openCLCount - ignoredCounters.openCLCount,
gvtester.py:      counters.cudaCount - ignoredCounters.cudaCount,
gvtester.py:    logging.info("Found {0} OpenCL kernels, {1} CUDA kernels and {2} miscellaneous tests".format(counters.openCLCount, counters.cudaCount, counters.miscCount))
gvtester.py:  global GPUVerifyExecutable
gvtester.py:  parser = argparse.ArgumentParser(description='A simple script to run GPUVerify on CUDA/OpenCL kernels in its test suite.')
gvtester.py:                      help="Pass a command line options to GPUVerify for all tests. This option can be specified multiple times.  E.g. --gvopt=--keep-temps --gvopt=--no-benign",
gvtester.py:                      metavar='GPUVerifyCmdLineOption')
gvtester.py:  parser.add_argument("--force-gpuverify-script", type=str, default=None, help="Force a different GPUVerify script to be used")
gvtester.py:  if args.force_gpuverify_script != None:
gvtester.py:    GPUVerifyExecutable = args.force_gpuverify_script
gvtester.py:    logging.info('Forcing GPUVerify script to be {}'.format(GPUVerifyExecutable))
gvtester.py:    return GPUVerifyTesterErrorCodes.GENERAL_ERROR
gvtester.py:    return GPUVerifyTesterErrorCodes.GENERAL_ERROR
gvtester.py:    return GPUVerifyTesterErrorCodes.FILE_SEARCH_ERROR
gvtester.py:    return GPUVerifyTesterErrorCodes.FILE_SEARCH_ERROR
gvtester.py:  logging.info("Running {0} OpenCL kernels, {1} CUDA kernels and {2} miscellaneous tests".format(
gvtester.py:    counters.openCLCount,
gvtester.py:    counters.cudaCount,
gvtester.py:    logging.error("Could not find any OpenCL, CUDA kernels or miscellaneous tests")
gvtester.py:    return GPUVerifyTesterErrorCodes.FILE_SEARCH_ERROR
gvtester.py:      tests.append(GPUVerifyTestKernel(kernelPath, args.time_as_csv, csvFile, getattr(args,'gvopt=') ))
gvtester.py:        return GPUVerifyTesterErrorCodes.KERNEL_PARSE_ERROR
gvtester.py:    if args.run_only_pass and test.expectedReturnCode != GPUVerifyErrorCodes.SUCCESS :
gvtester.py:    if args.run_only_xfail and test.expectedReturnCode == GPUVerifyErrorCodes.SUCCESS :
gvtester.py:    sys.exit(GPUVerifyTesterErrorCodes.GENERAL_ERROR)
gvtester.py:  return GPUVerifyTesterErrorCodes.SUCCESS
GPUVerifyCruncher/BoogieInterpreter.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyCruncher/BoogieInterpreter.cs:namespace GPUVerify
GPUVerifyCruncher/BoogieInterpreter.cs:        // Local and global IDs of the 2 threads modelled in GPUverify
GPUVerifyCruncher/BoogieInterpreter.cs:        // The GPU configuration
GPUVerifyCruncher/BoogieInterpreter.cs:        private GPU gpu = new GPU();
GPUVerifyCruncher/BoogieInterpreter.cs:                    Print.DebugMessage(gpu.ToString(), 1);
GPUVerifyCruncher/BoogieInterpreter.cs:            Tuple<BitVector, BitVector> dimX = GetID(gpu.BlockDim[DIMENSION.X] - 1);
GPUVerifyCruncher/BoogieInterpreter.cs:            Tuple<BitVector, BitVector> dimY = GetID(gpu.BlockDim[DIMENSION.Y] - 1);
GPUVerifyCruncher/BoogieInterpreter.cs:            Tuple<BitVector, BitVector> dimZ = GetID(gpu.BlockDim[DIMENSION.Z] - 1);
GPUVerifyCruncher/BoogieInterpreter.cs:            Tuple<BitVector, BitVector> dimX = GetID(gpu.GridDim[DIMENSION.X] - 1);
GPUVerifyCruncher/BoogieInterpreter.cs:            Tuple<BitVector, BitVector> dimY = GetID(gpu.GridDim[DIMENSION.Y] - 1);
GPUVerifyCruncher/BoogieInterpreter.cs:            Tuple<BitVector, BitVector> dimZ = GetID(gpu.GridDim[DIMENSION.Z] - 1);
GPUVerifyCruncher/BoogieInterpreter.cs:                                    gpu.BlockDim[DIMENSION.X] = right.Evaluation.ConvertToInt32();
GPUVerifyCruncher/BoogieInterpreter.cs:                                    memory.Store(left.Symbol, new BitVector(gpu.BlockDim[DIMENSION.X]));
GPUVerifyCruncher/BoogieInterpreter.cs:                                    gpu.BlockDim[DIMENSION.Y] = right.Evaluation.ConvertToInt32();
GPUVerifyCruncher/BoogieInterpreter.cs:                                    memory.Store(left.Symbol, new BitVector(gpu.BlockDim[DIMENSION.Y]));
GPUVerifyCruncher/BoogieInterpreter.cs:                                    gpu.BlockDim[DIMENSION.Z] = right.Evaluation.ConvertToInt32();
GPUVerifyCruncher/BoogieInterpreter.cs:                                    memory.Store(left.Symbol, new BitVector(gpu.BlockDim[DIMENSION.Z]));
GPUVerifyCruncher/BoogieInterpreter.cs:                                    gpu.GridDim[DIMENSION.X] = right.Evaluation.ConvertToInt32();
GPUVerifyCruncher/BoogieInterpreter.cs:                                    memory.Store(left.Symbol, new BitVector(gpu.GridDim[DIMENSION.X]));
GPUVerifyCruncher/BoogieInterpreter.cs:                                    gpu.GridDim[DIMENSION.Y] = right.Evaluation.ConvertToInt32();
GPUVerifyCruncher/BoogieInterpreter.cs:                                    memory.Store(left.Symbol, new BitVector(gpu.GridDim[DIMENSION.Y]));
GPUVerifyCruncher/BoogieInterpreter.cs:                                    gpu.GridDim[DIMENSION.Z] = right.Evaluation.ConvertToInt32();
GPUVerifyCruncher/BoogieInterpreter.cs:                                    memory.Store(left.Symbol, new BitVector(gpu.GridDim[DIMENSION.Z]));
GPUVerifyCruncher/BoogieInterpreter.cs:                                    gpu.GridOffset[DIMENSION.X] = right.Evaluation.ConvertToInt32();
GPUVerifyCruncher/BoogieInterpreter.cs:                                    memory.Store(left.Symbol, new BitVector(gpu.GridOffset[DIMENSION.X]));
GPUVerifyCruncher/BoogieInterpreter.cs:                                    gpu.GridOffset[DIMENSION.Y] = right.Evaluation.ConvertToInt32();
GPUVerifyCruncher/BoogieInterpreter.cs:                                    memory.Store(left.Symbol, new BitVector(gpu.GridOffset[DIMENSION.Y]));
GPUVerifyCruncher/BoogieInterpreter.cs:                                    gpu.GridOffset[DIMENSION.Z] = right.Evaluation.ConvertToInt32();
GPUVerifyCruncher/BoogieInterpreter.cs:                                    memory.Store(left.Symbol, new BitVector(gpu.GridOffset[DIMENSION.Z]));
GPUVerifyCruncher/BoogieInterpreter.cs:                Houdini.RefutedAnnotation annotation = GPUVerify.Utilities.GetRefutedAnnotation(program, assertBoolean, impl.Name);
GPUVerifyCruncher/Main.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyCruncher/Memory.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyCruncher/Memory.cs:namespace GPUVerify
GPUVerifyCruncher/ExprTree.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyCruncher/ExprTree.cs:namespace GPUVerify
GPUVerifyCruncher/RefutationEngine.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyCruncher/RefutationEngine.cs:    using GPUVerify;
GPUVerifyCruncher/RefutationEngine.cs:                // This is an initial attempt at hooking GPUVerify up with Staged Houdini.
GPUVerifyCruncher/RefutationEngine.cs:            if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DebugConcurrentHoudini)
GPUVerifyCruncher/RefutationEngine.cs:            Pipeline pipeline = ((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).Pipeline;
GPUVerifyCruncher/RefutationEngine.cs:                    GPUVerify.Utilities.IO.EmitProgram(ApplyInvariants(outcome), GetFileNameBase(), "cbpl");
GPUVerifyCruncher/RefutationEngine.cs:            if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).WriteKilledInvariantsToFile)
GPUVerifyCruncher/RefutationEngine.cs:            if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ReplaceLoopInvariantAssertions)
GPUVerifyCruncher/GPU.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyCruncher/GPU.cs:namespace GPUVerify
GPUVerifyCruncher/GPU.cs:    public class GPU
GPUVerifyCruncher/GPU.cs:        public GPU()
GPUVerifyCruncher/Print.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyCruncher/Print.cs:namespace GPUVerify
GPUVerifyCruncher/GPUVerifyCruncherCommandLineOptions.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyCruncher/GPUVerifyCruncherCommandLineOptions.cs:namespace GPUVerify
GPUVerifyCruncher/GPUVerifyCruncherCommandLineOptions.cs:    public class GPUVerifyCruncherCommandLineOptions : GVCommandLineOptions
GPUVerifyCruncher/GPUVerifyCruncherCommandLineOptions.cs:        public GPUVerifyCruncherCommandLineOptions()
GPUVerifyCruncher/Properties/AssemblyInfo.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyCruncher/Properties/AssemblyInfo.cs:[assembly: AssemblyTitle("GPUVerifyCruncher")]
GPUVerifyCruncher/Properties/AssemblyInfo.cs:[assembly: AssemblyProduct("GPUVerifyCruncher")]
GPUVerifyCruncher/BitVector.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyCruncher/BitVector.cs:namespace GPUVerify
GPUVerifyCruncher/GPUVerifyCruncher.csproj:    <RootNamespace>GPUVerifyCruncher</RootNamespace>
GPUVerifyCruncher/GPUVerifyCruncher.csproj:    <AssemblyName>GPUVerifyCruncher</AssemblyName>
GPUVerifyCruncher/GPUVerifyCruncher.csproj:    <Compile Include="GPUVerifyCruncher.cs" />
GPUVerifyCruncher/GPUVerifyCruncher.csproj:    <Compile Include="GPUVerifyCruncherCommandLineOptions.cs" />
GPUVerifyCruncher/GPUVerifyCruncher.csproj:    <Compile Include="GPU.cs" />
GPUVerifyCruncher/GPUVerifyCruncher.csproj:    <ProjectReference Include="..\GPUVerifyLib\GPUVerifyLib.csproj">
GPUVerifyCruncher/GPUVerifyCruncher.csproj:      <Name>GPUVerifyLib</Name>
GPUVerifyCruncher/GPUVerifyCruncher.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyCruncher/GPUVerifyCruncher.cs:namespace GPUVerify
GPUVerifyCruncher/GPUVerifyCruncher.cs:    public class GPUVerifyCruncher
GPUVerifyCruncher/GPUVerifyCruncher.cs:            CommandLineOptions.Install(new GPUVerifyCruncherCommandLineOptions());
GPUVerifyCruncher/GPUVerifyCruncher.cs:                ((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ParsePipelineString();
GPUVerifyCruncher/GPUVerifyCruncher.cs:                    Utilities.IO.ErrorWriteLine("GPUVerify: error: no input files were specified");
GPUVerifyCruncher/GPUVerifyCruncher.cs:                        Utilities.IO.ErrorWriteLine("GPUVerify: error: {0} is not a .bpl file", file);
GPUVerifyCruncher/GPUVerifyCruncher.cs:                if (GetCommandLineOptions().DebugGPUVerify)
GPUVerifyCruncher/GPUVerifyCruncher.cs:                    Console.Error.WriteLine("Exception thrown in GPUVerifyCruncher");
GPUVerifyCruncher/GPUVerifyCruncher.cs:        private static GPUVerifyCruncherCommandLineOptions GetCommandLineOptions()
GPUVerifyCruncher/GPUVerifyCruncher.cs:            return (GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo;
testsuite/CUDA/argument_promotion/kernel.cu:texture<unsigned, 1, cudaReadModeElementType> texDKey128;
testsuite/CUDA/annotation_tests/test_distinct/kernel.cu:#include <cuda.h>
testsuite/CUDA/annotation_tests/test_while_loop_invariant/kernel.cu:#include <cuda.h>
testsuite/CUDA/annotation_tests/test_assert/kernel.cu:#include <cuda.h>
testsuite/CUDA/annotation_tests/test_norace/kernel.cu:#include <cuda.h>
testsuite/CUDA/annotation_tests/test_contract/kernel.cu:#include <cuda.h>
testsuite/CUDA/annotation_tests/test_axiom/kernel.cu:#include <cuda.h>
testsuite/CUDA/annotation_tests/test_assume/kernel.cu:#include <cuda.h>
testsuite/CUDA/annotation_tests/test_for_loop_invariant/kernel.cu:#include <cuda.h>
testsuite/CUDA/annotation_tests/test_no_readwrite/kernel.cu:#include "cuda.h"
testsuite/CUDA/annotation_tests/test_all/kernel.cu:#include <cuda.h>
testsuite/CUDA/annotation_tests/test_at_most_one/kernel.cu:#include <cuda.h>
testsuite/CUDA/annotation_tests/test_ensures/kernel.cu:#include <cuda.h>
testsuite/CUDA/annotation_tests/test_requires/kernel.cu:#include <cuda.h>
testsuite/CUDA/annotation_tests/test_enabled_and_uniform/kernel.cu:#include <cuda.h>
testsuite/CUDA/cuda_arch/kernel.cu:#if __CUDA_ARCH__ < 350
testsuite/CUDA/cuda_arch/kernel.cu:#error Unexpected __CUDA_ARCH__
testsuite/CUDA/basicglobalarray/kernel.cu:#include "cuda.h"
testsuite/CUDA/curand_tests/pass/curand/kernel.cu:#include <cuda.h>
testsuite/CUDA/curand_tests/pass/curand_mtgp32/kernel.cu:#include <cuda.h>
testsuite/CUDA/curand_tests/fail/curand_mtgp32_block_race/kernel.cu:#include <cuda.h>
testsuite/CUDA/curand_tests/fail/init_race/kernel.cu:#include <cuda.h>
testsuite/CUDA/curand_tests/fail/curand_race/kernel.cu:#include <cuda.h>
testsuite/CUDA/curand_tests/fail/curand_mtgp32_race/kernel.cu:#include <cuda.h>
testsuite/CUDA/ternarytest2/kernel.cu:#include "cuda.h"
testsuite/CUDA/pointeranalysistests/testinterprocedural2/kernel.cu:#include "cuda.h"
testsuite/CUDA/pointeranalysistests/testinterprocedural3/kernel.cu:#include "cuda.h"
testsuite/CUDA/pointeranalysistests/testinterprocedural/kernel.cu:#include "cuda.h"
testsuite/CUDA/pointeranalysistests/testbasicaliasing/kernel.cu:#include "cuda.h"
testsuite/CUDA/noraceduetoreturn/kernel.cu:#include "cuda.h"
testsuite/CUDA/ternarytest/kernel.cu:#include "cuda.h"
testsuite/CUDA/simpleparampassing/kernel.cu:#include "cuda.h"
testsuite/CUDA/nestedinline/kernel.cu:#include <cuda.h>
testsuite/CUDA/reduced_strength_blockwise/kernel.cu:#include <cuda.h>
testsuite/CUDA/align/kernel.cu:#include <cuda.h>
testsuite/CUDA/cooperative_groups/pass/multiple_barriers/kernel.cu:#include <cuda.h>
testsuite/CUDA/cooperative_groups/pass/grid_barrier/kernel.cu:#include <cuda.h>
testsuite/CUDA/cooperative_groups/pass/block_barrier/kernel.cu:#include <cuda.h>
testsuite/CUDA/cooperative_groups/fail/divergence_grid_group/kernel.cu:#include <cuda.h>
testsuite/CUDA/cooperative_groups/fail/divergence_thread_block/kernel.cu:#include <cuda.h>
testsuite/CUDA/cooperative_groups/fail/race/kernel.cu:#include <cuda.h>
testsuite/CUDA/cooperative_groups/fail/block_divergence_grid_group/kernel.cu:#include <cuda.h>
testsuite/CUDA/multiplelocals/kernel.cu:#include "cuda.h"
testsuite/CUDA/ctimesgid/kernel.cu: * Test that GPUVerify generates a CTimesGid invariant.
testsuite/CUDA/mul24/kernel.cu:#include <cuda.h>
testsuite/CUDA/barrierconditionalkernelparam/kernel.cu:#include "cuda.h"
testsuite/CUDA/transitiveclosure/kernel.cu:#include <cuda.h>
testsuite/CUDA/memcpy_simplification/global_passed_to_call/kernel.cu:texture<unsigned, 1, cudaReadModeElementType> texDKey128;
testsuite/CUDA/globalarray/fail/kernel.cu:// The statically given values for A are not preserved when we translate CUDA
testsuite/CUDA/globalarray/fail/kernel.cu:// cf. testsuite/OpenCL/globalarray/pass2
testsuite/CUDA/ctimeslid/kernel.cu: * Test that GPUVerify generates a CTimesLid invariant.
testsuite/CUDA/pointertests/test4/kernel.cu:#include "cuda.h"
testsuite/CUDA/pointertests/test11/kernel.cu:#include "cuda.h"
testsuite/CUDA/pointertests/test13/kernel.cu:#include "cuda.h"
testsuite/CUDA/pointertests/test10/kernel.cu:#include "cuda.h"
testsuite/CUDA/pointertests/test_copy_between_pointers/kernel.cu:#include "cuda.h"
testsuite/CUDA/pointertests/test2/kernel.cu:#include "cuda.h"
testsuite/CUDA/pointertests/test12/kernel.cu:#include "cuda.h"
testsuite/CUDA/pointertests/test14/kernel.cu:#include "cuda.h"
testsuite/CUDA/pointertests/test_copy_between_memory_spaces/kernel.cu:#include "cuda.h"
testsuite/CUDA/pointertests/test9/kernel.cu:#include "cuda.h"
testsuite/CUDA/pointertests/test6/kernel.cu:#include "cuda.h"
testsuite/CUDA/pointertests/test8/kernel.cu:#include "cuda.h"
testsuite/CUDA/pointertests/test3/kernel.cu:#include "cuda.h"
testsuite/CUDA/pointertests/test_bad_pointer_procedure_call/kernel.cu:#include "cuda.h"
testsuite/CUDA/pointertests/test7/kernel.cu:#include "cuda.h"
testsuite/CUDA/pointertests/scanlargelike/kernel.cu:#include <cuda.h>
testsuite/CUDA/pointertests/cast/kernel.cu:#include "cuda.h"
testsuite/CUDA/pointertests/test1/kernel.cu:#include "cuda.h"
testsuite/CUDA/pointertests/test_pass_value_from_array/kernel.cu:#include "cuda.h"
testsuite/CUDA/pointertests/test5/kernel.cu:#include "cuda.h"
testsuite/CUDA/basicbarrier/kernel.cu:#include "cuda.h"
testsuite/CUDA/nonpointerparameter2/kernel.cu:#include "cuda.h"
testsuite/CUDA/struct/kernel.cu:#include <cuda.h>
testsuite/CUDA/casttofloat/kernel.cu:#include "cuda.h"
testsuite/CUDA/atomics/add_one/kernel.cu:#include <cuda.h>
testsuite/CUDA/atomics/add_tid/kernel.cu:#include <cuda.h>
testsuite/CUDA/atomics/add_zero/kernel.cu:#include <cuda.h>
testsuite/CUDA/atomics/definitions/kernel.cu:#include <cuda.h>
testsuite/CUDA/simplereturn/kernel.cu:#include "cuda.h"
testsuite/CUDA/always_inline/kernel.cu:#include <cuda.h>
testsuite/CUDA/warpsync/scan_warp/kernel.cu:#include <cuda.h>
testsuite/CUDA/warpsync/2d/kernel.cu:#include <cuda.h>
testsuite/CUDA/warpsync/shuffle/kernel.cu:#include <cuda.h>
testsuite/CUDA/warpsync/broken_shuffle/kernel.cu:#include <cuda.h>
testsuite/CUDA/warpsync/intragroup_scan/kernel.cu:#include <cuda.h>
testsuite/CUDA/scope/kernel.cu:#include <cuda.h>
testsuite/CUDA/local2darrayaccess/kernel.cu:#include "cuda.h"
testsuite/CUDA/floatcastrequired/kernel.cu:#include "cuda.h"
testsuite/CUDA/localarrayaccess/kernel.cu:#include "cuda.h"
testsuite/CUDA/nonpointerparameter1/kernel.cu:#include "cuda.h"
testsuite/CUDA/misc/pass/misc1/kernel.cu:#include <cuda.h>
testsuite/CUDA/misc/pass/misc1/kernel.cu:__global__ void helloCUDA(int x)
testsuite/CUDA/misc/pass/misc2/kernel.cu:#include <cuda.h>
testsuite/CUDA/misc/pass/misc2/kernel.cu:__global__ void helloCUDA(volatile int* p)
testsuite/CUDA/misc/fail/miscfail1/kernel.cu:#include <cuda.h>
testsuite/CUDA/misc/fail/miscfail3/kernel.cu://GPUVerify kernel analyser finished with 1 verified, 1 error
testsuite/CUDA/misc/fail/miscfail3/kernel.cu:// In CUDA providing the inline keyword should still keep a copy of
testsuite/CUDA/misc/fail/miscfail3/kernel.cu:// the function around (contrary to OpenCL). However, by default a
testsuite/CUDA/misc/fail/miscfail3/kernel.cu:// level used by GPUVerify.
testsuite/CUDA/misc/fail/miscfail4/kernel.cu://GPUVerify kernel analyser finished with 0 verified, 1 error
testsuite/CUDA/misc/fail/miscfail4/kernel.cu:// In CUDA providing static and __attribute__((always_inline)) should not
testsuite/CUDA/misc/fail/miscfail2/kernel.cu:#include <cuda.h>
testsuite/CUDA/misc/fail/miscfail2/kernel.cu:__global__ void helloCUDA(
testsuite/CUDA/floatrelationalop/kernel.cu:#include "cuda.h"
testsuite/CUDA/unusedreturn/kernel.cu:#include "cuda.h"
testsuite/CUDA/basic1/kernel.cu:#include "cuda.h"
testsuite/CUDA/notunaryoptest/kernel.cu:#include "cuda.h"
testsuite/CUDA/reduced_strength_with_requires/kernel.cu:#include <cuda.h>
testsuite/CUDA/fail_tests/shared_int/kernel.cu:#include "cuda.h"
testsuite/CUDA/fail_tests/race_on_shared/kernel.cu:#include <cuda.h>
testsuite/CUDA/test_for_get_group_id/kernel.cu:#include "cuda.h"
testsuite/CUDA/loop_unwind/kernel.cu:#include <cuda.h>
testsuite/CUDA/loop_unwind/kernel.cu:__global__ void helloCUDA(float *A)
testsuite/misc/version_info/version_info.misc:* This test checks that GPUVerify's
testsuite/misc/local_revision/local_revision.misc:* GPUVerifyRise4Fun depends on this functionality.
testsuite/OpenCL/annotation_tests/invariant_specification_mistakes/short_circuit_or/kernel.cl://xfail:GPUVERIFYVCGEN_ERROR
testsuite/OpenCL/annotation_tests/invariant_specification_mistakes/short_circuit_ternary/kernel.cl://xfail:GPUVERIFYVCGEN_ERROR
testsuite/OpenCL/annotation_tests/invariant_specification_mistakes/short_circuit_and/kernel.cl://xfail:GPUVERIFYVCGEN_ERROR
testsuite/OpenCL/annotation_tests/invariant_specification_mistakes/nowhere_near_loop_head/kernel.cl://xfail:GPUVERIFYVCGEN_ERROR
testsuite/OpenCL/transitiveclosuresimplified/kernel.cl:    // TODO: check that in OpenCL the order is 0=x, 1=y, 2=z (in AMP it is reversed)
testsuite/OpenCL/atomics/definitions_long/kernel.cl:#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
testsuite/OpenCL/atomics/definitions_long/kernel.cl:#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
testsuite/OpenCL/atomics/equality_fail/kernel.cl://xfail:GPUVERIFYVCGEN_ERROR
testsuite/OpenCL/atomics/equality_fail/kernel.cl://GPUVerify: error: --equality-abstraction cannot be used with atomics\.
testsuite/OpenCL/atomics/forloop/kernel.cl:// This is to test whether GPUVerify can correctly report the relevant atomic line
testsuite/OpenCL/misc/pass/misc16/kernel.cl:#pragma OPENCL EXTENSION cl_khr_fp64 : enable
testsuite/OpenCL/misc/pass/misc7/kernel.cl:#pragma OPENCL EXTENSION cl_khr_fp64 : enable
testsuite/OpenCL/misc/fail/miscfail6/kernel.cl://GPUVerify kernel analyser finished with 0 verified, 1 error
testsuite/OpenCL/vectortests/double4simpleaccess/kernel.cl:#pragma OPENCL EXTENSION cl_khr_fp64: enable
testsuite/OpenCL/vectortests/double2simpleaccess/kernel.cl:#pragma OPENCL EXTENSION cl_khr_fp64: enable
testsuite/OpenCL/vectortests/double8simpleaccess/kernel.cl:#pragma OPENCL EXTENSION cl_khr_fp64: enable
testsuite/OpenCL/vectortests/double2arithmetic/kernel.cl:#pragma OPENCL EXTENSION cl_khr_fp64: enable
testsuite/OpenCL/checkarrays/fail/arraydoesnotexist2/kernel.cl://xfail:GPUVERIFYVCGEN_ERROR
testsuite/OpenCL/checkarrays/fail/arraydoesnotexist1/kernel.cl://xfail:GPUVERIFYVCGEN_ERROR
GPUVerifyScript/argument_parser.py:"""Module for parsing GPUVerify command line arguments"""
GPUVerifyScript/argument_parser.py:    return "GPUVerify: COMMAND_LINE_ERROR error ({}): {}" \
GPUVerifyScript/argument_parser.py:  parser = __ArgumentParser(description = "GPUVerify frontend",
GPUVerifyScript/argument_parser.py:    usage = "gpuverify [options] <kernel>")
GPUVerifyScript/argument_parser.py:  language.add_argument("--opencl", dest = 'source_language',
GPUVerifyScript/argument_parser.py:    action = 'store_const', const = SourceLanguage.OpenCL,
GPUVerifyScript/argument_parser.py:    help = "Assume the kernel is an OpenCL kernel")
GPUVerifyScript/argument_parser.py:  language.add_argument("--cuda", dest = 'source_language',
GPUVerifyScript/argument_parser.py:    action = 'store_const', const = SourceLanguage.CUDA,
GPUVerifyScript/argument_parser.py:    help = "Assume the kernel is a CUDA kernel")
GPUVerifyScript/argument_parser.py:    help = "Specify the dimensions of an OpenCL work-group. This corresponds \
GPUVerifyScript/argument_parser.py:    help = "Specify dimensions of the OpenCL NDRange. This corresponds to the \
GPUVerifyScript/argument_parser.py:    help = "Specify the dimensions of a grid of OpenCL work-groups. Mutually \
GPUVerifyScript/argument_parser.py:    type = __offsets, help = "Specify the OpenCL global offset. This \
GPUVerifyScript/argument_parser.py:    help = "Specify the CUDA thread block size")
GPUVerifyScript/argument_parser.py:    help = "Specify the CUDA grid size")
GPUVerifyScript/argument_parser.py:    help = "Enable debugging of GPUVerify components: exceptions will not be \
GPUVerifyScript/argument_parser.py:    return SourceLanguage.OpenCL
GPUVerifyScript/argument_parser.py:    return SourceLanguage.CUDA
GPUVerifyScript/argument_parser.py:    return SourceLanguage.OpenCL
GPUVerifyScript/argument_parser.py:    return SourceLanguage.CUDA
GPUVerifyScript/argument_parser.py:    if args.source_language == SourceLanguage.OpenCL:
GPUVerifyScript/argument_parser.py:    elif args.source_language == SourceLanguage.CUDA:
GPUVerifyScript/argument_parser.py:  elif args.source_language == SourceLanguage.CUDA:
GPUVerifyScript/argument_parser.py:    parser.error("Cannot specify --global_offset for CUDA kernels")
GPUVerifyScript/json_loader.py:"""Module for loading JSON files with GPUVerify invocation data."""
GPUVerifyScript/json_loader.py:    return "GPUVerify: JSON_ERROR error ({}): {}" \
GPUVerifyScript/json_loader.py:def __process_opencl_entry(data, strict):
GPUVerifyScript/json_loader.py:  # Allow for future extension to CUDA
GPUVerifyScript/json_loader.py:  if data["language"] == "OpenCL":
GPUVerifyScript/json_loader.py:    __process_opencl_entry(data, strict)
GPUVerifyScript/json_loader.py:    raise JSONError("'language' value needs to be 'OpenCL'")
GPUVerifyScript/json_loader.py:  """Load GPUVerify invocation data from json_file object.
GPUVerifyScript/error_codes.py:"""Module defining the error codes used by GPUVerify and gvtester."""
GPUVerifyScript/error_codes.py:  GPUVERIFYVCGEN_ERROR = 5
GPUVerifyScript/constants.py:  OpenCL = 1
GPUVerifyScript/constants.py:  CUDA = 2
gvfindtools.templates/gvfindtools.dev.py:""" This module defines the paths that GPUVerify will use
gvfindtools.templates/gvfindtools.dev.py:    to run the various tools that GPUVerify Depends on.
gvfindtools.templates/gvfindtools.dev.py:# ENVIRONMENT. THEN COPY THIS FILE INTO THE ROOT GPUVERIFY DIRECTORY (where
gvfindtools.templates/gvfindtools.dev.py:# GPUVerify.py lives) AND RENAME IT TO "gvfindtools.py". "gvfindtools.py" WILL
gvfindtools.templates/gvfindtools.dev.py:# rootDir = r"c:\projects\gpuverify"
gvfindtools.templates/gvfindtools.dev.py:rootDir = "/home/dan/documents/projects/gpuverify"
gvfindtools.templates/gvfindtools.dev.py:# The path to the directory containing the GPUVerify binaries.
gvfindtools.templates/gvfindtools.dev.py:# GPUVerifyVCGen.exe, GPUVerifyCruncher.exe and GPUVerifyBoogieDriver.exe should be there
gvfindtools.templates/gvfindtools.dev.py:gpuVerifyBinDir = rootDir + "/gpuverify/Binaries"
gvfindtools.templates/gvfindtoolsdeploy.py:""" This module defines the paths that GPUVerify will use
gvfindtools.templates/gvfindtoolsdeploy.py:    to run the various tools that GPUVerify Depends on.
gvfindtools.templates/gvfindtoolsdeploy.py:gpuVerifyBinDir = None
gvfindtools.templates/gvfindtoolsdeploy.py:  global gpuVerifyBinDir
gvfindtools.templates/gvfindtoolsdeploy.py:  # The path to the directory containing the GPUVerify binaries.
gvfindtools.templates/gvfindtoolsdeploy.py:  # GPUVerifyVCGen.exe, GPUVerifyCruncher.exe and GPUVerifyBoogieDriver.exe should be there
gvfindtools.templates/gvfindtoolsdeploy.py:  gpuVerifyBinDir = pathPrefix + os.sep + "bin"
GPUVerifyBoogieDriver/GPUVerifyBoogieDriver.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyBoogieDriver/GPUVerifyBoogieDriver.cs:    using GPUVerify;
GPUVerifyBoogieDriver/GPUVerifyBoogieDriver.cs:    using ResultCounter = GPUVerify.KernelAnalyser.ResultCounter;
GPUVerifyBoogieDriver/GPUVerifyBoogieDriver.cs:    public class GPUVerifyBoogieDriver
GPUVerifyBoogieDriver/GPUVerifyBoogieDriver.cs:                    Utilities.IO.ErrorWriteLine("GPUVerify: error: no input files were specified");
GPUVerifyBoogieDriver/GPUVerifyBoogieDriver.cs:                        Utilities.IO.ErrorWriteLine("GPUVerify: error: {0} is not a .(c)bpl file", file);
GPUVerifyBoogieDriver/GPUVerifyBoogieDriver.cs:                if (GetCommandLineOptions().DebugGPUVerify)
GPUVerifyBoogieDriver/GPUVerifyBoogieDriver.cs:                    Console.Error.WriteLine("Exception thrown in GPUVerifyBoogieDriver");
GPUVerifyBoogieDriver/Properties/AssemblyInfo.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyBoogieDriver/Properties/AssemblyInfo.cs:[assembly: AssemblyTitle("GPUVerifyBoogieDriver")]
GPUVerifyBoogieDriver/Properties/AssemblyInfo.cs:[assembly: AssemblyProduct("GPUVerifyBoogieDriver")]
GPUVerifyBoogieDriver/GPUVerifyBoogieDriver.csproj:    <RootNamespace>GPUVerifyBoogieDriver</RootNamespace>
GPUVerifyBoogieDriver/GPUVerifyBoogieDriver.csproj:    <AssemblyName>GPUVerifyBoogieDriver</AssemblyName>
GPUVerifyBoogieDriver/GPUVerifyBoogieDriver.csproj:    <Compile Include="GPUVerifyBoogieDriver.cs" />
GPUVerifyBoogieDriver/GPUVerifyBoogieDriver.csproj:    <ProjectReference Include="..\GPUVerifyLib\GPUVerifyLib.csproj">
GPUVerifyBoogieDriver/GPUVerifyBoogieDriver.csproj:      <Name>GPUVerifyLib</Name>
README.md:# GPUVerify
README.md:![Build Status](https://github.com/mc-imperial/gpuverify/actions/workflows/build-and-test.yml/badge.svg)
README.md:GPUVerify is a static analyser for verifying race- and divergence-freedom of
README.md:GPU kernels written in OpenCL and CUDA.
README.md:[online](http://multicore.doc.ic.ac.uk/tools/GPUVerify/docs/)
GPUVerifyVCGen/UniformityAnalyser.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/UniformityAnalyser.cs:namespace GPUVerify
GPUVerifyVCGen/VariableDefinitionAnalysisRegion.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/VariableDefinitionAnalysisRegion.cs:namespace GPUVerify
GPUVerifyVCGen/VariableDefinitionAnalysisRegion.cs:        private GPUVerifier verifier;
GPUVerifyVCGen/VariableDefinitionAnalysisRegion.cs:        private VariableDefinitionAnalysisRegion(GPUVerifier v)
GPUVerifyVCGen/VariableDefinitionAnalysisRegion.cs:            private GPUVerifier verifier;
GPUVerifyVCGen/VariableDefinitionAnalysisRegion.cs:            public SubstitutionDuplicator(Dictionary<string, Expr> d, GPUVerifier v, string p)
GPUVerifyVCGen/VariableDefinitionAnalysisRegion.cs:        public static VariableDefinitionAnalysisRegion Analyse(Implementation impl, GPUVerifier verifier)
GPUVerifyVCGen/SmartBlockPredicator.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/SmartBlockPredicator.cs:namespace GPUVerify
GPUVerifyVCGen/CallSiteAnalyser.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/CallSiteAnalyser.cs:namespace GPUVerify
GPUVerifyVCGen/CallSiteAnalyser.cs:        private GPUVerifier verifier;
GPUVerifyVCGen/CallSiteAnalyser.cs:        public CallSiteAnalyser(GPUVerifier verifier)
GPUVerifyVCGen/RelationalPowerOfTwoAnalyser.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/RelationalPowerOfTwoAnalyser.cs:namespace GPUVerify
GPUVerifyVCGen/RelationalPowerOfTwoAnalyser.cs:        private GPUVerifier verifier;
GPUVerifyVCGen/RelationalPowerOfTwoAnalyser.cs:        public RelationalPowerOfTwoAnalyser(GPUVerifier verifier)
GPUVerifyVCGen/RelationalPowerOfTwoAnalyser.cs:            if (GPUVerifyVCGenCommandLineOptions.ShowMayBePowerOfTwoAnalysis)
GPUVerifyVCGen/AccessCollector.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/AccessCollector.cs:namespace GPUVerify
GPUVerifyVCGen/IKernelArrayInfo.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/IKernelArrayInfo.cs:namespace GPUVerify
GPUVerifyVCGen/LoopInvariantGenerator.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/LoopInvariantGenerator.cs:namespace GPUVerify
GPUVerifyVCGen/LoopInvariantGenerator.cs:        private GPUVerifier verifier;
GPUVerifyVCGen/LoopInvariantGenerator.cs:        private LoopInvariantGenerator(GPUVerifier verifier, Implementation impl)
GPUVerifyVCGen/LoopInvariantGenerator.cs:        public static void EstablishDisabledLoops(GPUVerifier verifier, Implementation impl)
GPUVerifyVCGen/LoopInvariantGenerator.cs:        public static void PreInstrument(GPUVerifier verifier, Implementation impl)
GPUVerifyVCGen/LoopInvariantGenerator.cs:        private static void GenerateCandidateForEnablednessWhenAccessingSharedArrays(GPUVerifier verifier, Implementation impl, IRegion region)
GPUVerifyVCGen/LoopInvariantGenerator.cs:        private static void GenerateCandidateForEnabledness(GPUVerifier verifier, Implementation impl, IRegion region)
GPUVerifyVCGen/LoopInvariantGenerator.cs:        private static Expr MaybeExtractGuard(GPUVerifier verifier, Implementation impl, Block b)
GPUVerifyVCGen/LoopInvariantGenerator.cs:        private static void GenerateCandidateForNonUniformGuardVariables(GPUVerifier verifier, Implementation impl, IRegion region)
GPUVerifyVCGen/LoopInvariantGenerator.cs:            if (!verifier.ContainsBarrierCall(region) && !GPUVerifyVCGenCommandLineOptions.WarpSync)
GPUVerifyVCGen/LoopInvariantGenerator.cs:        private static void GenerateCandidateForNonNegativeGuardVariables(GPUVerifier verifier, Implementation impl, IRegion region)
GPUVerifyVCGen/LoopInvariantGenerator.cs:        private static void GenerateCandidateForReducedStrengthStrideVariables(GPUVerifier verifier, Implementation impl, IRegion region)
GPUVerifyVCGen/LoopInvariantGenerator.cs:        private static void GenerateCandidateForLoopBounds(GPUVerifier verifier, Implementation impl, IRegion region)
GPUVerifyVCGen/LoopInvariantGenerator.cs:        public static void PostInstrument(GPUVerifier verifier, Implementation impl)
GPUVerifyVCGen/LoopInvariantGenerator.cs:            if (!verifier.ContainsBarrierCall(region) && !GPUVerifyVCGenCommandLineOptions.WarpSync)
GPUVerifyVCGen/LoopInvariantGenerator.cs:                    if (GPUVerifier.IsPredicate(lv))
GPUVerifyVCGen/LoopInvariantGenerator.cs:            return GPUVerifier.IsPredicate(Utilities.StripThreadIdentifier(((IdentifierExpr)nary.Args[0]).Name))
GPUVerifyVCGen/LoopInvariantGenerator.cs:                && GPUVerifier.IsPredicate(Utilities.StripThreadIdentifier(((IdentifierExpr)nary.Args[1]).Name));
GPUVerifyVCGen/LoopInvariantGenerator.cs:        private static bool AccessesGlobalArrayOrUnsafeBarrier(Cmd c, GPUVerifier verifier)
GPUVerifyVCGen/LoopInvariantGenerator.cs:                if (GPUVerifier.IsBarrier(call.Proc)
GPUVerifyVCGen/LoopInvariantGenerator.cs:        private static bool AccessesGlobalArrayOrUnsafeBarrier(IRegion region, GPUVerifier verifier)
GPUVerifyVCGen/AsymmetricExpressionFinder.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/AsymmetricExpressionFinder.cs:namespace GPUVerify
GPUVerifyVCGen/StrideConstraint.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/StrideConstraint.cs:namespace GPUVerify
GPUVerifyVCGen/StrideConstraint.cs:        public static StrideConstraint Bottom(GPUVerifier verifier, Expr e)
GPUVerifyVCGen/StrideConstraint.cs:        public Expr MaybeBuildPredicate(GPUVerifier verifier, Expr e)
GPUVerifyVCGen/StrideConstraint.cs:        private static StrideConstraint BuildAddStrideConstraint(GPUVerifier verifier, Expr e, StrideConstraint lhsc, StrideConstraint rhsc)
GPUVerifyVCGen/StrideConstraint.cs:        private static StrideConstraint BuildMulStrideConstraint(GPUVerifier verifier, Expr e, StrideConstraint lhsc, StrideConstraint rhsc)
GPUVerifyVCGen/StrideConstraint.cs:        public static StrideConstraint FromExpr(GPUVerifier verifier, Implementation impl, Expr e)
GPUVerifyVCGen/StrideConstraint.cs:                if (GPUVerifier.IsConstantInCurrentRegion(ie))
GPUVerifyVCGen/ExpressionSimplifier.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/ExpressionSimplifier.cs:namespace GPUVerify
GPUVerifyVCGen/UnaryBarrierInvariantDescriptor.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/UnaryBarrierInvariantDescriptor.cs:namespace GPUVerify
GPUVerifyVCGen/UnaryBarrierInvariantDescriptor.cs:            Expr predicate, Expr barrierInvariant, QKeyValue sourceLocationInfo, KernelDualiser dualiser, string procName, GPUVerifier verifier)
GPUVerifyVCGen/UnaryBarrierInvariantDescriptor.cs:            private GPUVerifier verifier;
GPUVerifyVCGen/UnaryBarrierInvariantDescriptor.cs:                Expr instantiationExpr, int thread, GPUVerifier verifier, string procName)
GPUVerifyVCGen/BarrierInvariantDescriptor.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/BarrierInvariantDescriptor.cs:namespace GPUVerify
GPUVerifyVCGen/BarrierInvariantDescriptor.cs:        protected GPUVerifier Verifier { get; private set; }
GPUVerifyVCGen/BarrierInvariantDescriptor.cs:            Expr predicate, Expr barrierInvariant, QKeyValue sourceLocationInfo, KernelDualiser dualiser, string procName, GPUVerifier verifier)
GPUVerifyVCGen/BarrierInvariantDescriptor.cs:            if (GPUVerifyVCGenCommandLineOptions.BarrierAccessChecks)
GPUVerifyVCGen/UninterpretedFunctionRemover.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/UninterpretedFunctionRemover.cs:namespace GPUVerify
GPUVerifyVCGen/IRaceInstrumenter.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/IRaceInstrumenter.cs:namespace GPUVerify
GPUVerifyVCGen/ReadCollector.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/ReadCollector.cs:namespace GPUVerify
GPUVerifyVCGen/InvariantGenerationRules/InvariantGenerationRule.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/InvariantGenerationRules/InvariantGenerationRule.cs:namespace GPUVerify.InvariantGenerationRules
GPUVerifyVCGen/InvariantGenerationRules/InvariantGenerationRule.cs:        protected GPUVerifier Verifier { get; }
GPUVerifyVCGen/InvariantGenerationRules/InvariantGenerationRule.cs:        public InvariantGenerationRule(GPUVerifier verifier)
GPUVerifyVCGen/InvariantGenerationRules/PowerOfTwoInvariantGenerator.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/InvariantGenerationRules/PowerOfTwoInvariantGenerator.cs:namespace GPUVerify.InvariantGenerationRules
GPUVerifyVCGen/InvariantGenerationRules/PowerOfTwoInvariantGenerator.cs:        public PowerOfTwoInvariantGenerator(GPUVerifier verifier)
GPUVerifyVCGen/IRegion.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/IRegion.cs:namespace GPUVerify
GPUVerifyVCGen/IntegerRepresentation.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/IntegerRepresentation.cs:namespace GPUVerify
GPUVerifyVCGen/IntegerRepresentation.cs:        private GPUVerifier verifier;
GPUVerifyVCGen/IntegerRepresentation.cs:        public BVIntegerRepresentation(GPUVerifier verifier)
GPUVerifyVCGen/IntegerRepresentation.cs:        private GPUVerifier verifier;
GPUVerifyVCGen/IntegerRepresentation.cs:        public MathIntegerRepresentation(GPUVerifier verifier)
GPUVerifyVCGen/WatchdogRaceInstrumenter.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/WatchdogRaceInstrumenter.cs:namespace GPUVerify
GPUVerifyVCGen/WatchdogRaceInstrumenter.cs:        public WatchdogRaceInstrumenter(GPUVerifier verifier)
GPUVerifyVCGen/WatchdogRaceInstrumenter.cs:            Variable accessHasOccurredVariable = GPUVerifier.MakeAccessHasOccurredVariable(v.Name, access);
GPUVerifyVCGen/WatchdogRaceInstrumenter.cs:            Variable accessBenignFlagVariable = GPUVerifier.MakeBenignFlagVariable(v.Name);
GPUVerifyVCGen/WatchdogRaceInstrumenter.cs:            if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access.IsReadOrWrite())
GPUVerifyVCGen/WatchdogRaceInstrumenter.cs:            if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access == AccessType.WRITE)
GPUVerifyVCGen/WatchdogRaceInstrumenter.cs:            GPUVerifier.AddInlineAttribute(logAccessImplementation);
GPUVerifyVCGen/OriginalRaceInstrumenter.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/OriginalRaceInstrumenter.cs:namespace GPUVerify
GPUVerifyVCGen/OriginalRaceInstrumenter.cs:        public OriginalRaceInstrumenter(GPUVerifier verifier)
GPUVerifyVCGen/OriginalRaceInstrumenter.cs:                GPUVerifier.MakeAccessHasOccurredVariable(v.Name, access);
GPUVerifyVCGen/OriginalRaceInstrumenter.cs:                GPUVerifier.MakeBenignFlagVariable(v.Name);
GPUVerifyVCGen/OriginalRaceInstrumenter.cs:            if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access.IsReadOrWrite())
GPUVerifyVCGen/OriginalRaceInstrumenter.cs:            if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access == AccessType.WRITE)
GPUVerifyVCGen/OriginalRaceInstrumenter.cs:            GPUVerifier.AddInlineAttribute(logAccessImplementation);
GPUVerifyVCGen/Properties/AssemblyInfo.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/Properties/AssemblyInfo.cs:[assembly: AssemblyTitle("GPUVerify")]
GPUVerifyVCGen/Properties/AssemblyInfo.cs:[assembly: AssemblyProduct("GPUVerify")]
GPUVerifyVCGen/ConstantWriteCollector.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/ConstantWriteCollector.cs:namespace GPUVerify
GPUVerifyVCGen/ConstantWriteInstrumenter.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/ConstantWriteInstrumenter.cs:namespace GPUVerify
GPUVerifyVCGen/ConstantWriteInstrumenter.cs:        private GPUVerifier verifier;
GPUVerifyVCGen/ConstantWriteInstrumenter.cs:        public ConstantWriteInstrumenter(GPUVerifier verifier)
GPUVerifyVCGen/NoAccessInstrumenter.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/NoAccessInstrumenter.cs:namespace GPUVerify
GPUVerifyVCGen/NoAccessInstrumenter.cs:        private GPUVerifier verifier;
GPUVerifyVCGen/NoAccessInstrumenter.cs:        public NoAccessInstrumenter(GPUVerifier verifier)
GPUVerifyVCGen/MayBePowerOfTwoAnalyser.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/MayBePowerOfTwoAnalyser.cs:namespace GPUVerify
GPUVerifyVCGen/MayBePowerOfTwoAnalyser.cs:        private GPUVerifier verifier;
GPUVerifyVCGen/MayBePowerOfTwoAnalyser.cs:        public MayBePowerOfTwoAnalyser(GPUVerifier verifier)
GPUVerifyVCGen/MayBePowerOfTwoAnalyser.cs:            if (GPUVerifyVCGenCommandLineOptions.ShowMayBePowerOfTwoAnalysis)
GPUVerifyVCGen/KernelArrayInfoLists.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/KernelArrayInfoLists.cs:namespace GPUVerify
GPUVerifyVCGen/GPUVerifyVCGen.csproj:    <RootNamespace>GPUVerify</RootNamespace>
GPUVerifyVCGen/GPUVerifyVCGen.csproj:    <AssemblyName>GPUVerifyVCGen</AssemblyName>
GPUVerifyVCGen/GPUVerifyVCGen.csproj:    <Compile Include="GPUVerifyVCGenCommandLineOptions.cs" />
GPUVerifyVCGen/GPUVerifyVCGen.csproj:    <Compile Include="GPUVerifier.cs" />
GPUVerifyVCGen/GPUVerifyVCGen.csproj:    <Compile Include="GPUVerifyVCGen.cs" />
GPUVerifyVCGen/GPUVerifyVCGen.csproj:    <ProjectReference Include="..\GPUVerifyLib\GPUVerifyLib.csproj">
GPUVerifyVCGen/GPUVerifyVCGen.csproj:      <Name>GPUVerifyLib</Name>
GPUVerifyVCGen/BarrierIntervalsAnalysis.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/BarrierIntervalsAnalysis.cs:namespace GPUVerify
GPUVerifyVCGen/BarrierIntervalsAnalysis.cs:        private GPUVerifier verifier;
GPUVerifyVCGen/BarrierIntervalsAnalysis.cs:        public BarrierIntervalsAnalysis(GPUVerifier verifier, BarrierStrength strength)
GPUVerifyVCGen/BarrierIntervalsAnalysis.cs:            ExtractCommandsIntoBlocks(impl, item => item is CallCmd && GPUVerifier.IsBarrier(((CallCmd)item).Proc));
GPUVerifyVCGen/BarrierIntervalsAnalysis.cs:            if (GPUVerifyVCGenCommandLineOptions.DebugGPUVerify)
GPUVerifyVCGen/BarrierIntervalsAnalysis.cs:            if (c == null || !GPUVerifier.IsBarrier(c.Proc))
GPUVerifyVCGen/BarrierIntervalsAnalysis.cs:            public HashSet<Variable> FindWrittenGroupSharedArrays(GPUVerifier verifier)
GPUVerifyVCGen/WriteCollector.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/WriteCollector.cs:namespace GPUVerify
GPUVerifyVCGen/GPUVerifier.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/GPUVerifier.cs:namespace GPUVerify
GPUVerifyVCGen/GPUVerifier.cs:    public class GPUVerifier : CheckingContext
GPUVerifyVCGen/GPUVerifier.cs:        public GPUVerifier(string filename, Program program, ResolutionContext rc)
GPUVerifyVCGen/GPUVerifier.cs:            this.IntRep = GPUVerifyVCGenCommandLineOptions.MathInt
GPUVerifyVCGen/GPUVerifier.cs:                Console.WriteLine("GPUVerify: error: _SIZE_T_TYPE size cannot be smaller than group_size_x size");
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.BarrierAccessChecks)
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.OnlyDivergence)
GPUVerifyVCGen/GPUVerifier.cs:                    "{0} GPUVerify format errors detected in {1}",
GPUVerifyVCGen/GPUVerifier.cs:                    GPUVerifyVCGenCommandLineOptions.InputFiles[GPUVerifyVCGenCommandLineOptions.InputFiles.Count - 1]);
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.EqualityAbstraction)
GPUVerifyVCGen/GPUVerifier.cs:                            Console.WriteLine("GPUVerify: error: --equality-abstraction cannot be used with atomics.");
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.CheckSingleNonInlinedImpl)
GPUVerifyVCGen/GPUVerifier.cs:                    Console.WriteLine("GPUVerify: warning: Found {0} non-inlined implementations.", nonInlinedImpls.Count());
GPUVerifyVCGen/GPUVerifier.cs:                Console.WriteLine("GPUVerify: error: exactly one _SIZE_T_TYPE bit-vector type must be specified");
GPUVerifyVCGen/GPUVerifier.cs:                Console.WriteLine("GPUVerify: error: exactly one group_size_x must be specified");
GPUVerifyVCGen/GPUVerifier.cs:                Console.WriteLine("GPUVerify: error: group_size_x must be of type int or bv");
GPUVerifyVCGen/GPUVerifier.cs:                        if (GPUVerifyVCGenCommandLineOptions.ArraysToCheck != null
GPUVerifyVCGen/GPUVerifier.cs:                            && !GPUVerifyVCGenCommandLineOptions.ArraysToCheck.Contains(GlobalArraySourceNames[(decl as Variable).Name]))
GPUVerifyVCGen/GPUVerifier.cs:                        if (GPUVerifyVCGenCommandLineOptions.ArraysToCheck != null
GPUVerifyVCGen/GPUVerifier.cs:                            && !GPUVerifyVCGenCommandLineOptions.ArraysToCheck.Contains(GlobalArraySourceNames[(decl as Variable).Name]))
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.EliminateRedundantReadInstrumentation)
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.PrintLoopStatistics)
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.IdentifySafeBarriers)
GPUVerifyVCGen/GPUVerifier.cs:                GPUVerifyVCGenCommandLineOptions.BarrierAccessChecks = false;
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.ArrayBoundsChecking)
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.RemovePrivateArrayAccesses)
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.RefinedAtomics)
GPUVerifyVCGen/GPUVerifier.cs:            if (!GPUVerifyVCGenCommandLineOptions.OnlyIntraGroupRaceChecking)
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.ShowUniformityAnalysis)
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.Inference)
GPUVerifyVCGen/GPUVerifier.cs:                    if (!GPUVerifyVCGenCommandLineOptions.DisableInessentialLoopDetection)
GPUVerifyVCGen/GPUVerifier.cs:                if (GPUVerifyVCGenCommandLineOptions.ShowStages)
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.KernelInterceptorParams.Count > 0)
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.BarrierAccessChecks)
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.ShowStages)
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.ShowStages)
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.ShowStages)
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.WarpSync)
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.ShowStages)
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.ShowStages)
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.ShowStages)
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.NonDeterminiseUninterpretedFunctions)
GPUVerifyVCGen/GPUVerifier.cs:                if (GPUVerifyVCGenCommandLineOptions.ShowStages)
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.OptimiseBarrierIntervals)
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.ShowStages)
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.Inference)
GPUVerifyVCGen/GPUVerifier.cs:                if (GPUVerifyVCGenCommandLineOptions.AbstractHoudini)
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.WarpSync)
GPUVerifyVCGen/GPUVerifier.cs:            var loopsOutputFile = Path.GetFileNameWithoutExtension(GPUVerifyVCGenCommandLineOptions.InputFiles[0]) + ".loops";
GPUVerifyVCGen/GPUVerifier.cs:            foreach (List<string> param_values in GPUVerifyVCGenCommandLineOptions.KernelInterceptorParams)
GPUVerifyVCGen/GPUVerifier.cs:                                var sourceLoc = new SourceLocationInfo(pc.Attributes, GPUVerifyVCGenCommandLineOptions.InputFiles[0], pc.tok);
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.DoUniformityAnalysis)
GPUVerifyVCGen/GPUVerifier.cs:                Program, GPUVerifyVCGenCommandLineOptions.DoUniformityAnalysis, entryPoints, nonUniformVars);
GPUVerifyVCGen/GPUVerifier.cs:                if (GPUVerifyVCGenCommandLineOptions.OnlyIntraGroupRaceChecking)
GPUVerifyVCGen/GPUVerifier.cs:            Expr warpsize = IntRep.GetLiteral(GPUVerifyVCGenCommandLineOptions.WarpSize, IdType);
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.OnlyIntraGroupRaceChecking)
GPUVerifyVCGen/GPUVerifier.cs:                if (GPUVerifyVCGenCommandLineOptions.OnlyIntraGroupRaceChecking)
GPUVerifyVCGen/GPUVerifier.cs:            if (!GPUVerifyVCGenCommandLineOptions.OnlyDivergence)
GPUVerifyVCGen/GPUVerifier.cs:                    var noAccessVars = GPUVerifyVCGenCommandLineOptions.BarrierAccessChecks ?
GPUVerifyVCGen/GPUVerifier.cs:                    var noAccessVars = GPUVerifyVCGenCommandLineOptions.BarrierAccessChecks ?
GPUVerifyVCGen/GPUVerifier.cs:            if (RaceInstrumentationUtil.RaceCheckingMethod != RaceCheckingMethod.ORIGINAL && !GPUVerifyVCGenCommandLineOptions.OnlyDivergence)
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.ArraysToCheck != null)
GPUVerifyVCGen/GPUVerifier.cs:                foreach (var v in GPUVerifyVCGenCommandLineOptions.ArraysToCheck)
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.DoNotGenerateCandidates.Contains(tag))
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.AdversarialAbstraction)
GPUVerifyVCGen/GPUVerifier.cs:            if (GPUVerifyVCGenCommandLineOptions.EqualityAbstraction)
GPUVerifyVCGen/GPUVerifyVCGen.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/GPUVerifyVCGen.cs:namespace GPUVerify
GPUVerifyVCGen/GPUVerifyVCGen.cs:    public class GPUVerifyVCGen
GPUVerifyVCGen/GPUVerifyVCGen.cs:                int showHelp = GPUVerifyVCGenCommandLineOptions.Parse(args);
GPUVerifyVCGen/GPUVerifyVCGen.cs:                    GPUVerifyVCGenCommandLineOptions.Usage();
GPUVerifyVCGen/GPUVerifyVCGen.cs:                if (GPUVerifyVCGenCommandLineOptions.InputFiles.Count < 1)
GPUVerifyVCGen/GPUVerifyVCGen.cs:                foreach (string file in GPUVerifyVCGenCommandLineOptions.InputFiles)
GPUVerifyVCGen/GPUVerifyVCGen.cs:                        Console.WriteLine("GPUVerify: error: {0} is not a .gbpl file", file);
GPUVerifyVCGen/GPUVerifyVCGen.cs:                if (GPUVerifyVCGenCommandLineOptions.DebugGPUVerify)
GPUVerifyVCGen/GPUVerifyVCGen.cs:                    Console.Error.WriteLine("Exception thrown in GPUVerifyBoogieDriver");
GPUVerifyVCGen/GPUVerifyVCGen.cs:            Program program = ParseBoogieProgram(GPUVerifyVCGenCommandLineOptions.InputFiles, false);
GPUVerifyVCGen/GPUVerifyVCGen.cs:            CommandLineOptions.Clo.PruneInfeasibleEdges = GPUVerifyVCGenCommandLineOptions.PruneInfeasibleEdges;
GPUVerifyVCGen/GPUVerifyVCGen.cs:                Console.WriteLine("{0} name resolution errors detected in {1}", rc.ErrorCount, GPUVerifyVCGenCommandLineOptions.InputFiles[GPUVerifyVCGenCommandLineOptions.InputFiles.Count - 1]);
GPUVerifyVCGen/GPUVerifyVCGen.cs:                Console.WriteLine("{0} type checking errors detected in {1}", errorCount, GPUVerifyVCGenCommandLineOptions.InputFiles[GPUVerifyVCGenCommandLineOptions.InputFiles.Count - 1]);
GPUVerifyVCGen/GPUVerifyVCGen.cs:            if (GPUVerifyVCGenCommandLineOptions.OutputFile != null)
GPUVerifyVCGen/GPUVerifyVCGen.cs:                fn = GPUVerifyVCGenCommandLineOptions.OutputFile;
GPUVerifyVCGen/GPUVerifyVCGen.cs:            else if (GPUVerifyVCGenCommandLineOptions.InputFiles.Count == 1)
GPUVerifyVCGen/GPUVerifyVCGen.cs:                var inputFile = GPUVerifyVCGenCommandLineOptions.InputFiles[0];
GPUVerifyVCGen/GPUVerifyVCGen.cs:            new GPUVerifier(fn, program, rc).DoIt();
GPUVerifyVCGen/GPUVerifyVCGen.cs:                    Console.WriteLine("GPUVerify: error opening file \"{0}\": {1}", bplFileName, e.Message);
GPUVerifyVCGen/BinaryBarrierInvariantDescriptor.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/BinaryBarrierInvariantDescriptor.cs:namespace GPUVerify
GPUVerifyVCGen/BinaryBarrierInvariantDescriptor.cs:            Expr predicate, Expr barrierInvariant, QKeyValue sourceLocationInfo, KernelDualiser dualiser, string procName, GPUVerifier verifier)
GPUVerifyVCGen/BinaryBarrierInvariantDescriptor.cs:            private GPUVerifier verifier;
GPUVerifyVCGen/BinaryBarrierInvariantDescriptor.cs:            public ThreadPairInstantiator(GPUVerifier verifier, Expr instantiationExpr1, Expr instantiationExpr2, int thread)
GPUVerifyVCGen/IConstantWriteInstrumenter.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/IConstantWriteInstrumenter.cs:namespace GPUVerify
GPUVerifyVCGen/UnstructuredRegion.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/UnstructuredRegion.cs:namespace GPUVerify
GPUVerifyVCGen/AccessRecord.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/AccessRecord.cs:namespace GPUVerify
GPUVerifyVCGen/AbstractHoudiniTransformation.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/AbstractHoudiniTransformation.cs:namespace GPUVerify
GPUVerifyVCGen/AbstractHoudiniTransformation.cs:        private GPUVerifier verifier;
GPUVerifyVCGen/AbstractHoudiniTransformation.cs:        public AbstractHoudiniTransformation(GPUVerifier verifier)
GPUVerifyVCGen/RaceInstrumenter.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/RaceInstrumenter.cs:namespace GPUVerify
GPUVerifyVCGen/RaceInstrumenter.cs:        public GPUVerifier Verifier { get; }
GPUVerifyVCGen/RaceInstrumenter.cs:        public RaceInstrumenter(GPUVerifier verifier)
GPUVerifyVCGen/RaceInstrumenter.cs:            if (!GPUVerifyVCGenCommandLineOptions.NoBenign)
GPUVerifyVCGen/RaceInstrumenter.cs:                new IdentifierExpr(Token.NoToken, GPUVerifier.MakeAccessHasOccurredVariable(v.Name, AccessType.WRITE)),
GPUVerifyVCGen/RaceInstrumenter.cs:            if (!GPUVerifyVCGenCommandLineOptions.WarpSync)
GPUVerifyVCGen/RaceInstrumenter.cs:            if (GPUVerifyVCGenCommandLineOptions.ShowAccessBreaking)
GPUVerifyVCGen/RaceInstrumenter.cs:            private GPUVerifier verifier;
GPUVerifyVCGen/RaceInstrumenter.cs:            public ComponentVisitor(GPUVerifier verifier)
GPUVerifyVCGen/RaceInstrumenter.cs:                // happens for CUDA kernels when 64-bit pointer bitwidths are used, because the
GPUVerifyVCGen/RaceInstrumenter.cs:                // thread and groups IDs used by CUDA are always 32 bits).
GPUVerifyVCGen/RaceInstrumenter.cs:                    invariant = Expr.Imp(new IdentifierExpr(Token.NoToken, GPUVerifier.MakeAccessHasOccurredVariable(v.Name, access)), invariant);
GPUVerifyVCGen/RaceInstrumenter.cs:            private GPUVerifier verifier;
GPUVerifyVCGen/RaceInstrumenter.cs:            public DistributeExprVisitor(GPUVerifier verifier)
GPUVerifyVCGen/RaceInstrumenter.cs:            if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access == AccessType.WRITE)
GPUVerifyVCGen/RaceInstrumenter.cs:            else if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access == AccessType.READ)
GPUVerifyVCGen/RaceInstrumenter.cs:                var lowerBoundInv = Expr.Imp(GPUVerifier.MakeAccessHasOccurredExpr(v.Name, access), Verifier.IntRep.MakeSle(lowerBound, OffsetXExpr(v, access, 1)));
GPUVerifyVCGen/RaceInstrumenter.cs:                var upperBoundInv = Expr.Imp(GPUVerifier.MakeAccessHasOccurredExpr(v.Name, access), Verifier.IntRep.MakeSlt(OffsetXExpr(v, access, 1), upperBound));
GPUVerifyVCGen/RaceInstrumenter.cs:            if (!GPUVerifyVCGenCommandLineOptions.NoBenign)
GPUVerifyVCGen/RaceInstrumenter.cs:                    resetCondition, Expr.Not(Expr.Ident(GPUVerifier.MakeAccessHasOccurredVariable(v.Name, kind))));
GPUVerifyVCGen/RaceInstrumenter.cs:            if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access.IsReadOrWrite())
GPUVerifyVCGen/RaceInstrumenter.cs:            if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access == AccessType.WRITE)
GPUVerifyVCGen/RaceInstrumenter.cs:            GPUVerifier.AddInlineAttribute(result);
GPUVerifyVCGen/RaceInstrumenter.cs:            GPUVerifier.AddInlineAttribute(result);
GPUVerifyVCGen/RaceInstrumenter.cs:            if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access.IsReadOrWrite())
GPUVerifyVCGen/RaceInstrumenter.cs:            Variable accessHasOccurredVariable = GPUVerifier.MakeAccessHasOccurredVariable(v.Name, AccessType.WRITE);
GPUVerifyVCGen/RaceInstrumenter.cs:            Variable accessBenignFlagVariable = GPUVerifier.MakeBenignFlagVariable(v.Name);
GPUVerifyVCGen/RaceInstrumenter.cs:            GPUVerifier.AddInlineAttribute(updateBenignFlagImplementation);
GPUVerifyVCGen/RaceInstrumenter.cs:                Variable writeReadBenignFlagVariable = GPUVerifier.MakeBenignFlagVariable(v.Name);
GPUVerifyVCGen/RaceInstrumenter.cs:                if (!GPUVerifyVCGenCommandLineOptions.NoBenign)
GPUVerifyVCGen/RaceInstrumenter.cs:                if (GPUVerifyVCGenCommandLineOptions.AtomicVsRead)
GPUVerifyVCGen/RaceInstrumenter.cs:                if (!GPUVerifyVCGenCommandLineOptions.NoBenign)
GPUVerifyVCGen/RaceInstrumenter.cs:                if (!GPUVerifyVCGenCommandLineOptions.NoBenign)
GPUVerifyVCGen/RaceInstrumenter.cs:                if (GPUVerifyVCGenCommandLineOptions.AtomicVsWrite)
GPUVerifyVCGen/RaceInstrumenter.cs:                if (GPUVerifyVCGenCommandLineOptions.AtomicVsWrite)
GPUVerifyVCGen/RaceInstrumenter.cs:                if (GPUVerifyVCGenCommandLineOptions.AtomicVsRead)
GPUVerifyVCGen/RaceInstrumenter.cs:            Variable accessHasOccurredVariable = GPUVerifier.MakeAccessHasOccurredVariable(v.Name, access);
GPUVerifyVCGen/RaceInstrumenter.cs:            if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access.IsReadOrWrite())
GPUVerifyVCGen/RaceInstrumenter.cs:            if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access == AccessType.WRITE)
GPUVerifyVCGen/RaceInstrumenter.cs:            IdentifierExpr readAccessOccurred1 = new IdentifierExpr(v.tok, GPUVerifier.MakeAccessHasOccurredVariable(v.Name, AccessType.READ));
GPUVerifyVCGen/RaceInstrumenter.cs:            IdentifierExpr writeAccessOccurred1 = new IdentifierExpr(v.tok, GPUVerifier.MakeAccessHasOccurredVariable(v.Name, AccessType.WRITE));
GPUVerifyVCGen/RaceInstrumenter.cs:            IdentifierExpr atomicAccessOccurred1 = new IdentifierExpr(v.tok, GPUVerifier.MakeAccessHasOccurredVariable(v.Name, AccessType.ATOMIC));
GPUVerifyVCGen/RaceInstrumenter.cs:            return new IdentifierExpr(v.tok, GPUVerifier.MakeAccessHasOccurredVariable(v.Name, access));
GPUVerifyVCGen/RaceInstrumenter.cs:                    new IdentifierExpr(Token.NoToken, GPUVerifier.MakeAccessHasOccurredVariable(v.Name, access)),
GPUVerifyVCGen/RaceInstrumenter.cs:                      new IdentifierExpr(v.tok, GPUVerifier.MakeAccessHasOccurredVariable(v.Name, access)),
GPUVerifyVCGen/RaceInstrumenter.cs:                    GPUVerifier.MakeAccessHasOccurredExpr(v.Name, access),
GPUVerifyVCGen/RaceInstrumenter.cs:                    GPUVerifier.MakeAccessHasOccurredExpr(v.Name, access),
GPUVerifyVCGen/RaceInstrumenter.cs:            private GPUVerifier verifier;
GPUVerifyVCGen/RaceInstrumenter.cs:                GPUVerifier.AddInlineAttribute(asyncWorkGroupCopyProcedure);
GPUVerifyVCGen/RaceInstrumenter.cs:                GPUVerifier.AddInlineAttribute(asyncWorkGroupCopyImplementation);
GPUVerifyVCGen/RaceInstrumenter.cs:                if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access == AccessType.WRITE)
GPUVerifyVCGen/RaceInstrumenter.cs:                if (!GPUVerifyVCGenCommandLineOptions.OnlyLog)
GPUVerifyVCGen/RaceInstrumenter.cs:                  access + " operation on " + v + " at " + GPUVerifyVCGenCommandLineOptions.InputFiles[0] + ":" +
GPUVerifyVCGen/RaceInstrumenter.cs:                if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access.IsReadOrWrite())
GPUVerifyVCGen/RaceInstrumenter.cs:                if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access == AccessType.WRITE)
GPUVerifyVCGen/GPUVerifyVCGenCommandLineOptions.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/GPUVerifyVCGenCommandLineOptions.cs:namespace GPUVerify
GPUVerifyVCGen/GPUVerifyVCGenCommandLineOptions.cs:    public class GPUVerifyVCGenCommandLineOptions
GPUVerifyVCGen/GPUVerifyVCGenCommandLineOptions.cs:        public static bool DebugGPUVerify { get; private set; } = false;
GPUVerifyVCGen/GPUVerifyVCGenCommandLineOptions.cs:        // Assigned in GPUVerifier when no barrier invariants occur
GPUVerifyVCGen/GPUVerifyVCGenCommandLineOptions.cs:                    case "-debugGPUVerify":
GPUVerifyVCGen/GPUVerifyVCGenCommandLineOptions.cs:                    case "/debugGPUVerify":
GPUVerifyVCGen/GPUVerifyVCGenCommandLineOptions.cs:                        DebugGPUVerify = true;
GPUVerifyVCGen/GPUVerifyVCGenCommandLineOptions.cs:            Console.WriteLine(@"GPUVerifyVCGen: usage:  GPUVerifyVCGen [ option ... ] [ filename ... ]
GPUVerifyVCGen/GPUVerifyVCGenCommandLineOptions.cs:  Debugging GPUVerifyVCGen
GPUVerifyVCGen/ReducedStrengthAnalysisRegion.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/ReducedStrengthAnalysisRegion.cs:namespace GPUVerify
GPUVerifyVCGen/ReducedStrengthAnalysisRegion.cs:        private GPUVerifier verifier;
GPUVerifyVCGen/ReducedStrengthAnalysisRegion.cs:        private ReducedStrengthAnalysisRegion(Implementation i, GPUVerifier v)
GPUVerifyVCGen/ReducedStrengthAnalysisRegion.cs:            public static StrideForm ComputeStrideForm(Variable v, Expr e, GPUVerifier verifier, HashSet<Variable> modSet)
GPUVerifyVCGen/ReducedStrengthAnalysisRegion.cs:        public static ReducedStrengthAnalysisRegion Analyse(Implementation impl, GPUVerifier verifier)
GPUVerifyVCGen/LiteralIndexedArrayEliminator.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/LiteralIndexedArrayEliminator.cs:namespace GPUVerify
GPUVerifyVCGen/LiteralIndexedArrayEliminator.cs:        private GPUVerifier verifier;
GPUVerifyVCGen/LiteralIndexedArrayEliminator.cs:        public LiteralIndexedArrayEliminator(GPUVerifier verifier)
GPUVerifyVCGen/LiteralIndexedArrayEliminator.cs:            public LiteralIndexVisitor(GPUVerifier verifier)
GPUVerifyVCGen/NullRaceInstrumenter.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/NullRaceInstrumenter.cs:namespace GPUVerify
GPUVerifyVCGen/KernelDualiser.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/KernelDualiser.cs:namespace GPUVerify
GPUVerifyVCGen/KernelDualiser.cs:        public GPUVerifier Verifier { get; }
GPUVerifyVCGen/KernelDualiser.cs:        public KernelDualiser(GPUVerifier verifier)
GPUVerifyVCGen/KernelDualiser.cs:                if (GPUVerifier.IsBarrier(call.Proc))
GPUVerifyVCGen/KernelDualiser.cs:                        if (GPUVerifyVCGenCommandLineOptions.BarrierAccessChecks)
GPUVerifyVCGen/KernelDualiser.cs:                if (GPUVerifier.IsBarrier(call.Proc))
GPUVerifyVCGen/KernelDualiser.cs:                    if (!GPUVerifyVCGenCommandLineOptions.AsymmetricAsserts && !ContainsAsymmetricExpression(a.Expr) && !isUniform)
GPUVerifyVCGen/KernelDualiser.cs:                        if (isShared && !GPUVerifyVCGenCommandLineOptions.OnlyIntraGroupRaceChecking)
GPUVerifyVCGen/KernelDualiser.cs:                        if (isShared && !GPUVerifyVCGenCommandLineOptions.OnlyIntraGroupRaceChecking)
GPUVerifyVCGen/KernelDualiser.cs:                        || (Verifier.IsGroupIdConstant(d as Variable) && !GPUVerifyVCGenCommandLineOptions.OnlyIntraGroupRaceChecking)))
GPUVerifyVCGen/KernelDualiser.cs:                            && !GPUVerifyVCGenCommandLineOptions.OnlyIntraGroupRaceChecking)
GPUVerifyVCGen/KernelDualiser.cs:                        if (!GPUVerifyVCGenCommandLineOptions.OnlyIntraGroupRaceChecking)
GPUVerifyVCGen/ArrayControlFlowAnalyser.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/ArrayControlFlowAnalyser.cs:namespace GPUVerify
GPUVerifyVCGen/ArrayControlFlowAnalyser.cs:        private GPUVerifier verifier;
GPUVerifyVCGen/ArrayControlFlowAnalyser.cs:        public ArrayControlFlowAnalyser(GPUVerifier verifier)
GPUVerifyVCGen/ArrayControlFlowAnalyser.cs:            if (GPUVerifyVCGenCommandLineOptions.ShowArrayControlFlowAnalysis)
GPUVerifyVCGen/ArrayControlFlowAnalyser.cs:                    else if (!GPUVerifier.IsBarrier(callCmd.Proc))
GPUVerifyVCGen/INoAccessInstrumenter.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/INoAccessInstrumenter.cs:namespace GPUVerify
GPUVerifyVCGen/VariableDualiser.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/VariableDualiser.cs:namespace GPUVerify
GPUVerifyVCGen/VariableDualiser.cs:        private GPUVerifier verifier;
GPUVerifyVCGen/VariableDualiser.cs:        public VariableDualiser(int id, GPUVerifier verifier, string procName)
GPUVerifyVCGen/VariableDualiser.cs:                if (GPUVerifyVCGenCommandLineOptions.OnlyIntraGroupRaceChecking)
GPUVerifyVCGen/VariableDualiser.cs:                        && !GPUVerifyVCGenCommandLineOptions.OnlyIntraGroupRaceChecking)
GPUVerifyVCGen/VariableDualiser.cs:                        && !GPUVerifyVCGenCommandLineOptions.OnlyIntraGroupRaceChecking)
GPUVerifyVCGen/VariableDualiser.cs:            if (QKeyValue.FindBoolAttribute(v.Attributes, "group_shared") && !GPUVerifyVCGenCommandLineOptions.OnlyIntraGroupRaceChecking)
GPUVerifyVCGen/ArrayBoundsChecker.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/ArrayBoundsChecker.cs:namespace GPUVerify
GPUVerifyVCGen/ArrayBoundsChecker.cs:        private GPUVerifier verifier;
GPUVerifyVCGen/ArrayBoundsChecker.cs:        public ArrayBoundsChecker(GPUVerifier verifier, Program program)
GPUVerifyVCGen/AdversarialAbstraction.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyVCGen/AdversarialAbstraction.cs:namespace GPUVerify
GPUVerifyVCGen/AdversarialAbstraction.cs:        private GPUVerifier verifier;
GPUVerifyVCGen/AdversarialAbstraction.cs:        public AdversarialAbstraction(GPUVerifier verifier)
GPUVerifyVCGen/AdversarialAbstraction.cs:            private GPUVerifier verifier;
GPUVerifyVCGen/AdversarialAbstraction.cs:            public AccessesAdversarialArrayVisitor(GPUVerifier verifier)
GPUVerify.py:from GPUVerifyScript.argument_parser import ArgumentParserError, parse_arguments
GPUVerify.py:from GPUVerifyScript.constants import AnalysisMode, SourceLanguage
GPUVerify.py:from GPUVerifyScript.error_codes import ErrorCodes
GPUVerify.py:from GPUVerifyScript.json_loader import JSONError, json_load
GPUVerify.py:    return "GPUVerify: CONFIGURATION_ERROR error ({}): {}".format(ErrorCodes.CONFIGURATION_ERROR,self.msg)
GPUVerify.py:  sys.stderr.write("GPUVerify requires Python to be equipped with the psutil module.\n")
GPUVerify.py:# Try to import the paths need for GPUVerify's tools
GPUVerify.py:Tools = ["clang", "opt", "bugle", "gpuverifyvcgen", "gpuverifycruncher", "gpuverifyboogiedriver"]
GPUVerify.py:Extensions = { 'clang': ".bc", 'opt': ".opt.bc", 'bugle': ".gbpl", 'gpuverifyvcgen': ".bpl", 'gpuverifycruncher': ".cbpl" }
GPUVerify.py:class GPUVerifyInstance (object):
GPUVerify.py:    if args.source_language == SourceLanguage.CUDA:
GPUVerify.py:    elif args.source_language == SourceLanguage.OpenCL:
GPUVerify.py:      defines += ["__OPENCL_VERSION__=120"]
GPUVerify.py:    if args.source_language == SourceLanguage.CUDA:
GPUVerify.py:    elif  args.source_language == SourceLanguage.OpenCL:
GPUVerify.py:    if args.source_language == SourceLanguage.CUDA:
GPUVerify.py:        options += [ "-target", "i386--" ] # gives nvptx-nvidia-cuda
GPUVerify.py:        options += [ "-target", "x86_64--" ] # gives nvptx64-nvidia-cuda
GPUVerify.py:      options += ["--cuda-device-only", "-nocudainc", "-nocudalib"]
GPUVerify.py:      options += ["--cuda-gpu-arch=sm_35", "-x", "cuda"]
GPUVerify.py:      options += ["-Xclang", "-fcuda-is-device", "-include", "cuda.h"]
GPUVerify.py:    elif args.source_language == SourceLanguage.OpenCL:
GPUVerify.py:      options += ["-Xclang", "-cl-std=CL1.2", "-O0", "-include", "opencl.h"]
GPUVerify.py:    # Must be added after include of opencl/cuda header
GPUVerify.py:    if args.source_language == SourceLanguage.CUDA:
GPUVerify.py:    elif args.source_language == SourceLanguage.OpenCL:
GPUVerify.py:      options.append("/debugGPUVerify")
GPUVerify.py:      options.append("/debugGPUVerify")
GPUVerify.py:    # See GPUVerifyLib/ToolExitCodes.cs
GPUVerify.py:      success, timeout = self.runTool("gpuverifyvcgen",
GPUVerify.py:              [gvfindtools.gpuVerifyBinDir + "/GPUVerifyVCGen.exe"] +
GPUVerify.py:      if success != 0: return ErrorCodes.GPUVERIFYVCGEN_ERROR
GPUVerify.py:      success, timeout = self.runTool("gpuverifycruncher",
GPUVerify.py:                [gvfindtools.gpuVerifyBinDir + os.sep + "GPUVerifyCruncher.exe"] +
GPUVerify.py:    success, timeout = self.runTool("gpuverifyboogiedriver",
GPUVerify.py:            [gvfindtools.gpuVerifyBinDir + "/GPUVerifyBoogieDriver.exe"] +
GPUVerify.py:        print("- no data races within " + ("work groups" if self.SL == SourceLanguage.OpenCL else "thread blocks"), file=self.outFile)
GPUVerify.py:          print("- no data races between " + ("work groups" if self.SL == SourceLanguage.OpenCL else "thread blocks"), file=self.outFile)
GPUVerify.py:  kernel_args.source_language = SourceLanguage.OpenCL
GPUVerify.py:  print("GPUVerify kernel analyzer checked {} kernels.".format(len(success) + len(failure)))
GPUVerify.py:  """ This wraps GPUVerify's real main function so
GPUVerify.py:  gv_instance = GPUVerifyInstance(args, out, err, cleanUpHandler)
GPUVerify.sln:Project("{FAE04EC0-301F-11D3-BF4B-00C04F79EFBC}") = "GPUVerifyVCGen", "GPUVerifyVCGen\GPUVerifyVCGen.csproj", "{E5D16606-06D0-434F-A9D7-7D079BC80229}"
GPUVerify.sln:Project("{FAE04EC0-301F-11D3-BF4B-00C04F79EFBC}") = "GPUVerifyBoogieDriver", "GPUVerifyBoogieDriver\GPUVerifyBoogieDriver.csproj", "{FD2A2C67-1BD6-4A1A-B65B-B057267E24A3}"
GPUVerify.sln:Project("{FAE04EC0-301F-11D3-BF4B-00C04F79EFBC}") = "GPUVerifyCruncher", "GPUVerifyCruncher\GPUVerifyCruncher.csproj", "{791E259B-B800-400F-8AA4-A92A565B3AA3}"
GPUVerify.sln:Project("{FAE04EC0-301F-11D3-BF4B-00C04F79EFBC}") = "GPUVerifyLib", "GPUVerifyLib\GPUVerifyLib.csproj", "{5E7E9AF7-4166-4082-B88B-F7766023D877}"
GPUVerify.sln:		StartupItem = GPUVerifyVCGen\GPUVerifyVCGen.csproj
deploy.py:# Try to import the paths need for GPUVerify's tools
deploy.py:GPUVerifyRoot = sys.path[0]
deploy.py:if os.path.isfile(os.path.join(GPUVerifyRoot, 'gvfindtools.py')):
deploy.py:sys.path.insert(0, os.path.join(GPUVerifyRoot, 'gvfindtools.templates'))
deploy.py:  des=('Deploys GPUVerify to a directory by copying the necessary '
deploy.py:      'files from the development directory so that GPUVerify can '
deploy.py:                      help = "The path to the directory that GPUVerify will be deployed to"
deploy.py:  FileCopy(os.path.join(GPUVerifyRoot, "Documentation"), "tutorial.rst", deployDir),
deploy.py:  # GPUVerify
deploy.py:  FileCopy(GPUVerifyRoot, 'GPUVerify.py', deployDir),
deploy.py:  FileCopy(GPUVerifyRoot, 'getversion.py', deployDir),
deploy.py:  FileCopy(GPUVerifyRoot, 'LICENSE.TXT', licenseDest),
deploy.py:  MoveFile(os.path.join(licenseDest, 'LICENSE.TXT'), os.path.join(licenseDest, 'gpuverify-boogie.txt')),
deploy.py:  IfUsing('posix',FileCopy(GPUVerifyRoot, 'gpuverify', deployDir)),
deploy.py:  IfUsing('nt',FileCopy(GPUVerifyRoot, 'GPUVerify.bat', deployDir)),
deploy.py:  FileCopy(os.path.join(GPUVerifyRoot, 'gvfindtools.templates'), 'gvfindtoolsdeploy.py', deployDir), # Note this will patched later
deploy.py:  RegexFileCopy(gvfindtools.gpuVerifyBinDir, r'^.+\.(dll|exe)$', gvfindtoolsdeploy.gpuVerifyBinDir),
deploy.py:  FileCopy(GPUVerifyRoot, 'gvtester.py', deployDir),
deploy.py:  DirCopy(os.path.join(GPUVerifyRoot ,'testsuite'), os.path.join(deployDir, 'testsuite')),
deploy.py:  DirCopy(os.path.join(GPUVerifyRoot ,'GPUVerifyScript'), os.path.join(deployDir, 'GPUVerifyScript'), copyOnlyRegex=r'^.+\.py$'),
deploy.py:  CreateFileFromString(versionString, os.path.join(deployDir, os.path.basename(getversion.GPUVerifyDeployVersionFile))),
deploy.py:    # Make a list of the assemblies that need embedding and the GPUVerify executable names
deploy.py:    _ignored, _ignored, files = next(os.walk(gvfindtools.gpuVerifyBinDir))
deploy.py:    gpuverifyExecutables = list(filter(lambda f: f.startswith('GPUVerify') and f.endswith('.exe'), files))
deploy.py:    assert len(gpuverifyExecutables) > 0
deploy.py:    for tool in gpuverifyExecutables:
deploy.py:        EmbedMonoRuntime(exePath = os.path.join(gvfindtoolsdeploy.gpuVerifyBinDir, tool),
deploy.py:                         outputPath = os.path.join(gvfindtoolsdeploy.gpuVerifyBinDir, tool + ".mono"),
deploy.py:                         assemblies = map(lambda a: os.path.join(gvfindtoolsdeploy.gpuVerifyBinDir, a), assemblies)))
deploy.py:    for fileToRemove in assemblies + gpuverifyExecutables:
deploy.py:          RemoveFile(os.path.join(gvfindtoolsdeploy.gpuVerifyBinDir, fileToRemove)))
deploy.py:    # Finally rename the bundled executables (e.g. GPUVerifyVCGen.exe.mono -> GPUVerifyVCGen.exe)
deploy.py:    for executable in gpuverifyExecutables:
deploy.py:        MoveFile(srcpath = os.path.join(gvfindtoolsdeploy.gpuVerifyBinDir, executable + '.mono'),
deploy.py:                 destpath = os.path.join(gvfindtoolsdeploy.gpuVerifyBinDir, executable)))
deploy.py:      deployActions.append(StripFile(os.path.join(gvfindtoolsdeploy.gpuVerifyBinDir, executable)))
license_banner.txt://                GPUVerify - a Verifier for GPU Kernels
GPUVerify.bat:python3 "%~dp0GPUVerify.py" %*
GPUVerifyLib/RaceInstrumentationUtil.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyLib/RaceInstrumentationUtil.cs:namespace GPUVerify
GPUVerifyLib/GPUVerifyErrorReporter.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyLib/GPUVerifyErrorReporter.cs:namespace GPUVerify
GPUVerifyLib/GPUVerifyErrorReporter.cs:    public class GPUVerifyErrorReporter
GPUVerifyLib/GPUVerifyErrorReporter.cs:        public GPUVerifyErrorReporter(Program program, string implName)
GPUVerifyLib/GPUVerifyErrorReporter.cs:                Console.WriteLine("GPUVerify: error: exactly one _SIZE_T_TYPE bit-vector type must be specified");
GPUVerifyLib/GPUVerifyErrorReporter.cs:            if (((GVCommandLineOptions)CommandLineOptions.Clo).SourceLanguage == SourceLanguage.CUDA)
GPUVerifyLib/GPUVerifyLib.csproj:    <RootNamespace>GPUVerifyLib</RootNamespace>
GPUVerifyLib/GPUVerifyLib.csproj:    <AssemblyName>GPUVerifyLib</AssemblyName>
GPUVerifyLib/GPUVerifyLib.csproj:    <Compile Include="GPUVerifyErrorReporter.cs" />
GPUVerifyLib/GVCommandLineOptions.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyLib/GVCommandLineOptions.cs:namespace GPUVerify
GPUVerifyLib/GVCommandLineOptions.cs:        OpenCL, CUDA
GPUVerifyLib/GVCommandLineOptions.cs:        public bool DebugGPUVerify { get; private set; } = false;
GPUVerifyLib/GVCommandLineOptions.cs:        public SourceLanguage SourceLanguage { get; private set; } = SourceLanguage.OpenCL;
GPUVerifyLib/GVCommandLineOptions.cs:            : base("GPUVerify", "GPUVerify kernel analyser")
GPUVerifyLib/GVCommandLineOptions.cs:                        SourceLanguage = SourceLanguage.OpenCL;
GPUVerifyLib/GVCommandLineOptions.cs:                        SourceLanguage = SourceLanguage.CUDA;
GPUVerifyLib/GVCommandLineOptions.cs:            if (name == "debugGPUVerify")
GPUVerifyLib/GVCommandLineOptions.cs:                DebugGPUVerify = true;
GPUVerifyLib/CheckForQuantifiersVisitor.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyLib/CheckForQuantifiersVisitor.cs:namespace GPUVerify
GPUVerifyLib/Utilities.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyLib/Utilities.cs:namespace GPUVerify
GPUVerifyLib/Utilities.cs:    /// Utility class for GPUVerify.
GPUVerifyLib/Utilities.cs:        /// IO utility class for GPUVerify.
GPUVerifyLib/Utilities.cs:                    Console.Error.WriteLine("GPUVerify has had trouble loading one of its components due to security settings.");
GPUVerifyLib/Utilities.cs:                    Console.Error.WriteLine("In order to run GPUVerify successfully you need to unblock the archive before unzipping it.");
GPUVerifyLib/Utilities.cs:                    Console.Error.WriteLine("Once this is done, unzip GPUVerify afresh and this issue should be resolved.");
GPUVerifyLib/Utilities.cs:                Console.Error.WriteLine("\nGPUVerify: an internal error has occurred, details written to " + DUMP_FILE + ".");
GPUVerifyLib/Utilities.cs:                Console.Error.WriteLine("Please consult the troubleshooting guide in the GPUVerify documentation");
GPUVerifyLib/Utilities.cs:                Console.Error.WriteLine("GPUVerify issue tracker:");
GPUVerifyLib/Utilities.cs:                Console.Error.WriteLine("  https://github.com/mc-imperial/gpuverify");
GPUVerifyLib/Utilities.cs:                    Console.Error.WriteLine("Hint: It looks like GPUVerify is having trouble invoking its");
GPUVerifyLib/ToolExitCodes.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyLib/ToolExitCodes.cs:namespace GPUVerify
GPUVerifyLib/ToolExitCodes.cs:        // If we have uncaught exceptions (i.e. with -DebugGPUVerify) then mono will exit with this exit code 1.
GPUVerifyLib/KernelAnalyser.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyLib/KernelAnalyser.cs:namespace GPUVerify
GPUVerifyLib/KernelAnalyser.cs:                // We subtract "VerificationErrors" here because its a bug report, not an error within GPUVerifyCruncher/GPUVerifyBoogieDriver
GPUVerifyLib/KernelAnalyser.cs:                        new GPUVerifyErrorReporter(program, implName).ReportCounterexample(error);
GPUVerifyLib/Properties/AssemblyInfo.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyLib/Properties/AssemblyInfo.cs:[assembly: AssemblyTitle("GPUVerifyLib")]
GPUVerifyLib/Properties/AssemblyInfo.cs:[assembly: AssemblyProduct("GPUVerifyLib")]
GPUVerifyLib/VariablesOccurringInExpressionVisitor.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyLib/VariablesOccurringInExpressionVisitor.cs:namespace GPUVerify
GPUVerifyLib/AccessType.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyLib/AccessType.cs:namespace GPUVerify
GPUVerifyLib/SourceLocationInfo.cs://                GPUVerify - a Verifier for GPU Kernels
GPUVerifyLib/SourceLocationInfo.cs:namespace GPUVerify
gpuverify:python3 $(dirname $0)/GPUVerify.py $@
.gitignore:# GPUVerifyBoogieDriver build files
.gitignore:GPUVerifyBoogieDriver/bin/*
.gitignore:GPUVerifyBoogieDriver/obj/*
.gitignore:# GPUVerifyVCGen build files
.gitignore:GPUVerifyVCGen/bin/*
.gitignore:GPUVerifyVCGen/obj/*
.gitignore:# GPUVerifyCruncher
.gitignore:GPUVerifyCruncher/obj/*
.gitignore:# GPUVerifyLib
.gitignore:GPUVerifyLib/obj/*
.gitignore:# GPUVerify Temporary files
.gitignore:utils/GPUVerifyRise4Fun/config.py
.gitignore:utils/GPUVerifyRise4Fun/venv
.gitignore:GPUVerify.v*.suo
.gitignore:utils/GPUVerifyRise4Fun/*-counter.pickle
.gitignore:# GPUVerify error report dump
utils/check_json.py:"""Utility for checking the validity of JSON files accepted by GPUVerify."""
utils/check_json.py:from GPUVerifyScript.json_loader import JSONError, json_load
utils/GPUVerifyRise4Fun/clientutil.py:This module contains code useful for building a GPUVerifyRise4Fun client
utils/GPUVerifyRise4Fun/production_server.py:      This program will run the GPUVerify Rise4Fun web service
utils/GPUVerifyRise4Fun/production_server.py:    logging.info("Starting GPUVerifyRise4Fun")
utils/GPUVerifyRise4Fun/gvapi.py:    API to GPUVerify
utils/GPUVerifyRise4Fun/gvapi.py:# Put GPUVerify.py module in search path
utils/GPUVerifyRise4Fun/gvapi.py:sys.path.insert(0, config.GPUVERIFY_ROOT_DIR)
utils/GPUVerifyRise4Fun/gvapi.py:from GPUVerifyScript.error_codes import ErrorCodes
utils/GPUVerifyRise4Fun/gvapi.py:ErrorCodes.GPUVERIFYVCGEN_ERROR:"Could not generate invariants and/or perform two-thread abstraction.",
utils/GPUVerifyRise4Fun/gvapi.py:ErrorCodes.CONFIGURATION_ERROR:"The web service has been incorrectly configured. Please report this issue to gpuverify-support@googlegroups.com",
utils/GPUVerifyRise4Fun/gvapi.py:class GPUVerifyObserver(object):
utils/GPUVerifyRise4Fun/gvapi.py:      Receive a notification of a completed GPUVerify command.
utils/GPUVerifyRise4Fun/gvapi.py:      returnCode : The return code given by GPUVerify
utils/GPUVerifyRise4Fun/gvapi.py:      output     : The output of the GPUVerify Tool
utils/GPUVerifyRise4Fun/gvapi.py:class GPUVerifyTool(object):
utils/GPUVerifyRise4Fun/gvapi.py:      rootPath : Is the root directory of the GPUVerify tool ( development or deploy)
utils/GPUVerifyRise4Fun/gvapi.py:      raise Exception('Path to GPUVerify root must exist')
utils/GPUVerifyRise4Fun/gvapi.py:    self.toolPath = os.path.join(rootPath,'GPUVerify.py')
utils/GPUVerifyRise4Fun/gvapi.py:      raise Exception('Could not find GPUVerify at "' + self.toolPath + '"')
utils/GPUVerifyRise4Fun/gvapi.py:        Register an observer (of type GPUVerifyObserver) that will receive notifications
utils/GPUVerifyRise4Fun/gvapi.py:        when the runCUDA() or runOpenCL() methods are executed.
utils/GPUVerifyRise4Fun/gvapi.py:    if not isinstance(observer, GPUVerifyObserver):
utils/GPUVerifyRise4Fun/gvapi.py:      to be passed to runOpencl() or runCUDA()
utils/GPUVerifyRise4Fun/gvapi.py:                 # OpenCL NDRange arguments
utils/GPUVerifyRise4Fun/gvapi.py:                 # CUDA grid arguments
utils/GPUVerifyRise4Fun/gvapi.py:  def runOpenCL(self, source, args, timeout=10):
utils/GPUVerifyRise4Fun/gvapi.py:        This function will excute GPUVerify on source code. This function
utils/GPUVerifyRise4Fun/gvapi.py:        an OpenCL kernel and a CUDA kernel.
utils/GPUVerifyRise4Fun/gvapi.py:        cmdArgs : A list of command line arguments to pass to GPUVerify
utils/GPUVerifyRise4Fun/gvapi.py:    f = tempfile.NamedTemporaryFile(prefix='gpuverify-source-',
utils/GPUVerifyRise4Fun/gvapi.py:        _logging.error('GPUVerify timed out (ErrorCode:{})'.format(response[0]))
utils/GPUVerifyRise4Fun/gvapi.py:  def runCUDA(self, source, args, timeout=10):
utils/GPUVerifyRise4Fun/gvapi.py:    tempDir = tempfile.mkdtemp(prefix='gpuverify-working-directory-temp',dir=self.tempDir)
utils/GPUVerifyRise4Fun/gvapi.py:                                    preexec_fn=os.setsid) # Make Sure GPUVerify can't kill us!
utils/GPUVerifyRise4Fun/config.py.template:version of GPUVerify and instead you can use a deployed version.
utils/GPUVerifyRise4Fun/config.py.template:GPUVERIFY_ROOT_DIR= '/data/dev/gpuverify-web-build' # Development or deploy root directory for GPUVerify
utils/GPUVerifyRise4Fun/config.py.template:GPUVERIFY_TEMP_DIR=None # The directory to place temporary files during GPUVerify execution. None will use system default.
utils/GPUVerifyRise4Fun/config.py.template:GPUVERIFY_TIMEOUT=30 # Number of seconds to wait for result before giving up
utils/GPUVerifyRise4Fun/config.py.template:# If set true the version number will include the output of running --version on GPUVerify in this repository.
utils/GPUVerifyRise4Fun/config.py.template:# You should set this to False if you have not configured the local version of GPUVerify
utils/GPUVerifyRise4Fun/config.py.template:# List of default arguments to pass to GPUVerify
utils/GPUVerifyRise4Fun/config.py.template:# Note you should not pass --timeout= ( see GPUVERIFY_TIMEOUT )
utils/GPUVerifyRise4Fun/config.py.template:GPUVERIFY_DEFAULT_ARGS= ['--verbose']
utils/GPUVerifyRise4Fun/tester.py:This script automatically tests the GPUVerifyRise4Fun webservice
utils/GPUVerifyRise4Fun/tester.py:    for lang in ['opencl', 'cuda']:
utils/GPUVerifyRise4Fun/opencl/syntax.py:This file provides the OpenCL C syntax definition for Rise4Fun web service
utils/GPUVerifyRise4Fun/opencl/samples/simple_barrier_divergence.cl: * which is what is required in OpenCL.
utils/GPUVerifyRise4Fun/observers/kernelcounter.py:class KernelCounterObserver(gvapi.GPUVerifyObserver):
utils/GPUVerifyRise4Fun/observers/kernelrecorder.py:# This observer records GPUVerify run information
utils/GPUVerifyRise4Fun/observers/kernelrecorder.py:class KernelRecorderObserver(gvapi.GPUVerifyObserver):
utils/GPUVerifyRise4Fun/observers/example.py:class ExampleObserver(gvapi.GPUVerifyObserver):
utils/GPUVerifyRise4Fun/webservice.py:cudaMetaData = {}
utils/GPUVerifyRise4Fun/webservice.py:openclMetaData = {} 
utils/GPUVerifyRise4Fun/webservice.py:  global cudaMetaData , openclMetaData, _gpuverifyObservers, _tool, _sourceCodeSanitiser
utils/GPUVerifyRise4Fun/webservice.py:  cudaMetaData = CUDAMetaData(app.config['SRC_ROOT'])
utils/GPUVerifyRise4Fun/webservice.py:  openclMetaData = OpenCLMetaData(app.config['SRC_ROOT'])
utils/GPUVerifyRise4Fun/webservice.py:  # Create GPUVerify tool instance
utils/GPUVerifyRise4Fun/webservice.py:  _tool = gvapi.GPUVerifyTool(app.config['GPUVERIFY_ROOT_DIR'], app.config['GPUVERIFY_TEMP_DIR'])
utils/GPUVerifyRise4Fun/webservice.py:  if lang == CUDAMetaData.folderName:
utils/GPUVerifyRise4Fun/webservice.py:    metaData = cudaMetaData.metadata
utils/GPUVerifyRise4Fun/webservice.py:    metaData = openclMetaData.metadata
utils/GPUVerifyRise4Fun/webservice.py:  if lang == CUDAMetaData.folderName:
utils/GPUVerifyRise4Fun/webservice.py:    metaData = cudaMetaData
utils/GPUVerifyRise4Fun/webservice.py:    metaData = openclMetaData
utils/GPUVerifyRise4Fun/webservice.py:def runGpuverify(lang):
utils/GPUVerifyRise4Fun/webservice.py:    _tool.filterCmdArgs(source, safeArgs, ignoredArgs, app.config['GPUVERIFY_DEFAULT_ARGS'])
utils/GPUVerifyRise4Fun/webservice.py:    if lang == CUDAMetaData.folderName:
utils/GPUVerifyRise4Fun/webservice.py:      returnMessage['Version'] = cudaMetaData.metadata['Version']
utils/GPUVerifyRise4Fun/webservice.py:      (returnCode, toolMessage) = _tool.runCUDA(source, 
utils/GPUVerifyRise4Fun/webservice.py:                                                timeout=app.config['GPUVERIFY_TIMEOUT']
utils/GPUVerifyRise4Fun/webservice.py:      returnMessage['Version'] = openclMetaData.metadata['Version']
utils/GPUVerifyRise4Fun/webservice.py:      (returnCode, toolMessage) = _tool.runOpenCL(source, 
utils/GPUVerifyRise4Fun/webservice.py:                                                  timeout=app.config['GPUVERIFY_TIMEOUT']
utils/GPUVerifyRise4Fun/webservice.py:def getGPUVerifyHelp():
utils/GPUVerifyRise4Fun/webservice.py:  (returnCode, toolMessage) = _tool.runOpenCL("",["--help"]);
utils/GPUVerifyRise4Fun/webservice.py:  parser = argparse.ArgumentParser(description='Run Development version of GPUVerifyRise4Fun web service')
utils/GPUVerifyRise4Fun/simpleclient.py:GPUVerifyRise4Fun instance with a query.
utils/GPUVerifyRise4Fun/simpleclient.py:be passed to GPUVerify instance running via
utils/GPUVerifyRise4Fun/simpleclient.py:GPUVerifyRise4Fun.
utils/GPUVerifyRise4Fun/simpleclient.py:    parser.add_argument('--gvhelp', action='store_true', help="Show GPUVerify help information and exit. You need to pass a kernel filename even though it won't be used")
utils/GPUVerifyRise4Fun/simpleclient.py:        logging.error("There must be at least two arguments passed to GPUVerifyRise4Fun (pass after the kernel filename)")
utils/GPUVerifyRise4Fun/simpleclient.py:        logging.info("Passing the following arguments to the GPUVerify executable" +
utils/GPUVerifyRise4Fun/simpleclient.py:                     " running on GPUVerifyRise4Fun:\n{}".format(pprint.pformat(otherArgs))
utils/GPUVerifyRise4Fun/simpleclient.py:    if knownArgs.kernel.endswith('.cl'): langStr='opencl'
utils/GPUVerifyRise4Fun/simpleclient.py:    if knownArgs.kernel.endswith('.cu'): langStr='cuda'
utils/GPUVerifyRise4Fun/simpleclient.py:    # Build source file to send to GPUVerifyRise4Fun
utils/GPUVerifyRise4Fun/simpleclient.py:    # Build first line which has GPUVerify arguments on it
utils/GPUVerifyRise4Fun/simpleclient.py:    logging.debug("Sending the following to GPUVerifyRise4Fun:\n{0}".format(source))
utils/GPUVerifyRise4Fun/service/upstart-job.conf:# This is a template for a GPUVerifyRise4Fun upstart job
utils/GPUVerifyRise4Fun/service/upstart-job.conf:description "This an upstart job to start GPUVerifyRise4fun"
utils/GPUVerifyRise4Fun/meta_data.py:      version, _NOT_USED = gvapi.GPUVerifyTool(config.GPUVERIFY_ROOT_DIR, 
utils/GPUVerifyRise4Fun/meta_data.py:                                               config.GPUVERIFY_TEMP_DIR).getVersionString()
utils/GPUVerifyRise4Fun/meta_data.py:        # Grab GPUVerify version from the repository that the webservice is in
utils/GPUVerifyRise4Fun/meta_data.py:        # which is not necessarily the same repository that GPUVerify is in.
utils/GPUVerifyRise4Fun/meta_data.py:        # This requires that the GPUVerify version at this location be 
utils/GPUVerifyRise4Fun/meta_data.py:        _logging.debug('Using path to local GPUVerify as "{}"'.format(pathToLocalGV))
utils/GPUVerifyRise4Fun/meta_data.py:        localVersion, _NOT_USED = gvapi.GPUVerifyTool(pathToLocalGV, 
utils/GPUVerifyRise4Fun/meta_data.py:                                                      config.GPUVERIFY_TEMP_DIR).getVersionString()
utils/GPUVerifyRise4Fun/meta_data.py:      # <x> : Version of GPUVerify that will be invoked by GPUVerifyRise4Fun
utils/GPUVerifyRise4Fun/meta_data.py:      # <y> : Version of GPUVerify Rise4Fun
utils/GPUVerifyRise4Fun/meta_data.py:        "Name": "GPUVerify",
utils/GPUVerifyRise4Fun/meta_data.py:        "DisplayName": "GPUVerify",
utils/GPUVerifyRise4Fun/meta_data.py:        "Email": "gpuverify-support@googlegroups.com",
utils/GPUVerifyRise4Fun/meta_data.py:        "SupportEmail": "gpuverify-support@googlegroups.com",
utils/GPUVerifyRise4Fun/meta_data.py:        "TermsOfUseUrl": "http://multicore.doc.ic.ac.uk/tools/GPUVerify/", # FIXME: NOT PROPER POLICY
utils/GPUVerifyRise4Fun/meta_data.py:        "PrivacyUrl": "http://multicore.doc.ic.ac.uk/tools/GPUVerify/", # FIXME: NOT PROPER POLICY
utils/GPUVerifyRise4Fun/meta_data.py:        "Title": "A verifier for CUDA/OpenCL kernels",
utils/GPUVerifyRise4Fun/meta_data.py:        "Url": "http://multicore.doc.ic.ac.uk/tools/GPUVerify/",
utils/GPUVerifyRise4Fun/meta_data.py:class OpenCLMetaData(BasicMetaData):
utils/GPUVerifyRise4Fun/meta_data.py:  folderName='opencl'
utils/GPUVerifyRise4Fun/meta_data.py:    import opencl.syntax
utils/GPUVerifyRise4Fun/meta_data.py:    # FIXME: use 'x-opencl' and provide own syntax definition
utils/GPUVerifyRise4Fun/meta_data.py:    self.metadata['Question'] = 'Is this OpenCL kernel correct?'
utils/GPUVerifyRise4Fun/meta_data.py:    self.metadata['DisplayName'] += '-OpenCL'
utils/GPUVerifyRise4Fun/meta_data.py:    self.metadata['Name'] += 'OpenCL'
utils/GPUVerifyRise4Fun/meta_data.py:    self.loadLanguageSyntax(opencl.syntax)
utils/GPUVerifyRise4Fun/meta_data.py:    _logging.debug("Generated OpenCL metadata:\n" + pprint.pformat(self.metadata))
utils/GPUVerifyRise4Fun/meta_data.py:class CUDAMetaData(BasicMetaData):
utils/GPUVerifyRise4Fun/meta_data.py:  folderName='cuda'
utils/GPUVerifyRise4Fun/meta_data.py:    import cuda.syntax
utils/GPUVerifyRise4Fun/meta_data.py:    # FIXME: use 'x-cuda' and provide own syntax definition
utils/GPUVerifyRise4Fun/meta_data.py:    self.metadata['Question'] = 'Is this CUDA kernel correct?'
utils/GPUVerifyRise4Fun/meta_data.py:    self.metadata['DisplayName'] += '-CUDA'
utils/GPUVerifyRise4Fun/meta_data.py:    self.metadata['Name'] += 'CUDA'
utils/GPUVerifyRise4Fun/meta_data.py:    self.loadLanguageSyntax(cuda.syntax)
utils/GPUVerifyRise4Fun/meta_data.py:    _logging.debug("Generated CUDA metadata:\n" + pprint.pformat(self.metadata))
utils/GPUVerifyRise4Fun/cuda/syntax.py:This file provides the CUDA C syntax definition for Rise4Fun web service
utils/GPUVerifyRise4Fun/cuda/samples/simple_barrier_divergence.cu: * which is what is required in CUDA.
utils/GPUVerifyRise4Fun/README:ABOUT GPUVerifyRise4Fun 
utils/GPUVerifyRise4Fun/README:online tool platform to allow use of the GPUVerify tool.
utils/GPUVerifyRise4Fun/README:- GPUVerify (development or deployed version)
utils/GPUVerifyRise4Fun/README:GPUVerifyRise4Fun has two different servers. A development web server (built in
utils/GPUVerifyRise4Fun/README:Right now there are two clients for GPUVerifyRise4Fun
utils/GPUVerifyRise4Fun/README:  on your machine and use the GPUVerifyRise4Fun service to verify it.
utils/GPUVerifyRise4Fun/README:* The web clients http://rise4fun.com/GPUVerify-OpenCL and
utils/GPUVerifyRise4Fun/README:  http://rise4fun.com/GPUVerify-CUDA
utils/GPUVerifyRise4Fun/README:loading all the sample kernels and asking the GPUVerifyRise4Fun service to
utils/GPUVerifyRise4Fun/README:CUDA kernels live in cuda/samples and OpenCL kernels live in opencl/samples.
utils/GPUVerifyRise4Fun/README:Tutorials for CUDA live in cuda/tutorial and tutorials for opencl live in
utils/GPUVerifyRise4Fun/README:opencl/tutorial. Each tutorial should be placed in its own folder along with any
utils/GPUVerifyRise4Fun/README:The "service/" folder contains scripts for integrating GPUVerifyRise4Fun into
utils/GPUVerifyRise4Fun/README:The GPUVerifyRise4Fun service implements the observer design pattern that
utils/GPUVerifyRise4Fun/README:This could be used for example to save a kernel that causes GPUVerify to
utils/GPUVerifyRise4Fun/README:   implement a class that descends from gvapi.GPUVerifyObserver that implements
Documentation/trouble_shooting.rst:If you think you have found a bug in GPUVerify, please report it via
Documentation/trouble_shooting.rst:our issue tracker on GitHub: https://github.com/mc-imperial/gpuverify/issues
Documentation/trouble_shooting.rst:when getting started with GPUVerify.
Documentation/trouble_shooting.rst:When running GPUVerify you see an error of the form::
Documentation/trouble_shooting.rst:     __main__.ConfigurationError: GPUVerify: CONFIGURATION_ERROR error (9): psutil required. `pip install psutil` to get it.
Documentation/trouble_shooting.rst:GPUVerify requires the python module `psutil <https://github.com/giampaolo/psutil>`_.
Documentation/trouble_shooting.rst:.. todo:: on Linux are all dynamic libraries installed that z3 and cvc4 were built against? Invoke each solver directly from the command line, or pass `--cruncher-opt=/proverOpt:VERBOSITY=100` to GPUVerify.
Documentation/advanced_features.rst:You can write assertions in GPU kernels using the special function
Documentation/advanced_features.rst:``__assert``.  Such assertions will be checked by GPUVerify.  For
Documentation/advanced_features.rst:example, consider the following CUDA kernel:
Documentation/advanced_features.rst:If we run GPUVerify specifying ``blockDim=8`` and ``gridDim=8`` then the tool verifies that the assertion cannot fail.  However, for large block and grid sizes it can fail::
Documentation/advanced_features.rst:  gpuverify --blockDim=64 --gridDim=64 failing-assert.cu
Documentation/advanced_features.rst:GPUVerify may be unable to verify a kernel due to limitations in
Documentation/advanced_features.rst:is treated by GPUVerify as a user-supplied loop invariant.  For
Documentation/advanced_features.rst:example, in the following (dumb) OpenCL kernel:
Documentation/advanced_features.rst:GPUVerify detects this potential problem::
Documentation/advanced_features.rst:  gpuverify --local_size=128 --num_groups=16 assert-as-invariant.cl
Documentation/advanced_features.rst:For readability, one can write ``__invariant`` as a synonym for ``__assert``.  This is recommended to state the intention that a given assertion is an invariant rather than a plain assertion, but currently GPUVerify does not check that this intention is satisifed.  In particular, if ``__invariant`` is used somewhere other than a loop head then it is treated as a plain assertion.
Documentation/advanced_features.rst:as reported by GPUVerify::
Documentation/advanced_features.rst:  gpuverify --local_size=1024 --num_groups=16 invariant-for-loop.cl
Documentation/advanced_features.rst:While programming using ``goto`` is not recommended, the ``goto`` keyword may be used in OpenCL.  If you write a loop using ``goto`` and wish to specify invariants for this loop then the rule is the same as usual: place invariants at the loop head.  This is illustrated for a ``goto`` version of the ``do-while`` example as follows:
Documentation/advanced_features.rst:Sometimes a kernel is correct only for certain input parameter values.  For example, the following CUDA kernel is not correct when executed by a single block of 1024 threads for *arbitrary* values of parameter ``sz``:
Documentation/advanced_features.rst:large that overflow can occur.  GPUVerify therefore rejects the
Documentation/advanced_features.rst:With this annotation, GPUVerify is able to verify race-freedom.
Documentation/advanced_features.rst:Inspecting invariants generated by GPUVerify
Documentation/advanced_features.rst:If you are looking to extend GPUVerify's invariant inference capabilities, it can be useful to look at the invariants the tool is generating.
Documentation/advanced_features.rst:To do this, run GPUVerify with the :ref:`keep-temps` option.  If you just want to see the candidate invariants the tool generates, and do not wish for verification to be attempted, you can use :ref:`stop-at-bpl` to tell GPUVerify to stop once is has generated the Boogie program that contains candidate invariants.
Documentation/advanced_features.rst:If you look in this file and search for ``:tag``, you may find some assertions that have the ``:tag`` attribute.  This attribute is a string indicating which invariant generation rule inside the ``GPUVerifyVCGen`` component of GPUVerify led to the candidate invariant being generated.
Documentation/advanced_features.rst:  testsuite/OpenCL/test_mod_invariants/global_direct/kernel.cl
Documentation/advanced_features.rst:    assert {:tag "user"} {:originated_from_invariant} {:line 12} {:col 7} {:fname "kernel.cl"} {:dir "/Users/nafe/work/autobuild/mac/gpuverify/testsuite/OpenCL/test_mod_invariants/global_direct"} {:thread 1} p0$1 ==> _c0 ==> (if $i.0$1 == v0$1 then 1bv1 else 0bv1) != 0bv1;
Documentation/advanced_features.rst:    assert {:tag "user"} {:originated_from_invariant} {:line 12} {:col 7} {:fname "kernel.cl"} {:dir "/Users/nafe/work/autobuild/mac/gpuverify/testsuite/OpenCL/test_mod_invariants/global_direct"} {:thread 2} p0$2 ==> _c0 ==> (if $i.0$2 == v0$2 then 1bv1 else 0bv1) != 0bv1;
Documentation/advanced_features.rst:Watching GPUVerify eliminate invariants
Documentation/advanced_features.rst:To see this kicking out of candidates in action, you can run GPUVerify with :ref:`verbose` and with ``--boogie-opt=/trace`` (see :ref:`boogie-opt`).  If you try this on::
Documentation/advanced_features.rst:  testsuite/OpenCL/test_mod_invariants/global_direct/kernel.cl
Documentation/developer_guide.rst:Building GPUVerify
Documentation/developer_guide.rst:The GPUVerify toolchain is a pipeline that uses different components.
Documentation/developer_guide.rst:GPUVerify requires python 3 and the python module `psutil <https://github.com/giampaolo/psutil>`_.
Documentation/developer_guide.rst:In addition to the common prerequisites Linux and Mac OS X builds of GPUVerify
Documentation/developer_guide.rst:To build GPUVerify follow this guide in a bash shell.
Documentation/developer_guide.rst:Note ``${BUILD_ROOT}`` refers to the location where you wish to build GPUVerify.
Documentation/developer_guide.rst:#. Get the LLVM and Clang sources (note that GPUVerify depends on LLVM 6.0)::
Documentation/developer_guide.rst:   Make a symbolic link; ``GPUVerify.py`` looks for ``z3.exe`` not ``z3``
Documentation/developer_guide.rst:   Make a symbolic link; ``GPUVerify.py`` looks for ``cvc4.exe`` not ``cvc4``
Documentation/developer_guide.rst:#. Get GPUVerify and build::
Documentation/developer_guide.rst:     $ git clone https://github.com/mc-imperial/gpuverify.git
Documentation/developer_guide.rst:     $ cd ${BUILD_ROOT}/gpuverify
Documentation/developer_guide.rst:     $ nuget restore GPUVerify.sln
Documentation/developer_guide.rst:               GPUVerify.sln
Documentation/developer_guide.rst:#. Configure GPUVerify front end.
Documentation/developer_guide.rst:   GPUVerify uses a front end python script (GPUVerify.py). This script needs
Documentation/developer_guide.rst:     $ cd ${BUILD_ROOT}/gpuverify
Documentation/developer_guide.rst:      # The path to the directory containing the GPUVerify binaries.
Documentation/developer_guide.rst:      # GPUVerifyVCGen.exe, GPUVerifyCruncher.exe and GPUVerifyBoogieDriver.exe should be there
Documentation/developer_guide.rst:      gpuVerifyBinDir = rootDir + "/gpuverify/Binaries"
Documentation/developer_guide.rst:    $ cd ${BUILD_ROOT}/gpuverify/Documentation
Documentation/developer_guide.rst:#. Run the GPUVerify test suite.
Documentation/developer_guide.rst:     $ cd ${BUILD_ROOT}/gpuverify
Documentation/developer_guide.rst:   To run the GPUVerify test suite using the CVC4 SMT Solver:
Documentation/developer_guide.rst:In addition to the common prerequisites a Windows build of GPUVerify requires
Documentation/developer_guide.rst:To build GPUVerify follow this guide in a powershell window.
Documentation/developer_guide.rst:Note ``${BUILD_ROOT}`` refers to where ever you wish to build GPUVerify.
Documentation/developer_guide.rst:We recommend that you build GPUVerify to a local hard drive like ``C:``
Documentation/developer_guide.rst:#. Get the LLVM and Clang sources (note that GPUVerify depends LLVM 6.0)::
Documentation/developer_guide.rst:   GPUVerify website and unzip this in ``${BUILD_ROOT}``. From the command
Documentation/developer_guide.rst:#. Get GPUVerify and build. You can do this by opening ``GPUVerify.sln``
Documentation/developer_guide.rst:      > git clone https://github.com/mc-imperial/gpuverify.git
Documentation/developer_guide.rst:      > cd ${BUILD_ROOT}\gpuverify
Documentation/developer_guide.rst:      > ${BUILD_ROOT}\nuget restore GPUVerify.sln
Documentation/developer_guide.rst:                GPUVerify.sln
Documentation/developer_guide.rst:#. Configure GPUVerify front end::
Documentation/developer_guide.rst:     > cd ${BUILD_ROOT}\gpuverify
Documentation/developer_guide.rst:      # The path to the directory containing the GPUVerify binaries.
Documentation/developer_guide.rst:      # GPUVerifyVCGen.exe, GPUVerifyCruncher.exe and GPUVerifyBoogieDriver.exe should be there
Documentation/developer_guide.rst:      gpuVerifyBinDir = rootDir + r"\gpuverify\Binaries"
Documentation/developer_guide.rst:    $ cd ${BUILD_ROOT}\gpuverify\Documentation
Documentation/developer_guide.rst:#. Run the GPUVerify test suite.
Documentation/developer_guide.rst:     $ cd ${BUILD_ROOT}\gpuverify
Documentation/developer_guide.rst:   To run the GPUVerify test suite using the CVC4 SMT Solver:
Documentation/developer_guide.rst:Deploying GPUVerify
Documentation/developer_guide.rst:To deploy a stand alone version of GPUVerify run::
Documentation/developer_guide.rst:  $ mkdir -p /path/to/deploy/gpuverify
Documentation/developer_guide.rst:  $ cd ${BUILD_ROOT}/gpuverify
Documentation/developer_guide.rst:  $ ./deploy.py /path/to/deploy/gpuverify
Documentation/developer_guide.rst:This will copy the necessary files to run a standalone copy of GPUVerify in an
Documentation/developer_guide.rst:  the deploy folder as ``gvfindtools.py`` for ``GPUVerify.py`` to use.
Documentation/developer_guide.rst:The GPUVerify repository has a pre-built version of Boogie inside it to make
Documentation/developer_guide.rst:in GPUVerify then follow the steps below for Linux and Mac OS X.::
Documentation/developer_guide.rst:      $ ls ${BUILD_ROOT}/gpuverify/BoogieBinaries \
Documentation/developer_guide.rst:             | xargs -I{} -t cp {} ${BUILD_ROOT}/gpuverify/BoogieBinaries
Documentation/developer_guide.rst:GPUVerify uses a python script ``gvtester.py`` to instrument the
Documentation/developer_guide.rst:GPUVerify.py front-end script with a series of tests. These tests are located in
Documentation/developer_guide.rst:Each test is a file named ``kernel.cu`` or ``kernel.cl`` (for CUDA and OpenCL
Documentation/developer_guide.rst:                |  "GPUVERIFYVCGEN_ERROR"
Documentation/developer_guide.rst:``GPUVerify.py``.
Documentation/developer_guide.rst:``GPUVerify.py``. ``<gv-arg>`` is a single ``GPUVerify.py`` command line
Documentation/developer_guide.rst:arguments. The path to the kernel for ``GPUVerify.py`` is implicitly passed as
Documentation/developer_guide.rst:the last command line argument to ``GPUVerify.py`` so it should **not** be
Documentation/developer_guide.rst:against the output of ``GPUVerify.py`` if ``GPUVerify.py``'s return code is not
Documentation/developer_guide.rst:    //GPUVerify:[ ]+error:[ ]*
Documentation/developer_guide.rst:    //GPUVerify: Try --help for list of options
Documentation/developer_guide.rst:broken anything. If you modify something in GPUVerify or add a new test you
Documentation/developer_guide.rst:- ``/home/person/gpuverify/testsuite/OpenCL/typestest``
Documentation/developer_guide.rst:- ``c:\program files\gpuverify\testsuite\OpenCL\typestest``
Documentation/developer_guide.rst:``testsuite/OpenCL/typestest`` so they are considered the same test and are
Documentation/developer_guide.rst:Adding additional GPUVerify error codes
Documentation/developer_guide.rst:``gvtester.py`` directly imports the GPUVerify codes so that it is aware of the
Documentation/developer_guide.rst:between the GPUVerify error codes and REGEX_MISMATCH_ERROR.
Documentation/developer_guide.rst:``GPUVerifyScript/error_codes.py``. Make sure your new error code has a value
Documentation/basic_usage.rst:Running GPUVerify
Documentation/basic_usage.rst:OpenCL
Documentation/basic_usage.rst:To invoke GPUVerify on an OpenCL kernel, do::
Documentation/basic_usage.rst:  gpuverify --local_size=<work-group-dimensions> --num_groups=<grid-dimensions> <OpenCL file>
Documentation/basic_usage.rst:Here, ``<work-group-dimensions>`` is a vector specifying the dimensionality of each work group, ``<grid-dimensions>`` is a vector specifying the dimensionality of the grid of work groups, and ``<OpenCL file>`` is an OpenCL file with extension ``.cl``.
Documentation/basic_usage.rst:  gpuverify --num_groups=[16,16] --local_size=[32,32] kernel.cl
Documentation/basic_usage.rst:  gpuverify --num_groups=1 --local_size=[32,32] kernel.cl
Documentation/basic_usage.rst:  gpuverify --num_groups=[1] --local_size=[32,32] kernel.cl
Documentation/basic_usage.rst:to the ``global_work_size`` and ``local_work_size`` parameters to the OpenCL
Documentation/basic_usage.rst:  gpuverify --local_size=[32,32] --global_size=[512,512] kernel.cl
Documentation/basic_usage.rst:  gpuverify --local_size=[32,32] --num_groups=[16,16] kernel.cl
Documentation/basic_usage.rst:CUDA
Documentation/basic_usage.rst:To invoke GPUVerify on a CUDA kernel, do::
Documentation/basic_usage.rst:  gpuverify --blockDim=<block-dimensions> --gridDim=<grid-dimensions> <CUDA file>
Documentation/basic_usage.rst:Here, ``<block-dimensions>`` is a vector specifying the dimensionality of each thread block, ``<grid-dimensions>`` is a vector specifying the dimensionality of the grid of thread blocks, and ``<CUDA file>`` is a CUDA file with extension ``.cu``.
Documentation/basic_usage.rst:  gpuverify --gridDim=[16,16] --blockDim=[32,32] kernel.cu
Documentation/basic_usage.rst:  gpuverify --gridDim=1 --blockDim=[32,32] kernel.cu
Documentation/basic_usage.rst:  gpuverify --gridDim=[1] --blockDim=[32,32] kernel.cu
Documentation/basic_usage.rst:By default, GPUVerify runs in verify mode. In this mode, the tool will
Documentation/basic_usage.rst:verified as free from the types of defects which GPUVerify can check
Documentation/basic_usage.rst:for. This verification result can be trusted, modulo bugs in GPUVerify
Documentation/basic_usage.rst:In verify mode, any defects reported by GPUVerify are **possible**
Documentation/basic_usage.rst:The ``--findbugs`` flag causes GPUVerify to run in *findbugs*
Documentation/basic_usage.rst:If GPUVerify reports that no defects were found when running in
Documentation/basic_usage.rst:loops are not abstracted using invariants. However, GPUVerify can
Documentation/basic_usage.rst:  example, if ``x`` is a ``float`` variable, GPUVerify does
Documentation/basic_usage.rst:Properties checked by GPUVerify
Documentation/basic_usage.rst:We now describe the key properties of GPU kernels that GPUVerify
Documentation/basic_usage.rst:checks, giving a mixture of small OpenCL and CUDA kernels as examples.
Documentation/basic_usage.rst:* OpenCL: an intra-group data race is a race between work items
Documentation/basic_usage.rst:* CUDA: an intra-group data race is a
Documentation/basic_usage.rst:Suppose the following OpenCL kernel is executed by a single work group
Documentation/basic_usage.rst:we run GPUVerify on the example::
Documentation/basic_usage.rst:  gpuverify --local_size=1024 --num_groups=1 intra-group.cl
Documentation/basic_usage.rst:* OpenCL: an inter-group data race is a race between work items in
Documentation/basic_usage.rst:* CUDA: an inter-group data race is a race between threads in different thread blocks.
Documentation/basic_usage.rst:Suppose the following CUDA kernel is executed by 8 thread blocks each
Documentation/basic_usage.rst:  1  #include <cuda.h>
Documentation/basic_usage.rst:intra-block thread indices. If we run GPUVerify on the example::
Documentation/basic_usage.rst:  gpuverify --blockDim=64 --gridDim=8 inter-group.cu
Documentation/basic_usage.rst:GPUVerify detects cases where a kernel breaks the rules for barrier synchronization in conditional code defined in the CUDA and OpenCL documentation. In particular, the tool checks that if a barrier occurs in a conditional statement then all threads must evaluate the condition uniformly, and if a barrier occurs inside a loop then all threads must execute the same number of loop iterations before synchronizing at the barrier.
Documentation/basic_usage.rst:GPUVerify rejects the following OpenCL kernel, executed by a single
Documentation/basic_usage.rst:  gpuverify --local_size=1024 --num_groups=1 barrier-div-opencl.cl
Documentation/basic_usage.rst:GPUVerify rejects the following CUDA kernel when, say, executed by a
Documentation/basic_usage.rst:  1  #include <cuda.h>
Documentation/basic_usage.rst:  gpuverify --blockDim=[16,16] --gridDim=[32,32] barrier-div-cuda.cu
Documentation/basic_usage.rst:  barrier-div-cuda.cu:6:5: error: barrier may be reached by non-uniform control flow
Documentation/basic_usage.rst:In the description of command line options, we follow OpenCL terminology, not CUDA terminology.  We thus refer to work items and work groups, not threads and thread blocks, and to local memory, not shared memory.
Documentation/basic_usage.rst:Display list of GPUVerify options.  Please report cases where GPUVerify claims to have an option not documented here, or if an option mentioned here is not listed by GPUVerify.
Documentation/basic_usage.rst:Run GPUVerify in *verify* mode (see :ref:`verifymode`).  This is the mode the tool uses by default.
Documentation/basic_usage.rst:By default, GPUVerify tries to tolerate certain kinds of (arguably) *benign* data races.  For example, if GPUVerify can figure out that in a write-write data race, both work items involved are guaranteed to write the same value to the memory location in question, it will not report the race.
Documentation/basic_usage.rst:Do not check for inter-work-group races.  In this mode, a kernel may be deemed correct even if it can exhibit races on global memory between work items in different work groups, as long as GPUVerify can prove that there are no data races (on global or local memory) between work items in the same work group.
Documentation/basic_usage.rst:With this option, GPUVerify will print the various sub-commands that are issued during the analysis process.  Also, output produced by the tools which GPUVerify invokes will be displayed.  If you are debugging, and are issuing print statements in one of the GPUVerify components, you will need to use ``--verbose`` to be able to see the results of this printing.
Documentation/basic_usage.rst:When GPUVerify finishes, print statistics about timing.
Documentation/basic_usage.rst:``--opencl``
Documentation/basic_usage.rst:Assume the kernel to verify is an OpenCL kernel. By default GPUVerify tries to detect whether the kernel to be verified is an OpenCL or CUDA kernel based on file extension and file contents. When detection fails, the kernel type can be explicitly specified to be OpenCL by passing this option.
Documentation/basic_usage.rst:``--cuda``
Documentation/basic_usage.rst:Assume the kernel to verify is a CUDA kernel. Similar to the ``--opencl`` option, this option can be used to explicitly specify the kernel type to be CUDA.
Documentation/basic_usage.rst:OpenCL-specific options
Documentation/basic_usage.rst:CUDA-specific options
Documentation/basic_usage.rst:In some cases this may speed-up verification. However, it is more likely that verification will fail, as GPUVerify's procedure contract inference capabilities are rather limited.
Documentation/basic_usage.rst:Normally, GPUVerify suppresses exceptions, dumping them to a file and printing a standard "internal error" message.  This option turns off this suppression.
Documentation/basic_usage.rst:Use this option to pass a command-line option directly to Clang, the front-end used by GPUVerify.
Documentation/basic_usage.rst:Use this option to pass a command-line option directly to opt, the LLVM optimizer used by GPUVerify.
Documentation/basic_usage.rst:Use this to pass a command-line option directly to Bugle, the component of GPUVerify that translates LLVM bitcode into Boogie.
Documentation/basic_usage.rst:Specify an option to be passed directly to Boogie.  For instance, if you want to see what Boogie is doing, you can use ``--boogie-opt=/trace``.  In this case you also need to pass :ref:`verbose` to GPUVerify.
Documentation/make.bat:	echo.^> qcollectiongenerator %BUILDDIR%\qthelp\GPUVerify.qhcp
Documentation/make.bat:	echo.^> assistant -collectionFile %BUILDDIR%\qthelp\GPUVerify.ghc
Documentation/index.rst:.. GPUVerify documentation master file, created by
Documentation/index.rst:Welcome to GPUVerify's documentation
Documentation/Makefile:	@echo "# qcollectiongenerator $(BUILDDIR)/qthelp/GPUVerify.qhcp"
Documentation/Makefile:	@echo "# assistant -collectionFile $(BUILDDIR)/qthelp/GPUVerify.qhc"
Documentation/Makefile:	@echo "# mkdir -p $$HOME/.local/share/devhelp/GPUVerify"
Documentation/Makefile:	@echo "# ln -s $(BUILDDIR)/devhelp $$HOME/.local/share/devhelp/GPUVerify"
Documentation/tutorial.rst:A very brief introduction to GPUVerify
Documentation/tutorial.rst:GPUVerify quickly.
Documentation/tutorial.rst:For more detailed instructions on using GPUVerify take a look at:
Documentation/tutorial.rst:- GPUVerify documentation:
Documentation/tutorial.rst:  http://multicore.doc.ic.ac.uk/tools/GPUVerify/docs/index.html
Documentation/tutorial.rst:* Python 2.7 or higher.  GPUVerify is coordinated by a Python script,
Documentation/tutorial.rst:* psutil.  GPUVerify relies on the psutil Python module.
Documentation/tutorial.rst:  of mono installed.  Please see the GPUVerify documentation for details of how
Documentation/tutorial.rst:  http://multicore.doc.ic.ac.uk/tools/GPUVerify/docs/developer_guide.html
Documentation/tutorial.rst:Running GPUVerify on some kernels
Documentation/tutorial.rst:Before you extract the GPUVerify archive you need to "unblock" it.
Documentation/tutorial.rst:**Adding GPUVerify to your path**
Documentation/tutorial.rst:After extracting the GPUVerify download, you should add the directory
Documentation/tutorial.rst:into which you have extracted GPUVerify to your ``PATH`` environment
Documentation/tutorial.rst:variable.  This directory contains the ``GPUVerify.py`` script.
Documentation/tutorial.rst:  testsuite/OpenCL/misc/fail/2d_array_race
Documentation/tutorial.rst:This is one of several hundred kernels that comprise the GPUVerify
Documentation/tutorial.rst:The top three comment lines are there for GPUVerify's testing tool and
Documentation/tutorial.rst:  GPUVerify --local_size=64,64 --num_groups=256,256 kernel.cl
Documentation/tutorial.rst:After a few seconds GPUVerify reports some data races for this kernel.
Documentation/tutorial.rst:GPUVerify has identified that two work items in work group (138, 0)
Documentation/tutorial.rst:  testsuite/OpenCL/async_work_group_copy/pass/test1
Documentation/tutorial.rst:and take a look at kernel.cl.  This example make use of OpenCL
Documentation/tutorial.rst:  GPUVerify --local_size=64 --num_groups=128 kernel.cl
Documentation/tutorial.rst:  GPUVerify kernel analyser finished with 1 verified, 0 errors
Documentation/tutorial.rst:"GPUVerify requires Python to be equipped with the psutil module."
Documentation/tutorial.rst:**Did you unblock the GPUVerify .zip file before extracting it?**
Documentation/tutorial.rst:then you forgot to unblock the GPUVerify .zip file before you
Documentation/tutorial.rst:**GPUVerify is taking a long time to give an answer for my kernel**
Documentation/tutorial.rst:Try running GPUVerify with the --infer-info option.  With this option
Documentation/tutorial.rst:If you observe this behaviour then GPUVerify is slowly but surely
Documentation/tutorial.rst:message then it could be that GPUVerify is stuck solving a very
Documentation/tutorial.rst:  if you know details of the values "width", "height" and "blockSize" should take then you can communicate these to GPUVerify as preconditions, using __requires clauses:
Documentation/tutorial.rst:  GPUVerify tends to do much better during verification when the
Documentation/tutorial.rst:  GPUVerify often succeeds in verifying kernels executed by very large
Documentation/tutorial.rst:**GPUVerify says that my kernel is incorrect, but it's not!**
Documentation/tutorial.rst:GPUVerify is a sound static verifier.  A result of this is that the
Documentation/tutorial.rst:If this issue is affecting your use of GPUVerify then please get in
Documentation/tutorial.rst:touch with the GPUVerify development team.  We can explain how you can
Documentation/tutorial.rst:extend GPUVerify so that it can handle kernels similar to yours fully
Documentation/json_format.rst:The JSON file format accepted by GPUVerify has the following form::
Documentation/json_format.rst:    "language"         : "OpenCL",
Documentation/json_format.rst:The value of ``language`` should always be the string ``OpenCL``.
Documentation/json_format.rst:The value of ``kernel_file`` is the name of an OpenCL source file
Documentation/json_format.rst:relevant OpenCL host-side functions that were invoked leading up to
Documentation/json_format.rst:GPUVerify). When ``global_offset`` is not specified, the offset is assumed to
Documentation/conf.py:# GPUVerify documentation build configuration file, created by
Documentation/conf.py:project = u'GPUVerify'
Documentation/conf.py:htmlhelp_basename = 'GPUVerifydoc'
Documentation/conf.py:  ('index', 'GPUVerify.tex', u'GPUVerify Documentation',
Documentation/conf.py:    ('index', 'gpuverify', u'GPUVerify Documentation',
Documentation/conf.py:  ('index', 'GPUVerify', u'GPUVerify Documentation',
Documentation/conf.py:   u'Multicore Programming Group, Imperial College London', 'GPUVerify', 'One line description of project.',
Documentation/examples/barrier-div-cuda-error.txt:barrier-div-cuda.cu:6:5: error: barrier may be reached by non-uniform control flow
Documentation/examples/barrier-div-cuda.cu:#include <cuda.h>
Documentation/examples/inter-group.cu:#include <cuda.h>
Documentation/examples/with-requires.cu:#include <cuda.h>
Documentation/examples/failing-assert.cu:#include <cuda.h>
Documentation/examples/needs-requires.cu:#include <cuda.h>
Documentation/installation.rst:Getting GPUVerify
Documentation/installation.rst:Prebuilt versions of GPUVerify are available as:
Documentation/installation.rst:These can be found on the GPUVerify `Download <http://multicore.doc.ic.ac.uk/tools/GPUVerify/download.php>`_ page.
Documentation/installation.rst:GPUVerify requires python 3 and the python module `psutil <https://github.com/giampaolo/psutil>`_.
Documentation/installation.rst:To install GPUVerify follow this guide in a bash shell.
Documentation/installation.rst:Note ``${INSTALL_ROOT}`` refers to where ever you wish to install GPUVerify.
Documentation/installation.rst:#. Download the Linux 64-bit toolchain zip file from the GPUVerify `Download <http://multicore.doc.ic.ac.uk/tools/GPUVerify/download.php>`_ page.
Documentation/installation.rst:      $ unzip GPUVerifyLinux64-nightly.zip
Documentation/installation.rst:   This should unpack the GPUVerify toolchain into a path like ``2013-06-11``, which is the date that the tool was packaged.
Documentation/installation.rst:#. Finally, run the GPUVerify test suite.::
Documentation/installation.rst:     $ cd ${INSTALL_ROOT}/2013-06-11/gpuverify
Documentation/installation.rst:To install GPUVerify follow this guide in a powershell window.
Documentation/installation.rst:Note ``${INSTALL_ROOT}`` refers to where ever you wish to build GPUVerify.
Documentation/installation.rst:We recommend that you install GPUVerify to a local hard drive like ``C:``
Documentation/installation.rst:#. Download the Windows 64-bit toolchain zip file from the GPUVerify `Download <http://multicore.doc.ic.ac.uk/tools/GPUVerify/download.php>`_ page.
Documentation/installation.rst:      > unzip GPUVerifyWindows64-nightly.zip
Documentation/installation.rst:   This should unpack the GPUVerify toolchain into a path like ``2013-06-11``, which is the date that the tool was packaged.
Documentation/installation.rst:#. Finally, run the GPUVerify test suite.::
Documentation/installation.rst:      > cd ${INSTALL_ROOT}\2013-06-11\gpuverify
Documentation/limitations.rst:Known Sources of Unsoundness in GPUVerify
Documentation/limitations.rst:* By defualt, GPUVerify does not perform any out-of-bounds checking. Hence, a
Documentation/limitations.rst:* GPUVerify assumes the input arrays to a kernel are all distinct. Not making
Documentation/limitations.rst:  by equipping the input arrays with restrict qualifiers. GPUVerify will issue
Documentation/limitations.rst:  input array. For example, for the the following CUDA kernel::
Documentation/limitations.rst:  GPUVerify will report it is assuming the the arguments ``a`` and ``b`` of
Documentation/limitations.rst:* GPUVerify's default pointer representation may cause false negatives to occur
Documentation/limitations.rst:  ``NULL``). For example, the following CUDA kernel will successfully verify::
Documentation/limitations.rst:* GPUVerify assumes that atomic operations do not overflow or underflow the
Documentation/limitations.rst:  annotation always evaluates to false. As a consequence, the following OpenCL
Documentation/overview.rst:Overview of GPUVerify
Documentation/overview.rst:GPUVerify is a tool for analysing graphics processing unit (GPU) kernels written in OpenCL and CUDA.
Documentation/overview.rst:GPUVerify is a *static analysis* tool, that is it works purely at
Documentation/overview.rst:compile time, without actually running the GPU kernel.

```
