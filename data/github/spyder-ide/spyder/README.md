# https://github.com/spyder-ide/spyder

```console
changelogs/Spyder-3.md:* [PR 4601](https://github.com/spyder-ide/spyder/pull/4601) - PR: Add pyopengl to setup.py to fix errors with some Nvidia/Intel drivers
changelogs/Spyder-6.md:* [PR 22575](https://github.com/spyder-ide/spyder/pull/22575) - PR: Add harmless OpenCL warning to bening errors (IPython console), by [@ccordoba12](https://github.com/ccordoba12) ([22551](https://github.com/spyder-ide/spyder/issues/22551))
changelogs/Spyder-2.md:    * syntax highlighting: added support for OpenCL, gettext files, patch/diff files, CSS and HTML files
spyder/plugins/completion/api.py:    'openClose': True,
spyder/plugins/ipythonconsole/widgets/client.py:            # Harmless warning from OpenCL on Windows.
spyder/plugins/help/utils/js/mathjax/jax/input/MathML/entities/c.js:(function(a){MathJax.Hub.Insert(a.Parse.Entity,{CHcy:"\u0427",COPY:"\u00A9",Cacute:"\u0106",CapitalDifferentialD:"\u2145",Cayleys:"\u212D",Ccaron:"\u010C",Ccedil:"\u00C7",Ccirc:"\u0108",Cconint:"\u2230",Cdot:"\u010A",Cedilla:"\u00B8",Chi:"\u03A7",ClockwiseContourIntegral:"\u2232",CloseCurlyDoubleQuote:"\u201D",CloseCurlyQuote:"\u2019",Colon:"\u2237",Colone:"\u2A74",Conint:"\u222F",CounterClockwiseContourIntegral:"\u2233",cacute:"\u0107",capand:"\u2A44",capbrcup:"\u2A49",capcap:"\u2A4B",capcup:"\u2A47",capdot:"\u2A40",caps:"\u2229\uFE00",caret:"\u2041",caron:"\u02C7",ccaps:"\u2A4D",ccaron:"\u010D",ccedil:"\u00E7",ccirc:"\u0109",ccups:"\u2A4C",ccupssm:"\u2A50",cdot:"\u010B",cedil:"\u00B8",cemptyv:"\u29B2",cent:"\u00A2",centerdot:"\u00B7",chcy:"\u0447",checkmark:"\u2713",cir:"\u25CB",cirE:"\u29C3",cire:"\u2257",cirfnint:"\u2A10",cirmid:"\u2AEF",cirscir:"\u29C2",clubsuit:"\u2663",colone:"\u2254",coloneq:"\u2254",comma:"\u002C",commat:"\u0040",compfn:"\u2218",complement:"\u2201",complexes:"\u2102",cong:"\u2245",congdot:"\u2A6D",conint:"\u222E",coprod:"\u2210",copy:"\u00A9",copysr:"\u2117",crarr:"\u21B5",cross:"\u2717",csub:"\u2ACF",csube:"\u2AD1",csup:"\u2AD0",csupe:"\u2AD2",cudarrl:"\u2938",cudarrr:"\u2935",cularrp:"\u293D",cupbrcap:"\u2A48",cupcap:"\u2A46",cupcup:"\u2A4A",cupdot:"\u228D",cupor:"\u2A45",cups:"\u222A\uFE00",curarrm:"\u293C",curlyeqprec:"\u22DE",curlyeqsucc:"\u22DF",curren:"\u00A4",curvearrowleft:"\u21B6",curvearrowright:"\u21B7",cuvee:"\u22CE",cuwed:"\u22CF",cwconint:"\u2232",cwint:"\u2231",cylcty:"\u232D"});MathJax.Ajax.loadComplete(a.entityDir+"/c.js")})(MathJax.InputJax.MathML);
spyder/plugins/help/utils/js/mathjax/extensions/a11y/mathjax-sre.js:this.invisibleComma_=sre.SemanticUtil.numberToUnicode(8291);this.commas=[",",this.invisibleComma_];this.ellipses="\u2026\u22ee\u22ef\u22f0\u22f1\ufe19".split("");this.fullStops=[".","\ufe52","\uff0e"];this.dashes="\u2012\u2013\u2014\u2015\u301c\ufe31\ufe32\ufe58".split("");this.primes="'\u2032\u2033\u2034\u2035\u2036\u2037\u2057".split("");this.openClosePairs={"(":")","[":"]","{":"}","\u2045":"\u2046","\u2329":"\u232a","\u2768":"\u2769","\u276a":"\u276b","\u276c":"\u276d","\u276e":"\u276f","\u2770":"\u2771",
spyder/plugins/help/utils/js/mathjax/extensions/a11y/mathjax-sre.js:"\u23a2":"\u23a5","\u23a3":"\u23a6","\u23a7":"\u23ab","\u23a8":"\u23ac","\u23a9":"\u23ad","\u23b0":"\u23b1","\u23b8":"\u23b9"};this.topBottomPairs={"\u23b4":"\u23b5","\u23dc":"\u23dd","\u23de":"\u23df","\u23e0":"\u23e1","\ufe35":"\ufe36","\ufe37":"\ufe38","\ufe39":"\ufe3a","\ufe3b":"\ufe3c","\ufe3d":"\ufe3e","\ufe3f":"\ufe40","\ufe41":"\ufe42","\ufe43":"\ufe44","\ufe47":"\ufe48"};this.leftFences=sre.SemanticUtil.objectsToKeys(this.openClosePairs);this.rightFences=sre.SemanticUtil.objectsToValues(this.openClosePairs);
spyder/plugins/help/utils/js/mathjax/extensions/a11y/mathjax-sre.js:sre.SemanticAttr.prototype.isMatchingFence_=function(a,b){return-1!=this.neutralFences.indexOf(a)?a==b:this.openClosePairs[a]==b||this.topBottomPairs[a]==b};sre.SemanticAttr.prototype.initMeaning_=function(){for(var a={},b=0,c;c=this.symbolSetToSemantic_[b];b++)c.set.forEach(function(b){a[b]={role:c.role||sre.SemanticAttr.Role.UNKNOWN,type:c.type||sre.SemanticAttr.Type.UNKNOWN,font:c.font||sre.SemanticAttr.Font.UNKNOWN}});return a};
spyder/plugins/editor/utils/languages.py:    'OpenCL': ('cl',),
spyder/plugins/editor/widgets/codeeditor/lsp_mixin.py:        self.open_close_notifications = sync_options.get("openClose", False)
spyder/plugins/editor/widgets/codeeditor/codeeditor.py:        'OpenCL': (sh.OpenCLSH, '//'),
spyder/config/utils.py:    (_("OpenCL files"), ('.cl', )),
spyder/locale/pl/LC_MESSAGES/spyder.po:msgid "OpenCL files"
spyder/locale/pl/LC_MESSAGES/spyder.po:msgstr "Pliki OpenCL"
spyder/locale/fr/LC_MESSAGES/spyder.po:msgid "OpenCL files"
spyder/locale/fr/LC_MESSAGES/spyder.po:msgstr "Fichiers OpenCL"
spyder/locale/spyder.pot:msgid "OpenCL files"
spyder/locale/ja/LC_MESSAGES/spyder.po:msgid "OpenCL files"
spyder/locale/ja/LC_MESSAGES/spyder.po:msgstr "OpenCLファイル"
spyder/locale/es/LC_MESSAGES/spyder.po:msgid "OpenCL files"
spyder/locale/es/LC_MESSAGES/spyder.po:msgstr "Archivos OpenCL"
spyder/locale/ru/LC_MESSAGES/spyder.po:msgid "OpenCL files"
spyder/locale/ru/LC_MESSAGES/spyder.po:msgstr "Файлы OpenCL"
spyder/locale/de/LC_MESSAGES/spyder.po:msgid "OpenCL files"
spyder/locale/de/LC_MESSAGES/spyder.po:msgstr "OpenCL-Dateien"
spyder/locale/hr/LC_MESSAGES/spyder.po:msgid "OpenCL files"
spyder/locale/hu/LC_MESSAGES/spyder.po:msgid "OpenCL files"
spyder/locale/hu/LC_MESSAGES/spyder.po:msgstr "OpenCL fájlok"
spyder/locale/fa/LC_MESSAGES/spyder.po:msgid "OpenCL files"
spyder/locale/fa/LC_MESSAGES/spyder.po:msgstr "فایل های OpenCL"
spyder/locale/zh_CN/LC_MESSAGES/spyder.po:msgid "OpenCL files"
spyder/locale/zh_CN/LC_MESSAGES/spyder.po:msgstr "OpenCL 文件"
spyder/locale/te/LC_MESSAGES/spyder.po:msgid "OpenCL files"
spyder/locale/uk/LC_MESSAGES/spyder.po:msgid "OpenCL files"
spyder/locale/uk/LC_MESSAGES/spyder.po:msgstr "Файли OpenCL"
spyder/locale/pt_BR/LC_MESSAGES/spyder.po:msgid "OpenCL files"
spyder/locale/pt_BR/LC_MESSAGES/spyder.po:msgstr "Arquivos OpenCL"
spyder/utils/syntaxhighlighters.py:def make_opencl_patterns():
spyder/utils/syntaxhighlighters.py:    kwstr2 = 'CL_FALSE, CL_TRUE, CL_PLATFORM_PROFILE, CL_PLATFORM_VERSION, CL_PLATFORM_NAME, CL_PLATFORM_VENDOR, CL_PLATFORM_EXTENSIONS, CL_DEVICE_TYPE_DEFAULT , CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ACCELERATOR, CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE, CL_DEVICE_VENDOR_ID, CL_DEVICE_MAX_COMPUTE_UNITS, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, CL_DEVICE_MAX_WORK_GROUP_SIZE, CL_DEVICE_MAX_WORK_ITEM_SIZES, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, CL_DEVICE_MAX_CLOCK_FREQUENCY, CL_DEVICE_ADDRESS_BITS, CL_DEVICE_MAX_READ_IMAGE_ARGS, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, CL_DEVICE_MAX_MEM_ALLOC_SIZE, CL_DEVICE_IMAGE2D_MAX_WIDTH, CL_DEVICE_IMAGE2D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_WIDTH, CL_DEVICE_IMAGE3D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_DEPTH, CL_DEVICE_IMAGE_SUPPORT, CL_DEVICE_MAX_PARAMETER_SIZE, CL_DEVICE_MAX_SAMPLERS, CL_DEVICE_MEM_BASE_ADDR_ALIGN, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, CL_DEVICE_SINGLE_FP_CONFIG, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, CL_DEVICE_GLOBAL_MEM_SIZE, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, CL_DEVICE_MAX_CONSTANT_ARGS, CL_DEVICE_LOCAL_MEM_TYPE, CL_DEVICE_LOCAL_MEM_SIZE, CL_DEVICE_ERROR_CORRECTION_SUPPORT, CL_DEVICE_PROFILING_TIMER_RESOLUTION, CL_DEVICE_ENDIAN_LITTLE, CL_DEVICE_AVAILABLE, CL_DEVICE_COMPILER_AVAILABLE, CL_DEVICE_EXECUTION_CAPABILITIES, CL_DEVICE_QUEUE_PROPERTIES, CL_DEVICE_NAME, CL_DEVICE_VENDOR, CL_DRIVER_VERSION, CL_DEVICE_PROFILE, CL_DEVICE_VERSION, CL_DEVICE_EXTENSIONS, CL_DEVICE_PLATFORM, CL_FP_DENORM, CL_FP_INF_NAN, CL_FP_ROUND_TO_NEAREST, CL_FP_ROUND_TO_ZERO, CL_FP_ROUND_TO_INF, CL_FP_FMA, CL_NONE, CL_READ_ONLY_CACHE, CL_READ_WRITE_CACHE, CL_LOCAL, CL_GLOBAL, CL_EXEC_KERNEL, CL_EXEC_NATIVE_KERNEL, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, CL_QUEUE_PROFILING_ENABLE, CL_CONTEXT_REFERENCE_COUNT, CL_CONTEXT_DEVICES, CL_CONTEXT_PROPERTIES, CL_CONTEXT_PLATFORM, CL_QUEUE_CONTEXT, CL_QUEUE_DEVICE, CL_QUEUE_REFERENCE_COUNT, CL_QUEUE_PROPERTIES, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY, CL_MEM_USE_HOST_PTR, CL_MEM_ALLOC_HOST_PTR, CL_MEM_COPY_HOST_PTR, CL_R, CL_A, CL_RG, CL_RA, CL_RGB, CL_RGBA, CL_BGRA, CL_ARGB, CL_INTENSITY, CL_LUMINANCE, CL_SNORM_INT8, CL_SNORM_INT16, CL_UNORM_INT8, CL_UNORM_INT16, CL_UNORM_SHORT_565, CL_UNORM_SHORT_555, CL_UNORM_INT_101010, CL_SIGNED_INT8, CL_SIGNED_INT16, CL_SIGNED_INT32, CL_UNSIGNED_INT8, CL_UNSIGNED_INT16, CL_UNSIGNED_INT32, CL_HALF_FLOAT, CL_FLOAT, CL_MEM_OBJECT_BUFFER, CL_MEM_OBJECT_IMAGE2D, CL_MEM_OBJECT_IMAGE3D, CL_MEM_TYPE, CL_MEM_FLAGS, CL_MEM_SIZECL_MEM_HOST_PTR, CL_MEM_HOST_PTR, CL_MEM_MAP_COUNT, CL_MEM_REFERENCE_COUNT, CL_MEM_CONTEXT, CL_IMAGE_FORMAT, CL_IMAGE_ELEMENT_SIZE, CL_IMAGE_ROW_PITCH, CL_IMAGE_SLICE_PITCH, CL_IMAGE_WIDTH, CL_IMAGE_HEIGHT, CL_IMAGE_DEPTH, CL_ADDRESS_NONE, CL_ADDRESS_CLAMP_TO_EDGE, CL_ADDRESS_CLAMP, CL_ADDRESS_REPEAT, CL_FILTER_NEAREST, CL_FILTER_LINEAR, CL_SAMPLER_REFERENCE_COUNT, CL_SAMPLER_CONTEXT, CL_SAMPLER_NORMALIZED_COORDS, CL_SAMPLER_ADDRESSING_MODE, CL_SAMPLER_FILTER_MODE, CL_MAP_READ, CL_MAP_WRITE, CL_PROGRAM_REFERENCE_COUNT, CL_PROGRAM_CONTEXT, CL_PROGRAM_NUM_DEVICES, CL_PROGRAM_DEVICES, CL_PROGRAM_SOURCE, CL_PROGRAM_BINARY_SIZES, CL_PROGRAM_BINARIES, CL_PROGRAM_BUILD_STATUS, CL_PROGRAM_BUILD_OPTIONS, CL_PROGRAM_BUILD_LOG, CL_BUILD_SUCCESS, CL_BUILD_NONE, CL_BUILD_ERROR, CL_BUILD_IN_PROGRESS, CL_KERNEL_FUNCTION_NAME, CL_KERNEL_NUM_ARGS, CL_KERNEL_REFERENCE_COUNT, CL_KERNEL_CONTEXT, CL_KERNEL_PROGRAM, CL_KERNEL_WORK_GROUP_SIZE, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, CL_KERNEL_LOCAL_MEM_SIZE, CL_EVENT_COMMAND_QUEUE, CL_EVENT_COMMAND_TYPE, CL_EVENT_REFERENCE_COUNT, CL_EVENT_COMMAND_EXECUTION_STATUS, CL_COMMAND_NDRANGE_KERNEL, CL_COMMAND_TASK, CL_COMMAND_NATIVE_KERNEL, CL_COMMAND_READ_BUFFER, CL_COMMAND_WRITE_BUFFER, CL_COMMAND_COPY_BUFFER, CL_COMMAND_READ_IMAGE, CL_COMMAND_WRITE_IMAGE, CL_COMMAND_COPY_IMAGE, CL_COMMAND_COPY_IMAGE_TO_BUFFER, CL_COMMAND_COPY_BUFFER_TO_IMAGE, CL_COMMAND_MAP_BUFFER, CL_COMMAND_MAP_IMAGE, CL_COMMAND_UNMAP_MEM_OBJECT, CL_COMMAND_MARKER, CL_COMMAND_ACQUIRE_GL_OBJECTS, CL_COMMAND_RELEASE_GL_OBJECTS, command execution status, CL_COMPLETE, CL_RUNNING, CL_SUBMITTED, CL_QUEUED, CL_PROFILING_COMMAND_QUEUED, CL_PROFILING_COMMAND_SUBMIT, CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END, CL_CHAR_BIT, CL_SCHAR_MAX, CL_SCHAR_MIN, CL_CHAR_MAX, CL_CHAR_MIN, CL_UCHAR_MAX, CL_SHRT_MAX, CL_SHRT_MIN, CL_USHRT_MAX, CL_INT_MAX, CL_INT_MIN, CL_UINT_MAX, CL_LONG_MAX, CL_LONG_MIN, CL_ULONG_MAX, CL_FLT_DIG, CL_FLT_MANT_DIG, CL_FLT_MAX_10_EXP, CL_FLT_MAX_EXP, CL_FLT_MIN_10_EXP, CL_FLT_MIN_EXP, CL_FLT_RADIX, CL_FLT_MAX, CL_FLT_MIN, CL_FLT_EPSILON, CL_DBL_DIG, CL_DBL_MANT_DIG, CL_DBL_MAX_10_EXP, CL_DBL_MAX_EXP, CL_DBL_MIN_10_EXP, CL_DBL_MIN_EXP, CL_DBL_RADIX, CL_DBL_MAX, CL_DBL_MIN, CL_DBL_EPSILON, CL_SUCCESS, CL_DEVICE_NOT_FOUND, CL_DEVICE_NOT_AVAILABLE, CL_COMPILER_NOT_AVAILABLE, CL_MEM_OBJECT_ALLOCATION_FAILURE, CL_OUT_OF_RESOURCES, CL_OUT_OF_HOST_MEMORY, CL_PROFILING_INFO_NOT_AVAILABLE, CL_MEM_COPY_OVERLAP, CL_IMAGE_FORMAT_MISMATCH, CL_IMAGE_FORMAT_NOT_SUPPORTED, CL_BUILD_PROGRAM_FAILURE, CL_MAP_FAILURE, CL_INVALID_VALUE, CL_INVALID_DEVICE_TYPE, CL_INVALID_PLATFORM, CL_INVALID_DEVICE, CL_INVALID_CONTEXT, CL_INVALID_QUEUE_PROPERTIES, CL_INVALID_COMMAND_QUEUE, CL_INVALID_HOST_PTR, CL_INVALID_MEM_OBJECT, CL_INVALID_IMAGE_FORMAT_DESCRIPTOR, CL_INVALID_IMAGE_SIZE, CL_INVALID_SAMPLER, CL_INVALID_BINARY, CL_INVALID_BUILD_OPTIONS, CL_INVALID_PROGRAM, CL_INVALID_PROGRAM_EXECUTABLE, CL_INVALID_KERNEL_NAME, CL_INVALID_KERNEL_DEFINITION, CL_INVALID_KERNEL, CL_INVALID_ARG_INDEX, CL_INVALID_ARG_VALUE, CL_INVALID_ARG_SIZE, CL_INVALID_KERNEL_ARGS, CL_INVALID_WORK_DIMENSION, CL_INVALID_WORK_GROUP_SIZE, CL_INVALID_WORK_ITEM_SIZE, CL_INVALID_GLOBAL_OFFSET, CL_INVALID_EVENT_WAIT_LIST, CL_INVALID_EVENT, CL_INVALID_OPERATION, CL_INVALID_GL_OBJECT, CL_INVALID_BUFFER_SIZE, CL_INVALID_MIP_LEVEL, CL_INVALID_GLOBAL_WORK_SIZE'
spyder/utils/syntaxhighlighters.py:class OpenCLSH(CppSH):
spyder/utils/syntaxhighlighters.py:    """OpenCL Syntax Highlighter"""
spyder/utils/syntaxhighlighters.py:    PROG = re.compile(make_opencl_patterns(), re.S)
spyder/widgets/simplecodeeditor.py:    'OpenCL': ('cl',),
spyder/widgets/simplecodeeditor.py:        'OpenCL': (sh.OpenCLSH, '//'),
external-deps/python-lsp-server/pylsp/python_lsp.py:                "openClose": True,

```