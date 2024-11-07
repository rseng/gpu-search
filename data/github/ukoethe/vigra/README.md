# https://github.com/ukoethe/vigra

```console
include/vigra/multi_opencl.hxx:#ifndef VIGRA_OPENCL_HXX
include/vigra/multi_opencl.hxx:#define VIGRA_OPENCL_HXX
include/vigra/multi_opencl.hxx:#include <OpenCL/opencl.h>
include/vigra/multi_opencl.hxx:#include <CL/opencl.h>
include/vigra/multi_opencl.hxx:#define VIGRA_OPENCL_VECTYPEN_INTEGER_TRAITS(basetype, n)               \
include/vigra/multi_opencl.hxx:#define VIGRA_OPENCL_VECTYPEN_REAL_TRAITS(basetype, n)                  \
include/vigra/multi_opencl.hxx:#define VIGRA_OPENCL_VECN_TRAITS(n)                          \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_VECTYPEN_INTEGER_TRAITS(cl_char, n);          \
include/vigra/multi_opencl.hxx:    VIGRA_OPENCL_VECTYPEN_INTEGER_TRAITS(cl_uchar, n);       \
include/vigra/multi_opencl.hxx:    VIGRA_OPENCL_VECTYPEN_INTEGER_TRAITS(cl_short, n);       \
include/vigra/multi_opencl.hxx:    VIGRA_OPENCL_VECTYPEN_INTEGER_TRAITS(cl_ushort, n);      \
include/vigra/multi_opencl.hxx:    VIGRA_OPENCL_VECTYPEN_INTEGER_TRAITS(cl_int, n);         \
include/vigra/multi_opencl.hxx:    VIGRA_OPENCL_VECTYPEN_INTEGER_TRAITS(cl_uint, n);        \
include/vigra/multi_opencl.hxx:    VIGRA_OPENCL_VECTYPEN_INTEGER_TRAITS(cl_long, n);        \
include/vigra/multi_opencl.hxx:    VIGRA_OPENCL_VECTYPEN_INTEGER_TRAITS(cl_ulong, n);       \
include/vigra/multi_opencl.hxx:    VIGRA_OPENCL_VECTYPEN_REAL_TRAITS(cl_float, n);          \
include/vigra/multi_opencl.hxx:    VIGRA_OPENCL_VECTYPEN_REAL_TRAITS(cl_double, n);
include/vigra/multi_opencl.hxx:VIGRA_OPENCL_VECN_TRAITS(2);
include/vigra/multi_opencl.hxx:VIGRA_OPENCL_VECN_TRAITS(3);
include/vigra/multi_opencl.hxx://VIGRA_OPENCL_VECN_TRAITS(4); // cl_type4 is the same as cl_type3
include/vigra/multi_opencl.hxx:VIGRA_OPENCL_VECN_TRAITS(8);
include/vigra/multi_opencl.hxx:VIGRA_OPENCL_VECN_TRAITS(16);
include/vigra/multi_opencl.hxx:#undef VIGRA_OPENCL_VECTYPEN_INTEGER_TRAITS
include/vigra/multi_opencl.hxx:#undef VIGRA_OPENCL_VECTYPEN_REAL_TRAITS
include/vigra/multi_opencl.hxx:#undef VIGRA_OPENCL_VECN_TRAITS
include/vigra/multi_opencl.hxx:/**     OpenCL 1.1 [6.2] - Convert operators */
include/vigra/multi_opencl.hxx:/**     OpenCL 1.1 [6.3] - Scalar/vector math operators */
include/vigra/multi_opencl.hxx:/**     OpenCL 1.1 [6.11.2] - Math Built-in Functions */
include/vigra/multi_opencl.hxx:/**     OpenCL 1.1 [6.11.3] - Integer Built-in Functions */
include/vigra/multi_opencl.hxx:/**     OpenCL 1.1 [6.11.4] - Common Built-in Functions */
include/vigra/multi_opencl.hxx:/**     OpenCL 1.1 [6.11.5] - Geometric Built-in Functions */
include/vigra/multi_opencl.hxx:/**     OpenCL 1.1 [6.11.6] - Relational Built-in Functions */
include/vigra/multi_opencl.hxx:/**     OpenCL 1.1 [6.11.7] - Vector Data Load/Store Built-in Functions */
include/vigra/multi_opencl.hxx:/**     OpenCL 1.1 [6.11.12] - Misc Vector Built-in Functions */
include/vigra/multi_opencl.hxx:/**     OpenCL 1.1 [6.11.12] - Image Read and Write Built-in Functions */
include/vigra/multi_opencl.hxx:/** \defgroup OpenCL-Accessors Accessors for OpenCL types
include/vigra/multi_opencl.hxx:    Encapsulate access to members of OpenCL vector types.
include/vigra/multi_opencl.hxx:    <b>\#include</b> \<vigra/multi_opencl.hxx\>
include/vigra/multi_opencl.hxx:    OpenCL 1.1 [6.1.7] - Vector Components
include/vigra/multi_opencl.hxx:    #include <vigra/multi_opencl.hxx>
include/vigra/multi_opencl.hxx:#define VIGRA_OPENCL_TYPE_ACCESSOR(basetype, n, NTH) \
include/vigra/multi_opencl.hxx:#define VIGRA_OPENCL_TYPE2_ACCESSORS(basetype)  \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 2, s0);  \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 2, s1);  \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 2, x);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 2, y);
include/vigra/multi_opencl.hxx:#define VIGRA_OPENCL_TYPE3_ACCESSORS(basetype)  \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 3, s0);  \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 3, s1);  \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 3, s2);  \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 3, x);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 3, y);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 3, z);
include/vigra/multi_opencl.hxx:#define VIGRA_OPENCL_TYPE4_ACCESSORS(basetype)   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 4, s0);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 4, s1);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 4, s2);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 4, s3);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 4, x);    \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 4, y);    \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 4, z);    \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 4, w);
include/vigra/multi_opencl.hxx:#define VIGRA_OPENCL_TYPE8_ACCESSORS(basetype)   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 8, s0);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 8, s1);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 8, s2);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 8, s3);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 8, s4);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 8, s5);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 8, s6);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 8, s7);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 8, s8);
include/vigra/multi_opencl.hxx:#define VIGRA_OPENCL_TYPE16_ACCESSORS(basetype)   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 16, s0);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 16, s1);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 16, s2);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 16, s3);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 16, s4);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 16, s5);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 16, s6);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 16, s7);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 16, s8);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 16, sa);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 16, sb);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 16, sc);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 16, sd);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 16, se);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 16, sf);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 16, sA);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 16, sB);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 16, sC);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 16, sD);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 16, sE);   \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE_ACCESSOR(basetype, 16, sF);
include/vigra/multi_opencl.hxx:#define VIGRA_OPENCL_ACCESSORS(basetype)  \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE2_ACCESSORS(basetype); \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE3_ACCESSORS(basetype); \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE4_ACCESSORS(basetype); \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE8_ACCESSORS(basetype); \
include/vigra/multi_opencl.hxx:  VIGRA_OPENCL_TYPE16_ACCESSORS(basetype);
include/vigra/multi_opencl.hxx:VIGRA_OPENCL_ACCESSORS(cl_char);
include/vigra/multi_opencl.hxx:VIGRA_OPENCL_ACCESSORS(cl_uchar);
include/vigra/multi_opencl.hxx:VIGRA_OPENCL_ACCESSORS(cl_short);
include/vigra/multi_opencl.hxx:VIGRA_OPENCL_ACCESSORS(cl_ushort);
include/vigra/multi_opencl.hxx:VIGRA_OPENCL_ACCESSORS(cl_int);
include/vigra/multi_opencl.hxx:VIGRA_OPENCL_ACCESSORS(cl_uint);
include/vigra/multi_opencl.hxx:VIGRA_OPENCL_ACCESSORS(cl_long);
include/vigra/multi_opencl.hxx:VIGRA_OPENCL_ACCESSORS(cl_ulong);
include/vigra/multi_opencl.hxx:VIGRA_OPENCL_ACCESSORS(cl_float);
include/vigra/multi_opencl.hxx:VIGRA_OPENCL_ACCESSORS(cl_double);
include/vigra/multi_opencl.hxx:#undef VIGRA_OPENCL_TYPE_ACCESSOR
include/vigra/multi_opencl.hxx:#undef VIGRA_OPENCL_TYPE2_ACCESSORS
include/vigra/multi_opencl.hxx:#undef VIGRA_OPENCL_TYPE3_ACCESSORS
include/vigra/multi_opencl.hxx:#undef VIGRA_OPENCL_TYPE4_ACCESSORS
include/vigra/multi_opencl.hxx:#undef VIGRA_OPENCL_TYPE8_ACCESSORS
include/vigra/multi_opencl.hxx:#undef VIGRA_OPENCL_TYPE16_ACCESSORS
include/vigra/multi_opencl.hxx:#undef VIGRA_OPENCL_ACCESSORS
include/vigra/multi_opencl.hxx:#endif // VIGRA_OPENCL_HXX

```
