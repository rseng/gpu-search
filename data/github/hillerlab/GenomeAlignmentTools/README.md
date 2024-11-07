# https://github.com/hillerlab/GenomeAlignmentTools

```console
kent/src/hg/inc/hgRelate.h:void hgPurgeExtFileTbl(int id, struct sqlConnection *conn, char *extFileTbl);
kent/src/hg/inc/hgRelate.h:void hgPurgeExtFile(int id, struct sqlConnection *conn);
kent/src/hg/inc/cart.h: * and hgPublicSessions: */
kent/src/hg/inc/cart.h:#define hgPublicSessionsPrefix "hgPS_"
kent/src/hg/inc/cart.h:#define hgPublicSessionsTableState hgPublicSessionsPrefix dataTableStateName
kent/src/hg/lib/hgRelate.c:void hgPurgeExtFileTbl(int id, struct sqlConnection *conn, char *extFileTbl)
kent/src/hg/lib/hgRelate.c:void hgPurgeExtFile(int id,  struct sqlConnection *conn)
kent/src/hg/lib/hgRelate.c:hgPurgeExtFileTbl(id, conn, "extFile");
kent/src/hg/lib/kgPutBack2.sql:CREATE TABLE kgPutBack2 (
kent/src/hg/lib/cart.c:    char *pubSessionsTableString = cartOptionalString(cart, hgPublicSessionsTableState);
kent/src/hg/lib/cart.c:	    cartSetString(cart, hgPublicSessionsTableState, pubSessionsTableString);
kent/src/hg/lib/cart.c:char *pubSessionsTableString = cartOptionalString(cart, hgPublicSessionsTableState);
kent/src/hg/lib/cart.c:    cartSetString(cart, hgPublicSessionsTableState, pubSessionsTableString);
kent/src/hg/lib/facetField.c:#define facetValStringPunc '~'
kent/src/hg/lib/facetField.c:    dyStringAppendC(dy, facetValStringPunc);
kent/src/hg/lib/facetField.c:    char *end = strchr(selectedFields, facetValStringPunc);
kent/src/inc/memgfx.h:#define _mgPutDot(mg, x, y, color) (*_mgPixAdr(mg,x,y) = (color))
kent/src/inc/memgfx.h:void _mgPutDotMultiply(struct memGfx *mg, int x, int y,Color color);
kent/src/inc/memgfx.h:INLINE void mgPutDot(struct memGfx *mg, int x, int y,Color color)
kent/src/inc/memgfx.h:            _mgPutDot(mg,x,y,color);
kent/src/inc/memgfx.h:            _mgPutDotMultiply(mg,x,y,color);
kent/src/inc/memgfx.h:void mgPutSeg(struct memGfx *mg, int x, int y, int width, Color *dots);
kent/src/inc/memgfx.h:void mgPutSegZeroClear(struct memGfx *mg, int x, int y, int width, Color *dots);
kent/src/lib/mgCircle.c:    mgPutDot(mg, xCen, yCen, color);
kent/src/lib/mgCircle.c:	mgPutDot(mg, xCen+x, yCen+y, color);
kent/src/lib/mgCircle.c:	mgPutDot(mg, xCen+x, yCen-y, color);
kent/src/lib/mgCircle.c:	mgPutDot(mg, xCen-x, yCen+y, color);
kent/src/lib/mgCircle.c:	mgPutDot(mg, xCen-x, yCen-y, color);
kent/src/lib/tests/input/mimeDecodeTest.txt:iOSBd7o4Xy3ABGPu4Ocjr1rodT8aaLaW1r4WuHuf7amtlC2oiKspCZyWPy4+U9znFdVZXupukC3m
kent/src/lib/tests/input/mimeDecodeTest.txt:Z2t3QoSQzKpAZjvPIGQF9zXrHhmz1jT9Chttd1GPUL9C2+4jj2BhngY9hXI/ECUDx54Bh4y2oO3v
kent/src/lib/tests/input/mimeDecodeTest.txt:CgPUwOfMib6FWx/wGl7OK+HQbrSektfX/Pc5Gy+O5lKW0/huSW9kOUFndo8RUfeJY4KkehH5V1Wm
kent/src/lib/tests/input/mimeDecodeTest.txt:Qw4YgERnWEABbRH/gbwwuEWLIQIrCZIP5fiRIKr5EDgYUM08GsCiMdZvaUKRBgPUgif+gcONRAHg
kent/src/lib/tests/input/mimeDecodeTest.txt:dVUXjjQzgpUEylpRwbJINXasyzoBpAUVyMAcEXj8K85UzIfcEsmszJcXJPgbLc48PWnFdTL8tDKH
kent/src/lib/errAbort.c:    boolean debugPushPopErr;        // generate stack dump on push/pop error
kent/src/lib/errAbort.c:    if (ptav->debugPushPopErr)
kent/src/lib/errAbort.c:    if (ptav->debugPushPopErr)
kent/src/lib/errAbort.c:    if (ptav->debugPushPopErr)
kent/src/lib/errAbort.c:    if (ptav->debugPushPopErr)
kent/src/lib/errAbort.c:ptav->debugPushPopErr = TRUE;
kent/src/lib/errAbort.c:    ptav->debugPushPopErr = FALSE;
kent/src/lib/memgfx.c:void _mgPutDotMultiply(struct memGfx *mg, int x, int y,Color color)
kent/src/lib/memgfx.c:static void mgPutSegMaybeZeroClear(struct memGfx *mg, int x, int y, int width, Color *dots, boolean zeroClear)
kent/src/lib/memgfx.c:    mgPutSegMaybeZeroClear(mg, xOff, yOff, width, dots, zeroClear);
kent/src/lib/memgfx.c:mgPutDot(img,x,y,MAKECOLOR_32(r,g,b));
kent/src/lib/memgfx.c:		mgPutDot(mg,x1,y1,color);
kent/src/lib/memgfx.c:		mgPutDot(mg,x1,y1,color);
kent/src/lib/memgfx.c:void mgPutSeg(struct memGfx *mg, int x, int y, int width, Color *dots)
kent/src/lib/memgfx.c:mgPutSegMaybeZeroClear(mg, x, y, width, dots, FALSE);
kent/src/lib/memgfx.c:void mgPutSegZeroClear(struct memGfx *mg, int x, int y, int width, Color *dots)
kent/src/lib/memgfx.c:mgPutSegMaybeZeroClear(mg, x, y, width, dots, TRUE);
kent/src/lib/memgfx.c:mgPutDot(mg, x, y, colorIx);
kent/src/lib/memgfx.c:               mgPutDot(mg, x1, y0, color); /*   I. Quadrant */
kent/src/lib/memgfx.c:               mgPutDot(mg, x0, y0, color); /*  II. Quadrant */
kent/src/lib/memgfx.c:               mgPutDot(mg, x0, y1, color); /* III. Quadrant */
kent/src/lib/memgfx.c:               mgPutDot(mg, x1, y1, color); /*  IV. Quadrant */
kent/src/lib/memgfx.c:           mgPutDot(mg, x0-1, y0, color); /* -> finish tip of ellipse */
kent/src/lib/memgfx.c:           mgPutDot(mg, x1+1, y0++, color);
kent/src/lib/memgfx.c:           mgPutDot(mg, x0-1, y1, color);
kent/src/lib/memgfx.c:           mgPutDot(mg, x1+1, y1--, color);
kent/src/lib/pipeline.c:static void plProcMemWrite(struct plProc* proc, int stdoutFd, int stderrFd, void *otherEndBuf, size_t otherEndBufSize)
kent/src/lib/pipeline.c:    plProcMemWrite(proc, procStdoutFd, stderrFd, otherEndBuf, otherEndBufSize);

```
