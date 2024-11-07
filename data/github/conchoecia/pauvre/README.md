# https://github.com/conchoecia/pauvre

```console
pauvre/bioawk/awkgram.y:%token	<i>	FINAL DOT ALL CCL NCCL CHAR OR STAR QUEST PLUS EMPTYRE
pauvre/bioawk/proctab.c:	(char *) "NCCL",	/* 273 */
pauvre/bioawk/proctab.c:	nullproc,	/* NCCL */
pauvre/bioawk/ytab.c:     NCCL = 273,
pauvre/bioawk/ytab.c:#define NCCL 273
pauvre/bioawk/ytab.c:  "MATCHOP", "FINAL", "DOT", "ALL", "CCL", "NCCL", "CHAR", "OR", "STAR",
pauvre/bioawk/ytab.h:     NCCL = 273,
pauvre/bioawk/ytab.h:#define NCCL 273
pauvre/bioawk/b.c:#define LEAF	case CCL: case NCCL: case CHAR: case DOT: case FINAL: case ALL:
pauvre/bioawk/b.c:	leaf (CCL, NCCL, CHAR, DOT, FINAL, ALL, EMPTYRE):
pauvre/bioawk/b.c:	case NCCL:
pauvre/bioawk/b.c:		np = op2(NCCL, NIL, (Node *) cclenter((char *) rlxstr));
pauvre/bioawk/b.c:	case CHAR: case DOT: case ALL: case EMPTYRE: case CCL: case NCCL: case '$': case '(':
pauvre/bioawk/b.c:					return NCCL;
pauvre/bioawk/b.c:			 || (k == NCCL && !member(c, (char *) f->re[p[i]].lval.up) && c != 0 && c != HAT)) {
pauvre/bioawk/b.c:		if (f->re[i].ltype == CCL || f->re[i].ltype == NCCL)
src/bioawk_mod/awkgram.y:%token	<i>	FINAL DOT ALL CCL NCCL CHAR OR STAR QUEST PLUS EMPTYRE
src/bioawk_mod/b.c:#define LEAF	case CCL: case NCCL: case CHAR: case DOT: case FINAL: case ALL:
src/bioawk_mod/b.c:	leaf (CCL, NCCL, CHAR, DOT, FINAL, ALL, EMPTYRE):
src/bioawk_mod/b.c:	case NCCL:
src/bioawk_mod/b.c:		np = op2(NCCL, NIL, (Node *) cclenter((char *) rlxstr));
src/bioawk_mod/b.c:	case CHAR: case DOT: case ALL: case EMPTYRE: case CCL: case NCCL: case '$': case '(':
src/bioawk_mod/b.c:					return NCCL;
src/bioawk_mod/b.c:			 || (k == NCCL && !member(c, (char *) f->re[p[i]].lval.up) && c != 0 && c != HAT)) {
src/bioawk_mod/b.c:		if (f->re[i].ltype == CCL || f->re[i].ltype == NCCL)

```
