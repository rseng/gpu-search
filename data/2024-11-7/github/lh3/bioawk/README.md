# https://github.com/lh3/bioawk

```console
awkgram.y:%token	<i>	FINAL DOT ALL CCL NCCL CHAR OR STAR QUEST PLUS EMPTYRE
b.c:#define LEAF	case CCL: case NCCL: case CHAR: case DOT: case FINAL: case ALL:
b.c:	leaf (CCL, NCCL, CHAR, DOT, FINAL, ALL, EMPTYRE):
b.c:	case NCCL:
b.c:		np = op2(NCCL, NIL, (Node *) cclenter((char *) rlxstr));
b.c:	case CHAR: case DOT: case ALL: case EMPTYRE: case CCL: case NCCL: case '$': case '(':
b.c:					return NCCL;
b.c:			 || (k == NCCL && !member(c, (char *) f->re[p[i]].lval.up) && c != 0 && c != HAT)) {
b.c:		if (f->re[i].ltype == CCL || f->re[i].ltype == NCCL)

```
