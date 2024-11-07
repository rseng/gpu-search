# https://github.com/vanheeringen-lab/gimmemotifs

```console
gimmemotifs/rocmetrics.py:logger = logging.getLogger("gimme.rocmetrics")
gimmemotifs/utils.py:from gimmemotifs.rocmetrics import ks_pvalue
gimmemotifs/__init__.py:from . import rocmetrics       # noqa: F401
gimmemotifs/stats.py:from gimmemotifs import rocmetrics
gimmemotifs/stats.py:        Names of metrics to calculate. See gimmemotifs.rocmetrics.__all__
gimmemotifs/stats.py:        stats = rocmetrics.__all__
gimmemotifs/stats.py:            func = getattr(rocmetrics, s)
gimmemotifs/stats.py:        Names of metrics to calculate. See gimmemotifs.rocmetrics.__all__
gimmemotifs/stats.py:            func = getattr(rocmetrics, s)
gimmemotifs/stats.py:            func = getattr(rocmetrics, stat)
test/test_08_prediction.py:        # for stat in rocmetrics.__all__:
test/test_02_metrics.py:from gimmemotifs.rocmetrics import pr_auc
test/test_06_stats.py:from gimmemotifs import rocmetrics
test/test_06_stats.py:            if "fg_table" not in kwargs or getattr(rocmetrics, f).input_type != "pos":
test/test_06_stats.py:                print(f, fg_table, getattr(rocmetrics, f).input_type)

```
