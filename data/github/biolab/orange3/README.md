# https://github.com/biolab/orange3

```console
Orange/regression/xgb.py:                 gpu_id=None,
Orange/regression/xgb.py:                         importance_type=importance_type, gpu_id=gpu_id,
Orange/regression/xgb.py:                 gpu_id=None,
Orange/regression/xgb.py:                         importance_type=importance_type, gpu_id=gpu_id,
Orange/classification/xgb.py:                 gpu_id=None,
Orange/classification/xgb.py:                         gpu_id=gpu_id,
Orange/classification/xgb.py:                 gpu_id=None,
Orange/classification/xgb.py:                         gpu_id=gpu_id,
Orange/misc/server_embedder.py:from httpx import AsyncClient, NetworkError, ReadTimeout, Response
Orange/misc/server_embedder.py:        async with AsyncClient(
Orange/misc/server_embedder.py:        client: AsyncClient,
Orange/misc/server_embedder.py:            self, client: AsyncClient, data: Union[bytes, Dict], url: str
Orange/misc/tests/test_server_embedder.py:_HTTPX_POST_METHOD = "httpx.AsyncClient.post"
Orange/base.py:                 gpu_ram_part=None,
Orange/base.py:                 gpu_cat_features_storage=None,
Orange/__init__.py:# A hack that prevents segmentation fault with Nvidia drives on Linux if Qt's browser window
Orange/widgets/widget.py:class OWWidget(OWBaseWidget, ProgressBarMixin, Report, openclass=True):
Orange/widgets/visualize/owtreeviewer2d.py:class OWTreeViewer2D(OWWidget, openclass=True):
Orange/widgets/visualize/utils/widget.py:class OWProjectionWidgetBase(OWWidget, openclass=True):
Orange/widgets/visualize/utils/widget.py:class OWDataProjectionWidget(OWProjectionWidgetBase, openclass=True):
Orange/widgets/visualize/utils/widget.py:class OWAnchorProjectionWidget(OWDataProjectionWidget, openclass=True):
Orange/widgets/visualize/utils/vizrank.py:    class __VizRankMixin(OWBaseWidget, openclass=True):  # pylint: disable=invalid_name
Orange/widgets/data/owpreprocess.py:class OWPreprocess(widget.OWWidget, openclass=True):
Orange/widgets/utils/tests/test_owlearnerwidget.py:        class WidgetA(OWBaseLearner, openclass=True):
Orange/widgets/utils/owbasesql.py:class OWBaseSql(OWWidget, openclass=True):
Orange/widgets/utils/filedialogs.py:class OWUrlDropBase(OWWidget, openclass=True):
Orange/widgets/utils/owlearnerwidget.py:class OWBaseLearner(OWWidget, metaclass=OWBaseLearnerMeta, openclass=True):
Orange/widgets/utils/save/owsavebase.py:class OWSaveBase(widget.OWWidget, openclass=True):
CHANGELOG.md:* Workaround for segfaults with Nvidia on Linux ([#3100](../../pull/3100))
i18n/si/tests-msgs.jaml:    httpx.AsyncClient.post: null

```
