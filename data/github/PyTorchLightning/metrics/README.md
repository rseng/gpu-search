# https://github.com/PyTorchLightning/metrics

```console
docs/source/pages/quickstart.rst:    # Optional if you do not need compile GPU support
docs/source/pages/quickstart.rst:    export USE_CUDA=0  # just to keep it simple
docs/source/pages/quickstart.rst:In order to use GPUs, you can enable them within the ``.devcontainer/devcontainer.json`` file.
docs/source/pages/overview.rst:    target = torch.tensor([1, 1, 0, 0], device=torch.device("cuda", 0))
docs/source/pages/overview.rst:    preds = torch.tensor([0, 1, 0, 0], device=torch.device("cuda", 0))
docs/source/pages/overview.rst:    confmat = BinaryAccuracy().to(torch.device("cuda", 0))
docs/source/pages/overview.rst:    print(out.device) # cuda:0
docs/source/pages/overview.rst:    metric = MulticlassAccuracy(num_classes=5).to("cuda")
docs/source/pages/overview.rst:    metric.update(torch.randint(5, (100,)).cuda(), torch.randint(5, (100,)).cuda())
docs/source/pages/overview.rst:the test dataset to only run on a single gpu or use a `join <https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel.join>`_
docs/source/pages/overview.rst:* In general ``pytorch`` had better support for 16-bit precision much earlier on GPU than CPU. Therefore, we
docs/source/pages/overview.rst:If you are running metrics on GPU and are encountering that you are running out of GPU VRAM then the following
docs/source/pages/overview.rst:  GPU memory is not filling up. The consequence will be that the ``compute`` method will be called on CPU instead
docs/source/pages/overview.rst:  of GPU. Only applies to metric states that are lists.
docs/paper_JOSS/paper.md:Another core feature of TorchMetrics is its ability to scale to multiple devices seamlessly. Modern deep learning models are often trained on hundreds of devices such as GPUs or TPUs (see @large_example1; @large_example2 for examples). This scale introduces the need to synchronize metrics across machines to get the correct value during training and evaluation. In distributed environments, TorchMetrics automatically accumulates across devices before reporting the calculated metric to the user.
docs/paper_JOSS/paper.md:TorchMetrics exhibits high test coverage on the various configurations, including all three major OS platforms (Linux, macOS, and Windows), and various Python, CUDA, and PyTorch versions. We test both minimum and latest package requirements for all combinations of OS and Python versions and include additional tests for each PyTorch version from 1.3 up to future development versions. On every pull request and merge to master, we run a full test suite. All standard tests run on CPU. In addition, we run all tests on a multi-GPU setting which reflects realistic Deep Learning workloads. For usability, we have auto-generated HTML documentation (hosted at [readthedocs](https://torchmetrics.readthedocs.io/en/stable/)) from the source code which updates in real-time with new merged pull requests.
CHANGELOG.md:- Updated `_safe_divide` to allow `Accuracy` to run on the GPU ([#2640](https://github.com/Lightning-AI/torchmetrics/pull/2640))
CHANGELOG.md:- Fixed `_cumsum` helper function in multi-gpu ([#2636](https://github.com/Lightning-AI/torchmetrics/pull/2636))
CHANGELOG.md:- Fixed IOU compute in cuda ([#1982](https://github.com/Lightning-AI/torchmetrics/pull/1982))
CHANGELOG.md:- Added support for deterministic evaluation on GPU for metrics that uses `torch.cumsum` operator ([#1499](https://github.com/Lightning-AI/metrics/pull/1499))
CHANGELOG.md:- Fixed "Sort currently does not support bool dtype on CUDA" error in MAP for empty preds ([#983](https://github.com/Lightning-AI/metrics/pull/983))
CHANGELOG.md:- Fixed `BestScore` on GPU ([#912](https://github.com/Lightning-AI/metrics/pull/912))
CHANGELOG.md:- Fixed `ConfusionMatrix`, `AUROC` and `AveragePrecision` on GPU when running in deterministic mode ([#900](https://github.com/Lightning-AI/metrics/pull/900))
CHANGELOG.md:- Fixed `torch.sort` currently does not support bool `dtype` on CUDA ([#665](https://github.com/Lightning-AI/metrics/pull/665))
CHANGELOG.md:- Fix edge case of AUROC with `average=weighted` on GPU ([#606](https://github.com/Lightning-AI/metrics/pull/606))
CHANGELOG.md:- Fixed `BootStrapper` metrics not working on GPU ([#462](https://github.com/Lightning-AI/metrics/pull/462))
tests/unittests/regression/test_r2.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/regression/test_r2.py:    def test_r2_half_gpu(self, adjusted, multioutput, preds, target, ref_metric):
tests/unittests/regression/test_r2.py:        """Test dtype support of the metric on GPU."""
tests/unittests/regression/test_r2.py:        self.run_precision_test_gpu(
tests/unittests/regression/test_mean_error.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/regression/test_mean_error.py:    def test_mean_error_half_gpu(self, preds, target, ref_metric, metric_class, metric_functional, sk_fn, metric_args):
tests/unittests/regression/test_mean_error.py:        """Test dtype support of the metric on GPU."""
tests/unittests/regression/test_mean_error.py:        self.run_precision_test_gpu(preds, target, metric_class, metric_functional)
tests/unittests/regression/test_pearson.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/regression/test_pearson.py:    def test_pearson_corrcoef_half_gpu(self, preds, target):
tests/unittests/regression/test_pearson.py:        """Test dtype support of the metric on GPU."""
tests/unittests/regression/test_pearson.py:        self.run_precision_test_gpu(preds, target, partial(PearsonCorrCoef, num_outputs=num_outputs), pearson_corrcoef)
tests/unittests/regression/test_kl_divergence.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/regression/test_kl_divergence.py:    def test_kldivergence_half_gpu(self, reduction, p, q, log_prob):
tests/unittests/regression/test_kl_divergence.py:        """Test dtype support of the metric on GPU."""
tests/unittests/regression/test_kl_divergence.py:        self.run_precision_test_gpu(p, q, KLDivergence, kl_divergence, {"log_prob": log_prob, "reduction": reduction})
tests/unittests/regression/test_rse.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/regression/test_rse.py:    def test_rse_half_gpu(self, squared, preds, target, ref_metric, num_outputs):
tests/unittests/regression/test_rse.py:        """Test dtype support of the metric on GPU."""
tests/unittests/regression/test_rse.py:        self.run_precision_test_gpu(
tests/unittests/regression/test_minkowski_distance.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/regression/test_minkowski_distance.py:    def test_minkowski_distance_half_gpu(self, preds, target, ref_metric, p):
tests/unittests/regression/test_minkowski_distance.py:        """Test dtype support of the metric on GPU."""
tests/unittests/regression/test_minkowski_distance.py:        self.run_precision_test_gpu(preds, target, MinkowskiDistance, minkowski_distance, metric_args={"p": p})
tests/unittests/regression/test_explained_variance.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/regression/test_explained_variance.py:    def test_explained_variance_half_gpu(self, multioutput, preds, target, ref_metric):
tests/unittests/regression/test_explained_variance.py:        """Test dtype support of the metric on GPU."""
tests/unittests/regression/test_explained_variance.py:        self.run_precision_test_gpu(preds, target, ExplainedVariance, explained_variance)
tests/unittests/regression/test_spearman.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/regression/test_spearman.py:    def test_spearman_corrcoef_half_gpu(self, preds, target):
tests/unittests/regression/test_spearman.py:        """Test dtype support of the metric on GPU."""
tests/unittests/regression/test_spearman.py:        self.run_precision_test_gpu(
tests/unittests/regression/test_concordance.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/regression/test_concordance.py:    def test_concordance_corrcoef_half_gpu(self, preds, target):
tests/unittests/regression/test_concordance.py:        """Test dtype support of the metric on GPU."""
tests/unittests/regression/test_concordance.py:        self.run_precision_test_gpu(
tests/unittests/regression/test_tweedie_deviance.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/regression/test_tweedie_deviance.py:    def test_deviance_scores_half_gpu(self, preds, target, power):
tests/unittests/regression/test_tweedie_deviance.py:        """Test dtype support of the metric on GPU."""
tests/unittests/regression/test_tweedie_deviance.py:        self.run_precision_test_gpu(
tests/unittests/audio/test_pesq.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/audio/test_pesq.py:    def test_pesq_half_gpu(self, preds, target, ref_metric, fs, mode):
tests/unittests/audio/test_pesq.py:        """Test dtype support of the metric on GPU."""
tests/unittests/audio/test_pesq.py:        self.run_precision_test_gpu(
tests/unittests/audio/test_pit.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/audio/test_pit.py:    def test_pit_half_gpu(self, preds, target, ref_metric, metric_func, mode, eval_func):
tests/unittests/audio/test_pit.py:        """Test dtype support of the metric on GPU."""
tests/unittests/audio/test_pit.py:        self.run_precision_test_gpu(
tests/unittests/audio/test_si_sdr.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/audio/test_si_sdr.py:    def test_si_sdr_half_gpu(self, preds, target, ref_metric, zero_mean):
tests/unittests/audio/test_si_sdr.py:        """Test dtype support of the metric on GPU."""
tests/unittests/audio/test_si_sdr.py:        self.run_precision_test_gpu(
tests/unittests/audio/test_sdr.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/audio/test_sdr.py:    def test_sdr_half_gpu(self, preds, target):
tests/unittests/audio/test_sdr.py:        """Test dtype support of the metric on GPU."""
tests/unittests/audio/test_sdr.py:        self.run_precision_test_gpu(
tests/unittests/audio/test_si_snr.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/audio/test_si_snr.py:    def test_si_snr_half_gpu(self, preds, target, ref_metric):
tests/unittests/audio/test_si_snr.py:        """Test dtype support of the metric on GPU."""
tests/unittests/audio/test_si_snr.py:        self.run_precision_test_gpu(
tests/unittests/audio/test_stoi.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/audio/test_stoi.py:    def test_stoi_half_gpu(self, preds, target, ref_metric, fs, extended):
tests/unittests/audio/test_stoi.py:        """Test dtype support of the metric on GPU."""
tests/unittests/audio/test_stoi.py:        self.run_precision_test_gpu(
tests/unittests/audio/test_c_si_snr.py:    def test_c_si_sdr_half_gpu(self, preds, target, ref_metric, zero_mean):
tests/unittests/audio/test_c_si_snr.py:        """Test dtype support of the metric on GPU."""
tests/unittests/audio/test_c_si_snr.py:        pytest.xfail("C-SI-SDR metric does not support gpu + half precision")
tests/unittests/audio/test_snr.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/audio/test_snr.py:    def test_snr_half_gpu(self, preds, target, ref_metric, zero_mean):
tests/unittests/audio/test_snr.py:        """Test dtype support of the metric on GPU."""
tests/unittests/audio/test_snr.py:        self.run_precision_test_gpu(
tests/unittests/audio/test_srmr.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/audio/test_srmr.py:    def test_srmr_half_gpu(self, preds, fs, fast, norm):
tests/unittests/audio/test_srmr.py:        """Test dtype support of the metric on GPU."""
tests/unittests/audio/test_srmr.py:        self.run_precision_test_gpu(
tests/unittests/audio/test_dnsmos.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/audio/test_dnsmos.py:    def test_dnsmos_cuda(self, preds: Tensor, fs: int, personalized: bool, ddp: bool, device="cuda:0"):
tests/unittests/audio/test_sa_sdr.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/audio/test_sa_sdr.py:    def test_sa_sdr_half_gpu(self, preds, target, scale_invariant, zero_mean):
tests/unittests/audio/test_sa_sdr.py:        """Test dtype support of the metric on GPU."""
tests/unittests/audio/test_sa_sdr.py:        self.run_precision_test_gpu(
tests/unittests/detection/test_map.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA availability")
tests/unittests/detection/test_map.py:    def test_map_gpu(self, backend, inputs):
tests/unittests/detection/test_map.py:        """Test predictions on single gpu."""
tests/unittests/detection/test_map.py:        metric = metric.to("cuda")
tests/unittests/detection/test_map.py:                apply_to_collection(preds, Tensor, lambda x: x.to("cuda")),
tests/unittests/detection/test_map.py:                apply_to_collection(targets, Tensor, lambda x: x.to("cuda")),
tests/unittests/detection/test_map.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA availability")
tests/unittests/detection/test_map.py:        metric = metric.to("cuda")
tests/unittests/detection/test_map.py:                apply_to_collection(preds, Tensor, lambda x: x.to("cuda")),
tests/unittests/detection/test_map.py:                apply_to_collection(targets, Tensor, lambda x: x.to("cuda")),
tests/unittests/detection/test_map.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/detection/test_map.py:        device = "cuda"
tests/unittests/retrieval/test_fallout.py:    def test_precision_gpu(self, indexes: Tensor, preds: Tensor, target: Tensor):
tests/unittests/retrieval/test_fallout.py:        """Test dtype support of the metric on GPU."""
tests/unittests/retrieval/test_fallout.py:        self.run_precision_test_gpu(
tests/unittests/retrieval/test_map.py:    def test_precision_gpu(self, indexes: Tensor, preds: Tensor, target: Tensor):
tests/unittests/retrieval/test_map.py:        """Test dtype support of the metric on GPU."""
tests/unittests/retrieval/test_map.py:        self.run_precision_test_gpu(
tests/unittests/retrieval/test_recall.py:    def test_precision_gpu(self, indexes: Tensor, preds: Tensor, target: Tensor):
tests/unittests/retrieval/test_recall.py:        """Test dtype support of the metric on GPU."""
tests/unittests/retrieval/test_recall.py:        self.run_precision_test_gpu(
tests/unittests/retrieval/test_mrr.py:    def test_precision_gpu(self, indexes: Tensor, preds: Tensor, target: Tensor):
tests/unittests/retrieval/test_mrr.py:        """Test dtype support of the metric on GPU."""
tests/unittests/retrieval/test_mrr.py:        self.run_precision_test_gpu(
tests/unittests/retrieval/test_ndcg.py:    def test_precision_gpu(self, indexes: Tensor, preds: Tensor, target: Tensor):
tests/unittests/retrieval/test_ndcg.py:        """Test dtype support of the metric on GPU."""
tests/unittests/retrieval/test_ndcg.py:        self.run_precision_test_gpu(
tests/unittests/retrieval/test_precision.py:    def test_precision_gpu(self, indexes: Tensor, preds: Tensor, target: Tensor):
tests/unittests/retrieval/test_precision.py:        """Test dtype support of the metric on GPU."""
tests/unittests/retrieval/test_precision.py:        self.run_precision_test_gpu(
tests/unittests/retrieval/helpers.py:    def run_precision_test_gpu(
tests/unittests/retrieval/helpers.py:        """Test dtype support of the metric on GPU."""
tests/unittests/retrieval/helpers.py:        if not torch.cuda.is_available():
tests/unittests/retrieval/helpers.py:            pytest.skip("Test requires GPU")
tests/unittests/retrieval/helpers.py:        super().run_precision_test_gpu(
tests/unittests/retrieval/test_hit_rate.py:    def test_precision_gpu(self, indexes: Tensor, preds: Tensor, target: Tensor):
tests/unittests/retrieval/test_hit_rate.py:        """Test dtype support of the metric on GPU."""
tests/unittests/retrieval/test_hit_rate.py:        self.run_precision_test_gpu(
tests/unittests/retrieval/test_r_precision.py:    def test_precision_gpu(self, indexes: Tensor, preds: Tensor, target: Tensor):
tests/unittests/retrieval/test_r_precision.py:        """Test dtype support of the metric on GPU."""
tests/unittests/retrieval/test_r_precision.py:        self.run_precision_test_gpu(
tests/unittests/retrieval/test_auroc.py:    def test_precision_gpu(self, indexes: Tensor, preds: Tensor, target: Tensor):
tests/unittests/retrieval/test_auroc.py:        """Test dtype support of the metric on GPU."""
tests/unittests/retrieval/test_auroc.py:        self.run_precision_test_gpu(
tests/unittests/multimodal/test_clip_score.py:@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_cohen_kappa.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_cohen_kappa.py:    def test_binary_confusion_matrix_dtypes_gpu(self, inputs, dtype):
tests/unittests/classification/test_cohen_kappa.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_cohen_kappa.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_cohen_kappa.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_cohen_kappa.py:    def test_multiclass_confusion_matrix_dtypes_gpu(self, inputs, dtype):
tests/unittests/classification/test_cohen_kappa.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_cohen_kappa.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_matthews_corrcoef.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_matthews_corrcoef.py:    def test_binary_matthews_corrcoef_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_matthews_corrcoef.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_matthews_corrcoef.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_matthews_corrcoef.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_matthews_corrcoef.py:    def test_multiclass_matthews_corrcoef_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_matthews_corrcoef.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_matthews_corrcoef.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_matthews_corrcoef.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_matthews_corrcoef.py:    def test_multilabel_matthews_corrcoef_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_matthews_corrcoef.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_matthews_corrcoef.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_precision_fixed_recall.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_precision_fixed_recall.py:    def test_binary_precision_at_fixed_recall_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_precision_fixed_recall.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_precision_fixed_recall.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_precision_fixed_recall.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_precision_fixed_recall.py:    def test_multiclass_precision_at_fixed_recall_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_precision_fixed_recall.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_precision_fixed_recall.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_precision_fixed_recall.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_precision_fixed_recall.py:    def test_multiclass_precision_at_fixed_recall_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_precision_fixed_recall.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_precision_fixed_recall.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_calibration_error.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_calibration_error.py:    def test_binary_calibration_error_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_calibration_error.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_calibration_error.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_calibration_error.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_calibration_error.py:    def test_multiclass_calibration_error_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_calibration_error.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_calibration_error.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_roc.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_roc.py:    def test_binary_roc_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_roc.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_roc.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_roc.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_roc.py:    def test_multiclass_roc_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_roc.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_roc.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_roc.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_roc.py:    def test_multiclass_roc_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_roc.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_roc.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_jaccard.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_jaccard.py:    def test_binary_jaccard_index_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_jaccard.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_jaccard.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_jaccard.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_jaccard.py:    def test_multiclass_jaccard_index_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_jaccard.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_jaccard.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_jaccard.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_jaccard.py:    def test_multilabel_jaccard_index_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_jaccard.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_jaccard.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_precision_recall_curve.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_precision_recall_curve.py:    def test_binary_precision_recall_curve_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_precision_recall_curve.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_precision_recall_curve.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_precision_recall_curve.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_precision_recall_curve.py:    def test_multiclass_precision_recall_curve_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_precision_recall_curve.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_precision_recall_curve.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_precision_recall_curve.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_precision_recall_curve.py:    def test_multiclass_precision_recall_curve_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_precision_recall_curve.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_precision_recall_curve.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_specificity_sensitivity.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_specificity_sensitivity.py:    def test_binary_specificity_at_sensitivity_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_specificity_sensitivity.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_specificity_sensitivity.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_specificity_sensitivity.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_specificity_sensitivity.py:    def test_multiclass_specificity_at_sensitivity_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_specificity_sensitivity.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_specificity_sensitivity.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_specificity_sensitivity.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_specificity_sensitivity.py:    def test_multiclass_specificity_at_sensitivity_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_specificity_sensitivity.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_specificity_sensitivity.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_confusion_matrix.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_confusion_matrix.py:    def test_binary_confusion_matrix_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_confusion_matrix.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_confusion_matrix.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_confusion_matrix.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_confusion_matrix.py:    def test_multiclass_confusion_matrix_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_confusion_matrix.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_confusion_matrix.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_confusion_matrix.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_confusion_matrix.py:    def test_multilabel_confusion_matrix_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_confusion_matrix.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_confusion_matrix.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_precision_recall.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_precision_recall.py:    def test_binary_precision_recall_half_gpu(self, inputs, module, functional, compare, dtype):
tests/unittests/classification/test_precision_recall.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_precision_recall.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_precision_recall.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_precision_recall.py:    def test_multiclass_precision_recall_half_gpu(self, inputs, module, functional, compare, dtype):
tests/unittests/classification/test_precision_recall.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_precision_recall.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_precision_recall.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_precision_recall.py:    def test_multilabel_precision_recall_half_gpu(self, inputs, module, functional, compare, dtype):
tests/unittests/classification/test_precision_recall.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_precision_recall.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_accuracy.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_accuracy.py:    def test_binary_accuracy_half_gpu(self, inputs, dtype):
tests/unittests/classification/test_accuracy.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_accuracy.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_accuracy.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_accuracy.py:    def test_multiclass_accuracy_half_gpu(self, inputs, dtype):
tests/unittests/classification/test_accuracy.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_accuracy.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_accuracy.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_accuracy.py:            # average=`macro` stays on GPU when `use_deterministic` is True. Otherwise syncs in `bincount`
tests/unittests/classification/test_accuracy.py:    def test_multiclass_accuracy_gpu_sync_points(
tests/unittests/classification/test_accuracy.py:        """Test GPU support of the metric, avoiding CPU sync points."""
tests/unittests/classification/test_accuracy.py:        # Wrap the default functional to attach `sync_debug_mode` as `run_precision_test_gpu` handles moving data
tests/unittests/classification/test_accuracy.py:        # onto the GPU, so we cannot set the debug mode outside the call
tests/unittests/classification/test_accuracy.py:            prev_sync_debug_mode = torch.cuda.get_sync_debug_mode()
tests/unittests/classification/test_accuracy.py:            torch.cuda.set_sync_debug_mode("error")
tests/unittests/classification/test_accuracy.py:                torch.cuda.set_sync_debug_mode(prev_sync_debug_mode)
tests/unittests/classification/test_accuracy.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_accuracy.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_accuracy.py:            # If you remove from this collection, please add items to `test_multiclass_accuracy_gpu_sync_points`
tests/unittests/classification/test_accuracy.py:    def test_multiclass_accuracy_gpu_sync_points_uptodate(
tests/unittests/classification/test_accuracy.py:        """Negative test for `test_multiclass_accuracy_gpu_sync_points`, to confirm completeness.
tests/unittests/classification/test_accuracy.py:        Tests that `test_multiclass_accuracy_gpu_sync_points` is kept up to date, explicitly validating that known
tests/unittests/classification/test_accuracy.py:        `test_multiclass_accuracy_gpu_sync_points`.
tests/unittests/classification/test_accuracy.py:        with pytest.raises(RuntimeError, match="called a synchronizing CUDA operation"):
tests/unittests/classification/test_accuracy.py:            self.test_multiclass_accuracy_gpu_sync_points(
tests/unittests/classification/test_accuracy.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_accuracy.py:    def test_multilabel_accuracy_half_gpu(self, inputs, dtype):
tests/unittests/classification/test_accuracy.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_accuracy.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_negative_predictive_value.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_negative_predictive_value.py:    def test_binary_negative_predictive_value_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_negative_predictive_value.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_negative_predictive_value.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_negative_predictive_value.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_negative_predictive_value.py:    def test_multiclass_negative_predictive_value_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_negative_predictive_value.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_negative_predictive_value.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_negative_predictive_value.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_negative_predictive_value.py:    def test_multilabel_negative_predictive_value_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_negative_predictive_value.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_negative_predictive_value.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_stat_scores.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_stat_scores.py:    def test_binary_stat_scores_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_stat_scores.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_stat_scores.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_stat_scores.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_stat_scores.py:    def test_multiclass_stat_scores_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_stat_scores.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_stat_scores.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_stat_scores.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_stat_scores.py:    def test_multilabel_stat_scores_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_stat_scores.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_stat_scores.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_ranking.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_ranking.py:    def test_multilabel_ranking_dtype_gpu(self, inputs, metric, functional_metric, ref_metric, dtype):
tests/unittests/classification/test_ranking.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_ranking.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_exact_match.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_exact_match.py:    def test_multiclass_exact_match_half_gpu(self, inputs, dtype):
tests/unittests/classification/test_exact_match.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_exact_match.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_exact_match.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_exact_match.py:    def test_multilabel_exact_match_half_gpu(self, inputs, dtype):
tests/unittests/classification/test_exact_match.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_exact_match.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_hinge.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_hinge.py:    def test_binary_hinge_loss_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_hinge.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_hinge.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_hinge.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_hinge.py:    def test_multiclass_hinge_loss_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_hinge.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_hinge.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_group_fairness.py:    def run_precision_test_gpu(
tests/unittests/classification/test_group_fairness.py:        """Test if a metric can be used with half precision tensors on gpu.
tests/unittests/classification/test_group_fairness.py:            device="cuda",
tests/unittests/classification/test_group_fairness.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_group_fairness.py:    def test_binary_fairness_half_gpu(self, inputs, dtype):
tests/unittests/classification/test_group_fairness.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_recall_fixed_precision.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_recall_fixed_precision.py:    def test_binary_recall_at_fixed_precision_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_recall_fixed_precision.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_recall_fixed_precision.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_recall_fixed_precision.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_recall_fixed_precision.py:    def test_multiclass_recall_at_fixed_precision_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_recall_fixed_precision.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_recall_fixed_precision.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_recall_fixed_precision.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_recall_fixed_precision.py:    def test_multiclass_recall_at_fixed_precision_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_recall_fixed_precision.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_recall_fixed_precision.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_auroc.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_auroc.py:    def test_binary_auroc_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_auroc.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_auroc.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_auroc.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_auroc.py:    def test_multiclass_auroc_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_auroc.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_auroc.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_auroc.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_auroc.py:    def test_multiclass_auroc_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_auroc.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_auroc.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_hamming_distance.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_hamming_distance.py:    def test_binary_hamming_distance_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_hamming_distance.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_hamming_distance.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_hamming_distance.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_hamming_distance.py:    def test_multiclass_hamming_distance_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_hamming_distance.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_hamming_distance.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_hamming_distance.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_hamming_distance.py:    def test_multilabel_hamming_distance_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_hamming_distance.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_hamming_distance.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_specificity.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_specificity.py:    def test_binary_specificity_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_specificity.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_specificity.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_specificity.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_specificity.py:    def test_multiclass_specificity_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_specificity.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_specificity.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_specificity.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_specificity.py:    def test_multilabel_specificity_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_specificity.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_specificity.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_sensitivity_specificity.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_sensitivity_specificity.py:    def test_binary_sensitivity_at_specificity_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_sensitivity_specificity.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_sensitivity_specificity.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_sensitivity_specificity.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_sensitivity_specificity.py:    def test_multiclass_sensitivity_at_specificity_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_sensitivity_specificity.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_sensitivity_specificity.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_sensitivity_specificity.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_sensitivity_specificity.py:    def test_multiclass_sensitivity_at_specificity_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_sensitivity_specificity.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_sensitivity_specificity.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_average_precision.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_average_precision.py:    def test_binary_average_precision_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_average_precision.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_average_precision.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_average_precision.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_average_precision.py:    def test_multiclass_average_precision_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_average_precision.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_average_precision.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_average_precision.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_average_precision.py:    def test_multiclass_average_precision_dtype_gpu(self, inputs, dtype):
tests/unittests/classification/test_average_precision.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_average_precision.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_f_beta.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_f_beta.py:    def test_binary_fbeta_score_half_gpu(self, inputs, module, functional, compare, dtype):
tests/unittests/classification/test_f_beta.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_f_beta.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_f_beta.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_f_beta.py:    def test_multiclass_fbeta_score_half_gpu(self, inputs, module, functional, compare, dtype):
tests/unittests/classification/test_f_beta.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_f_beta.py:        self.run_precision_test_gpu(
tests/unittests/classification/test_f_beta.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/classification/test_f_beta.py:    def test_multilabel_fbeta_score_half_gpu(self, inputs, module, functional, compare, dtype):
tests/unittests/classification/test_f_beta.py:        """Test dtype support of the metric on GPU."""
tests/unittests/classification/test_f_beta.py:        self.run_precision_test_gpu(
tests/unittests/text/test_perplexity.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/text/test_perplexity.py:    def test_perplexity_dtypes_gpu(self, preds, target, ignore_index, dtype):
tests/unittests/text/test_perplexity.py:        """Test dtype support of the metric on GPU."""
tests/unittests/text/test_perplexity.py:        self.run_precision_test_gpu(
tests/unittests/text/_helpers.py:        device: determine which device to run on, either 'cuda' or 'cpu'
tests/unittests/text/_helpers.py:        device: determine which device to run on, either 'cuda' or 'cpu'
tests/unittests/text/_helpers.py:        device: determine device, either "cpu" or "cuda"
tests/unittests/text/_helpers.py:    def run_precision_test_gpu(
tests/unittests/text/_helpers.py:        """Test if a metric can be used with half precision tensors on gpu.
tests/unittests/text/_helpers.py:            metric_module(**metric_args), metric_functional, preds, targets, device="cuda", **kwargs_update
tests/unittests/pairwise/test_pairwise_distance.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/pairwise/test_pairwise_distance.py:    def test_pairwise_half_gpu(self, x, y, metric_functional, sk_fn, reduction):
tests/unittests/pairwise/test_pairwise_distance.py:        """Test half precision support on gpu."""
tests/unittests/pairwise/test_pairwise_distance.py:        self.run_precision_test_gpu(x, y, None, metric_functional, metric_args={"reduction": reduction})
tests/unittests/wrappers/test_feature_share.py:@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
tests/unittests/wrappers/test_feature_share.py:    base_memory = torch.cuda.memory_allocated()
tests/unittests/wrappers/test_feature_share.py:    fid = FrechetInceptionDistance(feature=64).cuda()
tests/unittests/wrappers/test_feature_share.py:    inception = InceptionScore(feature=64).cuda()
tests/unittests/wrappers/test_feature_share.py:    kid = KernelInceptionDistance(feature=64, subset_size=5).cuda()
tests/unittests/wrappers/test_feature_share.py:    memory_before_fs = torch.cuda.memory_allocated()
tests/unittests/wrappers/test_feature_share.py:    torch.cuda.empty_cache()
tests/unittests/wrappers/test_feature_share.py:    feature_share = FeatureShare([fid, inception, kid]).cuda()
tests/unittests/wrappers/test_feature_share.py:    memory_after_fs = torch.cuda.memory_allocated()
tests/unittests/wrappers/test_feature_share.py:    img1 = torch.randint(255, (50, 3, 220, 220), dtype=torch.uint8).to("cuda")
tests/unittests/wrappers/test_feature_share.py:    img2 = torch.randint(255, (50, 3, 220, 220), dtype=torch.uint8).to("cuda")
tests/unittests/wrappers/test_feature_share.py:    assert "cuda" in str(res["FrechetInceptionDistance"].device)
tests/unittests/wrappers/test_feature_share.py:    assert "cuda" in str(res["InceptionScore"][0].device)
tests/unittests/wrappers/test_feature_share.py:    assert "cuda" in str(res["InceptionScore"][1].device)
tests/unittests/wrappers/test_feature_share.py:    assert "cuda" in str(res["KernelInceptionDistance"][0].device)
tests/unittests/wrappers/test_feature_share.py:    assert "cuda" in str(res["KernelInceptionDistance"][1].device)
tests/unittests/wrappers/test_bootstrapping.py:@pytest.mark.parametrize("device", ["cpu", "cuda"])
tests/unittests/wrappers/test_bootstrapping.py:    if device == "cuda" and not torch.cuda.is_available():
tests/unittests/wrappers/test_bootstrapping.py:        pytest.skip("Test with device='cuda' requires gpu")
tests/unittests/segmentation/test_utils.py:@pytest.mark.parametrize("device", ["cpu", "cuda"])
tests/unittests/segmentation/test_utils.py:    if device == "cuda" and not torch.cuda.is_available():
tests/unittests/segmentation/test_utils.py:        pytest.skip("CUDA device not available.")
tests/unittests/segmentation/test_utils.py:@pytest.mark.parametrize("device", ["cpu", "cuda"])
tests/unittests/segmentation/test_utils.py:    if device == "cuda" and not torch.cuda.is_available():
tests/unittests/segmentation/test_utils.py:        pytest.skip("CUDA device not available.")
tests/unittests/segmentation/test_utils.py:@pytest.mark.parametrize("device", ["cpu", "cuda"])
tests/unittests/segmentation/test_utils.py:    if device == "cuda" and not torch.cuda.is_available():
tests/unittests/segmentation/test_utils.py:        pytest.skip("CUDA device not available.")
tests/unittests/segmentation/test_utils.py:@pytest.mark.parametrize("device", ["cpu", "cuda"])
tests/unittests/segmentation/test_utils.py:    if device == "cuda" and not torch.cuda.is_available():
tests/unittests/segmentation/test_utils.py:        pytest.skip("CUDA device not available.")
tests/unittests/segmentation/test_utils.py:@pytest.mark.parametrize("device", ["cpu", "cuda"])
tests/unittests/segmentation/test_utils.py:    if device == "cuda" and not torch.cuda.is_available():
tests/unittests/segmentation/test_utils.py:        pytest.skip("CUDA device not available.")
tests/unittests/utilities/test_utilities.py:@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires gpu")
tests/unittests/utilities/test_utilities.py:    """Test that bincount works in deterministic setting on GPU."""
tests/unittests/utilities/test_utilities.py:@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU")
tests/unittests/utilities/test_utilities.py:    """Make sure that cumsum on gpu and deterministic mode still fails.
tests/unittests/utilities/test_utilities.py:    with pytest.raises(RuntimeError, match="cumsum_cuda_kernel does not have a deterministic implementation.*"):
tests/unittests/utilities/test_utilities.py:        torch.arange(10).float().cuda().cumsum(0)
tests/unittests/utilities/test_utilities.py:@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU")
tests/unittests/utilities/test_utilities.py:    # check that cumsum works as expected on non-default cuda device
tests/unittests/utilities/test_utilities.py:    device = torch.device("cuda:1") if torch.cuda.device_count() > 1 else torch.device("cuda:0")
tests/unittests/utilities/test_utilities.py:            TorchMetricsUserWarning, match="You are trying to use a metric in deterministic mode on GPU that.*"
tests/unittests/bases/test_collections.py:@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
tests/unittests/bases/test_collections.py:        assert metric.x.is_cuda is False
tests/unittests/bases/test_collections.py:    metric_collection = metric_collection.to(device="cuda")
tests/unittests/bases/test_collections.py:        assert metric.x.is_cuda
tests/unittests/bases/test_metric.py:@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
tests/unittests/bases/test_metric.py:    assert metric.x.is_cuda is False
tests/unittests/bases/test_metric.py:    metric = metric.to(device="cuda")
tests/unittests/bases/test_metric.py:    assert metric.x.is_cuda
tests/unittests/bases/test_metric.py:    assert metric.device == torch.device("cuda", index=0)
tests/unittests/bases/test_metric.py:@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
tests/unittests/bases/test_metric.py:    metric.to(device="cuda")
tests/unittests/bases/test_metric.py:    is_cuda = (
tests/unittests/bases/test_metric.py:        metric._forward_cache[0].is_cuda if isinstance(metric._forward_cache, list) else metric._forward_cache.is_cuda
tests/unittests/bases/test_metric.py:    assert is_cuda, "forward cache was not moved to the correct device"
tests/unittests/bases/test_metric.py:    is_cuda = metric._computed[0].is_cuda if isinstance(metric._computed, list) else metric._computed.is_cuda
tests/unittests/bases/test_metric.py:    assert is_cuda, "computed result was not moved to the correct device"
tests/unittests/bases/test_metric.py:@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
tests/unittests/bases/test_metric.py:    module.to(device="cuda")
tests/unittests/bases/test_metric.py:@pytest.mark.parametrize("device", ["cpu", "cuda"])
tests/unittests/bases/test_metric.py:    if not torch.cuda.is_available() and device == "cuda":
tests/unittests/bases/test_metric.py:        pytest.skip("Test requires GPU support")
tests/unittests/bases/test_metric.py:        return torch.cuda.memory_allocated()
tests/unittests/bases/test_metric.py:@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
tests/unittests/bases/test_metric.py:        return torch.cuda.memory_allocated() / 1024**2
tests/unittests/bases/test_metric.py:        _ = DummyListMetric(compute_with_cache=False).cuda()
tests/unittests/bases/test_metric.py:@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
tests/unittests/bases/test_metric.py:        return torch.cuda.memory_allocated() / 1024**2
tests/unittests/bases/test_metric.py:    m = DummyListMetric().cuda()
tests/unittests/bases/test_metric.py:        m(x=torch.randn(10000).cuda())
tests/unittests/bases/test_metric.py:@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires gpu")
tests/unittests/bases/test_metric.py:    preds = torch.tensor(range(10), device="cuda", dtype=torch.float)
tests/unittests/bases/test_metric.py:    target = torch.tensor(range(10), device="cuda", dtype=torch.float)
tests/unittests/bases/test_metric.py:@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/bases/test_metric.py:    x = torch.randn(10).cuda()
tests/unittests/bases/test_saving_loading.py:@pytest.mark.parametrize("in_device", ["cpu", "cuda"])
tests/unittests/bases/test_saving_loading.py:@pytest.mark.parametrize("out_device", ["cpu", "cuda"])
tests/unittests/bases/test_saving_loading.py:    if (in_device == "cuda" or out_device == "cuda") and not torch.cuda.is_available():
tests/unittests/bases/test_saving_loading.py:        pytest.skip("Test requires cuda, but GPU not available.")
tests/unittests/_helpers/testers.py:        device: determine which device to run on, either 'cuda' or 'cpu'
tests/unittests/_helpers/testers.py:        device: determine which device to run on, either 'cuda' or 'cpu'
tests/unittests/_helpers/testers.py:        device: determine device, either "cpu" or "cuda"
tests/unittests/_helpers/testers.py:    nb_gpus = torch.cuda.device_count()
tests/unittests/_helpers/testers.py:    # if nb_gpus > 1:
tests/unittests/_helpers/testers.py:    #     return f"cuda:{randrange(nb_gpus)}"
tests/unittests/_helpers/testers.py:    if nb_gpus:
tests/unittests/_helpers/testers.py:        return "cuda"
tests/unittests/_helpers/testers.py:    def run_precision_test_gpu(
tests/unittests/_helpers/testers.py:        """Test if a metric can be used with half precision tensors on gpu.
tests/unittests/_helpers/testers.py:            device="cuda",
tests/unittests/_helpers/__init__.py:    torch.cuda.manual_seed_all(seed)
tests/unittests/__init__.py:if torch.cuda.is_available():
tests/unittests/__init__.py:    torch.backends.cuda.matmul.allow_tf32 = False
tests/unittests/image/test_kid.py:@pytest.mark.skipif(not torch.cuda.is_available(), reason="test is too slow without gpu")
tests/unittests/image/test_kid.py:    metric = KernelInceptionDistance(feature=feature, subsets=1, subset_size=100).cuda()
tests/unittests/image/test_kid.py:        metric.update(img1[batch_size * i : batch_size * (i + 1)].cuda(), real=True)
tests/unittests/image/test_kid.py:        metric.update(img2[batch_size * i : batch_size * (i + 1)].cuda(), real=False)
tests/unittests/image/test_qnr.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/image/test_qnr.py:    def test_quality_with_no_reference_half_gpu(
tests/unittests/image/test_qnr.py:        """Test dtype support of the metric on GPU."""
tests/unittests/image/test_qnr.py:        self.run_precision_test_gpu(
tests/unittests/image/test_mifid.py:@pytest.mark.skipif(not torch.cuda.is_available(), reason="test is too slow without gpu")
tests/unittests/image/test_mifid.py:    metric = MemorizationInformedFrechetInceptionDistance(feature=768).cuda()
tests/unittests/image/test_mifid.py:        metric.update(img1[batch_size * i : batch_size * (i + 1)].cuda(), real=True)
tests/unittests/image/test_mifid.py:        metric.update(img2[batch_size * i : batch_size * (i + 1)].cuda(), real=False)
tests/unittests/image/test_perceptual_path_length.py:@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
tests/unittests/image/test_perceptual_path_length.py:        generator, num_samples=50000, conditional=False, lower_discard=None, upper_discard=None, device="cuda"
tests/unittests/image/test_d_s.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/image/test_d_s.py:    def test_d_s_half_gpu(self, preds, target, ms, pan, pan_lr, norm_order, window_size):
tests/unittests/image/test_d_s.py:        """Test dtype support of the metric on GPU."""
tests/unittests/image/test_d_s.py:        self.run_precision_test_gpu(
tests/unittests/image/test_uqi.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/image/test_uqi.py:    def test_uqi_half_gpu(self, preds, target, multichannel, kernel_size):
tests/unittests/image/test_uqi.py:        """Test dtype support of the metric on GPU."""
tests/unittests/image/test_uqi.py:        self.run_precision_test_gpu(
tests/unittests/image/test_fid.py:@pytest.mark.skipif(not torch.cuda.is_available(), reason="test is too slow without gpu")
tests/unittests/image/test_fid.py:    metric = FrechetInceptionDistance(feature=feature).cuda()
tests/unittests/image/test_fid.py:        metric.update(img1[batch_size * i : batch_size * (i + 1)].cuda(), real=True)
tests/unittests/image/test_fid.py:        metric.update(img2[batch_size * i : batch_size * (i + 1)].cuda(), real=False)
tests/unittests/image/test_tv.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/image/test_tv.py:    def test_sam_half_gpu(self, preds, target, reduction):
tests/unittests/image/test_tv.py:        """Test for half precision on GPU."""
tests/unittests/image/test_tv.py:        self.run_precision_test_gpu(preds, target, TotalVariationTester, _total_variaion_wrapped)
tests/unittests/image/test_csi.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/image/test_csi.py:    def test_csi_half_gpu(self, preds, target, threshold):
tests/unittests/image/test_csi.py:        """Test dtype support of the metric on GPU."""
tests/unittests/image/test_csi.py:        self.run_precision_test_gpu(
tests/unittests/image/test_inception.py:@pytest.mark.skipif(not torch.cuda.is_available(), reason="test is too slow without gpu")
tests/unittests/image/test_inception.py:    metric = InceptionScore(splits=1, compute_on_cpu=compute_on_cpu).cuda()
tests/unittests/image/test_inception.py:        metric.update(img1[batch_size * i : batch_size * (i + 1)].cuda())
tests/unittests/image/test_ergas.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/image/test_ergas.py:    def test_ergas_half_gpu(self, reduction, preds, target, ratio):
tests/unittests/image/test_ergas.py:        """Test dtype support of the metric on GPU."""
tests/unittests/image/test_ergas.py:        self.run_precision_test_gpu(
tests/unittests/image/test_psnr.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/image/test_psnr.py:    def test_psnr_half_gpu(self, preds, target, data_range, reduction, dim, base, ref_metric):
tests/unittests/image/test_psnr.py:        """Test dtype support of the metric on GPU."""
tests/unittests/image/test_psnr.py:        self.run_precision_test_gpu(
tests/unittests/image/test_psnrb.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/image/test_psnrb.py:    def test_psnr_half_gpu(self, preds, target):
tests/unittests/image/test_psnrb.py:        """Test that PSNRB metric works with half precision on gpu."""
tests/unittests/image/test_psnrb.py:        self.run_precision_test_gpu(
tests/unittests/image/test_lpips.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/image/test_lpips.py:    def test_lpips_half_gpu(self):
tests/unittests/image/test_lpips.py:        """Test dtype support of the metric on GPU."""
tests/unittests/image/test_lpips.py:        self.run_precision_test_gpu(_inputs.img1, _inputs.img2, LearnedPerceptualImagePatchSimilarity)
tests/unittests/image/test_d_lambda.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/image/test_d_lambda.py:    def test_d_lambda_half_gpu(self, preds, target, p):
tests/unittests/image/test_d_lambda.py:        """Test dtype support of the metric on GPU."""
tests/unittests/image/test_d_lambda.py:        self.run_precision_test_gpu(preds, target, SpectralDistortionIndex, spectral_distortion_index, {"p": p})
tests/unittests/image/test_sam.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/image/test_sam.py:    def test_sam_half_gpu(self, reduction, preds, target):
tests/unittests/image/test_sam.py:        """Test dtype support of the metric on GPU."""
tests/unittests/image/test_sam.py:        self.run_precision_test_gpu(preds, target, SpectralAngleMapper, spectral_angle_mapper)
tests/unittests/image/test_ssim.py:    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
tests/unittests/image/test_ssim.py:    def test_ssim_half_gpu(self, preds, target, sigma):
tests/unittests/image/test_ssim.py:        """Test dtype support of the metric on GPU."""
tests/unittests/image/test_ssim.py:        self.run_precision_test_gpu(
tests/unittests/conftest.py:NUM_PROCESSES = 2  # torch.cuda.device_count() if torch.cuda.is_available() else 2
tests/integrations/test_lightning.py:    # is_cuda = torch.cuda.is_available()
tests/integrations/test_lightning.py:    # cuda_extra = {"devices": int(is_cuda)} if is_cuda else {}
tests/integrations/test_lightning.py:        # **cuda_extra,
tests/integrations/test_lightning.py:    # is_cuda = torch.cuda.is_available()
tests/integrations/test_lightning.py:    # cuda_extra = {"devices": int(is_cuda)} if is_cuda else {}
tests/integrations/test_lightning.py:        # **cuda_extra,
README.md:**This can be run on CPU, single GPU or multi-GPUs!**
README.md:For the single GPU/CPU case:
README.md:device = "cuda" if torch.cuda.is_available() else "cpu"
README.md:Module metric usage remains the same when using multiple GPUs or multiple nodes.
README.md:    world_size = 2  # number of gpus to parallelize over
requirements/_integrate.txt:# ToDo: investigate and add validation with 2.0+ on GPU
requirements/audio.txt:onnxruntime >=1.12.0, <1.21 # installing onnxruntime_gpu-gpu failed on macos
dockers/README.md:docker image build -t torchmetrics:latest -f dockers/ubuntu-cuda/Dockerfile .
dockers/README.md:docker image build -t torchmetrics:ubuntu-cuda11.7.1-py3.9-torch1.13 \
dockers/README.md:  -f dockers/base-cuda/Dockerfile \
dockers/README.md:  --build-arg CUDA_VERSION=11.7.1 \
dockers/README.md:## Run docker image with GPUs
dockers/README.md:To run docker image with access to your GPUs, you need to install
dockers/README.md:curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
dockers/README.md:curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
dockers/README.md:sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
dockers/README.md:and later run the docker image with `--gpus all`. For example,
dockers/README.md:docker run --rm -it --gpus all torchmetrics:ubuntu-cuda11.7.1-py3.9-torch1.12
dockers/ubuntu-cuda/Dockerfile:ARG CUDA_VERSION=11.7.1
dockers/ubuntu-cuda/Dockerfile:FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}
dockers/ubuntu-cuda/Dockerfile:    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
dockers/ubuntu-cuda/Dockerfile:    CUDA_VERSION_MM=${CUDA_VERSION%.*} && \
dockers/ubuntu-cuda/Dockerfile:    CU_VERSION_MM=${CUDA_VERSION_MM//'.'/''} && \
dockers/ubuntu-cuda/Dockerfile:    # switch some packages to be GPU related
dockers/ubuntu-cuda/Dockerfile:    python assistant.py replace_str_requirements "onnxruntime" "onnxruntime_gpu" --req_files requirements/audio.txt && \
src/torchmetrics/regression/log_mse.py:        Half precision is only support on GPU for this metric.
src/torchmetrics/regression/kl_divergence.py:        Half precision is only support on GPU for this metric.
src/torchmetrics/audio/srmr.py:            setting `fast=True` may slow down the speed for calculating this metric on GPU.
src/torchmetrics/audio/sdr.py:        The metric currently does not seem to work with Pytorch v1.11 and specific GPU hardware.
src/torchmetrics/audio/dnsmos.py:        Install as ``pip install torchmetrics['audio']`` or alternatively `pip install librosa onnxruntime-gpu requests`
src/torchmetrics/audio/dnsmos.py:        (if you do not have GPU enabled machine install `onnxruntime` instead of `onnxruntime-gpu`)
src/torchmetrics/audio/dnsmos.py:        device: the device used for calculating DNSMOS, can be cpu or cuda:n, where n is the index of gpu.
src/torchmetrics/audio/dnsmos.py:                " Install as `pip install librosa onnxruntime-gpu requests`."
src/torchmetrics/detection/_mean_ap.py:        """Move list states to cpu to save GPU memory."""
src/torchmetrics/detection/_mean_ap.py:        # Convert to uint8 temporarily and back to bool, because "Sort currently does not support bool dtype on CUDA"
src/torchmetrics/detection/_mean_ap.py:        # Sort in PyTorch does not support bool types on CUDA (yet, 1.11.0)
src/torchmetrics/detection/_mean_ap.py:        dtype = torch.uint8 if det_scores.is_cuda and det_scores.dtype is torch.bool else det_scores.dtype
src/torchmetrics/detection/_mean_ap.py:        # Explicitly cast to uint8 to avoid error for bool inputs on CUDA to argsort
src/torchmetrics/functional/regression/log_mse.py:        Half precision is only support on GPU for this metric.
src/torchmetrics/functional/audio/srmr.py:            setting `fast=True` may slow down the speed for calculating this metric on GPU.
src/torchmetrics/functional/audio/srmr.py:        rank_zero_warn("`fast=True` may slow down the speed of SRMR metric on GPU.")
src/torchmetrics/functional/audio/sdr.py:        The metric currently does not seem to work with Pytorch v1.11 and specific GPU hardware.
src/torchmetrics/functional/audio/pit.py:    # reading from cache would be much faster than creating in CPU then moving to GPU
src/torchmetrics/functional/audio/dnsmos.py:    elif "CUDAExecutionProvider" in ort.get_available_providers():  # win or linux with cuda
src/torchmetrics/functional/audio/dnsmos.py:        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
src/torchmetrics/functional/audio/dnsmos.py:        as ``pip install torchmetrics['audio']`` or alternatively ``pip install librosa onnxruntime-gpu requests``
src/torchmetrics/functional/audio/dnsmos.py:        (if you do not have GPU enabled machine install ``onnxruntime`` instead of ``onnxruntime-gpu``)
src/torchmetrics/functional/audio/dnsmos.py:        device: the device used for calculating DNSMOS, can be cpu or cuda:n, where n is the index of gpu.
src/torchmetrics/functional/audio/dnsmos.py:            " Install as `pip install librosa onnxruntime-gpu requests`."
src/torchmetrics/functional/audio/dnsmos.py:            "CUDAExecutionProvider" in ort.get_available_providers()
src/torchmetrics/functional/audio/dnsmos.py:                rank_zero_warn(f"Failed to use GPU for DNSMOS, reverting to CPU. Error: {e}")
src/torchmetrics/functional/image/psnr.py:        Half precision is only support on GPU for this metric.
src/torchmetrics/metric.py:        """Move list states to cpu to save GPU memory."""
src/torchmetrics/metric.py:        This method is called by the base ``nn.Module`` class whenever `.to`, `.cuda`, `.float`, `.half` etc. methods
src/torchmetrics/utilities/data.py:    if x.dtype == torch.half and not x.is_cuda:
src/torchmetrics/utilities/data.py:    PyTorch currently does not support ``torch.bincount`` when running in deterministic mode on GPU or when running
src/torchmetrics/utilities/data.py:    if torch.are_deterministic_algorithms_enabled() and x.is_cuda and x.is_floating_point() and sys.platform != "win32":
src/torchmetrics/utilities/data.py:            "You are trying to use a metric in deterministic mode on GPU that uses `torch.cumsum`, which is currently "
src/torchmetrics/utilities/data.py:            "not supported. The tensor will be copied to the CPU memory to compute it and then copied back to GPU. "
src/conftest.py:        torch.cuda.manual_seed_all(seed)

```
