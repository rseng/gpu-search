# https://github.com/mne-tools/mne-icalabel

```console
mne_icalabel/utils/config.pyi:def _get_gpu_info() -> tuple[str | None, str | None]:
mne_icalabel/utils/config.pyi:    """Get the GPU information."""
mne_icalabel/utils/tests/test_config.py:from mne_icalabel.utils.config import _get_gpu_info, sys_info
mne_icalabel/utils/tests/test_config.py:def test_gpu_info():
mne_icalabel/utils/tests/test_config.py:    """Test getting GPU info."""
mne_icalabel/utils/tests/test_config.py:    version, renderer = _get_gpu_info()
mne_icalabel/utils/config.py:            version_, renderer = _get_gpu_info()
mne_icalabel/utils/config.py:def _get_gpu_info() -> tuple[Optional[str], Optional[str]]:
mne_icalabel/utils/config.py:    """Get the GPU information."""
mne_icalabel/utils/config.py:        from pyvista import GPUInfo
mne_icalabel/utils/config.py:        gi = GPUInfo()

```
