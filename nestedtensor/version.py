__version__ = '0.0.1.dev202032420+651f682'
git_version = '651f682618c705a647f56397091d31bbb96812f0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
