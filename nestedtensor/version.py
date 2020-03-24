__version__ = '0.0.1.dev202032420+84ce853'
git_version = '84ce85399f114d790a6905e074847b93a75267cc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
