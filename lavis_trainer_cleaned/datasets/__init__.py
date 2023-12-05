from .img2prompt_dataset import Image2PromptDataset
from .cls_datasets import ExampleClsDataset

from common.registry import registry

__all__ = [
    'Image2PromptDataset',
    'ExampleClsDataset',
]

def load_dataset(name, cfg=None):
    """
    Example

    >>> processor = load_dataset("example_cls_dataset", cfg=None)
    """
    dataset = registry.get_dataset_class(name).from_config(cfg)

    return dataset