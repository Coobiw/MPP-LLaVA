from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

class BaseDataset(Dataset):
    def __init__(
        self, transform,
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.transform = transform

        self.collater = default_collate

    @classmethod
    def from_config(cls,config):
        return cls(**config)
