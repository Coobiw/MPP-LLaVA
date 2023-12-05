from torch.utils.data import DataLoader

from common.dist_utils import get_world_size, get_rank
from torch.utils.data.distributed import DistributedSampler

def get_dataloaders(
    datasets,
    batch_size,
    batch_size_val,
    num_worker,
    ddp=False,
):
    trn_dataset = datasets['train']
    val_dataset = datasets['val']
    train_collator = getattr(trn_dataset,"collator",None)
    val_collator = getattr(val_dataset,"collator",None)

    dataloaders = {}
    train_sampler = DistributedSampler(
        trn_dataset,
        shuffle=True,
        num_replicas=get_world_size(),
        rank=get_rank()
    ) if ddp else None
    val_sampler = DistributedSampler(
        val_dataset,
        shuffle=False,
        num_replicas=get_world_size(),
        rank=get_rank()
    ) if ddp else None
    dataloaders['train'] = DataLoader(
        dataset=trn_dataset,
        shuffle=train_sampler is None,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_worker,
        drop_last=True,
        collate_fn=train_collator,
        sampler=train_sampler,
    )
    dataloaders['val'] = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        batch_size=batch_size_val,
        pin_memory=True,
        num_workers=num_worker,
        drop_last=False,
        collate_fn=val_collator,
        sampler=val_sampler,
    )
    return dataloaders