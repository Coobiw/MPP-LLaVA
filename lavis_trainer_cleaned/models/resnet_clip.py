import clip
import torch.nn as nn
import torch
import torch.nn.functional as F

from .base_model import BaseModel
from .utils import TextAttentionPool2d, interpolate_pos_embed, disabled_train, freeze_module, MLPHead

from common.registry import registry

@registry.register_model("resnet50_clip_image2prompt")
class resnet50_clip_image2prompt(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/resnet50_clip_default.yaml",
    }
    def __init__(self,
                base_ckpt='./ckpt/clip/openai/RN50.pt',
                input_resolution=224,
                output_dim=None,
                freeze_bn=False,
                head_scale=None,
                use_mlp_head=False,
                dropout_rate=0.,
        ):
        super().__init__()
        self.resnet50 = clip.load(base_ckpt, device="cpu")[0].visual
        self.resnet50 = self.resnet50.float()
        self.feature_dim = self.resnet50.attnpool.c_proj.out_features
        self.resnet50.attnpool.positional_embedding = nn.Parameter(
                interpolate_pos_embed(self.resnet50.attnpool.positional_embedding,input_resolution=input_resolution))
        if use_mlp_head and output_dim:
            self.head = MLPHead(self.feature_dim,output_dim,dropout_rate)
        else:
            self.head = nn.Linear(self.feature_dim,output_dim) if output_dim else nn.Identity()

        self.head_scale = head_scale

        if freeze_bn:
            print("Freezing BN ...")
            for layer in self.resnet50.modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    for param in layer.parameters():
                        param.requires_grad = False
                    layer.eval()
                    layer.train = disabled_train


    def forward(self,x):
        feat = self.resnet50(x)
        return self.head(feat)

    def train_step(self,samples):
        x,y = samples['images'], samples['prompt_embeddings']
        output = self(x)
        criterion = nn.CosineEmbeddingLoss()
        target = torch.ones(x.size(0)).to(x.device)
        loss = criterion(output, y, target)
        loss_dict = {"loss": loss.detach().clone()}
        return loss,loss_dict

    def get_optimizer_params(self, weight_decay, lr_scale=1):
        p_wd, p_non_wd = [], []
        if self.head_scale:
            p_head = []
            p_head_non_wd = []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            if self.head_scale and 'head' in n:
                if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                    p_head_non_wd.append(p)
                else:
                    p_head.append(p)
            else:
                if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                    p_non_wd.append(p)
                else:
                    p_wd.append(p)
        if self.head_scale:
            optim_params = [
                {"params": p_wd, "weight_decay": weight_decay, "lr_scale": lr_scale},
                {"params": p_non_wd, "weight_decay": 0, "lr_scale": lr_scale},
                {"params": p_head, "weight_decay": weight_decay, "lr_scale": self.head_scale},
                {"params": p_head_non_wd, "weight_decay": 0, "lr_scale": self.head_scale},
            ]
            print(f"head scale: {self.head_scale}")
        else:
            optim_params = [
                {"params": p_wd, "weight_decay": weight_decay, "lr_scale": lr_scale},
                {"params": p_non_wd, "weight_decay": 0, "lr_scale": lr_scale},
            ]
        return optim_params

    @classmethod
    def from_config(cls, cfg):
        base_ckpt = cfg.get('base_ckpt','./ckpt/clip/openai/RN50.pt')
        input_resolution = cfg.get("input_resolution",224)
        output_dim = cfg.get("output_dim",None)
        freeze_bn = cfg.get('freeze_bn',False)
        new_head = cfg.get('new_head',False)
        new_dim = cfg.get('new_dim',None)
        head_scale = cfg.get('head_scale',None)
        use_mlp_head = cfg.get('use_mlp_head',False)
        dropout_rate = cfg.get('dropout_rate',0.)

        model = cls(base_ckpt,input_resolution,output_dim,freeze_bn,head_scale,use_mlp_head,dropout_rate)

        load_finetuned = cfg.get("load_finetuned",False)  # init里load了pretrain的，所以不需要再load了
        if load_finetuned:
            model.load_checkpoint_from_config(cfg)
            if new_head:
                print('Establish new head ...')
                in_dim = model.head.in_features
                out_dim = new_dim if new_dim else output_dim
                model.head = nn.Linear(in_dim,out_dim)
        return model

@registry.register_model("resnet50_clip_cls")
class resnet50_clip_cls(resnet50_clip_image2prompt):
    def train_step(self,samples):
        x,y = samples['images'], samples['labels']
        output = self(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, y)
        loss_dict = {"loss": loss.detach().clone()}
        return loss,loss_dict

    def val_step(self,samples):
        x,y = samples['images'], samples['labels']
        output = self(x)
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss = criterion(output, y)
        loss_np = loss.detach().cpu().numpy().tolist()
        pred_np = output.detach().cpu().numpy().tolist()
        label_np = y.detach().cpu().numpy().tolist()
        ret = zip(loss_np,pred_np,label_np)
        return ret

if __name__ == "__main__":
    model = resnet50_clip_cls(input_resolution=512,output_dim=1)
    x = torch.ones((1,3,512,512))
    print(model(x).shape)