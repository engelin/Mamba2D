import torch
import torch.nn as nn
import lightning as L

from timm.loss import SoftTargetCrossEntropy
from schedulefree import AdamWScheduleFree
from torchmetrics.classification import Accuracy

class Mamba2DClassifier(L.LightningModule):
    def __init__(self,
                 # Model Params
                 backbone: torch.nn.Module,
                 head: torch.nn.Module,
                 
                 # Input Params
                 n_classes: int = 1000,
                 cutmix: bool = True,

                 # Optimiser Params
                 lr: float = 0.004,
                 warmup_pct: float = 0.05,
                 ):
        super().__init__()

        # Create model
        self.n_classes = n_classes

        self.backbone = backbone

        self.head = head

        self.norm = nn.LayerNorm(backbone.embed_dim[-1])

        # Losses
        if (cutmix):
            self.train_loss = SoftTargetCrossEntropy()
        else:
            self.train_loss = nn.CrossEntropyLoss()

        self.val_loss = nn.CrossEntropyLoss()

        # Metrics
        self.accuracy = Accuracy(task="multiclass", num_classes=n_classes)

        # Optimiser
        self.lr = lr

        # Check warmup_pct is within 0-1
        if (0 <= warmup_pct < 1): self.warmup_pct = warmup_pct
        else: raise ValueError(f"warmup_pct ({warmup_pct}) must be between 0.0 "
                                "(0%) and 1.0 (100%)!")

        # Set model to channels last if required
        if backbone.channel_last: self.to(memory_format=torch.channels_last)

    def setup(self, stage):
        print(self)

    def forward(self, x):
        x = self.backbone(x)

        # x.mean[-2,-1] applies avg pooling on H,W dims
        x = self.head(self.norm(x.mean([-2,-1])))

        return x

    # Set optimiser states for schedulefree AdamW
    def set_optim_mode(self, mode):
        optim = self.optimizers()
        if isinstance(optim, list):
            for opt in optim: getattr(opt, mode)()
        else: getattr(optim, mode)()

    def on_train_epoch_start(self): self.set_optim_mode('train')
    def on_train_epoch_end(self): self.set_optim_mode('eval')
    def on_save_checkpoint(self, checkpoint): self.set_optim_mode('eval')
    def on_validation_start(self): self.set_optim_mode('eval')
    def on_predict_start(self): self.set_optim_mode('eval')
    def on_test_start(self): self.set_optim_mode('eval')

    def training_step(self, batch, batch_idx):
        x, labels = batch

        preds = self(x)

        loss = self.train_loss(preds, labels)

        self.log_dict({
                "train/ce_loss" : loss,
                "train/accuracy" : self.accuracy(preds, labels),
            },
            batch_size = self.trainer.datamodule.batch_size,
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )

        return(loss)

    def validation_step(self, batch, batch_idx):
        x, labels = batch

        preds = self(x)

        loss = self.val_loss(preds, labels)

        self.log_dict({
                "val/ce_loss" : loss,
                "val/accuracy" : self.accuracy(preds, labels),
            },
            batch_size = self.trainer.datamodule.batch_size,
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        warmup_steps = int(self.trainer.estimated_stepping_batches * self.warmup_pct)
        return AdamWScheduleFree(params, self.lr, warmup_steps=warmup_steps, weight_decay=0.05)
