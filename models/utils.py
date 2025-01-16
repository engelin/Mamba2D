import os
import torch
import warnings
import lightning as L

from lightning import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_warn, rank_zero_info
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch.nn as nn
from einops import rearrange


from typing import Any, Dict, List, Optional

try:
    import amp_C
    apex_available = True
except Exception:
    apex_available = False

# Adapted from https://github.com/BioinfoMachineLearning/bio-diffusion/blob/e4bad15139815e562a27fb94dab0c31907522bc5/src/utils/__init__.py
class EMA(Callback):
    """
    Implements Exponential Moving Averaging (EMA).
    When training a model, this callback will maintain moving averages of the trained parameters.
    When evaluating, we use the moving averages copy of the trained parameters.
    When saving, we save an additional set of parameters with the prefix `ema`.
    Args:
        decay: The exponential decay used when calculating the moving average. Has to be between 0-1.
        apply_ema_every_n_steps: Apply EMA every n global steps.
        start_step: Start applying EMA from ``start_step`` global step onwards.
        save_ema_weights_in_callback_state: Enable saving EMA weights in callback state.
        evaluate_ema_weights_instead: Validate the EMA weights instead of the original weights.
            Note this means that when saving the model, the validation metrics are calculated with the EMA weights.

    Adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/callbacks/ema.py
    """

    def __init__(
        self,
        decay: float,
        apply_ema_every_n_steps: int = 1,
        start_step: int = 0,
        save_ema_weights_in_callback_state: bool = False,
        evaluate_ema_weights_instead: bool = False,
    ):
        if not apex_available:
            rank_zero_warn(
                "EMA has better performance when Apex is installed: https://github.com/NVIDIA/apex#installation."
            )
        if not (0 <= decay <= 1):
            raise MisconfigurationException("EMA decay value must be between 0 and 1")
        self._ema_model_weights: Optional[List[torch.Tensor]] = None
        self._overflow_buf: Optional[torch.Tensor] = None
        self._cur_step: Optional[int] = None
        self._weights_buffer: Optional[List[torch.Tensor]] = None
        self.apply_ema_every_n_steps = apply_ema_every_n_steps
        self.start_step = start_step
        self.save_ema_weights_in_callback_state = save_ema_weights_in_callback_state
        self.evaluate_ema_weights_instead = evaluate_ema_weights_instead
        self.decay = decay

    def on_train_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        print("Creating EMA weights copy.")
        if self._ema_model_weights is None:
            self._ema_model_weights = [p.detach().clone() for p in pl_module.state_dict().values()]
        # ensure that all the weights are on the correct device
        self._ema_model_weights = [p.to(pl_module.device) for p in self._ema_model_weights]
        self._overflow_buf = torch.IntTensor([0]).to(pl_module.device)

    def ema(self, pl_module: "L.LightningModule") -> None:
        if apex_available and pl_module.device.type == "cuda":
            return self.apply_multi_tensor_ema(pl_module)
        return self.apply_ema(pl_module)

    def apply_multi_tensor_ema(self, pl_module: "L.LightningModule") -> None:
        model_weights = list(pl_module.state_dict().values())
        amp_C.multi_tensor_axpby(
            65536,
            self._overflow_buf,
            [self._ema_model_weights, model_weights, self._ema_model_weights],
            self.decay,
            1 - self.decay,
            -1,
        )

    def apply_ema(self, pl_module: "L.LightningModule") -> None:
        for orig_weight, ema_weight in zip(list(pl_module.state_dict().values()), self._ema_model_weights):
            if ema_weight.data.dtype != torch.long and orig_weight.data.dtype != torch.long:
                # ensure that non-trainable parameters (e.g., feature distributions) are not included in EMA weight averaging
                diff = ema_weight.data - orig_weight.data
                diff.mul_(1.0 - self.decay)
                ema_weight.sub_(diff)

    def should_apply_ema(self, step: int) -> bool:
        return step != self._cur_step and step >= self.start_step and step % self.apply_ema_every_n_steps == 0

    def on_train_batch_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if self.should_apply_ema(trainer.global_step):
            self._cur_step = trainer.global_step
            self.ema(pl_module)

    def state_dict(self) -> Dict[str, Any]:
        if self.save_ema_weights_in_callback_state:
            return dict(cur_step=self._cur_step, ema_weights=self._ema_model_weights)
        return dict(cur_step=self._cur_step)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._cur_step = state_dict["cur_step"]
        # when loading within apps such as NeMo, EMA weights will be loaded by the experiment manager separately
        if self._ema_model_weights is None:
            self._ema_model_weights = state_dict.get("ema_weights")

    def on_load_checkpoint(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        checkpoint_callback = trainer.checkpoint_callback

        if trainer.ckpt_path and checkpoint_callback is not None:
            ext = checkpoint_callback.FILE_EXTENSION
            if trainer.ckpt_path.endswith(f"-EMA{ext}"):
                print(
                    "loading EMA based weights. "
                    "The callback will treat the loaded EMA weights as the main weights"
                    " and create a new EMA copy when training."
                )
                return
            ema_path = trainer.ckpt_path.replace(ext, f"-EMA{ext}")
            if os.path.exists(ema_path):
                ema_state_dict = torch.load(ema_path, map_location=torch.device("cpu"))
                self._ema_model_weights = ema_state_dict["state_dict"].values()
                del ema_state_dict
                print("EMA weights have been loaded successfully. Continuing training with saved EMA weights.")
            else:
                warnings.warn(
                    "we were unable to find the associated EMA weights when re-loading, "
                    "training will start with new EMA weights.",
                    UserWarning,
                )

    def replace_model_weights(self, pl_module: "L.LightningModule") -> None:
        self._weights_buffer = [p.detach().clone().to("cpu") for p in pl_module.state_dict().values()]
        new_state_dict = {k: v for k, v in zip(pl_module.state_dict().keys(), self._ema_model_weights)}
        pl_module.load_state_dict(new_state_dict)

    def restore_original_weights(self, pl_module: "L.LightningModule") -> None:
        state_dict = pl_module.state_dict()
        new_state_dict = {k: v for k, v in zip(state_dict.keys(), self._weights_buffer)}
        pl_module.load_state_dict(new_state_dict)
        del self._weights_buffer

    @property
    def ema_initialized(self) -> bool:
        return self._ema_model_weights is not None

    def on_validation_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.replace_model_weights(pl_module)

    def on_validation_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.restore_original_weights(pl_module)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.replace_model_weights(pl_module)

    def on_test_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.restore_original_weights(pl_module)

# Adapted from https://github.com/BioinfoMachineLearning/bio-diffusion/blob/e4bad15139815e562a27fb94dab0c31907522bc5/src/utils/__init__.py
class EMAModelCheckpoint(ModelCheckpoint):
    """
    Light wrapper around Lightning's `ModelCheckpoint` to, upon request, save an EMA copy of the model as well.

    Adapted from: https://github.com/NVIDIA/NeMo/blob/be0804f61e82dd0f63da7f9fe8a4d8388e330b18/nemo/utils/exp_manager.py#L744
    """

    def __init__(self, **kwargs):
        # call the parent class constructor with the provided kwargs
        super().__init__(**kwargs)

    def _get_ema_callback(self, trainer: "L.Trainer") -> Optional[EMA]:
        ema_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, EMA):
                ema_callback = callback
        return ema_callback

    def _save_checkpoint(self, trainer: "L.Trainer", filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        ema_callback = self._get_ema_callback(trainer)
        if ema_callback is not None:
            # save EMA copy of the model as well
            ema_callback.replace_model_weights(trainer.lightning_module)
            filepath = self._ema_format_filepath(filepath)
            if self.verbose:
                rank_zero_info(f"Saving EMA weights to separate checkpoint {filepath}")
            super()._save_checkpoint(trainer, filepath)
            ema_callback.restore_original_weights(trainer.lightning_module)

    def _ema_format_filepath(self, filepath: str) -> str:
        return filepath.replace(self.FILE_EXTENSION, f"-EMA{self.FILE_EXTENSION}")

        # Adapted from https://github.com/NVIDIA/NeMo/pull/5169#issuecomment-2047055655
    def _update_best_and_save(self, current: torch.Tensor, trainer: "pl.Trainer",
                            monitor_candidates: Dict[str, torch.Tensor]) -> None:
        k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k

        del_filepath = None
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)

        # do not save nan, replace with +/- inf
        if isinstance(current, torch.Tensor) and torch.isnan(current):
            current = torch.tensor(float("inf" if self.mode == "min" else "-inf"), device=current.device)

        filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer, del_filepath)

        # save the current score
        self.current_score = current
        self.best_k_models[filepath] = current

        if len(self.best_k_models) == k:
            # monitor dict has reached k elements
            _op = max if self.mode == "min" else min
            self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
            self.kth_value = self.best_k_models[self.kth_best_model_path]

        _op = min if self.mode == "min" else max
        self.best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
        self.best_model_score = self.best_k_models[self.best_model_path]

        if self.verbose:
            epoch = monitor_candidates["epoch"]
            step = monitor_candidates["step"]
            rank_zero_info(
                f"Epoch {epoch:d}, global step {step:d}: {self.monitor!r} reached {current:0.5f}"
                f" (best {self.best_model_score:0.5f}), saving model to {filepath!r} as top {k}"
            )
        self._save_checkpoint(trainer, filepath)

        if del_filepath is not None and filepath != del_filepath:
            self._remove_checkpoint(trainer, del_filepath)
            self._remove_checkpoint(trainer, del_filepath.replace(self.FILE_EXTENSION, f"-EMA{self.FILE_EXTENSION}"))

# Adding padding and unpadding to keep dims even when windowing/patching
# https://stackoverflow.com/questions/66028743/how-to-handle-odd-resolutions-in-unet-architecture-pytorch
def pad_to(x, stride):
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = torch.nn.functional.pad(x, pads, "constant", 0)

    return out, pads

def unpad(x, pad, sr_factor=1):
    if pad[2]+pad[3] > 0:
        x = x[:,:,pad[2]*sr_factor:-pad[3]*sr_factor,:]
    if pad[0]+pad[1] > 0:
        x = x[:,:,:,pad[0]*sr_factor:-pad[1]*sr_factor]
    return x

# Implement LayerNorm for 2D inputs
class LayerNorm2D(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        x = rearrange(x, "b h w c -> b c h w")

        return(x)

### Adapt following blocks from MetaFormer:
### https://github.com/sail-sg/metaformer/blob/main/metaformer_baselines.py

class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True, 
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias


class Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Adapted from timm
    """
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1
        
        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        
    def forward(self, x):
        B, H, W, C = x.shape

        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
    
class MLP(nn.Module):
    def __init__(self,
                 embed_dim: int = 64,
                 expand_factor: int = 2,
                 dropout: float = 0.,
                 act: torch.nn.Module = nn.GELU
                 ):
        super().__init__()

        self.embed_dim = embed_dim
        self.inner_dim = embed_dim * expand_factor
        self.dropout = dropout
    
        # Create block
        self.fc1 = nn.Linear(self.embed_dim, self.inner_dim)
        self.act = act()
        self.drop1 = nn.Dropout(p=self.dropout)
        self.fc2 = nn.Linear(self.inner_dim, self.embed_dim)
        self.drop2 = nn.Dropout(p=self.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.

    - Added inverted residual connection for better performance. 
    """
    def __init__(self, dim, expansion_ratio=2,
        act1_layer=StarReLU, act2_layer=nn.Identity, 
        bias=False, kernel_size=7, padding=3, residual = False,
        **kwargs,):
        super().__init__()
        self.residual = residual
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias) # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        shortcut = x 

        x = self.pwconv1(x) 
        x = self.act1(x)
        
        x = x.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)

        x = self.act2(x)
        x = self.pwconv2(x)
        x = x + shortcut if self.residual else x # Inverted residual (in the bottlenech layers)
        return x


class Downsampling(nn.Module):
    """
    Downsampling implemented by a layer of convolution + normalization.
    """
    def __init__(self, in_channels, out_channels, 
        kernel_size, stride=1, padding=0, 
        pre_norm=None, post_norm =None):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding)
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)
        x = self.conv(x)
        x = self.post_norm(x)
        return x


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class MlpHead(nn.Module):
    """ MLP classification head
    """
    def __init__(self,
                 dim: int,
                 n_classes: int = 1000,
                 mlp_ratio: int = 4,
                 head_dropout: float = 0.,
                 bias: bool =True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_features)
        self.fc2 = nn.Linear(hidden_features, n_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x

###--------------------------------------------------------------------------###
