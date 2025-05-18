import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"

import os
import sys
import torch
import importlib.util

# Load .so
this_dir = os.path.dirname(__file__)
so_path = os.path.join(this_dir, "wf_cuda/wavefront_cuda.cpython-311-x86_64-linux-gnu.so")

if not os.path.exists(so_path):
    raise ImportError(f"[ERROR] Prebuilt extension not found at {so_path}")

spec = importlib.util.spec_from_file_location("wavefront_cuda", so_path)
wavefront_cuda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wavefront_cuda)

# Bind .so
wf_fwd = wavefront_cuda.wf_fwd
wf_bwd = wavefront_cuda.wf_bwd

class wf_cuda_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, deltaAT, deltaAL, BXT, BXL):
        # ΔAT : (B, H, W, ED, N)
        # ΔAL : (B, H, W, ED, N)
        # BXT : (B, H, W, ED, N)
        # BXL : (B, H, W, ED, N)

        hs = torch.zeros_like(deltaAT, device=deltaAT.device)

        # Launch kernel
        wf_fwd(hs, deltaAT, deltaAL, BXT, BXL)

        ctx.save_for_backward(hs, deltaAT, deltaAL)

        return hs

    def backward(ctx, grad_output):
        # ctx: 
        #   - hs  : (B, H, W, ED, N)
        #   - ΔAT : (B, H, W, ED, N)
        #   - ΔAL : (B, H, W, ED, N)
        # grad_output: (B, H, W, ED, N)

        hs, deltaAT, deltaAL = ctx.saved_tensors

        # Special init to avoid tweaks inside loop
        omega = torch.full_like(hs, 0.5, device=hs.device)
        # dDAT/L is used as a mask for special cases in calculating
        # omega and gets overwritten for final result
        dDAT = torch.ones_like(hs, device=hs.device)
        dDAL = torch.ones_like(hs, device=hs.device)
        # mask top/left row/col special cases + bake in doubling
        dDAT[:,:,0,:,:] = 2
        dDAL[:,0,:,:,:] = 2

        # For 2* on top/left row/col
        dBX = torch.ones_like(hs, device=hs.device)
        dBX[:,1:,1:,:,:] = 0.5

        # Launch kernel
        wf_bwd(hs, deltaAT, deltaAL, grad_output, dDAT, dDAL, omega, dBX)

        # Fudge cols/rows as appropriate
        dBXT = dBX.clone(); dBXT[:,0,:,:,:] = 0
        dBXL = dBX.clone(); dBXL[:,1:,0,:,:] = 0

        return dDAT, dDAL, dBXT, dBXL

# Just to make fn name pretty
@torch.compiler.disable
def wavefront_scan_cuda(deltaAT, deltaAL, BXT, BXL):
    return wf_cuda_fn.apply(deltaAT, deltaAL, BXT, BXL)
