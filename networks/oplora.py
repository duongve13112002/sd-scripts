"""OPLoRA: orthogonal-projection LoRA for anti catastrophic-forgetting (LoRA training only).

Confines each LoRA update to the orthogonal complement of the base weight's top-k singular
subspace, so the base model's dominant directions are provably preserved while the adapter
learns in the remaining subspace. There is no teacher and no extra forward pass: the basis is
computed once from each base weight (SVD), and after every optimizer step the up/down factors
are projected back into the orthogonal complement.

For a base weight W = U S V^T with top-k U_k (out x k) and V_k (in x k), and a LoRA delta
dW = up @ down, the projected delta is P_L dW P_R with P_L = I - U_k U_k^T and P_R = I - V_k V_k^T.
This factors into a cheap low-rank correction of the factors:

    up'  = up   - U_k (U_k^T up)
    down' = down - (down V_k) V_k^T

After projection U_k^T up' = 0 and down' V_k = 0, so dW' is orthogonal to the top-k left and
right singular subspaces; hence W + dW' keeps W's top-k singular triples unchanged.

This is LoRA-only: it operates on the LoRA factors. The arguments are registered only by the
LoRA trainer's parser, so passing them to a full fine-tune script is rejected by argparse.
"""

import argparse
import logging

import torch

from library.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def is_oplora_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "oplora", False))


def _all_loras(network):
    loras = []
    for attr in ("text_encoder_loras", "unet_loras"):
        loras.extend(getattr(network, attr, None) or [])
    return loras


def _compute_basis(weight: torch.Tensor, rank: int, use_lowrank_svd: bool):
    """Top-k left/right singular vectors of a (possibly conv) weight, as fp32 CPU tensors.

    Returns ``(U_k [out, k], V_k [in_flat, k])`` with orthonormal columns, or ``None`` if the
    weight is too small for the requested rank.
    """
    w = weight.detach().reshape(weight.shape[0], -1).float().cpu()  # (out, in_flat)
    k = min(rank, w.shape[0], w.shape[1])
    if k <= 0:
        return None
    if use_lowrank_svd:
        # randomized SVD with a little oversampling for a stable top-k basis
        q = min(w.shape[0], w.shape[1], k + 8)
        u, _, v = torch.svd_lowrank(w, q=q)  # w ~ u diag(s) v^T, v is (in_flat, q)
        return u[:, :k].contiguous(), v[:, :k].contiguous()
    u, _, vh = torch.linalg.svd(w, full_matrices=False)
    return u[:, :k].contiguous(), vh[:k, :].t().contiguous()


class OPLoRAManager:
    """Builds and applies the orthogonal projection for every projectable LoRA module of a network.

    The basis is built from the base weights before ``network.apply_to`` deletes ``org_module``.
    ``project`` is called after each optimizer step to push the updated factors back into the
    orthogonal complement.
    """

    def __init__(self, network, rank: int, use_lowrank_svd: bool):
        self.network = network
        self.bases = {}
        skipped_split = 0
        for lora in _all_loras(network):
            # split-qkv LoRA (FLUX) packs several outputs into one base weight; an exact
            # per-split projection does not factor back into low-rank factors, so it is skipped.
            if getattr(lora, "split_dims", None) is not None:
                skipped_split += 1
                continue
            org_module = getattr(lora, "org_module", None)
            if org_module is None or not hasattr(org_module, "weight"):
                continue
            basis = _compute_basis(org_module.weight, rank, use_lowrank_svd)
            if basis is not None:
                self.bases[id(lora)] = basis
        msg = f"OPLoRA: orthogonal projection enabled for {len(self.bases)} modules (rank {rank})"
        if skipped_split:
            msg += f", {skipped_split} split-qkv modules left unprojected"
        logger.info(msg)

    @torch.no_grad()
    def project(self) -> None:
        for lora in _all_loras(self.network):
            basis = self.bases.get(id(lora))
            if basis is None:
                continue
            up = lora.lora_up.weight
            down = lora.lora_down.weight
            u_k, v_k = basis
            if u_k.device != up.device:
                u_k = u_k.to(up.device)
                v_k = v_k.to(up.device)
                self.bases[id(lora)] = (u_k, v_k)
            up2d = up.reshape(up.shape[0], -1).float()  # (out, r)
            down2d = down.reshape(down.shape[0], -1).float()  # (r, in_flat)
            up2d = up2d - u_k @ (u_k.t() @ up2d)
            down2d = down2d - (down2d @ v_k) @ v_k.t()
            up.copy_(up2d.reshape(up.shape).to(up.dtype))
            down.copy_(down2d.reshape(down.shape).to(down.dtype))


def create_oplora_manager(args: argparse.Namespace, network):
    """Build the OPLoRA manager from the raw network (before ``apply_to``), or None if disabled."""
    if not is_oplora_enabled(args):
        return None
    rank = int(getattr(args, "oplora_rank", 0))
    if rank <= 0:
        raise ValueError("--oplora requires --oplora_rank > 0 (the top-k singular subspace to preserve).")
    use_lowrank = not getattr(args, "oplora_full_svd", False)
    return OPLoRAManager(network, rank, use_lowrank)
