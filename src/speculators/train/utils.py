import logging
import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import init_device_mesh, device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
is_distributed = "LOCAL_RANK" in os.environ

logger = logging.getLogger("speculators")


def maybe_setup_distributed(master_addr: str = None, master_port: int = None, nnodes: int = 1, nproc_per_node: int = 1) -> tuple[int, int, int, bool, Optional[device_mesh]]:
    """Sets up distributed training if the process was launched with `torchrun`.
    If not, returns single process training.

    Based on of https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html#initialize-ddp-with-torch-distributed-run-torchrun

    Args:
        master_addr: Address of the master node (for multi-node training)
        master_port: Port of the master node (for multi-node training)
        nnodes: Total number of nodes
        nproc_per_node: Number of processes per node

    Returns:
        tuple[int, int, int, bool, device_mesh | None]: Local rank, world size, rank, is_distributed, and device mesh.
    """
    if not is_distributed:
        # No distributed training
        return 0, 1, 0, False, None

    torch.accelerator.set_device_index(local_rank)
    acc = torch.accelerator.current_accelerator()
    if acc is None:
        raise ValueError("No accelerator found")
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend, device_id=local_rank)

    rank = dist.get_rank()

    world_size = dist.get_world_size()
    # # Set environment variables for multi-node training if provided
    # if master_addr is not None:
    #     os.environ["MASTER_ADDR"] = master_addr
    # if master_port is not None:
    #     os.environ["MASTER_PORT"] = str(master_port)
    device_type = acc.type if acc is not None else "cuda"
    if nproc_per_node == 1 and world_size > 1:
        nproc_per_node = world_size
    mesh = init_device_mesh(device_type, (nnodes, nproc_per_node), mesh_dim_names=('dp', 'tp'))
    print("get=====mesh:", mesh)
    logger.info(
        f"Started distributed with local_rank={local_rank}, world_size={world_size}, rank={rank}",
        extra={"override_rank0_filter": True},
    )
    return local_rank, world_size, rank, True, mesh


def maybe_destroy_distributed():
    """Destroys the distributed process group if using distributed training."""
    if not is_distributed:
        # No distributed training
        return

    dist.destroy_process_group()
    logger.info(
        f"Destroyed distributed with local_rank={local_rank}, world_size={world_size}",
        extra={"override_rank0_filter": True},
    )


def apply_fully_sharded(model: torch.nn.Module, mesh: Optional[device_mesh] = None):
    """Applies torch FSDP fully_shard to the model, wrapping layers in FSDPModule.

    Assumes the model has a `layers` attribute containing the decoder layers.
    Model should be validated with SpeculatorModel.verify_training_compatible()
    before calling this function.
    
    Args:
        model: The model to apply FSDP to
        mesh: Device mesh for FSDP. If provided, enables hybrid sharding (fully shard within node, replicate across nodes)
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    # For hybrid sharding: shard within node, replicate across nodes
    # We use the device mesh to specify the sharding strategy
    sharding_strategy = "fully_shard" if mesh is None else "hybrid_shard"
    
    for layer in model.layers:  # type: ignore[union-attr]
        fully_shard(layer, mp_policy=mp_policy, mesh=mesh)

    fully_shard(model, mp_policy=mp_policy, mesh=mesh)

    return model

