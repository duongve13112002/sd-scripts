from __future__ import division
from __future__ import unicode_literals

from typing import Iterable, Optional
import weakref
import copy
import contextlib
from .optimizers.optimizer_utils import copy_stochastic

import torch


# Partially based on:
# https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py
class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.

    Args:
        parameters: Iterable of `torch.nn.Parameter` (typically from
            `model.parameters()`).
            Note that EMA is computed on *all* provided parameters,
            regardless of whether or not they have `requires_grad = True`;
            this allows a single EMA object to be consistantly used even
            if which parameters are trainable changes step to step.

            If you want to some parameters in the EMA, do not pass them
            to the object in the first place. For example:

                ExponentialMovingAverage(
                    parameters=[p for p in model.parameters() if p.requires_grad],
                    decay=0.9
                )

            will ignore parameters that do not require grad.

        decay: The exponential decay.

        use_num_updates: Whether to use number of updates when computing
            averages.
    """

    def __init__(
            self,
            parameters: Iterable[torch.nn.Parameter] = None,
            decay: float = 0.995,
            use_num_updates: bool = False,
            # feeds back the decay to the parameter
            use_feedback: bool = False,
            param_multiplier: float = 1.0,
            device: Optional[torch.device] = None,
            accelerator=None,  # Required for multi-GPU
    ):
        if parameters is None:
            raise ValueError("parameters must be provided")
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        if param_multiplier <= 0.0:
            raise ValueError('param_multiplier must be positive')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.use_feedback = use_feedback
        self.param_multiplier = param_multiplier
        self.device = device
        self.accelerator = accelerator

        # ==================== MULTI-GPU HANDLING ====================
        # In multi-GPU training:
        # - Only MAIN process stores shadow parameters (save memory)
        # - Other processes skip shadow param creation
        # - Shadow params are broadcasted when needed for sampling

        self.is_main_process = (
            accelerator is None or accelerator.is_main_process
        )
        self.num_processes = (
            1 if accelerator is None else accelerator.num_processes
        )

        parameters = list(parameters)

        self.shadow_param_shapes = [p.shape for p in parameters]
        self.shadow_param_dtypes = [p.dtype for p in parameters]

        if self.is_main_process:
            # Main process: Create shadow parameters
            self.shadow_params = [
                p.clone().detach().to(device) if device else p.clone().detach()
                for p in parameters
            ]

            if accelerator is not None:
                accelerator.print(
                    f"EMA: Main process storing {len(self.shadow_params)} shadow parameters"
                )
        else:
            # Other processes: Just store parameter shapes for later
            # This saves MASSIVE memory in multi-GPU training
            self.shadow_params = None
            if accelerator is not None:
                accelerator.print(
                    f"EMA: Worker process skipping shadow parameter creation (memory saving)"
                )
        self.collected_params = None
        self._is_train_mode = True
        # By maintaining only a weakref to each parameter,
        # we maintain the old GC behaviour of ExponentialMovingAverage:
        # if the model goes out of scope but the ExponentialMovingAverage
        # is kept, no references to the model or its parameters will be
        # maintained, and the model will be cleaned up.
        self._params_refs = [weakref.ref(p) for p in parameters]

    def _get_parameters(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]]
    ) -> Iterable[torch.nn.Parameter]:

        if parameters is None:
            parameters = [p() for p in self._params_refs]
            if any(p is None for p in parameters):
                raise ValueError(
                    "(One of) the parameters with which this "
                    "ExponentialMovingAverage "
                    "was initialized no longer exists (was garbage collected);"
                    " please either provide `parameters` explicitly or keep "
                    "the model to which they belong from being garbage "
                    "collected."
                )
            return parameters
        else:
            parameters = list(parameters)

            # For multi-GPU: we may receive parameters from different processes
            # Ensure we have the right number
            if self.is_main_process and self.shadow_params is not None:
                expected_len = len(self.shadow_params)
            elif hasattr(self, 'shadow_param_shapes'):
                expected_len = len(self.shadow_param_shapes)
            else:
                expected_len = len(self._params_refs)


            if len(parameters) != expected_len:
                raise ValueError(
                    "Number of parameters passed as argument is different "
                    "from number of shadow parameters maintained by this "
                    "ExponentialMovingAverage"
                )
            return parameters

    def _sync_shadow_params_to_workers(self):
        """Broadcast shadow parameters from main process to all workers"""
        if self.accelerator is None or self.num_processes == 1:
            return

        # ==================== FIX 1: Wait before starting ====================
        self.accelerator.wait_for_everyone()
        # ====================================================================

        if not self.is_main_process and self.shadow_params is None:
            # Workers: Create empty shadow params
            self.shadow_params = [
                torch.zeros(shape, dtype=dtype, device=self.device)
                for shape, dtype in zip(self.shadow_param_shapes, self.shadow_param_dtypes)
            ]

        # ==================== FIX 2: Ensure same iteration count ====================
        # All processes must know the number of params to broadcast
        if self.is_main_process:
            num_params = len(self.shadow_params)
        else:
            num_params = len(self.shadow_param_shapes)

        # Broadcast count (sanity check)
        if self.accelerator.distributed_type != "NO":
            import torch.distributed as dist
            num_params_tensor = torch.tensor([num_params], device=self.accelerator.device)
            dist.broadcast(num_params_tensor, src=0)

            if not self.is_main_process:
                assert num_params_tensor.item() == num_params, \
                    f"Param count mismatch: expected {num_params_tensor.item()}, got {num_params}"
        # ============================================================================

        # Now safe to broadcast
        for i in range(num_params):
            shadow_param = self.shadow_params[i]

            # Move to GPU for efficient broadcast
            shadow_on_gpu = shadow_param.to(self.accelerator.device)

            if self.accelerator.distributed_type != "NO":
                import torch.distributed as dist
                dist.broadcast(shadow_on_gpu, src=0)

            # Move back to storage device
            self.shadow_params[i] = shadow_on_gpu.to(self.device)

        # ==================== FIX 3: Wait after finishing ====================
        self.accelerator.wait_for_everyone()
        # ====================================================================

        if not self.is_main_process:
            self.accelerator.print("EMA: Worker received shadow parameters")

    def _cleanup_worker_shadow_params(self):
        """
        Clean up shadow params on worker processes after sampling.
        Saves memory during training.
        """
        if not self.is_main_process and self.shadow_params is not None:
            # Workers can delete shadow params after sampling
            del self.shadow_params
            self.shadow_params = None
            torch.cuda.empty_cache()


    def update(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Update EMA parameters (ONLY on main process in multi-GPU).

        In multi-GPU training:
        - Only main process performs update (others skip)
        - Ensures consistent EMA state across processes
        - Parameters are already synced by DDP/FSDP before this call

        Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the same set of
                parameters used to initialize this object. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """

        # ==================== MULTI-GPU: Only main updates ====================
        if not self.is_main_process:
            # Worker processes: Skip update to save computation
            # Main process will handle it and broadcast when needed
            return

        if self.shadow_params is None:
            raise RuntimeError("Shadow parameters not initialized on main process")
        # ======================================================================

        parameters = self._get_parameters(parameters)

        # Calculate decay
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(
                decay,
                (1 + self.num_updates) / (10 + self.num_updates)
            )
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            for s_param, param in zip(self.shadow_params, parameters):
                # ==================== MULTI-GPU: Handle DDP parameters ====================
                # In DDP, param.data contains the local copy
                # In FSDP, param might be sharded - need to handle carefully

                # Get the actual parameter data (unwrap if needed)
                if hasattr(param, '_data') and param._data is not None:
                    param_data = param._data  # FSDP sharded param
                else:
                    param_data = param.data

                # Move to same device as shadow param for computation
                param_on_device = param_data.to(s_param.device)
                # ==========================================================================

                # Convert to fp32 for precision
                s_param_float = s_param.float()
                if s_param.dtype != torch.float32:
                    s_param_float = s_param_float.to(torch.float32)
                param_float = param_on_device
                if param_float.dtype != torch.float32:
                    param_float = param_float.to(torch.float32)
                if s_param_float.device != param_float.device:
                    self.accelerator.print(f"s_param_float and param_float are on different devices: {s_param_float.device} - {param_float.device} -> load occurred on {param_float.device}")
                    s_param_float = s_param_float.to(param_float.device)
                    param_float = param_float.to(param_float.device)
                tmp = (s_param_float - param_float)
                # tmp will be a new tensor so we can do in-place
                tmp.mul_(one_minus_decay)
                s_param_float.sub_(tmp)

                update_param = False
                if self.use_feedback:
                    # make feedback 10x decay
                    param_float.add_(tmp * 10)
                    update_param = True

                if self.param_multiplier != 1.0:
                    param_float.mul_(self.param_multiplier)
                    update_param = True

                # Write s_param_float back to s_param (stochastic rounding for non-fp32)
                # When s_param is fp32, s_param_float IS s_param (same tensor), no copy needed
                if s_param.dtype != torch.float32:
                    copy_stochastic(s_param, s_param_float)

                # If we modified the param (feedback/multiplier), copy back to original device
                if update_param:
                    param_back = param_float.to(param.device)

                    if param.dtype != torch.float32:
                        copy_stochastic(param.data, param_back)
                    else:
                        param.data.copy_(param_back)


    def copy_to(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Copy current averaged parameters into given collection of parameters.
        In multi-GPU:
        - Shadow params are broadcasted from main to workers first
        - Then copied to each GPU's model replica

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """

        # ==================== MULTI-GPU: Sync shadow params first ====================
        if self.accelerator is not None and self.num_processes > 1:
            self._sync_shadow_params_to_workers()
        # =============================================================================

        if self.shadow_params is None:
            raise RuntimeError("Shadow parameters not available")


        parameters = self._get_parameters(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

        if self.accelerator is not None and not self.is_main_process:
            self.accelerator.print("EMA: Worker copied shadow params to model")

    def store(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Save the current parameters for restoring later.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored. If `None`, the parameters of with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        self.collected_params = [
            param.clone()
            for param in parameters
        ]

    def restore(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        if self.collected_params is None:
            raise RuntimeError(
                "This ExponentialMovingAverage has no `store()`ed weights "
                "to `restore()`"
            )
        parameters = self._get_parameters(parameters)
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

        # ==================== MULTI-GPU: Cleanup ====================
        # Clear collected params to free memory
        self.collected_params = None

        # Workers can also cleanup shadow params now
        if not self.is_main_process:
            self._cleanup_worker_shadow_params()
        # ============================================================

    @contextlib.contextmanager
    def average_parameters(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ):
        r"""
        Context manager for validation/inference with averaged parameters.

        In multi-GPU:
        - Broadcasts shadow params to all workers
        - Each worker uses EMA for its model replica
        - Cleanup after sampling

        Equivalent to:

            ema.store()
            ema.copy_to()
            try:
                ...
            finally:
                ema.restore()

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """

        # ==================== MULTI-GPU: Synchronize ====================
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        # ================================================================

        parameters = self._get_parameters(parameters)
        self.store(parameters)
        self.copy_to(parameters)
        try:
            yield
        finally:
            self.restore(parameters)

            # ==================== MULTI-GPU: Cleanup ====================
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
            # ============================================================

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly

        if not self.is_main_process:
            return  # Workers don't store shadow params during training

        if self.shadow_params is None:
            return

        self.device = device

        self.shadow_params = [
            p.to(device=device, dtype=dtype)
            if p.is_floating_point()
            else p.to(device=device)
            for p in self.shadow_params
        ]
        if self.collected_params is not None:
            self.collected_params = [
                p.to(device=device, dtype=dtype)
                if p.is_floating_point()
                else p.to(device=device)
                for p in self.collected_params
            ]
        return

    def state_dict(self) -> dict:
        r"""Returns the state of the ExponentialMovingAverage as a dict."""
        # Following PyTorch conventions, references to tensors are returned:
        # "returns a reference to the state and not its copy!" -
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        if not self.is_main_process:
            return {}  # Workers return empty dict

        if self.shadow_params is None:
            return {}

        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
            "collected_params": self.collected_params
        }

    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads the ExponentialMovingAverage state.
        In multi-GPU: Only main process loads shadow params
        Args:
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        if not state_dict:  # Empty dict from worker
            return

        if not self.is_main_process:
            return  # Workers don't load shadow params

        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)
        self.decay = state_dict["decay"]
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.num_updates = state_dict["num_updates"]
        assert self.num_updates is None or isinstance(self.num_updates, int), \
            "Invalid num_updates"

        self.shadow_params = state_dict["shadow_params"]
        assert isinstance(self.shadow_params, list), \
            "shadow_params must be a list"
        assert all(
            isinstance(p, torch.Tensor) for p in self.shadow_params
        ), "shadow_params must all be Tensors"

        self.collected_params = state_dict["collected_params"]
        if self.collected_params is not None:
            assert isinstance(self.collected_params, list), \
                "collected_params must be a list"
            assert all(
                isinstance(p, torch.Tensor) for p in self.collected_params
            ), "collected_params must all be Tensors"
            assert len(self.collected_params) == len(self.shadow_params), \
                "collected_params and shadow_params had different lengths"

        if len(self.shadow_params) == len(self._params_refs):
            # Consistant with torch.optim.Optimizer, cast things to consistant
            # device and dtype with the parameters
            params = [p() for p in self._params_refs]
            # If parameters have been garbage collected, just load the state
            # we were given without change.
            if not any(p is None for p in params):
                # ^ parameter references are still good
                for i, p in enumerate(params):
                    self.shadow_params[i] = self.shadow_params[i].to(
                        device=p.device, dtype=p.dtype
                    )
                    if self.collected_params is not None:
                        self.collected_params[i] = self.collected_params[i].to(
                            device=p.device, dtype=p.dtype
                        )
        else:
            raise ValueError(
                "Tried to `load_state_dict()` with the wrong number of "
                "parameters in the saved state."
            )
        if self.device is not None:
            self.to(self.device)

    def eval(self):
        if self._is_train_mode:
            with torch.no_grad():
                self.store()
                self.copy_to()
                self._is_train_mode = False

    def train(self):
        if not self._is_train_mode:
            with torch.no_grad():
                self.restore()
                self._is_train_mode = True
