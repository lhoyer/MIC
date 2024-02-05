""" Memory Bank Wrapper """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import warnings
from typing import Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
import torch.distributed as dist

@torch.no_grad()
def concat_all_gather(x: torch.Tensor) -> torch.Tensor:
    """Returns concatenated instances of x gathered from all gpus.

    This code was taken and adapted from here:
    https://github.com/facebookresearch/moco.

    """
    output = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(output, x, async_op=False)
    output = torch.cat(output, dim=0)
    return output

class MemoryBankModule(Module):
    """Memory bank implementation

    This is a parent class to all loss functions implemented by the lightly
    Python package. This way, any loss can be used with a memory bank if
    desired.

    Attributes:
        size:
            Size of the memory bank as (num_features, dim) tuple. If num_features is 0
            then the memory bank is disabled. Deprecated: If only a single integer is
            passed, it is interpreted as the number of features and the feature
            dimension is inferred from the first batch stored in the memory bank.
            Leaving out the feature dimension might lead to errors in distributed
            training.
        gather_distributed:
            If True then negatives from all gpus are gathered before the memory bank
            is updated. This results in more frequent updates of the memory bank and
            keeps the memory bank contents independent of the number of gpus. But it has
            the drawback that synchronization between processes is required and
            diversity of the memory bank content is reduced.
        feature_dim_first:
            If True, the memory bank returns features with shape (dim, num_features).
            If False, the memory bank returns features with shape (num_features, dim).

    Examples:
        >>> class MyLossFunction(MemoryBankModule):
        >>>
        >>>     def __init__(self, memory_bank_size: Tuple[int, int] = (2 ** 16, 128)):
        >>>         super().__init__(memory_bank_size)
        >>>
        >>>     def forward(self, output: Tensor, labels: Union[Tensor, None] = None):
        >>>         output, negatives = super().forward(output)
        >>>
        >>>         if negatives is not None:
        >>>             # evaluate loss with negative samples
        >>>         else:
        >>>             # evaluate loss without negative samples

    """

    def __init__(
        self,
        num_classes: int = 15,
        size: Union[int, Sequence[int]] = 65536,
        gather_distributed: bool = False,
        feature_dim_first: bool = False,
        pixel_update_freq: int = 10,
        ignore_label: int = -1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_label = ignore_label

        size_tuple = (
            (
                num_classes,
                size,
            )
            if isinstance(size, int)
            else tuple(size)
        )

        if any(x < 0 for x in size_tuple):
            raise ValueError(
                f"Illegal memory bank size {size}, all entries must be non-negative."
            )

        self.pixel_update_freq = pixel_update_freq
        self.size = size_tuple
        self.gather_distributed = gather_distributed
        self.feature_dim_first = feature_dim_first
        self.pixel_queue: Tensor

        self.register_buffer(
            "pixel_queue",
            tensor=torch.empty(size=size_tuple, dtype=torch.float),
            persistent=False,
        )
        self.pixel_queue_ptr: Tensor
        self.register_buffer(
            "pixel_queue_ptr",
            tensor=torch.empty(self.num_classes, dtype=torch.long),
            persistent=False,
        )

        self.segment_queue: Tensor
        self.register_buffer(
            "segment_queue",
            tensor=torch.empty(size=size_tuple, dtype=torch.float),
            persistent=False,
        )
        self.segment_queue_ptr: Tensor
        self.register_buffer(
            "segment_queue_ptr",
            tensor=torch.empty(self.num_classes, dtype=torch.long),
            persistent=False,
        )

        if isinstance(size, int) and size > 0:
            warnings.warn(
                (
                    f"Memory bank size 'size={size}' does not specify feature "
                    "dimension. It is recommended to set the feature dimension with "
                    "'size=(n, dim)' when creating the memory bank. Distributed "
                    "training might fail if the feature dimension is not set."
                ),
                UserWarning,
            )
        elif len(size_tuple) > 1:
            self._init_memory_bank(size=size_tuple)

    @torch.no_grad()
    def _init_memory_bank(self, size: Tuple[int, ...]) -> None:
        """Initialize the memory bank.

        Args:
            size:
                Size of the memory bank as (num_features, dim) tuple.

        """
        self.pixel_queue = torch.randn(size).type_as(self.pixel_queue).cuda()
        self.pixel_queue = torch.nn.functional.normalize(
            self.pixel_queue, dim=-1
        ).cuda()
        self.pixel_queue_ptr = (
            torch.zeros(self.num_classes).type_as(self.pixel_queue_ptr).cuda()
        )

        self.segment_queue = torch.randn(size).type_as(self.segment_queue).cuda()
        self.segment_queue = torch.nn.functional.normalize(
            self.segment_queue, dim=-1
        ).cuda()
        self.segment_queue_ptr = (
            torch.zeros(self.num_classes).type_as(self.segment_queue_ptr).cuda()
        )

    @torch.no_grad()
    def _dequeue_and_enqueue(
        self, keys: Tensor, labels: Tensor, weight: Union[Tensor, None] = None
    ) -> None:
        """Dequeue the oldest batch and add the latest one

        Args:
            keys:
                The latest batch of keys to add to the memory bank.
            labels:
                Corresponding class labels.
        """
        if self.gather_distributed:
            keys = concat_all_gather(keys)
            labels = concat_all_gather(labels)

        batch_size = keys.shape[0]
        feat_dim = keys.shape[1]

        for bs in range(batch_size):
            this_feat = keys[bs].contiguous().view(feat_dim, -1)
            this_label = labels[bs].contiguous().view(-1)
            if weight is not None:
                this_w = weight[bs].contiguous().view(-1)
                this_feat = this_feat[:, this_w > 0]
                this_label = this_label[this_w > 0]

            this_label_ids = torch.unique(this_label)
            this_label_ids = [x for x in this_label_ids if x != self.ignore_label]

            for lb in this_label_ids:
                idxs = (this_label == lb).nonzero()

                # segment enqueue and dequeue
                feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)

                ptr = int(self.segment_queue_ptr[lb])
                self.segment_queue[lb, ptr, :] = torch.nn.functional.normalize(
                    feat.view(-1), p=2, dim=0
                )
                self.segment_queue_ptr[lb] = (
                    self.segment_queue_ptr[lb] + 1
                ) % self.size[0]

                # pixel enqueue and dequeue
                num_pixel = idxs.shape[0]
                perm = torch.randperm(num_pixel)
                K = min(num_pixel, self.pixel_update_freq, self.size[0])
                feat = this_feat[:, perm[:K]]
                feat = torch.transpose(feat, 0, 1)
                ptr = int(self.pixel_queue_ptr[lb])

                if ptr + K >= self.size[0]:
                    self.pixel_queue[lb, -K:, :] = torch.nn.functional.normalize(
                        feat, p=2, dim=1
                    )
                    self.pixel_queue_ptr[lb] = 0
                else:
                    self.pixel_queue[
                        lb, ptr : ptr + K, :
                    ] = torch.nn.functional.normalize(feat, p=2, dim=1)
                    self.pixel_queue_ptr[lb] = (
                        self.pixel_queue_ptr[lb] + 1
                    ) % self.size[0]

    def forward(
        self,
        output: Tensor,
        labels: Union[Tensor, None] = None,
        weight: Union[Tensor, None] = None,
        update: bool = True,
    ) -> Tuple[Tensor, Union[Tensor, None]]:
        """Query memory bank for additional negative samples

        Args:
            output:
                The output of the model.
            labels:
                Classification labels: memory bank is updated separately for each class.
            weight:
                Weight values for each labels. If 0, the pixed value won't be used.
            update:
                If True, the memory bank will be updated with the current output. Default: True.

        Returns:
            The output if the memory bank is of size 0, otherwise the output
            and the entries from the memory bank. Entries from the memory bank have
            shape (dim, num_features) if feature_dim_first is True and
            (num_features, dim) otherwise.

        """

        # no memory bank, return the output
        if self.size[0] == 0:
            return output, None

        # Initialize the memory bank if it is not already done.
        if self.pixel_queue.ndim == 2:
            dim = output.shape[1]
            self._init_memory_bank(size=(*self.size, dim))

        if update:
            self._dequeue_and_enqueue(output, labels, weight)

        # query and update memory bank
        pixel_queue = self.pixel_queue.clone().detach()
        segment_queue = self.segment_queue.clone().detach()
        if self.feature_dim_first:
            # swap bank size and feature dimension for backwards compatibility
            pixel_queue = pixel_queue.transpose(0, -1)
            segment_queue = segment_queue.transpose(0, -1)

        # only update memory bank if we later do backward pass (gradient)

        # if update:
        #     self._dequeue_and_enqueue(output, labels, weight)

        return torch.cat((segment_queue, pixel_queue), dim=1)
