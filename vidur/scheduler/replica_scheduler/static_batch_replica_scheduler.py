from math import ceil
from typing import List

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)


class StaticBatchReplicaScheduler(BaseReplicaScheduler):
    """Static batching scheduler with paged attention memory management.

    Once a batch is dispatched, it runs to completion — no request can leave
    or join mid-batch. Memory is managed with block-based KV cache (paged
    attention), allocating blocks incrementally as decode progresses.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._num_running_batches = 0
        self._running_batches: List[Batch] = []

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1

        # Free memory for completed requests
        completed_ids = [r.id for r in batch.requests if r.completed]
        if completed_ids:
            self.free(*completed_ids)

        if batch.all_requests_completed:
            return

        # Re-queue the batch so _get_next_batch continues it with the same requests
        self._running_batches.append(batch)

    def _get_next_batch(self) -> Batch:
        # Priority 1: continue a running static batch
        if self._running_batches:
            old_batch = self._running_batches.pop(0)
            requests = []
            num_tokens = []

            for request in old_batch.requests:
                if request.completed:
                    continue

                # Allocate 1 more block for decode if current blocks are full
                tokens_allocated = (
                    self._allocation_map.get(request.id, 0) * self._config.block_size
                )
                if request.num_processed_tokens >= tokens_allocated:
                    if self.can_allocate(1):
                        self.allocate(request.id, 1)

                next_num_tokens = self._get_request_next_num_tokens(request)
                requests.append(request)
                num_tokens.append(next_num_tokens)

            if requests:
                return Batch(self._replica_id, requests, num_tokens)
            return None

        # Priority 2: form a new static batch from the request queue
        requests = []
        num_tokens = []

        while self._request_queue:
            if len(requests) >= self._max_batch_size:
                break

            request = self._request_queue[0]
            num_blocks_needed = ceil(
                request.num_prefill_tokens / self._config.block_size
            )

            if not self.can_allocate(num_blocks_needed):
                break

            request = self._request_queue.pop(0)
            self.allocate(request.id, num_blocks_needed)
            next_num_tokens = self._get_request_next_num_tokens(request)
            requests.append(request)
            num_tokens.append(next_num_tokens)

        if requests:
            return Batch(self._replica_id, requests, num_tokens)
        return None
