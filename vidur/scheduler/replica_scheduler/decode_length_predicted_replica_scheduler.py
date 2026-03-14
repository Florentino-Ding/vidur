from typing import Dict, List, Optional

from vidur.entities import Batch, Request
from vidur.scheduler.replica_scheduler.decode_length_predictor import DecodeLengthPredictor
from vidur.scheduler.replica_scheduler.vllm_replica_scheduler import VLLMReplicaScheduler


class DecodeLengthPredictedReplicaScheduler(VLLMReplicaScheduler):
    """
    Replica scheduler that batches requests by predicted decode length.

    Two additions over VLLMReplicaScheduler:
    1. Decode-length prediction with configurable per-request latency overhead
       and optional Gaussian noise (noise_std=0 gives the oracle/exact value).
    2. Batching by similar predicted decode lengths: at each scheduling round,
       a sliding-window algorithm selects the largest group of prediction-ready
       requests whose predicted lengths span at most `similarity_tolerance` tokens,
       then applies the standard vLLM memory constraints within that group.

    All preemption and memory management logic is inherited unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._predictor = DecodeLengthPredictor(
            noise_std=self._config.prediction_noise_std
        )
        self._predicted_lengths: Dict[int, float] = {}
        self._prediction_ready_at: Dict[int, float] = {}
        # Convert ms → seconds (simulator time unit is seconds)
        self._prediction_latency_s: float = self._config.prediction_latency_ms / 1000.0
        self._tolerance: int = self._config.similarity_tolerance

    # ------------------------------------------------------------------
    # add_request: run prediction immediately, record when it's ready
    # ------------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        super().add_request(request)
        self._predicted_lengths[request.id] = self._predictor.predict(request)
        self._prediction_ready_at[request.id] = (
            request.arrived_at + self._prediction_latency_s
        )

    # ------------------------------------------------------------------
    # on_batch_end: clean up per-request prediction state when done
    # ------------------------------------------------------------------

    def on_batch_end(self, batch: Batch) -> None:
        super().on_batch_end(batch)
        for request in batch.requests:
            if request.completed:
                self._predicted_lengths.pop(request.id, None)
                self._prediction_ready_at.pop(request.id, None)

    # ------------------------------------------------------------------
    # _get_next_batch: prediction-aware, similarity-grouped batching
    # ------------------------------------------------------------------

    def _get_next_batch(self) -> Optional[Batch]:
        # Preempted requests always take priority (matches vLLM semantics)
        if not self._request_queue and self._preempted_requests:
            return self._schedule_preempted()

        # Split queue: requests whose prediction has completed vs still pending
        ready, _pending = [], []
        for req in self._request_queue:
            if self._prediction_ready_at.get(req.id, 0.0) <= self._current_time:
                ready.append(req)
            else:
                _pending.append(req)

        if not ready:
            return None

        # Sort prediction-ready requests by predicted decode length
        ready.sort(key=lambda r: self._predicted_lengths[r.id])

        # Find the largest group within similarity_tolerance
        group = self._find_best_group(ready)

        # Apply vLLM-style memory / batch-size constraints within the group
        requests: List[Request] = []
        num_tokens: List[int] = []
        selected_ids = set()

        for request in group:
            next_num_tokens = self._get_request_next_num_tokens(request)

            if not self._can_allocate_request(request):
                break

            new_num_tokens = num_tokens + [next_num_tokens]
            if len(new_num_tokens) * max(new_num_tokens) > self._config.max_tokens_in_batch:
                break

            if len(self._allocation_map) == self._config.batch_size_cap:
                break

            if len(requests) == self._max_micro_batch_size:
                break

            self._allocate_request(request)
            requests.append(request)
            num_tokens.append(next_num_tokens)
            selected_ids.add(request.id)

        if not requests:
            # Memory was full even for the best group; try preempted queue
            return self._schedule_preempted() if self._preempted_requests else None

        # Remove selected requests from the main queue; preserve relative order
        self._request_queue = [r for r in self._request_queue if r.id not in selected_ids]

        return Batch(self._replica_id, requests, num_tokens)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_best_group(self, sorted_ready: List[Request]) -> List[Request]:
        """
        Two-pointer sliding window over requests sorted by predicted decode length.
        Returns the largest contiguous window where
            max_predicted - min_predicted <= similarity_tolerance.
        """
        if not sorted_ready:
            return []

        best_lo, best_hi = 0, 0
        lo = 0
        for hi in range(len(sorted_ready)):
            max_p = self._predicted_lengths[sorted_ready[hi].id]
            min_p = self._predicted_lengths[sorted_ready[lo].id]
            while max_p - min_p > self._tolerance:
                lo += 1
                min_p = self._predicted_lengths[sorted_ready[lo].id]
            if hi - lo > best_hi - best_lo:
                best_lo, best_hi = lo, hi

        return sorted_ready[best_lo : best_hi + 1]

    def _schedule_preempted(self) -> Optional[Batch]:
        """
        Handle preempted requests with the same logic as VLLMReplicaScheduler.
        Duplicated here so _get_next_batch can call it as a fallback without
        altering the parent's _get_next_batch flow.
        """
        requests: List[Request] = []
        num_tokens: List[int] = []
        self._preempted_requests.sort(key=lambda r: r.arrived_at)

        while self._preempted_requests:
            if len(requests) == self._max_micro_batch_size:
                break

            request = self._preempted_requests.pop(0)

            while not self._can_allocate_request(request):
                if self._preempted_requests:
                    victim = self._preempted_requests.pop(-1)
                    victim.restart()
                    self.free(victim.id)
                    self._request_queue = [victim] + self._request_queue
                else:
                    request.restart()
                    self.free(request.id)
                    self._request_queue = [request] + self._request_queue
                    break
            else:
                self._allocate_request(request)
                next_num_tokens = self._get_request_next_num_tokens(request)
                requests.append(request)
                num_tokens.append(next_num_tokens)

        if not requests:
            return None
        return Batch(self._replica_id, requests, num_tokens)
