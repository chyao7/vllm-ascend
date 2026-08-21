# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType
from types import SimpleNamespace

from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID

from vllm_ascend.core.recompute_scheduler import RecomputeScheduler


def _make_scheduler(*, enforce_eager: bool, inject: bool | None = None):
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.requests = {}
    scheduler.is_kv_producer = False
    scheduler.is_hybrid_model = False
    scheduler.is_mtp_kv_consumer = True
    scheduler.num_spec_tokens = 1
    scheduler.max_model_len = 1024
    scheduler.log_stats = False
    scheduler.connector = None
    scheduler.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=enforce_eager)
    )
    if inject is None:
        scheduler._inject_mtp_kv_placeholders = not enforce_eager
    else:
        scheduler._inject_mtp_kv_placeholders = inject
    enqueued_requests = []

    def enqueue_waiting_request(self, request):
        enqueued_requests.append(request)

    scheduler._enqueue_waiting_request = MethodType(enqueue_waiting_request, scheduler)
    return scheduler, enqueued_requests


def test_pd_consumer_first_step_injects_placeholder_spec_tokens():
    scheduler, enqueued_requests = _make_scheduler(enforce_eager=False)

    request = Request(
        request_id="pd-consumer-first-step",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )

    scheduler.add_request(request)

    assert enqueued_requests == [request]
    assert scheduler.requests[request.request_id] is request
    assert request.spec_token_ids == [PLACEHOLDER_TOKEN_ID]
    assert request.num_tokens_with_spec == request.num_tokens + 1


def test_pd_consumer_skips_placeholder_spec_tokens_when_enforce_eager():
    scheduler, enqueued_requests = _make_scheduler(enforce_eager=True)

    request = Request(
        request_id="pd-consumer-eager",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )

    scheduler.add_request(request)

    assert enqueued_requests == [request]
    assert request.spec_token_ids == []
