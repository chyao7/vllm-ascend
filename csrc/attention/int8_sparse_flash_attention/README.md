# Int8SparseFlashAttention

Ascend **910B** sparse flash attention with **packed int8 KV nope**, **bf16/fp16 rope**, and **per-tile fp32 scales**.

## Formula

For each gathered KV token:

```text
# Packed layout (logical D=516 = 512 int8 + 4 fp32 scales; 528 bytes/row):
scale[g] = fp32 at byte offset 512 + g*4
kv_dequant[tile g] = int8[tile g] * scale[g]

attention = softmax(Q @ K_dequant^T * scale_value) @ V_dequant
```

Rope (D=64) is **not quantized** and is copied as bf16/fp16 from `key_rope`.

## Supported platform

| Product | Support |
|---------|---------|
| Atlas A2 (ascend910b) | Yes |
| Others | No |

## Inputs

| Name | Dtype | Shape (PA_BSND example) | Notes |
|------|-------|-------------------------|-------|
| query | fp16/bf16 | `(T, N1, 512)` TND | nope |
| query_rope | fp16/bf16 | `(T, N1, 64)` | required |
| key | **int8** | `(block_num, block_size, 1, 528)` | packed nope cache (logical D=516) |
| value | **int8** | same as key | shared KV |
| key_rope | fp16/bf16 | `(block_num, block_size, 1, 64)` | not quantized |
| sparse_indices | int32 | `(T, 1, topk)` | from lightning_indexer |

Packed layout (910B sparse C8): logical **D=516** = 512 int8 + 4 fp32 scales; **528 bytes** per row in torch.int8 cache.

## Attributes

| Name | Type | Default | Notes |
|------|------|---------|-------|
| scale_value | float | 1.0 | attention softmax scale |
| key_scale | float | 1.0 | deprecated, ignored |
| key_offset | float | 0.0 | deprecated, ignored |

## Implementation notes

- Based on `sparse_flash_attention` arch22 pipeline.
- **Vec0** (`MergeKv`): gather int8 nope + bf16 rope, per-tile dequant to Q dtype, write merged workspace.
- GM stride 528 with per-128-dim fp32 dequant from embedded scale metadata.
- **Cube/Vec1/Vec2**: unchanged Flash Attention on dequantized KV.

## Python

```python
out, _, _ = torch.ops._C_ascend.npu_int8_sparse_flash_attention(
    query=ql_nope,
    key=kv_int8,          # D=528 packed
    value=kv_int8,
    sparse_indices=topk_indices,
    scale_value=attn_scale,
    block_table=block_table,
    actual_seq_lengths_query=cum_q_lens,
    actual_seq_lengths_kv=seq_lens,
    query_rope=q_pe,
    key_rope=key_rope_bf16,
    layout_query="TND",
    layout_kv="PA_BSND",
    sparse_mode=3,
    attention_mode=2,
)
```

## Build

Registered in `csrc/build_aclnn.sh` for `ascend910b`.

```bash
bash csrc/build_aclnn.sh $(pwd) ascend910b
```
