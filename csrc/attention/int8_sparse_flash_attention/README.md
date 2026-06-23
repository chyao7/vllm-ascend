# Int8SparseFlashAttention

Ascend **910B** sparse flash attention with **int8 KV nope**, **bf16/fp16 rope**, and optional **packed per-tile scales**.

## Formula

For each gathered KV token:

```text
# Global mode (key D=512):
kv_dequant = kv_int8 * key_scale + key_offset

# Packed mode (key D=528 = 512 int8 + 16B fp32 scales):
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
| key | **int8** | `(block_num, block_size, 1, 512)` or `(…, 528)` packed | nope cache |
| value | **int8** | same as key | shared KV |
| key_rope | fp16/bf16 | `(block_num, block_size, 1, 64)` | not quantized |
| sparse_indices | int32 | `(T, 1, topk)` | from lightning_indexer |

Packed layout (910B sparse C8): last dim **528** = 512 int8 + 4×fp32 per-token scales. Kernel auto-detects when `key.shape[-1] == 528`.

## Attributes

| Name | Type | Default | Notes |
|------|------|---------|-------|
| scale_value | float | 1.0 | attention softmax scale |
| key_scale | float | 1.0 | global KV dequant scale (D=512 only) |
| key_offset | float | 0.0 | global KV dequant offset (D=512 only) |

## Implementation notes

- Based on `sparse_flash_attention` arch22 pipeline.
- **Vec0** (`MergeKv`): gather int8 nope + bf16 rope, dequant to Q dtype, write merged workspace.
- Packed mode uses GM stride 528 and per-128-dim fp32 dequant from embedded scale metadata.
- **Cube/Vec1/Vec2**: unchanged Flash Attention on dequantized KV.

## Python

```python
out, _, _ = torch.ops._C_ascend.npu_int8_sparse_flash_attention(
    query=ql_nope,
    key=kv_int8,          # D=512 global quant, or D=528 packed
    value=kv_int8,
    sparse_indices=topk_indices,
    scale_value=attn_scale,
    key_scale=0.02,       # ignored when key D=528
    key_offset=0.0,
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
