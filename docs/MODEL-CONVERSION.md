# Qwen 系小模型 → OpenVINO IR / MLX 转换方法论

> 2026-11-06/07 两天实战沉淀。适用于 Qwen3.5 家族（含 SFT 变体）在任何非官方运行时的部署。
> 实验对象：`hindsight-ai/hs-decide-1-flash`（Qwen3.5-0.8B System One 决策模型微调，Apache-2.0）。

## TL;DR — 三条路的死活

| 路线 | 结果 | 死因/要点 |
|---|---|---|
| pip 装新 optimum/transformers 转 IR | ❌ | optimum 2.3.0 与 transformers 5.19+ 互斥：`qwen3_5_text` 不在新 tf 注册表时 optimum 走 ONNX 导出，撞 `aten::linalg_solve_triangular` 不支持 |
| **旧工具链容器转 IR** | ✅ | `optimum 2.3.0 + transformers 5.2.0` 黄金组合，30s 出 IR；关键：找本机历史成功转换的容器镜像（`openviking-openvino-services:v2.0.0`），别用 pip 拼 |
| **MLX 壳化转换** | ✅ | `model_type: qwen3_5_text → qwen3_5` + 参数包进 `text_config`（借多模态壳过注册表），权重零改动，mlx_lm.convert 直出 |

## 死结复盘：架构注册墙

`qwen3_5`（多模态壳，tf 5.2 认识）vs `qwen3_5_text`（纯文本类，tf 5.19+ 才认识）——
同一个底座两个注册名，工具链版本夹在中间就是死锁。判断方法：看转换成功先例的
`openvino_config.json`（记录了 optimum/transformers 版本），照方抓药。

## INT8 量化（Intel iGPU 生产档）

```bash
optimum-cli export openvino -m <model_dir> \
  --task text-generation --library transformers --framework pt \
  --weight-format int8 <out_dir>   # output 是位置参数, 不是 --output!
```

- int8_asym per-channel，205/205 层全覆盖
- bf16 1.5G → int8 755M；数值对拍 top1 一致（判别式模型看 argmax，不看困惑度）

## N150 iGPU 实测上限（两次独立实验互证）

| 输入 | INT8@iGPU 耗时 | 状态 |
|---|---|---|
| ~100 tok | 1.2s | ✅ |
| 970 tok | 4.9s | ✅ |
| 2371 tok | 14.9s | ✅ |
| 3000 tok | 26.8~29.3s | ✅（需 LARGE_ALLOCATIONS） |
| 3300 tok | — | ❌ 共享内存池（CL_OUT_OF_RESOURCES） |
| 4300+ tok | — | ❌ 单算子 4GB（`GPU_ENABLE_LARGE_ALLOCATIONS: "YES"` 可破到 4000，代价是寻址模式变慢） |

- 正确键名：`{"GPU_ENABLE_LARGE_ALLOCATIONS": "YES"}`（字符串值！无 HINT 后缀！）
- 旧 v7 意图模型同机同墙（4096 贴边/4408 死）——两代模型同一硬件天花板
- **生产口径：`max_request_tokens=2000`**（硬上限 50%，覆盖并发挤占+分词器差异）

## 消费方对接参数（Hindsight）

```env
HINDSIGHT_API_DECISION_MODEL_PROVIDER=typesafe   # 或 hindsight
HINDSIGHT_API_DECISION_MODEL_MAX_REQUEST_TOKENS=2000  # 默认 26000 需下调
```

超长请求 Hindsight 自动对半拆分（源码 `_split`），无内容丢失。

## MLX 路线（Mac）

- mlx-lm ≥0.32 原生认 `qwen3_5`；`qwen3_5_text` 用壳化 config 绕过（见 TL;DR）
- M4 Pro bf16：5882tok=1.9s / 14642=5.4s / 24561=10.4s，32K 无墙
- 量化档：`--q-bits 8`（8.5bpw）/ `--q-bits 4`（4.5bpw）/ `--quant-predicate mixed_3_6`（3.7bpw）
- **vMLX 限制**：session 链走 mlx_lm 注册表，encoder 家族（ModernBERT：hs-embed-1/hs-rerank-1）无法注册（`Model type modernbert not supported`）——已提 vmlx#388；decoder 型 embedding（Qwen3-Embedding）正常
