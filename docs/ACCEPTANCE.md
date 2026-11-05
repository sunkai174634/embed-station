# 验收记录与命令

> 本文件区分本机可验收与 Linux Intel iGPU 真机验收。每条记录必须填写：输入、运行命令、真实结果/日志、判定。未执行写「待验证」，不能把测试代码存在当成测试通过。

## A. 本机自动化

- [x] Python 3.10 兼容依赖锁定；`uv lock --check --python 3.10` 通过，Ruff 0.17.3 的 check/format 均通过。当前环境无法完成 `uv sync`/pytest runner 下载；改为在最终 amd64 镜像的 Python 3.10 下直接执行 5 个项目测试函数，5/5 通过（不是 pytest runner 报告）。
- [x] 假引擎 API：单字符串/数组、空数组拒绝、模型不匹配、批量保序、请求 ID、SQLite 请求/事件记录和模型发现等 5 项测试函数通过。
- [x] 并发 worker、取消隔离：假引擎 2-worker 测试通过；NAS 上两个并发 HTTP-worker 调用由单 InferRequest 锁保护，重复真机 smoke 仍需在完整矩阵收尾时补证。
- [ ] 真实 tokenizer padding 混合结果与逐条向量数值一致性尚未比较；有限值、1024 维和 L2 norm≈1 已在 NAS iGPU 上测过。
- [ ] Dashboard 已做 HTTP 路由 smoke，但未用真实浏览器验收交互、筛选/暂停刷新和恶意错误文本渲染。
- [ ] SQLite 单请求写入已通过 HTTP smoke；重启恢复、并发写和只读/不可写场景未测。
- [x] Compose `config --quiet`、锁文件检查、`git diff --check` 通过。amd64 镜像构建、运行依赖导入、非 root 用户检查均通过。

本机与镜像实际执行记录：

```text
代码版本：未提交工作树；Python >=3.10，依赖锁按 Python 3.10 解析
基础镜像：embed-station-runtime:2026.3（本机复用现有共享 OpenVINO 2026.3 runtime）
Ruff 0.17.3：All checks passed；11 files already formatted
项目测试函数：在镜像 Python 3.10 下 5/5 通过；pytest CLI 未运行（本机 uv 依赖同步下载超时）
镜像：linux/amd64，默认用户 openvino；关键模块导入、compileall 通过
本机假引擎 HTTP：Dashboard、health、models、embeddings、SQLite record 均通过
Mac 无 Intel GPU；以上本机结果不代表 GPU 验收
```

## B. NAS 目标机（隔离临时容器，不改生产 stack）

- [x] OpenVINO 2026.3 / Intel GPU runtime / 模型 IR 与 tokenizer 资产核对；NAS `AVAILABLE_DEVICES=[CPU,GPU]`，设备名 `Intel(R) Graphics (iGPU)`；无 GPU 时服务代码拒绝启动且不回退 CPU。
- [x] GPU 初始化后 `/ready` 返回 ready；HTTP `/health` 报 `ready=true`、设备和维度符合预期。
- [x] 真实 HTTP `POST /v1/embeddings` 数组输入返回 200；返回 2 条 1024 维向量，token usage=144；短/长输入专门探针 token 数 3/577；值有限且 L2 norm 均为 1。
- [x] Dashboard 根路由 HTTP、状态/请求 API 与请求记录通过 loopback 临时 smoke；未做浏览器交互或重启恢复验收。
- [x] 性能基准已完成：jobs=1/2/4 × batch=1/2/4/8 × 短/中/长三档，36 组配置、零错误；P50/P95、吞吐、CPU/内存已记录（见下方性能结果）。GPU 利用率因主机无监控接口未测。
- [x] 容器以默认非 root `openvino` 用户跑通 GPU + HTTP smoke；模型只读挂载、临时 `/data` 可写；测试容器为 `--rm` 并已清理。NAS 现有生产服务健康状态正常。
- [ ] 没有做持久 Compose 部署、Portainer stack 或常驻服务变更；基础镜像 alias 和本次 app 镜像仅用于构建/实验验证。

真机执行记录：

```text
镜像：embed-station:gpu-smoke；NAS config image ID sha256:2841cc011667bad325354f0c5877f90111aa55d35d8a022f87b99ac51bd150fa；linux/amd64；用户 openvino
模型：NAS 已有 Qwen3-Embedding-0.6B INT8 OpenVINO IR + 配套 tokenizer，模型目录只读
运行时：OpenVINO 2026.3；Intel(R) Graphics (iGPU)；jobs=1
真模型推理：短/长混合输入 token 数 3/577；1024 维、finite=true、L2 norms=[1.0,1.0]
批量语义：混合 batch 与逐条调用逐维最大绝对误差=[0.0,0.0]
真实 HTTP：Dashboard PASS；/ready ready；/health device=iGPU；/v1/embeddings 2 条成功、usage=144 tokens；SQLite request_records=3
容器用户：非 root openvino；GPU 组 44/992；监听仅 127.0.0.1:33040
收尾：容器、NAS /tmp 脚本和归档已清理；现有生产服务健康；未改生产路由或 stack
```

## 性能结果（NAS N150 iGPU 实测，2026-11-04）

模型 Qwen3-Embedding-0.6B INT8 OpenVINO IR；OpenVINO 2026.3；GPU 推理为单 InferRequest 串行；每配置预热 1 次 + 2 次 warmup + 10 轮稳态（jobs>1 时每轮并发提交 jobs 个请求）；全程零错误。

P50 延迟（ms，括号内为吞吐 texts/s）：

| 负载 | tokens | jobs=1 b1/b2/b4/b8 | jobs=2 b1/b2/b4/b8 | jobs=4 b1/b2/b4/b8 |
|---|---:|---|---|---|
| 短 | 12 | 71/103/184/324 (13.8/19.3/21.8/25.8) | 106/206/281/413 (14.3/19.5/25.9/30.4) | 188/278/423/779 (15.0/21.1/27.8/30.6) |
| 中 | 52 | 197/304/525/1185 (5.1/6.5/7.6/6.7) | 391/607/1050/2369 (5.1/6.5/7.6/6.8) | 786/1213/1576/4146 (5.1/6.6/7.6/6.8) |
| 长 | 420 | 1309/2549/5053/10060 (0.76/0.79/0.79/0.80) | 2616/5095/10101/20105 (0.76/0.78/0.79/0.80) | 5231/7645/20192/40217 (0.76/0.79/0.79/0.80) |

资源（docker stats 采样）：CPU 平均 107-119%、峰值 368%（4 核）；容器内存平均约 1.36-1.39 GiB、峰值约 1.98 GiB。GPU 利用率无法采样（主机无 intel_gpu_top/可读引擎计数器），未测。

长文本专门实测（2026-11-04 补充，单条串行，5 次稳态）：

| 输入 tokens | 预热 ms | 稳态 P50 ms | 吞吐 tok/s |
|---:|---:|---:|---:|
| 2029 | 8629 | 9791 | 207 |
| 2549 | 12117 | 13597 | 188 |

注意：长文本吞吐随长度衰减（330→207→188），高于 ~500 token 后不再是线性 330；2048 token 单条约 9.8s。旧 sidecar 同机实测：769 tok 2831ms（272 tok/s）、1665 tok 7435ms（224 tok/s）、外推 2048 ≈ 9.5s——新旧在 2048 档基本打平（9.8s vs ~9.5s），新服务优势区间在 ≤1024 token（短/中文本快 15-49%）。
结论：

1. **吞吐上限由 GPU 串行决定**：≤500 tok 时 ~330 tok/s，随长度衰减（2048 tok 时 207 tok/s）；提高 jobs 只是把排队从 GPU 外移到 HTTP 层。
2. **延迟随"batch × 并发"线性叠加**：长文本 b8 在 jobs=1/2/4 分别 10s/20s/40s——并发请求把彼此延迟乘上并发数。
3. **batch=4 是中短文本吞吐甜点**：中文本 7.6 texts/s、396 tok/s（b8 反降，b8 批超线性变慢）。
4. **短文本场景 jobs>1 有小收益**：b8 下 25.8→30.6 texts/s（+19%），代价是单请求延迟 ×3。
5. **推荐配置**：交互/低延迟场景 `INFER_JOBS=1, BATCH_SIZE=1`；批量索引场景 `INFER_JOBS=1, BATCH_SIZE=4`（吞吐优先且不拖慢单请求）；不建议 jobs>1，除非能接受延迟线性放大。
