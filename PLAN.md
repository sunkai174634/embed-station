# Embed Station — 最终实施计划

## 目标与边界

构建一个独立、通用的 Embedding 服务：在 NAS 级 Intel iGPU 上执行 Qwen3-Embedding-0.6B OpenVINO IR 推理，提供标准 `POST /v1/embeddings`、可视化 Dashboard、逐请求记录与服务日志。项目、镜像和文档不包含任何特定调用方名称；客户端通过通用 API 接入。首个正式运行目标是 Linux/amd64、Intel N150 级 iGPU。

此计划是新工程，不移植旧工程的调度器、输入截断、双车道、意图模型或专用客户端。旧代码只用于核对 OpenVINO 加载、模型输入输出、tokenizer 与归一化的已验证实现；新工程独立组织，避免复制业务耦合和冗余。

## v1 功能清单

1. **Embedding API**：`POST /v1/embeddings` 接受单字符串或非空字符串数组；返回 OpenAI 格式 `object`、`data[{object,index,embedding}]`、`model`、`usage`。`GET /v1/models` 返回实际装载模型。输入错误返回清晰 4xx；模型/设备失败不得伪装为成功。模型 ID 必须与配置一致。支持数组输入、保序、L2 归一化和正确的 Qwen3 last-token pooling（padding 方向须经 tokenizer 配置与实测验证）。不得在接口中暴露原文或任意客户标识。
2. **GPU-only 推理**：从只读模型目录加载 OpenVINO IR 与本地 tokenizer；编译到 `GPU`，无 GPU、缺模型或编译失败则启动失败，不自动落 CPU。CPU 做 tokenizer 属于前处理，不是推理回退。推理层维护模型对象、设备信息和可安全并发的推理请求池；不要让多个请求同时操作同一个 `InferRequest`。
3. **并发与批处理**：使用 OpenVINO 官方 `AsyncInferQueue`（多 `InferRequest`）或等价、经证明安全的请求池；从可验证的 `jobs=1` 基线开始。实现一个简单、可关闭的短窗口微批处理器，支持请求内多文本，也能合并相近时刻的请求；默认以正确性与低延迟为先，批大小/窗口/jobs 由目标机基准决定。不要预设“必须串行”，也不要承诺并行一定更快。必须正确处理完成回调、异常、取消和响应归属；不得在 FastAPI 事件循环上做长时间阻塞推理。
4. **Dashboard**：同容器同源静态页面，至少三部分：服务状态（GPU 名称/模型/维度/启动时间/近期延迟及成功率）、逐请求记录（时间、请求 ID、输入条数、token 数、耗时、结果或错误；可按 ID 定位）、服务日志（时间、级别、事件、Request ID、详细错误摘要；可筛选与暂停刷新）。展示真实数据，不造演示数值。默认不储存输入原文、向量或原文摘要，避免泄露；如需查看请求内容，另行讨论隐私和授权。
5. **可追溯观测**：结构化 JSON 日志输出 stdout；SQLite 持久化请求元数据和可供 Dashboard 查询的关键服务事件，数据目录独立卷；无 Redis/外部 DB/nginx 第二容器。重启后历史可查；未落库的异常终止应标明记录语义，不声称精确一次。默认不自动删除历史；文档说明磁盘增长、备份和清理策略。浏览器通过受限分页 API 拉取，禁止一次返回全部历史。
6. **健康与部署**：`GET /health` 为进程存活探针，`GET /ready` 仅在模型已加载且 GPU 可实际执行时成功；Docker HEALTHCHECK 调 readiness；`Dockerfile` 与 Compose 示例应适配 Linux/amd64、`/dev/dri` 和只读模型卷、可持久化数据卷。容器非 root 运行的权限须核实；不要硬编码 NAS 的 group id、内网 IP 或模型宿主路径。默认只为可信内网使用，不把无鉴权 Dashboard 暴露公网；文档给出反向代理鉴权建议。

## 明确不做

- CPU 推理回退；额外模型、rerank、意图处理、聊天、热切换。
- 预设的输入截断、优先级车道、旧协议扩展字段、固定 512-token 假设。
- 压测前拍脑袋给出“<100ms”“3.6×”或镜像体积等硬指标。
- 把旧工程里的专用路径、名称、服务接线或部署配置带入本工程。
- v1 就上线 Prometheus、Redis、分布式协调、复杂鉴权或多模型集群。

**输入长度原则**：不要静默截断。使用模型 tokenizer 的实际能力以及真实 GPU 运行结果；超出已验证能力时必须明确失败并留可诊断日志。任何安全上限必须由目标硬件实测决定并文档化，不从旧工程照搬。

## 项目产物

- `README.md`：部署、配置、API 样例、Dashboard、隐私/磁盘说明、性能报告链接。
- `docs/ARCHITECTURE.md`：模块边界、数据流、并发语义、失败路径。
- `docs/ACCEPTANCE.md`：命令和验收记录表，区分本地与 NAS 真机。
- `src/embed_station/`：清晰职责的 settings / runtime / inference / batch / telemetry / api 模块；`static/`：无前端构建链的页面；`tests/`：单元+集成测试。
- 锁定依赖的 `pyproject.toml`、Dockerfile、Compose 示例、`.env.example`、许可证说明、`.gitignore`。

## 实施顺序与当前状态

- [x] 1. 盘点模型 IR 输入输出、tokenizer/padding/pooling 行为及目标部署资产；列实证，不复制旧架构。**已完成**：NAS iGPU 实际加载 IR；混合短长 batch 与分别单条请求的 1024 维向量逐维最大绝对误差均为 0.0。
- [x] 2. 写最小引擎与 API；用假引擎测试基础向量契约、批量保序与错误语义。**本机已完成**：API、模型列表、校验和假引擎 HTTP 测试通过；NAS 真模型 HTTP 也返回正确向量。
- [ ] 3. 实现安全的 OpenVINO 推理与可关闭微批，先建立 jobs=1 / batch=1 基准，再让目标机实验选择配置。**基准已完成**：36 组配置实测零错误；N150 上 AsyncInferQueue 并发回调会 SIGSEGV，已改单 InferRequest + 锁；结论为交互场景 jobs=1/batch=1、批量索引 jobs=1/batch=4，详见 ACCEPTANCE 性能结果。
- [ ] 4. 实现 SQLite 记录、日志查询 API 和 Dashboard；确认重启保留历史、分页与敏感数据默认不落盘。**实现部分完成**：请求记录、事件查询、分页、XSS 防护和 HTTP Dashboard smoke 通过；真实浏览器交互、进程重启恢复待验。
- [ ] 5. Docker/Compose 打包，完成自动化测试、可构建验证与真机 GPU 验收。**部分完成**：linux/amd64 镜像已构建并在 NAS 用临时非 root 容器完成真实模型与 HTTP smoke；完整 pytest CLI、本机 uv sync、性能基准及持久部署待验。

## 本机已验证

- [x] Python 3.10 `uv lock --check` 通过；Ruff 0.17.3 check / format 通过。
- [x] 5 个项目测试函数在 Python 3.10 amd64 应用镜像中执行通过；pytest CLI 未运行，本机依赖同步下载超时。
- [x] Compose `config --quiet` 与 `git diff --check` 通过。
- [x] `linux/amd64` 镜像构建、关键模块导入、非 root 用户及假引擎 HTTP smoke 通过。
- [x] NAS N150 iGPU 真实推理及非 root 临时 HTTP smoke 通过；详见 `docs/ACCEPTANCE.md`。
- [ ] 浏览器实测、SQLite 重启恢复及性能基准待做。

## 技术验证门槛

- 单元：API 契约、请求内批量保序、池并发安全、callback 异常、取消、无 GPU 启动失败、SQLite 并发写/分页、Dashboard 数据一致性与 XSS 防护。
- 集成：OpenAI SDK 或标准 HTTP 真调用；向量有限、维度与模型一致、L2 范数接近 1；padding 混合长短文本后与单条结果一致（合理数值误差）；模型重启后观测历史仍在。
- 目标机：记录硬件/驱动/OpenVINO 版本、模型文件及量化、预热次数、输入 token 分布；串行比较 batch=1/2/4/8 与 jobs=1/2/4，单请求 P50/P95、吞吐、内存/GPU 占用、失败率。独立隔离实验，不压现有生产服务；没有真实测量就标「待测」。

## 提交与授权边界

Pi 只在本机 `~/Projects/embed-station` 写代码与运行本地测试；不更改任何常驻 agent 配置，不部署或修改 NAS，不推送远端仓库，不写入凭据，不删除旧工程。遇到缺少 Linux Intel GPU 时提交真实测试结果与明确的真机待验收项，不能虚构性能和部署完成状态。代码写完由 BayMax 再做独立验收。