# DraftReasoner 架构文档（ARCHITECTURE）

> `draftreasoner` 是一个 **ReAct 范式、工具增强的机械工程图纸理解智能体**，面向 MechVQA 基准做问答。
> 核心思想：**LLM 自行判断“要不要调工具、调哪个工具”**；能由确定性工具（OCR/尺寸链计算）给出精确值的，
> 让工具读，避免 LLM 看图误读（如 Φ61 读成 Φ62）。

---

## 1. 设计原则

- **ReAct 循环**：`agent(LLM) → ToolNode → (回 agent | 收尾)`，LLM 是决策者。
- **工具=精确证据**：工具返回 `{ok, data, evidence, error}` 的 JSON observation，回灌给 LLM 继续推理。
- **确定性优先**：GeometrySolve+OCR 读精确尺寸并计算（GC）；VLM 工具（AnnotationExtract/ViewAlign）做语义抽取。
- **证据可溯源**：每个工具产出 `Evidence`（source/claim/confidence），供最终答案解释。
- **可复现、可离线测**：无 API key 用 `NullProvider`；工具主链路可离线验证（测试）。

---

## 2. 整体数据流（ReAct）

```text
   Question + Drawing(image)
             |
             v
   MechAgent (入口) -- plan_for(subcategory) -> [ "engineer" ]
             |
             v
   EngineerAgent.run --(langchain)--> ReAct 循环 (agents/react_engine.py)
             |
             +---------------------------------------------------------------+
             |    agent 节点: LLM 结合(question + 图 + 观测) 决定动作           |
             |       -> 需要工具? 产出 AIMessage(tool_calls)  <-----------------+
             |       -> 可以回答? 产出 AIMessage(content=<answer>)            |
             |                        |                                      |
             |                        v                                      |
             |    ToolNode: 执行所选工具(FigureParse/AnnotationExtract/       |
             |                ViewAlign/GeometrySolve/StdKB)                  |
             |                        |  -> ToolMessage(observation json)     |
             |                        v                                      |
             |    条件边: 有 tool_calls 且 < MAX_TOOL_ROUNDS(=5) ?  ----------+ (回 agent)
             |                        | 否则 END
             |                        v
             +------------------ 输出 <think>...</think><answer>...</answer>
             |
             v
   evaluate: extract_answer -> LLM judge (语义等价) -> 0/1 -> 聚合
```

- `agents/react_engine.py` 用 **LangGraph `StateGraph`** 搭建：`START -> agent -> ToolNode -> (conditional) 回 agent | END`。
- `agents/react_tools.py` 把 5 个 MechAgent 工具包装成 LangChain `tool`，交给 `ToolNode` 与 `llm.bind_tools`。

---

## 3. 目录结构（ReAct-only）

```text
DraftReasoner/
├── main.py                  # 入口 -> cli.app
├── pyproject.toml           # 依赖/入口点 draftreasoner:main
├── .env.example .gitignore .python-version uv.lock
├── data/                    # MechVQA_test(图+benchmark) + MechVQA_train&val
├── tests/test_smoke.py      # ReAct 视角冒烟测试
└── src/draftreasoner/
    ├── __init__.py          # __version__ + main
    ├── __main__.py          # python -m draftreasoner
    ├── core/                # 配置/状态/运行时
    │   ├── config.py        #   Settings / _load_dotenv / resolve_image_path
    │   ├── state.py         #   Evidence / AgentHandoff / AgentState
    │   ├── media.py         #   image_data_uri / crop_to（共享图片工具）
    │   └── agent.py         #   MechAgent（入口薄封装）
    ├── agents/
    │   ├── base.py          #   BaseAgent / AgentContext / AgentResult
    │   ├── registry.py      #   register_agent / get_agent / build_agents
    │   ├── engineer.py      #   EngineerAgent（ReAct 工人）
    │   ├── react_engine.py  #   ReAct 循环（LangGraph StateGraph）
    │   └── react_tools.py   #   工具 -> LangChain 适配（wrap_react_tools/REACT_ALLOWED）
    ├── tools/
    │   ├── base.py          #   Tool / ToolResult
    │   ├── registry.py      #   register / build_tools / get_tool
    │   ├── figure_parse.py · annotation_extract.py · view_align.py
    │   ├── geometry_solve.py · std_kb.py
    │   └── vision.py · ocr.py          # VLM 调用助手 / OCR 读数
    ├── providers/
    │   ├── vlm.py           #   BaseProvider / NullProvider / create_provider
    │   └── langchain_provider.py # LangChainProvider(ChatOpenAI)
    ├── prompts/agent.py     # REACT_SYSTEM / JUDGE / ANNOTATION / VIEW / ANSWER_TEMPLATE
    ├── evaluation/          # judge(extract_answer/score) · benchmark(load/evaluate)
    └── cli/app.py           # tools / agents / run / eval
```

---

## 4. 核心抽象（数据模型）

### 4.1 `core/state.py`
| 类型 | 字段 | 说明 |
|---|---|---|
| `Evidence` | `source, claim, confidence, detail` | 一条可回查证据；`as_dict()` |
| `AgentHandoff` | `from_agent, to_agent, instruction, result` | 智能体间移交（为多智能体预留） |
| `AgentState` | 见下 | 一条问题的运行时状态 |

`AgentState`：`question/image_path/metadata`、`route`(保留)、`evidence[]`、`reasoning/answer/final_output`、`confidence`、`extras`、`handoffs/subagent_results`、`retries/max_retries`。属性 `subcategory/capability/language`；方法 `add_evidence()`、`evidence_summary()`。

### 4.2 `tools/base.py`
`ToolResult(ok, data, error, evidence, confidence)`；`Tool`（`name/description` + 抽象 `run(**kwargs)` + `ok()/fail()`）。工具**只产结果与证据，不写答案**。

### 4.3 `agents/base.py`
`AgentContext(settings, provider, tools)`；`AgentResult(ok, answer, reasoning, confidence, evidence, handoffs, final_output)`；`BaseAgent`（`name/description/can_handle()/run(state, ctx)`）。

### 4.4 `providers/vlm.py`
`BaseProvider`（抽象 `chat`/`judge`）、`NullProvider`（离线）、`create_provider`（有 key→LangChainProvider）。**全程不直接用 openai 客户端**。

---

## 5. 运行时：`core/agent.py` 的 `MechAgent`

薄入口。`__init__` 建 provider/tools/agents；`run(record)`/`answer(...)` 依据 `plan_for(subcategory)`（默认 `["engineer"]`）调度 `EngineerAgent`，取其 `final_output`。多智能体扩展点保留：`PLANS` + 注册新 `BaseAgent`，supervisor 用 LLM 合并。

---

## 6. 单元智能体：`agents/engineer.py` 的 `EngineerAgent`

唯一单元，`run()` 直接调用 `agents/react_engine.run_react(state, ctx)` 启动 ReAct 循环。

---

## 7. ReAct 循环：`agents/react_engine.py`

- **`_ReactState`**：`messages: Annotated[list[BaseMessage], add_messages]`。
- **`agent` 节点**：`llm.invoke(messages)`（`llm` 已 `bind_tools`），返回 `AIMessage`（可能带 `tool_calls`，或直接 `content`）。
- **`tools` 节点**：`langgraph.prebuilt.ToolNode(tools)` 执行所选工具，返回 `ToolMessage`（observation）。
- **条件边 `_route`**：末条有 `tool_calls` 且 `AIMessage+tool_calls` 次数 `< MAX_TOOL_ROUNDS(=5)` → 回 `agent`；否则 `END`。
- **收尾**：`_last_text(messages)` 取最近非空 AI 内容作为答案（避免达上限被强制收尾时为空）。
- **启动**：`SystemMessage(REACT_SYSTEM)` + `HumanMessage(question + image)`；`agent.invoke(...)`。

> 若 `provider` 为 `NullProvider`（无 key）则抛 `RuntimeError`（ReAct 必须有真实 LLM）。

### `agents/react_tools.py`
把 5 个工具（`REACT_ALLOWED`）包装成 LangChain `tool`。`run(_focus)` 绑定 `image_path/provider/question`，返回 `json.dumps({ok,data,evidence,error})` 供模型观察。

---

## 8. 工具层

| 工具 | 负责 | 实现 | 说明 |
|---|---|---|---|
| `FigureParse` | 复合图切分/bbox | 版面空白缝 | 前置定位 |
| `AnnotationExtract` | 尺寸/基准/公差/粗糙度 | VLM(`ANNOTATION_PROMPT`) | 结构化为证据 |
| `ViewAlign` | 视图/剖切对应(PM) | VLM(`VIEW_PROMPT`) | 结构化为证据 |
| `GeometrySolve` | 尺寸链计算(GC) | OCR(`ocr.read_dimensions`)+`parse_gc_formula` | **确定性**，精确值 |
| `StdKB` | GB/T 查表(CJ) | 规则表 | 确定性 |

- `tools/vision.py`：`call_json/parse_json/claims_from/has_provider`，供 VLM 工具、`crop_to/image_data_uri` 来自 `core/media.py`。
- `tools/ocr.py`：RapidOCR 读直径（裁绘图区→过滤 `^[0OΦØDd]` 头 token→取最大两个=外/内径）。
- 依赖：`core/media.py` 提供 `image_data_uri()`/`crop_to()`，被 `providers/langchain_provider`、`agents/react_engine`、`tools/vision` 共用。

---

## 9. Prompt 层：`prompts/agent.py`

- `REACT_SYSTEM`：ReAct 主提示词（机械工程师 + 工具说明 + “优先用 GeometrySolve/OCR，不要猜”）。
- `ANSWER_TEMPLATE`（`<think>/<answer>`）、`JUDGE_PROMPT`、`ANNOTATION_PROMPT`、`VIEW_PROMPT`。

---

## 10. Provider 层：`providers/`

- `vlm.py`：抽象 + `NullProvider` + `create_provider`。
- `langchain_provider.py`：`LangChainProvider` 用 `langchain_openai.ChatOpenAI`；`chat()` 转 LangChain 消息并在需要时把图附为 `image_url`；`judge()` 复用 `JUDGE_PROMPT`。
- 换后端：实现 `BaseProvider` 并在 `create_provider` 返回。

---

## 11. 评测层：`evaluation/`

- `judge.py`：`extract_answer()`（取 `<answer>`，回退 `</think>` 后/全文）、`score()`（`provider.judge` 0/1）。
- `benchmark.py`：`load_benchmark()`；`evaluate()` 逐条 `agent.run` → `extract_answer` → `score`，聚合 `total` 与按 `subcategory/capability/difficulty/language` 的 `{n,correct,accuracy}`，保留 `per_record`。

---

## 12. CLI：`cli/app.py`

```bash
uv run python main.py tools               # 列出工具
uv run python main.py agents              # 列出单元智能体
uv run python main.py run --index 0       # 跑一条
uv run python main.py eval --limit 100    # 评测
```

---

## 13. 测试：`tests/test_smoke.py`

GeometrySolve 尺寸链、StdKB、5 工具注册、`extract_answer`、`wrap_react_tools`、ReAct 无 provider 守卫（`NullProvider` → RuntimeError）。全部离线可跑。

---

## 14. 配置参考

| 变量 | 默认 | 说明 |
|---|---|---|
| `DR_API_KEY` | 空 | 模型 key（ReAct 必须） |
| `DR_MODEL` | `gpt-4o` | 模型名 |
| `DR_BASE_URL` | 空 | OpenAI 兼容端点 |
| `DR_TEMPERATURE` | `0.0` | 温度 |
| `DR_BACKEND` | `react` | 唯一范式 |
| `DR_MAX_RETRIES` | `2` | 兼容字段 |
| `DR_RETRY_CONF_FLOOR` | `0.6` | 兼容字段 |
| `DR_VERBOSE` | `true` | 评测是否逐条打印 |

`_load_dotenv()` 读 `.env`，缺失回退 `.env.example`，去行内注释。

---

## 15. 依赖

```toml
dependencies = [
    "pillow>=11.0.0",
    "langchain>=1.2.15",
    "langchain-openai>=1.1.13",
    "langgraph>=1.1.6",
    "rapidocr-onnxruntime>=1.4.4",
]
```

---

## 16. 扩展指南

- **加工具**：`tools/<name>.py` 写 `@register` 的 `Tool` 子类；在 `agents/react_tools.py: REACT_ALLOWED` 加名字即可被 LLM 调用。
- **换 provider**：实现 `BaseProvider`，在 `providers/vlm.py: create_provider` 返回。
- **多智能体**：注册新 `BaseAgent` + 改 `core/agent.py: PLANS`。

---

## 17. 已知边界与建议

- **ReAct 的 LLM 可能“不信任并覆盖”工具的精确值**：实测 GC 时模型把 OCR 读对的 `Φ61` 覆盖成自己看图读的 `Φ67`（得 8mm，正确为 5mm）。纯 ReAct 在“精确尺寸”上反而不如“工具值为权威”的管线。
- **建议**：给 ReAct 加“**确定性锚点**”——一旦确定性工具（如 GeometrySolve）返回 `confidence>=0.95` 的精确数值，最终答案强制采用，LLM 不得用自己的看图读数覆盖。或在 `REACT_SYSTEM` 中强化“工具数值为权威”。
- **成本/延迟**：ReAct 每轮一次模型调用，`MAX_TOOL_ROUNDS=5`；批量评测需控制 `--limit`。
