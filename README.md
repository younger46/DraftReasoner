# DraftReasoner — MechAgent

A **ReAct-style, tool-augmented agent** for mechanical drawing VQA, built to run on the
MechVQA benchmark. Structured like `mokioclaw` (`core / agents / tools / providers /
prompts / evaluation / cli`), with LangGraph + LangChain + a deterministic OCR pipeline.

## Architecture

One `engineer` agent runs a LangGraph **ReAct loop** (`agents/react_engine.py`):

```
START → agent(LLM 决定动作) → ToolNode(执行所选工具) → (有 tool_calls 且在轮数内? → 回 agent)
                                                      → 否则 END → 输出答案
```

The LLM decides **whether to call a tool and which one**; tools are exposed to it via
`agents/react_tools.py`. Tools that read exact values are trusted more than the LLM's own
reading (OCR/GeometrySolve).

## Tools

| Tool | 负责 | 实现 |
|---|---|---|
| FigureParse 图解析 | 复合图切子图/bbox | 版面空白缝切分 |
| AnnotationExtract 标注提取 | 尺寸/基准/公差/粗糙度 | VLM 结构化抽取 |
| ViewAlign 视图对齐 | 视图/剖切对应 (PM) | VLM 结构化抽取 |
| GeometrySolve 几何求解 | 尺寸链计算 (GC) | OCR 读数 + 确定性计算 |
| StdKB 标准库 | GB/T 查表 (CJ) | 规则表 |

## Layout

```text
src/draftreasoner/
  core/        config(.env) · state(Evidence/AgentState) · media(图片工具) · agent(入口 MechAgent)
  agents/      base · registry · engineer(ReAct 工人) · react_engine(LangGraph 循环) · react_tools(工具→LangChain)
  tools/       base · registry · figure_parse/annotation_extract/view_align/geometry_solve/std_kb · vision · ocr
  providers/   vlm(BaseProvider/Null) · langchain_provider(ChatOpenAI)
  prompts/     agent.py(REACT_SYSTEM/judge/annotation/view)
  evaluation/  judge(extract_answer/score) · benchmark(load/evaluate)
  cli/         app.py(tools/agents/run/eval)
data/          MechVQA_test(benchmark + images)
```

## Quickstart

```bash
uv sync                           # install deps
cp .env.example .env              # DR_API_KEY / DR_MODEL / DR_BASE_URL / DR_BACKEND=react
uv run python main.py tools       # list registered tools
uv run python main.py run --index 0
uv run python main.py eval --limit 100   # score with LLM judge
```

Requires credentials (`DR_API_KEY`) — the ReAct loop needs a real provider.

## Extending

**Add a tool** — `tools/<name>.py` with a `@register` `Tool` subclass (`name`, `description`,
`run(**kwargs) -> ToolResult`); then add its name to `agents/react_tools.py: REACT_ALLOWED`
so the LLM can call it.

**Swap the model/provider** — implement `BaseProvider` and return it from
`providers/vlm.py: create_provider`.

Note: `GeometrySolve`(OCR) is deterministic and precise; `AnnotationExtract`/`ViewAlign`
rely on the vision model. In a ReAct loop the LLM may override a tool value, so for
critical numeric tasks the tool's high-confidence value should take precedence.
