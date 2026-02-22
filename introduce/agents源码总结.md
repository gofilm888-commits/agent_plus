# Agents 模块源码总结

> HelloAgents 智能体实现层完整解读，涵盖 7 种 Agent 范式：SimpleAgent、FunctionCallAgent、ReActAgent、ReflectionAgent、PlanAndSolveAgent、ToolAwareSimpleAgent、SkillAgent。

---

## 一、模块概览

### 1.1 定位与职责

`agents` 模块提供多种 Agent 实现，均继承自 `core.Agent` 基类，实现 `run(input_text, **kwargs) -> str` 接口。不同 Agent 采用不同的推理范式与工具调用方式。

### 1.2 目录结构

```
agents/
├── __init__.py           # 导出 7 种 Agent
├── simple_agent.py       # 简单对话 + 可选工具（文本标记解析）
├── function_call_agent.py # OpenAI Function Calling
├── react_agent.py        # ReAct（Thought-Action-Observation）
├── reflection_agent.py   # 反思迭代优化
├── plan_solve_agent.py   # 规划-执行
├── tool_aware_agent.py   # SimpleAgent + 工具调用监听
└── skill_agent.py       # 技能驱动（继承 FunctionCallAgent）
```

### 1.3 Agent 对比总览

| Agent | 工具调用方式 | 典型场景 |
|-------|--------------|----------|
| SimpleAgent | 文本标记 `[TOOL_CALL:name:params]` | 简单对话、轻量工具 |
| FunctionCallAgent | OpenAI 原生 function calling | 结构化工具调用 |
| ReActAgent | `tool_name[input]` 格式解析 | 推理+行动交替 |
| ReflectionAgent | 无工具 | 迭代优化（代码、文档） |
| PlanAndSolveAgent | 无工具 | 多步骤规划执行 |
| ToolAwareSimpleAgent | 同 SimpleAgent + 监听 | 日志、调试、追踪 |
| SkillAgent | 同 FunctionCallAgent，技能即工具 | 技能驱动、按需加载 |

---

## 二、SimpleAgent

### 2.1 类定义与定位

**文件**：simple_agent.py

**定位**：简单对话 Agent，支持可选的工具调用。工具调用通过**文本标记**在 LLM 输出中解析，而非原生 function calling。

### 2.2 构造函数

```python
def __init__(
    self,
    name: str,
    llm: HelloAgentsLLM,
    system_prompt: Optional[str] = None,
    config: Optional[Config] = None,
    tool_registry: Optional['ToolRegistry'] = None,
    enable_tool_calling: bool = True
):
```

- `tool_registry` 为 None 时，`enable_tool_calling` 强制为 False
- `enable_tool_calling = enable_tool_calling and tool_registry is not None`

### 2.3 工具调用格式

LLM 需在输出中使用以下格式：

```
[TOOL_CALL:{tool_name}:{parameters}]
```

**参数格式**：
- 多参数：`key=value` 逗号分隔，如 `[TOOL_CALL:calculator_multiply:a=12,b=8]`
- 单参数：`key=value` 或直接文本
- JSON：`{...}` 格式也可解析

### 2.4 核心方法

| 方法 | 说明 |
|------|------|
| `_get_enhanced_system_prompt()` | 注入工具描述与调用格式说明 |
| `_parse_tool_calls(text)` | 正则 `\[TOOL_CALL:([^:]+):([^\]]+)\]` 解析 |
| `_parse_tool_parameters(tool_name, parameters)` | 解析参数字符串为 dict，支持 JSON、key=value、简单推断 |
| `_convert_parameter_types(tool_name, param_dict)` | 按工具定义转换类型 |
| `_infer_action(tool_name, param_dict)` | 为 memory/rag 等推断 action |
| `_execute_tool_call(tool_name, parameters)` | 执行工具并返回格式化结果 |

### 2.5 run 流程

1. 构建 messages（system + history + user）
2. 若无工具：直接 `llm.invoke`，保存历史，返回
3. 若有工具：循环 `max_tool_iterations`（默认 3）：
   - 调用 `llm.invoke`
   - 解析 `_parse_tool_calls(response)`
   - 若有工具调用：执行、将结果作为 user 消息追加、继续循环
   - 若无：作为最终回答，退出
4. 超迭代次数且无最终回答：再调用一次 `llm.invoke` 获取回答
5. 保存到 `_history`，返回

### 2.6 工具管理

- `add_tool(tool, auto_expand=True)`：注册工具，支持展开
- `remove_tool(tool_name)`：调用 `tool_registry.unregister_tool`（注意：ToolRegistry 实际为 `unregister`）
- `list_tools()`：列出工具名
- `has_tools()`：是否启用工具

### 2.7 stream_run

- 无工具时：`llm.stream_invoke` 逐块 yield
- 有工具时：未实现流式工具调用，使用非流式逻辑

---

## 三、FunctionCallAgent

### 3.1 类定义与定位

**文件**：function_call_agent.py

**定位**：基于 **OpenAI 原生 function calling**，通过 `tools`、`tool_choice` 参数调用，模型返回 `tool_calls`，Agent 执行后以 `role: tool` 消息追加。

### 3.2 构造函数

```python
def __init__(
    self,
    name: str,
    llm: HelloAgentsLLM,
    system_prompt: Optional[str] = None,
    config: Optional[Config] = None,
    tool_registry: Optional["ToolRegistry"] = None,
    enable_tool_calling: bool = True,
    default_tool_choice: Union[str, dict] = "auto",
    max_tool_iterations: int = 3,
):
```

- `default_tool_choice`：`"auto"`（模型决定）、`"none"`（不调用）、或 `{"type":"function","function":{"name":"xxx"}}` 强制调用
- `max_tool_iterations`：工具调用最大轮数

### 3.3 核心方法

| 方法 | 说明 |
|------|------|
| `_get_system_prompt()` | 注入工具描述，提示“通过函数调用使用以下工具” |
| `_build_tool_schemas()` | 从 ToolRegistry 生成 OpenAI function schema |
| `_extract_message_content(raw_content)` | 处理 str/list 等 content 格式 |
| `_parse_function_call_arguments(arguments)` | `json.loads` 解析参数 |
| `_convert_parameter_types(tool_name, param_dict)` | 类型转换 |
| `_execute_tool_call(tool_name, arguments)` | 执行 Tool 或 register_function 注册的函数 |
| `_invoke_with_tools(messages, tools, tool_choice, **kwargs)` | 直接调用 `llm._client.chat.completions.create(tools=..., tool_choice=...)` |

### 3.4 Tool Schema 构建

- **Tool 对象**：遍历 `get_all_tools()`，每个 tool 的 `get_parameters()` 转为 `properties`、`required`
- **register_function**：访问 `_functions`，每个函数固定 `input` 参数
- 参数类型映射：`_map_parameter_type` 确保为 JSON Schema 允许类型

### 3.5 run 流程

1. 构建 messages（system + history + user）
2. 若无 tool_schemas：直接 `llm.invoke`，返回
3. 循环 `max_tool_iterations`：
   - `_invoke_with_tools(messages, tools, tool_choice)`
   - 解析 `assistant_message.tool_calls`
   - 若有：追加 assistant（含 tool_calls）、逐条执行、追加 tool 消息，`current_iteration += 1`，继续
   - 若无：`final_response = content`，退出
4. 若超迭代且无 final_response：`tool_choice="none"` 再调用一次，强制文本回答
5. 保存历史，返回

### 3.6 add_tool 与 expandable

- 若 tool 有 `auto_expand` 且 `get_expanded_tools()` 非空：注册所有子工具
- 否则：`register_tool(tool)`

### 3.7 stream_run

- 当前实现：`result = self.run(...); yield result`，非真正流式

---

## 四、ReActAgent

### 4.1 类定义与定位

**文件**：react_agent.py

**定位**：**ReAct（Reasoning and Acting）** 范式，结合推理与行动。每轮输出 `Thought:` 与 `Action:`，Action 格式为 `tool_name[input]` 或 `Finish[结论]`。

### 4.2 默认提示词模板

```python
DEFAULT_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手...
## 可用工具
{tools}

## 工作流程
Thought: 分析问题...
Action: tool_name[input] 或 Finish[研究结论]

## 当前任务
**Question:** {question}

## 执行历史
{history}
"""
```

### 4.3 构造函数

```python
def __init__(
    self,
    name: str,
    llm: HelloAgentsLLM,
    tool_registry: Optional[ToolRegistry] = None,
    system_prompt: Optional[str] = None,
    config: Optional[Config] = None,
    max_steps: int = 5,
    custom_prompt: Optional[str] = None
):
```

- `tool_registry` 为 None 时创建空 `ToolRegistry()`
- `max_steps`：最大 Thought-Action-Observation 轮数
- `custom_prompt`：可覆盖默认模板

### 4.4 核心方法

| 方法 | 说明 |
|------|------|
| `_parse_output(text)` | 正则提取 `Thought:`、`Action:` |
| `_parse_action(action_text)` | 正则 `(\w+)\[(.*)\]` 提取 tool_name、input |
| `_parse_action_input(action_text)` | 提取 `Finish[xxx]` 中的结论 |

### 4.5 run 流程

1. `current_history = []`，`current_step = 0`
2. 循环 `while current_step < max_steps`：
   - 构建 prompt（tools、question、history）
   - `llm.invoke(messages)`
   - `_parse_output(response)` → thought, action
   - 若 action 以 `Finish` 开头：解析结论，保存历史，返回
   - 否则：`_parse_action(action)` → tool_name, tool_input
   - `tool_registry.execute_tool(tool_name, tool_input)`
   - 追加 `Action:`、`Observation:` 到 `current_history`
3. 超步数：返回“无法在限定步数内完成”

### 4.6 工具执行

- `execute_tool(name, input_text)`：ToolRegistry 将 `input_text` 包装为 `{"input": input_text}` 传入 `tool.run()`
- 部分工具需 `action`、`query` 等，可能需适配

### 4.7 add_tool 与 MCP

- 若 tool 有 `auto_expand` 和 `_available_tools`：为每个 MCP 子工具创建包装 Tool，`run` 时调用 `call_tool`
- 否则：直接 `register_tool(tool)`

---

## 五、ReflectionAgent

### 5.1 类定义与定位

**文件**：reflection_agent.py

**定位**：**自我反思与迭代优化**，无工具调用。流程：初始执行 → 反思 → 优化 → 迭代，直到反思认为“无需改进”。

### 5.2 内部 Memory 类

```python
class Memory:
    records: List[Dict]  # {"type": "execution"|"reflection", "content": str}
    add_record(record_type, content)
    get_trajectory() -> str   # 格式化所有记录
    get_last_execution() -> str
```

### 5.3 默认提示词

```python
DEFAULT_PROMPTS = {
    "initial": "任务: {task}\n请提供一个完整、准确的回答。",
    "reflect": "原始任务: {task}\n当前回答: {content}\n请分析质量，指出不足，提出改进建议。若已很好则回答'无需改进'。",
    "refine": "原始任务: {task}\n上一轮回答: {last_attempt}\n反馈意见: {feedback}\n请提供改进后的回答。"
}
```

### 5.4 构造函数

```python
def __init__(
    self,
    name: str,
    llm: HelloAgentsLLM,
    system_prompt: Optional[str] = None,
    config: Optional[Config] = None,
    max_iterations: int = 3,
    custom_prompts: Optional[Dict[str, str]] = None
):
```

### 5.5 run 流程

1. 重置 `memory = Memory()`
2. 初始执行：`prompts["initial"]` → `_get_llm_response` → `add_record("execution", ...)`
3. 循环 `max_iterations`：
   - 反思：`prompts["reflect"]` → feedback → `add_record("reflection", ...)`
   - 若 feedback 含“无需改进”或“no need for improvement”：退出
   - 优化：`prompts["refine"]` → refined_result → `add_record("execution", ...)`
4. 返回 `memory.get_last_execution()`

### 5.6 适用场景

代码生成、文档写作、分析报告等需要多轮迭代优化的任务。

---

## 六、PlanAndSolveAgent

### 6.1 类定义与定位

**文件**：plan_solve_agent.py

**定位**：**规划-执行**范式。先由 Planner 将问题分解为步骤列表，再由 Executor 逐步执行，每步仅输出当前步骤答案。

### 6.2 Planner 类

```python
class Planner:
    def plan(self, question: str, **kwargs) -> List[str]:
```

- 使用 `DEFAULT_PLANNER_PROMPT`，要求输出 Python 列表格式
- 从响应中提取 ````python [...] ``` 块，`ast.literal_eval` 解析
- 返回步骤字符串列表

### 6.3 Executor 类

```python
class Executor:
    def execute(self, question: str, plan: List[str], **kwargs) -> str:
```

- 逐步执行，每步 prompt 含：原始问题、完整计划、历史步骤与结果、当前步骤
- 仅输出当前步骤答案，追加到 history
- 返回最后一步的答案

### 6.4 构造函数

```python
def __init__(
    self,
    name: str,
    llm: HelloAgentsLLM,
    system_prompt: Optional[str] = None,
    config: Optional[Config] = None,
    custom_prompts: Optional[Dict[str, str]] = None  # {"planner": "", "executor": ""}
):
```

### 6.5 run 流程

1. `plan = self.planner.plan(input_text)`
2. 若 plan 为空：返回“无法生成有效的行动计划”
3. `final_answer = self.executor.execute(input_text, plan)`
4. 保存历史，返回

### 6.6 适用场景

多步骤推理、数学问题、复杂分析等。

---

## 七、ToolAwareSimpleAgent

### 7.1 类定义与定位

**文件**：tool_aware_agent.py

**定位**：**SimpleAgent 子类**，增加工具调用监听、增强解析、流式工具调用支持。

### 7.2 构造函数

```python
def __init__(
    self,
    *args: Any,
    tool_call_listener: Optional[Callable[[dict], None]] = None,
    **kwargs: Any
):
```

- `tool_call_listener`：回调，接收 `{agent_name, tool_name, raw_parameters, parsed_parameters, result}`

### 7.3 重写方法

| 方法 | 说明 |
|------|------|
| `_execute_tool_call` | 执行后调用 `tool_call_listener` |
| `_parse_tool_calls` | 支持嵌套 `[]`、字符串内 `[]`，正确匹配结束位置 |
| `_sanitize_parameters` | 清理参数：类型转换、tags 解析、字符串规范化 |
| `_normalize_string` | 移除多余引号、补全未闭合括号 |
| `stream_run` | 流式输出中检测 `[TOOL_CALL:...]`，执行后继续流式 |

### 7.4 流式工具调用逻辑

- 使用 `process_residual` 生成器，在流式 chunk 中查找完整 `[TOOL_CALL:...]`
- 遇到完整调用：暂停 yield，执行工具，将结果追加为 user 消息，下一轮流式
- 支持 `max_tool_iterations` 轮

### 7.5 静态方法

- `attach_registry(agent, registry)`：附加 ToolRegistry
- `_sanitize_parameters`、`_normalize_string`、`_coerce_sequence`：参数清理

---

## 八、SkillAgent

### 8.1 类定义与定位

**文件**：skill_agent.py

**定位**：**技能驱动**，继承 FunctionCallAgent。从技能目录加载 SKILL.md，将技能注册为 SkillTool，通过 function call 实现渐进式披露。

### 8.2 构造函数

```python
def __init__(
    self,
    name: str,
    llm: HelloAgentsLLM,
    skill_dir: Optional[Union[str, Path]] = None,
    skill_registry: Optional["SkillRegistry"] = None,
    system_prompt: Optional[str] = None,
    config: Optional[Config] = None,
    tool_registry: Optional["ToolRegistry"] = None,
    max_tool_iterations: int = 3,
):
```

- `skill_dir` 与 `skill_registry` 二选一
- 若提供 `skill_dir`：`SkillRegistry.load_from_directory(base_dir)`
- `registry.register_to_tool_registry(tool_registry)`：将技能注册为工具
- 调用 `super().__init__` 传入 tool_registry、enable_tool_calling=True

### 8.3 默认系统提示

```python
"你是一个可靠的 AI 助理，能够通过调用技能获取领域专业知识。"
"当用户问题涉及特定领域时，请主动调用相关技能获取详细指导，再基于技能内容给出回答。"
```

### 8.4 list_skills

- `return self.skill_registry.list_skills()`：列出已加载技能名

### 8.5 与 FunctionCallAgent 的关系

- 完全复用 FunctionCallAgent 的 run、tool 执行逻辑
- 仅扩展初始化：加载技能、注册为工具、设置默认 system_prompt

---

## 九、继承关系图

```
                    Agent (core)
                         │
        ┌────────────────┼────────────────┬─────────────────┐
        │                │                │                 │
        ▼                ▼                ▼                 ▼
  SimpleAgent    FunctionCallAgent  ReflectionAgent  PlanAndSolveAgent
        │                │
        │                │
        ▼                ▼
  ToolAwareSimpleAgent  SkillAgent
  (继承 SimpleAgent)    (继承 FunctionCallAgent)

  ReActAgent (独立继承 Agent)
```

---

## 十、工具调用方式对比

| Agent | 工具调用机制 | 解析方式 |
|-------|--------------|----------|
| SimpleAgent | 文本标记 `[TOOL_CALL:name:params]` | 正则解析，需 LLM 按格式输出 |
| FunctionCallAgent | OpenAI function calling | 模型原生返回 tool_calls |
| ReActAgent | `tool_name[input]` | 正则解析 Action 行 |
| ToolAwareSimpleAgent | 同 SimpleAgent | 增强解析，支持嵌套 |
| SkillAgent | 同 FunctionCallAgent | 技能即工具 |

---

## 十一、使用示例

### SimpleAgent

```python
from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools.builtin import SearchTool

llm = HelloAgentsLLM(model="gpt-4")
registry = ToolRegistry()
registry.register_tool(SearchTool())
agent = SimpleAgent(name="助手", llm=llm, tool_registry=registry)
print(agent.run("搜索 Python 最新版本"))
```

### FunctionCallAgent

```python
from hello_agents import FunctionCallAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools.builtin import SearchTool, CalculatorTool

llm = HelloAgentsLLM(model="gpt-4")
registry = ToolRegistry()
registry.register_tool(SearchTool())
registry.register_tool(CalculatorTool())
agent = FunctionCallAgent(name="助手", llm=llm, tool_registry=registry)
print(agent.run("计算 123 * 456"))
```

### ReActAgent

```python
from hello_agents import ReActAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools.builtin import SearchTool

llm = HelloAgentsLLM(model="gpt-4")
registry = ToolRegistry()
registry.register_tool(SearchTool())
agent = ReActAgent(name="研究助手", llm=llm, tool_registry=registry, max_steps=5)
print(agent.run("北京今天天气怎么样？"))
```

### ReflectionAgent

```python
from hello_agents import ReflectionAgent, HelloAgentsLLM

llm = HelloAgentsLLM(model="gpt-4")
agent = ReflectionAgent(name="代码助手", llm=llm, max_iterations=3)
print(agent.run("写一个 Python 快速排序函数"))
```

### PlanAndSolveAgent

```python
from hello_agents import PlanAndSolveAgent, HelloAgentsLLM

llm = HelloAgentsLLM(model="gpt-4")
agent = PlanAndSolveAgent(name="解题助手", llm=llm)
print(agent.run("小明有 10 个苹果，吃了 3 个，又买了 5 个，现在有几个？"))
```

### SkillAgent

```python
from hello_agents import SkillAgent, HelloAgentsLLM
from pathlib import Path

llm = HelloAgentsLLM(model="gpt-4")
agent = SkillAgent(
    name="技能助手",
    llm=llm,
    skill_dir=Path(".cursor/skills")
)
print(agent.run("如何创建 Cursor 规则？"))
```

---

## 十二、注意事项与扩展

1. **SimpleAgent.remove_tool**：调用 `unregister_tool`，ToolRegistry 实际为 `unregister`，可能需适配
2. **ReActAgent.execute_tool**：ToolRegistry 将 input 包装为 `{"input": input_text}`，复杂工具需适配
3. **FunctionCallAgent**：依赖 `llm._client`，需确保 LLM 使用 OpenAI 兼容客户端
4. **ToolAwareSimpleAgent**：流式工具调用在 chunk 边界可能截断，需保留 residual 缓冲
5. **扩展**：继承对应 Agent，重写 `run` 或 `_execute_tool_call` 等即可定制

---

*文档基于 hello_agents agents 模块源码整理，适用于理解与二次开发。*
