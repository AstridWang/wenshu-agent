"""问数 Agent 第 1 版：用自然语言提问，它自己看表结构、写 SQL、查数、给结论。

用法：
    .venv/bin/python agent.py "哪个渠道的退款率最高？"
    .venv/bin/python agent.py            # 不带问题则进入连续提问模式
"""
import json
import os
import sqlite3
import sys
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).parent
DB_PATH = HERE / "edu.db"
MODEL = "deepseek-flash"
MAX_STEPS = 10  # 最多循环多少轮，防止模型陷入死循环一直烧钱
MAX_ROWS = 50   # 查询结果最多返回多少行，防止把几千行数据塞进上下文

# ---------------------------------------------------------------------------
# 一、提示词：给模型的"岗位说明"
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是一家在线教育公司的数据分析助手。用户会用自然语言提问，你通过查询数据库来回答。

背景：数据库里的数据截至 2026-09-30，今天按 2026-10-01 计算。只有问题里出现"上个月""最近"这类相对时间时才需要用到这一点；问题没有指定时间范围时，用全部数据回答，不要额外附上其他时间段的结果。

工作方式：
1. 先调用 get_schema 了解有哪些表和字段，不要凭猜测写 SQL。
2. 用 run_sql 查询数据。数据库是 SQLite，只能执行 SELECT。列别名用中文（如 AS 退款率），方便直接展示。
3. 如果 SQL 报错或结果不合理，分析原因后修正重试。
4. 得到足够的数据后，调用 submit_answer 提交答案，不要直接用文字回复。

回答要求：
- 结论先行：conclusion 直接回答问题并带上关键数字，不超过 100 字。用户问什么答什么，不要扩展到没问的话题。
- 结论里的每个数字都必须来自查询结果，不得编造。
- definitions 只列结论里出现的指标，每个指标一条，最多 3 条。问题里的词有歧义时，在结论里用半句话说明你采用的理解。
- 数据不足以回答时：conclusion 只说明算不了以及缺什么数据。不要用假设、分摊或估算去凑出数字，也不要改为回答另一个问题。发现算不了就立刻提交，不必继续查询。
- 结果是时间趋势或多项对比时，提供 chart。"""

# ---------------------------------------------------------------------------
# 二、工具：模型只能"说"它想调用什么，真正执行的是下面这些普通函数
# ---------------------------------------------------------------------------


def get_schema():
    """返回所有表的建表语句和 3 行样例数据。"""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    parts = []
    for name, ddl in conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
    ):
        rows = conn.execute(f"SELECT * FROM {name} LIMIT 3").fetchall()
        parts.append(f"{ddl}\n-- 样例数据: {rows}")
    conn.close()
    return "\n\n".join(parts)


def run_sql(sql):
    """执行一条只读 SQL，返回列名和结果行。出错时把错误信息返回给模型，让它自己修。"""
    if not sql.strip().lower().startswith(("select", "with")):
        return "错误：只允许执行 SELECT 查询。"
    # mode=ro 以只读方式打开数据库，即使模型写出修改语句也改不了数据
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        cur = conn.execute(sql)
        columns = [d[0] for d in cur.description]
        rows = cur.fetchmany(MAX_ROWS + 1)
    except sqlite3.Error as e:
        return f"SQL 执行出错：{e}"
    finally:
        conn.close()
    note = ""
    if len(rows) > MAX_ROWS:
        rows = rows[:MAX_ROWS]
        note = f"\n（结果超过 {MAX_ROWS} 行，只显示前 {MAX_ROWS} 行，请用聚合或加筛选条件）"
    return json.dumps({"columns": columns, "rows": rows}, ensure_ascii=False) + note


# 工具说明书：告诉模型有哪些工具、各自干什么、需要什么参数
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": "查看数据库里所有表的结构（字段名、类型）和样例数据。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "在 SQLite 数据库上执行一条 SELECT 查询，返回结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "要执行的 SQL 语句"}
                },
                "required": ["sql"],
            },
        },
    },
]

# submit_answer 不是用来"做事"的工具，而是让模型按固定栏目交卷，
# 这样界面可以把结论、口径、图表分开展示，而不是拿到一整段自由发挥的文字。
TOOLS.append(
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "提交最终答案。得到足够数据后必须调用它来结束。",
            "parameters": {
                "type": "object",
                "properties": {
                    "conclusion": {
                        "type": "string",
                        "description": "直接回答问题的结论，1~3 句话，包含关键数字。不要标题和表格。",
                    },
                    "definitions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "结论中每个指标的计算口径，每条一句，如“退款率 = 退款订单数 ÷ 总订单数”。没有用到指标时给空数组。",
                    },
                    "scope": {
                        "type": "string",
                        "description": "数据范围：时间段和筛选条件，如“2026-04 至 2026-09，全部渠道”。",
                    },
                    "detail": {
                        "type": "string",
                        "description": "可选。对结论的补充，Markdown 格式，可含表格，不超过 200 字，只写与问题直接相关的内容。没有必要补充时不填。",
                    },
                    "chart": {
                        "type": "object",
                        "description": "可选。图表数据直接取自某一次查询的结果。",
                        "properties": {
                            "query_id": {
                                "type": "integer",
                                "description": "用哪一次查询的结果画图，即 run_sql 返回内容开头的查询编号",
                            },
                            "type": {
                                "type": "string",
                                "enum": ["bar", "line"],
                                "description": "line 用于时间趋势，bar 用于类别对比",
                            },
                            "x": {"type": "string", "description": "横轴列名"},
                            "y": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "纵轴的数值列名。多列时量纲必须相同",
                            },
                            "series": {
                                "type": "string",
                                "description": "可选。用来区分多条线或多组柱的分类列名",
                            },
                            "title": {"type": "string", "description": "图表标题"},
                        },
                        "required": ["query_id", "type", "x", "y"],
                    },
                },
                "required": ["conclusion", "definitions", "scope"],
            },
        },
    }
)

TOOL_FUNCTIONS = {"get_schema": get_schema, "run_sql": run_sql}


def format_answer(final):
    """把结构化答案拼成一段纯文字，给命令行、评测和多轮对话的历史用。"""
    parts = [final.get("conclusion", "")]
    if final.get("definitions"):
        parts.append("计算口径：" + "；".join(final["definitions"]))
    if final.get("scope"):
        parts.append("数据范围：" + final["scope"])
    if final.get("detail"):
        parts.append(final["detail"])
    return "\n\n".join(p for p in parts if p)

# ---------------------------------------------------------------------------
# 三、Agent 循环
# ---------------------------------------------------------------------------


def find_api_key():
    """优先读环境变量，其次读同目录下的 .env 文件。找不到返回 None。"""
    key = os.environ.get("DEEPSEEK_API_KEY")
    env_file = HERE / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("DEEPSEEK_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'")
    return key or None


def load_api_key():
    key = find_api_key()
    if not key:
        sys.exit("没有找到 API Key：请把 Key 填进 .env 文件的 DEEPSEEK_API_KEY= 后面")
    return key


def ask(client, question, verbose=True, history=None, on_step=None):
    """跑完一个问题，返回 (最终答案, 运行记录)。运行记录后面做评测要用。

    history: 之前几轮的问答 [{"role": "user"/"assistant", "content": ...}]，用于追问。
    on_step: 每执行完一次工具就回调 on_step(工具名, 参数, 结果)，界面用它实时显示过程。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *(history or []),
        {"role": "user", "content": question},
    ]
    record = {"question": question, "steps": [], "input_tokens": 0, "output_tokens": 0}
    query_count = 0

    for step in range(1, MAX_STEPS + 1):
        # 每一轮都把完整的 messages 重新发给模型，模型自己不记得任何东西
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS
        )
        record["input_tokens"] += response.usage.prompt_tokens
        record["output_tokens"] += response.usage.completion_tokens
        message = response.choices[0].message
        # 把模型这一轮的回复原样追加进历史
        messages.append(message.model_dump(exclude_none=True))

        # 模型没按要求交卷而是直接回了文字：把这段文字当作结论，保证不中断
        if not message.tool_calls:
            record["final"] = {"conclusion": message.content or "", "definitions": [], "scope": ""}
            record["answer"] = message.content
            return message.content, record

        for call in message.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            # 模型交卷了：这是循环的正常出口
            if name == "submit_answer" and args.get("conclusion"):
                record["final"] = args
                record["answer"] = format_answer(args)
                return record["answer"], record

            # 其他工具：执行后把结果作为 tool 消息发回去
            try:
                result = TOOL_FUNCTIONS[name](**args)
            except Exception as e:  # 模型给了不存在的工具名或错误的参数
                result = f"工具调用失败：{e}"
            step_record = {"tool": name, "args": args, "result": result}
            content = result
            if name == "run_sql":
                # 给每次查询编号，模型画图时用编号指明"用哪次查询的结果"
                query_count += 1
                step_record["query_id"] = query_count
                content = f"[查询编号 {query_count}] {result}"
            record["steps"].append(step_record)
            if on_step:
                on_step(name, args, result)
            if verbose:
                print(f"\n[第 {step} 轮] 调用 {name} {args.get('sql', '')}")
                print(f"  返回: {result[:300]}{'…' if len(result) > 300 else ''}")
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": content}
            )

    message = f"超过 {MAX_STEPS} 轮仍未得出结论，已停止。"
    record["final"] = {"conclusion": message, "definitions": [], "scope": ""}
    record["answer"] = None
    return message, record


def main():
    if not DB_PATH.exists():
        sys.exit("没有找到 edu.db，请先运行：.venv/bin/python create_db.py")
    client = OpenAI(api_key=load_api_key(), base_url="https://api.deepseek.com")

    def run(question):
        answer, record = ask(client, question)
        print(f"\n===== 回答 =====\n{answer}")
        print(
            f"\n（共调用工具 {len(record['steps'])} 次，"
            f"输入 {record['input_tokens']} token，输出 {record['output_tokens']} token）"
        )

    if len(sys.argv) > 1:
        run(" ".join(sys.argv[1:]))
        return
    print("问数 Agent 已启动，输入问题后回车，输入 q 退出。")
    while True:
        question = input("\n你的问题> ").strip()
        if question.lower() in ("q", "quit", "exit"):
            break
        if question:
            run(question)


if __name__ == "__main__":
    main()
