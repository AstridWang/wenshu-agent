"""问数 Agent 的网页界面。

启动：.venv/bin/streamlit run app.py
"""
import json

import pandas as pd
import streamlit as st
from openai import OpenAI

import create_db
from agent import DB_PATH, MODEL, ask, find_api_key

EXAMPLES = [
    "哪个渠道的退款率最高？",
    "9月信息流投放的效率和之前比有什么变化？",
    "各学科的投放ROI分别是多少？",
    "付费渠道和免费渠道的客单价差多少？",
]

st.set_page_config(page_title="问数 Agent", page_icon="📊", layout="centered")
# 模型的回答里常带大标题，在对话气泡里显得过大，统一缩小
st.markdown(
    """<style>
    [data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3 { font-size: 1.15rem; padding: 0.6rem 0 0.3rem; }
    </style>""",
    unsafe_allow_html=True,
)

# 部署到云端时仓库里没有数据库文件，首次启动自动生成
if not DB_PATH.exists():
    create_db.main()


def get_api_key():
    """依次从 .env / 环境变量、云端密钥配置、侧边栏输入框获取 Key。"""
    key = find_api_key()
    if not key:
        try:
            key = st.secrets["DEEPSEEK_API_KEY"]
        except Exception:  # 本地没有配置 secrets 文件时会报错，忽略即可
            key = None
    if not key:
        key = st.sidebar.text_input(
            "DeepSeek API Key",
            type="password",
            help="Key 只保存在当前浏览器会话里，不会被记录。可在 platform.deepseek.com 获取。",
        )
    return key


def show_steps(steps):
    """把 Agent 的每一步工具调用展示出来：SQL 用代码块，查询结果用表格。"""
    for i, step in enumerate(steps, 1):
        if step["tool"] == "get_schema":
            st.markdown(f"**第 {i} 步 · 查看表结构**")
            continue
        st.markdown(f"**第 {i} 步 · 执行查询**")
        args = step["args"]
        st.code(args.get("sql", "") if isinstance(args, dict) else str(args), language="sql")
        try:
            # run_sql 成功时返回 JSON（后面可能跟一句行数提示），出错时返回普通文字
            data, end = json.JSONDecoder().raw_decode(step["result"])
            st.dataframe(
                pd.DataFrame(data["rows"], columns=data["columns"]),
                hide_index=True,
                use_container_width=True,
            )
            if step["result"][end:].strip():
                st.caption(step["result"][end:].strip())
        except ValueError:
            st.error(step["result"])


def show_assistant(turn):
    with st.expander(f"分析过程（调用工具 {len(turn['steps'])} 次）"):
        show_steps(turn["steps"])
    st.markdown(turn["answer"])
    st.caption(
        f"输入 {turn['input_tokens']} token · 输出 {turn['output_tokens']} token · 模型 {MODEL}"
    )


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("数据说明")
    st.markdown(
        "一家在线教育公司 2026 年 4–9 月的**模拟数据**：\n"
        "- **渠道**：5 个，分付费和免费\n"
        "- **产品**：5 个，覆盖数学、英语、语文\n"
        "- **订单**：约 6000 条，含退款状态\n"
        "- **投放花费**：按月、按渠道"
    )
    st.header("试试这些问题")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state.pending = example
    st.divider()
    if st.button("清空对话", use_container_width=True):
        st.session_state.turns = []

api_key = get_api_key()

# ---------------------------------------------------------------------------
# 主区域
# ---------------------------------------------------------------------------
st.title("📊 问数 Agent")
st.caption("用大白话提问，Agent 自己看表结构、写 SQL、查数并给出结论。每一步都可以展开检查。")

# turns 里每个元素是一轮问答：{"question", "answer", "steps", "input_tokens", "output_tokens"}
turns = st.session_state.setdefault("turns", [])

for turn in turns:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        show_assistant(turn)

question = st.chat_input("输入你的问题，例如：哪个渠道的退款率最高？")
question = question or st.session_state.pop("pending", None)

if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if not api_key:
            st.warning("请先在左侧填入 DeepSeek API Key。")
            st.stop()
        # 把之前几轮的问题和结论带上，这样可以追问"那按月看呢"
        history = []
        for turn in turns:
            history.append({"role": "user", "content": turn["question"]})
            history.append({"role": "assistant", "content": turn["answer"]})

        with st.status("正在分析…", expanded=True) as status:

            def on_step(name, args, result):
                if name == "get_schema":
                    st.write("查看表结构")
                else:
                    st.write("执行查询")
                    st.code(args.get("sql", "") if isinstance(args, dict) else str(args), language="sql")

            try:
                client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                answer, record = ask(
                    client, question, verbose=False, history=history, on_step=on_step
                )
            except Exception as e:
                status.update(label="出错了", state="error")
                st.error(f"调用模型失败：{e}")
                st.stop()
            status.update(label="分析完成", state="complete", expanded=False)

    turns.append(
        {
            "question": question,
            "answer": answer,
            "steps": record["steps"],
            "input_tokens": record["input_tokens"],
            "output_tokens": record["output_tokens"],
        }
    )
    st.rerun()  # 重新渲染，让这一轮按统一的历史样式显示
