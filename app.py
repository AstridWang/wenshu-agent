"""问数 Agent 的网页界面。

启动：.venv/bin/streamlit run app.py
"""
import html
import json
import time
import uuid

import pandas as pd
import streamlit as st
from openai import OpenAI

import create_db
from agent import DB_PATH, MODEL, ask, find_api_key, format_answer
from usage_log import list_sessions, load_session, log_query, set_feedback

EXAMPLES = [
    ("渠道质量", "哪个渠道的退款率最高？"),
    ("投放效率", "9月信息流投放的效率和之前比有什么变化？"),
    ("产品销售", "上个月卖得最好的产品是哪个？"),
    ("数据边界", "各学科的投放ROI分别是多少？"),
]

st.set_page_config(page_title="问数 Agent", page_icon="📊", layout="centered")
st.markdown(
    """<style>
    /* 补充说明里模型偶尔会用标题，统一缩小 */
    [data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3 { font-size: 1.05rem; padding: 0.5rem 0 0.2rem; }
    .definition-card { background: rgba(128,128,128,0.08); border-radius: 8px;
        padding: 0.7rem 1rem; font-size: 0.86rem; line-height: 1.7; margin: 0.4rem 0 0.8rem; }
    .definition-card b { font-weight: 600; }
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


def result_to_df(result):
    """把 run_sql 的返回值转成表格。出错或无法解析时返回 None。"""
    try:
        # 成功时是 JSON（后面可能跟一句行数提示），出错时是普通文字
        data, _ = json.JSONDecoder().raw_decode(result)
        return pd.DataFrame(data["rows"], columns=data["columns"])
    except (ValueError, KeyError, TypeError):
        return None


def show_chart(chart, steps):
    """按模型给的图表配置画图。数据直接取自对应那次查询的结果，而不是模型转述的数字。"""
    step = next((s for s in steps if s.get("query_id") == chart.get("query_id")), None)
    df = result_to_df(step["result"]) if step else None
    x, ys, series = chart.get("x"), chart.get("y") or [], chart.get("series")
    # 模型给的列名可能和查询结果对不上，对不上就不画，不影响其余内容
    if df is None or x not in df.columns or not ys or any(y not in df.columns for y in ys):
        return
    if series and (series not in df.columns or len(ys) > 1):
        series = None
    for y in ys:
        df[y] = pd.to_numeric(df[y], errors="coerce")
    # 模型有时把量级差很远的指标（如 ROI 和获客成本）放进同一张图，小的那条会被压成直线。
    # 只保留和第一个指标量级接近的列。
    first_max = df[ys[0]].abs().max() or 1
    ys = [y for y in ys if 0.05 <= (df[y].abs().max() or 0) / first_max <= 20]
    if chart.get("title"):
        st.markdown(f"**{chart['title']}**")
    draw = st.line_chart if chart.get("type") == "line" else st.bar_chart
    draw(df, x=x, y=ys[0] if len(ys) == 1 else ys, color=series, height=300)


def show_steps(steps):
    """展示 Agent 的每一步工具调用：SQL 用代码块，查询结果用表格。"""
    for i, step in enumerate(steps, 1):
        if step["tool"] == "get_schema":
            st.markdown(f"**第 {i} 步 · 查看表结构**")
            continue
        st.markdown(f"**第 {i} 步 · 执行查询**")
        st.code(step["args"].get("sql", ""), language="sql")
        df = result_to_df(step["result"])
        if df is None:
            st.error(step["result"])
        else:
            st.dataframe(df, hide_index=True, use_container_width=True)


def on_feedback(log_id):
    value = st.session_state.get(f"feedback_{log_id}")  # 0 点踩，1 点赞，None 未选
    set_feedback(log_id, None if value is None else (1 if value == 1 else -1))


def show_assistant(turn):
    final = turn["final"]
    # 1. 结论
    st.markdown(final.get("conclusion", ""))
    # 2. 图表
    if final.get("chart"):
        show_chart(final["chart"], turn["steps"])
    # 3. 口径卡片
    # 这些文字来自模型，转义后再放进 HTML
    lines = [f"<b>计算口径</b>　{html.escape(str(d))}" for d in final.get("definitions") or []]
    if final.get("scope"):
        lines.append(f"<b>数据范围</b>　{html.escape(str(final['scope']))}")
    if lines:
        st.markdown(f"<div class='definition-card'>{'<br>'.join(lines)}</div>", unsafe_allow_html=True)
    # 4. 补充说明和分析过程默认收起
    if final.get("detail"):
        with st.expander("补充说明"):
            st.markdown(final["detail"])
    with st.expander(f"分析过程（调用工具 {len(turn['steps'])} 次）"):
        show_steps(turn["steps"])
    # 5. 反馈和本次消耗
    left, right = st.columns([1, 3])
    with left:
        st.feedback(
            "thumbs",
            key=f"feedback_{turn['log_id']}",
            on_change=on_feedback,
            args=(turn["log_id"],),
        )
    with right:
        st.caption(
            f"耗时 {turn['seconds']:.0f} 秒 · 输入 {turn['input_tokens']} token · "
            f"输出 {turn['output_tokens']} token · {MODEL}"
        )


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------
def open_session(session_id):
    """重新打开一个历史对话。之后的提问会接在这个对话后面。"""
    turns = load_session(session_id)
    for turn in turns:
        turn["answer"] = format_answer(turn["final"])
        # 恢复之前点过的赞或踩（反馈组件里 1 是赞，0 是踩）
        if turn["feedback"] is not None:
            st.session_state[f"feedback_{turn['log_id']}"] = 1 if turn["feedback"] == 1 else 0
    st.session_state.session_id = session_id
    st.session_state.turns = turns


api_key = get_api_key()
session_id = st.session_state.setdefault("session_id", uuid.uuid4().hex)
# turns 里每个元素是一轮问答
turns = st.session_state.setdefault("turns", [])

with st.sidebar:
    if st.button("＋ 新对话", use_container_width=True, type="primary"):
        st.session_state.turns = []
        st.session_state.session_id = uuid.uuid4().hex
        st.rerun()

    st.subheader("历史对话")
    sessions = list_sessions()
    if not sessions:
        st.caption("还没有历史对话")
    for sid, first_question, started_at, count in sessions:
        title = first_question if len(first_question) <= 16 else first_question[:16] + "…"
        label = f"{'● ' if sid == session_id else ''}{title}"
        if st.button(
            label,
            key=f"session_{sid}",
            use_container_width=True,
            help=f"{started_at[:16].replace('T', ' ')} · {count} 个问题",
        ):
            open_session(sid)
            st.rerun()

    st.divider()
    with st.expander("数据说明"):
        st.markdown(
            "一家在线教育公司 2026 年 4–9 月的**模拟数据**：\n"
            "- **渠道**：5 个，分付费和免费\n"
            "- **产品**：5 个，覆盖数学、英语、语文\n"
            "- **订单**：约 6000 条，含退款状态\n"
            "- **投放花费**：按月、按渠道"
        )

# ---------------------------------------------------------------------------
# 主区域
# ---------------------------------------------------------------------------
if not turns and "pending" not in st.session_state:
    # 欢迎页：还没有提问时显示
    st.title("📊 问数 Agent")
    st.markdown(
        "用大白话问数据。我会自己查表、写 SQL、算数，然后告诉你**结论、计算口径和依据**。"
    )
    st.markdown("##### 可以这样问")
    columns = st.columns(2)
    for i, (tag, example) in enumerate(EXAMPLES):
        with columns[i % 2]:
            if st.button(f"**{tag}**\n\n{example}", key=f"example_{i}", use_container_width=True):
                st.session_state.pending = example
                st.rerun()
    st.caption(
        "每条回答都会标明指标是怎么算的，并可展开查看执行过的每一条 SQL。"
        "遇到数据不支持的问题，会直接说明算不了。"
    )

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
        # 把之前几轮的问题和答案带上，这样可以追问"那按月看呢"
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
                    st.code(args.get("sql", ""), language="sql")

            start = time.time()
            try:
                client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                answer, record = ask(
                    client, question, verbose=False, history=history, on_step=on_step
                )
            except Exception as e:
                status.update(label="出错了", state="error")
                st.error(f"调用模型失败：{e}")
                st.stop()
            seconds = time.time() - start
            status.update(label="分析完成", state="complete", expanded=False)

    turns.append(
        {
            "question": question,
            "answer": answer,
            "final": record["final"],
            "steps": record["steps"],
            "input_tokens": record["input_tokens"],
            "output_tokens": record["output_tokens"],
            "seconds": seconds,
            "log_id": log_query(session_id, len(turns) + 1, question, record, seconds),
        }
    )
    st.rerun()  # 重新渲染，让这一轮按统一的历史样式显示
