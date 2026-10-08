"""使用日志：把每一次提问和用户反馈记进 usage.db。

这张表就是这个产品的"埋点数据"，之后可以直接用 SQL 分析：
采纳率（点赞占比）、单次成本、平均步数、哪类问题容易失败等。
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

LOG_DB = Path(__file__).parent / "usage.db"


def _connect():
    conn = sqlite3.connect(LOG_DB)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS query_log (
            log_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at    TEXT,     -- 提问时间
            session_id    TEXT,     -- 一次打开页面算一个会话
            turn_index    INTEGER,  -- 该会话里的第几个问题（从 1 开始），大于 1 即为追问
            question      TEXT,
            conclusion    TEXT,
            answer_json   TEXT,     -- 完整的结构化答案
            sqls_json     TEXT,     -- 本次执行过的全部 SQL
            steps_json    TEXT,     -- 完整的分析过程（含每次查询的结果），用于重新打开历史对话
            tool_calls    INTEGER,  -- 工具调用次数
            sql_errors    INTEGER,  -- 其中 SQL 报错的次数
            has_chart     INTEGER,  -- 是否给出了图表
            input_tokens  INTEGER,
            output_tokens INTEGER,
            seconds       REAL,     -- 从提问到出答案的耗时
            feedback      INTEGER,  -- 1 点赞，-1 点踩，空表示没有反馈
            feedback_at   TEXT
        )
        """
    )
    # 早期版本建的表没有 steps_json 这一列，补上
    if "steps_json" not in [row[1] for row in conn.execute("PRAGMA table_info(query_log)")]:
        conn.execute("ALTER TABLE query_log ADD COLUMN steps_json TEXT")
    return conn


def log_query(session_id, turn_index, question, record, seconds):
    """记录一次提问，返回 log_id（后面写反馈时要用）。"""
    final = record.get("final", {})
    sqls = [s["args"].get("sql", "") for s in record["steps"] if s["tool"] == "run_sql"]
    conn = _connect()
    cur = conn.execute(
        """INSERT INTO query_log (created_at, session_id, turn_index, question, conclusion,
               answer_json, sqls_json, steps_json, tool_calls, sql_errors, has_chart,
               input_tokens, output_tokens, seconds)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            datetime.now().isoformat(timespec="seconds"),
            session_id,
            turn_index,
            question,
            final.get("conclusion", ""),
            json.dumps(final, ensure_ascii=False),
            json.dumps(sqls, ensure_ascii=False),
            json.dumps(record["steps"], ensure_ascii=False),
            len(record["steps"]),
            sum("SQL 执行出错" in s["result"] for s in record["steps"]),
            int(bool(final.get("chart"))),
            record["input_tokens"],
            record["output_tokens"],
            round(seconds, 1),
        ),
    )
    conn.commit()
    log_id = cur.lastrowid
    conn.close()
    return log_id


def set_feedback(log_id, value):
    """value: 1 点赞，-1 点踩，None 取消。"""
    conn = _connect()
    conn.execute(
        "UPDATE query_log SET feedback = ?, feedback_at = ? WHERE log_id = ?",
        (value, datetime.now().isoformat(timespec="seconds") if value else None, log_id),
    )
    conn.commit()
    conn.close()


def list_sessions(limit=30):
    """返回最近的历史对话：[(session_id, 第一个问题, 开始时间, 问题数)]，最新的在前。"""
    conn = _connect()
    rows = conn.execute(
        """SELECT session_id,
                  (SELECT question FROM query_log q2
                    WHERE q2.session_id = q1.session_id ORDER BY log_id LIMIT 1),
                  MIN(created_at), COUNT(*)
             FROM query_log q1
            GROUP BY session_id
            ORDER BY MAX(log_id) DESC
            LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return rows


def load_session(session_id):
    """读出一个历史对话的全部问答，格式与界面里的 turns 一致。"""
    conn = _connect()
    rows = conn.execute(
        """SELECT log_id, question, answer_json, steps_json,
                  input_tokens, output_tokens, seconds, feedback
             FROM query_log WHERE session_id = ? ORDER BY log_id""",
        (session_id,),
    ).fetchall()
    conn.close()
    return [
        {
            "log_id": log_id,
            "question": question,
            "final": json.loads(answer_json or "{}"),
            "steps": json.loads(steps_json or "[]"),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "seconds": seconds or 0,
            "feedback": feedback,
        }
        for log_id, question, answer_json, steps_json, input_tokens, output_tokens, seconds, feedback in rows
    ]

