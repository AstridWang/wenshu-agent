"""批量运行评测集，把每道题的回答、步骤、token 存下来，并打印汇总表。

用法：
    .venv/bin/python run_eval.py v1            # 跑全部题目，结果存到 results/v1.json
    .venv/bin/python run_eval.py v1 Q01 Q03    # 只跑指定的题

"v1" 是这次运行的版本标签。以后改了提示词再跑一次，换个标签（如 v2），
两份结果就可以并排对比。
"""
import json
import sys
import time
from pathlib import Path

from openai import OpenAI

from agent import MODEL, SYSTEM_PROMPT, ask, load_api_key

HERE = Path(__file__).parent
EVAL_SET = HERE / "eval_set.json"
RESULTS_DIR = HERE / "results"


def main():
    if len(sys.argv) < 2:
        sys.exit("请给这次运行起个版本标签，例如：.venv/bin/python run_eval.py v1")
    label, only_ids = sys.argv[1], set(sys.argv[2:])

    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    if only_ids:
        cases = [c for c in cases if c["id"] in only_ids]
    client = OpenAI(api_key=load_api_key(), base_url="https://api.deepseek.com")

    results = []
    for case in cases:
        print(f"正在跑 {case['id']}：{case['question']}")
        start = time.time()
        try:
            answer, record = ask(client, case["question"], verbose=False)
        except Exception as e:  # 单题出错不影响其他题
            answer, record = f"运行出错：{e}", {"steps": [], "input_tokens": 0, "output_tokens": 0}
        results.append(
            {
                **case,
                "answer": answer,
                "steps": record["steps"],
                "tool_calls": len(record["steps"]),
                "sql_errors": sum("SQL 执行出错" in s["result"] for s in record["steps"]),
                "input_tokens": record["input_tokens"],
                "output_tokens": record["output_tokens"],
                "seconds": round(time.time() - start, 1),
            }
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    out_file = RESULTS_DIR / f"{label}.json"
    out_file.write_text(
        json.dumps(
            # 把当时用的模型和提示词一起存下来，以后才说得清这份结果是哪个版本跑出来的
            {"label": label, "model": MODEL, "system_prompt": SYSTEM_PROMPT, "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n{'题号':<6}{'类型':<8}{'工具调用':>8}{'SQL报错':>8}{'输入token':>10}{'输出token':>10}{'耗时(秒)':>9}")
    for r in results:
        print(
            f"{r['id']:<6}{r['category']:<8}{r['tool_calls']:>10}{r['sql_errors']:>9}"
            f"{r['input_tokens']:>12}{r['output_tokens']:>12}{r['seconds']:>10}"
        )
    n = len(results) or 1
    print(
        f"\n共 {len(results)} 题，平均工具调用 {sum(r['tool_calls'] for r in results) / n:.1f} 次，"
        f"平均输入 {sum(r['input_tokens'] for r in results) / n:.0f} token，"
        f"平均输出 {sum(r['output_tokens'] for r in results) / n:.0f} token"
    )
    print(f"完整结果已保存到 {out_file.relative_to(HERE)}")


if __name__ == "__main__":
    main()
