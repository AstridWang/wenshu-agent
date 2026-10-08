"""生成练习用的在线教育模拟数据库 edu.db（全部是编造数据）。

数据里故意埋了几个"已知答案"，后面做评测时用来判断 Agent 答得对不对：
  1. 达人直播渠道的退款率远高于其他渠道（约 65%）
  2. 英语产品线在付费渠道的实收金额是三个学科里最低的
  3. 9 月信息流投放的花费上涨约五成，但订单量基本没变
  4. 投放花费只记到渠道，没有拆到产品，所以"各学科的投放 ROI"算不出来
     （用来测试 Agent 会不会在数据不足时老实说算不了，而不是硬编一个数）
"""
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

random.seed(42)  # 固定随机种子，每次生成的数据完全一样，评测答案才稳定
DB_PATH = Path(__file__).parent / "edu.db"

CHANNELS = [
    # (id, 名称, 类型, 每月订单量基数, 退款概率)
    (1, "信息流投放", "付费", 320, 0.18),
    (2, "达人直播", "付费", 260, 0.65),
    (3, "短视频自播", "付费", 180, 0.22),
    (4, "老带新", "免费", 150, 0.08),
    (5, "自然流量", "免费", 120, 0.12),
]

PRODUCTS = [
    # (id, 名称, 学科, 价格)
    (1, "数学思维训练营", "数学", 2999),
    (2, "数学思维年课", "数学", 8999),
    (3, "英语启蒙训练营", "英语", 1999),
    (4, "英语分级阅读年课", "英语", 6999),
    (5, "语文素养训练营", "语文", 2499),
]

MONTHS = ["2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09"]

# 每个付费渠道每月的投放花费（元）；9 月信息流花费明显上涨
AD_SPEND = {
    1: [520000, 540000, 530000, 560000, 550000, 820000],
    2: [610000, 630000, 600000, 650000, 640000, 660000],
    3: [280000, 290000, 300000, 310000, 300000, 320000],
}


def random_day(month):
    y, m = map(int, month.split("-"))
    first = date(y, m, 1)
    next_first = date(y + (m == 12), m % 12 + 1, 1)
    return first + timedelta(days=random.randrange((next_first - first).days))


def main():
    DB_PATH.unlink(missing_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(
        """
        CREATE TABLE channels (
            channel_id   INTEGER PRIMARY KEY,
            channel_name TEXT,
            channel_type TEXT
        );
        CREATE TABLE products (
            product_id   INTEGER PRIMARY KEY,
            product_name TEXT,
            subject      TEXT,
            price        INTEGER
        );
        CREATE TABLE orders (
            order_id    INTEGER PRIMARY KEY,
            user_id     INTEGER,
            channel_id  INTEGER,
            product_id  INTEGER,
            order_date  TEXT,
            amount      INTEGER,
            status      TEXT,
            refund_date TEXT
        );
        CREATE TABLE ad_spend (
            month      TEXT,
            channel_id INTEGER,
            spend      INTEGER
        );
        """
    )
    conn.executemany(
        "INSERT INTO channels VALUES (?,?,?)", [c[:3] for c in CHANNELS]
    )
    conn.executemany("INSERT INTO products VALUES (?,?,?,?)", PRODUCTS)

    orders = []
    order_id = 0
    for month in MONTHS:
        for channel_id, _, _, base, refund_p in CHANNELS:
            for _ in range(int(base * random.uniform(0.9, 1.1))):
                order_id += 1
                # 英语产品在付费渠道卖得少，所以英语线的投放 ROI 偏低
                weights = [30, 12, 8, 4, 20] if channel_id <= 3 else [25, 15, 20, 12, 18]
                product = random.choices(PRODUCTS, weights=weights)[0]
                day = random_day(month)
                refunded = random.random() < refund_p
                orders.append(
                    (
                        order_id,
                        random.randint(10000, 99999),
                        channel_id,
                        product[0],
                        day.isoformat(),
                        product[3],
                        "refunded" if refunded else "paid",
                        (day + timedelta(days=random.randint(1, 20))).isoformat()
                        if refunded
                        else None,
                    )
                )
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?)", orders)

    conn.executemany(
        "INSERT INTO ad_spend VALUES (?,?,?)",
        [(m, cid, spends[i]) for cid, spends in AD_SPEND.items() for i, m in enumerate(MONTHS)],
    )
    conn.commit()
    print(f"已生成 {DB_PATH.name}：{len(orders)} 条订单，{len(MONTHS)} 个月，{len(CHANNELS)} 个渠道")
    conn.close()


if __name__ == "__main__":
    main()
