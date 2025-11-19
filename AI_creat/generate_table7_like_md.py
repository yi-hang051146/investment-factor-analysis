import pandas as pd
from pathlib import Path

BASE = Path('d:/python_workspace/projects/Pricing-day3')
OUTPUT_DIR = BASE / 'output'
RESULTS_FILE = OUTPUT_DIR / 'anomaly_pricing_results.csv'
TABLE7_MD_FILE = OUTPUT_DIR / 'table7_like_pricing_power.md'

# 需要展示的异象顺序，可以根据需要调整或扩展
ANOMALY_ORDER = [
    'value',            # Book-to-Market
    'profitability',    # Operating Profitability
    'investment_ag',    # Total Assets Growth
    'investment_cash',  # Cash Holdings Growth
    'investment_rece',  # Receivables Growth
    'investment_invt',  # Inventories Growth
    'investment_ppe',   # PPE Growth
    'size',             # Size Anomaly
]

# 将内部的 anomaly 名称转成更易读的标签（可按论文风格调整）
ANOMALY_LABELS = {
    'value': 'Book-to-Market',
    'profitability': 'Operating Profit',
    'investment_ag': 'Total Assets',
    'investment_cash': 'Cash Holdings',
    'investment_rece': 'Receivables',
    'investment_invt': 'Inventories',
    'investment_ppe': 'PPE',
    'size': 'Size',
}


def extract_row(res: pd.DataFrame, anomaly: str) -> dict | None:
    """从回归结果中提取某个异象在 AG_1f / PPE_1f 下的 alpha / t / beta / t。"""
    sub = res[res['anomaly'] == anomaly].set_index('spec')
    if not {'AG_1f', 'PPE_1f'}.issubset(sub.index):
        # 如果缺少某一规格，跳过该异象
        return None

    ag = sub.loc['AG_1f']
    ppe = sub.loc['PPE_1f']

    return {
        'Anomaly': ANOMALY_LABELS.get(anomaly, anomaly),
        # AG 单因子
        'AG_alpha': ag.get('alpha', float('nan')),
        'AG_t_alpha': ag.get('t_alpha', float('nan')),
        'AG_beta': ag.get('beta_CMA_AG', float('nan')),
        'AG_t_beta': ag.get('t_CMA_AG', float('nan')),
        # PPE 单因子
        'PPE_alpha': ppe.get('alpha', float('nan')),
        'PPE_t_alpha': ppe.get('t_alpha', float('nan')),
        'PPE_beta': ppe.get('beta_CMA_PPEG', float('nan')),
        'PPE_t_beta': ppe.get('t_CMA_PPEG', float('nan')),
    }


def to_markdown_table(df: pd.DataFrame) -> str:
    """将 DataFrame 转为 Markdown 表格字符串，保留 2 位小数。"""
    df_fmt = df.copy()
    for col in df_fmt.columns:
        if col != 'Anomaly':
            df_fmt[col] = df_fmt[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    # 手动构造 markdown 表头，类似论文 Table 7 的结构
    header = [
        "| Anomaly | AG-α | AG-t(α) | AG-β | AG-t(β) | PPE-α | PPE-t(α) | PPE-β | PPE-t(β) |",
        "|---------|------|---------|------|---------|-------|----------|-------|----------|",
    ]
    rows = []
    for _, r in df_fmt.iterrows():
        rows.append(
            f"| {r['Anomaly']} | {r['AG_alpha']} | {r['AG_t_alpha']} | {r['AG_beta']} | {r['AG_t_beta']} | "
            f"{r['PPE_alpha']} | {r['PPE_t_alpha']} | {r['PPE_beta']} | {r['PPE_t_beta']} |"
        )
    return "\n".join(header + rows) + "\n"


def main():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"找不到回归结果文件: {RESULTS_FILE}")

    res = pd.read_csv(RESULTS_FILE)
    if 'anomaly' not in res.columns or 'spec' not in res.columns:
        raise ValueError("anomaly_pricing_results.csv 格式不正确，缺少 'anomaly' 或 'spec' 列。")

    rows: list[dict] = []
    for a in ANOMALY_ORDER:
        row = extract_row(res, a)
        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError("在回归结果中未找到任何 AG_1f / PPE_1f 的匹配记录，无法生成表格。")

    df_table = pd.DataFrame(rows)
    md_text = to_markdown_table(df_table)

    TABLE7_MD_FILE.write_text(md_text, encoding='utf-8')
    print(f"已生成 Table 7 风格的 Markdown 表格: {TABLE7_MD_FILE}")


if __name__ == '__main__':
    main()

