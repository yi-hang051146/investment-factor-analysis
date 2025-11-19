import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from typing import Dict, List

print("Script started (size-neutral anomalies)")

BASE = Path('d:/python_workspace/projects/Pricing-day3')
DATA_PATH = BASE / 'data' / 'cleaned_data'
OUTPUT_PATH = BASE / 'output'
OUTPUT_PATH.mkdir(exist_ok=True)

PANEL_FILE = DATA_PATH / 'master_panel_filtered.csv'
FACTORS_FILE = OUTPUT_PATH / 'factors.csv'

ANOMALY_PORTFOLIO_RETURNS_FILE = OUTPUT_PATH / 'anomaly_portfolio_returns_size_neutral.csv'
ANOMALY_LONG_SHORT_RETURNS_FILE = OUTPUT_PATH / 'anomaly_long_short_returns_size_neutral.csv'
ANOMALY_PRICING_RESULTS_FILE = OUTPUT_PATH / 'anomaly_pricing_results_size_neutral.csv'
# 新增：输出规模中性异象定价能力的 Markdown 表
ANOMALY_PRICING_MD_FILE = OUTPUT_PATH / 'table_anomaly_pricing_size_neutral.md'

START_PERIOD = pd.Period('2008-07', freq='M')
END_PERIOD = pd.Period('2023-06', freq='M')

DECILES = 10
DECILE_LABELS = [f'P{i}' for i in range(1, DECILES + 1)]

ANOMALIES: Dict[str, str] = {
    'size': 'market_cap',
    'value': 'book_to_market',
    'profitability': 'op_profitability',
    'investment_ag': 'asset_growth',
    'investment_cash': 'cash_growth',
    'investment_rece': 'receivables_growth',
    'investment_invt': 'inventory_growth',
    'investment_ppe': 'ppe_growth',
}

REG_SPECS: Dict[str, List[str]] = {
    'AG_1f': ['CMA_AG'],
    'PPE_1f': ['CMA_PPEG'],
    'FF5_AG': ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA_AG'],
    'FF5_PPE': ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA_PPEG'],
}


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL_FILE)
    required_cols = {
        'stock_code', 'year_month', 'monthly_return', 'market_cap',
        'book_to_market', 'op_profitability', 'asset_growth', 'cash_growth',
        'receivables_growth', 'inventory_growth', 'ppe_growth', 'rf_monthly',
    }
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f'面板缺少必要列: {missing}')

    df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
    df['year_month'] = pd.PeriodIndex(df['year_month'].astype(str), freq='M')

    numeric_cols = list(required_cols - {'stock_code', 'year_month'})
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.sort_values(['stock_code', 'year_month'])
    df['ME_lag'] = df.groupby('stock_code')['market_cap'].shift(1)
    df['ME_lag'] = df['ME_lag'].fillna(df['market_cap'])

    df = df[(df['year_month'] >= START_PERIOD) & (df['year_month'] <= END_PERIOD)]
    return df


def load_factors() -> pd.DataFrame:
    df = pd.read_csv(FACTORS_FILE)
    if 'year_month' not in df.columns:
        raise KeyError('factors.csv 缺少 year_month 列，请确认文件格式。')
    df['year_month'] = pd.PeriodIndex(df['year_month'].astype(str), freq='M')

    needed = set().union(*REG_SPECS.values())
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f'factors.csv 缺少必要因子列: {missing}')

    df = df.set_index('year_month')[sorted(needed)]
    df = df.loc[(df.index >= START_PERIOD) & (df.index <= END_PERIOD)]
    return df


def _robust_deciles(series: pd.Series, q: int) -> pd.Series:
    s = series.dropna()
    if len(s) < q or s.nunique() < q:
        return None
    labels = list(range(1, q + 1))
    try:
        cats = pd.qcut(s, q=q, labels=labels, duplicates='drop')
        if len(set(int(x) for x in cats)) < q:
            raise ValueError('qcut bins insufficient')
        return cats.astype(int)
    except Exception:
        try:
            ranks = s.rank(method='first')
            bins = np.linspace(0, len(ranks), q + 1)
            cut = pd.cut(ranks, bins=bins, labels=labels, include_lowest=True)
            return cut.astype(int)
        except Exception:
            return None


def build_size_x_anomaly_sorts(panel: pd.DataFrame, anomaly_col: str, size_col: str = 'market_cap') -> pd.DataFrame:
    """在每年 7 月，先按市值分 small/big，再在每个 size 组内部按异象做 10 分位分组。
    返回 formation 截面：stock_code, formation_year, size_group(S/B), decile(1..10)
    """
    df = panel.copy()
    year = df['year_month'].dt.year
    month = df['year_month'].dt.month
    form_mask = (month == 7)
    form_df = df[form_mask].copy()
    form_df['formation_year'] = year[form_mask]

    records = []
    for y, g in form_df.groupby('formation_year'):
        g = g.dropna(subset=[size_col, anomaly_col])
        if g.empty:
            continue
        # 1) 按市值分两组：Small / Big
        try:
            size_label = pd.qcut(g[size_col], q=2, labels=['S', 'B'])
        except Exception:
            # 如果 qcut 失败，改用中位数
            median_size = g[size_col].median()
            size_label = np.where(g[size_col] <= median_size, 'S', 'B')
        g = g.assign(size_group=size_label)

        for size_grp, sub in g.groupby('size_group'):
            dec = _robust_deciles(sub[anomaly_col], DECILES)
            if dec is None:
                print(f"[WARN] {anomaly_col} {y} 年 size={size_grp} 无法形成 {DECILES} 组，跳过该 size。")
                continue
            tmp = sub[['stock_code']].copy()
            tmp['formation_year'] = y
            tmp['size_group'] = size_grp
            tmp['decile'] = dec
            records.append(tmp)

    if not records:
        return pd.DataFrame(columns=['stock_code', 'formation_year', 'size_group', 'decile'])

    return pd.concat(records, ignore_index=True)


def expand_to_holding_size_neutral(panel: pd.DataFrame, form_sorts: pd.DataFrame) -> pd.DataFrame:
    """将 size×anomaly formation 标签扩展到持有期：formation 年 7 月 ~ 次年 6 月。"""
    df = panel.copy()
    df['year'] = df['year_month'].dt.year
    df['month'] = df['year_month'].dt.month

    df['size_group'] = np.nan
    df['decile'] = np.nan

    if form_sorts.empty:
        return df

    for y, g in form_sorts.groupby('formation_year'):
        start = pd.Period(f'{y}-07', freq='M')
        end = pd.Period(f'{y+1}-06', freq='M')
        hold_mask = (df['year_month'] >= start) & (df['year_month'] <= end)
        codes = g['stock_code'].unique()
        sub_mask = hold_mask & df['stock_code'].isin(codes)

        # code -> (size_group, decile)
        size_map = dict(zip(g['stock_code'], g['size_group']))
        dec_map = dict(zip(g['stock_code'], g['decile']))

        df.loc[sub_mask, 'size_group'] = df.loc[sub_mask, 'stock_code'].map(size_map)
        df.loc[sub_mask, 'decile'] = df.loc[sub_mask, 'stock_code'].map(dec_map)

    return df


def compute_decile_returns_by_size(panel_with_tags: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """计算 size=S/B 内部各 decile 的市值加权收益，返回字典 { 'S': df, 'B': df }。
    每个 df 为 year_month × P1..P10。
    """
    df = panel_with_tags.copy()
    df = df.dropna(subset=['size_group', 'decile', 'monthly_return', 'ME_lag'])
    if df.empty:
        idx = pd.period_range(START_PERIOD, END_PERIOD, freq='M')
        empty = pd.DataFrame(index=idx, columns=DECILE_LABELS, dtype=float)
        empty.index.name = 'year_month'
        return {'S': empty.copy(), 'B': empty.copy()}

    df['decile'] = df['decile'].astype(int)
    df['weight'] = df['ME_lag']

    def vw_ret(g: pd.DataFrame) -> float:
        w = g['weight'].sum()
        if w <= 0:
            return np.nan
        return float(np.average(g['monthly_return'], weights=g['weight']))

    results: Dict[str, pd.DataFrame] = {}
    for size_grp, sub in df.groupby('size_group'):
        ret = sub.groupby(['year_month', 'decile'])['monthly_return'].apply(
            lambda s: vw_ret(sub.loc[s.index])
        )
        ret = ret.reset_index(name='vw_return')
        pivot = ret.pivot(index='year_month', columns='decile', values='vw_return')
        pivot.index = pd.PeriodIndex(pivot.index, freq='M')
        pivot.index.name = 'year_month'
        cols = {}
        for d in range(1, DECILES + 1):
            label = f'P{d}'
            if d in pivot.columns:
                cols[d] = label
            else:
                pivot[label] = np.nan
        pivot = pivot.rename(columns=cols)
        pivot = pivot[[f'P{i}' for i in range(1, DECILES + 1)]].sort_index()
        pivot = pivot[(pivot.index >= START_PERIOD) & (pivot.index <= END_PERIOD)]
        results[str(size_grp)] = pivot

    # 若某个 size 组不存在，补空表
    idx = pd.period_range(START_PERIOD, END_PERIOD, freq='M')
    for g in ['S', 'B']:
        if g not in results:
            empty = pd.DataFrame(index=idx, columns=DECILE_LABELS, dtype=float)
            empty.index.name = 'year_month'
            results[g] = empty

    return results


def run_regression(long_short: pd.Series, factors: pd.DataFrame, factor_list: List[str]) -> Dict[str, float]:
    if not isinstance(long_short.index, pd.PeriodIndex):
        long_short.index = pd.PeriodIndex(long_short.index.astype(str), freq='M')

    merged = pd.concat([
        long_short.rename('long_short'),
        factors[factor_list]
    ], axis=1).dropna()

    if merged.empty or len(merged) < 12:
        return {}

    y = merged['long_short'] * 100
    X = merged[factor_list] * 100
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})

    out: Dict[str, float] = {
        'alpha': float(model.params['const']),
        't_alpha': float(model.tvalues['const']),
        'r_squared': float(model.rsquared),
    }
    for fac in factor_list:
        out[f'beta_{fac}'] = float(model.params.get(fac, np.nan))
        out[f't_{fac}'] = float(model.tvalues.get(fac, np.nan))
    out['n_obs'] = int(model.nobs)
    return out


def run_anomaly_analysis_size_neutral():
    panel = load_panel()
    factors = load_factors()

    portfolio_records = []
    long_short_records = []
    regression_records = []

    for anomaly_name, col in ANOMALIES.items():
        print(f">>> [Size-neutral] 处理中异象: {anomaly_name} ({col})")
        form_sorts = build_size_x_anomaly_sorts(panel, col)
        if form_sorts.empty:
            print(f"[WARN] {anomaly_name} 无 formation 截面，跳过。")
            continue

        panel_tags = expand_to_holding_size_neutral(panel, form_sorts)
        size_deciles = compute_decile_returns_by_size(panel_tags)

        # 合成规模中性 decile 收益：对 S/B 同权平均
        dec_S = size_deciles['S']
        dec_B = size_deciles['B']
        dec_neutral = 0.5 * (dec_S + dec_B)

        # 记录 decile 收益（长表）
        tidy = dec_neutral.reset_index().melt(
            id_vars='year_month', value_vars=[f'P{i}' for i in range(1, DECILES + 1)],
            var_name='decile', value_name='vw_return'
        )
        tidy['year_month'] = tidy['year_month'].astype(str)
        tidy['anomaly'] = anomaly_name
        tidy['size_neutral'] = True
        portfolio_records.append(tidy)

        # 多空组合 P10 - P1（规模中性版）
        long_short = dec_neutral['P10'] - dec_neutral['P1']
        ls_df = long_short.to_frame(name='long_short')
        ls_df['anomaly'] = anomaly_name
        ls_df['size_neutral'] = True
        long_short_records.append(ls_df)

        # 回归
        for spec_name, fac_list in REG_SPECS.items():
            metrics = run_regression(long_short, factors, fac_list)
            if not metrics:
                metrics = {'alpha': np.nan, 't_alpha': np.nan, 'r_squared': np.nan, 'n_obs': 0}
                for f in fac_list:
                    metrics[f'beta_{f}'] = np.nan
                    metrics[f't_{f}'] = np.nan
            record = {'anomaly': anomaly_name, 'spec': spec_name, 'size_neutral': True}
            record.update(metrics)
            regression_records.append(record)

    if portfolio_records:
        portfolio_df = pd.concat(portfolio_records, ignore_index=True)
        portfolio_df.to_csv(ANOMALY_PORTFOLIO_RETURNS_FILE, index=False)

    if long_short_records:
        ls_df_all = pd.concat(long_short_records).reset_index().rename(columns={'index': 'year_month'})
        ls_df_all['year_month'] = ls_df_all['year_month'].astype(str)
        ls_df_all.to_csv(ANOMALY_LONG_SHORT_RETURNS_FILE, index=False)

    reg_df = None
    if regression_records:
        reg_df = pd.DataFrame(regression_records)
        reg_df = reg_df.set_index(['anomaly', 'spec', 'size_neutral']).sort_index()
        reg_df.to_csv(ANOMALY_PRICING_RESULTS_FILE)
        print("\n=== [Size-neutral] 回归结果预览 ===")
        print(reg_df[['alpha', 't_alpha', 'r_squared', 'n_obs']])

    # 新增：生成类似 Table 7 的横向 Markdown 表
    if reg_df is not None:
        # 只保留规模中性的结果
        reg_sn = reg_df.xs(True, level='size_neutral')

        # ---------------- 新增：两种表输出 ----------------
        # 1) 内部变量名版本（已有）：table_anomaly_pricing_size_neutral.md
        rows_raw = []
        for anomaly in ANOMALIES.keys():
            if anomaly not in reg_sn.index.get_level_values('anomaly'):
                continue
            sub = reg_sn.loc[anomaly]
            if 'AG_1f' not in sub.index or 'PPE_1f' not in sub.index:
                continue
            ag = sub.loc['AG_1f']
            ppe = sub.loc['PPE_1f']
            rows_raw.append({
                'Anomaly': anomaly,
                'AG-α': float(ag['alpha']),
                'AG-t(α)': float(ag['t_alpha']),
                'AG-β': float(ag.get('beta_CMA_AG', np.nan)),
                'AG-t(β)': float(ag.get('t_CMA_AG', np.nan)),
                'PPE-α': float(ppe['alpha']),
                'PPE-t(α)': float(ppe['t_alpha']),
                'PPE-β': float(ppe.get('beta_CMA_PPEG', np.nan)),
                'PPE-t(β)': float(ppe.get('t_CMA_PPEG', np.nan)),
            })
        if rows_raw:
            table_df_raw = pd.DataFrame(rows_raw)
            for col in table_df_raw.columns:
                if col != 'Anomaly':
                    table_df_raw[col] = table_df_raw[col].astype(float).round(2)

            lines_raw: List[str] = []
            lines_raw.append('# Size-neutral Anomaly Pricing Power (AG vs PPE)')
            lines_raw.append('本表基于规模中性的多空组合（Small/Big 取均值），比较 AG 与 PPE 一因子模型对各异象的定价能力。')
            lines_raw.append('')
            lines_raw.append('| Anomaly | AG-α | AG-t(α) | AG-β | AG-t(β) | PPE-α | PPE-t(α) | PPE-β | PPE-t(β) |')
            lines_raw.append('|---------|------|---------|------|---------|-------|----------|-------|----------|')
            for _, r in table_df_raw.iterrows():
                lines_raw.append(
                    f"| {r['Anomaly']} | {r['AG-α']:.2f} | {r['AG-t(α)']:.2f} | {r['AG-β']:.2f} | {r['AG-t(β)']:.2f} | "
                    f"{r['PPE-α']:.2f} | {r['PPE-t(α)']:.2f} | {r['PPE-β']:.2f} | {r['PPE-t(β)']:.2f} |"
                )
            ANOMALY_PRICING_MD_FILE.write_text('\n'.join(lines_raw), encoding='utf-8')
            print(f"\n[INFO] 已生成规模中性异象定价 Markdown 表: {ANOMALY_PRICING_MD_FILE}")

        # 2) Table7-like 版本：行名和顺序对齐论文
        DISPLAY_ORDER = [
            'value',            # Book-to-Market
            'profitability',    # Operating Profit
            'investment_ag',    # Total Assets
            'investment_cash',  # Cash Holdings
            'investment_rece',  # Receivables
            'investment_invt',  # Inventories
            'investment_ppe',   # PPE
            'size',             # Size
        ]
        DISPLAY_LABELS = {
            'value': 'Book-to-Market',
            'profitability': 'Operating Profit',
            'investment_ag': 'Total Assets',
            'investment_cash': 'Cash Holdings',
            'investment_rece': 'Receivables',
            'investment_invt': 'Inventories',
            'investment_ppe': 'PPE',
            'size': 'Size',
        }

        rows_t7 = []
        for anomaly in DISPLAY_ORDER:
            if anomaly not in reg_sn.index.get_level_values('anomaly'):
                continue
            sub = reg_sn.loc[anomaly]
            if isinstance(sub, pd.Series):
                # 只有一个 spec，跳过
                continue
            if 'AG_1f' not in sub.index or 'PPE_1f' not in sub.index:
                continue
            ag = sub.loc['AG_1f']
            ppe = sub.loc['PPE_1f']
            rows_t7.append({
                'Anomaly': DISPLAY_LABELS.get(anomaly, anomaly),
                'AG-α': float(ag['alpha']),
                'AG-t(α)': float(ag['t_alpha']),
                'AG-β': float(ag.get('beta_CMA_AG', np.nan)),
                'AG-t(β)': float(ag.get('t_CMA_AG', np.nan)),
                'PPE-α': float(ppe['alpha']),
                'PPE-t(α)': float(ppe['t_alpha']),
                'PPE-β': float(ppe.get('beta_CMA_PPEG', np.nan)),
                'PPE-t(β)': float(ppe.get('t_CMA_PPEG', np.nan)),
            })

        if rows_t7:
            table_df_t7 = pd.DataFrame(rows_t7)
            for col in table_df_t7.columns:
                if col != 'Anomaly':
                    table_df_t7[col] = table_df_t7[col].astype(float).round(2)

            lines_t7: List[str] = []
            lines_t7.append('# Size-neutral Anomaly Pricing Power (AG vs PPE) — Table7-like')
            lines_t7.append('本表在规模中性基础上，对齐论文 Table 7 的异象名称与行顺序。')
            lines_t7.append('')
            lines_t7.append('| Anomaly | AG-α | AG-t(α) | AG-β | AG-t(β) | PPE-α | PPE-t(α) | PPE-β | PPE-t(β) |')
            lines_t7.append('|------------------|------|---------|------|---------|-------|----------|-------|----------|')
            for _, r in table_df_t7.iterrows():
                lines_t7.append(
                    f"| {r['Anomaly']} | {r['AG-α']:.2f} | {r['AG-t(α)']:.2f} | {r['AG-β']:.2f} | {r['AG-t(β)']:.2f} | "
                    f"{r['PPE-α']:.2f} | {r['PPE-t(α)']:.2f} | {r['PPE-β']:.2f} | {r['PPE-t(β)']:.2f} |"
                )
            # 输出到一个新的 md 文件
            t7_file = OUTPUT_PATH / 'table7_like_pricing_power_size_neutral.md'
            t7_file.write_text('\n'.join(lines_t7), encoding='utf-8')
            print(f"[INFO] 已生成 Table7-like（规模中性）Markdown 表: {t7_file}")

    print("\n=== 规模中性异象定价分析完成 ===")


def main():
    try:
        run_anomaly_analysis_size_neutral()
    except Exception as e:
        print(f"[ERROR] 脚本运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
