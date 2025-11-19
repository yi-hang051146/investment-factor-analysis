import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from typing import Dict, List

print("Script started")

# ============================================
# 异象定价分析脚本（按论文逻辑重构版）
# ============================================
# 输入：
#   1) data/cleaned_data/master_panel_filtered.csv
#   2) output/factors.csv （需包含 Mkt-RF, SMB, HML, RMW, CMA_AG, CMA_PPEG）
# 逻辑：
#   - 每年 7 月用上一年的特征做 10 分位分组
#   - 形成组从当年 7 月持有到次年 6 月（12 个月）
#   - 计算每个十分位组合的月度市值加权收益
#   - 构造多空组合：P10 - P1
#   - 对每个异象的多空收益，用四种模型回归：
#       * AG_1f     : long_short 对 CMA_AG
#       * PPE_1f    : long_short 对 CMA_PPEG
#       * FF5_AG    : Mkt-RF, SMB, HML, RMW, CMA_AG
#       * FF5_PPE   : Mkt-RF, SMB, HML, RMW, CMA_PPEG
#   - 使用 Newey-West HAC(4) 标准误
# 输出：
#   - anomaly_portfolio_returns.csv：year_month, anomaly, decile(P1..P10), vw_return
#   - anomaly_long_short_returns.csv：year_month, anomaly, long_short
#   - anomaly_pricing_results.csv：横向比较表（每行一个 anomaly+spec）
# ============================================

BASE = Path('d:/python_workspace/projects/Pricing-day3')
DATA_PATH = BASE / 'data' / 'cleaned_data'
OUTPUT_PATH = BASE / 'output'
OUTPUT_PATH.mkdir(exist_ok=True)

PANEL_FILE = DATA_PATH / 'master_panel_filtered.csv'
FACTORS_FILE = OUTPUT_PATH / 'factors.csv'

ANOMALY_PORTFOLIO_RETURNS_FILE = OUTPUT_PATH / 'anomaly_portfolio_returns.csv'
ANOMALY_LONG_SHORT_RETURNS_FILE = OUTPUT_PATH / 'anomaly_long_short_returns.csv'
ANOMALY_PRICING_RESULTS_FILE = OUTPUT_PATH / 'anomaly_pricing_results.csv'

START_PERIOD = pd.Period('2008-07', freq='M')
END_PERIOD = pd.Period('2023-06', freq='M')

# 采用 10 分组
DECILES = 10
DECILE_LABELS = [f'P{i}' for i in range(1, DECILES + 1)]

# 异象变量对应的列名（使用 formation 月的特征值）
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

# 回归规格：名称 -> 使用的因子列表
REG_SPECS: Dict[str, List[str]] = {
    'AG_1f': ['CMA_AG'],
    'PPE_1f': ['CMA_PPEG'],
    'FF5_AG': ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA_AG'],
    'FF5_PPE': ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA_PPEG'],
}

# --------------------------------------------
# 数据加载
# --------------------------------------------

def load_panel() -> pd.DataFrame:
    """加载清洗后面板，保证必要列存在，并添加上一月市值 ME_lag。"""
    print("Loading panel...")
    required_cols = {
        'stock_code', 'year_month', 'monthly_return', 'market_cap',
        'book_to_market', 'op_profitability', 'asset_growth', 'cash_growth',
        'receivables_growth', 'inventory_growth', 'ppe_growth', 'rf_monthly',
    }
    df = pd.read_csv(PANEL_FILE)
    print(f"Panel loaded with shape: {df.shape}")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f'面板缺少必要列: {missing}')

    df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
    df['year_month'] = pd.PeriodIndex(df['year_month'].astype(str), freq='M')

    numeric_cols = list(required_cols - {'stock_code', 'year_month'})
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.sort_values(['stock_code', 'year_month'])
    # 上一月市值用于持有期权重；首月缺失可用当月市值补
    df['ME_lag'] = df.groupby('stock_code')['market_cap'].shift(1)
    df['ME_lag'] = df['ME_lag'].fillna(df['market_cap'])

    df = df[(df['year_month'] >= START_PERIOD) & (df['year_month'] <= END_PERIOD)]
    print(f"Panel filtered to shape: {df.shape}")
    return df


def load_factors() -> pd.DataFrame:
    """加载因子数据，要求包含 Mkt-RF, SMB, HML, RMW, CMA_AG, CMA_PPEG。"""
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

# --------------------------------------------
# Formation 年 7 月分组 & 持有期标签扩散
# --------------------------------------------

def _robust_deciles(series: pd.Series, q: int) -> pd.Series:
    """稳健的十分位分组：先 qcut，失败则用 rank+cut；唯一值不足时返回 None。"""
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


def build_formation_deciles(panel: pd.DataFrame, anomaly_col: str) -> pd.DataFrame:
    """在每年 7 月按指定异象做 10 分位分组，返回 formation 截面带 decile 标签。"""
    df = panel.copy()
    year = df['year_month'].dt.year
    month = df['year_month'].dt.month
    # formation 月：每年 7 月
    form_mask = (month == 7)
    form_df = df[form_mask].copy()
    form_df['formation_year'] = year[form_mask]

    records = []
    for y, g in form_df.groupby('formation_year'):
        col = anomaly_col
        dec = _robust_deciles(g[col], DECILES)
        if dec is None:
            print(f"[WARN] {anomaly_col} {y} 年 formation 无法形成 {DECILES} 组，跳过该年。")
            continue
        tmp = g[['stock_code', 'formation_year']].copy()
        tmp['decile'] = dec
        records.append(tmp)

    if not records:
        return pd.DataFrame(columns=['stock_code', 'formation_year', 'decile'])

    formation_deciles = pd.concat(records, ignore_index=True)
    return formation_deciles


def expand_to_holding(panel: pd.DataFrame, formation_deciles: pd.DataFrame) -> pd.DataFrame:
    """将 formation decile 标签扩展到持有期：formation 年 7 月 ~ 次年 6 月。"""
    df = panel.copy()
    df['year'] = df['year_month'].dt.year
    df['month'] = df['year_month'].dt.month

    # 初始化 decile 列为空
    df['decile'] = np.nan

    if formation_deciles.empty:
        return df

    # 按 formation_year 循环，将对应股票在持有期内打上组标签
    for y, g in formation_deciles.groupby('formation_year'):
        start = pd.Period(f'{y}-07', freq='M')
        end = pd.Period(f'{y+1}-06', freq='M')
        hold_mask = (df['year_month'] >= start) & (df['year_month'] <= end)

        # 只对在 formation 截面出现过的股票赋值
        codes = g['stock_code'].unique()
        sub_mask = hold_mask & df['stock_code'].isin(codes)

        # 构造 code -> decile 映射
        mapping = dict(zip(g['stock_code'], g['decile']))
        df.loc[sub_mask, 'decile'] = df.loc[sub_mask, 'stock_code'].map(mapping)

    return df

# --------------------------------------------
# 计算组合收益 & 多空序列
# --------------------------------------------

def compute_decile_returns(panel_with_decile: pd.DataFrame) -> pd.DataFrame:
    """给定已打上 decile 的面板，计算 P1..P10 市值加权月度收益。"""
    df = panel_with_decile.copy()
    df = df.dropna(subset=['decile', 'monthly_return', 'ME_lag'])
    if df.empty:
        idx = pd.period_range(START_PERIOD, END_PERIOD, freq='M')
        out = pd.DataFrame(index=idx, columns=DECILE_LABELS, dtype=float)
        out.index.name = 'year_month'
        return out

    df['decile'] = df['decile'].astype(int)
    df['weight'] = df['ME_lag']

    # 市值加权收益
    def vw_ret(g: pd.DataFrame) -> float:
        w = g['weight'].sum()
        if w <= 0:
            return np.nan
        return float(np.average(g['monthly_return'], weights=g['weight']))

    # groupby.apply 返回的是 Series：index=(year_month, decile)，value=vw_return
    ret = df.groupby(['year_month', 'decile'])['monthly_return'].apply(
        lambda s: vw_ret(df.loc[s.index])
    )
    ret = ret.reset_index(name='vw_return')

    # pivot 成 year_month × P1..P10
    pivot = ret.pivot(index='year_month', columns='decile', values='vw_return')
    pivot.index = pd.PeriodIndex(pivot.index, freq='M')
    pivot.index.name = 'year_month'

    # 确保所有 decile 列存在
    cols = {}
    for d in range(1, DECILES + 1):
        label = f'P{d}'
        if d in pivot.columns:
            cols[d] = label
        else:
            pivot[label] = np.nan
    pivot = pivot.rename(columns=cols)
    pivot = pivot[[f'P{i}' for i in range(1, DECILES + 1)]].sort_index()

    # 限制时间窗
    pivot = pivot[(pivot.index >= START_PERIOD) & (pivot.index <= END_PERIOD)]
    return pivot

# --------------------------------------------
# 回归模块
# --------------------------------------------

def run_regression(long_short: pd.Series, factors: pd.DataFrame, factor_list: List[str]) -> Dict[str, float]:
    """对 long_short 超额收益序列进行多因子回归。返回 alpha, t_alpha, R2, 以及各因子 beta/t。"""
    if not isinstance(long_short.index, pd.PeriodIndex):
        long_short.index = pd.PeriodIndex(long_short.index.astype(str), freq='M')

    # 只保留共同样本
    merged = pd.concat([
        long_short.rename('long_short'),
        factors[factor_list]
    ], axis=1).dropna()

    if merged.empty or len(merged) < 12:
        # 至少要求一年数据
        return {}

    # 注意：long_short 已经是组合超额收益，这里不再减 RF
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

# --------------------------------------------
# 主流程
# --------------------------------------------

def run_anomaly_analysis():
    panel = load_panel()
    factors = load_factors()

    print("[DEBUG] panel year_month 范围:", panel['year_month'].min(), '->', panel['year_month'].max())
    print("[DEBUG] factors index 范围:", factors.index.min(), '->', factors.index.max())

    portfolio_records = []
    long_short_records = []
    regression_records = []

    for anomaly_name, col in ANOMALIES.items():
        print(f">>> 处理中异象: {anomaly_name} ({col})")

        # 1. formation 年 7 月分组
        form_dec = build_formation_deciles(panel, col)
        if form_dec.empty:
            print(f"[WARN] {anomaly_name} 无 formation 截面，跳过。")
            continue

        # 2. 扩展到持有期（当年 7 月 ~ 次年 6 月）
        panel_with_dec = expand_to_holding(panel, form_dec)

        # 3. 计算各 decile 月度价值加权收益
        dec_ret = compute_decile_returns(panel_with_dec)

        # 4. 记录 decile 收益（长表）
        tidy = dec_ret.reset_index().melt(
            id_vars='year_month', value_vars=[f'P{i}' for i in range(1, DECILES + 1)],
            var_name='decile', value_name='vw_return'
        )
        tidy['year_month'] = tidy['year_month'].astype(str)
        tidy['anomaly'] = anomaly_name
        portfolio_records.append(tidy)

        # 5. 多空组合 P10 - P1
        long_short = dec_ret['P10'] - dec_ret['P1']
        ls_df = long_short.to_frame(name='long_short')
        ls_df['anomaly'] = anomaly_name
        long_short_records.append(ls_df)

        # 6. 对多空组合做多种模型回归
        for spec_name, fac_list in REG_SPECS.items():
            metrics = run_regression(long_short, factors, fac_list)
            if not metrics:
                # 填 NaN 占位
                metrics = {'alpha': np.nan, 't_alpha': np.nan, 'r_squared': np.nan, 'n_obs': 0}
                for f in fac_list:
                    metrics[f'beta_{f}'] = np.nan
                    metrics[f't_{f}'] = np.nan
            record = {'anomaly': anomaly_name, 'spec': spec_name}
            record.update(metrics)
            regression_records.append(record)

    # 汇总并写出
    if portfolio_records:
        portfolio_df = pd.concat(portfolio_records, ignore_index=True)
        portfolio_df.to_csv(ANOMALY_PORTFOLIO_RETURNS_FILE, index=False)
    else:
        print("[WARN] 无 decile 收益结果，未写 anomaly_portfolio_returns.csv")

    if long_short_records:
        ls_df_all = pd.concat(long_short_records).reset_index().rename(columns={'index': 'year_month'})
        ls_df_all['year_month'] = ls_df_all['year_month'].astype(str)
        ls_df_all.to_csv(ANOMALY_LONG_SHORT_RETURNS_FILE, index=False)
    else:
        print("[WARN] 无多空收益结果，未写 anomaly_long_short_returns.csv")

    if regression_records:
        reg_df = pd.DataFrame(regression_records)
        # 横向比较表：行为 anomaly+spec，列为 alpha/t/R2 以及 beta/t
        reg_df = reg_df.set_index(['anomaly', 'spec']).sort_index()
        reg_df.to_csv(ANOMALY_PRICING_RESULTS_FILE)
        print("\n=== 回归结果（alpha / t_alpha / R^2 / n_obs）预览 ===")
        print(reg_df[['alpha', 't_alpha', 'r_squared', 'n_obs']])
    else:
        print("[WARN] 无回归结果，未写 anomaly_pricing_results.csv")

    print("\n=== 异象定价分析完成 ===")


def main():
    try:
        run_anomaly_analysis()
    except Exception as e:
        print(f"[ERROR] 脚本运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
