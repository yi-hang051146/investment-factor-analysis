import pandas as pd
import numpy as np
from pathlib import Path
from typing import List


START_DATE = '2008-07'
END_DATE = '2023-06'
FORMATION_MONTH = 6  # 投资组合形成月（用于停牌当月剔除）
# 当日停牌时间阈值（小时）：若单次停牌 Timeperd < 此阈值，则视为“短停牌”，不计入停牌天数
SUSP_SHORT_HOURS_THRESHOLD = 2.0

RAW_PATH = Path("d:/python_workspace/projects/Pricing-day3/data/raw_data")
CLEAN_PATH = Path("d:/python_workspace/projects/Pricing-day3/data/cleaned_data")
CLEAN_PATH.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# 整体流程简述（便于回顾 & 二次开发）
# 1. load_raw                读取原始 CSV（市场 / 财务 / 无风险利率 / 指数）
# 2. clean_risk_free_rate    日度 1D 拆借利率 → 月度无风险利率
# 3. clean_market_return     沪深300 指数收益清洗
# 4. clean_market_data       个股月度收益 + 流通市值 + 行业过滤
# 5. clean_financial_data    年报（12 月合并报表）基础财务字段提取
# 6. compute_financial_indicators  构造派生指标（资产/现金/应收/存货/PPE 增长、盈利能力）
# 7. align_annual_to_monthly 年 t 年报 → (t+1)7 月 ~ (t+2)6 月 月度面板对齐
# 8. 合并 rf & 指数收益      添加无风险利率、市场指数收益
# 9. apply_paper_filters     按论文规则生成 tradable 股票并过滤
# 10. 输出各阶段 CSV         供因子构建脚本使用
# ----------------------------------------------------------------------------
# 简化目标：减少重复逻辑 + 提升可读性 + 保持原输出结构
# ----------------------------------------------------------------------------

# =============================
# 通用辅助函数
# =============================

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名：去除首尾空格与 BOM 隐藏字符。"""
    df.columns = [str(c).strip().replace('\ufeff', '') for c in df.columns]
    return df

def _safe_to_numeric(df: pd.DataFrame, cols: List[str]):
    """批量安全数字化转换。"""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

# ------------------ Load Raw Data ------------------

def load_raw():
    """
    加载原始 CSV 数据。
    返回 dict: {'market','financial','rf','idx'} → DataFrame
    """
    files = {
        "market": "TRD_Mnth(Merge Query).csv",
        "financial": "FS_Combas(Merge Query).csv",
        "rf": "CMMPI_Uimd.csv",
        "idx": "IDX_Idxtrdmth.csv",
    }
    out = {}
    for k, v in files.items():
        p = RAW_PATH / v
        for enc in ("utf-8", "gbk"):
            try:
                df = pd.read_csv(p, encoding=enc)
                out[k] = _standardize_columns(df)
                break
            except Exception:
                continue
        if k not in out:
            raise RuntimeError(f"无法读取文件: {p}，请检查路径或编码。")
    return out

# ------------------ Clean Risk Free ------------------

def clean_risk_free_rate(df, start: str = START_DATE, end: str = END_DATE):
    """无风险利率：筛选 1d 、日度→月均→简单折算为月度利率。"""
    rf_df = df[df.get('Uimd01', '') == '1d'].copy()
    rf_df = rf_df.rename(columns={'Datesgn': 'date', 'Uimd07': 'rf_rate'})
    rf_df['date'] = pd.to_datetime(rf_df['date'], errors='coerce')
    rf_df = rf_df[(rf_df['date'] >= pd.to_datetime(start + '-01')) & (rf_df['date'] <= pd.to_datetime(end + '-28'))]
    rf_df['rf_rate'] = pd.to_numeric(rf_df['rf_rate'], errors='coerce') / 100.0
    rf_df = rf_df.dropna(subset=['rf_rate'])
    monthly = rf_df.set_index('date')['rf_rate'].resample('M').mean().reset_index()
    monthly['rf_monthly'] = monthly['rf_rate'] / 12.0
    monthly['year_month'] = monthly['date'].dt.to_period('M').astype(str)
    monthly = monthly[(monthly['year_month'] >= start) & (monthly['year_month'] <= end)]
    return monthly[['year_month', 'rf_monthly']]

# ------------------ Clean Market Return ------------------

def clean_market_return(df, start: str = START_DATE, end: str = END_DATE):
    """沪深300指数月度收益；清理基础列 → 过滤时间窗口。"""
    for col in ['Indexcd', 'Month', 'Idxrtn']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.replace('\ufeff', '')
    mkt_df = df.copy()
    mkt_df = mkt_df.rename(columns={'Month': 'year_month', 'Idxrtn': 'market_return'})
    mkt_df['year_month'] = mkt_df['year_month'].astype(str)
    mkt_df['market_return'] = pd.to_numeric(mkt_df['market_return'], errors='coerce')
    mkt_df = mkt_df[(mkt_df['year_month'] >= start) & (mkt_df['year_month'] <= end)]
    return mkt_df[['year_month', 'market_return']]

# ------------------ Column Finder ------------------

def _find_col(cols: List[str], desired: str, alt: List[str] = None):
    """兼容带前缀字段名的列查找；优先精确 → 后缀匹配 → 备用别名。"""
    if desired in cols:
        return desired
    for c in cols:
        if c.endswith('.' + desired):
            return c
    if alt:
        for a in alt:
            if a in cols:
                return a
        for a in alt:
            for c in cols:
                if c.endswith('.' + a):
                    return c
    return None

# ------------------ Clean Market ------------------

def clean_market_data(df, start: str = START_DATE, end: str = END_DATE):
    """市场月度数据清洗：代码/时间标准化 + 必需列映射 + 行业剔除 + 单位转换。"""
    df = df.copy()
    cols = df.columns.tolist()
    c_stk = _find_col(cols, 'Stkcd')
    c_mnt = _find_col(cols, 'Trdmnt')
    c_ret = _find_col(cols, 'Mretwd')
    c_mcap = _find_col(cols, 'Msmvosd')
    c_ind = _find_col(cols, 'Nnindcd')
    c_listdt = _find_col(cols, 'Listdt')
    missing = [n for n, v in [('Stkcd', c_stk), ('Trdmnt', c_mnt), ('Mretwd', c_ret)] if v is None]
    if missing:
        raise ValueError(f"市场数据缺少必要列: {missing}")
    rename_map = {c_stk: 'stock_code', c_mnt: 'year_month', c_ret: 'monthly_return'}
    if c_mcap: rename_map[c_mcap] = 'market_cap'
    if c_ind: rename_map[c_ind] = 'industry_code'
    if c_listdt: rename_map[c_listdt] = 'list_date'
    df = df.rename(columns=rename_map)
    df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
    df['year_month'] = pd.to_datetime(df['year_month'], errors='coerce').dt.to_period('M').astype(str)
    if 'list_date' in df.columns:
        df['list_date'] = pd.to_datetime(df['list_date'], errors='coerce')
    df['monthly_return'] = pd.to_numeric(df['monthly_return'], errors='coerce')
    if 'market_cap' in df.columns:
        df['market_cap'] = pd.to_numeric(df['market_cap'], errors='coerce') * 1000
    else:
        df['market_cap'] = np.nan
    df = df[(df['year_month'] >= start) & (df['year_month'] <= end)]
    df = df.dropna(subset=['monthly_return'])
    if 'industry_code' in df.columns:
        df = df[~df['industry_code'].astype(str).str.startswith('J')]  # 剔除金融业
    keep = ['stock_code', 'year_month', 'monthly_return', 'market_cap']
    if 'industry_code' in df.columns: keep.append('industry_code')
    if 'list_date' in df.columns: keep.append('list_date')
    return df[keep]

# ------------------ Clean Financial ------------------

def clean_financial_data(df, market_cleaned):
    """年报财务字段提取 + 仅合并报表 + 年末月份 + 与市场样本交叉。"""
    df = df.copy()
    cols = df.columns.tolist()
    fp = lambda d, al=None: _find_col(cols, d, al)
    m = {
        'Stkcd': fp('Stkcd'), 'Accper': fp('Accper'), 'Typrep': fp('Typrep'),
        'A001000000': fp('A001000000'), 'A003000000': fp('A003000000'), 'A001212000': fp('A001212000'),
        'B001100000': fp('B001100000'), 'B001200000': fp('B001200000'),
        'A001101000': fp('A001101000'), 'A001111000': fp('A001111000'), 'A001123000': fp('A001123000'),
        'A001103000': fp('A001103000'), 'A001104000': fp('A001104000'), 'A001121000': fp('A001121000'),
        'ShortName': fp('ShortName')
    }
    required = ['Stkcd', 'Accper', 'Typrep', 'A001000000', 'A003000000', 'A001212000', 'B001100000', 'B001200000']
    miss = [k for k in required if m.get(k) is None]
    if miss:
        raise ValueError(f"财务文件缺少必要列: {miss}")
    df = df.rename(columns={
        m['Stkcd']: 'stock_code', m['Accper']: 'acc_period', m['Typrep']: 'report_type',
        m['A001000000']: 'total_assets', m['A003000000']: 'book_value', m['A001212000']: 'ppe',
        m['B001100000']: 'total_revenue', m['B001200000']: 'total_op_cost',
        **({m['A001101000']: 'cash'} if m.get('A001101000') else {}),
        **({m['A001111000']: 'receivables'} if m.get('A001111000') else {}),
        **({m['A001123000']: 'inventory'} if m.get('A001123000') else {}),
        **({m['A001104000']: 'accounts_receivable'} if m.get('A001104000') else {}),
        **({m['A001103000']: 'notes_receivable'} if m.get('A001103000') else {}),
        **({m['A001121000']: 'inventory_raw'} if m.get('A001121000') else {}),
        **({m['ShortName']: 'short_name'} if m.get('ShortName') else {})
    })
    df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
    df['acc_period'] = pd.to_datetime(df['acc_period'], errors='coerce')
    df = df[(df['report_type'] == 'A') & (df['acc_period'].dt.month == 12)]
    valid = set(market_cleaned['stock_code'].unique())
    df = df[df['stock_code'].isin(valid)]
    num_cols = ['total_assets', 'book_value', 'ppe', 'total_revenue', 'total_op_cost', 'cash', 'receivables', 'inventory', 'accounts_receivable', 'notes_receivable', 'inventory_raw']
    _safe_to_numeric(df, num_cols)
    df['year'] = df['acc_period'].dt.year
    df = df.sort_values(['stock_code', 'year'])
    # 组装应收 + 存货净额兼容
    if 'receivables' not in df.columns:
        if 'accounts_receivable' in df.columns and 'notes_receivable' in df.columns:
            df['receivables'] = df['accounts_receivable'] + df['notes_receivable']
        elif 'accounts_receivable' in df.columns:
            df['receivables'] = df['accounts_receivable']
        elif 'notes_receivable' in df.columns:
            df['receivables'] = df['notes_receivable']
    if 'inventory' not in df.columns and 'inventory_raw' in df.columns:
        df['inventory'] = df['inventory_raw']
    keep = ['stock_code', 'year', 'total_assets', 'book_value', 'ppe', 'total_revenue', 'total_op_cost']
    for extra in ['cash', 'receivables', 'inventory', 'short_name']:
        if extra in df.columns: keep.append(extra)
    return df[keep]

# ------------------ Compute Annual Indicators ------------------

def compute_financial_indicators(fin_df: pd.DataFrame) -> pd.DataFrame:
    """构造派生指标：盈利能力 + 各类增长率。"""
    fin_df = fin_df.copy()
    fin_df['operating_profit'] = fin_df['total_revenue'] - fin_df['total_op_cost']
    fin_df['op_profitability'] = np.where(fin_df['book_value'] > 0, fin_df['operating_profit'] / fin_df['book_value'], np.nan)
    fin_df['asset_growth'] = fin_df.groupby('stock_code')['total_assets'].pct_change()
    if 'cash' in fin_df.columns:
        fin_df['cash_growth'] = fin_df.groupby('stock_code')['cash'].pct_change()
    if 'receivables' in fin_df.columns:
        fin_df['receivables_growth'] = fin_df.groupby('stock_code')['receivables'].pct_change()
    if 'inventory' in fin_df.columns:
        fin_df['inventory_growth'] = fin_df.groupby('stock_code')['inventory'].pct_change()
    fin_df['ppe_growth'] = fin_df.groupby('stock_code')['ppe'].pct_change()
    return fin_df

# ------------------ Lag Join Annual to Monthly ------------------

def align_annual_to_monthly(fin_indicators: pd.DataFrame, monthly_market: pd.DataFrame) -> pd.DataFrame:
    """年 t 年报适用于 (t+1)7 月 ~ (t+2)6 月；按同月市值计算 B/M。"""
    market = monthly_market.copy()
    market['year_month_dt'] = pd.to_datetime(market['year_month'] + '-01')
    fin = fin_indicators.copy()
    fin['effective_start'] = pd.to_datetime((fin['year'] + 1).astype(str) + '-07-01')
    fin['effective_end'] = pd.to_datetime((fin['year'] + 2).astype(str) + '-06-30')
    fin_drop = fin.drop(columns=['year'])
    merged = market.merge(fin_drop, on='stock_code', how='left')
    merged = merged[merged['year_month_dt'].between(merged['effective_start'], merged['effective_end'])]
    merged = merged.drop(columns=['effective_start', 'effective_end', 'year_month_dt'])
    merged['book_to_market'] = np.where(merged['market_cap'] > 0, merged['book_value'] / merged['market_cap'], np.nan)
    return merged

# ------------------ Suspension Handling & Filters ------------------

def _load_suspensions(path: Path) -> pd.DataFrame:
    """读取停牌原始表；若不存在返回空框架。

    说明：
    - 使用 TSR_Stkstat / TSR_Stkstat_new 文件；字段含义参考 DES 说明文件。
    - Timeperd 表示停牌期间在交易日内累计停牌时长（小时）。
    - 若 Timeperd < SUSP_SHORT_HOURS_THRESHOLD，则认为该次停牌对整日交易影响有限，
      在停牌天数统计中忽略（不展开为停牌工作日）。
    """
    if not path.exists():
        return pd.DataFrame(columns=['stock_code', 'Type', 'Suspdate', 'Resmdate', 'Timeperd'])
    df = pd.read_csv(path)
    df.columns = [c.split('.')[-1] for c in df.columns]
    df = df.rename(columns={'Stkcd': 'stock_code'})
    df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
    df['Type'] = pd.to_numeric(df['Type'], errors='coerce')
    df['Suspdate'] = pd.to_datetime(df['Suspdate'], errors='coerce')
    df['Resmdate'] = pd.to_datetime(df['Resmdate'], errors='coerce')
    # Timeperd: 交易停复牌时间长度（小时），-8888 表示已退市或无效
    if 'Timeperd' in df.columns:
        df['Timeperd'] = pd.to_numeric(df['Timeperd'], errors='coerce')
        # 过滤已退市标记与明显异常值
        df = df[df['Timeperd'] != -8888]
        # 仅保留停牌时长 >= 阈值的记录；短停牌不计入日度停牌统计
        df = df[(df['Timeperd'].isna()) | (df['Timeperd'] >= SUSP_SHORT_HOURS_THRESHOLD)]
    else:
        df['Timeperd'] = np.nan
    # 仅保留 1/2/3 类型
    df = df[df['Type'].isin([1, 2, 3])]
    df['Resm_filled'] = df['Resmdate'].fillna(pd.to_datetime('2100-01-01'))
    return df

def _expand_susp_days(susp: pd.DataFrame, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """将每次停牌区间展开为工作日列表。

    注意：_load_suspensions 已根据 Timeperd 和 SUSP_SHORT_HOURS_THRESHOLD 过滤掉短停牌记录，
    因此此处展开的工作日只代表“对整日交易有实质影响”的停牌日。
    """
    if susp.empty:
        return pd.DataFrame(columns=['stock_code', 'date'])
    rows = []
    min_date = pd.to_datetime(start + '-01')
    max_date = pd.to_datetime(end + '-28')
    for _, r in susp.iterrows():
        if pd.isna(r['Suspdate']):
            continue
        s = max(r['Suspdate'], min_date)
        e = min(r['Resm_filled'], max_date)
        if s > e:
            continue
        bdays = pd.bdate_range(s, e)
        if len(bdays) == 0:
            continue
        rows.append(pd.DataFrame({'stock_code': r['stock_code'], 'date': bdays}))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=['stock_code', 'date'])
    out['year_month'] = out['date'].dt.to_period('M').astype(str)
    return out

def _build_tradable_flags(panel: pd.DataFrame, susp: pd.DataFrame) -> pd.DataFrame:
    """依据论文规则生成可交易标记。"""
    susp_days = _expand_susp_days(susp)
    day_cnt = susp_days.groupby(['stock_code', 'year_month']).size().rename('halt_days_month').reset_index() if not susp_days.empty else pd.DataFrame(columns=['stock_code', 'year_month', 'halt_days_month'])
    panel = panel.copy()
    panel['year_month'] = panel['year_month'].astype(str)
    panel['month_start'] = pd.to_datetime(panel['year_month'] + '-01')
    flags = panel[['stock_code', 'year_month', 'market_cap', 'book_value', 'month_start']].drop_duplicates()
    flags = flags.merge(day_cnt, on=['stock_code', 'year_month'], how='left')
    flags['halt_days_month'] = flags['halt_days_month'].fillna(0).astype(int)
    # 上月事件次数
    if not susp.empty:
        susp['event_month'] = susp['Suspdate'].dt.to_period('M').astype(str)
        evt_cnt = susp.groupby(['stock_code', 'event_month']).size().rename('events_month').reset_index()
    else:
        evt_cnt = pd.DataFrame(columns=['stock_code', 'event_month', 'events_month'])
    flags['prev_month'] = (flags['month_start'] - pd.offsets.MonthBegin(1)).dt.to_period('M').astype(str)
    flags = flags.merge(evt_cnt.rename(columns={'event_month': 'prev_month'}), on=['stock_code', 'prev_month'], how='left')
    flags['events_prev_month'] = flags['events_month'].fillna(0).astype(int)
    flags = flags.drop(columns=['events_month'])
    # 过去12个月停牌累积
    flags = flags.sort_values(['stock_code', 'year_month'])
    flags['halt_days_12m'] = flags.groupby('stock_code')['halt_days_month'].rolling(12, min_periods=1).sum().reset_index(level=0, drop=True)
    # 形成月停牌 & 上市时长
    flags['is_formation_month'] = flags['month_start'].dt.month == FORMATION_MONTH
    if 'list_date' in panel.columns:
        ld = panel[['stock_code', 'list_date']].drop_duplicates('stock_code')
        flags = flags.merge(ld, on='stock_code', how='left')
        flags['listed6m'] = (flags['month_start'] - flags['list_date']).dt.days >= 183
    else:
        flags['listed6m'] = True
    # 市值底部30%剔除（保留 top70 标记）
    def _mark_top70(g):
        if g['market_cap'].notna().sum() == 0:
            g['top70'] = False
            return g
        q30 = g['market_cap'].quantile(0.3)
        g['top70'] = g['market_cap'] > q30
        return g
    flags = flags.groupby('year_month', group_keys=False).apply(_mark_top70)
    # ST 标记
    if 'short_name' in panel.columns:
        sn = panel[['stock_code', 'short_name']].drop_duplicates('stock_code')
        flags = flags.merge(sn, on='stock_code', how='left')
        flags['is_st'] = flags['short_name'].astype(str).str.contains('ST', case=False, regex=False)
    else:
        flags['is_st'] = False
    # 形成月停牌（直接引用当月是否有停牌天数>0）
    flags['formation_halted'] = flags['is_formation_month'] & (flags['halt_days_month'] > 0)
    # 综合可交易条件
    flags['tradable'] = (
        flags['listed6m'] &
        (flags['events_prev_month'] < 5) &
        (flags['halt_days_12m'] < 120) &
        (~flags['formation_halted']) &
        flags['top70'] &
        (~flags['is_st']) &
        (flags['book_value'] > 0)
    )
    return flags

def apply_paper_filters(panel: pd.DataFrame, suspension_path: Path, start: str = START_DATE, end: str = END_DATE, write_report: bool = True) -> pd.DataFrame:
    """对齐后面板 → 停牌处理 → 生成并应用论文过滤；返回过滤后面板。"""
    susp = _load_suspensions(suspension_path)
    flags = _build_tradable_flags(panel, susp)
    if write_report:
        summary = [f"总记录: {len(flags)}", f"可交易: {flags['tradable'].sum()} ({flags['tradable'].mean():.2%})"]
        rule_map = {
            '上市<6月': ~flags['listed6m'], '上月事件>=5': flags['events_prev_month'] >= 5,
            '过去12月停牌>=120': flags['halt_days_12m'] >= 120, '形成月停牌': flags['formation_halted'],
            '市值底部30%': ~flags['top70'], 'ST/*ST': flags['is_st'], '权益<=0': flags['book_value'] <= 0
        }
        for name, m in rule_map.items():
            summary.append(f"{name} 剔除行数: {m.sum()}")

    flags.to_csv(CLEAN_PATH / 'tradable_monthly.csv', index=False)
    panel_f = panel.merge(flags[['stock_code', 'year_month', 'tradable']], on=['stock_code', 'year_month'], how='left')
    panel_f = panel_f[panel_f['tradable']].copy()
    panel_f.to_csv(CLEAN_PATH / 'master_panel_filtered.csv', index=False)
    return panel_f

# ------------------ Build Master Panel ------------------

def build_master_panel():
    """一键整合全流程，返回各阶段 DataFrame 便于后续因子构建。"""
    raw = load_raw()
    rf = clean_risk_free_rate(raw['rf'])
    mkt_ret = clean_market_return(raw['idx'])
    market = clean_market_data(raw['market'])
    fin = clean_financial_data(raw['financial'], market)
    fin_ind = compute_financial_indicators(fin)
    panel = align_annual_to_monthly(fin_ind, market)
    panel = panel.merge(rf, on='year_month', how='left').merge(mkt_ret, on='year_month', how='left')
    try:
        panel_filtered = apply_paper_filters(panel, RAW_PATH / 'TSR_Stkstat.csv')
    except Exception as e:
        print('过滤执行失败，返回未过滤面板。错误：', e)
        panel_filtered = panel
    return {
        'rf': rf, 'mkt_ret': mkt_ret, 'market': market,
        'financial': fin, 'fin_ind': fin_ind, 'panel': panel,
        'panel_filtered': panel_filtered
    }

if __name__ == '__main__':
    data_sets = build_master_panel()
    for name, df in data_sets.items():
        print(f"\n=== {name} ===")
        print(df.head())
        print(df.info())
