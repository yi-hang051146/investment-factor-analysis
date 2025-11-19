import pandas as pd
import numpy as np
from pathlib import Path

# =============================
# 投资因子构建脚本概述（简化版）
# =============================
# 目标：基于已清洗并过滤后的月度面板（master_panel_filtered.csv）
# 构建 Fama-French 经典因子：Mkt-RF / SMB / HML / RMW / CMA(AG)
# 以及论文关注的投资拆解：CMA_PPEG / CMA_CASH / CMA_RECE / CMA_INVT
# 并输出：
#   1. 因子时间序列 factors.csv
#   2. 投资组合（标签组合）月度加权收益 portfolio_returns.csv
#   3. 因子相关性矩阵 table_correlation_matrix.md（若存在全部列）
#   4. 增长类底层指标描述性统计 table_growth_stats.md
# 流程：加载面板 → 形成月(6月)分箱 → 标签外推到持有期(7~次年6) → 计算组合加权收益 → 构建因子差分 → 写出文件
# 说明：分组使用 qcut；失败回退到基于排序近似等频；市值权重采用上一月流通市值（缺失用当月补）。
# =============================

# 路径配置
BASE = Path('d:/python_workspace/projects/Pricing-day3')
CLEAN = BASE / 'data' / 'cleaned_data'
OUTPUT = BASE / 'output'
OUTPUT.mkdir(exist_ok=True)

PANEL_FILTERED = CLEAN / 'master_panel_filtered.csv'

OUTPUT_FACTORS = OUTPUT / 'factors.csv'
OUTPUT_PORT_RET = OUTPUT / 'portfolio_returns.csv'
OUTPUT_CORR_MD = OUTPUT / 'table_correlation_matrix.md'
OUTPUT_GROWTH_MD = OUTPUT / 'table_growth_stats.md'

FORMATION_MONTH = 6  # 组合形成月份（6=6月）

# =============================
# 基础工具函数
# =============================

def load_panel() -> pd.DataFrame:
    """加载过滤后的面板数据（仅使用 master_panel_filtered.csv）。
    要求列：stock_code / year_month / monthly_return / market_cap / book_to_market / op_profitability
            / asset_growth / ppe_growth / cash_growth / receivables_growth / inventory_growth
            / rf_monthly / market_return / book_value
    若缺列直接抛错，避免静默错误。
    """
    path = PANEL_FILTERED
    df = pd.read_csv(path)
    df['year_month'] = df['year_month'].astype(str)
    need = ['stock_code','year_month','monthly_return','market_cap','book_to_market','op_profitability',
            'asset_growth','ppe_growth','cash_growth','receivables_growth','inventory_growth','rf_monthly',
            'market_return','book_value']
    # 标准化股票代码
    df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
    return df[need]

def add_lag_me(df: pd.DataFrame) -> pd.DataFrame:
    """添加上一月市值（ME_lag），用于价值加权；首月缺失用当月市值补。"""
    out = df.sort_values(['stock_code','year_month']).copy()
    out['ME_lag'] = out.groupby('stock_code')['market_cap'].shift(1)
    out['ME_lag'] = out['ME_lag'].fillna(out['market_cap'])
    return out


def year_month_to_period(s: pd.Series) -> pd.Series:
    """将形如 'YYYY-MM' 或 'YYYYMM' 的年月字符串序列转换为 pandas 的月度 Period（period[M]）。
    返回与输入等长的 Series；若解析失败将由 pd.to_datetime 抛出异常。
    """
    return pd.to_datetime(s.astype(str) + '-01').dt.to_period('M')

def robust_qcut(series: pd.Series, quantiles, labels):
    """稳健分箱：优先使用 qcut；若唯一值不足或 qcut 边界重复则回退到基于排序近似分箱。
    返回：分类 Series 或 None（指示该维度放弃分组）。"""
    s = series.dropna()
    if s.nunique() < len(labels):
        return None
    try:
        return pd.qcut(s, quantiles, labels=labels, duplicates='drop')
    except Exception:
        try:
            ranks = s.rank(method='first')
            # 使用分位数切分排序值
            cuts = np.quantile(ranks, quantiles)
            cuts = np.unique(cuts)
            if len(cuts) - 1 < len(labels):
                return None
            # 只取前 len(labels) 个区间标签
            cat = pd.cut(ranks, bins=cuts, labels=labels[:len(cuts)-1], include_lowest=True)
            return cat
        except Exception:
            return None

# =============================
# 形成月分组与标签生成
# =============================

def form_portfolios(panel: pd.DataFrame, formation_month: int = FORMATION_MONTH) -> pd.DataFrame:
    """在形成月（默认 6 月）进行多维分箱：Size / BM / OP / AG / PPE / CASH / RECE / INVT。
    返回：包含所有标签与原始股票代码的形成月截面 DataFrame。"""
    df = panel.copy()
    df['period'] = year_month_to_period(df['year_month'])
    df['year'] = df['period'].dt.year
    df['month'] = df['period'].dt.month
    form_df = df[df['month'] == formation_month].copy()

    # 各维度分组
    group_specs = [
        ('market_cap',[0,0.5,1],['S','B'],'size_grp'),
        ('book_to_market',[0,0.3,0.7,1],['L','M','H'],'bm_grp'),
        ('op_profitability',[0,0.3,0.7,1],['W','M','R'],'op_grp'),
        ('asset_growth',[0,0.3,0.7,1],['A','M','C'],'inv_grp'),
        ('ppe_growth',[0,0.3,0.7,1],['A','M','C'],'ppe_grp'),
        ('cash_growth',[0,0.3,0.7,1],['A','M','C'],'cash_grp'),
        ('receivables_growth',[0,0.3,0.7,1],['A','M','C'],'rece_grp'),
        ('inventory_growth',[0,0.3,0.7,1],['A','M','C'],'invt_grp'),
    ]
    for col, q, labels, out_col in group_specs:
        grp = robust_qcut(form_df[col], q, labels)
        if grp is not None:
            form_df.loc[grp.index, out_col] = grp
    needed = ['size_grp','bm_grp','op_grp','inv_grp','ppe_grp','cash_grp','rece_grp','invt_grp']
    form_df = form_df.dropna(subset=needed)

    # 统一标签生成
    tag_map = {
        'tag_bm': ('size_grp','bm_grp'),
        'tag_op': ('size_grp','op_grp'),
        'tag_inv': ('size_grp','inv_grp'),
        'tag_ppe': ('size_grp','ppe_grp'),
        'tag_cash': ('size_grp','cash_grp'),
        'tag_rece': ('size_grp','rece_grp'),
        'tag_invt': ('size_grp','invt_grp'),
    }
    for tag, (g1,g2) in tag_map.items():
        form_df[tag] = form_df[g1].astype(str) + '_' + form_df[g2].astype(str)

    keep_cols = ['stock_code','year'] + needed + list(tag_map.keys())
    return form_df[keep_cols]

# =============================
# 标签外推到持有期（7 月 ~ 次年 6 月）
# =============================

def expand_portfolios(panel: pd.DataFrame, formation_df: pd.DataFrame, formation_month: int = FORMATION_MONTH) -> pd.DataFrame:
    """将形成月的标签应用到后续 12 个月持有期：当年 7 月 ~ 次年 6 月。"""
    df = panel.copy()
    df['period'] = year_month_to_period(df['year_month'])
    df['year'] = df['period'].dt.year
    tag_cols = ['tag_bm','tag_op','tag_inv','tag_ppe','tag_cash','tag_rece','tag_invt']
    for tc in tag_cols:
        df[tc] = pd.Series(index=df.index, dtype='object')
    for yr, grp in formation_df.groupby('year'):
        start = pd.Period(f'{yr}-{formation_month+1:02d}')  # 7 月
        end = pd.Period(f'{yr+1}-{(formation_month):02d}')  # 次年 6 月
        hold_mask = (df['period'] >= start) & (df['period'] <= end)
        sub_idx = df.index[hold_mask & df['stock_code'].isin(grp['stock_code'])]
        mapping_dict = {tc: dict(zip(grp['stock_code'], grp[tc])) for tc in tag_cols}
        for tc in tag_cols:
            df.loc[sub_idx, tc] = df.loc[sub_idx, 'stock_code'].map(mapping_dict[tc]).astype('object')
    df = df.dropna(subset=tag_cols)
    return df

# =============================
# 计算组合价值加权收益
# =============================

def weighted_portfolio_returns(panel_with_tags: pd.DataFrame) -> dict:
    """按标签与月份分组，使用上一月市值加权计算组合收益。返回 {tag_type: DataFrame}。"""
    df = add_lag_me(panel_with_tags)
    df['weight'] = df['ME_lag'].fillna(0)
    portfolios = {}
    for tag_type in ['tag_bm','tag_op','tag_inv','tag_ppe','tag_cash','tag_rece','tag_invt']:
        grp = df.groupby(['year_month', tag_type])
        ret = grp.apply(lambda g: np.average(g['monthly_return'], weights=g['weight']) if g['weight'].sum()>0 else np.nan)
        ret = ret.reset_index(name='ret')
        portfolios[tag_type] = ret
    return portfolios

# =============================
# 构建因子
# =============================

def compute_factors(portfolios: dict, panel_with_tags: pd.DataFrame) -> pd.DataFrame:
    """依据经典差分公式构建各类因子；SMB 取三类（BM/OP/INV）小盘-大盘均值。"""
    def pivot(ret_df):
        return ret_df.pivot(index='year_month', columns=ret_df.columns[1], values='ret')

    bm_p = pivot(portfolios['tag_bm'])
    op_p = pivot(portfolios['tag_op'])
    inv_p = pivot(portfolios['tag_inv'])
    ppe_p = pivot(portfolios['tag_ppe'])
    cash_p = pivot(portfolios['tag_cash'])
    rece_p = pivot(portfolios['tag_rece'])
    invt_p = pivot(portfolios['tag_invt'])

    rf = panel_with_tags.groupby('year_month')['rf_monthly'].first()
    mkt = panel_with_tags.groupby('year_month')['market_return'].first()
    mkt_rf = mkt - rf

    def smb_component(pivot_df: pd.DataFrame):
        small = [c for c in pivot_df.columns if str(c).startswith('S_')]
        big = [c for c in pivot_df.columns if str(c).startswith('B_')]
        if not small or not big:
            return pd.Series(index=pivot_df.index, dtype='float64')
        return pivot_df[small].mean(axis=1) - pivot_df[big].mean(axis=1)

    smb_bm = smb_component(bm_p)
    smb_op = smb_component(op_p)
    smb_inv = smb_component(inv_p)
    smb_ppe = smb_component(ppe_p)  # PPE 维度的 SMB 分量（可用于诊断）
    smb = pd.concat([smb_bm, smb_op, smb_inv], axis=1).mean(axis=1)

    # 差分公式（若某列缺失返回 NaN）
    def diff_factor(pivot_df, high_code, low_code):
        h = pivot_df.get(high_code)
        l = pivot_df.get(low_code)
        if h is None or l is None:
            return pd.Series(index=pivot_df.index, dtype='float64')
        return (h) - (l)

    # HML：高 BM - 低 BM（取 S/B 平均形式）
    hml = ((bm_p.get('S_H') + bm_p.get('B_H'))/2) - ((bm_p.get('S_L') + bm_p.get('B_L'))/2)
    rmw = ((op_p.get('S_R') + op_p.get('B_R'))/2) - ((op_p.get('S_W') + op_p.get('B_W'))/2)
    cma_ag = ((inv_p.get('S_C') + inv_p.get('B_C'))/2) - ((inv_p.get('S_A') + inv_p.get('B_A'))/2)
    cma_ppeg = ((ppe_p.get('S_C') + ppe_p.get('B_C'))/2) - ((ppe_p.get('S_A') + ppe_p.get('B_A'))/2)
    cma_cash = ((cash_p.get('S_C') + cash_p.get('B_C'))/2) - ((cash_p.get('S_A') + cash_p.get('B_A'))/2)
    cma_rece = ((rece_p.get('S_C') + rece_p.get('B_C'))/2) - ((rece_p.get('S_A') + rece_p.get('B_A'))/2)
    cma_invt = ((invt_p.get('S_C') + invt_p.get('B_C'))/2) - ((invt_p.get('S_A') + invt_p.get('B_A'))/2)

    factors = pd.DataFrame({
        'Mkt-RF': mkt_rf,
        'SMB': smb,
        'SMB_BM': smb_bm,
        'SMB_OP': smb_op,
        'SMB_INV': smb_inv,
        'HML': hml,
        'RMW': rmw,
        'CMA_AG': cma_ag,
        'CMA_PPEG': cma_ppeg,
        'CMA_CASH': cma_cash,
        'CMA_RECE': cma_rece,
        'CMA_INVT': cma_invt,
        'SMB_PPE': smb_ppe
    }).sort_index()
    return factors

# =============================
# 主流程
# =============================

def run_factor_construction():
    panel = load_panel()
    formation_df = form_portfolios(panel)
    panel_with_tags = expand_portfolios(panel, formation_df)
    portfolios = weighted_portfolio_returns(panel_with_tags)

    # 汇总所有组合收益
    all_port_list = []
    for tag_type, df_ret in portfolios.items():
        df_ret['group_type'] = tag_type
        all_port_list.append(df_ret)
    all_ports_df = pd.concat(all_port_list, ignore_index=True)

    factors = compute_factors(portfolios, panel_with_tags)
    factors.to_csv(OUTPUT_FACTORS, index=True)
    all_ports_df.to_csv(OUTPUT_PORT_RET, index=False)

    # 因子相关性矩阵写入 markdown
    try:
        corr = factors.corr()
        md_lines = ["# 因子相关性矩阵", corr.to_markdown()]
        OUTPUT_CORR_MD.write_text('\n\n'.join(md_lines), encoding='utf-8')
    except Exception as e:
        print('相关性矩阵写出失败:', e)

    # 输出 AG, CASH, RECE, INVT, PPE 相关性矩阵到控制台
    try:
        sub_factors = factors[['CMA_AG', 'CMA_CASH', 'CMA_RECE', 'CMA_INVT', 'CMA_PPEG']]
        sub_corr = sub_factors.corr()
        print("\nAG, CASH, RECE, INVT, PPE 相关性矩阵:")
        print(sub_corr)
    except Exception as e:
        print('子集相关性计算失败:', e)

    # 增长类指标描述性统计
    growth_cols = ['asset_growth','cash_growth','receivables_growth','inventory_growth','ppe_growth']
    growth_stats = panel[growth_cols].replace([np.inf,-np.inf], np.nan).describe().T[['mean','std','min','max','50%']].rename(columns={'50%':'median'})
    try:
        OUTPUT_GROWTH_MD.write_text('# 增长类指标描述性统计\n\n' + growth_stats.to_markdown(), encoding='utf-8')
    except Exception as e:
        print('增长统计写出失败:', e)

    # 控制台输出简要
    print(f'因子文件保存 -> {OUTPUT_FACTORS}')
    print(f'组合收益文件保存 -> {OUTPUT_PORT_RET}')
    print(f'相关性矩阵 -> {OUTPUT_CORR_MD}')
    print(f'增长统计 -> {OUTPUT_GROWTH_MD}')
    print('因子首行预览:')
    print(factors.head())

if __name__ == '__main__':
    run_factor_construction()
