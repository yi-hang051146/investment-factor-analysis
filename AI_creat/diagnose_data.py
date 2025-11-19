import sys
from pathlib import Path
import pandas as pd

RAW = Path('d:/python_workspace/projects/Pricing-day3/data/raw_data')
FILES = {
    'market': 'TRD_Mnth(Merge Query).csv',
    'financial': 'FS_Combas(Merge Query).csv',
    'rf': 'CMMPI_Uimd.csv',
    'status': 'TSR_Stkstat.csv',
}

EXPECTED_COLS = {
    'market': ['Stkcd','Trdmnt','Msmvttl','Mretwd','Nnindcd'],
    'financial': ['Stkcd','Accper','Typrep','A001000000','A003000000','A001212000','B001100000','B001200000'],
    'rf': ['Datesgn','Uimd01','Uimd07'],
    'status': ['Stkcd','Annctime','Type','Suspdate','Resmdate']
}

ENCODINGS = ['gbk','utf-8','utf-8-sig','latin1']
SEPARATORS = [',',';','\t']


def try_read(path: Path):
    last_err = None
    for enc in ENCODINGS:
        for sep in SEPARATORS:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine='python')
                if df.shape[1] == 1 and sep != '\t':
                    # likely wrong sep, continue
                    continue
                return enc, sep, df
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f'Failed to read {path.name} with encodings {ENCODINGS} & seps {SEPARATORS}: {last_err}')


def infer_prefix(cols):
    # 返回公共前缀（如 TRD_Mnth. 或 FS_Combas.）
    parts = [c.split('.')[0] for c in cols if '.' in c]
    if not parts:
        return None
    from collections import Counter
    cnt = Counter(parts)
    prefix, _ = cnt.most_common(1)[0]
    return prefix


def check_file(key, filename):
    p = RAW / filename
    print(f'\n=== {key}: {p.name} ===')
    if not p.exists():
        print('MISSING: file not found')
        return None
    print(f'File size: {p.stat().st_size} bytes')
    # 打印首行原始文本
    try:
        with open(p,'rb') as fh:
            raw_head = fh.read(300).decode('utf-8','ignore')
            print('RAW HEAD:', raw_head.splitlines()[0][:200])
    except Exception:
        pass
    try:
        enc, sep, df = try_read(p)
        cols = [str(c).strip().replace('\ufeff','') for c in df.columns]
        df.columns = cols
        print(f'ENCODING: {enc}  SEP: {repr(sep)}  SHAPE: {df.shape}')
        print('COLUMNS:', cols[:25])
        prefix = infer_prefix(cols)
        if prefix:
            print('Detected column prefix:', prefix)
        expected = EXPECTED_COLS[key]
        # 如果有前缀则剥离前缀再匹配
        if prefix:
            stripped = [c.split('.',1)[-1] for c in cols]
            missing = [c for c in expected if c not in stripped]
        else:
            missing = [c for c in expected if c not in cols]
        if missing:
            print('MISSING EXPECTED COLUMNS:', missing)
        else:
            print('All expected columns present (considering prefix)')
        print('HEAD:')
        print(df.head(3))
        # 日期字段范围
        if key == 'market':
            date_col = prefix + '.Trdmnt' if prefix and prefix + '.Trdmnt' in cols else 'Trdmnt'
            if date_col in cols:
                try:
                    dt = pd.to_datetime(df[date_col])
                    print('Trdmnt range:', dt.min(), '->', dt.max())
                except Exception as e:
                    print('Trdmnt parse error:', e)
        if key == 'financial':
            date_col = prefix + '.Accper' if prefix and prefix + '.Accper' in cols else 'Accper'
            if date_col in cols:
                try:
                    dt = pd.to_datetime(df[date_col])
                    print('Accper range:', dt.min(), '->', dt.max())
                except Exception as e:
                    print('Accper parse error:', e)
        if key == 'rf':
            if 'Datesgn' in cols:
                try:
                    dt = pd.to_datetime(df['Datesgn'])
                    print('Datesgn range:', dt.min(), '->', dt.max())
                except Exception as e:
                    print('Datesgn parse error:', e)
        return df
    except Exception as e:
        print('READ ERROR:', e)
        return None


def main():
    print('Raw path:', RAW)
    loaded = {}
    for k, fname in FILES.items():
        loaded[k] = check_file(k, fname)
    print('\nDone diagnostics.')

if __name__ == '__main__':
    main()
