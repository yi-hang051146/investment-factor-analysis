import sys
from pathlib import Path

try:
    from PyPDF2 import PdfReader
except Exception as e:
    print("ERROR: PyPDF2 not installed or failed to import:", e)
    sys.exit(2)

def extract_text(path):
    reader = PdfReader(path)
    texts = []
    for i, page in enumerate(reader.pages):
        try:
            texts.append(page.extract_text() or "")
        except Exception as e:
            texts.append(f"[ERROR extracting page {i}: {e}]")
    return texts

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extract_pdf_text.py <pdf-path> [max_chars]")
        sys.exit(1)
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"ERROR: file not found: {p}")
        sys.exit(1)
    texts = extract_text(p)
    total_pages = len(texts)
    joined = "\n\n---PAGE_BREAK---\n\n".join(texts)
    max_chars = None
    if len(sys.argv) >= 3:
        try:
            max_chars = int(sys.argv[2])
        except:
            pass
    print(f"PDF path: {p}\nPages: {total_pages}\n")
    if max_chars:
        print(joined[:max_chars])
    else:
        # print up to ~4000 chars by default
        print(joined[:4000])
    # exit normally
    sys.exit(0)

