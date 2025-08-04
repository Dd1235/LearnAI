from pathlib import Path

from img2table.document import Image
from img2table.ocr import TesseractOCR

IMG_PATH = "telugu2.png"
OUT_DIR = Path("trials")
OUT_DIR.mkdir(exist_ok=True, parents=True)

ocr = TesseractOCR(lang="tel", psm=6, n_threads=1)

doc = Image(IMG_PATH)
tables = doc.extract_tables(
    ocr=ocr,
    implicit_rows=False,
    implicit_columns=False,
    borderless_tables=False,
    min_confidence=10,
)

if not tables:
    raise RuntimeError(" No tables detected.  Check image quality or tune parameters.")

for idx, tbl in enumerate(tables, 1):
    html_path = OUT_DIR / f"table_{idx}.html"
    csv_path = OUT_DIR / f"table_{idx}.csv"

    html_path.write_text(tbl.html, encoding="utf-8")
    tbl.df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n=== Table {idx} ===")
    print(tbl.df.head())
    print(f"✔ Saved HTML → {html_path}")
    print(f"✔ Saved CSV  → {csv_path}")
