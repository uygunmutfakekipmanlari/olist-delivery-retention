"""Download the Olist public dataset from Olist's own repository into data/raw/.

Source: https://github.com/olist/work-at-olist-data (datasets/), also on Kaggle as
"Brazilian E-Commerce Public Dataset by Olist". License: CC BY-NC-SA 4.0.
The raw files are not committed to this repository.
"""
import subprocess
import shutil
import tempfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
REPO = "https://github.com/olist/work-at-olist-data"
NEEDED = ["olist_orders_dataset.csv", "olist_customers_dataset.csv", "olist_order_items_dataset.csv",
          "olist_products_dataset.csv", "olist_order_reviews_dataset.csv",
          "product_category_name_translation.csv"]


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if all((RAW / f).exists() for f in NEEDED):
        print("data/raw already complete")
        return
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "--depth", "1", REPO, tmp], check=True)
        for f in NEEDED:
            shutil.copy(Path(tmp) / "datasets" / f, RAW / f)
    print(f"copied {len(NEEDED)} files to {RAW}")


if __name__ == "__main__":
    main()
