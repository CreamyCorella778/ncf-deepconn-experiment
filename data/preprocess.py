import argparse
import pandas as pd
import json
from pathlib import Path
import sys
from typing import Optional
from datasets import load_dataset
import gc
import numpy as np
import pickle

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# ============================================================
# 1. TẢI AMAZON REVIEWS QUA load_dataset (Hugging Face)
# ============================================================


def load_amazon_reviews_hf(
    category: str = "All_Beauty",
    split: str = "full",
    max_records: Optional[int] = None,
    trust_remote_code: bool = True,
) -> pd.DataFrame:
    """
    Tải Amazon Reviews 2023 từ Hugging Face bằng load_dataset.

    Tham số:
        category: tên category, ví dụ "All_Beauty", "Books", "Software"
                  (tương ứng subset "raw_review_{category}")
        split: "full" hoặc "train"/"test" nếu dùng subset đã chia sẵn
        max_records: giới hạn số dòng (hữu ích khi thử nghiệm)
        trust_remote_code: cần thiết cho một số subset của dataset này

    Trả về DataFrame với các cột gốc từ dataset.
    """
    subset_name = f"raw_review_{category}"
    print(f"[Amazon] Đang tải: {subset_name} (split={split})...")

    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        subset_name,
        split=split,
        trust_remote_code=trust_remote_code,
    )

    # Nếu cần giới hạn số dòng (tránh tràn RAM khi thử nghiệm)
    if max_records is not None:
        dataset = dataset.select(range(min(max_records, len(dataset))))

    df = dataset.to_pandas()
    print(f"[Amazon] Đã tải {len(df):,} bản ghi. Các cột: {list(df.columns)}")
    return df


# ============================================================
# 2. CHUẨN HÓA SCHEMA AMAZON
# ============================================================


def normalize_amazon_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ánh xạ trường dữ liệu Amazon Reviews 2023 về schema chuẩn:
        user_id, item_id, rating, timestamp, review_text

    Lưu ý về tên cột trong dataset gốc (raw_review_*):
        - user_id       (đã đúng tên)
        - parent_asin   -> item_id  (hoặc 'asin' tùy phiên bản)
        - rating        (đã đúng tên)
        - timestamp     (Unix timestamp, mili-giây)
        - text          -> review_text
    """
    # Xác định cột item_id: ưu tiên parent_asin, fallback asin
    if "parent_asin" in df.columns:
        item_col = "parent_asin"
    elif "asin" in df.columns:
        item_col = "asin"
    else:
        raise KeyError("Không tìm thấy cột 'parent_asin' hoặc 'asin' trong DataFrame.")

    # Xác định cột review text
    if "text" in df.columns:
        text_col = "text"
    elif "reviewText" in df.columns:
        text_col = "reviewText"
    else:
        text_col = None  # có thể không có review text

    # Xây dựng DataFrame chuẩn
    out = pd.DataFrame(
        {
            "user_id": df["user_id"],
            "item_id": df[item_col],
            "rating": df["rating"],
            "timestamp": df["timestamp"],
            "review_text": df[text_col] if text_col else "",
        }
    )

    return out


# ============================================================
# 3. HÀM LÀM SẠCH CHUNG (DÙNG CHO CẢ AMAZON VÀ YELP)
# ============================================================


def clean_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thực hiện đúng quy trình làm sạch:
      1. Loại bỏ bản ghi thiếu user_id, item_id, rating.
      2. review_text rỗng -> thay bằng "".
      3. Loại bỏ trùng lặp tuyệt đối.
      4. Với mỗi cặp (user_id, item_id) bị đánh giá nhiều lần,
         chỉ giữ bản ghi có timestamp mới nhất.
    """
    original_len = len(df)

    # --- Bước 1: Loại bỏ missing ở các cột bắt buộc ---
    required_cols = ["user_id", "item_id", "rating"]
    df = df.dropna(subset=required_cols).copy()
    print(f"  - Sau khi bỏ missing: {len(df):,} / {original_len:,}")

    # --- Bước 2: Xử lý review_text rỗng ---
    df["review_text"] = df["review_text"].fillna("").astype(str)

    # --- Bước 3: Loại bỏ trùng lặp tuyệt đối ---
    before = len(df)
    df = df.drop_duplicates()
    print(
        f"  - Sau khi bỏ trùng lặp tuyệt đối: {len(df):,} (đã bỏ {before - len(df):,})"
    )

    # --- Bước 4: Giữ bản ghi mới nhất cho mỗi (user_id, item_id) ---
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        df = df.sort_values("timestamp", ascending=False, na_position="last")
        df = df.drop_duplicates(subset=["user_id", "item_id"], keep="first")
    else:
        df = df.drop_duplicates(subset=["user_id", "item_id"], keep="first")

    print(
        f"  - Sau khi khử trùng (user_id, item_id) giữ timestamp mới nhất: {len(df):,}"
    )

    df = df[["user_id", "item_id", "rating", "timestamp", "review_text"]].reset_index(
        drop=True
    )
    return df


def build_review_docs(
    train_df: pd.DataFrame,
    all_df: pd.DataFrame = None,
    max_review_chars: int = 1000,
    max_doc_chars: int = 50_000,
):
    """
    Gom review CHỈ TỪ TẬP TRAIN để tránh data leakage.

    Chiến lược chống OOM:
      - Tách hoàn toàn khỏi pyarrow: chuyển sang list Python thuần.
      - Cắt mỗi review tối đa max_review_chars ký tự.
      - Cắt mỗi document (user/item) tối đa max_doc_chars ký tự.
      - Duyệt bằng vòng lặp for trên zip(), không dùng groupby().apply().
    """
    # 1. Lấy list Python thuần, không giữ Arrow
    users = train_df["user_idx"].to_numpy(dtype=np.int64, copy=True)
    items = train_df["item_idx"].to_numpy(dtype=np.int64, copy=True)
    texts = train_df["review_text"].astype(object).to_numpy(dtype=object, copy=True)
    del train_df
    gc.collect()

    # 2. Làm sạch + cắt ngắn từng review, lưu vào dict theo user và item
    user_parts = {}  # user_idx -> list[str]
    item_parts = {}  # item_idx -> list[str]
    user_len = {}
    item_len = {}

    for u, i, raw in zip(users.tolist(), items.tolist(), texts.tolist()):
        c = clean_text(raw)[:max_review_chars]
        if not c:
            continue

        # user
        lst = user_parts.get(u)
        if lst is None:
            user_parts[u] = [c]
            user_len[u] = len(c)
        elif user_len[u] < max_doc_chars:
            room = max_doc_chars - user_len[u]
            piece = c[:room]
            lst.append(piece)
            user_len[u] += len(piece)

        # item
        lst = item_parts.get(i)
        if lst is None:
            item_parts[i] = [c]
            item_len[i] = len(c)
        elif item_len[i] < max_doc_chars:
            room = max_doc_chars - item_len[i]
            piece = c[:room]
            lst.append(piece)
            item_len[i] += len(piece)

    # 3. Nối thành chuỗi cuối cùng
    user_docs = {u: " ".join(parts)[:max_doc_chars] for u, parts in user_parts.items()}
    item_docs = {i: " ".join(parts)[:max_doc_chars] for i, parts in item_parts.items()}

    del user_parts, item_parts, user_len, item_len, users, items, texts
    gc.collect()

    if all_df is not None:
        all_users = set(all_df["user_idx"].unique().tolist())
        all_items = set(all_df["item_idx"].unique().tolist())
        for u in all_users - user_docs.keys():
            user_docs[u] = ""
        for i in all_items - item_docs.keys():
            item_docs[i] = ""
    return user_docs, item_docs


def build_review_docs_streaming(
    train_df: pd.DataFrame,
    all_df: pd.DataFrame = None,
    user_out_path: str = None,
    item_out_path: str = None,
    max_review_chars: int = 1000,
    max_doc_chars: int = 50_000,
    chunk_size: int = 5000,
):
    """
    Gom review CHỈ TỪ TẬP TRAIN, ghi pickle theo chunk để tránh OOM.

    Thay vì trả về dict, ghi trực tiếp ra file pickle.
    Mỗi chunk được ghi ngay, không giữ toàn bộ trong RAM.
    """
    # 1. Lấy mảng numpy, không dùng .tolist() để giảm overhead
    users = train_df["user_idx"].to_numpy(dtype=np.int64, copy=True)
    items = train_df["item_idx"].to_numpy(dtype=np.int64, copy=True)
    texts = train_df["review_text"].astype(object).to_numpy(dtype=object, copy=True)
    del train_df
    gc.collect()

    print(f"[build_review_docs] {len(users):,} dòng, xử lý theo chunk {chunk_size}...")

    # 2. Gom theo user — ghi theo chunk
    # Sắp xếp theo user_idx để các dòng cùng user nằm liền nhau
    order = np.argsort(users, kind="mergesort")
    users_sorted = users[order]
    texts_sorted = texts[order]
    del order
    gc.collect()

    _write_grouped_pickle(
        ids=users_sorted,
        texts=texts_sorted,
        out_path=user_out_path,
        max_review_chars=max_review_chars,
        max_doc_chars=max_doc_chars,
    )
    del users_sorted, texts_sorted
    gc.collect()

    # 3. Gom theo item
    order = np.argsort(items, kind="mergesort")
    items_sorted = items[order]
    texts_sorted = texts[order]
    del order, users, items, texts
    gc.collect()

    _write_grouped_pickle(
        ids=items_sorted,
        texts=texts_sorted,
        out_path=item_out_path,
        max_review_chars=max_review_chars,
        max_doc_chars=max_doc_chars,
    )
    del items_sorted, texts_sorted
    gc.collect()

    # 4. Điền ID thiếu (cold-start) từ all_df
    if all_df is not None:
        _fill_missing_ids(all_df["user_idx"].unique(), user_out_path)
        _fill_missing_ids(all_df["item_idx"].unique(), item_out_path)


def _write_grouped_pickle(
    ids: np.ndarray,
    texts: np.ndarray,
    out_path: str,
    max_review_chars: int,
    max_doc_chars: int,
):
    """Ghi pickle theo từng nhóm ID, giải phóng ngay sau mỗi nhóm."""
    if out_path is None:
        return

    n = len(ids)
    i = 0
    with open(out_path, "wb") as f:
        while i < n:
            cur_id = ids[i]
            j = i
            parts = []
            total = 0
            while j < n and ids[j] == cur_id:
                c = clean_text(texts[j])[:max_review_chars]
                if c:
                    extra = 1 if parts else 0
                    if total + len(c) + extra <= max_doc_chars:
                        parts.append(c)
                        total += len(c) + extra
                    elif total == 0:
                        # review đầu tiên dài hơn max_doc_chars — vẫn cắt
                        parts.append(c[:max_doc_chars])
                        total = max_doc_chars
                j += 1
            doc = " ".join(parts)[:max_doc_chars] if parts else ""
            pickle.dump((int(cur_id), doc), f, protocol=pickle.HIGHEST_PROTOCOL)
            del parts, doc
            i = j
    print(f"  -> ghi xong {out_path} ({n:,} dòng)")


def _fill_missing_ids(all_ids: np.ndarray, out_path: str):
    """Đọc lại pickle dạng tuple (id, doc), thêm các id thiếu = '', ghi lại."""
    if out_path is None:
        return
    existing = {}
    with open(out_path, "rb") as f:
        while True:
            try:
                k, v = pickle.load(f)
                existing[k] = v
            except EOFError:
                break
    added = 0
    for uid in all_ids.tolist():
        if uid not in existing:
            existing[int(uid)] = ""
            added += 1
    with open(out_path, "wb") as f:
        pickle.dump(existing, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  -> bổ sung {added:,} ID thiếu vào {out_path}")


def load_review_pickle(path: str) -> dict:
    """Đọc file pickle dạng tuple (id, doc) hoặc dict, chuẩn hóa về dict."""
    with open(path, "rb") as f:
        try:
            obj = pickle.load(f)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # Nếu là stream tuple, đọc lại từ đầu
    result = {}
    with open(path, "rb") as f:
        while True:
            try:
                k, v = pickle.load(f)
                result[k] = v
            except EOFError:
                break
    return result


# ============================================================
# 4. YELP (GIỮ NGUYÊN LOGIC TỪ TRƯỚC)
# ============================================================


def load_yelp_reviews(
    review_json_path: str, max_records: Optional[int] = None
) -> pd.DataFrame:
    path = Path(review_json_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file Yelp: {review_json_path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_records and i >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            date_str = obj.get("date")
            ts = int(pd.Timestamp(date_str).timestamp()) if date_str else None

            records.append(
                {
                    "user_id": obj.get("user_id"),
                    "item_id": obj.get("business_id"),
                    "rating": obj.get("stars"),
                    "timestamp": ts,
                    "review_text": obj.get("text", ""),
                }
            )

    df = pd.DataFrame(records)
    print(f"[Yelp] Đọc {len(df):,} bản ghi từ {path}")
    return df


"""
prepare_pipeline.py
-------------------
Bước 2–6: K-Core → Label Encoding → Làm sạch text & gom review
          → Chia tập theo timestamp → Xuất parquet/pkl.

Chạy sau khi đã có df_amazon_clean và df_yelp_clean từ bước 1.
"""

import re
import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ============================================================
# BƯỚC 2: LỌC K-CORE
# ============================================================
def apply_k_core(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Lặp lọc cho đến khi mọi user và item đều có >= k tương tác."""
    while True:
        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()
        valid_users = user_counts[user_counts >= k].index
        valid_items = item_counts[item_counts >= k].index
        new_df = df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]
        if len(new_df) == len(df):
            break
        df = new_df
    return df.reset_index(drop=True)


# ============================================================
# BƯỚC 3: LABEL ENCODING
# ============================================================
def encode_ids(df: pd.DataFrame):
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    df = df.copy()
    df["user_idx"] = user_enc.fit_transform(df["user_id"])
    df["item_idx"] = item_enc.fit_transform(df["item_id"])
    return df, user_enc, item_enc


# ============================================================
# BƯỚC 4: LÀM SẠCH TEXT
# ============================================================
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


def clean_text(s: str) -> str:
    s = str(s).lower()
    s = _URL_RE.sub(" ", s)
    s = _HTML_RE.sub(" ", s)
    s = _NON_ALNUM_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


# ============================================================
# BƯỚC 5: CHIA TẬP THEO TIMESTAMP (LEAVE-ONE-OUT)
# ============================================================
def split_leave_one_out(df: pd.DataFrame):
    """
    Leave-One-Out theo timestamp, tối ưu bộ nhớ:
      - Chỉ giữ các cột cần thiết.
      - Dùng argsort + slicing thay boolean mask.
    """
    # Chỉ giữ cột cần thiết — QUAN TRỌNG: loại review_text khỏi split
    df = df[["user_idx", "item_idx", "rating", "timestamp"]].copy()

    # Sắp xếp theo (user, timestamp)
    df = df.sort_values(["user_idx", "timestamp"], kind="mergesort").reset_index(
        drop=True
    )

    # Tính rank từ cuối cho mỗi user bằng numpy (nhanh + ít RAM hơn groupby.cumcount)
    user_arr = df["user_idx"].values
    n = len(user_arr)
    # Đếm ngược: vị trí cuối cùng của mỗi user có rank=0, trước đó rank=1, ...
    # Tính bằng cách đếm số lần xuất hiện của user trong phần còn lại
    # Cách đơn giản: dùng pandas nhưng trên mảng nhỏ
    counts = pd.Series(user_arr).groupby(user_arr).cumcount(ascending=False).values

    # Lấy index bằng numpy
    test_idx = np.where(counts == 0)[0]
    val_idx = np.where(counts == 1)[0]
    train_idx = np.where(counts >= 2)[0]

    # Tạo DataFrame con bằng .iloc (chỉ copy phần cần)
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    del df, counts, user_arr
    gc.collect()
    return train_df, val_df, test_df


# ============================================================
# PIPELINE CHO MỘT DATASET
# ============================================================
def process_dataset(
    name: str, df: pd.DataFrame, k: int, out_dir: Path, use_kcore: bool = True
):
    print(f"\n{'='*60}\n>>> Xử lý {name.upper()} (k={k}, kcore={use_kcore})\n{'='*60}")
    print(
        f"Trước: {len(df):,} bản ghi, {df.user_id.nunique():,} user, {df.item_id.nunique():,} item"
    )

    if use_kcore:
        df = apply_k_core(df, k=k)
        print(
            f"Sau K-Core: {len(df):,} bản ghi, {df.user_id.nunique():,} user, {df.item_id.nunique():,} item"
        )
    else:
        print("Bỏ qua K-Core")

    if len(df) == 0:
        print(f"[CẢNH BÁO] {name}: không còn dữ liệu.")
        return

    df, user_enc, item_enc = encode_ids(df)
    gc.collect()

    # Split trên bản không có review_text để tiết kiệm RAM
    train_idx_df, val_df, test_df = split_leave_one_out(df)
    print(
        f"Split: train={len(train_idx_df):,} | val={len(val_df):,} | test={len(test_df):,}"
    )

    # Lấy lại review_text cho train bằng merge
    train_df = train_idx_df.merge(
        df[["user_idx", "item_idx", "timestamp", "review_text"]],
        on=["user_idx", "item_idx", "timestamp"],
        how="left",
    )
    del train_idx_df
    gc.collect()

    # Xuất parquet — làm TRƯỚC khi del train_df
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["user_idx", "item_idx", "rating", "timestamp"]
    train_df[cols].to_parquet(out_dir / "train_ratings.parquet", index=False)
    val_df[cols].to_parquet(out_dir / "val_ratings.parquet", index=False)
    test_df[cols].to_parquet(out_dir / "test_ratings.parquet", index=False)

    # Ghi review pickle theo streaming (tránh OOM)
    build_review_docs_streaming(
        train_df=train_df,
        all_df=df,
        user_out_path=str(out_dir / "user_reviews.pkl"),
        item_out_path=str(out_dir / "item_reviews.pkl"),
        max_review_chars=500,
        max_doc_chars=20_000,
    )

    # Lưu encoder
    with open(out_dir / "encoders.pkl", "wb") as f:
        pickle.dump(
            {"user": user_enc, "item": item_enc}, f, protocol=pickle.HIGHEST_PROTOCOL
        )

    del train_df
    gc.collect()

    print(f"Đã xuất vào: {out_dir.resolve()}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Clean and export Amazon or Yelp review data."
    )
    parser.add_argument("--dataset", choices=("amazon", "yelp"), required=True)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--amazon-category", default="Movies_and_TV")
    parser.add_argument(
        "--yelp-file", type=Path, default=RAW / "yelp_academic_dataset_review.json"
    )
    args = parser.parse_args()

    if args.max_records is not None and args.max_records < 1:
        parser.error("--max-records must be a positive integer")

    if args.dataset == "amazon":
        raw_df = load_amazon_reviews_hf(
            category=args.amazon_category, max_records=args.max_records
        )
        clean_df = clean_and_deduplicate(normalize_amazon_schema(raw_df))
        process_dataset("amazon", clean_df, k=3, out_dir=PROCESSED / "amazon")
        return

    raw_df = load_yelp_reviews(str(args.yelp_file), max_records=args.max_records)
    clean_df = clean_and_deduplicate(raw_df)
    process_dataset("yelp", clean_df, k=5, out_dir=PROCESSED / "yelp")


if __name__ == "__main__":
    main()
