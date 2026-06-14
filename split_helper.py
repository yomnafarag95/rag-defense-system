"""
split_helper.py
───────────────
Deterministic, hash-based dataset split helper for RAG-Shield.
Guarantees train/test splits are disjoint and reproducible, regardless
of ordering, duplicate templates, or sorting.
"""
import hashlib

def get_split(text: str, test_ratio: float = 0.10) -> str:
    """
    Deterministically partition a text string into 'train' or 'test' splits
    based on its MD5 hash. Ensures identical text always maps to the same split.
    
    test_ratio: proportion of text space mapped to test set.
    """
    if not text:
        return "train"
    h = hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()
    # Convert first 8 hex characters to integer (values between 0 and 2^32 - 1)
    val = int(h[:8], 16)
    limit = int(test_ratio * 4294967295) # 2^32 - 1
    return "test" if val < limit else "train"
