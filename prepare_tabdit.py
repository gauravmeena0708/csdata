import sys
import os
import pandas as pd
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

# Add csdata to path if not installed
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from csdata.download import fetch, fetch_health_heritage
from csdata.health_heritage import build_health_heritage_frame
from csdata.registry import load_spec
from csdata.transforms import apply_transform
from csdata.render import render_name
from csdata.prepare import _impute

def save_tabdit_format(df, out_path, num_cols, cat_cols, target_col):
    out_path.mkdir(parents=True, exist_ok=True)
    
    # TabDiT expects 'user' column
    df_copy = df.copy()
    df_copy['user'] = range(len(df_copy))
    df_copy['user'] = df_copy['user'].astype(str)
    
    # Parent table: user and label
    parent_df = df_copy[['user', target_col]].copy()
    parent_df.rename(columns={target_col: 'label'}, inplace=True)
    
    # Trans table: user and all features
    trans_cols = ['user'] + num_cols + cat_cols
    trans_df = df_copy[trans_cols].copy()
    
    # Ensure columns are string types for categorical, float/int for numerical if needed
    for c in cat_cols:
        trans_df[c] = trans_df[c].astype(str)
        
    parent_df.to_parquet(out_path / 'true_parent.parquet', index=False)
    trans_df.to_parquet(out_path / 'true_trans.parquet', index=False)
    
    print(f"Saved TabDiT format data to {out_path}")

def prepare_for_tabdit(name: str, out_dir: str):
    out_dir_path = Path(out_dir)
    
    spec = load_spec(name)
    
    if name == "health_heritage":
        raw_paths = fetch_health_heritage(spec.source["url"], name=name, cache_dir=None)
        df = build_health_heritage_frame(raw_paths)
    else:
        raw_path = fetch(spec.source["url"], name, None, spec.source.get("archive_member"))
        read_excel_opts = spec.source.get("read_excel")
        if read_excel_opts is not None or Path(raw_path).suffix.lower() in {".xls", ".xlsx"}:
            df = pd.read_excel(raw_path, **dict(read_excel_opts or {}))
        else:
            df = pd.read_csv(raw_path, **dict(spec.source.get("read_csv", {})))
            
    df = apply_transform(name, df)
    
    # Get column specs to know target and features
    info = render_name(spec, naming="real")
    num_cols = info["numerical_columns"]
    cat_cols = list(info.get("categorical_columns", []))
    target_col = info["target_column"]
    
    # Impute missing values as csdata does
    df = _impute(df, num_cols, cat_cols)
    
    # Split as per csdata standard (80/20, seed 42)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    
    # Save full, train, and test in TabDiT format
    save_tabdit_format(df, out_dir_path / "full", num_cols, cat_cols, target_col)
    save_tabdit_format(train_df, out_dir_path / "train", num_cols, cat_cols, target_col)
    save_tabdit_format(test_df, out_dir_path / "test", num_cols, cat_cols, target_col)
    
    # Save the standard info.json in the root out_dir
    (out_dir_path / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Finished processing {name}. Saved to {out_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python prepare_tabdit.py <dataset_name> <out_dir>")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    out_dir = sys.argv[2]
    prepare_for_tabdit(dataset_name, out_dir)
