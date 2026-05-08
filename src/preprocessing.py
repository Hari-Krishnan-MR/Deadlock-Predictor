# preprocessing.py
# Project: Deadlock Predictor - Deadlock-1.0
# Purpose: Data loading and cleaning pipeline
# Author: Hari Krishnan M R
# Date: May 2026

import pandas as pd
import numpy as np
import os

COLUMNS = [
    'timestamp',
    'missing_info',
    'job_id',
    'task_index',
    'machine_id',
    'event_type',
    'user',
    'scheduling_class',
    'priority',
    'cpu_requested',
    'memory_requested',
    'disk_requested',
    'different_machine'
]

EVENT_TYPES = {
    0: 'SUBMIT',
    1: 'SCHEDULE',
    2: 'EVICT',
    3: 'FAIL',
    4: 'FINISH',
    5: 'KILL',
    6: 'LOST',
    7: 'UPDATE_PENDING',
    8: 'UPDATE_RUNNING'
}

FAILURE_EVENTS  = [2, 3, 5]
NORMAL_EVENTS   = [4]
TERMINAL_EVENTS = FAILURE_EVENTS + NORMAL_EVENTS


def load_trace_parts(path, n_parts=10):
    """
    Load n_parts CSV files from Google Cluster Trace.

    Parameters
    ----------
    path : str
        Folder containing CSV files
    n_parts : int
        Number of parts to load (default 10)

    Returns
    -------
    pd.DataFrame
        Combined raw dataframe
    """
    files = sorted([
        f for f in os.listdir(path)
        if f.endswith('.csv')
    ])[:n_parts]

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {path}"
        )

    dfs = []
    for file in files:
        filepath = os.path.join(path, file)
        df = pd.read_csv(
            filepath,
            header=None,
            names=COLUMNS
        )
        dfs.append(df)
        print(f"  Loaded {file}: {len(df):,} rows")

    df_raw = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows loaded: {len(df_raw):,}")
    return df_raw


def clean_trace(df_raw):
    """
    Apply all cleaning steps to raw trace.

    Steps:
    1. Drop missing_info (99.98% null)
    2. Remove duplicates
    3. Drop null job_id or task_index
    4. Fill machine_id nulls with -1
    5. Forward-fill resource columns
    6. Add event_label column

    Parameters
    ----------
    df_raw : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with zero nulls
    """
    df = df_raw.copy()
    print(f"Starting shape: {df.shape}")

    # Step 1 — Drop missing_info
    df = df.drop(columns=['missing_info'])

    # Step 2 — Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"Duplicates removed: {before - len(df):,}")

    # Step 3 — Drop null IDs
    before = len(df)
    df = df.dropna(subset=['job_id', 'task_index'])
    print(f"Rows dropped (null IDs): {before - len(df):,}")

    # Step 4 — Fill machine_id nulls
    df['machine_id'] = (
        df['machine_id'].fillna(-1).astype('int64')
    )

    # Step 5 — Forward fill resource columns
    resource_cols = [
        'cpu_requested',
        'memory_requested',
        'disk_requested',
        'different_machine'
    ]
    df = df.sort_values(
        ['job_id', 'task_index', 'timestamp']
    )
    for col in resource_cols:
        df[col] = (
            df.groupby(['job_id', 'task_index'])[col]
            .ffill()
            .fillna(0)
        )

    # Step 6 — Add event label
    df['event_label'] = df['event_type'].map(EVENT_TYPES)

    print(f"Final shape     : {df.shape}")
    print(f"Remaining nulls : {df.isnull().sum().sum()}")
    return df


def create_labels(df_clean):
    """
    Create binary labels from cleaned trace.

    label=1 : EVICT, FAIL, or KILL
    label=0 : FINISH

    Parameters
    ----------
    df_clean : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        One row per task with binary label
    """
    df_sorted = df_clean.sort_values('timestamp')
    final_events = (
        df_sorted
        .groupby(['job_id', 'task_index'])
        .last()
        .reset_index()
    )

    print(f"Unique tasks: {len(final_events):,}")

    # Keep only terminal events
    final_events = final_events[
        final_events['event_type'].isin(TERMINAL_EVENTS)
    ]
    print(f"Terminal tasks: {len(final_events):,}")

    # Create binary label
    final_events['label'] = (
        final_events['event_type']
        .isin(FAILURE_EVENTS)
        .astype(int)
    )

    counts = final_events['label'].value_counts()
    print(f"Label 0: {counts.get(0,0):,}")
    print(f"Label 1: {counts.get(1,0):,}")

    return final_events


def get_missing_summary(df):
    """
    Return missing value summary.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    missing_count = df.isnull().sum()
    missing_pct   = (
        missing_count / len(df) * 100
    ).round(2)

    return pd.DataFrame({
        'Missing Count' : missing_count,
        'Missing %'     : missing_pct
    })