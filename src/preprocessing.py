# features.py
# Project: Deadlock Predictor - Deadlock-1.0
# Purpose: Novel feature engineering functions
# Author: Hari Krishnan M R
# Date: May 2026

import pandas as pd
import numpy as np


def calculate_scheduling_delay(df_clean):
    """
    Novel Feature 1: Scheduling Delay (seconds)
    
    Time between task SUBMIT and SCHEDULE events.
    Long delay means scheduler struggled to find
    a machine - early sign of resource contention.
    
    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned trace from preprocessing.py
        
    Returns
    -------
    pd.DataFrame
        job_id, task_index, scheduling_delay_sec
    """
    submit_times = (
        df_clean[df_clean['event_type'] == 0]
        .groupby(['job_id', 'task_index'])['timestamp']
        .min()
        .reset_index()
        .rename(columns={'timestamp': 'submit_time'})
    )

    schedule_times = (
        df_clean[df_clean['event_type'] == 1]
        .groupby(['job_id', 'task_index'])['timestamp']
        .min()
        .reset_index()
        .rename(columns={'timestamp': 'schedule_time'})
    )

    delay_df = submit_times.merge(
        schedule_times,
        on=['job_id', 'task_index'],
        how='inner'
    )

    delay_df['scheduling_delay_sec'] = (
        (delay_df['schedule_time'] - delay_df['submit_time'])
        / 1_000_000
    )

    # Remove negative delays (trace anomalies)
    delay_df = delay_df[
        delay_df['scheduling_delay_sec'] >= 0
    ]

    print(f"scheduling_delay_sec created: "
          f"{len(delay_df):,} tasks")
    return delay_df[
        ['job_id', 'task_index', 'scheduling_delay_sec']
    ]


def calculate_execution_duration(df_clean, df_label):
    """
    Novel Feature 2: Execution Duration (seconds)
    
    Time between SCHEDULE and final terminal event.
    Deadlock-like tasks show bimodal pattern:
    - Fast kills: immediate resource conflict
    - Slow stalls: deadlock-like waiting
    
    This is the single most important feature
    (causes 5.9% F1 drop when removed).
    
    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned trace from preprocessing.py
    df_label : pd.DataFrame
        Labelled tasks from create_labels()
        
    Returns
    -------
    pd.DataFrame
        job_id, task_index, execution_duration_sec
    """
    sched_times = (
        df_clean[df_clean['event_type'] == 1]
        .groupby(['job_id', 'task_index'])['timestamp']
        .min()
        .reset_index()
        .rename(columns={'timestamp': 'sched_time'})
    )

    final_times = (
        df_label[['job_id', 'task_index', 'timestamp']]
        .rename(columns={'timestamp': 'final_time'})
    )

    duration_df = sched_times.merge(
        final_times,
        on=['job_id', 'task_index'],
        how='inner'
    )

    duration_df['execution_duration_sec'] = (
        (duration_df['final_time'] - duration_df['sched_time'])
        / 1_000_000
    )

    # Remove negative durations
    duration_df = duration_df[
        duration_df['execution_duration_sec'] >= 0
    ]

    print(f"execution_duration_sec created: "
          f"{len(duration_df):,} tasks")
    return duration_df[
        ['job_id', 'task_index', 'execution_duration_sec']
    ]


def calculate_machine_failure_rate(df_clean):
    """
    Novel Feature 3: Machine Failure Rate
    
    Proportion of failure events per machine.
    Tasks on historically unreliable machines
    have higher deadlock risk regardless of their
    own attributes.
    
    Formula: (FAIL+KILL+EVICT on machine) / 
             (all events on machine)
    
    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned trace from preprocessing.py
        
    Returns
    -------
    pd.DataFrame
        machine_id, machine_failure_rate
    """
    # Only real machine assignments
    machine_df = df_clean[
        df_clean['machine_id'] != -1
    ].copy()

    total_per_machine = (
        machine_df
        .groupby('machine_id')['event_type']
        .count()
        .reset_index()
        .rename(columns={'event_type': 'total_events'})
    )

    # EVICT=2, FAIL=3, KILL=5
    failure_events = machine_df[
        machine_df['event_type'].isin([2, 3, 5])
    ]
    fail_per_machine = (
        failure_events
        .groupby('machine_id')['event_type']
        .count()
        .reset_index()
        .rename(columns={'event_type': 'failure_events'})
    )

    machine_stats = total_per_machine.merge(
        fail_per_machine,
        on='machine_id',
        how='left'
    )
    machine_stats['failure_events'] = (
        machine_stats['failure_events'].fillna(0)
    )
    machine_stats['machine_failure_rate'] = (
        machine_stats['failure_events'] /
        machine_stats['total_events']
    )

    print(f"machine_failure_rate created: "
          f"{len(machine_stats):,} machines")
    return machine_stats[
        ['machine_id', 'machine_failure_rate']
    ]


def calculate_concurrency_density(df_clean,
                                   window_seconds=300):
    """
    Novel Feature 4: Concurrency Density
    
    Number of tasks scheduled on same machine
    within a time window. High concurrency means
    more tasks competing for shared resources.
    
    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned trace from preprocessing.py
    window_seconds : int
        Time window size in seconds (default 300)
        
    Returns
    -------
    pd.DataFrame
        job_id, task_index, concurrency_density
    """
    scheduled = df_clean[
        (df_clean['event_type'] == 1) &
        (df_clean['machine_id'] != -1)
    ].copy()

    scheduled['timestamp_sec'] = (
        scheduled['timestamp'] / 1_000_000
    )
    scheduled['time_window'] = (
        scheduled['timestamp_sec'] // window_seconds
    ).astype(int)

    concurrency = (
        scheduled
        .groupby(['machine_id', 'time_window'])
        .size()
        .reset_index()
        .rename(columns={0: 'concurrency_density'})
    )

    scheduled = scheduled.merge(
        concurrency,
        on=['machine_id', 'time_window'],
        how='left'
    )

    concurrency_per_task = (
        scheduled
        .groupby(['job_id', 'task_index'])
        ['concurrency_density']
        .first()
        .reset_index()
    )

    print(f"concurrency_density created: "
          f"{len(concurrency_per_task):,} tasks")
    return concurrency_per_task


def calculate_resource_overcommit(df_clean):
    """
    Novel Feature 5: Resource Over-commitment Ratio
    
    How much CPU this task requested compared to
    the average on its machine.
    
    Formula: task cpu_requested / 
             mean(cpu_requested on machine)
    
    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned trace from preprocessing.py
        
    Returns
    -------
    pd.DataFrame
        job_id, task_index, resource_overcommit_ratio
    """
    sched_resources = df_clean[
        (df_clean['event_type'] == 1) &
        (df_clean['machine_id'] != -1) &
        (df_clean['cpu_requested'] > 0)
    ].copy()

    mean_cpu = (
        sched_resources
        .groupby('machine_id')['cpu_requested']
        .mean()
        .reset_index()
        .rename(columns={
            'cpu_requested': 'mean_cpu_on_machine'
        })
    )

    sched_resources = sched_resources.merge(
        mean_cpu,
        on='machine_id',
        how='left'
    )

    sched_resources['resource_overcommit_ratio'] = (
        sched_resources['cpu_requested'] /
        sched_resources['mean_cpu_on_machine']
    )

    overcommit_per_task = (
        sched_resources
        .groupby(['job_id', 'task_index'])
        ['resource_overcommit_ratio']
        .first()
        .reset_index()
    )

    print(f"resource_overcommit_ratio created: "
          f"{len(overcommit_per_task):,} tasks")
    return overcommit_per_task


def calculate_priority_gap(df_clean):
    """
    Novel Feature 6: Priority Gap
    
    Severity of priority inversion - how much lower
    this scheduled task's priority is compared to
    the highest priority task waiting in the same
    time window.
    
    A classic deadlock precursor: low priority tasks
    scheduled while high priority tasks wait.
    
    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned trace from preprocessing.py
        
    Returns
    -------
    pd.DataFrame
        job_id, task_index, priority_inversion_flag
    """
    waiting = df_clean[
        df_clean['event_type'] == 0
    ][['job_id', 'task_index', 'timestamp',
       'priority']].copy()

    waiting['timestamp_sec'] = (
        waiting['timestamp'] / 1_000_000
    )
    waiting['time_window'] = (
        waiting['timestamp_sec'] // 60
    ).astype(int)

    max_waiting_priority = (
        waiting
        .groupby('time_window')['priority']
        .max()
        .reset_index()
        .rename(columns={
            'priority': 'max_waiting_priority'
        })
    )

    sched_prio = df_clean[
        df_clean['event_type'] == 1
    ][['job_id', 'task_index', 'timestamp',
       'priority']].copy()

    sched_prio['timestamp_sec'] = (
        sched_prio['timestamp'] / 1_000_000
    )
    sched_prio['time_window'] = (
        sched_prio['timestamp_sec'] // 60
    ).astype(int)

    sched_prio = sched_prio.merge(
        max_waiting_priority,
        on='time_window',
        how='left'
    )

    sched_prio['priority_inversion_flag'] = (
        sched_prio['max_waiting_priority'] -
        sched_prio['priority']
    )

    priority_per_task = (
        sched_prio
        .groupby(['job_id', 'task_index'])
        ['priority_inversion_flag']
        .first()
        .reset_index()
    )

    print(f"priority_inversion_flag created: "
          f"{len(priority_per_task):,} tasks")
    return priority_per_task


def build_feature_matrix(df_label,
                          delay_df,
                          duration_df,
                          machine_stats,
                          concurrency_per_task,
                          overcommit_per_task,
                          priority_per_task):
    """
    Merge all features into final feature matrix.
    
    Combines 6 raw features and 6 novel features
    into one dataframe ready for model training.
    
    Parameters
    ----------
    df_label : pd.DataFrame
        Labelled tasks from create_labels()
    delay_df : pd.DataFrame
        From calculate_scheduling_delay()
    duration_df : pd.DataFrame
        From calculate_execution_duration()
    machine_stats : pd.DataFrame
        From calculate_machine_failure_rate()
    concurrency_per_task : pd.DataFrame
        From calculate_concurrency_density()
    overcommit_per_task : pd.DataFrame
        From calculate_resource_overcommit()
    priority_per_task : pd.DataFrame
        From calculate_priority_gap()
        
    Returns
    -------
    pd.DataFrame
        Final feature matrix with 12 features + label
    """
    RAW_FEATURES = [
        'job_id', 'task_index', 'machine_id',
        'scheduling_class', 'priority',
        'cpu_requested', 'memory_requested',
        'disk_requested', 'different_machine',
        'label'
    ]

    feature_matrix = df_label[RAW_FEATURES].copy()
    print(f"Base shape: {feature_matrix.shape}")

    # Merge novel features one by one
    merges = [
        (delay_df,
         ['job_id', 'task_index'],
         'scheduling_delay_sec'),
        (duration_df,
         ['job_id', 'task_index'],
         'execution_duration_sec'),
        (machine_stats,
         ['machine_id'],
         'machine_failure_rate'),
        (concurrency_per_task,
         ['job_id', 'task_index'],
         'concurrency_density'),
        (overcommit_per_task,
         ['job_id', 'task_index'],
         'resource_overcommit_ratio'),
        (priority_per_task,
         ['job_id', 'task_index'],
         'priority_inversion_flag'),
    ]

    for df, keys, col in merges:
        feature_matrix = feature_matrix.merge(
            df, on=keys, how='left'
        )
        print(f"After {col}: {feature_matrix.shape}")

    # Fill nulls with 0 for tasks missing features
    novel_cols = [
        'scheduling_delay_sec',
        'execution_duration_sec',
        'machine_failure_rate',
        'concurrency_density',
        'resource_overcommit_ratio',
        'priority_inversion_flag'
    ]
    feature_matrix[novel_cols] = (
        feature_matrix[novel_cols].fillna(0)
    )

    print(f"\nFinal shape  : {feature_matrix.shape}")
    print(f"Null check   : "
          f"{feature_matrix.isnull().sum().sum()}")
    return feature_matrix