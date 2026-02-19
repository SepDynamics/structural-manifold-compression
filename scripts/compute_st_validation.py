import pandas as pd
import numpy as np
import json
import os

def load_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            flat_row = {
                'instrument': row['instrument'],
                'ts': row['ts'],
                'regime': row['regime'],
                'coherence': row['coherence'],
                'entropy': row['entropy'],
                'repetitions': row['repetitions'],
                'roc_prev_pct': row['roc_prev_pct'],
                'roc_forward_60': row['roc_forward_pct'].get('60', np.nan),
                'roc_forward_240': row['roc_forward_pct'].get('240', np.nan),
            }
            data.append(flat_row)
    df = pd.DataFrame(data)
    # Using k=1.0 for decay constant lambda as suggested in draft
    df['ST'] = df['repetitions'] * df['coherence'] * np.exp(-1.0 * df['entropy'])
    return df

def analyze_instrument(df, instrument):
    print(f"\n{'='*50}\nAnalyzing {instrument}\n{'='*50}")
    
    df = df.dropna(subset=['roc_forward_60', 'roc_forward_240'])
    prev_signs = np.sign(df['roc_prev_pct'])
    
    for hor in [60, 240]:
        df[f'aligned_roc_{hor}'] = -prev_signs * df[f'roc_forward_{hor}']
        df[f'reversal_{hor}'] = df[f'aligned_roc_{hor}'] > 0

    st_50 = df['ST'].quantile(0.50)
    st_75 = df['ST'].quantile(0.75)
    st_90 = df['ST'].quantile(0.90)
    print(f"ST 50th percentile: {st_50:.6f}")
    print(f"ST 75th percentile: {st_75:.6f}")
    print(f"ST 90th percentile: {st_90:.6f}")

    # Baseline: regime == 'mean_revert'
    baseline_mask = df['regime'] == 'mean_revert'
    print("\n--- Baseline: Regime = 'mean_revert' ---")
    b_df = df[baseline_mask]
    print(f"Count: {len(b_df)}")
    if len(b_df) > 0:
        for hor in [60, 240]:
            mean_roc = b_df[f'aligned_roc_{hor}'].mean() * 100
            win_rate = b_df[f'reversal_{hor}'].mean() * 100
            print(f"Horizon {hor}m | Aligned ROC: {mean_roc:.4f}% | Reversal Rate: {win_rate:.2f}%")

    for pct_name, st_thresh in [('50th', st_50), ('75th', st_75), ('90th', st_90)]:
        print(f"\n--- ST > {pct_name} percentile ({st_thresh:.6f}) ---")
        for min_rep in [1, 2, 3]:
            mask = (df['ST'] > st_thresh) & (df['repetitions'] >= min_rep)
            s_df = df[mask]
            print(f"\nMin Reps: {min_rep} (Count: {len(s_df)})")
            if len(s_df) == 0:
                continue
            for hor in [60, 240]:
                mean_roc = s_df[f'aligned_roc_{hor}'].mean() * 100
                win_rate = s_df[f'reversal_{hor}'].mean() * 100
                print(f"  Horizon {hor}m | Aligned ROC: {mean_roc:.4f}% | Reversal Rate: {win_rate:.2f}%")

if __name__ == '__main__':
    data_dir = '/sep/structural-manifold-compression/data/ml_corpus/docs/evidence/roc_history/gameplan/subsets/'
    eur_file = os.path.join(data_dir, 'gates_2020-11-13_2021-04-09_EUR_USD.jsonl')
    jpy_file = os.path.join(data_dir, 'gates_2020-11-13_2021-04-09_USD_JPY.jsonl')
    
    if os.path.exists(eur_file):
        analyze_instrument(load_data(eur_file), 'EUR_USD')
    if os.path.exists(jpy_file):
        analyze_instrument(load_data(jpy_file), 'USD_JPY')

