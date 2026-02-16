"""
analyze_pipelines.py - Comprehensive analysis of GP vs Transformer pipelines.

Combines functionality from:
- compare_pipelines.py: Side-by-side R² and dataset comparisons
- compare_training_characteristics.py: Training dynamics analysis
- Expected performance analysis based on training space differences

All outputs saved to a single directory.

Usage:
    .pixi/envs/default/bin/python analyze_pipelines.py \\
        --gp-dir active_learning_gp_output \\
        --transformer-dir active_learning_output \\
        --output-dir pipeline_analysis
"""

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def parse_log_for_metrics(log_path):
    """Extract iteration metrics from log file."""
    iterations = []
    al_r2, baseline_r2 = [], []
    al_train_loss, al_val_loss = [], []
    baseline_train_loss, baseline_val_loss = [], []
    n_train, n_val = [], []

    iter_re = re.compile(r"=== (?:Global|GP Active Learning) Iteration (\d+) ===")
    split_re = re.compile(r"Split: n_train=(\d+), n_val=(\d+)")
    gp_size_re = re.compile(r"Training set size:\s*(\d+),\s*Validation set size:\s*(\d+)")
    al_metrics_re = re.compile(r"AL metrics: train_loss=([\d.]+), val_loss=([\d.]+), R²=([-\d.]+)")
    base_metrics_re = re.compile(r"Baseline metrics: train_loss=([\d.]+), val_loss=([\d.]+), R²=([-\d.]+)")

    current_iter = None
    current_n_train, current_n_val = None, None

    with open(log_path) as f:
        for line in f:
            m = iter_re.search(line)
            if m:
                current_iter = int(m.group(1))
                current_n_train, current_n_val = None, None
                continue

            m = split_re.search(line) or gp_size_re.search(line)
            if m and current_iter is not None:
                current_n_train = int(m.group(1))
                current_n_val = int(m.group(2))
                continue

            m = al_metrics_re.search(line)
            if m and current_iter is not None:
                al_train_loss.append(float(m.group(1)))
                al_val_loss.append(float(m.group(2)))
                al_r2.append(float(m.group(3)))
                continue

            m = base_metrics_re.search(line)
            if m and current_iter is not None:
                baseline_train_loss.append(float(m.group(1)))
                baseline_val_loss.append(float(m.group(2)))
                baseline_r2.append(float(m.group(3)))

                iterations.append(current_iter)
                n_train.append(current_n_train)
                n_val.append(current_n_val)
                current_iter = None

    return pd.DataFrame({
        'iteration': iterations,
        'al_r2': al_r2,
        'baseline_r2': baseline_r2,
        'al_train_loss': al_train_loss,
        'al_val_loss': al_val_loss,
        'baseline_train_loss': baseline_train_loss,
        'baseline_val_loss': baseline_val_loss,
        'n_train': n_train,
        'n_val': n_val,
    })


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_r2_comparison_detailed(gp_df, trans_df, output_dir):
    """Comprehensive R² comparison plots."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Colors
    gp_color = '#1f77b4'
    gp_light = '#7eb3d6'
    trans_color = '#ff7f0e'
    trans_light = '#ffbb78'

    # Plot 1: R² progression (both AL and Baseline)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(gp_df['iteration'], gp_df['al_r2'],
             'o-', color=gp_color, linewidth=2.5, markersize=6,
             label='GP Active Learning', alpha=0.9)
    ax1.plot(gp_df['iteration'], gp_df['baseline_r2'],
             's--', color=gp_light, linewidth=2, markersize=5,
             label='GP Baseline', alpha=0.7)
    ax1.plot(trans_df['iteration'], trans_df['al_r2'],
             '^-', color=trans_color, linewidth=2.5, markersize=6,
             label='Transformer Active Learning', alpha=0.9)
    ax1.plot(trans_df['iteration'], trans_df['baseline_r2'],
             'd--', color=trans_light, linewidth=2, markersize=5,
             label='Transformer Baseline', alpha=0.7)

    ax1.set_xlabel('Iteration', fontsize=13)
    ax1.set_ylabel('R² Score (Physical Space)', fontsize=13)
    ax1.set_title('R² Score Progression: GP vs Transformer', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='best', ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Plot 2: AL vs Baseline comparison
    ax2 = fig.add_subplot(gs[1, 0])
    gp_advantage = np.array(gp_df['al_r2']) - np.array(gp_df['baseline_r2'])
    trans_advantage = np.array(trans_df['al_r2']) - np.array(trans_df['baseline_r2'])

    ax2.plot(gp_df['iteration'], gp_advantage,
             'o-', color=gp_color, linewidth=2.5, markersize=6,
             label='GP', alpha=0.9)
    ax2.plot(trans_df['iteration'], trans_advantage,
             '^-', color=trans_color, linewidth=2.5, markersize=6,
             label='Transformer', alpha=0.9)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.fill_between(gp_df['iteration'], 0, gp_advantage,
                      where=(gp_advantage > 0), alpha=0.2, color=gp_color)
    ax2.fill_between(trans_df['iteration'], 0, trans_advantage,
                      where=(trans_advantage > 0), alpha=0.2, color=trans_color)

    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('R² Difference (AL - Baseline)', fontsize=12)
    ax2.set_title('Active Learning Benefit', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Direct AL comparison
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(gp_df['iteration'], gp_df['al_r2'],
             'o-', color=gp_color, linewidth=2.5, markersize=7,
             label='GP AL', alpha=0.9)
    ax3.plot(trans_df['iteration'], trans_df['al_r2'],
             '^-', color=trans_color, linewidth=2.5, markersize=7,
             label='Transformer AL', alpha=0.9)

    # Highlight winner
    final_gp = gp_df['al_r2'].iloc[-1]
    final_trans = trans_df['al_r2'].iloc[-1]
    winner = "GP" if final_gp > final_trans else "Transformer"
    diff = abs(final_gp - final_trans)

    ax3.text(0.05, 0.95, f'Winner: {winner}\nΔR² = {diff:.4f}',
             transform=ax3.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Active Learning R²', fontsize=12)
    ax3.set_title('Active Learning Only Comparison', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)

    plt.savefig(output_dir / 'r2_comparison_comprehensive.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'r2_comparison_comprehensive.png'}")


def plot_dataset_growth(gp_df, trans_df, output_dir):
    """Plot dataset size growth."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Handle None values in dataset sizes
    gp_train = np.array([x if x is not None else 0 for x in gp_df['n_train']])
    gp_val = np.array([x if x is not None else 0 for x in gp_df['n_val']])
    trans_train = np.array([x if x is not None else 0 for x in trans_df['n_train']])
    trans_val = np.array([x if x is not None else 0 for x in trans_df['n_val']])

    gp_total = gp_train + gp_val
    trans_total = trans_train + trans_val

    gp_has_data = np.any(gp_total > 0)
    trans_has_data = np.any(trans_total > 0)

    if gp_has_data:
        ax.plot(gp_df['iteration'], gp_total,
                'o-', color='#1f77b4', linewidth=2.5, markersize=7,
                label='GP Pipeline', alpha=0.9)
    if trans_has_data:
        ax.plot(trans_df['iteration'], trans_total,
                '^-', color='#ff7f0e', linewidth=2.5, markersize=7,
                label='Transformer Pipeline', alpha=0.9)

    ax.set_xlabel('Iteration', fontsize=13)
    ax.set_ylabel('Total Dataset Size (Train + Val)', fontsize=13)
    ax.set_title('Dataset Growth Over Iterations', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)

    # Add annotations for final sizes
    if gp_has_data and gp_total[-1] > 0:
        ax.text(gp_df['iteration'].iloc[-1], gp_total[-1], f'  {int(gp_total[-1]):,}',
                va='center', fontsize=10, color='#1f77b4', fontweight='bold')
    if trans_has_data and trans_total[-1] > 0:
        ax.text(trans_df['iteration'].iloc[-1], trans_total[-1], f'  {int(trans_total[-1]):,}',
                va='center', fontsize=10, color='#ff7f0e', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_growth.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'dataset_growth.png'}")


def plot_loss_comparison(gp_df, trans_df, output_dir):
    """Plot training/validation loss comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # GP losses
    axes[0, 0].semilogy(gp_df['iteration'], gp_df['al_train_loss'],
                        'o-', label='AL Train', color='#1f77b4', linewidth=2, alpha=0.8)
    axes[0, 0].semilogy(gp_df['iteration'], gp_df['al_val_loss'],
                        's-', label='AL Val', color='#aec7e8', linewidth=2, alpha=0.8)
    axes[0, 0].set_title('GP Pipeline: Active Learning Losses', fontsize=13, fontweight='bold')
    axes[0, 0].set_ylabel('Loss (MSE, log scale)', fontsize=11)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(gp_df['iteration'], gp_df['baseline_train_loss'],
                        'o-', label='Baseline Train', color='#ff7f0e', linewidth=2, alpha=0.8)
    axes[0, 1].semilogy(gp_df['iteration'], gp_df['baseline_val_loss'],
                        's-', label='Baseline Val', color='#ffbb78', linewidth=2, alpha=0.8)
    axes[0, 1].set_title('GP Pipeline: Baseline Losses', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Transformer losses
    axes[1, 0].semilogy(trans_df['iteration'], trans_df['al_train_loss'],
                        '^-', label='AL Train', color='#1f77b4', linewidth=2, alpha=0.8)
    axes[1, 0].semilogy(trans_df['iteration'], trans_df['al_val_loss'],
                        'd-', label='AL Val', color='#aec7e8', linewidth=2, alpha=0.8)
    axes[1, 0].set_title('Transformer Pipeline: Active Learning Losses', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Iteration', fontsize=11)
    axes[1, 0].set_ylabel('Loss (MSE, log scale)', fontsize=11)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].semilogy(trans_df['iteration'], trans_df['baseline_train_loss'],
                        '^-', label='Baseline Train', color='#ff7f0e', linewidth=2, alpha=0.8)
    axes[1, 1].semilogy(trans_df['iteration'], trans_df['baseline_val_loss'],
                        'd-', label='Baseline Val', color='#ffbb78', linewidth=2, alpha=0.8)
    axes[1, 1].set_title('Transformer Pipeline: Baseline Losses', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Iteration', fontsize=11)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'loss_comparison.png'}")


def plot_training_efficiency(gp_df, trans_df, output_dir):
    """Plot training efficiency metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: R² improvement rate
    ax1 = axes[0]
    gp_r2_diff = np.diff(gp_df['al_r2'].values, prepend=0)
    trans_r2_diff = np.diff(trans_df['al_r2'].values, prepend=0)

    ax1.plot(gp_df['iteration'], gp_r2_diff,
             'o-', color='#1f77b4', linewidth=2, markersize=5, label='GP', alpha=0.8)
    ax1.plot(trans_df['iteration'], trans_r2_diff,
             '^-', color='#ff7f0e', linewidth=2, markersize=5, label='Transformer', alpha=0.8)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('ΔR² (improvement from previous iteration)', fontsize=12)
    ax1.set_title('R² Improvement Rate', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cumulative R² improvement
    ax2 = axes[1]
    gp_cumulative = np.cumsum(gp_r2_diff)
    trans_cumulative = np.cumsum(trans_r2_diff)

    ax2.plot(gp_df['iteration'], gp_cumulative,
             'o-', color='#1f77b4', linewidth=2.5, markersize=6, label='GP', alpha=0.9)
    ax2.plot(trans_df['iteration'], trans_cumulative,
             '^-', color='#ff7f0e', linewidth=2.5, markersize=6, label='Transformer', alpha=0.9)

    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Cumulative R² Gain', fontsize=12)
    ax2.set_title('Cumulative Learning Progress', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_efficiency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'training_efficiency.png'}")


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def create_comprehensive_report(gp_df, trans_df, output_dir):
    """Create comprehensive text summary."""
    report = []
    report.append("=" * 90)
    report.append("COMPREHENSIVE GP vs TRANSFORMER PIPELINE ANALYSIS")
    report.append("=" * 90)
    report.append("")

    gp_final = gp_df.iloc[-1]
    trans_final = trans_df.iloc[-1]

    # Section 1: Final Performance
    report.append("1. FINAL PERFORMANCE (Iteration 40)")
    report.append("-" * 90)
    report.append(f"{'Metric':<45} {'GP':<22} {'Transformer':<22}")
    report.append("-" * 90)
    report.append(f"{'Active Learning R²':<45} {gp_final['al_r2']:<22.6f} {trans_final['al_r2']:<22.6f}")
    report.append(f"{'Baseline R²':<45} {gp_final['baseline_r2']:<22.6f} {trans_final['baseline_r2']:<22.6f}")

    gp_advantage = gp_final['al_r2'] - gp_final['baseline_r2']
    trans_advantage = trans_final['al_r2'] - trans_final['baseline_r2']
    report.append(f"{'AL Advantage (AL - Baseline)':<45} {gp_advantage:<22.6f} {trans_advantage:<22.6f}")

    gp_final_size = (gp_final['n_train'] or 0) + (gp_final['n_val'] or 0)
    trans_final_size = (trans_final['n_train'] or 0) + (trans_final['n_val'] or 0)
    gp_size_str = f"{gp_final_size:,}" if gp_final_size > 0 else "N/A"
    trans_size_str = f"{trans_final_size:,}" if trans_final_size > 0 else "N/A"
    report.append(f"{'Final Dataset Size':<45} {gp_size_str:<22} {trans_size_str:<22}")

    report.append(f"{'Final Train Loss':<45} {gp_final['al_train_loss']:<22.6f} {trans_final['al_train_loss']:<22.6f}")
    report.append(f"{'Final Val Loss':<45} {gp_final['al_val_loss']:<22.6f} {trans_final['al_val_loss']:<22.6f}")
    report.append("")

    # Section 2: Winner Analysis
    report.append("2. WINNER ANALYSIS")
    report.append("-" * 90)
    if gp_final['al_r2'] > trans_final['al_r2']:
        diff = gp_final['al_r2'] - trans_final['al_r2']
        report.append(f"🏆 GP WINS with R² = {gp_final['al_r2']:.6f} (Δ = +{diff:.6f})")
    elif trans_final['al_r2'] > gp_final['al_r2']:
        diff = trans_final['al_r2'] - gp_final['al_r2']
        report.append(f"🏆 TRANSFORMER WINS with R² = {trans_final['al_r2']:.6f} (Δ = +{diff:.6f})")
    else:
        report.append("🤝 TIE - Both achieve the same R² score")

    # Active Learning effectiveness
    report.append("")
    if gp_advantage < 0:
        report.append(f"⚠️  GP: Active Learning UNDERPERFORMS baseline by {abs(gp_advantage):.6f}")
    else:
        report.append(f"✓  GP: Active Learning OUTPERFORMS baseline by {gp_advantage:.6f}")

    if trans_advantage < 0:
        report.append(f"⚠️  Transformer: Active Learning UNDERPERFORMS baseline by {abs(trans_advantage):.6f}")
    else:
        report.append(f"✓  Transformer: Active Learning OUTPERFORMS baseline by {trans_advantage:.6f}")
    report.append("")

    # Section 3: Training Space Differences
    report.append("3. TRAINING SPACE DIFFERENCES")
    report.append("-" * 90)
    report.append("GP Pipeline:")
    report.append("  • Training target space: LOG-TRANSFORMED space")
    report.append("    Formula: log(Y / 0.12) for DMRD")
    report.append("  • Optimization focus: RELATIVE ERRORS (multiplicative)")
    report.append("  • R² evaluation: Physical space (for fair comparison)")
    report.append("  • Expected strengths:")
    report.append("    - Better for small relic density values")
    report.append("    - Handles wide dynamic ranges well")
    report.append("    - More sample-efficient")
    report.append("    - Natural for multiplicative physics relationships")
    report.append("")
    report.append("Transformer Pipeline:")
    report.append("  • Training target space: Z-SCORE NORMALIZED space")
    report.append("    Formula: (Y - mean) / std")
    report.append("  • Optimization focus: ABSOLUTE ERRORS (additive)")
    report.append("  • R² evaluation: Physical space")
    report.append("  • Expected strengths:")
    report.append("    - Better for bulk of distribution (near mean)")
    report.append("    - Good for complex non-linear patterns with sufficient data")
    report.append("    - Better interpolation in well-sampled regions")
    report.append("")

    # Section 4: Data Efficiency
    report.append("4. DATA EFFICIENCY")
    report.append("-" * 90)
    if gp_final_size > 0 and trans_final_size > 0:
        gp_points_per_iter = (gp_final_size - 50000) / 40
        trans_points_per_iter = (trans_final_size - 50000) / 40

        report.append(f"Average points added per iteration:")
        report.append(f"  GP:          {gp_points_per_iter:>10.1f}")
        report.append(f"  Transformer: {trans_points_per_iter:>10.1f}")
        if gp_points_per_iter > 0:
            ratio = trans_points_per_iter / gp_points_per_iter
            report.append(f"  Ratio:       {ratio:>10.1f}x (Transformer uses {ratio:.1f}x more data)")
    else:
        report.append("Dataset size information not available in logs.")
    report.append("")

    # Section 5: Key Insights
    report.append("5. KEY INSIGHTS & INTERPRETATIONS")
    report.append("-" * 90)

    # Insight 1: Why does baseline outperform AL?
    if gp_advantage < 0 and trans_advantage < 0:
        report.append("CRITICAL FINDING: Both AL strategies underperform random baseline!")
        report.append("")
        report.append("Possible explanations:")
        report.append("  1. Active Learning explores 'difficult' regions that hurt generalization")
        report.append("  2. Uncertainty metrics don't correlate with model improvement")
        report.append("  3. Random sampling provides better coverage of parameter space")
        report.append("  4. Selection strategy (entropy_batch/top_k) may need tuning")
        report.append("")
        report.append("Recommendations:")
        report.append("  • Try different selection strategies (e.g., diversity-based)")
        report.append("  • Add exploration bonus to balance uncertainty exploitation")
        report.append("  • Consider stratified random sampling as baseline")
        report.append("  • Analyze which parameter regions AL is selecting")
        report.append("")

    # Insight 2: Training space impact
    if gp_final['al_r2'] > trans_final['al_r2']:
        report.append("LOG-SPACE TRAINING ADVANTAGE:")
        report.append("  GP's log-space training appears to better capture the relic density")
        report.append("  distribution structure, even when evaluated in physical space.")
        report.append("  This suggests relative error optimization is more effective for")
        report.append("  this physics problem than absolute error optimization.")
    else:
        report.append("DATA VOLUME COMPENSATES FOR TRAINING SPACE:")
        report.append("  Despite training in normalized space, Transformer achieves")
        report.append("  comparable or better performance, likely due to:")
        report.append(f"  - {trans_final_size:,} total samples vs GP's data")
        report.append("  - More expressive model architecture")
        report.append("  - Better optimization in high-dimensional space")

    report.append("")
    report.append("=" * 90)

    report_text = "\n".join(report)

    with open(output_dir / 'analysis_report.txt', 'w') as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\nSaved: {output_dir / 'analysis_report.txt'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive analysis of GP vs Transformer active learning pipelines."
    )
    parser.add_argument('--gp-dir', required=True, help='GP pipeline output directory')
    parser.add_argument('--transformer-dir', required=True, help='Transformer pipeline output directory')
    parser.add_argument('--output-dir', default='pipeline_analysis', help='Output directory')
    args = parser.parse_args()

    gp_dir = Path(args.gp_dir)
    trans_dir = Path(args.transformer_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("COMPREHENSIVE PIPELINE ANALYSIS")
    print("=" * 90)
    print(f"GP directory:          {gp_dir}")
    print(f"Transformer directory: {trans_dir}")
    print(f"Output directory:      {output_dir}")
    print()

    # Parse logs
    print("Parsing logs...")
    gp_log = gp_dir / 'active_learning.log'
    trans_log = trans_dir / 'active_learning.log'

    if not gp_log.exists():
        print(f"Error: GP log not found: {gp_log}")
        return
    if not trans_log.exists():
        print(f"Error: Transformer log not found: {trans_log}")
        return

    gp_df = parse_log_for_metrics(gp_log)
    trans_df = parse_log_for_metrics(trans_log)

    print(f"  GP:          {len(gp_df)} iterations parsed")
    print(f"  Transformer: {len(trans_df)} iterations parsed")
    print()

    # Generate all visualizations
    print("Generating visualizations...")
    print("-" * 90)
    plot_r2_comparison_detailed(gp_df, trans_df, output_dir)
    plot_dataset_growth(gp_df, trans_df, output_dir)
    plot_loss_comparison(gp_df, trans_df, output_dir)
    plot_training_efficiency(gp_df, trans_df, output_dir)

    # Generate report
    print()
    create_comprehensive_report(gp_df, trans_df, output_dir)

    # Save metrics
    gp_df.to_csv(output_dir / 'gp_metrics.csv', index=False)
    trans_df.to_csv(output_dir / 'transformer_metrics.csv', index=False)
    print(f"Saved metrics: {output_dir}/gp_metrics.csv, {output_dir}/transformer_metrics.csv")

    # Summary of outputs
    print()
    print("=" * 90)
    print("ANALYSIS COMPLETE!")
    print("=" * 90)
    print(f"All results saved to: {output_dir}/")
    print()
    print("Generated files:")
    print("  📊 r2_comparison_comprehensive.png - Detailed R² comparison")
    print("  📊 dataset_growth.png              - Dataset size over time")
    print("  📊 loss_comparison.png             - Training/validation losses")
    print("  📊 training_efficiency.png         - Learning rate and efficiency")
    print("  📄 analysis_report.txt             - Comprehensive text report")
    print("  📄 gp_metrics.csv                  - GP metrics table")
    print("  📄 transformer_metrics.csv         - Transformer metrics table")
    print("=" * 90)


if __name__ == "__main__":
    main()
