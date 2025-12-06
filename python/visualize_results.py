"""
visualize_results.py
Generates visualizations for CSTR digital twin results and anomaly detection

Author: [Your Name]
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_results(filepath='../data/data_with_anomalies.csv'):
    """
    Load results with anomaly flags
    
    Args:
        filepath: Path to results CSV file
        
    Returns:
        DataFrame with results
    """
    print("Loading results...")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}\n"
                                "Please run anomaly_detection.py first")
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} data points")
    
    return df

def plot_temperature_control(df, save_path='plots/temperature_control.png'):
    """
    Plot reactor temperature with setpoint and anomalies
    
    Args:
        df: DataFrame with results
        save_path: Path to save figure
    """
    print("\nGenerating temperature control plot...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Temperature vs Setpoint
    ax1.plot(df['Time_min'], df['T_reactor_K'], 'b-', linewidth=1.5, label='Reactor Temperature')
    ax1.plot(df['Time_min'], df['Tsp_setpoint_K'], 'r--', linewidth=1.5, label='Setpoint')
    
    # Mark anomalies
    anomalies = df[df['anomaly_any']]
    if len(anomalies) > 0:
        ax1.scatter(anomalies['Time_min'], anomalies['T_reactor_K'], 
                   c='red', s=50, marker='x', label='Anomalies', zorder=5)
    
    ax1.set_xlabel('Time (min)', fontsize=11)
    ax1.set_ylabel('Temperature (K)', fontsize=11)
    ax1.set_title('Reactor Temperature Control with Anomaly Detection', fontsize=13, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Temperature Error
    ax2.plot(df['Time_min'], df['T_error'], 'g-', linewidth=1.5)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
    ax2.fill_between(df['Time_min'], -5, 5, alpha=0.2, color='green', label='Acceptable Range (±5K)')
    
    ax2.set_xlabel('Time (min)', fontsize=11)
    ax2.set_ylabel('Temperature Error (K)', fontsize=11)
    ax2.set_title('Temperature Deviation from Setpoint', fontsize=13, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    plt.close()

def plot_control_action(df, save_path='plots/control_action.png'):
    """
    Plot coolant flow rate and disturbances
    
    Args:
        df: DataFrame with results
        save_path: Path to save figure
    """
    print("Generating control action plot...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Coolant Flow Rate
    ax1.plot(df['Time_min'], df['Fc_coolant_L_per_min'], 'm-', linewidth=1.5)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=0.8, label='Min Limit')
    ax1.axhline(y=100, color='r', linestyle='--', linewidth=0.8, label='Max Limit')
    
    # Mark saturation events
    saturated = df[df['anomaly_saturation']]
    if len(saturated) > 0:
        ax1.scatter(saturated['Time_min'], saturated['Fc_coolant_L_per_min'],
                   c='red', s=30, marker='o', alpha=0.5, label='Saturation')
    
    ax1.set_xlabel('Time (min)', fontsize=11)
    ax1.set_ylabel('Flow Rate (L/min)', fontsize=11)
    ax1.set_title('Coolant Flow Rate (Manipulated Variable)', fontsize=13, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-5, 105])
    
    # Plot 2: Feed Temperature (Disturbance)
    ax2.plot(df['Time_min'], df['Tf_feed_K'], 'orange', linewidth=1.5)
    ax2.set_xlabel('Time (min)', fontsize=11)
    ax2.set_ylabel('Temperature (K)', fontsize=11)
    ax2.set_title('Feed Temperature (Disturbance)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    plt.close()

def plot_anomaly_analysis(df, save_path='plots/anomaly_analysis.png'):
    """
    Plot anomaly detection analysis
    
    Args:
        df: DataFrame with results
        save_path: Path to save figure
    """
    print("Generating anomaly analysis plot...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Z-score over time
    ax1.plot(df['Time_min'], df['T_zscore'], 'b-', linewidth=1, alpha=0.7)
    ax1.axhline(y=3, color='r', linestyle='--', linewidth=1, label='±3σ threshold')
    ax1.axhline(y=-3, color='r', linestyle='--', linewidth=1)
    ax1.fill_between(df['Time_min'], -3, 3, alpha=0.2, color='green')
    ax1.set_xlabel('Time (min)', fontsize=10)
    ax1.set_ylabel('Z-score', fontsize=10)
    ax1.set_title('Statistical Outlier Detection (3-Sigma Rule)', fontsize=11, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Rate of change
    ax2.plot(df['Time_min'], df['dT_dt'], 'g-', linewidth=1, alpha=0.7)
    ax2.axhline(y=5, color='r', linestyle='--', linewidth=1, label='±5 K/min threshold')
    ax2.axhline(y=-5, color='r', linestyle='--', linewidth=1)
    ax2.fill_between(df['Time_min'], -5, 5, alpha=0.2, color='green')
    ax2.set_xlabel('Time (min)', fontsize=10)
    ax2.set_ylabel('dT/dt (K/min)', fontsize=10)
    ax2.set_title('Rate-of-Change Anomaly Detection', fontsize=11, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Anomaly type distribution
    anomaly_counts = {
        'Statistical': df['anomaly_statistical'].sum(),
        'Rate': df['anomaly_rate'].sum(),
        'Residual': df['anomaly_residual'].sum(),
        'Saturation': df['anomaly_saturation'].sum()
    }
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    ax3.bar(anomaly_counts.keys(), anomaly_counts.values(), color=colors, edgecolor='black')
    ax3.set_ylabel('Count', fontsize=10)
    ax3.set_title('Anomaly Type Distribution', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for i, (key, value) in enumerate(anomaly_counts.items()):
        ax3.text(i, value + 0.5, str(value), ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Anomaly timeline
    anomaly_times = df[df['anomaly_any']]['Time_min'].values
    if len(anomaly_times) > 0:
        ax4.hist(anomaly_times, bins=20, color='red', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Time (min)', fontsize=10)
        ax4.set_ylabel('Anomaly Count', fontsize=10)
        ax4.set_title('Anomaly Frequency Over Time', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No Anomalies Detected', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Anomaly Frequency Over Time', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    plt.close()

def plot_performance_metrics(df, save_path='plots/performance_metrics.png'):
    """
    Plot control performance metrics
    
    Args:
        df: DataFrame with results
        save_path: Path to save figure
    """
    print("Generating performance metrics plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Temperature distribution
    ax1.hist(df['T_reactor_K'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(df['T_reactor_K'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {df["T_reactor_K"].mean():.2f} K')
    ax1.set_xlabel('Temperature (K)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Reactor Temperature Distribution', fontsize=13, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Error distribution
    ax2.hist(df['T_error'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='green', linestyle='--', linewidth=2, label='Zero Error')
    ax2.axvline(df['T_error'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {df["T_error"].mean():.2f} K')
    ax2.set_xlabel('Temperature Error (K)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Control Error Distribution', fontsize=13, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    plt.close()

def generate_summary_report(df):
    """
    Generate text summary report
    
    Args:
        df: DataFrame with results
    """
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    
    print("\nControl Performance:")
    print(f"  Mean Temperature: {df['T_reactor_K'].mean():.2f} K")
    print(f"  Std Dev Temperature: {df['T_reactor_K'].std():.2f} K")
    print(f"  Mean Error: {df['T_error'].mean():.2f} K")
    print(f"  RMS Error: {np.sqrt((df['T_error']**2).mean()):.2f} K")
    print(f"  Max Error: {df['T_error'].abs().max():.2f} K")
    
    print("\nCoolant Flow Statistics:")
    print(f"  Mean Flow: {df['Fc_coolant_L_per_min'].mean():.2f} L/min")
    print(f"  Min Flow: {df['Fc_coolant_L_per_min'].min():.2f} L/min")
    print(f"  Max Flow: {df['Fc_coolant_L_per_min'].max():.2f} L/min")
    
    print("\nAnomaly Detection Results:")
    print(f"  Total Anomalies: {df['anomaly_any'].sum()} ({100*df['anomaly_any'].sum()/len(df):.2f}%)")
    print(f"  Statistical Outliers: {df['anomaly_statistical'].sum()}")
    print(f"  Rate Anomalies: {df['anomaly_rate'].sum()}")
    print(f"  Residual Anomalies: {df['anomaly_residual'].sum()}")
    print(f"  Control Saturation: {df['anomaly_saturation'].sum()}")
    
    print("\n" + "="*60)

def main():
    """
    Main visualization pipeline
    """
    print("="*60)
    print("CSTR Digital Twin - Visualization")
    print("="*60)
    
    # Load results
    df = load_results()
    
    # Generate all plots
    plot_temperature_control(df)
    plot_control_action(df)
    plot_anomaly_analysis(df)
    plot_performance_metrics(df)
    
    # Generate summary report
    generate_summary_report(df)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("Check the 'plots/' directory for all figures")
    print("="*60)

if __name__ == "__main__":
    main()
