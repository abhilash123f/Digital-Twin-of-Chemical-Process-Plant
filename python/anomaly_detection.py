"""
anomaly_detection.py
Detects anomalies in CSTR process data using statistical methods
Implements predictive maintenance alerts

Author: [Your Name]
Date: December 2025
"""

import pandas as pd
import numpy as np
import os

class AnomalyDetector:
    """
    Statistical anomaly detection for CSTR process data
    """
    
    def __init__(self, sigma_threshold=3, rate_threshold=5):
        """
        Initialize anomaly detector
        
        Args:
            sigma_threshold: Number of standard deviations for outlier detection
            rate_threshold: Maximum allowable rate of change (K/min)
        """
        self.sigma_threshold = sigma_threshold
        self.rate_threshold = rate_threshold
        self.anomalies = []
        
    def detect_statistical_outliers(self, df):
        """
        Detect outliers using 3-sigma rule
        
        Args:
            df: DataFrame with preprocessed data
            
        Returns:
            Boolean series indicating anomalies
        """
        print(f"\nDetecting statistical outliers (>{self.sigma_threshold}-sigma)...")
        
        # Use z-score calculated in preprocessing
        outliers = np.abs(df['T_zscore']) > self.sigma_threshold
        
        n_outliers = outliers.sum()
        print(f"Found {n_outliers} statistical outliers ({100*n_outliers/len(df):.2f}%)")
        
        return outliers
    
    def detect_rate_anomalies(self, df):
        """
        Detect anomalies based on rate of change
        
        Args:
            df: DataFrame with preprocessed data
            
        Returns:
            Boolean series indicating anomalies
        """
        print(f"\nDetecting rate-of-change anomalies (>|{self.rate_threshold}| K/min)...")
        
        # Check if temperature change rate exceeds threshold
        rate_anomalies = np.abs(df['dT_dt']) > self.rate_threshold
        
        n_rate_anomalies = rate_anomalies.sum()
        print(f"Found {n_rate_anomalies} rate anomalies ({100*n_rate_anomalies/len(df):.2f}%)")
        
        return rate_anomalies
    
    def detect_residual_anomalies(self, df, tolerance=5):
        """
        Detect anomalies based on deviation from setpoint
        
        Args:
            df: DataFrame with preprocessed data
            tolerance: Maximum allowable deviation from setpoint (K)
            
        Returns:
            Boolean series indicating anomalies
        """
        print(f"\nDetecting residual anomalies (>|{tolerance}| K from setpoint)...")
        
        # Check if temperature error exceeds tolerance
        residual_anomalies = np.abs(df['T_error']) > tolerance
        
        n_residual = residual_anomalies.sum()
        print(f"Found {n_residual} residual anomalies ({100*n_residual/len(df):.2f}%)")
        
        return residual_anomalies
    
    def detect_control_saturation(self, df):
        """
        Detect when controller is saturated (at limits)
        
        Args:
            df: DataFrame with preprocessed data
            
        Returns:
            Boolean series indicating saturation
        """
        print("\nDetecting control saturation...")
        
        # Check if coolant flow is at min (0) or max (100)
        saturation = (df['Fc_coolant_L_per_min'] <= 0.1) | (df['Fc_coolant_L_per_min'] >= 99.9)
        
        n_saturation = saturation.sum()
        print(f"Found {n_saturation} saturation events ({100*n_saturation/len(df):.2f}%)")
        
        return saturation
    
    def combine_anomalies(self, df):
        """
        Combine all anomaly detection methods
        
        Args:
            df: DataFrame with preprocessed data
            
        Returns:
            DataFrame with anomaly flags
        """
        print("\n" + "="*60)
        print("Running all anomaly detection methods...")
        print("="*60)
        
        # Run all detection methods
        df['anomaly_statistical'] = self.detect_statistical_outliers(df)
        df['anomaly_rate'] = self.detect_rate_anomalies(df)
        df['anomaly_residual'] = self.detect_residual_anomalies(df, tolerance=5)
        df['anomaly_saturation'] = self.detect_control_saturation(df)
        
        # Combined anomaly flag (any method triggers)
        df['anomaly_any'] = (df['anomaly_statistical'] | 
                             df['anomaly_rate'] | 
                             df['anomaly_residual'] | 
                             df['anomaly_saturation'])
        
        n_total = df['anomaly_any'].sum()
        print(f"\n{'='*60}")
        print(f"Total anomalies detected: {n_total} ({100*n_total/len(df):.2f}%)")
        print(f"{'='*60}")
        
        return df
    
    def generate_anomaly_log(self, df):
        """
        Generate detailed log of anomalies
        
        Args:
            df: DataFrame with anomaly flags
            
        Returns:
            DataFrame with anomaly details
        """
        print("\nGenerating anomaly log...")
        
        # Filter to only anomalous points
        anomaly_df = df[df['anomaly_any']].copy()
        
        if len(anomaly_df) == 0:
            print("No anomalies to log")
            return pd.DataFrame()
        
        # Determine anomaly type
        def get_anomaly_type(row):
            types = []
            if row['anomaly_statistical']:
                types.append('Statistical Outlier')
            if row['anomaly_rate']:
                types.append('Rate Anomaly')
            if row['anomaly_residual']:
                types.append('Residual Anomaly')
            if row['anomaly_saturation']:
                types.append('Control Saturation')
            return ', '.join(types)
        
        anomaly_df['anomaly_type'] = anomaly_df.apply(get_anomaly_type, axis=1)
        
        # Select relevant columns for log
        log_columns = ['Time_min', 'T_reactor_K', 'Tsp_setpoint_K', 'T_error', 
                       'dT_dt', 'Fc_coolant_L_per_min', 'anomaly_type']
        
        anomaly_log = anomaly_df[log_columns].copy()
        
        print(f"Anomaly log created with {len(anomaly_log)} entries")
        
        return anomaly_log

def load_preprocessed_data(filepath='../data/preprocessed_data.csv'):
    """
    Load preprocessed data
    
    Args:
        filepath: Path to preprocessed CSV file
        
    Returns:
        DataFrame with preprocessed data
    """
    print("Loading preprocessed data...")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Preprocessed data not found: {filepath}\n"
                                "Please run data_preprocessing.py first")
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} data points")
    
    return df

def save_results(df, anomaly_log):
    """
    Save anomaly detection results
    
    Args:
        df: DataFrame with anomaly flags
        anomaly_log: DataFrame with anomaly details
    """
    print("\nSaving results...")
    
    # Save full data with anomaly flags
    df.to_csv('../data/data_with_anomalies.csv', index=False)
    print("Saved: ../data/data_with_anomalies.csv")
    
    # Save anomaly log
    if len(anomaly_log) > 0:
        anomaly_log.to_csv('../data/anomaly_log.csv', index=False)
        print("Saved: ../data/anomaly_log.csv")
    else:
        print("No anomalies to save")

def main():
    """
    Main anomaly detection pipeline
    """
    print("="*60)
    print("CSTR Digital Twin - Anomaly Detection")
    print("="*60)
    
    # Load preprocessed data
    df = load_preprocessed_data()
    
    # Initialize detector
    detector = AnomalyDetector(sigma_threshold=3, rate_threshold=5)
    
    # Run anomaly detection
    df = detector.combine_anomalies(df)
    
    # Generate anomaly log
    anomaly_log = detector.generate_anomaly_log(df)
    
    # Save results
    save_results(df, anomaly_log)
    
    # Display sample anomalies
    if len(anomaly_log) > 0:
        print("\n" + "="*60)
        print("Sample Anomalies (first 5)")
        print("="*60)
        print(anomaly_log.head().to_string())
    
    # Summary statistics
    print("\n" + "="*60)
    print("Anomaly Detection Summary")
    print("="*60)
    print(f"Total data points: {len(df)}")
    print(f"Statistical outliers: {df['anomaly_statistical'].sum()}")
    print(f"Rate anomalies: {df['anomaly_rate'].sum()}")
    print(f"Residual anomalies: {df['anomaly_residual'].sum()}")
    print(f"Control saturation: {df['anomaly_saturation'].sum()}")
    print(f"Total unique anomalies: {df['anomaly_any'].sum()}")
    
    print("\n" + "="*60)
    print("Anomaly detection complete!")
    print("Next step: Run visualize_results.py")
    print("="*60)

if __name__ == "__main__":
    main()
