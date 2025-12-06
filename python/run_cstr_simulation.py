"""
run_cstr_simulation.py
Pure Python implementation of CSTR simulation (alternative to MATLAB)
Runs the complete simulation and generates data for anomaly detection

Author: [Your Name]
Date: December 2025
"""

import numpy as np
from scipy.integrate import odeint
import pandas as pd
import matplotlib.pyplot as plt
import os

# Parameters
class CSTRParams:
    """CSTR process parameters"""
    # Reactor Parameters
    V = 100              # Reactor volume (L)
    rho = 1000           # Density (g/L)
    Cp = 0.239           # Heat capacity (J/g·K)
    dH = -5e4            # Heat of reaction (J/mol) - exothermic
    E = 11843            # Activation energy (J/mol)
    k0 = 7.2e10          # Pre-exponential factor (1/min)
    R = 8.314            # Gas constant (J/mol·K)
    
    # Heat Transfer
    UA = 5e4             # Overall heat transfer coefficient × Area (J/min·K)
    
    # Cooling Jacket
    Vc = 10              # Jacket volume (L)
    rhoc = 1000          # Coolant density (g/L)
    Cpc = 0.239          # Coolant heat capacity (J/g·K)
    
    # Operating Conditions
    F = 100              # Feed flow rate (L/min)
    Tf_nom = 350         # Nominal feed temperature (K)
    CAf_nom = 1.0        # Nominal feed concentration (mol/L)
    Tcf = 350            # Coolant feed temperature (K)
    Tsp = 350            # Temperature setpoint (K)
    
    # Initial Conditions
    T0 = 350             # Initial reactor temperature (K)
    Tc0 = 350            # Initial coolant temperature (K)
    CA0 = 0.5            # Initial concentration (mol/L)
    
    # Controller Parameters
    Kp = 10              # Proportional gain
    Ki = 0.5             # Integral gain
    Kd = 2               # Derivative gain
    Kff = 0.8            # Feedforward gain
    
    # Constraints
    Fc_min = 0           # Minimum coolant flow (L/min)
    Fc_max = 100         # Maximum coolant flow (L/min)
    
    # Simulation
    t_final = 100        # Simulation time (min)
    dt = 0.1             # Time step (min)
    
    # Disturbances
    t_setpoint_change = 10
    Tsp_new = 360
    t_Tf_disturbance = 30
    Tf_disturbed = 360
    t_CAf_disturbance = 50
    CAf_disturbed = 1.2

params = CSTRParams()

def cstr_model(x, t, Fc, Tf, CAf):
    """
    CSTR differential equations
    x = [T, Tc, CA]
    """
    T, Tc, CA = x
    
    # Reaction rate (Arrhenius)
    k = params.k0 * np.exp(-params.E / (params.R * T))
    r = k * CA
    
    # Reactor temperature dynamics
    Q_in = params.F * params.rho * params.Cp * (Tf - T) / (params.V * params.rho * params.Cp)
    Q_gen = (-params.dH) * r / (params.rho * params.Cp)
    Q_transfer = params.UA * (T - Tc) / (params.V * params.rho * params.Cp)
    dT_dt = Q_in + Q_gen - Q_transfer
    
    # Coolant temperature dynamics
    Q_coolant_flow = Fc * params.rhoc * params.Cpc * (params.Tcf - Tc) / (params.Vc * params.rhoc * params.Cpc)
    Q_coolant_transfer = params.UA * (T - Tc) / (params.Vc * params.rhoc * params.Cpc)
    dTc_dt = Q_coolant_flow + Q_coolant_transfer
    
    # Concentration dynamics
    dCA_flow = params.F * (CAf - CA) / params.V
    dCA_reaction = -r
    dCA_dt = dCA_flow + dCA_reaction
    
    return [dT_dt, dTc_dt, dCA_dt]

def run_simulation():
    """Run complete CSTR simulation with PID + Feedforward control"""
    
    print("="*60)
    print("CSTR Digital Twin Simulation (Python)")
    print("="*60)
    
    # Time vector
    t_span = np.arange(0, params.t_final + params.dt, params.dt)
    n_steps = len(t_span)
    
    # Initialize storage
    T_history = np.zeros(n_steps)
    Tc_history = np.zeros(n_steps)
    CA_history = np.zeros(n_steps)
    Fc_history = np.zeros(n_steps)
    Tf_history = np.zeros(n_steps)
    CAf_history = np.zeros(n_steps)
    Tsp_history = np.zeros(n_steps)
    
    # Initial state
    x = [params.T0, params.Tc0, params.CA0]
    error_integral = 0
    error_prev = 0
    
    print("\nRunning simulation...")
    
    for i, t in enumerate(t_span):
        # Progress
        if i % (n_steps // 10) == 0:
            print(f"Progress: {100*i//n_steps}%")
        
        # Disturbances
        Tsp = params.Tsp_new if t >= params.t_setpoint_change else params.Tsp
        Tf = params.Tf_disturbed if t >= params.t_Tf_disturbance else params.Tf_nom
        CAf = params.CAf_disturbed if t >= params.t_CAf_disturbance else params.CAf_nom
        
        # PID Controller
        T_measured = x[0]
        error = Tsp - T_measured
        
        # P term
        P_term = params.Kp * error
        
        # I term with anti-windup
        error_integral += error * params.dt
        error_integral = np.clip(error_integral, -100, 100)
        I_term = params.Ki * error_integral
        
        # D term
        error_derivative = (error - error_prev) / params.dt
        D_term = params.Kd * error_derivative
        
        u_pid = P_term + I_term + D_term
        
        # Feedforward Controller
        Tf_deviation = Tf - params.Tf_nom
        u_ff = params.Kff * Tf_deviation
        
        # Combined control with saturation
        Fc = np.clip(u_pid + u_ff, params.Fc_min, params.Fc_max)
        
        # Store values
        T_history[i] = x[0]
        Tc_history[i] = x[1]
        CA_history[i] = x[2]
        Fc_history[i] = Fc
        Tf_history[i] = Tf
        CAf_history[i] = CAf
        Tsp_history[i] = Tsp
        
        # Integrate to next step
        if i < n_steps - 1:
            t_next = [t, t + params.dt]
            x_next = odeint(cstr_model, x, t_next, args=(Fc, Tf, CAf))
            x = x_next[-1]
        
        error_prev = error
    
    print("Progress: 100%")
    print("\nSimulation complete!")
    
    # Create DataFrame
    results = pd.DataFrame({
        'Time_min': t_span,
        'T_reactor_K': T_history,
        'T_coolant_K': Tc_history,
        'CA_mol_per_L': CA_history,
        'Fc_coolant_L_per_min': Fc_history,
        'Tf_feed_K': Tf_history,
        'CAf_feed_mol_per_L': CAf_history,
        'Tsp_setpoint_K': Tsp_history
    })
    
    # Save to CSV
    output_file = '../data/simulation_output.csv'
    results.to_csv(output_file, index=False)
    print(f"\nData exported to: {output_file}")
    
    # Calculate performance metrics
    print("\n" + "="*60)
    print("Control Performance Metrics")
    print("="*60)
    
    # Find setpoint change response
    idx_sp = np.where(t_span >= params.t_setpoint_change)[0][0]
    T_after = T_history[idx_sp:]
    
    # Overshoot
    T_max = np.max(T_after)
    overshoot = ((T_max - params.Tsp_new) / params.Tsp_new) * 100
    print(f"Overshoot: {overshoot:.2f}%")
    
    # Settling time (2% band)
    settling_band = 0.02 * params.Tsp_new
    settled = np.where(np.abs(T_after - params.Tsp_new) <= settling_band)[0]
    if len(settled) > 0:
        settling_time = t_span[idx_sp + settled[0]] - params.t_setpoint_change
        print(f"Settling Time (2% band): {settling_time:.2f} min")
    
    # Generate plots
    print("\nGenerating plots...")
    generate_plots(results)
    
    print("\n" + "="*60)
    print("Ready for Python anomaly detection!")
    print("Next step: python data_preprocessing.py")
    print("="*60)
    
    return results

def generate_plots(df):
    """Generate simulation result plots"""
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # Plot 1: Reactor Temperature
    axes[0, 0].plot(df['Time_min'], df['T_reactor_K'], 'b-', linewidth=1.5, label='T_reactor')
    axes[0, 0].plot(df['Time_min'], df['Tsp_setpoint_K'], 'r--', linewidth=1.5, label='Setpoint')
    axes[0, 0].set_xlabel('Time (min)')
    axes[0, 0].set_ylabel('Temperature (K)')
    axes[0, 0].set_title('Reactor Temperature vs Setpoint')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Coolant Temperature
    axes[0, 1].plot(df['Time_min'], df['T_coolant_K'], 'g-', linewidth=1.5)
    axes[0, 1].set_xlabel('Time (min)')
    axes[0, 1].set_ylabel('Temperature (K)')
    axes[0, 1].set_title('Coolant Temperature')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Coolant Flow Rate
    axes[1, 0].plot(df['Time_min'], df['Fc_coolant_L_per_min'], 'm-', linewidth=1.5)
    axes[1, 0].set_xlabel('Time (min)')
    axes[1, 0].set_ylabel('Flow Rate (L/min)')
    axes[1, 0].set_title('Coolant Flow Rate (Control Action)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Concentration
    axes[1, 1].plot(df['Time_min'], df['CA_mol_per_L'], 'c-', linewidth=1.5)
    axes[1, 1].set_xlabel('Time (min)')
    axes[1, 1].set_ylabel('Concentration (mol/L)')
    axes[1, 1].set_title('Reactant Concentration')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: Feed Temperature
    axes[2, 0].plot(df['Time_min'], df['Tf_feed_K'], 'orange', linewidth=1.5)
    axes[2, 0].set_xlabel('Time (min)')
    axes[2, 0].set_ylabel('Temperature (K)')
    axes[2, 0].set_title('Feed Temperature (Disturbance)')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 6: Feed Concentration
    axes[2, 1].plot(df['Time_min'], df['CAf_feed_mol_per_L'], 'k-', linewidth=1.5)
    axes[2, 1].set_xlabel('Time (min)')
    axes[2, 1].set_ylabel('Concentration (mol/L)')
    axes[2, 1].set_title('Feed Concentration (Disturbance)')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.suptitle('CSTR Digital Twin - Simulation Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    os.makedirs('../data', exist_ok=True)
    plt.savefig('../data/simulation_results.png', dpi=300, bbox_inches='tight')
    print("Plots saved to: ../data/simulation_results.png")
    plt.close()

if __name__ == "__main__":
    results = run_simulation()
