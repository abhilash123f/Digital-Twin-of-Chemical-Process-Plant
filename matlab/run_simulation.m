% run_simulation.m
% Main simulation script for CSTR digital twin
% Runs the process model with PID + Feedforward control
% Exports data to CSV for Python anomaly detection
%
% Author: [Your Name]
% Date: December 2025

%% Initialize
clear all;
close all;
clc;

fprintf('=== CSTR Digital Twin Simulation ===\n\n');

%% Load parameters
cstr_parameters;

%% Simulation setup
t_span = 0:params.dt:params.t_final;
n_steps = length(t_span);

% Initialize state vector
x0 = [params.T0; params.Tc0; params.CA0];

% Initialize storage arrays
T_history = zeros(n_steps, 1);
Tc_history = zeros(n_steps, 1);
CA_history = zeros(n_steps, 1);
Fc_history = zeros(n_steps, 1);
Tf_history = zeros(n_steps, 1);
CAf_history = zeros(n_steps, 1);
Tsp_history = zeros(n_steps, 1);
error_integral = 0;
error_prev = 0;

%% Simulation loop
fprintf('Running simulation...\n');
fprintf('Progress: ');

x = x0;  % Current state

for i = 1:n_steps
    t = t_span(i);
    
    % Progress indicator
    if mod(i, floor(n_steps/10)) == 0
        fprintf('%d%% ', round(100*i/n_steps));
    end
    
    %% Define disturbances based on time
    % Setpoint changes
    if t < params.t_setpoint_change
        Tsp = params.Tsp;
    else
        Tsp = params.Tsp_new;
    end
    
    % Feed temperature disturbance
    if t < params.t_Tf_disturbance
        Tf = params.Tf_nom;
    else
        Tf = params.Tf_disturbed;
    end
    
    % Feed concentration disturbance
    if t < params.t_CAf_disturbance
        CAf = params.CAf_nom;
    else
        CAf = params.CAf_disturbed;
    end
    
    disturbances.Tf = Tf;
    disturbances.CAf = CAf;
    
    %% PID Controller
    % Calculate error
    T_measured = x(1);
    error = Tsp - T_measured;
    
    % Proportional term
    P_term = params.Kp * error;
    
    % Integral term (with anti-windup)
    error_integral = error_integral + error * params.dt;
    % Simple anti-windup: limit integral
    error_integral = max(min(error_integral, 100), -100);
    I_term = params.Ki * error_integral;
    
    % Derivative term
    error_derivative = (error - error_prev) / params.dt;
    D_term = params.Kd * error_derivative;
    
    % PID output
    u_pid = P_term + I_term + D_term;
    
    %% Feedforward Controller
    % Compensate for feed temperature disturbance
    Tf_deviation = Tf - params.Tf_nom;
    u_ff = params.Kff * Tf_deviation;
    
    %% Combined control signal
    Fc = u_pid + u_ff;
    
    % Apply constraints (saturation)
    Fc = max(params.Fc_min, min(params.Fc_max, Fc));
    
    %% Store current values
    T_history(i) = x(1);
    Tc_history(i) = x(2);
    CA_history(i) = x(3);
    Fc_history(i) = Fc;
    Tf_history(i) = Tf;
    CAf_history(i) = CAf;
    Tsp_history(i) = Tsp;
    
    %% Integrate to next time step (if not last step)
    if i < n_steps
        % Use ode45 for one time step
        [~, x_next] = ode45(@(t, x) cstr_model(t, x, Fc, disturbances, params), ...
                            [t, t+params.dt], x);
        x = x_next(end, :)';  % Update state
    end
    
    % Update previous error for derivative term
    error_prev = error;
end

fprintf('\nSimulation complete!\n\n');

%% Create results table
results_table = table(t_span', T_history, Tc_history, CA_history, Fc_history, ...
                      Tf_history, CAf_history, Tsp_history, ...
                      'VariableNames', {'Time_min', 'T_reactor_K', 'T_coolant_K', ...
                      'CA_mol_per_L', 'Fc_coolant_L_per_min', 'Tf_feed_K', ...
                      'CAf_feed_mol_per_L', 'Tsp_setpoint_K'});

%% Export to CSV (simulated OPC-UA data)
output_file = '../data/simulation_output.csv';
writetable(results_table, output_file);
fprintf('Data exported to: %s\n', output_file);
fprintf('This CSV simulates OPC-UA historian data for Python analysis.\n\n');

%% Generate plots
fprintf('Generating plots...\n');

figure('Position', [100, 100, 1200, 800]);

% Plot 1: Reactor Temperature
subplot(3, 2, 1);
plot(t_span, T_history, 'b-', 'LineWidth', 1.5);
hold on;
plot(t_span, Tsp_history, 'r--', 'LineWidth', 1.5);
xlabel('Time (min)');
ylabel('Temperature (K)');
title('Reactor Temperature vs Setpoint');
legend('T_{reactor}', 'T_{setpoint}', 'Location', 'best');
grid on;

% Plot 2: Coolant Temperature
subplot(3, 2, 2);
plot(t_span, Tc_history, 'g-', 'LineWidth', 1.5);
xlabel('Time (min)');
ylabel('Temperature (K)');
title('Coolant Temperature');
grid on;

% Plot 3: Coolant Flow Rate (Manipulated Variable)
subplot(3, 2, 3);
plot(t_span, Fc_history, 'm-', 'LineWidth', 1.5);
xlabel('Time (min)');
ylabel('Flow Rate (L/min)');
title('Coolant Flow Rate (Control Action)');
grid on;

% Plot 4: Concentration
subplot(3, 2, 4);
plot(t_span, CA_history, 'c-', 'LineWidth', 1.5);
xlabel('Time (min)');
ylabel('Concentration (mol/L)');
title('Reactant Concentration');
grid on;

% Plot 5: Feed Temperature (Disturbance)
subplot(3, 2, 5);
plot(t_span, Tf_history, 'r-', 'LineWidth', 1.5);
xlabel('Time (min)');
ylabel('Temperature (K)');
title('Feed Temperature (Disturbance)');
grid on;

% Plot 6: Feed Concentration (Disturbance)
subplot(3, 2, 6);
plot(t_span, CAf_history, 'k-', 'LineWidth', 1.5);
xlabel('Time (min)');
ylabel('Concentration (mol/L)');
title('Feed Concentration (Disturbance)');
grid on;

sgtitle('CSTR Digital Twin - Simulation Results', 'FontSize', 14, 'FontWeight', 'bold');

% Save figure
saveas(gcf, '../data/simulation_results.png');
fprintf('Plots saved to: ../data/simulation_results.png\n');

%% Calculate performance metrics
fprintf('\n=== Control Performance Metrics ===\n');

% Find setpoint change response
idx_setpoint = find(t_span >= params.t_setpoint_change, 1);
T_after_setpoint = T_history(idx_setpoint:end);
t_after_setpoint = t_span(idx_setpoint:end);

% Overshoot
T_final = params.Tsp_new;
T_max = max(T_after_setpoint);
overshoot = ((T_max - T_final) / T_final) * 100;
fprintf('Overshoot: %.2f%%\n', overshoot);

% Settling time (within 2% of final value)
settling_band = 0.02 * T_final;
settled_idx = find(abs(T_after_setpoint - T_final) <= settling_band, 1);
if ~isempty(settled_idx)
    settling_time = t_after_setpoint(settled_idx) - params.t_setpoint_change;
    fprintf('Settling Time (2%% band): %.2f min\n', settling_time);
else
    fprintf('Settling Time: Not achieved within simulation time\n');
end

% Rise time (10% to 90% of final value)
T_initial = params.Tsp;
T_10 = T_initial + 0.1 * (T_final - T_initial);
T_90 = T_initial + 0.9 * (T_final - T_initial);
idx_10 = find(T_after_setpoint >= T_10, 1);
idx_90 = find(T_after_setpoint >= T_90, 1);
if ~isempty(idx_10) && ~isempty(idx_90)
    rise_time = t_after_setpoint(idx_90) - t_after_setpoint(idx_10);
    fprintf('Rise Time (10%%-90%%): %.2f min\n', rise_time);
end

fprintf('\n=== Simulation Summary ===\n');
fprintf('Total simulation time: %.1f min\n', params.t_final);
fprintf('Number of data points: %d\n', n_steps);
fprintf('Sampling interval: %.2f min\n', params.dt);
fprintf('\nReady for Python anomaly detection!\n');
fprintf('Next step: Run python/anomaly_detection.py\n\n');
