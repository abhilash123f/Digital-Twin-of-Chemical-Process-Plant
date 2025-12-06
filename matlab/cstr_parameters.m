% cstr_parameters.m
% Defines all parameters and constants for the CSTR digital twin
% Author: [Your Name]
% Date: December 2025

%% Clear workspace
clear all;
clc;

fprintf('Loading CSTR parameters...\n');

%% Physical Constants
global params;

% Reactor Parameters
params.V = 100;              % Reactor volume (L)
params.rho = 1000;           % Density (g/L)
params.Cp = 0.239;           % Heat capacity (J/g·K)
params.dH = -5e4;            % Heat of reaction (J/mol) - exothermic (negative)
params.E = 11843;            % Activation energy (J/mol)
params.k0 = 7.2e10;          % Pre-exponential factor (1/min)
params.R = 8.314;            % Gas constant (J/mol·K)

% Heat Transfer Parameters
params.UA = 5e4;             % Overall heat transfer coefficient × Area (J/min·K)

% Cooling Jacket Parameters
params.Vc = 10;              % Jacket volume (L)
params.rhoc = 1000;          % Coolant density (g/L)
params.Cpc = 0.239;          % Coolant heat capacity (J/g·K)

% Nominal Operating Conditions
params.F = 100;              % Feed flow rate (L/min)
params.Tf_nom = 350;         % Nominal feed temperature (K)
params.CAf_nom = 1.0;        % Nominal feed concentration (mol/L)
params.Tcf = 350;            % Coolant feed temperature (K)
params.Tsp = 350;            % Temperature setpoint (K)

% Initial Conditions (Steady State)
params.T0 = 350;             % Initial reactor temperature (K)
params.Tc0 = 350;            % Initial coolant temperature (K)
params.CA0 = 0.5;            % Initial concentration (mol/L)
params.Fc0 = 50;             % Initial coolant flow rate (L/min)

%% Controller Parameters

% PID Controller Gains (tuned using Ziegler-Nichols + manual adjustment)
params.Kp = 10;              % Proportional gain
params.Ki = 0.5;             % Integral gain
params.Kd = 2;               % Derivative gain

% Feedforward Controller
params.Kff = 0.8;            % Feedforward gain (tuned based on heat balance)

% Controller Constraints
params.Fc_min = 0;           % Minimum coolant flow (L/min)
params.Fc_max = 100;         % Maximum coolant flow (L/min)

%% Simulation Parameters
params.t_final = 100;        % Simulation time (min)
params.dt = 0.1;             % Time step for data logging (min)

%% Disturbance Scenarios
% These define when disturbances occur during simulation

% Setpoint change
params.t_setpoint_change = 10;    % Time of setpoint change (min)
params.Tsp_new = 360;             % New setpoint (K)

% Feed temperature disturbance
params.t_Tf_disturbance = 30;     % Time of Tf disturbance (min)
params.Tf_disturbed = 360;        % Disturbed feed temperature (K)

% Feed concentration disturbance
params.t_CAf_disturbance = 50;    % Time of CAf disturbance (min)
params.CAf_disturbed = 1.2;       % Disturbed feed concentration (mol/L)

%% Anomaly Detection Thresholds (for Python)
% These are saved to a config file for Python to read

params.anomaly_sigma = 3;         % Number of standard deviations for anomaly
params.anomaly_rate_threshold = 5; % Max allowable dT/dt (K/min)
params.anomaly_window = 20;       % Rolling window size for statistics (samples)

%% Display Parameters
fprintf('\n=== CSTR Digital Twin Parameters ===\n');
fprintf('Reactor Volume: %.1f L\n', params.V);
fprintf('Nominal Temperature: %.1f K\n', params.Tsp);
fprintf('Heat of Reaction: %.2e J/mol\n', params.dH);
fprintf('PID Gains: Kp=%.1f, Ki=%.2f, Kd=%.1f\n', params.Kp, params.Ki, params.Kd);
fprintf('Feedforward Gain: %.2f\n', params.Kff);
fprintf('Simulation Time: %.1f min\n', params.t_final);
fprintf('Parameters loaded successfully!\n\n');

%% Save parameters to file for Python access
% Create a simple text file with key parameters
fid = fopen('../data/parameters.txt', 'w');
fprintf(fid, 'CSTR Digital Twin Parameters\n');
fprintf(fid, '============================\n');
fprintf(fid, 'Reactor_Volume_L: %.2f\n', params.V);
fprintf(fid, 'Setpoint_K: %.2f\n', params.Tsp);
fprintf(fid, 'Kp: %.2f\n', params.Kp);
fprintf(fid, 'Ki: %.2f\n', params.Ki);
fprintf(fid, 'Kd: %.2f\n', params.Kd);
fprintf(fid, 'Kff: %.2f\n', params.Kff);
fprintf(fid, 'Anomaly_Sigma: %.2f\n', params.anomaly_sigma);
fprintf(fid, 'Anomaly_Rate_Threshold_K_per_min: %.2f\n', params.anomaly_rate_threshold);
fclose(fid);

fprintf('Parameters exported to ../data/parameters.txt\n');
