function dxdt = cstr_model(t, x, u, disturbances, params)
% cstr_model.m
% ODE function for CSTR dynamics
% This function defines the differential equations governing the CSTR
%
% INPUTS:
%   t            - Current time (min)
%   x            - State vector [T; Tc; CA]
%                  T  = Reactor temperature (K)
%                  Tc = Coolant temperature (K)
%                  CA = Reactant concentration (mol/L)
%   u            - Control input: Coolant flow rate Fc (L/min)
%   disturbances - Structure with Tf and CAf
%   params       - Parameter structure from cstr_parameters.m
%
% OUTPUT:
%   dxdt         - Time derivatives [dT/dt; dTc/dt; dCA/dt]
%
% EQUATIONS:
%   Reactor energy balance, coolant energy balance, mass balance

%% Extract states
T = x(1);      % Reactor temperature (K)
Tc = x(2);     % Coolant temperature (K)
CA = x(3);     % Concentration (mol/L)

%% Extract control input
Fc = u;        % Coolant flow rate (L/min)

%% Extract disturbances
Tf = disturbances.Tf;      % Feed temperature (K)
CAf = disturbances.CAf;    % Feed concentration (mol/L)

%% Extract parameters
V = params.V;
rho = params.rho;
Cp = params.Cp;
dH = params.dH;
E = params.E;
k0 = params.k0;
R = params.R;
UA = params.UA;
Vc = params.Vc;
rhoc = params.rhoc;
Cpc = params.Cpc;
F = params.F;
Tcf = params.Tcf;

%% Calculate reaction rate
% Arrhenius equation: k = k0 * exp(-E/RT)
k = k0 * exp(-E / (R * T));
reaction_rate = k * CA;  % First-order reaction (mol/L·min)

%% Reactor temperature dynamics (dT/dt)
% Energy balance: Accumulation = Input - Output + Generation - Heat Transfer

% Input energy from feed
Q_in = F * rho * Cp * (Tf - T) / (V * rho * Cp);

% Heat generation from reaction (exothermic, so dH is negative)
Q_gen = (-dH) * reaction_rate / (rho * Cp);

% Heat transfer to cooling jacket
Q_transfer = UA * (T - Tc) / (V * rho * Cp);

% Overall reactor temperature rate of change
dT_dt = Q_in + Q_gen - Q_transfer;

%% Coolant temperature dynamics (dTc/dt)
% Energy balance for cooling jacket

% Coolant input/output
Q_coolant_flow = Fc * rhoc * Cpc * (Tcf - Tc) / (Vc * rhoc * Cpc);

% Heat transfer from reactor
Q_coolant_transfer = UA * (T - Tc) / (Vc * rhoc * Cpc);

% Overall coolant temperature rate of change
dTc_dt = Q_coolant_flow + Q_coolant_transfer;

%% Concentration dynamics (dCA/dt)
% Mass balance: Accumulation = Input - Output - Consumption

% Input/output due to flow
dCA_flow = F * (CAf - CA) / V;

% Consumption due to reaction
dCA_reaction = -reaction_rate;

% Overall concentration rate of change
dCA_dt = dCA_flow + dCA_reaction;

%% Return derivative vector
dxdt = [dT_dt; dTc_dt; dCA_dt];

end
