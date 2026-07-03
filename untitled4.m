%% Financial Risk Assessment and Portfolio Optimization via Monte Carlo
% Author: Goldfish Prodigy
% Description: Uses Geometric Brownian Motion (GBM) and multivariate 
%              normal distributions to simulate future asset paths, 
%              calculating portfolio Value at Risk (VaR) and optimizing returns.

clear; clc; close all;

%% 1. Baseline Portfolio Configuration (3 Assets)
num_assets = 3;
num_simulations = 10000; % Total number of Monte Carlo paths
trading_days = 252;      % Horizon: 1 Year of trading days

% Expected annual returns for the assets
expected_returns = [0.12; 0.08; 0.15]; 

% Annualized volatilities (standard deviations)
volatilities = [0.18; 0.10; 0.25];

% Correlation Matrix (Linear Algebra framework defining asset relationships)
R = [ 1.0,  0.3, -0.2;
      0.3,  1.0,  0.1;
     -0.2,  0.1,  1.0];

% Initial portfolio allocation weights (must sum to 1.0)
weights = [0.4; 0.4; 0.2]; 
initial_portfolio_value = 1000000; % $1,000,000 USD

%% 2. Construct Covariance Matrix and Apply Cholesky Decomposition
% Convert correlation to covariance matrix: Sigma = D * R * D
D = diag(volatilities);
Sigma = D * R * D;

% Cholesky Decomposition splits Covariance into lower triangular matrix L
% This allows us to inject linear correlations into independent random variables
L = chol(Sigma, 'lower');

%% 3. Execute Monte Carlo Simulation Loop
dt = 1 / trading_days;
portfolio_final_values = zeros(num_simulations, 1);

% Pre-calculate daily drift coefficients for GBM: (mu - 0.5 * sigma^2) * dt
drift = (expected_returns - 0.5 * (volatilities.^2)) * dt;

for sim = 1:num_simulations
    % Generate uncorrelated daily random shocks from normal distribution
    Z = randn(num_assets, trading_days);
    
    % Correlate the random shocks using our Cholesky matrix
    correlated_Z = L * Z;
    
    % Simulate daily asset pricing paths using Geometric Brownian Motion
    % S(t) = S(0) * exp(drift + volatility * shock)
    asset_returns = exp(drift + correlated_Z * sqrt(dt));
    
    % Compute cumulative asset returns across the time horizon
    cumulative_asset_returns = prod(asset_returns, 2);
    
    % Final portfolio valuation for this specific path
    final_asset_values = initial_portfolio_value * (weights .* cumulative_asset_returns);
    portfolio_final_values(sim) = sum(final_asset_values);
end

%% 4. Risk Assessment: Value at Risk (VaR) Metrics
% Calculate individual paths' total net returns
portfolio_net_returns = (portfolio_final_values - initial_portfolio_value) / initial_portfolio_value;

% Sort returns to find percentiles
confidence_level = 0.95;
sorted_returns = sort(portfolio_net_returns);
cutoff_index = round((1 - confidence_level) * num_simulations);

% 95% Value at Risk (VaR)
VaR_95 = -sorted_returns(cutoff_index);
VaR_dollar = VaR_95 * initial_portfolio_value;

%% 5. Visualization Matrix
figure('Name', 'Monte Carlo Risk Assessment Engine', 'Position', [100, 100, 1000, 500]);

% Histogram distribution of final portfolio valuations
histogram(portfolio_final_values / 1e6, 50, 'FaceColor', [0.2 0.6 0.5], 'EdgeColor', 'w');
hold on;

% Draw boundary line showing the threshold for the 5% worst-case scenarios
var_threshold_line = (initial_portfolio_value - VaR_dollar) / 1e6;
line([var_threshold_line, var_threshold_line], ylim, 'Color', 'r', 'LineWidth', 2.5, 'LineStyle', '--');

grid on;
xlabel('Ending Portfolio Value ($ Millions)');
ylabel('Frequency Count');
title('10,000 Path Monte Carlo Value Distribution Profile');
legend('Simulated Endpoints', sprintf('95%% Value at Risk Threshold'), 'Location', 'best');

%% 6. Diagnostic Console Summary
fprintf('=== Portfolio Risk Analysis ===\n');
fprintf('Initial Portfolio Principle: $%,.2f\n', initial_portfolio_value);
fprintf('Expected Value (Mean Path): $%,.2f\n', mean(portfolio_final_values));
fprintf('----------------------------------------\n');
fprintf('Calculated 95%% VaR (Percentage): %.2f%%\n', VaR_95 * 100);
fprintf('Calculated 95%% VaR (Absolute Cash): $%,.2f\n', VaR_dollar);
fprintf('Interpretation: There is a 5%% chance that the portfolio will lose more than $%,.2f over a 1-year horizon.\n', VaR_dollar);