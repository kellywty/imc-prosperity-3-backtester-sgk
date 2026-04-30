import numpy as np
from scipy.optimize import minimize

def simulate_round_5_portfolio():
    # 1. Game Parameters
    BUDGET = 1_000_000
    NUM_SIMULATIONS = 10000
    
    # ==========================================
    # 2. THE MASTER DIAL: Risk Aversion Penalty
    # ==========================================
    # 0.0 = Pure EV (Teammate's logic: trades almost everything)
    # 0.5 = Moderate Risk Aversion (Cuts out the noise/low-edge trades)
    # 1.0 = High Risk Aversion (Only the absolute safest bets)
    RISK_AVERSION_PENALTY = 0.3
    
    # 3. Qualitative Estimates for ALL 9 Products
    # Format: 'Product': (Expected Return, Uncertainty/Standard Deviation)
    estimates = {
        'Lava Cake': (-0.30, 0.20),         # High conviction short
        'Thermalite Core': (0.20, 0.15),    # High conviction long
        'Ashes of Phoenix': (-0.20, 0.25),  # Moderate short, high uncertainty
        'Scoria Paste': (0.10, 0.10),       # Moderate long
        'Sulfur Reactor': (0.10, 0.05),     # Moderate long, very low uncertainty
        'Magma Ink': (0.05, 0.10),          # Stale news, tiny edge
        'Pyroflex Cells': (-0.05, 0.10),    # Small short
        'Obsidian Cutlery': (-0.02, 0.10),  # Ambiguous news, almost zero edge
        'Volcanic Incense': (-0.10, 0.40)   # Massive uncertainty (retail trap)
    }
    
    assets = list(estimates.keys())
    n_assets = len(assets)
    
    # Generate Monte Carlo scenarios
    np.random.seed(42)
    simulated_returns = np.zeros((NUM_SIMULATIONS, n_assets))
    for i, asset in enumerate(assets):
        mu, std = estimates[asset]
        simulated_returns[:, i] = np.random.normal(mu, std, NUM_SIMULATIONS)
        
        # If shorting, invert the return for PnL calculation
        if mu < 0:
            simulated_returns[:, i] = -simulated_returns[:, i]

    # 4. The Objective Function (Now with Risk Penalty)
    def objective(allocations):
        fees = np.sum((allocations / 100.0)**2 * BUDGET)
        investments = (allocations / 100.0) * BUDGET
        simulated_pnl = np.dot(simulated_returns, investments) - fees
        
        expected_pnl = np.mean(simulated_pnl)
        pnl_std_dev = np.std(simulated_pnl)
        
        # Risk-Adjusted Score: Maximize EV, but penalize volatility
        risk_adjusted_score = expected_pnl - (RISK_AVERSION_PENALTY * pnl_std_dev)
        
        # Scipy minimize needs a negative value to find the maximum
        return -risk_adjusted_score

    # 5. Constraints and Bounds
    constraints = ({'type': 'ineq', 'fun': lambda x: 100.0 - np.sum(x)})
    bounds = tuple((0, 100) for _ in range(n_assets))
    initial_guess = np.ones(n_assets) * (50.0 / n_assets)

    # 6. Run the Optimizer
    result = minimize(
        objective, 
        initial_guess, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    # 7. Output the Results
    optimal_allocations = result.x
    
    print(f"\n--- OPTIMAL PORTFOLIO (Risk Penalty: {RISK_AVERSION_PENALTY}) ---")
    print(f"{'PRODUCT':<20} | {'EXPECTED RETURN':<15} | {'OPTIMAL ALLOCATION %'}")
    print("-" * 60)
    for i, asset in enumerate(assets):
        mu = estimates[asset][0]
        direction = "SHORT" if mu < 0 else "LONG " if mu > 0 else "NONE "
        
        # Clean up near-zero outputs for readability
        alloc = optimal_allocations[i] if optimal_allocations[i] > 0.01 else 0.0
        
        print(f"{asset:<20} | {direction} {abs(mu*100):>5.1f}%   | {alloc:>6.2f}%")
        
    print("-" * 60)
    print(f"Total Budget Used: {np.sum(optimal_allocations):.2f}%")
    
    # Calculate pure EV for the final output
    final_fees = np.sum((optimal_allocations / 100.0)**2 * BUDGET)
    final_investments = (optimal_allocations / 100.0) * BUDGET
    final_pnl = np.mean(np.dot(simulated_returns, final_investments)) - final_fees
    print(f"Expected Pure PnL: {final_pnl:,.2f} XIRECs\n")

if __name__ == "__main__":
    simulate_round_5_portfolio()
