import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes and Greeks functions
def BlackScholes(r, S, K, T, sigma, type='C'):
    d1 = (np.log(S/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if type == 'C':
        price = S * norm.cdf(d1, 0, 1) - K * np.exp(-r * T) * norm.cdf(d2, 0, 1)
    elif type == 'P':
        price = K * np.exp(-r * T) * norm.cdf(-d2, 0, 1) - S * norm.cdf(-d1, 0, 1)
    return price

def delta_calc(r, S, K, T, sigma, type='C'):
    d1 = (np.log(S/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    if type == 'C':
        delta_value = norm.cdf(d1, 0, 1)
    elif type == 'P':
        delta_value = -norm.cdf(-d1, 0, 1)
    return delta_value

def gamma_calc(r, S, K, T, sigma):
    d1 = (np.log(S/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    gamma_value = norm.pdf(d1, 0, 1) / (S * sigma * np.sqrt(T))
    return gamma_value

def vega_calc(r, S, K, T, sigma):
    d1 = (np.log(S/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    vega_value = S * norm.pdf(d1, 0, 1) * np.sqrt(T)
    return vega_value

def theta_calc(r, S, K, T, sigma, type='C'):
    d1 = (np.log(S/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if type == 'C':
        theta_value = -S * norm.pdf(d1, 0, 1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2, 0, 1)
    elif type == 'P':
        theta_value = -S * norm.pdf(d1, 0, 1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2, 0, 1)
    return theta_value

def rho_calc(r, S, K, T, sigma, type='C'):
    d1 = (np.log(S/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if type == 'C':
        rho_value = T * K * np.exp(-r * T) * norm.cdf(d2, 0, 1)
    elif type == 'P':
        rho_value = -T * K * np.exp(-r * T) * norm.cdf(-d2, 0, 1)
    return rho_value

# Streamlit UI
st.title("Black-Scholes Option Pricing Model")

# Inputs from the user
current_price = st.number_input("Current Price", min_value=0.0, value=229.849) # New input added
S = st.number_input("Underlying Asset Price (S)", min_value=0.0, max_value=10000.0, value=5626.02, step=1.0)
K = st.number_input("Strike Price (K)", min_value=0.0, max_value=10000.0, value=5700.00, step=1.0)
T_days = st.number_input("Time to Maturity (T) in days", min_value=1, max_value=365, value=30, step=1)
T = T_days / 365  # Convert days to years
sigma = st.number_input("Volatility (sigma)", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
r = st.number_input("Risk-Free Rate (r)", min_value=0.0, max_value=0.2, value=0.015, step=0.001)
purchase_price = st.number_input("Purchase Price", min_value=0.0, value=229.849)
current_market_price = st.number_input("Current Market Price", min_value=0.0, value=current_price) # Updated to use current_price

# Calculate option price and Greeks
call_price = BlackScholes(r, S, K, T, sigma, type='C')
put_price = BlackScholes(r, S, K, T, sigma, type='P')

call_delta = delta_calc(r, S, K, T, sigma, type='C')
put_delta = delta_calc(r, S, K, T, sigma, type='P')

call_gamma = gamma_calc(r, S, K, T, sigma)
put_gamma = call_gamma

call_vega = vega_calc(r, S, K, T, sigma)
put_vega = call_vega

call_theta = theta_calc(r, S, K, T, sigma, type='C')
put_theta = theta_calc(r, S, K, T, sigma, type='P')

call_rho = rho_calc(r, S, K, T, sigma, type='C')
put_rho = rho_calc(r, S, K, T, sigma, type='P')

# Create a DataFrame for the table
data = {
    'Metric': ['Price', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho'],
    'Call': [f'${call_price:.2f}', f'{call_delta:.4f}', f'{call_gamma:.4f}', f'{call_vega:.4f}', f'{call_theta:.4f}', f'{call_rho:.4f}'],
    'Put': [f'${put_price:.2f}', f'{put_delta:.4f}', f'{put_gamma:.4f}', f'{put_vega:.4f}', f'{put_theta:.4f}', f'{put_rho:.4f}']
}

df = pd.DataFrame(data)

# Display the table
st.subheader("Option Pricing and Greeks")
st.dataframe(df, width=800)

# Calculate PnL
pnl = current_market_price - purchase_price
st.subheader("Profit and Loss (PnL)")
if pnl > 0:
    st.markdown(f"<h3 style='color: green;'>PnL: ${pnl:.2f}</h3>", unsafe_allow_html=True)
else:
    st.markdown(f"<h3 style='color: red;'>PnL: ${pnl:.2f}</h3>", unsafe_allow_html=True)

# Generate heatmaps
st.subheader("Heatmaps")

# Option Price Heatmap
S_range = np.linspace(5000, 6000, 100)
T_range = np.linspace(10/365, 90/365, 100)
price_matrix = np.zeros((len(S_range), len(T_range)))
for i, S_val in enumerate(S_range):
    for j, T_val in enumerate(T_range):
        price_matrix[i, j] = BlackScholes(r, S_val, K, T_val, sigma, type='C')

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(price_matrix, xticklabels=10, yticklabels=10, cmap='YlGnBu', cbar=True, ax=ax, cbar_kws={'format': '$%.2f'})
ax.set_xlabel('Time to Maturity (Years)')
ax.set_ylabel('Underlying Asset Price')
ax.set_title('Call Option Price Heatmap')
st.pyplot(fig)


#PnL matrix for heatmap
pnl_matrix = np.zeros((len(S_range), len(T_range)))
for i, S_val in enumerate(S_range):
    for j, T_val in enumerate(T_range):
        # You can choose either call or put PnL
        option_price = BlackScholes(r, S_val, K, T_val, sigma, type='C')
        pnl_matrix[i, j] = option_price - purchase_price

#PnL Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(pnl_matrix, xticklabels=10, yticklabels=10, cmap='RdYlGn', cbar=True, ax=ax, center=0, cbar_kws={'format': '$%.0f'})
ax.set_xlabel('Time to Maturity (Years)')
ax.set_ylabel('Underlying Asset Price')
ax.set_title('PnL Heatmap (Red: Negative, Green: Positive)')
st.pyplot(fig)
