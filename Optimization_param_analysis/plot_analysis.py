import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

# Load the results from the CSV file
input_file = "/home/stud/casperc/bhome/Project3_DL_sigpros/Optimization_param_analysis/optimization_results_3.csv"
results_df = pd.read_csv(input_file)
print(results_df.head())
# Plotting time vs ftol
plt.figure(figsize=(10, 6))
orders = results_df['order'].unique()
for order in orders:
    subset = results_df[results_df['order'] == order]
    plt.errorbar(subset['ftol'], subset['time_mean'], yerr=subset['time_std'], label=f'Order {order}', capsize=5)

plt.xscale('log')
plt.xlabel('ftol')
plt.ylabel('Time (s)')
plt.title('Time vs ftol for different orders')
plt.legend()
plt.grid(True)
plt.savefig("/home/stud/casperc/bhome/Project3_DL_sigpros/Optimization_param_analysis/time_vs_ftol.png")

# plot error vs ftol
plt.figure(figsize=(10, 6))
orders = results_df['order'].unique()
for order in orders:
    subset = results_df[results_df['order'] == order]
    plt.errorbar(subset['ftol'], subset['error_mean'], yerr=subset['error_std'], label=f'Order {order}', capsize=5)
    
plt.xscale('log')
plt.xlabel('ftol')
plt.ylabel('Error')
plt.title('Error vs ftol for different orders')
plt.legend()
plt.grid(True)
plt.savefig("/home/stud/casperc/bhome/Project3_DL_sigpros/Optimization_param_analysis/error_vs_ftol.png")

# plot time vs maxiter
plt.figure(figsize=(10, 6))
orders = results_df['order'].unique()
for order in orders:
    subset = results_df[results_df['order'] == order]
    plt.errorbar(subset['maxiter'], subset['time_mean'], yerr=subset['time_std'], label=f'Order {order}', capsize=5)
    
plt.xlabel('maxiter')
plt.ylabel('Time (s)')
plt.title('Time vs maxiter for different orders')
plt.legend()
plt.grid(True)
plt.savefig("/home/stud/casperc/bhome/Project3_DL_sigpros/Optimization_param_analysis/time_vs_maxiter.png")

# plot error vs maxiter
plt.figure(figsize=(10, 6))
orders = results_df['order'].unique()
for order in orders:
    subset = results_df[results_df['order'] == order]
    plt.errorbar(subset['maxiter'], subset['error_mean'], yerr=subset['error_std'], label=f'Order {order}', capsize=5)
    
plt.xlabel('maxiter')
plt.ylabel('Error')
plt.title('Error vs maxiter for different orders')
plt.legend()
plt.grid(True)
plt.savefig("/home/stud/casperc/bhome/Project3_DL_sigpros/Optimization_param_analysis/error_vs_maxiter.png")
