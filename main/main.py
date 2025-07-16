import numpy as np
import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_path)
print(src_path)

from MMM import MMM
from utils import prepare_mmm_data, plot_data_overview
from dotenv import load_dotenv
load_dotenv("/home/ashish/Desktop/202418007/Marketing_Mix_Model/.env")
DATA = os.getenv("DATA")

data = prepare_mmm_data(DATA)
plot_data_overview(data)

media_data = data['media_data']
target_data = data['target_data']
extra_features = data['extra_features']
competitor_data = data['competitor_data']



print("Initializing Enhanced MMM...")
model = MMM(
    model_name="enhanced_hill_adstock",
    degrees_seasonality=2,
    weekday_seasonality=True
)

print("Fitting model...")
model.fit(
    media_data=media_data,
    target_data=target_data,
    extra_features=extra_features,
    competitor_data=competitor_data,
    num_warmup=500,
    num_samples=500,
    num_chains=2,
    use_svi=False  
)

print("Model fitted successfully!")

print("Generating predictions...")
predictions = model.predict(
    media_data=media_data,
    extra_features=extra_features,
    competitor_data=competitor_data
)

print(f"Prediction MAPE: {np.mean(np.abs((predictions['mean'] - target_data) / target_data)) * 100:.2f}%")

print("Computing media contributions...")
contributions_result = model.compute_media_contributions(
    media_data=media_data,
    extra_features=extra_features,
    competitor_data=competitor_data
)

mean_contributions = contributions_result['mean_contributions']
total_contribution = contributions_result['total_contribution']

print("Mean media contributions (last period):")
print(mean_contributions[-1])

print("Total contribution across time periods:")
print(total_contribution)

print("Optimizing budget...")
total_budget = 100.0  
budget_result = model.optimize_budget(
    total_budget=total_budget,
    media_data_historical=media_data,
    extra_features=extra_features,
    competitor_data=competitor_data,
    risk_tolerance=0.2
)

print("Optimal budget allocation:")
print(budget_result['optimal_allocation'])

print("Allocation shares:")
print(budget_result['allocation_shares'])

print("Expected return:")
print(budget_result['expected_return'])

print("Optimization success:", budget_result['optimization_success'])
print("Message:", budget_result['optimization_message'])

print("Done.")