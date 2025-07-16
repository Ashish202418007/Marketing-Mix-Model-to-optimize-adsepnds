import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_mmm_data(csv_file_path):
    """
    Prepare MMM data from the simulated CSV file for EnhancedLightweightMMM
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the CSV file containing simulated MMM data
        
    Returns:
    --------
    dict: Dictionary containing all prepared data arrays and metadata
    """
    
    print("Loading simulated MMM data...")
    df = pd.read_csv(csv_file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Data loaded: {df.shape[0]} time periods, {df.shape[1]} features")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    target_data = df['outcome'].values
    
    # media channels 
    media_columns = [col for col in df.columns if col.startswith('media_')]
    media_data = df[media_columns].values
    n_media_channels = len(media_columns)
    
    print(f"Media channels found: {media_columns}")
    print(f"Media data shape: {media_data.shape}")
    
    #  competitor data
    competitor_columns = [col for col in df.columns if col.startswith('competitor_')]
    competitor_data = df[competitor_columns].values if competitor_columns else None
    
    if competitor_data is not None:
        print(f"Competitor channels found: {competitor_columns}")
        print(f"Competitor data shape: {competitor_data.shape}")
    else:
        print("No competitor data found")
    
    # control variables 
    control_columns = [col for col in df.columns if col.startswith('control_')]
    extra_features = df[control_columns].values if control_columns else None
    
    if extra_features is not None:
        print(f"Control variables found: {control_columns}")
        print(f"Control data shape: {extra_features.shape}")
    else:
        print("No control variables found")
    
    #  seasonality components 
    seasonal_columns = [col for col in df.columns if col.startswith('seasonal_')]
    seasonal_data = df[seasonal_columns].values if seasonal_columns else None
    
    if seasonal_data is not None:
        print(f"Seasonal components found: {seasonal_columns}")
        print(f"Seasonal data shape: {seasonal_data.shape}")
    
    #  interaction effects
    interaction_columns = [col for col in df.columns if col.startswith('interaction_')]
    interaction_data = df[interaction_columns].values if interaction_columns else None
    
    if interaction_data is not None:
        print(f"Interaction effects found: {interaction_columns}")
        print(f" Interaction data shape: {interaction_data.shape}")
    
    combined_extra_features = []
    combined_feature_names = []
    
    if extra_features is not None:
        combined_extra_features.append(extra_features)
        combined_feature_names.extend(control_columns)
    
    if seasonal_data is not None:
        combined_extra_features.append(seasonal_data)
        combined_feature_names.extend(seasonal_columns)
    

    if interaction_data is not None:
        combined_extra_features.append(interaction_data)
        combined_feature_names.extend(interaction_columns)
    
    if combined_extra_features:
        final_extra_features = np.concatenate(combined_extra_features, axis=1)
    else:
        final_extra_features = None
    
    ground_truth = {
        'media_channels': media_columns,
        'competitor_channels': competitor_columns if competitor_columns else [],
        'control_variables': control_columns if control_columns else [],
        'seasonal_components': seasonal_columns if seasonal_columns else [],
        'interaction_effects': interaction_columns if interaction_columns else [],
        'total_features': combined_feature_names
    }
    
    #  final data summary
    print("\n" + "="*50)
    print(" FINAL DATA SUMMARY")
    print("="*50)
    print(f"Target shape: {target_data.shape}")
    print(f"Media data shape: {media_data.shape}")
    print(f"Competitor data shape: {competitor_data.shape if competitor_data is not None else 'None'}")
    print(f"Extra features shape: {final_extra_features.shape if final_extra_features is not None else 'None'}")
    print(f"Total extra features: {len(combined_feature_names)}")
    
    return {
        'target_data': target_data,
        'media_data': media_data,
        'competitor_data': competitor_data,
        'extra_features': final_extra_features,
        'dates': df['date'].values,
        'ground_truth': ground_truth,
        'raw_df': df,
        'media_columns': media_columns,
        'feature_names': combined_feature_names
    }

def plot_data_overview(data_dict):
    """
    Create overview plots of the prepared data
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('MMM Data Overview', fontsize=16, fontweight='bold')
    
    dates = data_dict['dates']
    

    axes[0, 0].plot(dates, data_dict['target_data'], color='darkblue', linewidth=2)
    axes[0, 0].set_title('Target Variable (Outcome)')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].tick_params(axis='x', rotation=45)

    media_data = data_dict['media_data']
    for i, channel in enumerate(data_dict['media_columns']):
        axes[0, 1].plot(dates, media_data[:, i], label=channel.replace('media_', '').title(), alpha=0.7)
    axes[0, 1].set_title('Media Spend Over Time')
    axes[0, 1].set_ylabel('Spend')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    

    if data_dict['competitor_data'] is not None:
        competitor_data = data_dict['competitor_data']
        for i in range(competitor_data.shape[1]):
            axes[1, 0].plot(dates, competitor_data[:, i], 
                          label=f"Competitor {i+1}", alpha=0.7)
        axes[1, 0].set_title('Competitor Spend Over Time')
        axes[1, 0].set_ylabel('Spend')
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)
    else:
        axes[1, 0].text(0.5, 0.5, 'No Competitor Data', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Competitor Data')

    if data_dict['extra_features'] is not None:
        extra_features = data_dict['extra_features']

        n_controls_to_plot = min(3, extra_features.shape[1])
        for i in range(n_controls_to_plot):
            feature_name = data_dict['feature_names'][i] if i < len(data_dict['feature_names']) else f'Feature {i+1}'
            axes[1, 1].plot(dates, extra_features[:, i], 
                          label=feature_name.replace('control_', '').title(), alpha=0.7)
        axes[1, 1].set_title('Control Variables (Sample)')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Control Variables', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Control Variables')
    
    plt.tight_layout()
    plt.show()