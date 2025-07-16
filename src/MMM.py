import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
from numpyro.infer import SVI, Trace_ELBO
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class MMM:
    """
    Enhanced LightweightMMM with advanced features:
    - Competitive effects
    - Cross-channel interactions
    - Time-varying coefficients
    - Power transformations
    - Risk-adjusted optimization
    """
    
    def __init__(self, 
                 model_name: str = "enhanced_hill_adstock",
                 degrees_seasonality: int = 2,
                 weekday_seasonality: bool = True,
                 adstock_max_lag: int = 7,
                 convolve_func: str = "Adstock"):
        
        self.model_name = model_name
        self.degrees_seasonality = degrees_seasonality
        self.weekday_seasonality = weekday_seasonality
        self.adstock_max_lag = adstock_max_lag
        self.convolve_func = convolve_func
        self.mcmc = None
        self.posterior_samples = None
        self.guide = None
        self.svi = None
        
    def _apply_power_adstock(self, x: jnp.ndarray, lambda_adstock: float, rho_power: float) -> jnp.ndarray:
        """Apply power-transformed adstock transformation"""
        def scan_fn(carry, x_t):
            adstock_t = x_t**rho_power + lambda_adstock * carry
            return adstock_t, adstock_t
        
        _, adstock_data = jax.lax.scan(scan_fn, init=0.0, xs=x)
        return adstock_data
    
    def _apply_hill_saturation(self, x: jnp.ndarray, K_sat: float, S_sat: float, nu_sat: float) -> jnp.ndarray:
        """Apply Hill saturation transformation"""
        denominator = jnp.power(S_sat, nu_sat) + jnp.power(x, nu_sat)
        return K_sat * jnp.power(x, nu_sat) / denominator
    
    def _apply_competitive_effect(self, x: jnp.ndarray, competitor_spend: jnp.ndarray, 
                                 phi_comp: float, kappa_comp: float) -> jnp.ndarray:
        """Apply competitive interference effect"""
        if competitor_spend is None:
            return x
        
        competitive_saturation = jnp.power(competitor_spend, phi_comp) / (
            jnp.power(competitor_spend, phi_comp) + jnp.power(kappa_comp, phi_comp)
        )
        competitive_factor = 1 - competitive_saturation
        return x * competitive_factor
    
    def _create_seasonality_features(self, n_time_periods: int) -> jnp.ndarray:
        """Create seasonality features"""
        time_arange = jnp.arange(n_time_periods, dtype=jnp.float32)
        
        # Annual seasonality
        seasonality_features = []
        for i in range(1, self.degrees_seasonality + 1):
            cos_features = jnp.cos(2 * jnp.pi * i * time_arange / 52.0)
            sin_features = jnp.sin(2 * jnp.pi * i * time_arange / 52.0)
            seasonality_features.extend([cos_features, sin_features])
        
        # Weekday seasonality
        if self.weekday_seasonality:
            for i in range(1, 4):  
                cos_features = jnp.cos(2 * jnp.pi * i * time_arange / 7.0)
                sin_features = jnp.sin(2 * jnp.pi * i * time_arange / 7.0)
                seasonality_features.extend([cos_features, sin_features])
        
        return jnp.stack(seasonality_features, axis=-1)
    
    def _compute_cross_channel_interactions(self, transformed_media: jnp.ndarray, 
                                          delta_interact: jnp.ndarray) -> jnp.ndarray:
        """Compute pairwise cross-channel interactions"""
        n_media_channels = transformed_media.shape[1]
        interaction_contrib = 0.0
        
        idx = 0
        for i in range(n_media_channels):
            for j in range(i + 1, n_media_channels):
                interaction_contrib += (delta_interact[idx] * 
                                      transformed_media[:, i] * 
                                      transformed_media[:, j])
                idx += 1
        
        return interaction_contrib
    
    def _model_definition(self, media_data: jnp.ndarray, 
                         extra_features: Optional[jnp.ndarray] = None,
                         competitor_data: Optional[jnp.ndarray] = None,
                         target_data: Optional[jnp.ndarray] = None) -> None:
        """Enhanced LightweightMMM model definition"""
        
        n_time_periods, n_media_channels = media_data.shape
        n_extra_features = extra_features.shape[1] if extra_features is not None else 0
        n_interactions = n_media_channels * (n_media_channels - 1) // 2
        
        # Seasonality features
        seasonality_features = self._create_seasonality_features(n_time_periods)
        n_seasonality_features = seasonality_features.shape[1]
        
        # === PRIORS ===
        
        # Intercept
        intercept = numpyro.sample("intercept", dist.Normal(0, 1))
        
        # Trend
        trend_coef = numpyro.sample("trend_coef", dist.Normal(0, 0.1))
        
        # Seasonality coefficients
        seasonality_coef = numpyro.sample("seasonality_coef", 
                                        dist.Normal(0, 0.1).expand([n_seasonality_features]))
        
        # Media channel parameters
        with numpyro.plate("media_channels", n_media_channels):
            beta_media = numpyro.sample("beta_media", dist.Gamma(1, 1))
            
            # Time-varying seasonality multiplier
            seasonal_multiplier = numpyro.sample("seasonal_multiplier", dist.Normal(0, 0.1))
            
            # Adstock parameters
            lambda_adstock = numpyro.sample("lambda_adstock", dist.Beta(2, 2))
            rho_power = numpyro.sample("rho_power", 
                                     dist.TruncatedNormal(1, 0.2, low=0.5, high=1.5))
            
            # Hill saturation parameters
            K_sat = numpyro.sample("K_sat", dist.Gamma(1, 1))
            S_sat = numpyro.sample("S_sat", dist.Gamma(1, 1))
            nu_sat = numpyro.sample("nu_sat", dist.Gamma(1, 1))
            
            # Competitive effect parameters
            if competitor_data is not None:
                phi_comp = numpyro.sample("phi_comp", dist.Gamma(1, 1))
                kappa_comp = numpyro.sample("kappa_comp", dist.Gamma(1, 1))
            else:
                phi_comp = kappa_comp = None
        
        # Cross-channel interaction parameters with shrinkage
        if n_interactions > 0:
            tau_global = numpyro.sample("tau_global", dist.HalfCauchy(0.1))
            delta_interact = numpyro.sample("delta_interact", 
                                          dist.Normal(0, tau_global).expand([n_interactions]))
        else:
            delta_interact = jnp.array([])
        
        # Extra features coefficients
        if n_extra_features > 0:
            gamma_extra = numpyro.sample("gamma_extra", 
                                       dist.Normal(0, 1).expand([n_extra_features]))
        
        # Error term
        sigma = numpyro.sample("sigma", dist.HalfNormal(1))
        
        
        # Apply media transformations
        transformed_media = []
        for i in range(n_media_channels):
            # Power adstock
            adstocked = self._apply_power_adstock(
                media_data[:, i], lambda_adstock[i], rho_power[i]
            )
            
            # Hill saturation
            saturated = self._apply_hill_saturation(
                adstocked, K_sat[i], S_sat[i], nu_sat[i]
            )
            
            # Competitive effect
            if competitor_data is not None:
                final_media = self._apply_competitive_effect(
                    saturated, competitor_data[:, i], phi_comp[i], kappa_comp[i]
                )
            else:
                final_media = saturated
            
            transformed_media.append(final_media)
        
        transformed_media = jnp.stack(transformed_media, axis=1)
        
        # === MODEL PREDICTION ===
        
        # Time trend
        time_arange = jnp.arange(n_time_periods, dtype=jnp.float32)
        trend_contribution = trend_coef * time_arange
        
        # Seasonality contribution
        seasonality_contribution = jnp.sum(seasonality_coef * seasonality_features, axis=1)
        
        # Time-varying media coefficients
        seasonal_factor = jnp.cos(2 * jnp.pi * time_arange / 52.0) 
        beta_media = beta_media[None, :]  
        seasonal_factor = seasonal_factor[:, None]  
        seasonal_multiplier = seasonal_multiplier[None, :]  
        time_varying_beta = beta_media * (1 + seasonal_multiplier * seasonal_factor)

        
        # Media contribution with time-varying coefficients
        media_contribution = jnp.sum(time_varying_beta * transformed_media, axis=1)
        
        # Cross-channel interactions
        interaction_contribution = self._compute_cross_channel_interactions(
            transformed_media, delta_interact
        )
        
        # Extra features contribution
        extra_contribution = (jnp.sum(gamma_extra * extra_features, axis=1) 
                            if n_extra_features > 0 else 0.0)
        
        # Total prediction
        mu = (intercept + 
              trend_contribution + 
              seasonality_contribution + 
              media_contribution + 
              interaction_contribution + 
              extra_contribution)
        
        # Observation model
        numpyro.sample("target", dist.Normal(mu, sigma), obs=target_data)
    
    def fit(self, media_data: np.ndarray, 
            target_data: np.ndarray,
            extra_features: Optional[np.ndarray] = None,
            competitor_data: Optional[np.ndarray] = None,
            num_warmup: int = 1000,
            num_samples: int = 1000,
            num_chains: int = 2,
            target_accept_prob: float = 0.8,
            max_tree_depth: int = 10,
            use_svi: bool = False,
            svi_num_steps: int = 10000) -> None:
        """
        Fits the MMM model
        
        Args:
            media_data: Media spend data [n_time_periods, n_media_channels]
            target_data: Target variable (e.g., sales, conversions)
            extra_features: Additional control variables
            competitor_data: Competitor spend data
            num_warmup: Number of warmup samples for MCMC
            num_samples: Number of samples for MCMC
            num_chains: Number of MCMC chains
            target_accept_prob: Target acceptance probability
            max_tree_depth: Maximum tree depth for NUTS
            use_svi: Whether to use SVI instead of MCMC
            svi_num_steps: Number of SVI optimization steps
        """
        
        # Convert to JAX arrays
        media_data_jax = jnp.array(media_data)
        target_data_jax = jnp.array(target_data)
        extra_features_jax = jnp.array(extra_features) if extra_features is not None else None
        competitor_data_jax = jnp.array(competitor_data) if competitor_data is not None else None
        
        # Model arguments
        model_args = {
            'media_data': media_data_jax,
            'target_data': target_data_jax,
            'extra_features': extra_features_jax,
            'competitor_data': competitor_data_jax
        }
        
        if use_svi:
            # Variational inference
            self.guide = AutoLowRankMultivariateNormal(self._model_definition, rank=20)
            optimizer = numpyro.optim.Adam(step_size=0.01)
            self.svi = SVI(self._model_definition, self.guide, optimizer, loss=Trace_ELBO())
            
            # Run SVI
            svi_result = self.svi.run(
                jax.random.PRNGKey(0), 
                svi_num_steps, 
                **model_args
            )
            
            # Get posterior samples
            posterior_samples = Predictive(
                self.guide, 
                params=svi_result.params, 
                num_samples=num_samples
            )(jax.random.PRNGKey(1), **{k: v for k, v in model_args.items() if k != 'target_data'})
            
            self.posterior_samples = posterior_samples
            
        else:
            nuts_kernel = NUTS(
                self._model_definition,
                target_accept_prob=target_accept_prob,
                max_tree_depth=max_tree_depth
            )
            
            self.mcmc = MCMC(
                nuts_kernel,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains
            )
            
            # Run MCMC
            self.mcmc.run(jax.random.PRNGKey(0), **model_args)
            self.posterior_samples = self.mcmc.get_samples()
    
    def predict(self, media_data: np.ndarray, 
                extra_features: Optional[np.ndarray] = None,
                competitor_data: Optional[np.ndarray] = None,
                num_samples: int = 1000) -> Dict[str, np.ndarray]:
        """
        Generate predictions from the fitted model
        
        Returns:
            Dictionary with predictions and uncertainty intervals
        """
        
        if self.posterior_samples is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to JAX arrays
        media_data_jax = jnp.array(media_data)
        extra_features_jax = jnp.array(extra_features) if extra_features is not None else None
        competitor_data_jax = jnp.array(competitor_data) if competitor_data is not None else None
        
        # Predictive model
        predictive = Predictive(
            self._model_definition,
            posterior_samples=self.posterior_samples,
            num_samples=num_samples
        )
        
        # Generate predictions
        predictions = predictive(
            jax.random.PRNGKey(1),
            media_data=media_data_jax,
            extra_features=extra_features_jax,
            competitor_data=competitor_data_jax
        )
        
        pred_target = predictions['target']
        
        return {
            'predictions': np.array(pred_target),
            'mean': np.mean(pred_target, axis=0),
            'median': np.median(pred_target, axis=0),
            'lower_ci': np.percentile(pred_target, 2.5, axis=0),
            'upper_ci': np.percentile(pred_target, 97.5, axis=0)
        }
    
    def compute_media_contributions(self, media_data: np.ndarray,
                                  extra_features: Optional[np.ndarray] = None,
                                  competitor_data: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Compute individual media channel contributions
        
        Returns:
            Dictionary with contribution decomposition
        """
        
        if self.posterior_samples is None:
            raise ValueError("Model must be fitted before computing contributions")
        

        media_data_jax = jnp.array(media_data)
        extra_features_jax = jnp.array(extra_features) if extra_features is not None else None
        competitor_data_jax = jnp.array(competitor_data) if competitor_data is not None else None
        
        n_time_periods, n_media_channels = media_data.shape
        n_samples = len(list(self.posterior_samples.values())[0])
        

        beta_media = self.posterior_samples['beta_media']
        seasonal_multiplier = self.posterior_samples['seasonal_multiplier']
        lambda_adstock = self.posterior_samples['lambda_adstock']
        rho_power = self.posterior_samples['rho_power']
        K_sat = self.posterior_samples['K_sat']
        S_sat = self.posterior_samples['S_sat']
        nu_sat = self.posterior_samples['nu_sat']
        
        # Compute contributions for each sample
        contributions = np.zeros((n_samples, n_time_periods, n_media_channels))
        
        for s in range(n_samples):
            time_arange = jnp.arange(n_time_periods, dtype=jnp.float32)
            seasonal_factor = jnp.cos(2 * jnp.pi * time_arange / 52.0)
            
            for c in range(n_media_channels):
                # Apply transformations
                adstocked = self._apply_power_adstock(
                    media_data_jax[:, c], lambda_adstock[s, c], rho_power[s, c]
                )
                
                saturated = self._apply_hill_saturation(
                    adstocked, K_sat[s, c], S_sat[s, c], nu_sat[s, c]
                )
                
                if competitor_data is not None:
                    phi_comp = self.posterior_samples['phi_comp'][s, c]
                    kappa_comp = self.posterior_samples['kappa_comp'][s, c]
                    final_media = self._apply_competitive_effect(
                        saturated, competitor_data_jax[:, c], phi_comp, kappa_comp
                    )
                else:
                    final_media = saturated
                
                # Time-varying coefficient
                seasonal_factor = seasonal_factor.reshape(-1, 1, 1)

                time_varying_beta = beta_media[None, :, :] * ( 1 + seasonal_multiplier[None, :, :] * seasonal_factor)

                # Channel contribution
                final_media_reshaped = final_media[:, None, None] 
                contributions[s, :, c] = (time_varying_beta[:, s, c] * media_data[:, c])

        return {
            'contributions': contributions,
            'mean_contributions': np.mean(contributions, axis=0),
            'total_contribution': np.sum(np.mean(contributions, axis=0), axis=1)
        }
    
    def optimize_budget(self, total_budget: float,
                   media_data_historical: np.ndarray,
                   extra_features: Optional[np.ndarray] = None,
                   competitor_data: Optional[np.ndarray] = None,
                   bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                   risk_tolerance: float = 0.1,
                   num_samples: int = 500,
                   planning_horizon: int = 4,
                   spend_distribution: str = 'equal') -> Dict[str, Any]:
        """
        Optimize budget allocation with proper multi-period evaluation
        
        Args:
            total_budget: Total budget to allocate
            media_data_historical: Historical media data for baseline
            bounds: Optional bounds for each channel {channel_idx: (min, max)}
            risk_tolerance: Risk tolerance parameter (higher = more risk averse)
            num_samples: Number of posterior samples for uncertainty quantification
            planning_horizon: Number of future periods to optimize over
            spend_distribution: How to distribute spend over time ('equal', 'front_loaded', 'back_loaded')
            
        Returns:
            Dictionary with optimal allocation and expected returns
        """
        
        if self.posterior_samples is None:
            raise ValueError("Model must be fitted before optimizing budget")
        
        n_media_channels = media_data_historical.shape[1]
        n_time_periods = media_data_historical.shape[0]
        
        if bounds is None:
            bounds_list = [(0, total_budget) for _ in range(n_media_channels)]
        else:
            bounds_list = [bounds.get(i, (0, total_budget)) for i in range(n_media_channels)]
        
        # Sample posterior indices once for consistent evaluation
        total_posterior_samples = len(list(self.posterior_samples.values())[0])
        sample_indices = np.random.choice(
            total_posterior_samples, 
            size=min(num_samples, total_posterior_samples), 
            replace=False
        )
        
        time_weights = self._get_time_weights(planning_horizon, spend_distribution)
        
        def objective(allocation):
            """Multi-period risk-adjusted objective function"""
            
            future_media = np.zeros((planning_horizon, n_media_channels))
            
            # Distribute allocation across time periods
            for t in range(planning_horizon):
                future_media[t] = allocation * time_weights[t]
            
            extended_media = np.vstack([media_data_historical, future_media])
            
            # Extend other features if provided
            if extra_features is not None:
                extended_extra = np.vstack([
                    extra_features, 
                    np.tile(extra_features[-1], (planning_horizon, 1))
                ])
            else:
                extended_extra = None
                
            if competitor_data is not None:
                extended_competitor = np.vstack([
                    competitor_data,
                    np.tile(competitor_data[-1], (planning_horizon, 1))
                ])
            else:
                extended_competitor = None
            
            # Predict outcomes across all samples
            predicted_outcomes = []
            
            for idx in sample_indices:
                sample = {k: v[idx] for k, v in self.posterior_samples.items()}
                
                # Predict for extended period
                pred = self._predict_with_sample(
                    sample, extended_media, extended_extra, extended_competitor
                )
                
                # Sum incremental returns from future periods
                baseline_pred = self._predict_with_sample(
                    sample, media_data_historical, extra_features, competitor_data
                )
                
                # Calculate incremental lift
                incremental_return = np.sum(pred[n_time_periods:]) - np.sum(baseline_pred[-planning_horizon:])
                predicted_outcomes.append(incremental_return)
            
            predicted_outcomes = np.array(predicted_outcomes)
            
            # Risk-adjusted return with diminishing returns penalty
            mean_return = np.mean(predicted_outcomes)
            std_return = np.std(predicted_outcomes)
            
            # Add saturation penalty to encourage diversification
            saturation_penalty = self._calculate_saturation_penalty(allocation)
            
            return mean_return - risk_tolerance * std_return - saturation_penalty
        
        #channel effectiveness
        self._diagnose_channel_effectiveness(total_budget, objective)
        
        # Budget constraint
        budget_constraint = {'type': 'eq', 'fun': lambda x: total_budget - np.sum(x)}
        
        # Initial guess 
        np.random.seed(42)  
        initial_guess = np.random.dirichlet(np.ones(n_media_channels)) * total_budget
        
        best_result = None
        best_value = -np.inf
        
        for seed in range(5): 
            np.random.seed(seed)
            init_guess = np.random.dirichlet(np.ones(n_media_channels)) * total_budget
            
            result = minimize(
                lambda x: -objective(x), 
                x0=init_guess,
                bounds=bounds_list,
                constraints=budget_constraint,
                method='SLSQP',
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success and -result.fun > best_value:
                best_result = result
                best_value = -result.fun
        
        if best_result is None:
            raise RuntimeError("Optimization failed for all starting points")
        
        optimal_allocation = best_result.x
        expected_return = objective(optimal_allocation)  # Evaluate directly
        
        # Calculate metrics
        roi_per_channel = self._calculate_channel_roi(optimal_allocation)
        saturation_levels = self._calculate_saturation_levels(optimal_allocation)
        
        return {
            'optimal_allocation': optimal_allocation,
            'expected_return': expected_return,
            'allocation_shares': optimal_allocation / total_budget,
            'roi_per_channel': roi_per_channel,
            'saturation_levels': saturation_levels,
            'optimization_success': best_result.success,
            'optimization_message': best_result.message,
            'planning_horizon': planning_horizon,
            'total_iterations': best_result.nit
        }

    def _get_time_weights(self, planning_horizon: int, distribution: str) -> np.ndarray:
        """Generate time distribution weights"""
        if distribution == 'equal':
            return np.ones(planning_horizon) / planning_horizon
        elif distribution == 'front_loaded':
            weights = np.exp(-np.arange(planning_horizon) * 0.3)
            return weights / weights.sum()
        elif distribution == 'back_loaded':
            weights = np.exp(np.arange(planning_horizon) * 0.3)
            return weights / weights.sum()
        else:
            return np.ones(planning_horizon) / planning_horizon

    def _calculate_saturation_penalty(self, allocation: np.ndarray) -> float:
        """Calculate penalty for over-saturated channels"""
        if 'K_sat' not in self.posterior_samples:
            return 0.0
        
        K_sat_mean = np.mean(self.posterior_samples['K_sat'], axis=0)
        S_sat_mean = np.mean(self.posterior_samples['S_sat'], axis=0)
        
        penalty = 0.0
        for c in range(len(allocation)):
            # If allocation approaches saturation point, add penalty
            saturation_ratio = allocation[c] / (K_sat_mean[c] + 1e-8)
            if saturation_ratio > 0.8:  # 80% of saturation
                penalty += (saturation_ratio - 0.8) ** 2 * 1000
        
        return penalty

    def _diagnose_channel_effectiveness(self, total_budget: float, objective_func):
        """Diagnostic function to check individual channel effectiveness"""
        n_channels = len(self.posterior_samples['beta_media'][0])
        
        print("Channel Effectiveness Diagnosis:")
        print("-" * 50)
        
        for i in range(n_channels):
            # Test channel with full budget
            test_allocation = np.zeros(n_channels)
            test_allocation[i] = total_budget
            channel_return = objective_func(test_allocation)
            
            print(f"Channel {i}: Full budget return = {channel_return:.2f}")
        
        # Test equal allocation
        equal_allocation = np.full(n_channels, total_budget / n_channels)
        equal_return = objective_func(equal_allocation)
        print(f"Equal allocation return = {equal_return:.2f}")
        print("-" * 50)

    def _calculate_channel_roi(self, allocation: np.ndarray) -> np.ndarray:
        """Calculate ROI for each channel at given allocation"""
        n_channels = len(allocation)
        roi_values = np.zeros(n_channels)
        
        for i in range(n_channels):
            if allocation[i] > 0:
                # Calculate marginal ROI
                test_allocation = allocation.copy()
                test_allocation[i] *= 1.1  # 10% increase
                
                # This would need the objective function accessible
                # Simplified version - you'd need to implement proper marginal calculation
                roi_values[i] = allocation[i] / (allocation[i] + 1e-8)  # Placeholder
        
        return roi_values

    def _calculate_saturation_levels(self, allocation: np.ndarray) -> np.ndarray:
        """Calculate saturation level for each channel"""
        if 'K_sat' not in self.posterior_samples:
            return np.zeros(len(allocation))
        
        K_sat_mean = np.mean(self.posterior_samples['K_sat'], axis=0)
        saturation_levels = allocation / (K_sat_mean + allocation)
        
        return saturation_levels
    
    def _predict_with_sample(self, sample: Dict[str, float], 
                           media_data: np.ndarray,
                           extra_features: Optional[np.ndarray] = None,
                           competitor_data: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict with a single posterior sample"""
        
        media_data_jax = jnp.array(media_data)
        extra_features_jax = jnp.array(extra_features) if extra_features is not None else None
        competitor_data_jax = jnp.array(competitor_data) if competitor_data is not None else None
        
        n_time_periods, n_media_channels = media_data.shape
        
        # Seasonality features
        seasonality_features = self._create_seasonality_features(n_time_periods)
        
        # Extract parameters
        intercept = sample['intercept']
        trend_coef = sample['trend_coef']
        seasonality_coef = sample['seasonality_coef']
        beta_media = sample['beta_media']
        seasonal_multiplier = sample['seasonal_multiplier']
        
        # Transform media data
        transformed_media = []
        for i in range(n_media_channels):
            # Apply transformations
            adstocked = self._apply_power_adstock(
                media_data_jax[:, i], 
                sample['lambda_adstock'][i], 
                sample['rho_power'][i]
            )
            
            saturated = self._apply_hill_saturation(
                adstocked, 
                sample['K_sat'][i], 
                sample['S_sat'][i], 
                sample['nu_sat'][i]
            )
            
            if competitor_data is not None:
                final_media = self._apply_competitive_effect(
                    saturated, 
                    competitor_data_jax[:, i], 
                    sample['phi_comp'][i], 
                    sample['kappa_comp'][i]
                )
            else:
                final_media = saturated
            
            transformed_media.append(final_media)
        
        transformed_media = jnp.stack(transformed_media, axis=1)
        
        # Compute prediction
        time_arange = jnp.arange(n_time_periods, dtype=jnp.float32)
        
        # Time trend
        trend_contribution = trend_coef * time_arange
        
        # Seasonality
        seasonality_contribution = jnp.sum(seasonality_coef * seasonality_features, axis=1)
        
        # Time-varying media effects
        seasonal_factor = jnp.cos(2 * jnp.pi * time_arange / 52.0)  # (104,)
        seasonal_effect = 1 + seasonal_multiplier * seasonal_factor[:, None]  # (104, 4)
        time_varying_beta = beta_media[None, :] * seasonal_effect  # (104, 4)

        media_contribution = jnp.sum(time_varying_beta * transformed_media, axis=1)
        
        # Cross-channel interactions
        n_interactions = n_media_channels * (n_media_channels - 1) // 2
        if n_interactions > 0:
            delta_interact = sample['delta_interact']
            interaction_contribution = self._compute_cross_channel_interactions(
                transformed_media, delta_interact
            )
        else:
            interaction_contribution = 0.0
        
        # Extra features
        extra_contribution = (jnp.sum(sample['gamma_extra'] * extra_features_jax, axis=1) 
                            if extra_features_jax is not None else 0.0)
        
        # Total prediction
        mu = (intercept + 
              trend_contribution + 
              seasonality_contribution + 
              media_contribution + 
              interaction_contribution + 
              extra_contribution)
        
        return np.array(mu)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the fitted model"""
        
        if self.posterior_samples is None:
            raise ValueError("Model must be fitted before getting summary")
        
        summary = {}
        
        for param_name, param_values in self.posterior_samples.items():
            if param_values.ndim == 1:
                summary[param_name] = {
                    'mean': float(np.mean(param_values)),
                    'std': float(np.std(param_values)),
                    'median': float(np.median(param_values)),
                    'q025': float(np.percentile(param_values, 2.5)),
                    'q975': float(np.percentile(param_values, 97.5))
                }
            else:
                summary[param_name] = {
                    'mean': np.mean(param_values, axis=0).tolist(),
                    'std': np.std(param_values, axis=0).tolist(),
                    'median': np.median(param_values, axis=0).tolist(),
                    'q025': np.percentile(param_values, 2.5, axis=0).tolist(),
                    'q975': np.percentile(param_values, 97.5, axis=0).tolist()
                }
        
        return summary
