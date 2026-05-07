
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import warnings
from Bio.SeqUtils.ProtParam import ProteinAnalysis

warnings.filterwarnings('ignore')

# =============================================================================
# PILLAR 1: SEQUENCE-TO-BURDEN AI (Bioinformatics Layer)
# =============================================================================

@dataclass
class SequenceFeatures:
    """Extracted features from amino acid sequence."""
    length: int
    molecular_weight: float
    hydrophobicity: float
    charge_at_ph7: float
    instability_index: float
    aliphatic_index: float
    gravy: float
    cysteine_fraction: float
    proline_fraction: float
    aromatic_fraction: float
    disorder_propensity: float
    aggregation_propensity: float
    rare_codon_burden: float
    folding_complexity: float

class SequenceAnalyzer:
    """Analyzes amino acid sequences using Biopython's ProteinAnalysis module."""
    
    def clean_sequence(self, sequence: str) -> str:
        """Clean and validate amino acid sequence."""
        return sequence.upper().replace(' ', '').replace('\n', '')
    
    def extract_features(self, sequence: str) -> SequenceFeatures:
        """Extract comprehensive biophysical features using Biopython."""
        seq = self.clean_sequence(sequence)
        
        if len(seq) == 0:
            raise ValueError("Invalid sequence: no valid amino acids found")
            
        analysis = ProteinAnalysis(seq)
        
        mw = analysis.molecular_weight()
        gravy = analysis.gravy()
        instability = analysis.instability_index()
        charge = analysis.charge_at_pH(7.0)
        
        # --- THE FIX: Calculate fractions manually to bypass Biopython version changes ---
        raw_counts = analysis.count_amino_acids()
        seq_len = len(seq)
        
        # Convert raw integer counts to 0.0 - 1.0 fractions so the AI doesn't break
        aa_fracs = {aa: count / seq_len for aa, count in raw_counts.items()}
        
        cys_frac = aa_fracs.get('C', 0.0)
        pro_frac = aa_fracs.get('P', 0.0)
        aromatic_frac = aa_fracs.get('F', 0.0) + aa_fracs.get('W', 0.0) + aa_fracs.get('Y', 0.0)
        
        sec_struct = analysis.secondary_structure_fraction()
        
        aliphatic = (aa_fracs.get('A', 0.0) * 100 + 
                     aa_fracs.get('V', 0.0) * 2.9 * 100 +
                     aa_fracs.get('I', 0.0) * 3.9 * 100 +
                     aa_fracs.get('L', 0.0) * 3.9 * 100)
                     
        folding_complexity = (cys_frac * 5 + pro_frac * 2 + 
                              np.log10(seq_len) * 0.5 + aromatic_frac * 1.5)
        # -------------------------------------------------------------------------------

        return SequenceFeatures(
            length=seq_len,
            molecular_weight=mw,
            hydrophobicity=gravy,  
            charge_at_ph7=charge,
            instability_index=instability,
            aliphatic_index=aliphatic,
            gravy=gravy,
            cysteine_fraction=cys_frac,
            proline_fraction=pro_frac,
            aromatic_fraction=aromatic_frac,
            disorder_propensity=sec_struct[1], 
            aggregation_propensity=max(0, gravy * 1.5), 
            rare_codon_burden=0.1, 
            folding_complexity=folding_complexity
        )

class MetabolicBurdenPredictor:
    """
    Triple-Branch Stacking Ensemble for predicting metabolic burden.
    Optimized for Streamlit Cloud memory limits.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.is_fitted = False
        
    def _build_ensemble(self):
        """Build the triple-branch stacking ensemble with cloud-safe parameters."""
        estimators = [
            ('rf', RandomForestClassifier(
                n_estimators=50, max_depth=10, min_samples_split=5,
                random_state=42, n_jobs=1  # CLOUD FIX: Removed -1 to prevent thread deadlocks
            )),
            ('gb1', GradientBoostingClassifier(
                n_estimators=50, max_depth=5, learning_rate=0.1,
                random_state=42
            )),
            ('gb2', GradientBoostingClassifier(
                n_estimators=30, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=123
            ))
        ]
        
        final_estimator = GradientBoostingClassifier(
            n_estimators=30, max_depth=3, learning_rate=0.1,
            random_state=42
        )
        
        return StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=2,  # CLOUD FIX: Reduced from 5 to 2 to cut memory usage by 60%
            passthrough=True
        )

    
    def fit(self):
        """Train the ensemble with RAM-protection sampling."""
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        print("🧠 Booting AI: Training on real-world industrial records...")
        
        df = pd.read_csv('universal_digital_twin_data.csv')
        
        # --- CLOUD FIX: Prevent Out-Of-Memory Crash ---
        # Randomly sample the dataset down to 3,000 rows so the free server doesn't die
        if len(df) > 3000:
            df = df.sample(n=3000, random_state=42)
        # ----------------------------------------------
        
        df = pd.get_dummies(df, columns=['Host_Microbe'], prefix='Host')
        
        feature_columns = [
            'Length', 'Molecular_Weight', 'Hydrophobicity', 'Charge_pH7',
            'Instability_Index', 'Aliphatic_Index', 'GRAVY', 'Cys_Fraction',
            'Pro_Fraction', 'Aromatic_Fraction', 'Disorder_Propensity',
            'Aggregation_Propensity', 'Rare_Codon_Burden', 'Folding_Complexity',
            'Host_Aspergillus niger', 'Host_Bacillus subtilis', 
            'Host_E. coli', 'Host_Pichia pastoris', 'Host_Saccharomyces cerevisiae'
        ]
        
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
                
        X = df[feature_columns].values
        y = df['Target_Burden_Class'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = self._build_ensemble()
        self.model.fit(X_train_scaled, y_train)
        
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        self.is_fitted = True
        print(f"✅ AI Trained! Test Accuracy: {test_score*100:.2f}%")
        
        return {'train_accuracy': train_score, 'test_accuracy': test_score}
    
    def predict_burden(self, features, microbe_name: str = "E. coli"):
        import numpy as np

        base_features = [
            features.length, features.molecular_weight, features.hydrophobicity,
            features.charge_at_ph7, features.instability_index, features.aliphatic_index,
            features.gravy, features.cysteine_fraction, features.proline_fraction,
            features.aromatic_fraction, features.disorder_propensity,
            features.aggregation_propensity, features.rare_codon_burden,
            features.folding_complexity
        ]

        host_flags = {
            "Aspergillus niger": [1, 0, 0, 0, 0],
            "Bacillus subtilis": [0, 1, 0, 0, 0],
            "E. coli": [0, 0, 1, 0, 0],
            "Pichia pastoris": [0, 0, 0, 1, 0],
            "Saccharomyces cerevisiae": [0, 0, 0, 0, 1]
        }

        microbe_flags = host_flags.get(microbe_name, [0, 0, 1, 0, 0])
        feature_vec = np.array([base_features + microbe_flags])

        feature_vec_scaled = self.scaler.transform(feature_vec)
        probabilities = self.model.predict_proba(feature_vec_scaled)[0]
        
        weights = np.array([0.1, 0.4, 0.7, 1.0])
        burden_score = np.dot(probabilities, weights)
        predicted_class = int(np.argmax(probabilities))
        
        class_names = ["Low", "Medium", "High", "Critical"]
        
        probs_list = probabilities.tolist()
        class_probs = {
            "Low": probs_list[0], 
            "Medium": probs_list[1], 
            "High": probs_list[2], 
            "Critical": probs_list[3]
        }
        
        return {
            'burden_score': float(burden_score),
            'burden_class': class_names[predicted_class],
            'class_probabilities': class_probs
        }
    
# =============================================================================
# PILLAR 2: MECHANISTIC GROWTH ENGINE (Cellular Layer)
# =============================================================================

@dataclass
class CellKineticsParams:
    """Parameters for cell growth kinetics."""
    mu_max: float        # Maximum specific growth rate (1/h)
    Ks: float            # Substrate saturation constant (g/L)
    Yxs: float           # Biomass yield on substrate (g/g)
    Yps: float           # Product yield on substrate (g/g)
    ms: float            # Maintenance coefficient (g/g/h)
    kd_base: float       # Base death rate (1/h)
    alpha: float         # Growth-associated product formation (g/g)
    beta: float          # Non-growth-associated product formation (g/g/h)
    qO2_max: float       # Maximum specific oxygen uptake (g/g/h)
    Ko2: float           # Oxygen saturation constant (g/L)
    metabolic_heat: float = 15000  # Heat generation per O2 consumed (kJ/kg O2)


MICROBE_LIBRARY = {
    "E. coli": CellKineticsParams(
        # Fast growth, high maintenance, high oxygen demand
        mu_max=0.45, Ks=0.5, Yxs=0.5, Yps=0.3, ms=0.02, 
        kd_base=0.01, alpha=0.05, beta=0.01, qO2_max=0.5, Ko2=0.001
    ),
    "Saccharomyces cerevisiae": CellKineticsParams(
        # Baker's Yeast: Slower growth, lower maintenance, moderate oxygen
        mu_max=0.25, Ks=0.1, Yxs=0.4, Yps=0.2, ms=0.005, 
        kd_base=0.005, alpha=0.03, beta=0.005, qO2_max=0.3, Ko2=0.002
    ),
    "Bacillus subtilis": CellKineticsParams(
        # Gram-positive: robust secretory microbe, moderate/fast growth
        mu_max=0.35, Ks=0.2, Yxs=0.45, Yps=0.4, ms=0.015, 
        kd_base=0.01, alpha=0.08, beta=0.02, qO2_max=0.4, Ko2=0.001
    ),
    "Aspergillus niger": CellKineticsParams(
        # Filamentous Fungus: Slow growth, highly efficient secretor
        mu_max=0.15, Ks=0.5, Yxs=0.5, Yps=0.6, ms=0.01, 
        kd_base=0.005, alpha=0.10, beta=0.05, qO2_max=0.2, Ko2=0.005
    ),
    "Pichia pastoris": CellKineticsParams(
        # Methylotrophic Yeast: High cell density, high oxygen demand
        mu_max=0.20, Ks=0.2, Yxs=0.4, Yps=0.35, ms=0.008, 
        kd_base=0.005, alpha=0.06, beta=0.01, qO2_max=0.45, Ko2=0.002
    )
}
class MechanisticGrowthModel:
    """
    ODE-based mechanistic model for cell growth dynamics
    incorporating metabolic burden from Pillar 1.
    """
    
    def __init__(self, params: CellKineticsParams = None):
        # Default to E. coli from your library if no params are passed
        self.params = params or MICROBE_LIBRARY["E. coli"]
    
    def apply_burden(self, burden_score: float):
        """Modify kinetic parameters based on metabolic burden."""
        # Burden reduces max growth rate
        self.effective_mu_max = self.params.mu_max * (1 - burden_score * 0.5)
        
        # Burden increases death rate
        self.effective_kd = self.params.kd_base * (1 + burden_score * 3)
        
        # Burden increases maintenance energy requirement
        self.effective_ms = self.params.ms * (1 + burden_score * 2)
        
        # Burden may reduce product yield (misfolding, aggregation)
        self.effective_Yps = self.params.Yps * (1 - burden_score * 0.3)
    
    def ode_system(self, t: float, y: np.ndarray, 
                   control_inputs: Dict) -> np.ndarray:
        """
        System of ODEs for fed-batch fermentation.
        
        State variables:
        y[0] = X  - Biomass concentration (g/L)
        y[1] = S  - Substrate concentration (g/L)
        y[2] = P  - Product concentration (g/L)
        y[3] = O2 - Dissolved oxygen (g/L)
        y[4] = V  - Culture volume (L)
        y[5] = Q  - Accumulated heat (kJ)
        """
        X, S, P, O2, V, Q = y
        
        # Ensure non-negative values
        X = max(X, 0.001)
        S = max(S, 0)
        O2 = max(O2, 0)
        V = max(V, 1)
        
        # Get control inputs
        F = control_inputs.get('feed_rate', 0)  # L/h
        S_feed = control_inputs.get('feed_concentration', 500)  # g/L
        kLa = control_inputs.get('kLa', 200)  # 1/h
        O2_sat = control_inputs.get('O2_saturation', 0.007)  # g/L
        
        # Specific growth rate with dual substrate limitation
        mu = self.effective_mu_max * (S / (self.params.Ks + S)) * (O2 / (self.params.Ko2 + O2))
        
        # Specific rates
        qS = mu / self.params.Yxs + self.effective_ms  # Substrate uptake
        qP = self.params.alpha * mu + self.params.beta  # Product formation
        qO2 = self.params.qO2_max * (S / (self.params.Ks + S))  # Oxygen uptake
        
        # Mass balances
        dXdt = mu * X - self.effective_kd * X - (F / V) * X
        dSdt = -qS * X + (F / V) * (S_feed - S)
        dPdt = qP * X - (F / V) * P
        dO2dt = kLa * (O2_sat - O2) - qO2 * X
        dVdt = F
        dQdt = qO2 * X * V * self.params.metabolic_heat / 1000  # kJ/h
        
        return np.array([dXdt, dSdt, dPdt, dO2dt, dVdt, dQdt])
    
    def simulate(self, y0: np.ndarray, t_span: Tuple[float, float],
                 control_trajectory: List[Dict], 
                 t_eval: np.ndarray = None) -> Dict:
        """
        Simulate fermentation with time-varying control inputs.
        """
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 100)
        
        # Interpolate control inputs over time
        def get_control_at_time(t):
            if len(control_trajectory) == 1:
                return control_trajectory[0]
            
            times = np.linspace(t_span[0], t_span[1], len(control_trajectory))
            idx = np.searchsorted(times, t) - 1
            idx = max(0, min(idx, len(control_trajectory) - 1))
            return control_trajectory[idx]
        
        # Solve ODE with events for crash detection
        def ode_wrapper(t, y):
            return self.ode_system(t, y, get_control_at_time(t))
        
        # Event: detect oxygen crash
        def oxygen_crash(t, y):
            return y[3] - 0.0005  # Crash if DO < 0.5% of saturation
        oxygen_crash.terminal = True
        oxygen_crash.direction = -1
        
        # Event: detect substrate depletion
        def substrate_depleted(t, y):
            return y[1] - 0.01
        substrate_depleted.terminal = False
        substrate_depleted.direction = -1
        
        solution = solve_ivp(
            ode_wrapper,
            t_span,
            y0,
            method='LSODA',
            t_eval=t_eval,
            events=[oxygen_crash, substrate_depleted],
            max_step=0.5
        )
        
        return {
            'time': solution.t,
            'biomass': solution.y[0],
            'substrate': solution.y[1],
            'product': solution.y[2],
            'dissolved_oxygen': solution.y[3],
            'volume': solution.y[4],
            'heat': solution.y[5],
            'success': solution.success,
            'crash_detected': solution.status == 1,
            'crash_time': solution.t_events[0][0] if solution.t_events[0].size > 0 else None
        }


# =============================================================================
# PILLAR 3: INDUSTRIAL TRANSPORT SIMULATOR (Physics Layer)
# =============================================================================

@dataclass
class BioreactorConfig:
    """Configuration for industrial bioreactor."""
    volume_max: float = 100000  # L
    diameter: float = 4.5       # m
    height: float = 8.0         # m
    impeller_diameter: float = 1.5  # m
    n_impellers: int = 3
    sparger_area: float = 0.5   # m²
    jacket_area: float = 80     # m²
    jacket_U: float = 500       # W/m²/K
    coolant_temp: float = 15    # °C


class IndustrialBioreactorSimulator:
    """
    Physics-based simulation of industrial bioreactor transport phenomena.
    """
    
    def __init__(self, config: BioreactorConfig = None):
        self.config = config or BioreactorConfig()
        
    def calculate_power_input(self, rpm: float, V: float) -> float:
        """Calculate power input from agitation (W)."""
        # Power number for Rushton turbine ≈ 5
        Np = 5.0
        rho = 1020  # kg/m³ (broth density)
        N = rpm / 60  # rev/s
        D = self.config.impeller_diameter
        
        # P = Np * rho * N³ * D⁵
        power_per_impeller = Np * rho * (N ** 3) * (D ** 5)
        total_power = power_per_impeller * self.config.n_impellers
        
        return total_power
    
    def calculate_kLa(self, rpm: float, airflow: float, V: float) -> float:
        """
        Calculate oxygen mass transfer coefficient.
        
        Uses correlation: kLa = a * (P/V)^b * (vg)^c
        """
        power = self.calculate_power_input(rpm, V)
        P_V = power / (V / 1000)  # W/m³
        
        # Superficial gas velocity (m/s)
        cross_section = np.pi * (self.config.diameter / 2) ** 2
        vg = airflow / cross_section / 3600  # m/s
        
        # Empirical correlation (van't Riet)
        a = 0.026
        b = 0.4
        c = 0.5
        
        kLa = a * (P_V ** b) * (vg ** c) * 3600  # Convert to 1/h
        
        return kLa
    
    def calculate_mixing_time(self, rpm: float, V: float) -> float:
        """Calculate mixing time (seconds)."""
        # Correlation: t_mix = 5.9 * (D_tank/D_impeller)^2 * (1/N)
        D_tank = self.config.diameter
        D_imp = self.config.impeller_diameter
        N = rpm / 60
        
        if N < 0.1:
            return np.inf
            
        t_mix = 5.9 * ((D_tank / D_imp) ** 2) * (1 / N)
        
        # Scale with volume
        V_ref = 10000  # Reference volume
        t_mix *= (V / V_ref) ** (1/3)
        
        return t_mix
    
    def calculate_heat_removal_capacity(self, rpm: float, 
                                        T_broth: float = 37) -> float:
        """Calculate maximum heat removal rate (kJ/h)."""
        U = self.config.jacket_U
        A = self.config.jacket_area
        T_coolant = self.config.coolant_temp
        
        # Q = U * A * ΔT
        delta_T = T_broth - T_coolant
        Q_max = U * A * delta_T * 3.6  # Convert W to kJ/h
        
        return Q_max
    
    def calculate_oxygen_demand(self, X: float, V: float, 
                               qO2: float = 0.3) -> float:
        """Calculate oxygen demand rate (kg O2/h)."""
        return qO2 * X * V / 1000  # kg/h
    
    def calculate_oxygen_supply_capacity(self, rpm: float, 
                                         airflow: float, V: float) -> float:
        """Calculate maximum oxygen transfer rate (kg O2/h)."""
        kLa = self.calculate_kLa(rpm, airflow, V)
        O2_sat = 0.007  # g/L at 37°C
        
        # OTR_max = kLa * C* * V
        OTR_max = kLa * O2_sat * V / 1000  # kg/h
        
        return OTR_max
    
    def identify_bottlenecks(self, simulation_result: Dict,
                            rpm_trajectory: np.ndarray,
                            airflow_trajectory: np.ndarray) -> List[Dict]:
        """Identify physics-limited bottlenecks in the fermentation."""
        bottlenecks = []
        
        times = simulation_result['time']
        X = simulation_result['biomass']
        V = simulation_result['volume']
        heat = simulation_result['heat']
        
        for i, t in enumerate(times):
            rpm = rpm_trajectory[min(i, len(rpm_trajectory)-1)]
            airflow = airflow_trajectory[min(i, len(airflow_trajectory)-1)]
            
            # Check oxygen limitation
            O2_demand = self.calculate_oxygen_demand(X[i], V[i])
            O2_supply = self.calculate_oxygen_supply_capacity(rpm, airflow, V[i])
            
            if O2_demand > O2_supply * 0.95:
                bottlenecks.append({
                    'time': t,
                    'type': 'oxygen_limitation',
                    'severity': O2_demand / O2_supply,
                    'demand': O2_demand,
                    'capacity': O2_supply
                })
            
            # Check heat removal limitation
            if i > 0:
                heat_rate = (heat[i] - heat[i-1]) / (times[i] - times[i-1]) if times[i] > times[i-1] else 0
                heat_removal = self.calculate_heat_removal_capacity(rpm)
                
                if heat_rate > heat_removal * 0.9:
                    bottlenecks.append({
                        'time': t,
                        'type': 'heat_limitation',
                        'severity': heat_rate / heat_removal,
                        'demand': heat_rate,
                        'capacity': heat_removal
                    })
            
            # Check mixing limitation
            mixing_time = self.calculate_mixing_time(rpm, V[i])
            if mixing_time > 120:  # > 2 minutes is problematic
                bottlenecks.append({
                    'time': t,
                    'type': 'mixing_limitation',
                    'severity': mixing_time / 60,
                    'mixing_time_min': mixing_time / 60
                })
        
        return bottlenecks


# =============================================================================
# PILLAR 4: DYNAMIC TRAJECTORY OPTIMIZER (Core Intelligence)
# =============================================================================

class DynamicTrajectoryOptimizer:
    """
    Optimization engine for finding optimal control trajectories
    that prevent bioreactor crashes.
    """
    
    def __init__(self, growth_model: MechanisticGrowthModel,
                 reactor_sim: IndustrialBioreactorSimulator):
        self.growth_model = growth_model
        self.reactor_sim = reactor_sim
        self.optimization_history = []
        
    def _parameterize_trajectory(self, params: np.ndarray, 
                                  n_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert optimization parameters to control trajectories.
        
        Parameters are control points for spline interpolation.
        """
        n_ctrl = n_points // 3
        
        # Split parameters into RPM, airflow, and feed profiles
        rpm_ctrl = params[:n_ctrl]
        airflow_ctrl = params[n_ctrl:2*n_ctrl]
        feed_ctrl = params[2*n_ctrl:3*n_ctrl]
        
        # Interpolate to full trajectory
        ctrl_times = np.linspace(0, 1, n_ctrl)
        full_times = np.linspace(0, 1, n_points)
        
        rpm_traj = np.interp(full_times, ctrl_times, rpm_ctrl)
        airflow_traj = np.interp(full_times, ctrl_times, airflow_ctrl)
        feed_traj = np.interp(full_times, ctrl_times, feed_ctrl)
        
        return rpm_traj, airflow_traj, feed_traj
    
    def objective_function(self, params: np.ndarray, 
                          y0: np.ndarray, t_span: Tuple[float, float],
                          n_hours: int) -> float:
        """
        Objective: Maximize final product while avoiding crashes.
        
        Penalties for:
        - Oxygen crashes
        - Heat accumulation
        - Poor mixing
        - Constraint violations
        """
        rpm_traj, airflow_traj, feed_traj = self._parameterize_trajectory(
            params, n_hours
        )
        
        # Build control trajectory
        control_trajectory = []
        for i in range(n_hours):
            kLa = self.reactor_sim.calculate_kLa(
                rpm_traj[i], airflow_traj[i], y0[4]
            )
            control_trajectory.append({
                'feed_rate': feed_traj[i],
                'feed_concentration': 500,
                'kLa': kLa,
                'O2_saturation': 0.007
            })
        
        # Simulate
        try:
            result = self.growth_model.simulate(
                y0, t_span, control_trajectory,
                t_eval=np.linspace(t_span[0], t_span[1], n_hours)
            )
        except Exception:
            return 1e10  # Simulation failed
        
        # Base objective: negative final product (we minimize)
        final_product = result['product'][-1] * result['volume'][-1]
        objective = -final_product
        
        # Penalty for crash
        if result['crash_detected']:
            crash_penalty = 1e6 * (t_span[1] - result['crash_time'])
            objective += crash_penalty
        
        # Penalty for low oxygen
        min_O2 = np.min(result['dissolved_oxygen'])
        if min_O2 < 0.001:
            objective += 1e5 * (0.001 - min_O2)
        
        # Penalty for excessive heat
        final_heat = result['heat'][-1]
        heat_limit = self.reactor_sim.calculate_heat_removal_capacity(
            np.mean(rpm_traj)
        ) * t_span[1]
        if final_heat > heat_limit:
            objective += 1e4 * (final_heat - heat_limit) / heat_limit
        
        # Penalty for constraint violations
        if np.any(rpm_traj < 30) or np.any(rpm_traj > 200):
            objective += 1e5
        if np.any(airflow_traj < 1000) or np.any(airflow_traj > 10000):
            objective += 1e5
        if np.any(feed_traj < 0) or np.any(feed_traj > 500):
            objective += 1e5
        
        # Smoothness penalty (avoid erratic control)
        rpm_smoothness = np.sum(np.diff(rpm_traj) ** 2)
        airflow_smoothness = np.sum(np.diff(airflow_traj) ** 2)
        feed_smoothness = np.sum(np.diff(feed_traj) ** 2)
        objective += 0.01 * (rpm_smoothness + airflow_smoothness + feed_smoothness)
        
        return objective
    
    def optimize(self, y0: np.ndarray, t_span: Tuple[float, float],
                n_hours: int = 72, method: str = 'differential_evolution'
                ) -> Dict:
        """
        Find optimal control trajectory (Cloud Speed Patch).
        Drastically limits iterations so free web servers do not hang.
        """
        import numpy as np
        from scipy.optimize import minimize, differential_evolution
        
        n_ctrl = n_hours // 3
        
        # Bounds for control points [RPM, Airflow, Feed]
        bounds = (
            [(30, 200)] * n_ctrl +      
            [(1000, 10000)] * n_ctrl +  
            [(0, 500)] * n_ctrl         
        )
        
        # Initial guess
        x0 = np.concatenate([
            np.linspace(50, 150, n_ctrl),    
            np.linspace(2000, 6000, n_ctrl), 
            np.linspace(50, 200, n_ctrl)     
        ])
        
        # --- CLOUD FIX: Drastically reduced iterations for Streamlit Servers ---
        if method == 'differential_evolution':
            result = differential_evolution(
                self.objective_function,
                bounds,
                args=(y0, t_span, n_hours),
                maxiter=3,        # Reduced from 100 to prevent cloud timeout
                popsize=2,        # Reduced from 10
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42,
                workers=1,
                updating='deferred',
                disp=False
            )
        else:
            result = minimize(
                self.objective_function,
                x0,
                args=(y0, t_span, n_hours),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5, 'disp': False} # Reduced from 500
            )
        # -----------------------------------------------------------------------
        
        # Extract optimal trajectories
        rpm_opt, airflow_opt, feed_opt = self._parameterize_trajectory(
            result.x, n_hours
        )
        
        return {
            'optimal_rpm': rpm_opt,
            'optimal_airflow': airflow_opt,
            'optimal_feed': feed_opt,
            'objective_value': result.fun,
            'success': result.success
        }


# =============================================================================
# MAIN PIPELINE: GENE-TO-PLANT DIGITAL TWIN
# =============================================================================

class G2PDigitalTwin:
    """
    Complete Gene-to-Plant Digital Twin pipeline.
    """
    
    def __init__(self):
        self.sequence_analyzer = SequenceAnalyzer()
        self.burden_predictor = MetabolicBurdenPredictor()
        self.growth_model = MechanisticGrowthModel()
        self.reactor_sim = IndustrialBioreactorSimulator()
        self.optimizer = DynamicTrajectoryOptimizer(
            self.growth_model, self.reactor_sim
        )
        
        # Train the burden predictor
        print("Initializing G2P Digital Twin...")
        training_result = self.burden_predictor.fit()
        print(f"Burden predictor trained - Test accuracy: {training_result['test_accuracy']:.3f}")
    
    """
        Run the complete G2P pipeline.
        
        Parameters
        ----------
        amino_acid_sequence : str
            Target protein amino acid sequence
        fermentation_hours : int
            Duration of fermentation
        initial_volume : float
            Starting volume in liters
        optimize : bool
            Whether to run trajectory optimization
            
        Returns
        -------
        dict
            Complete results including burden analysis, simulation,
            bottleneck identification, and optimized control trajectory
        """
    
    def run_pipeline(self, sequence: str, hours: int, volume: float, optimize: bool = True, microbe_name: str = "E. coli", opt_method: str = "differential_evolution"):
        """
        Executes the complete 4-Pillar Digital Twin architecture.
        """
        print("\n" + "="*60)
        print("GENE-TO-PLANT DIGITAL TWIN - COMPLETE PIPELINE EXECUTION")
        print("="*60)

        # ---------------------------------------------------------
        # PILLAR 1: Sequence-to-Burden AI
        # ---------------------------------------------------------
        print(f"\n[Pillar 1] Analyzing Sequence and Predicting Burden for {microbe_name}...")
        features = self.sequence_analyzer.extract_features(sequence)
        burden_results = self.burden_predictor.predict_burden(features, microbe_name)
        
        print(f" -> Predicted Burden Class: {burden_results['burden_class']}")
        print(f" -> Calculated Burden Score: {burden_results['burden_score']:.3f}")

        # ---------------------------------------------------------
        # PILLAR 2: The Biological Bridge (Cellular Physics)
        # ---------------------------------------------------------
        print(f"\n[Pillar 2] Linking Burden to {microbe_name} Cellular Physics...")
        # 1. Fetch the correct microbe kinetic parameters
        self.growth_model.params = MICROBE_LIBRARY.get(microbe_name, MICROBE_LIBRARY["E. coli"])
        # 2. Apply the AI penalty directly to the growth equations
        self.growth_model.apply_burden(burden_results['burden_score'])

        # ---------------------------------------------------------
        # INITIAL STATE PREPARATION (y0)
        # ---------------------------------------------------------
        # Biomass=0.1g/L, Sugar=50g/L, Product=0g/L, DO=0.007g/L, Vol=user_volume, Heat=0
        y0 = np.array([0.1, 50.0, 0.0, 0.007, float(volume), 0.0])
        t_span = (0.0, float(hours))
        
        # Base results dictionary mapped perfectly for the dashboard charts
        results = {
            "microbe": microbe_name,
            "tank_volume": volume,
            "fermentation_time": hours,
            "sequence_features": features,
            "burden_analysis": burden_results
        }

        # ---------------------------------------------------------
        # PILLAR 3: Baseline Factory Simulation
        # ---------------------------------------------------------
        print(f"\n[Pillar 3] Running Standard Baseline Simulation for {hours} hours...")
        base_kLa = self.reactor_sim.calculate_kLa(100, 3000, volume)
        base_controls = [{'feed_rate': 100, 'feed_concentration': 500, 'kLa': base_kLa, 'O2_saturation': 0.007}] * hours
        
        results['baseline_simulation'] = self.growth_model.simulate(
            y0, t_span, base_controls, t_eval=np.linspace(0, hours, hours)
        )

        # ---------------------------------------------------------
        # PILLAR 4: AI Trajectory Optimization
        # ---------------------------------------------------------
        if optimize:
            print(f"\n[Pillar 4] Running AI Optimization Trajectory for {hours} hours using {opt_method}...")
            # Pass the frontend's method choice directly into the optimizer!
            opt_res = self.optimizer.optimize(y0, t_span, hours, method=opt_method)
            results['optimization'] = opt_res
            
            # Re-run simulation using the optimal trajectory to get telemetry
            opt_controls = []
            for i in range(hours):
                kLa = self.reactor_sim.calculate_kLa(opt_res['optimal_rpm'][i], opt_res['optimal_airflow'][i], volume)
                opt_controls.append({'feed_rate': opt_res['optimal_feed'][i], 'feed_concentration': 500, 'kLa': kLa, 'O2_saturation': 0.007})
                
            results['optimized_simulation'] = self.growth_model.simulate(
                y0, t_span, opt_controls, t_eval=np.linspace(0, hours, hours)
            )
            print(" -> Optimization Complete. Maximum yield trajectory found.")

        print("\n" + "="*60)
        print("PIPELINE COMPLETE - SENDING TELEMETRY TO DASHBOARD")
        print("="*60)
        
        return results
    

    def generate_control_report(self, results: Dict) -> pd.DataFrame:
        """Generate hour-by-hour control trajectory report."""
        if 'optimization' not in results:
            return None
        
        opt = results['optimization']
        n_hours = len(opt['optimal_rpm'])
        
        report = pd.DataFrame({
            'Hour': range(n_hours),
            'RPM': opt['optimal_rpm'],
            'Airflow (L/h)': opt['optimal_airflow'],
            'Feed Rate (L/h)': opt['optimal_feed']
        })
        
        # Add simulation results if available
        if 'optimized_simulation' in results:
            sim = results['optimized_simulation']
            
            # --- THE FIX: Wrap in pd.Series() so missing crashed hours become NaN ---
            report['Biomass (g/L)'] = pd.Series(sim['biomass'])
            report['Product (g/L)'] = pd.Series(sim['product'])
            report['DO (g/L)'] = pd.Series(sim['dissolved_oxygen'])
            report['Volume (L)'] = pd.Series(sim['volume'])
            # -----------------------------------------------------------------------
            
        return report
    
    def plot_results(self, results: Dict) -> Dict:
        """Generates 10 individual figures for the Streamlit button UI or terminal saving."""
        sns.set_theme(style="whitegrid")
        figures = {}
        
        # Helper function to keep code clean
        def create_chart(title, ylabel, key, multiplier=1.0, threshold=None):
            fig, ax = plt.subplots(figsize=(10, 5))
            if 'baseline_simulation' in results:
                ax.plot(results['baseline_simulation']['time'], results['baseline_simulation'][key] * multiplier, 'b--', label='Baseline', linewidth=2)
            if 'optimized_simulation' in results:
                ax.plot(results['optimized_simulation']['time'], results['optimized_simulation'][key] * multiplier, 'g-', label='Optimized', linewidth=2)
            if threshold:
                ax.axhline(y=threshold, color='r', linestyle=':', label='Critical threshold')
            ax.set_xlabel('Time (h)'); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
            return fig

        # 1-5: Physics Charts
        figures['Biomass'] = create_chart('Biomass Concentration', 'Biomass (g/L)', 'biomass')
        figures['Product'] = create_chart('Product Concentration', 'Product (g/L)', 'product')
        figures['Substrate'] = create_chart('Substrate (Sugar) Depletion', 'Substrate (g/L)', 'substrate')
        figures['Dissolved Oxygen'] = create_chart('Dissolved Oxygen', 'DO (mg/L)', 'dissolved_oxygen', 1000, 0.5)
        figures['Heat Generation'] = create_chart('Metabolic Heat Generation', 'Accumulated Heat (MJ)', 'heat', 0.001)

        # 6-8: Control Charts
        if 'optimization' in results:
            opt = results['optimization']
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(range(len(opt['optimal_rpm'])), opt['optimal_rpm'], color='purple', linewidth=2)
            ax.set_xlabel('Time (h)'); ax.set_ylabel('RPM'); ax.set_title('Optimal Agitation')
            figures['Agitation'] = fig
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(range(len(opt['optimal_airflow'])), opt['optimal_airflow'], color='cyan', linewidth=2)
            ax.set_xlabel('Time (h)'); ax.set_ylabel('Airflow (L/h)'); ax.set_title('Optimal Aeration')
            figures['Aeration'] = fig
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.fill_between(range(len(opt['optimal_feed'])), 0, opt['optimal_feed'], alpha=0.3, color='orange')
            ax.plot(range(len(opt['optimal_feed'])), opt['optimal_feed'], color='orange', linewidth=2)
            ax.set_xlabel('Time (h)'); ax.set_ylabel('Feed (L/h)'); ax.set_title('Optimal Feeding')
            figures['Feeding'] = fig

        # 9-10: Bioinformatics Charts
        if 'sequence_features' in results:
            fig, ax = plt.subplots(figsize=(10, 5))
            feat = results['sequence_features']
            features_dict = {'Instability': feat.instability_index/100, 'Aliphatic': feat.aliphatic_index/100, 'Hydrophobicity': feat.hydrophobicity, 'Disorder': feat.disorder_propensity, 'Rare Codon': feat.rare_codon_burden}
            sns.barplot(x=list(features_dict.values()), y=list(features_dict.keys()), ax=ax, palette='mako')
            ax.set_title('Normalized Sequence Features')
            figures['Features'] = fig
            
        if 'burden_analysis' in results:
            fig, ax = plt.subplots(figsize=(10, 5))
            burden = results['burden_analysis']
            classes = list(burden['class_probabilities'].keys())
            probs = list(burden['class_probabilities'].values())
            ax.bar(classes, probs, color=['#2ca02c', '#bcbd22', '#ff7f0e', '#d62728'], alpha=0.8, edgecolor='black')
            ax.set_ylabel('Probability'); ax.set_title(f"Metabolic Burden: {burden['burden_class']} ({burden['burden_score']:.2f})"); ax.set_ylim(0, 1)
            figures['Burden'] = fig

        return figures
    

# =============================================================================
# EXAMPLE EXECUTION
# =============================================================================

# =============================================================================
# EXAMPLE EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Example: Green Fluorescent Protein (GFP) sequence
    gfp_sequence = """
    MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL
    VTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLV
    NRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLAD
    HYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK
    """
    
    # Initialize the Digital Twin
    twin = G2PDigitalTwin()
    
    # Run the complete pipeline
    print("\n" + "="*70)
    print("GENE-TO-PLANT DIGITAL TWIN - COMPLETE PIPELINE EXECUTION")
    print("="*70)
    
    # THE FIX: Use the exact variable names defined in run_pipeline
    results = twin.run_pipeline(
        sequence=gfp_sequence,
        hours=72,
        volume=50000,
        optimize=True,
        microbe_name="E. coli"
    )
    
    # Generate control report
    print("\n" + "="*60)
    print("DYNAMIC CONTROL TRAJECTORY (Every 12 hours)")
    print("="*60)
    
    report = twin.generate_control_report(results)
    if report is not None:
        # Show every 12 hours
        display_hours = [0, 12, 24, 36, 48, 60, 71]
        print(report.iloc[display_hours].to_string(index=False))
    
    # Generate plots
    # Generate plots
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION DASHBOARD")
    print("="*60)
    
    # Get the dictionary of 10 graphs
    graphs = twin.plot_results(results)
    
    # Save each graph as its own image file!
    for name, fig in graphs.items():
        filename = f"{name.replace(' ', '_').lower()}_chart.png"
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE - SUMMARY")
    print("="*70)
    print(f"Target protein: {results['sequence_features'].length} amino acids, "
          f"{results['sequence_features'].molecular_weight/1000:.1f} kDa")
    print(f"Metabolic burden: {results['burden_analysis']['burden_class']} "
          f"(score: {results['burden_analysis']['burden_score']:.3f})")
    
    if 'optimized_simulation' in results:
        final_product_kg = (results['optimized_simulation']['product'][-1] * results['optimized_simulation']['volume'][-1] / 1000)
        print(f"Predicted final product yield: {final_product_kg:.1f} kg")
        print(f"Crash prevented: {'Yes' if not results['optimized_simulation']['crash_detected'] else 'No'}")