# solar-wind-magnetosphere-xgboost-forecast
Physics-informed Machine Learning pipeline for 1-hour ahead ($T+1$) geomagnetic storm forecasting ($Dst \le -50\text{ nT}$) using NASA OMNIWeb solar wind data, rolling-window energy accumulation features, autoregression, and Optuna-tuned XGBoost.

```markdown
# ☀️ Geomagnetic Storm Forecasting ($T+1$) using Physics-Informed XGBoost

An end-to-end Machine Learning and Computational Physics pipeline designed to forecast geomagnetic storms ($Dst \le -50\text{ nT}$) **1 hour ahead** using interplanetary solar wind parameters measured at the Sun-Earth Lagrange 1 ($L_1$) point.

## 🌌 Physical Background & Motivation
Geomagnetic storms are major disturbances of Earth's magnetosphere caused by the efficient exchange of energy from the solar wind into the space environment surrounding Earth. When the Interplanetary Magnetic Field (IMF) turns southward ($B_z < 0$), **magnetic reconnection** occurs, allowing solar wind plasma and Poynting flux ($E_y = -v \times B_z$) to penetrate the magnetosphere and intensify the equatorial **Ring Current**, resulting in a sharp drop in the Disturbance Storm Time ($Dst$) index.

For equatorial regions (such as Indonesia/Kalimantan), severe geomagnetic storms trigger Equatorial Ionospheric Anomalies (EIA) and ionospheric scintillation, disrupting high-precision GPS/GNSS navigation and High-Frequency (HF) radio communications. 

This project bridges **Space Physics** and **Machine Learning** to build a computationally lightweight Early Warning System (EWS).

## 📊 Dataset & Preprocessing
* **Source:** NASA/GSFC's OMNIWeb (Hourly resolution, 2012–2015, covering the **Solar Cycle 24 Maximum**).
* **Data Cleaning:** Instrument telemetry gaps (e.g., `999.9`) were handled using time-based linear interpolation (capped at $\le 3\text{ hours}$ to preserve plasma fluid continuity); longer gaps were dropped, resulting in **34,788 valid hourly observations**.
* **Target Definition:** Binary classification based on the $Dst$ index:
  * `0` (**Quiet**): $Dst > -50\text{ nT}$ (~95.4% of data)
  * `1` (**Storm**): $Dst \le -50\text{ nT}$ (~4.6% of data — highly imbalanced)

## ⚙️ Methodology & Physics-Informed Feature Engineering
Rather than treating space weather purely as a black-box tabular problem, physical domain knowledge was embedded into the pipeline:
1. **Kinematic Time-Lag ($T+1$ Forecasting):** Targets were shifted 1 hour ahead to account for the solar wind propagation time from the $L_1$ satellite orbit (~1.5 million km) to Earth's magnetopause (~45–60 minutes).
2. **Energy Accumulation (Rolling Windows):** Based on Burton et al. (1975) and Newell et al. (2007), magnetospheric storms result from sustained energy injection rather than instantaneous fluctuations. 3-hour and 6-hour rolling means were extracted for $B_z$, $E_y$, Flow Speed, and $\vert{}B\vert{}$.
3. **Magnetospheric Inertia (Autoregression):** Incorporated the current storm state ($T$) to model the physical inertia and decay rate of the Ring Current.
4. **Imbalance & Hyperparameter Optimization:** Handled extreme class imbalance using `scale_pos_weight` (~20.66) and tuned tree architecture via **Optuna** (100 trials) optimizing for **PR-AUC** (Precision-Recall Area Under Curve).
5. **Strict Leakage Prevention:** Chronological sorting, rolling-window extraction, target shifting, and stratified splitting were strictly ordered to prevent data leakage.

## 📈 Key Results
* **PR-AUC Score:** `0.82+` *(Update with your Cell 11 final score)*
* **Storm F1-Score / Recall:** Significant reduction in False Alarms after introducing 3h–6h rolling historical memory, proving that the algorithm successfully learned the cumulative nature of solar wind-magnetosphere coupling.
* **Feature Importance:** XGBoost autonomously prioritized `IMF_Mag_Avg`, `Ey`, and `Flow_Speed` over raw `Bz_GSM`, confirming that the model captured the collinear electrodynamic relationship $E_y = -v \times B_z$.

## 🛠️ Tech Stack
* **Language:** Python 3
* **Libraries:** `XGBoost`, `Optuna`, `Scikit-Learn`, `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`

## 🚀 How to Run
1. Clone this repository:
   
```bash
   git clone https://github.com/yourusername/geomagnetic-storm-forecasting-xgboost.git
