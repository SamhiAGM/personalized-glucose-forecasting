# personalized-glucose-forecasting
Explainable classical machine learning for 30-minute glucose forecasting and metabolic response clustering using the CGMacros dataset from PhysioNet.
## Data Preprocessing and Feature Engineering

The preprocessing and feature-engineering pipeline prepares the ShanghaiT2DM and CGMacros datasets for 30-minute blood glucose forecasting.

### ShanghaiT2DM

- 100 participants
- 111,753 final ML-ready observations
- 25 approved model features
- Prediction target: glucose at t + 30 minutes
- Duplicate participant/timestamps: 0
- Missing prediction targets: 0
- Incorrect target horizons: 0
- Final audit status: PASS

### CGMacros

- 45 participants
- 678,070 final ML-ready observations
- 39 approved model features
- Prediction target: glucose at t + 30 minutes
- Duplicate participant/timestamps: 0
- Missing prediction targets: 0
- Incorrect target horizons: 0
- Final audit status: PASS

### Feature Groups

The preprocessing pipeline creates features from:

- Recent glucose history
- Time and temporal information
- Meal and nutrition information
- Participant and clinical characteristics
- Activity and heart-rate information where supported

### Final Generated Datasets

The pipeline generates:

- `data/processed/shanghai_ml_ready.csv`
- `data/processed/cgmacros_ml_ready.csv`

Raw and generated datasets are excluded from Git and must be stored or generated locally.

### Important Modeling Rules

- Split train, validation, and test data by participant ID rather than randomly splitting rows.
- Fit missing-value imputation using training participants only.
- Fit scaling using training participants only.
- `participant_id`, `timestamp`, and `target_timestamp_30min` are metadata and should not be model inputs.
- `target_glucose_30min` is the regression target and must never be included in the input feature matrix.
- ShanghaiT2DM and CGMacros should not be directly concatenated until their feature definitions and units are harmonized.