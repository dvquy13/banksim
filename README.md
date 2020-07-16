# BankSim Data Exploration
Source: https://www.kaggle.com/ntnu-testimon/banksim1/data#

# Set up
1. Grant permission to scripts: `chmod 744 -R ./cmd`
2. Set up env vars: `source .env`
3. Making directories: `./cmd/01_setup.sh`

# Naming
## Data
|name|base|meaning|
|---|---|---|
|`fe1`|None|basic numerical features' stats aggregated by customer|
|`fe2`|`fe1`|add basic categorical features' stats|
|`fe21`|`fe2`|cross-validation with full data instead of train data|
|`fe3`|None|generate features from featuretools, not using gender info. Depth = 2|
|`fe32`|`fe3`|using gender proportion by merchant info, depth = 3, adding interesting values when generating features|
|`fe321`|`fe32`|remove features building from `customer` value|
