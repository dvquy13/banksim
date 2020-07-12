source .env

mkdir -p data/raw
mkdir -p data/raw/.gitkeep
mkdir -p data/interim
mkdir -p data/interim/.gitkeep
mkdir -p data/process
mkdir -p data/process/.gitkeep
mkdir -p data/output
mkdir -p data/output/.gitkeep

mkdir -p notebooks/wip

# TODO: download data from Kaggle: https://www.kaggle.com/ntnu-testimon/banksim1/data#

# Get envsubst
sudo apt-get install gettext-base