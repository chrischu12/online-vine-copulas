set -e
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^online_copula_exp "; then
    conda create -y -n online_copula_exp python=3.12
fi
conda activate online_copula_exp

git submodule update --init --recursive VineCopulas
git submodule update --init --recursive ondil

pip install -r req.txt

cd VineCopulas
git fetch
git checkout dev || git checkout -b dev origin/dev
git pull
pip install -e .

cd ..
cd ondil
git fetch
git checkout bivariate_copula || git checkout -b bivariate_copula origin/bivariate_copula
git pull
pip install -e .
