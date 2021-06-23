# Readme for ForceBalance Poltype Wrapper

* Install modified forcebalance for tinker
```
git clone https://github.com/bdw2292/forcebalance.git
```

* ForceBalance conda environment
```
conda create --name FBTest python=3.8 --yes
conda activate FBTest
conda install -c conda-forge ndcctools --yes
conda install -c conda-forge forcebalance --yes
conda install pip --yes
pip install pyilt2
conda install -c conda-forge cirpy --yes
conda install -c conda-forge pint --yes
conda install -c conda-forge tqdm --yes
conda install -c anaconda beautifulsoup4 --yes
conda install -c anaconda pandas --yes
conda install openbabel --yes
conda install -c conda-forge rdkit --yes
```

* If you want to use Tinker9's gpu dynamics, make sure that your bashrc for each Tinker9 env contains the below export statement as well as function for dynamics_gpu (to call GPU dynamics and not CPU dynamics) 
```
export GPUDYNAMICS=True

```

* https://ilthermo.boulder.nist.gov/
* Be careful by default tries first name it finds with molecule match, but may not be correct name. Use printnames to find possible names that are found in database. Use truename to only grab data from the correct molecule name (according to the database)

* Make sure Tinker is sourced in your path so program can use xyzedit!
* Environment Bashrc Example For CPU Tinker
```
conda activate FBTest
export PATH=/home/bdw2292/NewestTinker/bin:$PATH
```
* Make sure to copy files from modified forcebalance to conda site packages (for example mine are located /home/bdw2292/miniconda3/envs/FBTest/lib/python3.8/site-packages/forcebalance/) 
* If you just want to use the NIST parser
```
python forcebalancepoltypeinterface.py --smiles=C(Cl)(Cl)(Cl)Cl
python forcebalancepoltypeinterface.py --smiles=C(Cl)(Cl)(Cl)Cl --printnames
python forcebalancepoltypeinterface.py --smiles=C(Cl)(Cl)(Cl)Cl --truename=Tetrachloromethane
python forcebalancepoltypeinterface.py --truename=Water
```
* If you want to use NIST parser and grab data/files from poltype job do set up Force Balance job
* Make a folder for input/outputs from ForceBalence and navigate to that folder
```
python /home/bdw2292/FBInterface/forcebalancepoltypeinterface.py --poltypepath=/home/bdw2292/TestDaemonSubmissionScript/DrugTestMolecules/944 --vdwtypes=401,402 
```
```
poltypepath        Path to completed poltypejob
vdwtypes           List of vdw types to parameterize
liquid_equ_steps   Equilibrium steps MD 
liquid_prod_steps  Production dynamic steps MD
liquid_timestep    Time step for liquid MD
liquid_interval    
gas_equ_steps      Equilibrium steps MD 
gas_prod_steps     Production dynamic steps MD
gas_timestep      
gas_interval
md_threads         CPU cores
liquid_prod_time   Total time liquid sim
gas_prod_time      Total time gas sim
nvtprops           This controls whether to include surface tension or not (NVT) property
skipNIST           This will skip parsing NIST and generation of data.csv
density            This will input density for liquid.xyz



```
