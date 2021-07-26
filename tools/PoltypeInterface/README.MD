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
conda install openbabel --yes
conda install -c conda-forge rdkit --yes
```

* If you want to use Tinker9's gpu dynamics, make sure that your bashrc for each Tinker9 env contains the below export statement as well as function for dynamics_gpu (to call GPU dynamics and not CPU dynamics) 
```
export GPUDYNAMICS=True
```

* Make sure Tinker is sourced in your path so program can use xyzedit!
* Environment Bashrc Example For CPU Tinker
```
conda activate FBTest
export PATH=/home/bdw2292/NewestTinker/bin:$PATH
```
* Make sure to copy files from modified forcebalance to conda site packages (for example mine are located /home/bdw2292/miniconda3/envs/FBTest/lib/python3.8/site-packages/forcebalance/) 
* Make a folder for input/outputs from ForceBalence and navigate to that folder
* Make a file called forcebalancepoltype.ini , inputs will go in here
* Call script via 
```
python /home/bdw2292/FBInterface/forcebalancepoltypeinterface.py
```
* Example input
```
poltypepathlist=/home/bdw2292/PoltypeJobs/HalogenPaperMolecules/CHI3,/home/bdw2292/PoltypeJobs/HalogenPaperMolecules/CF4 
vdwtypeslist=401 402 , 401 402  
temperature_list= 300 250 , 300 230 350 
pressure_list= 1 1 , 1 1 1 
density_list=1000 1000 , 1000 1000 1000 
fittypestogether=401 402 , 401 402
```
* In the example above, fittypestogether reads fit 401 from molecule 1 and 401 from molecule 2 together, and also fit 402 from molecule 1 and 402 from molecule 2 together
* Alternative example input
```
poltypepathlist=/home/bdw2292/PoltypeJobs/HalogenPaperMolecules/CHI3,/home/bdw2292/PoltypeJobs/HalogenPaperMolecules/CF4 
csvexpdatafile=ExpHalogenData.csv
vdwtypeslist=401 402 , 401 402  
```
* Can use UNK for when unknown data property for input arrays
* If using csvexpdatafile, please use correct units as below and the column headers in example csv file, make sure the molecule name under the Name column are same as poltype molecule folder name
* By default all types in key files are reassigned new type numbers so there is no overlap while fitting together. So input types will be converted internally.
* temperature_list, pressure_list and density_list are required.
* Density units are Kg/m^3
* Enthalpy of vaporization units kJ/mol
* Thermal expansion units 1/K
* Isothermal compressibility units are 1/bar
* Isobaric heat capacity units are kJ/mol.K
* pressure units are atm
* Experimental data can be found at https://webbook.nist.gov/ or https://www.aiche.org/dippr
* QM energies from dimers in poltype (normalized) >=15 kcal/mol are removed and not use for forcebalance.
```
poltypepathlist                                      Comma seperated list of Paths to completed poltypejobs
vdwtypes                                             Comma seperated list of vdw types lists to parameterize
temperature_list                                     Comma seperated list of temperatures 
pressure_list                                        Comma seperated list of pressures 
enthalpy_of_vaporization_list                        Comma seperated list of heat of vaporizations 
enthalpy_of_vaporization_err_list                    Comma seperated list of heat of vaporizations error 
surface_tension_list                                 Comma seperated list of surface tensions 
surface_tension_err_list                             Comma seperated list of surface tension errors 
relative_permittivity_list                           Comma seperated list of relative permittivity
relative_permittivity_err_list                       Comma seperated list of relative permittivity errors
isothermal_compressibility_list                      Comma seperated list of isothermal_compressibility
isothermal_compressibility_err_list                  Comma seperated list of isothermal_compressibility errors
isobaric_coefficient_of_volume_expansion_list        Comma seperated list of isobaric_coefficient_of_volume_expansion
isobaric_coefficient_of_volume_expansion_err_list    Comma seperated list of isobaric_coefficient_of_volume_expansion errors
heat_capacity_at_constant_pressure_list              Comma seperated list of heat_capacity
heat_capacity_at_constant_pressure_err_list          Comma seperated list of heat_capacity errors
density_list                                         Comma seperated list of density
density_err_list                                     Comma seperated list of density errors
citation_list                                        Comma seperated list of citations
liquid_equ_steps                                     Equilibrium steps MD 
liquid_prod_steps                                    Production dynamic steps MD
liquid_timestep                                      Time step for liquid MD
liquid_interval                                      
gas_equ_steps                                        Equilibrium steps MD 
gas_prod_steps                                       Production dynamic steps MD
gas_timestep      
gas_interval
md_threads                                           CPU cores
liquid_prod_time                                     Total time liquid sim
gas_prod_time                                        Total time gas sim
nvtprops                                             This controls whether to include surface tension or not (NVT) property
```
