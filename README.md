****Revealing the Impact of Aggregations in the Graph-based Molecular Machine Learning: Electrostatic Interaction versus Pooling Methods****

Requirements

For users with CUDA version > 11.7

eelGNN requires new environment with python=3.11 from anaconda. We used conda 23.1.0
conda create -n eel_gnn python=3.11
After creating new environment, the following commands are required to install packages.

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
conda install pandas
pip install -U scikit-learn
pip install rdkit
conda install openpyxl


If you want to use eelGNN_espaloma, you have to install additional packages via GitHub - choderalab/espaloma_charge: Standalone charge assignment from Espaloma framework. or taking following commands

conda install -c dglteam/label/th20_cu117 dgl
pip install espaloma_charge
pip install packaging

% For users with CUDA --version <= 11.7 or whom above command doesn’t work

You can install older packages using below commands, but in this case eelGNN_espaloma is unavailable since it supports python >= 3.11
conda create -n alt_env python=3.10
conda install pytorch==1.13.1 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
conda install pandas
pip install -U scikit-learn
pip install rdkit
conda install openpyxl



Implementation
(Optional) 1. Calculate partial charge of molecules and save charge data as a pkl file. This procedure is not required for eelGNN_Pauling. In addition, the charge of chromophore data is prepared in both espaloma and gasteiger charge.

2. Determine the set of number as following
First one: Which property do you want to predict?
①	maximum absorption wavelength [nm]
②	maximum emission wavelength [nm]
③	fluorescence lifetime (log10 value)
④	photoluminescence quantum yield (log10 value)
⑤	extinction coefficient (log10 value)
⑥	absorption bandwidth [cm-1]
⑦	emission bandwidth [cm-1]
⑧	solubility (ESOL)
⑨	molar mass (ESOL) 
Second one: Which model do you want to use?
①	eelGCN_Espaloma
②	eelGCN_Gasteiger
③	eelGCN_Pauling
④	GCN
Third one: How many edge types do you want for intermolecular edges?
①	1
②	2
③	4
Fourth one: Which aggregation do you want to use?
①	add
②	mean
Fifth one: Which pooling do you want to use?
①	add
②	mean

3. execute the main.py with your settings.
For example, you can input
python main.py 5 2 1 2 1
to predict extinction coefficient with eelGCN_Gasteiger, using single intermolecular edge types and using mean aggregation and add pooling.
