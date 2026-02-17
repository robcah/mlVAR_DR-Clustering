# Dr+Clustering from mlVAR features
## Code for replication
* Data will be accesible by request at discretion of the University of Manchester

Supporting materials for paper: _Emergence of clusters of subjects with psychosis related symptoms from mlVAR feature space using unsupervised machine learning methods_.

1. Install Anaconda, R and RStudio (recommended)

1. in RStudio (or R) run the file:
`mlVAR_FeatureExtraction.qmd`
This produces the file `mlvar_participantsstatistics.csv`

1. From an Anaconda prompt install environment `mlvar_DRC`:\
`conda env create --file mlvar_DRC.yaml`\
It is possible you will need to install `Microsoft Visual C++ v14.0` or greater.

1. Activate environment and run Jupyter:\
`conda activate mlVAR_DRC`\
`jupyter notebook`

1. Run and follow the instructions from jupyter notebook `mlVAR_DRClustering.ipynb` to produce the analysis on the exploration of Dr and clustering tools to find the most robust combination and label participants by emerging clusters. This will produce the file `BRC_data+clusters.csv`.

1. in RStudio (or R) run the file:\
`mlVAR_FeatureExtraction.qmd`
This will produce the files: `mlvar_cluster#_temporal.csv`, `mlvar_cluster#_contemporaneous.csv`, `mlvar_cluster#_between.csv`\
Where # is a number from 0 to 3.

1. in Jupyter notebook run the file `mlVAR_ManuscriptPlots.ipynb`to reproduce the figures in the manuscript.



