import magic

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

X= magic.io.load_csv('GSE74596.csv', cell_axis='column')
X=X.values
print(X.shape)

#libsize = X.sum(axis=1)
#plt.hist(libsize, bins=50)
#plt.axvline(1000, c='r')
#plt.show()
print("X:",X)
'''
print(type(X))
X  = magic.preprocessing.library_size_normalize(X)
print("X:",X)
X = np.sqrt(X)
print("X:",X)
'''
X = X+1
X = np.log2(X)
print("X:",X)
magic_op = magic.MAGIC(k=5)


X_magic  = magic_op.fit_transform(X)
print(X_magic.shape)
print("X:",X_magic)
#print(type(X_magic))
'''
print(X_magic.values.T)

genes = list(X_magic.T.index)
low_genes = []
for gene in genes:
    low_genes.append(gene.lower())
print(low_genes[0:5])
'''
'''
bmmsc_data = magic.io.load_csv('https://github.com/KrishnaswamyLab/PHATE/raw/master/data/BMMC_myeloid.csv.gz')
bmmsc_data.head()
'''
'''
import magic

# Plotting and miscellaneous imports
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#%matplotlib inline

# Load single-cell RNA-seq data
scdata = magic.mg.SCData.from_csv('GSE74596.csv',
                                  data_type='sc-seq', cell_axis=1, normalize=False)
scdata = scdata.normalize_scseq_data() 
scdata.log_transform_scseq_data(pseudocount=1)

print(scdata)

#fig, ax = scdata.plot_molecules_per_cell_and_gene()
#fig.show()


# Minimum molecules/cell value
CELL_MIN = 0

# Maximum molecules/cell values
CELL_MAX = 1000000

# Minimum number of nonzero cells/gene 
# (None if no filtering desired)
GENE_NONZERO = None

# Minimum number of molecules/gene
# (None if no filtering desired)
GENE_MOLECULES = None

#scdata.filter_scseq_data(filter_cell_min=CELL_MIN, filter_cell_max=CELL_MAX, 
#                         filter_gene_nonzero=GENE_NONZERO, filter_gene_mols=GENE_MOLECULES)

print(scdata)
'''
#scdata = scdata.normalize_scseq_data()
#scdata.save('scdata.p')
#scdata = magic.mg.SCdata.load('scdata.p')
'''
#fig, ax = scdata.plot_pca_variance_explained(n_components=150, random=True)
#print(type(fig))
#fig.show()


scdata.run_magic(n_pca_components=20, random_pca=True, t=None, compute_t_make_plots=True, 
                 t_max=12, compute_t_n_genes=500, k=30, 
                 ka=10, epsilon=1, rescale_percent=99)
''
fig, ax = scdata.magic.scatter_gene_expression(['MAGIC 0610007c21rik', 'MAGIC 0610007l01rik'], color ='MAGIC 0610007p08rik')
fig.show()

fig, ax = scdata.magic.scatter_gene_expression(['MAGIC 0610007p14rik', 'MAGIC 0610007p22rik'], color ='MAGIC 0610009b14rik')
fig.show()
''

data = scdata.data.values.T
#data.to_csv("test.csv")
print(type(data))
print(data.shape)

''
CSVFileName = "GSE74596_magic_test.csv"
scdata.to_csv(CSVFileName)
'''
