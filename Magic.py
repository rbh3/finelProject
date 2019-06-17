import magic

import numpy as np

def txt_to_csv(filename,start_row):
    txt = open(filename,'r')
    csv = open(filename[:-4]+'.csv','w',encoding='utf-8')
    count = 0
    for line in txt:
        count+=1
        if count>=start_row:
            csv.write(line.replace("\t",","))
    csv.close()



def magic_process(filename,start_row):
    
    txt_to_csv(filename,start_row)
    filename = filename[:-4]+".csv"

    X = magic.io.load_csv(filename, cell_axis='column', encoding='utf-8')
    if len(X) == 0:
        raise SyntaxError('File not on correct format')

    try:
        X = np.log2(X+1)
        magic_op = magic.MAGIC(k=2,t=3, a=20, verbose=0)
        X_magic  = magic_op.fit_transform(X)
    except:
        raise EOFError('File not on correct format')

    genes = list(X_magic.T.index)
    low_genes = []
    try:
        for gene in genes:
             low_genes.append(gene.lower())
    except:
        raise IndexError('A problem in translating your genes, maybe you need conversion file')


    return X_magic.values.T, low_genes
