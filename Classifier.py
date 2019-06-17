import pickle

import math
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


def get_series_data(filename,data_line,data_end, isLabeld, isTitled, offset=1):
    """
    This function opens a .txt file, reads until the type_line, uses those entries as labels (array y).
    Then it reads until data_line and creates an array of expression data (specifically log2(expression+1))
    Offset is only used for weird GSE files where the labels present parsing difficulties
    """
    data_line = int(data_line)
    data_end = int(data_end)
    bad_columns = [0] #columns that we throw out due to missing label/data

    f = open(filename, 'r')
    labels = []
    titles = []

    #read down until data_line
    for _ in range(data_line):
        line = f.readline()
        if isLabeld == 'true' and _ == 0:
            labels = line.split('\t')
            labels = [element.lower().rstrip() for element in labels] ; labels
        if isTitled == 'true' and _ == 1:
            titles = line.split('\t')
            titles = [element.lower().rstrip() for element in titles] ; titles
            print(titles)

    if len(titles) > 0:
        del titles[0]
    #Set up X - the array of data
    X = []
    gene_ids = []

    count = data_line
    while(data_end != 0 and count < data_end) or (data_end == 0 and ("!series_matrix_table_end" not in line and line != '')):
        count+= 1
        if count%5000 == 0:
            print("reading line: ",count)

        line_split = line.split("\t")
        gene_ids.append(line_split[0].strip('"').lower()) #first element in the row is the gene id
        line_flt = [] #we generate a line of floats - the data across a single row
        for i in range(len(line_split)):
            try:
                line_flt.append((math.log(float(line_split[i])+1)/math.log(2))) #/scale)
            except:
                bad_columns.append(i)
                line_flt.append(line_split[i].strip('"'))
        X.append(line_flt)
        line = f.readline()
    
    #Turn into an np.array 
    X = np.array(X)

    if(len(X) == 0):
        return 'Empty'

    #Remove the columns with missing data/labels
    bad_columns = list(set(bad_columns))
    bad_columns.sort(key = lambda x:-x) #we have to remove backwards
    for i in bad_columns:
        X = np.delete(X,i,1)

    f.close()

    return X, gene_ids, labels, titles




def gene_code_map(filename, data_row, data_end, symbol_col, id_col):
    '''
    This function opens up the txt file with gene ids and symbols to create conversion mappings
    '''
    symbol_col -= 1
    id_col -= 1
    id_to_symbol_map = {}
    symbol_to_id_map = {}
    f = open(filename, 'r')
    for _ in range(data_row):
        line = f.readline()
        #print(line)

    count = 0
    while(line !="!platform_table_end" and count<data_end) or line != "":
        count+= 1
        if count%5000 == 0:
            print("reading line: ",count)

        line_split = line.split(sep="\t")
        try:
             if line_split[id_col].lower().strip('\n') not in id_to_symbol_map:
                id_to_symbol_map[str(line_split[id_col]).lower().strip('\n')] = str(line_split[symbol_col]).lower().strip('\n')
             if line_split[symbol_col].lower().strip('\n') not in symbol_to_id_map:
                symbol_to_id_map[line_split[symbol_col].lower().strip('\n')] = line_split[id_col].lower().strip('\n')
        except:
            raise SyntaxError()
        line = f.readline()

    f.close()
    return id_to_symbol_map

def gene_symbol_to_affy(gene_list):
    '''
    Takes in a list of gene symbols present in the query set and creates a mapping of mutual genes in query set and microarray set
    '''
    symbol_to_affy = pickle.load(open("Testing/symbol_to_AffyID_map", "rb")) #a saved mapping for the microarray set

    id_to_affy = {}
    genes = set(gene_list)
    print(gene_list[0:5])
    for key in symbol_to_affy:
        if key in genes:
            id_to_affy[key] = symbol_to_affy[key]

    return id_to_affy


def gene_to_Affy_map(ID_to_symbol_map,gene_id_genes,X):
    '''
    Takes in gene id to symbol map and uses saved symbol to affy map to create a direct mapping of gene id to affy
    '''
    
    symbol_to_affy = pickle.load(open("Testing/symbol_to_AffyID_map", "rb"))
    translated_genes = 0
    non_translated = 0
    ID_to_Affy = {}
    gene_expression = {} #keeps track of current average expression for gene
    symbols_seen = set()
    for gene in gene_id_genes:

        try:           
            if ID_to_symbol_map[gene] in symbols_seen:
                #If there are multiple instances of a gene, choose the gene ID with the higher reported mean expression according to standard practice

                if np.mean(X[gene_id_genes.index(gene),:]) > gene_expression[ID_to_symbol_map[gene]]:
                    ID_to_Affy[gene]=symbol_to_affy[ID_to_symbol_map[gene]]
                    gene_expression[ID_to_symbol_map[gene]] = np.mean(X[gene_id_genes.index(gene),:])
                    
            else:
                ID_to_Affy[gene]=symbol_to_affy[ID_to_symbol_map[gene]]

                gene_expression[ID_to_symbol_map[gene]] = np.mean(X[gene_id_genes.index(gene),:])
                translated_genes += 1
                symbols_seen.add(ID_to_symbol_map[gene])
        except:
            non_translated += 1
    print("number of translated genes: ",translated_genes," not translated: ",non_translated)
    return ID_to_Affy




def reduce_to_good_rows(train_data,test_data,included_affy_file,train_genes,gene_to_Affy,ID_genes):
    '''
    Before distribution matching, we need to reduce the reference and query set to only mutually shared genes
    '''
    X_train = train_data.astype(np.float)
    X_test = test_data.astype(np.float)
    if(X_test.shape[1] == 0):
        return 'NO TEST GENES'
    print(X_test)
    print(X_test.shape)
    print("train genes: ",len(train_genes))

        
    included_Affy_indices = pickle.load(open(included_affy_file, "rb"))


    ID_genes = np.array(ID_genes)
    Affy_to_gene = {}
    #set up the conversions in both directions
    for entry in list(gene_to_Affy.keys()):
        Affy_to_gene[gene_to_Affy[entry]] = entry
            

    X = [] #newly formated array of query data in Affy order
    X_affy_good_rows = [] #For distances, we only want to use features that were successfully converted
    input_X_good_rows = [] #X, but without the non-convertable rows
    count = 0
    genes_converted = 0

    ID_genes_list = ID_genes.tolist()
    X_test_list = X_test.tolist()

    #for each gene we want to include
    for i in included_Affy_indices: 
        gene = train_genes[i]
        
        if count%300==0:
            print("count: ",count)
        if gene in Affy_to_gene:
            #convert if possible
                
            ID_gene = Affy_to_gene[gene].lower()
            index = ID_genes_list.index(ID_gene) #find the corresponding row in the query set
            
            genes_converted+=1
            if genes_converted==1 or genes_converted==2:
                print("GENE USED IN PLOT:",Affy_to_gene[gene].lower())
            input_X_good_rows.append(X_test_list[index]) 
            X_affy_good_rows.append(X_train[i,:].tolist())           
  
        count+=1
    
                
    print("genes converted: ",genes_converted," out of ",count)
    
    X_affy_good_rows = np.array(X_affy_good_rows) #GSE data for rows that were successfully converted
    input_X_good_rows = np.array(input_X_good_rows) #query data for rows that were successfully converted

    return X_affy_good_rows, input_X_good_rows

# Hack notes: predict_proba works - it gives a vector of all the confidences according to each

def ravidSVM(X_train, y_train, X_test, k, type_map):
    print('Calculating using SVM')
    clf = svm.SVC(kernel="rbf", probability=True)
    clf.fit(X_train.transpose(), y_train)
    pred = clf.predict(X_test.transpose())
    pred_translated=[]
    for item in pred:
        pred_translated.append(type_map[item])
    print(pred_translated)
    conf = clf.predict_proba(X_test.transpose())
    confArr = []
    for ind, confLevel in enumerate(conf):
        confArr.insert(ind, calcConf(confLevel, type_map, clf.classes_))
    print(confArr)
    return pred_translated, confArr


def dorRandomForst(X_train, y_train, X_test, k, type_map):
    print('Calculating using RANDOM FOREST')
    RanFor = RandomForestClassifier(n_estimators=100)
    RanFor.fit(X_train.transpose(), y_train)
    pred = RanFor.predict(X_test.transpose())
    pred_translated = []
    for item in pred:
        pred_translated.append(type_map[item])
    print(pred_translated)
    conf = RanFor.predict_proba(X_test.transpose())
    confArr = []
    for ind, confLevel in enumerate(conf):
        confArr.insert(ind, calcConf(confLevel, type_map, RanFor.classes_))
    print(confArr)
    return pred_translated, confArr

def calcConf(confLine,typeMap, setClasses):
    confArr = {}
    for ind,cls in enumerate(setClasses):
        myType = typeMap[cls]
        if confArr.get(myType) is None:
            confArr[myType] = confLine[ind]
        else:
            confArr[myType] += confLine[ind]
    key_max = max(confArr.keys(), key=(lambda k: confArr[k]))
    return confArr[key_max]

def KNN_sort_filtered(X_train,y_train,X_test,k,type_map):
    print('Calculating using KNN')
    predicted_types = []
    confidences = []

    d,n = X_test.shape
    d,n_train = X_train.shape
    correct=0
    for i in range(n): #for each point in reference set
        frequency = {}
        closest_examples = []
        for j in range(n_train):
            dist = np.linalg.norm(X_train[:,j]-X_test[:,i])
            closest_examples.append((j,dist))
        closest_examples.sort(key= lambda x:x[1])
        #find the types of k closest and return mode
        for j in range(k):
            if type_map[y_train[closest_examples[j][0]]] in frequency:
                frequency[type_map[y_train[closest_examples[j][0]]]] +=1
            else:
                frequency[type_map[y_train[closest_examples[j][0]]]]=1
        key_list = list(frequency.keys())
        key_list.sort(key=lambda x:frequency[x])
        cell_type = key_list[-1]
        predicted_types.append(cell_type)
        confidences.append(frequency[cell_type]/k)

    return predicted_types,confidences#,X_train,X_test

def classification(X_train, y_train, X_test, included_affy_file, train_genes_file, algo, k=10, platform="affy",
                      genes_list=None):
    '''
    This function takes in SCALED variance filtered training and test data, finds the mode of k closest neighbors for each sample of X_test
    train_genes_file - list of genes remaining after variance filtering that should be used
    platform - either "affy" or mapping of query set gene ID to Affy ID
    gene_list - list of query set gene IDs
    impute - should missing genes in query set be filled in with mean of reference set
    '''

    # If platform not affy we need to convert as much as possible
    if platform != "affy":
        Affy_genes = pickle.load(open(train_genes_file, "rb"))

        included_Affy_indices = pickle.load(open(included_affy_file, "rb"))
        print("Top two included genes in good rows: ", Affy_genes[included_Affy_indices[0]],
              Affy_genes[included_Affy_indices[1]])

        if platform == "xloc":
            ID_genes = pickle.load(open("Testing/gene_ids_genes_fpkm", "rb"))
            ID_genes = np.array(ID_genes)
            gene_to_Affy = pickle.load(open("Testing/Xloc_to_Affy", "rb"))
            Affy_to_gene = {}
            # set up the conversions in both directions
            for entry in list(gene_to_Affy.keys()):
                Affy_to_gene[gene_to_Affy[entry]] = entry
        else:
            gene_to_Affy = platform
            ID_genes = genes_list
            ID_genes = np.array(ID_genes)
            Affy_to_gene = {}
            # set up the conversions in both directions
            for entry in list(gene_to_Affy.keys()):
                Affy_to_gene[gene_to_Affy[entry]] = entry

        X_affy_good_rows = []  # For distances, we only want to use features that were successfully converted
        input_X_good_rows = []  # newly formatted query data including only mutual genes in the same order as the reference data
        count = 0
        genes_converted = 0

        ID_genes_list = ID_genes.tolist()
        ID_genes_set = set(ID_genes_list)
        X_test_list = X_test.tolist()
        X_test_mean = np.mean(X_test, axis=0)

        # for each gene we want to include
        for i in included_Affy_indices:
            gene = Affy_genes[i]

            if gene in Affy_to_gene and Affy_to_gene[gene] in ID_genes_set:
                # convert if possible
                ID_gene = Affy_to_gene[gene]
                index = ID_genes_list.index(ID_gene)

                genes_converted += 1
                input_X_good_rows.append(X_test_list[index])
                X_affy_good_rows.append(X_train[i, :].tolist())
            count += 1

        print("genes converted: ", genes_converted, " out of ", count)

        X_test = np.array(X_affy_good_rows)  # reference set rows that were successfully converted
        X_train = np.array(input_X_good_rows)  # query set rows that were successfully converted

    type_map = {'ccd11b': 'other', 'b': 'b', 'cd19': 'b', 'sc': 'other', 'fi': 'other', 'frc': 'other', 'bec': 'other',
                'lec': 'other',
                'ep': 'other', 'st': 'other', 't': 't4', 'nkt': 'nkt', 'prob': 'b', 'preb': 'b', 'pret': 't4',
                'mo': 'other', 'b1b': 'b1ab',
                'b1a': 'b1ab', 'dc': 'dc', 'gn': 'gn', 'nk': 'nk', 'mf': 'mf', 'tgd': 'tgd', 'cd4': 't4',
                'mlp': 'other', 'cd8': 't8',
                't8': 't8', 'b1ab': 'b1ab', 'treg': 'treg', 't4': 't4', 'dn': 'other', 'eo': 'other', 'ilc1': 'other',
                'ilc2': 'other',
                'ilc3': 'other', 'ba': 'other', 'mechi': 'other', 'mc': 'other', 'ccd11b-': 'other', 'b-cells': 'b',
                'nucleated': 'other', 'lt-hsc': 'other',
                'monocytes': 'other', 'cd4+': 't4', 'cd8+': 't8', 'granulocytes': 'gn', 'macrophage': 'mf',
                'hsc': 'other', 'ilc': 'other'}
    if algo == 'KNN':
        return KNN_sort_filtered(X_train,y_train,X_test,k,type_map)
    if algo == 'Random Forest':
        return dorRandomForst(X_train,y_train,X_test,k,type_map)
    if algo == 'SVM':
        return ravidSVM(X_train,y_train,X_test,k,type_map)

def match_dist(X_ref,X_query): #, one-to-one scaling is ok
    '''
    Matches the distribution of the query set to the reference set.
    For each sample, genes are sorted by expression values, then the distribution of the average of the reference set is copied over
    query file and reference file should already contain only mutual, convertable genes
    '''
    
    print("X_ref shape: ",X_ref.shape,"query shape: ", X_query.shape)
    if len(X_ref) == 0 or len(X_query) == 0:
        return 'File not on correct format'
    X_ref_mean = np.median(X_ref,axis=1)
    X_ref_sort = X_ref_mean.tolist()
    X_ref_sort.sort() #Representative distribution of the reference set
    
    d,n = X_query.shape
    X_out=[]
    for i in range(n):
        X_list = [] #Will be a list of tuples of each gene's expression level and original index
        for j in range(d):
            X_list.append((X_query[j,i],j))
            
        X_list.sort(key= lambda x:x[0]) #Sort by gene expression level
        for j in range(len(X_list)):
            X_list[j] = (X_ref_sort[j],X_list[j][1]) #copy over expression from representative of reference set
            
        X_list.sort(key= lambda x:x[1]) #sort by original index
        X_row = [x[0] for x in X_list] #a row in the output matrix is a list of only the copied over expression values
        X_out.append(X_row)
    X_out = np.array(X_out).T
    
    return X_out
