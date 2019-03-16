import pdb
import numpy as np
import itertools
import math
import string
import pylab
import matplotlib
#import keras

#np.random.seed(0)
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import SGD, Adam
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import Callback
from keras.datasets import mnist
from keras import backend as K
from keras.initializers import VarianceScaling
from matplotlib import pyplot as plt

#feature selection tools
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA as sklearnPCA

# Utilities
import pickle






def get_series_data(filename,data_line,data_end, isLabeld, offset=1):
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
    #read down until data_line
    for _ in range(data_line):
        line = f.readline()
        if isLabeld == 'true' and _ == 0:
            labels = line.split('\t')

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

    #Remove the columns with missing data/labels
    bad_columns = list(set(bad_columns))
    bad_columns.sort(key = lambda x:-x) #we have to remove backwards
    for i in bad_columns:
        X = np.delete(X,i,1)


    return X,gene_ids,labels




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
        if line_split[id_col].lower() not in id_to_symbol_map:
            id_to_symbol_map[str(line_split[id_col]).lower()] = str(line_split[symbol_col]).lower().replace('\n', '')
        if line_split[symbol_col].lower() not in symbol_to_id_map:
            symbol_to_id_map[line_split[symbol_col].lower().replace('\n', '')] = line_split[id_col].lower()

        line = f.readline()
    
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
        
        

def KNN_sort_filtered(X_train,y_train,X_test,included_affy_file,train_genes_file,k=10,platform="affy",genes_list=None):
    '''
    This function takes in SCALED variance filtered training and test data, finds the mode of k closest neighbors for each sample of X_test
    train_genes_file - list of genes remaining after variance filtering that should be used
    platform - either "affy" or mapping of query set gene ID to Affy ID
    gene_list - list of query set gene IDs
    impute - should missing genes in query set be filled in with mean of reference set
    '''
    predicted_types = []
    confidences = []


    #If platform not affy we need to convert as much as possible
    if platform != "affy":
        Affy_genes = pickle.load(open(train_genes_file, "rb"))
        
        included_Affy_indices = pickle.load(open(included_affy_file, "rb"))
        print("Top two included genes in good rows: ",Affy_genes[included_Affy_indices[0]],Affy_genes[included_Affy_indices[1]])

        if platform=="xloc":
            ID_genes = pickle.load(open("Testing/gene_ids_genes_fpkm", "rb"))
            ID_genes = np.array(ID_genes)
            gene_to_Affy = pickle.load(open("Testing/Xloc_to_Affy", "rb"))
            Affy_to_gene = {}
            #set up the conversions in both directions
            for entry in list(gene_to_Affy.keys()):
                Affy_to_gene[gene_to_Affy[entry]] = entry
        else:
            gene_to_Affy = platform 
            ID_genes = genes_list 
            ID_genes = np.array(ID_genes)
            Affy_to_gene = {}
            #set up the conversions in both directions
            for entry in list(gene_to_Affy.keys()):
                Affy_to_gene[gene_to_Affy[entry]] = entry
            
            

    


        X_affy_good_rows = [] #For distances, we only want to use features that were successfully converted
        input_X_good_rows = [] #newly formatted query data including only mutual genes in the same order as the reference data
        count = 0
        genes_converted = 0
   
        
        
        ID_genes_list = ID_genes.tolist()
        ID_genes_set = set(ID_genes_list)
        X_test_list = X_test.tolist()
        X_test_mean = np.mean(X_test,axis=0)

        #for each gene we want to include
        for i in included_Affy_indices: 
            gene = Affy_genes[i]
            
            
            if gene in Affy_to_gene and Affy_to_gene[gene] in ID_genes_set:
                #convert if possible
                ID_gene = Affy_to_gene[gene]
                index = ID_genes_list.index(ID_gene)
                

                genes_converted+=1
                input_X_good_rows.append(X_test_list[index]) 
                X_affy_good_rows.append(X_train[i,:].tolist())
            count+=1
    
                
        print("genes converted: ",genes_converted," out of ",count)
    
        X_test = np.array(X_affy_good_rows) #reference set rows that were successfully converted
        X_train = np.array(input_X_good_rows) #query set rows that were successfully converted
        
        #pickle.dump(X_test, open("Testing/X_test_good_rows", "wb"))
        #pickle.dump(X_train, open("Testing/X_train_good_rows", "wb"))

    type_map = {'ccd11b': 'other', 'b': 'b', 'cd19': 'b', 'sc': 'other', 'fi': 'other', 'frc': 'other', 'bec': 'other', 'lec': 'other',
                'ep': 'other', 'st': 'other', 't': 't4', 'nkt': 'nkt', 'prob': 'b', 'preb': 'b', 'pret': 't4', 'mo': 'other', 'b1b': 'b1ab',
                'b1a': 'b1ab', 'dc': 'dc', 'gn': 'gn', 'nk': 'nk', 'mf': 'mf', 'tgd': 'tgd', 'cd4': 't4', 'mlp': 'other', 'cd8': 't8',
                't8': 't8', 'b1ab': 'b1ab', 'treg': 'treg', 't4': 't4', 'dn': 'other', 'eo': 'other', 'ilc1': 'other', 'ilc2': 'other',
                'ilc3': 'other', 'ba': 'other', 'mechi': 'other', 'mc': 'other', 'ccd11b-': 'other','b-cells':'b','nucleated':'other','lt-hsc':'other',
                'monocytes':'other','cd4+':'t4','cd8+':'t8','granulocytes':'gn','macrophage':'mf','hsc':'other','ilc':'other'}

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


def column_scale(X,scale="mean"):
    '''
    Scales data by dividing each column by its max, median, or mean as specified by scale
    '''
    d,n = X.shape
    column_maxes = np.amax(X,axis=0)
    column_medians = np.median(X,axis=0)
    column_means = np.mean(X,axis=0)

    for i in range(n):
        if scale=="max":
            X[:,i] = X[:,i]/column_maxes[i]
        elif scale=="median":
            X[:,i] = X[:,i]/column_medians[i]
        elif scale=="mean":
            X[:,i] = X[:,i]/column_means[i]
        elif scale=="none":
            pass
    return X



def shuffle(X_file,y_file,train_size):
    '''
    FOR TESTING CLASSIFIER
    This function takes in a dataset, shuffles it, then splits it into training and testing datasets
    '''
    X = pickle.load(open(X_file, "rb")).astype(np.float)
    y = pickle.load(open(y_file, "rb"))
    d,n = X.shape
    data = X.tolist()
    data.append(y.tolist())
    data = np.array(data)
    data = data.T
    np.random.shuffle(data)
    data = data.T
    
    X_shuffle = data[0:d,:]
    y_shuffle = data[d,:]
    return X_shuffle[:,0:train_size],y_shuffle[0:train_size],X_shuffle[:,train_size:],y_shuffle[train_size:]





def match_dist(X_ref,X_query): #, one-to-one scaling is ok
    '''
    Matches the distribution of the query set to the reference set.
    For each sample, genes are sorted by expression values, then the distribution of the average of the reference set is copied over
    query file and reference file should already contain only mutual, convertable genes
    '''
    
    print("X_ref shape: ",X_ref.shape,"query shape: ", X_query.shape)
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



def covariance_predict(X,y,X_test,threshold=.25,types=11):
    d,n = X.shape
    
    if types==20:
        type_map = {'ccd11b': 'cd4', 'b': 'b', 'cd19': 'b', 'sc': 'hsc', 'fi': 'stromal', 'frc': 'stromal', 'bec': 'stromal', 'lec': 'stromal',
                'ep': 'stromal', 'st': 'stromal', 't': 't4', 'nkt': 'nkt', 'prob': 'b', 'preb': 'b', 'pret': 't4', 'mo': 'mo', 'b1b': 'b1ab',
                'b1a': 'b1ab', 'dc': 'dc', 'gn': 'gn', 'nk': 'nk', 'mf': 'mf', 'tgd': 'tgd', 'cd4': 't4', 'mlp': 'other', 'cd8': 't8',
                't8': 't8', 'b1ab': 'b1ab', 'treg': 'treg', 't4': 't4', 'dn': 'stromal', 'eo': 'eo', 'ilc1': 'ilc', 'ilc2': 'ilc',
                'ilc3': 'ilc', 'ba': 'other', 'mechi': 'ep', 'mc': 'mc', 'ccd11b-': 't4', 'b-cells':'b','nucleated':'other','lt-hsc':'hsc',
                'monocytes':'mo','cd4+':'t4','cd8+':'t8','granulocytes':'gn', 'iap':'other', 'mmp4':'other','mmp3':'other', 'b1b':'other',
                'sthsc':'hsc', 'lthsc':'hsc','bec':'other', 'frc':'other', 'l1210':'other','macrophage':'mf','hsc':'hsc','ilc':'ilc'}
    else:
        type_map = {'ccd11b': 'other', 'b': 'b', 'cd19': 'b', 'sc': 'other', 'fi': 'other', 'frc': 'other', 'bec': 'other', 'lec': 'other',
                'ep': 'other', 'st': 'other', 't': 't4', 'nkt': 'nkt', 'prob': 'b', 'preb': 'b', 'pret': 't4', 'mo': 'other', 'b1b': 'b1ab',
                'b1a': 'b1ab', 'dc': 'dc', 'gn': 'gn', 'nk': 'nk', 'mf': 'mf', 'tgd': 'tgd', 'cd4': 't4', 'mlp': 'other', 'cd8': 't8',
                't8': 't8', 'b1ab': 'b1ab', 'treg': 'treg', 't4': 't4', 'dn': 'other', 'eo': 'other', 'ilc1': 'other', 'ilc2': 'other',
                'ilc3': 'other', 'ba': 'other', 'mechi': 'other', 'mc': 'other', 'ccd11b-': 'other','b-cells':'b','nucleated':'other','lt-hsc':'other',
                'monocytes':'other','cd4+':'t4','cd8+':'t8','granulocytes':'gn','macrophage':'mf','hsc':'other','ilc':'other'}
        

    rep_types = {}
    for cell_type in list(type_map.keys()):
        if type_map[cell_type] not in rep_types:
            #make a new vector for that rep_type
            rep_types[type_map[cell_type]] = [[cell_type],np.ones(d)/10,0] #Initialize the vector to .1 everywhere to prevent 0's that can cause issues with np.coercoff
        else:
            #add type to existing rep_type
            rep_types[type_map[cell_type]][0].append(cell_type)

    for i in range(len(y.tolist())):
        rep_types[type_map[y[i].lower()]][2]+=1 #keep track of how many samples are of this type
        rep_types[type_map[y[i].lower()]][1] += X[:,i] #and the sum of expressions for this type

        
    for rep in rep_types:
        if rep_types[rep][2] == 0:
            pass
        else:
            rep_types[rep][1] = rep_types[rep][1]/rep_types[rep][2] #average the samples to get a representative vector for each type

    del rep_types['other']


    d_test,n_test = X_test.shape
    wrong = 0
    correct = 0
    predictions = []
    confidences = []
    for i in range(n_test):
        #print("t_cell avg: ",t_cell_avg.T.shape)
        #print(X_affy[:,i].T.shape)
        covs = []
        for rep in rep_types:
            if np.corrcoef(np.stack((rep_types[rep][1].T,X_test[:,i].T),axis=0))[0,1] < 1:
                covs.append((np.corrcoef(np.stack((rep_types[rep][1].T,X_test[:,i].T),axis=0))[0,1],rep))
                #print(np.stack((rep_types[rep][1].T,X_test[:,i].T),axis=0))
                #print(np.corrcoef(np.stack((rep_types[rep][1].T,X_test[:,i].T),axis=0))[0,1])
                #print(np.corrcoef(np.stack((rep_types[rep][1].T,X_test[:,i].T),axis=0))[0,1]<1)

        cov,pred = max(covs, key=lambda x:x[0])
        predictions.append(pred)
        confidences.append(cov)
        
    return predictions,confidences


"""
Below is code used for creating the neural network classifier, but is not used in the webapp
"""

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.keys = ['loss', 'acc', 'val_loss', 'val_acc']
        self.values = {}
        for k in self.keys:
            self.values['batch_'+k] = []
            self.values['epoch_'+k] = []

    def on_batch_end(self, batch, logs={}):
        for k in self.keys:
            bk = 'batch_'+k
            if k in logs:
                self.values[bk].append(logs[k])

    def on_epoch_end(self, epoch, logs={}):
        for k in self.keys:
            ek = 'epoch_'+k
            if k in logs:
                self.values[ek].append(logs[k])

    def plot(self, keys):
        for key in keys:
            plt.plot(np.arange(len(self.values[key])), np.array(self.values[key]), label=key)
        plt.legend()

def run_keras(X_train, y_train, X_val, y_val, X_test, y_test, layers, epochs, split=0, verbose=True):
    # Model specification
    model = Sequential()
    for layer in layers:
        model.add(layer)
    # Define the optimization
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=["accuracy"])
    N = X_train.shape[0]
    # Pick batch size
    batch = 32 if N > 1000 else 1     # batch size
    history = LossHistory()
    # Fit the model
    if X_val is None:
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_split=split,
                  callbacks=[history], verbose=verbose)
    else:
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_data=(X_val, y_val),
                  callbacks=[history], verbose=verbose)
    # Evaluate the model on validation data, if any
    if X_val is not None or split > 0:
        val_acc, val_loss = history.values['epoch_val_acc'][-1], history.values['epoch_val_loss'][-1]
        print ("\nLoss on validation set:"  + str(val_loss) + " Accuracy on validation set: " + str(val_acc))
    else:
        val_acc = None
    # Evaluate the model on test data, if any
    if X_test is not None:
        test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=batch)
        print ("\nLoss on test set:"  + str(test_loss) + " Accuracy on test set: " + str(test_acc))
    else:
        test_acc = None
    return model, history, val_acc, test_acc

def keras_trials(trials, X_train,Y_train,X_test,Y_test,layers,split=.1):
    val_acc = 0
    test_acc = 0
    for trial in range(trials):
        # Reset the weights
        # See https://github.com/keras-team/keras/issues/341
        session = K.get_session()
        for layer in layers:
            for v in layer.__dict__:
                v_arg = getattr(layer, v)
                if hasattr(v_arg, 'initializer'):
                    initializer_func = getattr(v_arg, 'initializer')
                    initializer_func.run(session=session)
        # Run the model
        model, history, vacc, tacc = run_keras(X_train,Y_train,None,None,X_test,Y_test,layers,1,split,verbose=False)
        val_acc += vacc if vacc else 0
        test_acc += tacc if tacc else 0

    #print("Average test accuracy: ",test_acc/trials)
    return model,test_acc/trials




def fit_model(X,y,separate="11",filename="",trials=5,save_file=None):
    
    classes = []
    gene_ids = []
    
    num_classes = 10 #len(classes)
    print("X shape: ",X.shape)
    print("Y shape: ",y.shape)

    d,n = X.shape

    #For separating into 11 different cell types, we have mapping from the 25 specific types to 11 general:
    type_map = {'ccd11b': 'other', 'b': 'b', 'cd19': 'b', 'sc': 'other', 'fi': 'other', 'frc': 'other', 'bec': 'other', 'lec': 'other',
            'ep': 'other', 'st': 'other', 't': 't4', 'nkt': 'nkt', 'prob': 'b', 'preb': 'b', 'pret': 't4', 'mo': 'other', 'b1b': 'b1ab',
            'b1a': 'b1ab', 'dc': 'dc', 'gn': 'gn', 'nk': 'nk', 'mf': 'mf', 'tgd': 'tgd', 'cd4': 't4', 'mlp': 'other', 'cd8': 't8',
            't8': 't8', 'b1ab': 'b1ab', 'treg': 'treg', 't4': 't4', 'dn': 'other', 'eo': 'other', 'ilc1': 'other', 'ilc2': 'other',
            'ilc3': 'other', 'ba': 'other', 'mechi': 'other', 'mc': 'other', 'ccd11b-': 'other'}

    
    cell_type_to_int = {}
    int_to_cell_type = {}
    cell_types = list(set(list(type_map.values())))
    print('cell_types ',cell_types)
    num_classes = len(cell_types)
    
    for i in range(len(cell_types)):
        cell_type_to_int[cell_types[i]]=i
        int_to_cell_type[i] = cell_types[i]

    cell_labels = []
    for label in y.tolist():
        if label in type_map:
            cell_labels.append(cell_type_to_int[type_map[label]])
        else:
            cell_labels.append(cell_type_to_int['Other'])
    y = np.array(cell_labels)
    pickle.dump(int_to_cell_type, open("Testing/int_to_cell_type_for_"+ save_file +"_" + str(num_classes) + "_types", "wb"))
           
    
    
    test_acc=0
    for i in range(trials):

        data = X.tolist()
        data.append(y.tolist())
        data = np.array(data)
        data = data.T
        np.random.shuffle(data)
        data = data.T
        X_shuffle = data[0:d,:].T
        y_shuffle = data[d,:].T

        print("done shuffling")

        layers = [Dense(input_dim=d, units=500, activation='relu'),
                  Dense(units=250, activation='relu'),
                  Dense(units=num_classes, activation="softmax")]

        test_split = 1.2
        X_train = X_shuffle[0:math.floor(n/test_split),:]
        Y_train = y_shuffle[0:math.floor(n/test_split)]
        X_test = X_shuffle[math.floor(n/test_split):,:]
        Y_test = y_shuffle[math.floor(n/test_split):]

        print("num classes = ",num_classes)
        Y_train = np_utils.to_categorical(Y_train, num_classes)
        Y_test = np_utils.to_categorical(Y_test, num_classes)

        model,acc = keras_trials(1,X_train,Y_train,X_test,Y_test,layers)
        test_acc+=acc
        if save_file != None:
            model.save("Testing/" + save_file + ".h5")
        else:
            model.save("Testing/" + filename[5:-4] + "_" + str(num_classes) + "_types_model.h5")
    print("average test accuracy: ",test_acc/trials)
    return test_acc/trials, model


#Used for visualization, not for prediction
def plot_two_genes(X,y,X2):
    type_map = {'ccd11b': 'cd4', 'b': 'b', 'cd19': 'b', 'sc': 'hsc', 'fi': 'stromal', 'frc': 'stromal', 'bec': 'stromal', 'lec': 'stromal',
                'ep': 'stromal', 'st': 'stromal', 't': 't4', 'nkt': 'nkt', 'prob': 'b', 'preb': 'b', 'pret': 't4', 'mo': 'mo', 'b1b': 'b1ab',
                'b1a': 'b1ab', 'dc': 'dc', 'gn': 'gn', 'nk': 'nk', 'mf': 'mf', 'tgd': 'tgd', 'cd4': 't4', 'mlp': 'other', 'cd8': 't8',
                't8': 't8', 'b1ab': 'b1ab', 'treg': 'treg', 't4': 't4', 'dn': 'stromal', 'eo': 'eo', 'ilc1': 'ilc', 'ilc2': 'ilc',
                'ilc3': 'ilc', 'ba': 'other', 'mechi': 'ep', 'mc': 'mc', 'ccd11b-': 't4', 'b-cells':'b','nucleated':'other','lt-hsc':'hsc',
                'monocytes':'mo','cd4+':'t4','cd8+':'t8','granulocytes':'gn', 'iap':'other', 'mmp4':'other','mmp3':'other', 'b1b':'other',
                'sthsc':'hsc', 'lthsc':'hsc','bec':'other', 'frc':'other', 'l1210':'other','macrophage':'mf','hsc':'hsc','ilc':'ilc'}
    
    pylab.clf()
    pylab.cla()
    pylab.close()
    d,n = X.shape

    
    
    cell_type = type_map[y[0]]
    legend = [type_map[y[0]]]
    i=0
    row_val = []
    col_val = []
    cell_types_array = {}
    for i in range(n):
        if type_map[y[i]] not in cell_types_array:
            cell_types_array[type_map[y[i]]] = ([],[])
    for i in range(n):
        cell_types_array[type_map[y[i]]][0].append(X[0,i])
        cell_types_array[type_map[y[i]]][1].append(X[1,i])
    legend_list = []
    i=0
    for cell_type in cell_types_array:
        i+=1
        if i<8:
            pylab.plot(cell_types_array[cell_type][0],cell_types_array[cell_type][1],'o')
            legend_list.append(cell_type)
        elif i<20 and i>10:
            pylab.plot(cell_types_array[cell_type][0],cell_types_array[cell_type][1],'^')
            legend_list.append(cell_type)
        else:
            pass
        
        
    '''
    while(i<n):
        while(type_map[y[i]]==cell_type):
            row_val.append(X[0,i])
            col_val.append(X[1,i])
            i+=1
            if i>=n:
                break
        pylab.plot(row_val,col_val,'o')
        if i<n:
            row_val = []
            col_val = []
            cell_type=type_map[y[i]]
            legend.append(type_map[y[i]])
    '''
    row_val = X2[0,:].tolist()
    col_val = X2[1,:].tolist()
    legend_list.append("Test Set - nkt")
    pylab.plot(row_val,col_val,'x')
    pylab.legend(legend_list)
    pylab.title("Distribution Matched, Magic Imputed SC and Microarray data plotted by expression of two genes")
    print("LEGEND: ",legend_list)
    pylab.show()
