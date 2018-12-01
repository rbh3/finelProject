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



def get_series_data(filename,data_line,data_end,offset=1):
    """
    This function opens a .txt file, reads until the type_line, uses those entries as labels (array y).
    Then it reads until data_line and creates an array of expression data (specifically log2(expression+1))
    Offset is only used for weird GSE files where the labels present parsing difficulties
    """
    bad_columns = [] #columns that we throw out due to missing label/data


    #read down until data_line
    for _ in range(data_line):
        line = f.readline()

    #Set up X - the array of data
    X = []
    gene_ids = []

    count = data_line
    while(line!="!series_matrix_table_end" and count<data_end):
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

    pickle.dump(X, open("Testing/input_unscaled_data_for_"+ filename[:-4], "wb"))
    pickle.dump(gene_ids, open("Testing/gene_ids_"+ filename[:-4], "wb"))

    print("X: ",X)
    print("shape of X: ",X.shape)

    return X,gene_ids


#TEST: get_series_data('data/GSE15907_series_matrix.txt',59,77,24999)


def Affy_gene_code_map():
    id_to_symbol_map = {}
    symbol_to_id_map = {}
    filename = 'GeneID_symbol_map.txt'
    f = open(filename, 'r')
    for _ in range(29):
        line = f.readline()

    count = 0
    while(line!="" and count<100000):
        count+= 1
        if count%5000 == 0:
            print("reading line: ",count)
        try:
            line_split = line.split(sep="\t")
            id_to_symbol_map[line_split[0]] = line_split[2].lower()
            symbol_to_id_map[line_split[2].lower()] = line_split[0]
        except:
            pass
        line = f.readline()
    pickle.dump(id_to_symbol_map, open("Testing/AffyID_to_symbol_map", "wb"))
    pickle.dump(symbol_to_id_map, open("Testing/symbol_to_AffyID_map", "wb"))
    #print(list(symbol_to_id_map.keys())[0:5])
    return id_to_symbol_map


def Xloc_gene_code_map():
    id_to_symbol_map = {}
    symbol_to_id_map = {}
    filename = 'genes_attr.txt'
    f = open(filename, 'r')
    for _ in range(2):
        line = f.readline()

    count = 0
    while(line!="" and count<100000):
        count+= 1
        if count%5000 == 0:
            print("reading line: ",count)
        try:
            line_split = line.split(sep="\t")
            id_to_symbol_map[line_split[0]] = line_split[4].lower()
            symbol_to_id_map[line_split[4].lower()] = line_split[0]
        except:
            pass
        line = f.readline()
    pickle.dump(id_to_symbol_map, open("Testing/XlocID_to_symbol_map", "wb"))
    pickle.dump(symbol_to_id_map, open("Testing/symbol_to_XlocID_map", "wb"))
    print(list(symbol_to_id_map.keys())[0:5])
    return id_to_symbol_map

def gene_code_map(filename, data_row, data_end, symbol_col, id_col,save_file="unknown"):
    #opens up the txt file with gene ids and symbols to create conversion dictionaries
    id_to_symbol_map = {}
    symbol_to_id_map = {}
    f = open(filename, 'r')
    for _ in range(data_row):
        line = f.readline()

    count = 0
    while(line!="!platform_table_end" and line != "" and count<data_end):
        count+= 1
        if count%5000 == 0:
            print("reading line: ",count)
            #print(line)
        try:
            line_split = line.split(sep="\t")
            if line_split[id_col] not in id_to_symbol_map:
                id_to_symbol_map[line_split[id_col.lower()]] = line_split[symbol_col].lower()
            if line_split[symbol_col].lower() not in symbol_to_id_map:
                symbol_to_id_map[line_split[symbol_col].lower()] = line_split[id_col.lower()]
        except:
            pass
        line = f.readline()
    pickle.dump(id_to_symbol_map, open("Testing/"+save_file+"_ID_to_symbol_map", "wb"))
    pickle.dump(symbol_to_id_map, open("Testing/"+save_file+"_symbol_to_ID_map", "wb"))
    print(list(symbol_to_id_map.keys())[0:5])
    print(list(id_to_symbol_map.keys())[0:5])
    print(id_to_symbol_map[list(id_to_symbol_map.keys())[0]])
    return id_to_symbol_map

def gene_symbol_to_affy(file):
    symbol_to_affy = pickle.load(open("Testing/symbol_to_AffyID_map", "rb"))
    gene_list = pickle.load(open("Testing/gene_ids_"+file, "rb"))
    id_to_affy = {}
    genes = set(gene_list)
    print(gene_list[0:5])
    for key in symbol_to_affy:
        #print(key)
        if key in genes:
            #print("converted: ",key)
            id_to_affy[key] = symbol_to_affy[key]
    pickle.dump(id_to_affy, open("Testing/"+file+"_to_affy", "wb"))

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




def fit_model(X,y,separate="11",filename="",trials=5,save_file=None): #filename,type_line,data_line,data_end,offset=0,trials=5,var_threshold=0,feature_selector=None,new_dimension=100):
    #X,y,classes,gene_ids = get_series_data(filename,type_line,data_line,data_end,offset,variance_threshold=var_threshold,feature_selector=feature_selector,new_dimension=new_dimension)#'data/genes_fpkm.txt',1,2,68879)
    
    classes = []
    gene_ids = []
    
    num_classes = 10 #len(classes)
    print("X shape: ",X.shape)
    print("Y shape: ",y.shape)

    d,n = X.shape

    
    #For separating into 11 different cell types, we have mapping from the 25 specific types to 11 general:
    type_map_11 = type_map = {'ccd11b': 'other', 'b': 'b', 'cd19': 'b', 'sc': 'other', 'fi': 'other', 'frc': 'other', 'bec': 'other', 'lec': 'other',
            'ep': 'other', 'st': 'other', 't': 't4', 'nkt': 'nkt', 'prob': 'b', 'preb': 'b', 'pret': 't4', 'mo': 'other', 'b1b': 'b1ab',
            'b1a': 'b1ab', 'dc': 'dc', 'gn': 'gn', 'nk': 'nk', 'mf': 'mf', 'tgd': 'tgd', 'cd4': 't4', 'mlp': 'other', 'cd8': 't8',
            't8': 't8', 'b1ab': 'b1ab', 'treg': 'treg', 't4': 't4', 'dn': 'other', 'eo': 'other', 'ilc1': 'other', 'ilc2': 'other',
            'ilc3': 'other', 'ba': 'other', 'mechi': 'other', 'mc': 'other', 'ccd11b-': 'other'}
    type_map_T = {'CCD11b':'Other','T':'T', 'NKT':'T', 'preT':'T','Tgd':'T', 'CD4':'T','CD8':'T',"T8":"T","Treg":"T","T4":"T"}
    type_map_T_specific = {'CCD11b':'Other','T':'T4', 'NKT':'NKT', 'preT':'T4','Tgd':'Tgd', 'CD4':'T4','CD8':'T8',"T8":"T8","Treg":"Treg","T4":"T4"}
    if separate == "T":
        type_map = type_map_T
    elif separate == "T_specific":
        type_map = type_map_T_specific
    else:
        type_map = type_map_11
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
        #print(X)
        #print(y)
        data = X.tolist()
        data.append(y.tolist())
        data = np.array(data)
        #print("datashape ",data.shape)
        data = data.T
        np.random.shuffle(data)
        #print("datashate ",data.shape)
        data = data.T
        #print("datashate ",data.shape)
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
        #print(X_train.shape,Y_train.shape)


        print("num classes = ",num_classes)
        #print("pre: ",Y_train)
        Y_train = np_utils.to_categorical(Y_train, num_classes)
        #print("post: ",Y_train)
        Y_test = np_utils.to_categorical(Y_test, num_classes)



        model,acc = keras_trials(1,X_train,Y_train,X_test,Y_test,layers)
        test_acc+=acc
        if save_file != None:
            model.save("Testing/" + save_file + ".h5")
        else:
            model.save("Testing/" + filename[5:-4] + "_" + str(num_classes) + "_types_model.h5")
    print("average test accuracy: ",test_acc/trials)
    return test_acc/trials





#fit_model('data/GSE37448_series_matrix.txt',26,65,24987,offset=1,var_threshold=2,feature_selector='pca',new_dimension = 100)

'''
output = []
for i in range(12,15):#[2,1.85,1.75,1.5,1,.5]:
    output.append((i/2,))
print(output)
#
'''
#print("X: ",X)



'''
#%%%%%%%%%%%%%%%%%%%%
#PREDICTION PROCESS
#%%%%%%%%%%%%%%%%%%%%
model = load_model("Testing/GSE15907_series_matrix_model.h5")
X_sample = pickle.load(open("Testing/Sample_X", "rb")).T
X_rebuild = [float(i) for i in X_sample.tolist()[0]]
X_sample = np.array([X_rebuild])
pca = pickle.load(open("Testing/pca_for_GSE15907_series_matrix", "rb"))
print("Shape of sample_X: ",X_sample.shape)
print("type of X: ",type(X_sample))
X_dummy = np.array([[0 for i in range(24922)]])
print("Shape of dummy_X: ",X_dummy.shape)
print("type of dummy_X: ",type(X_dummy))

print("X_dummy: ",X_dummy)
print("X_sample: ",X_sample)

X = pca.transform(X_dummy)
X2 = pca.transform(X_sample)
print("Shape of pca_X: ",X2.shape)
pred = model.predict(X2)
print(np.argmax(pred,axis=1))
int_to_type = pickle.load(open("Testing/int_to_cell_type_for_GSE37448_series_matrix", "rb"))
print(int_to_type[np.argmax(pred,axis=1)[0]])
'''


def test_25_types(input_file_name,label_file_name):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Testing against other data set - using 25 types
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    model = load_model("Testing/GSE15907_series_matrix_model.h5")
    int_to_type = pickle.load(open("Testing/int_to_cell_type_for_GSE15907_series_matrix", "rb"))
    X= pickle.load(open(input_file_name, "rb")).T.astype(np.float)
    y= pickle.load(open(label_file_name, "rb"))
    print('Shape of X other data',X.shape)
    X_sample = pickle.load(open("Testing/Sample_X", "rb")).T
    print("Shape of sample_X: ",X_sample.shape)
    pca = pickle.load(open("Testing/pca_for_GSE15907_series_matrix", "rb"))
    X = pca.transform(X)
    n,d = X.shape
    correct = 0
    incorrect = 0
    unsure = 0
    unfair = 0
    threshold = .5
    for i in range(n):
        pred = model.predict(X[i:i+1,:])
        #print(int_to_type)
        #print(pred)
        prob = pred[0,np.argmax(pred,axis=1)[0]]

        if y[i] in list(int_to_type.values()):
            if prob>threshold:
                print('prediction: ',int_to_type[np.argmax(pred,axis=1)[0]],' probability: ',pred[0,np.argmax(pred,axis=1)[0]],' actual: ',y[i])
                if int_to_type[np.argmax(pred,axis=1)[0]]==y[i]:
                    correct+= 1
                else:
                    incorrect+=1
            else:
                unsure += 1
        else:
            unfair += 1
    print("correct: ",correct," wrong: ",incorrect," unsure: ",unsure,' unfair: ',unfair)

#test_25_types("Testing/input_data_for_GSE37448_series_matrix","Testing/labels_for_GSE37448_series_matrix")
#test_25_types("Testing/fpkm_converted_to_Affy","Testing/labels_for_genes_fpkm")

def test_10_types(input_file_name,label_file_name,model,pca,int_to_type_file):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Testing against other data set - using 10 types
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    filename = 'X_unscaled_combo'
    type_map = {'ccd11b': 'other', 'b': 'b', 'cd19': 'b', 'sc': 'other', 'fi': 'other', 'frc': 'other', 'bec': 'other', 'lec': 'other',
            'ep': 'other', 'st': 'other', 't': 't4', 'nkt': 'nkt', 'prob': 'b', 'preb': 'b', 'pret': 't4', 'mo': 'other', 'b1b': 'b1ab',
            'b1a': 'b1ab', 'dc': 'dc', 'gn': 'gn', 'nk': 'nk', 'mf': 'mf', 'tgd': 'tgd', 'cd4': 't4', 'mlp': 'other', 'cd8': 't8',
            't8': 't8', 'b1ab': 'b1ab', 'treg': 'treg', 't4': 't4', 'dn': 'other', 'eo': 'other', 'ilc1': 'other', 'ilc2': 'other',
            'ilc3': 'other', 'ba': 'other', 'mechi': 'other', 'mc': 'other', 'ccd11b-': 'other'}
    model = load_model(model) #caled_c_12_types_model.h5") #
    int_to_type = pickle.load(open("Testing/"+int_to_type_file, "rb"))
    X= pickle.load(open(input_file_name, "rb")).T.astype(np.float)
    y= pickle.load(open(label_file_name, "rb"))
    print('Shape of X other data',X.shape)
    X_sample = pickle.load(open("Testing/Sample_X", "rb")).T
    print("Shape of sample_X: ",X_sample.shape)
    pca = pickle.load(open(pca, "rb")) #pickle.load(open("Testing/pca_for_GSE15907_series_matrix", "rb")) #
    X = pca.transform(X)

    type_to_int = {}
    for key in int_to_type:
        type_to_int[int_to_type[key]] = key
    
    n,d = X.shape
    correct = 0
    incorrect = 0
    unsure = 0
    unfair = set()
    threshold = .6
    confusion_matrix = np.zeros((12,12))
    for i in range(n):#n):
        pred = model.predict(X[i:i+1,:])
        #print(int_to_type)
        #print(pred)
        prob = pred[0,np.argmax(pred,axis=1)[0]]
        likely_type = np.argmax(pred,axis=1)[0]
        #print(likely_type)
        confusion_matrix[np.argmax(pred,axis=1)[0],type_to_int[type_map[y[i]]]] += 1

        if y[i] in type_map:
            if prob>threshold:
                '''
                if type_map[y[i]]=='Other':
                    #print('prediction: ',int_to_type[np.argmax(pred,axis=1)[0]],' probability: ',pred[0,np.argmax(pred,axis=1)[0]],' actual: ',type_map[y[i]],' which is: ',y[i])
                else:
                    #print('prediction: ',int_to_type[np.argmax(pred,axis=1)[0]],' probability: ',pred[0,np.argmax(pred,axis=1)[0]],' actual: ',type_map[y[i]])
                '''
                if int_to_type[np.argmax(pred,axis=1)[0]]==type_map[y[i]]:
                    correct+= 1
                else:
                    incorrect+=1
                    print('prediction: ',int_to_type[np.argmax(pred,axis=1)[0]],' probability: ',pred[0,np.argmax(pred,axis=1)[0]],' actual: ',type_map[y[i]])
            else:
                unsure += 1
        else:
            unfair.add(y[i])
    print("correct: ",correct," wrong: ",incorrect," unsure: ",unsure,' unfair: ',unfair)
    '''
    print(confusion_matrix)
    fig, ax = plt.subplots()

    # Hide axes
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False)

    
    clust_data = confusion_matrix
    collabel=tuple([int_to_type[x]for x in range(12)])
    rowlabel = tuple([int_to_type[x]for x in range(12)])
    plt.title("Confusion Matrix - 12 types")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    ax.table(cellText=clust_data,colLabels=collabel,rowLabels = rowlabel,loc='center')
    plt.show()
    '''

#test_10_types("Testing/input_data_for_GSE37448_series_matrix","Testing/labels_for_GSE37448_series_matrix")
#test_10_types("Testing/fpkm_converted_to_filtered_Affy","Testing/labels_for_genes_fpkm")
#test_10_types("Testing/pca_transformed_X_for_variance_expression_filtered_"+"X_unscaled_combo","Testing/y_combo")

#Xloc_gene_code_map()

#model, history, val_acc, test_acc = run_keras(X_train,Y_train,None,None,X_test,Y_test,layers,1,split = .1,verbose=False)
#model.save('C:\Users\caboonie\Documents\ImmunoGenML\MicroarrayFull')


#fit_model('data/genes_fpkm.txt',1,2,68880,offset=0,var_threshold=2,feature_selector='pca',new_dimension = 100)        
'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Procedure for Xloc to AffyID map
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pickle.load(open("Testing/gene_ids_genes_fpkm", "rb"))

fpkm_genes = pickle.load(open("Testing/gene_ids_genes_fpkm", "rb"))
print(fpkm_genes[0:3])
XlocID_to_symbol_map = pickle.load(open("Testing/XlocID_to_symbol_map", "rb"))
symbol_to_affy = pickle.load(open("Testing/symbol_to_AffyID_map", "rb"))
translated_genes = 0
non_translated = 0
XlocID_to_AffyID = {}
for gene in fpkm_genes:
    try:
        XlocID_to_AffyID[gene]=symbol_to_affy[XlocID_to_symbol_map[gene]]
        translated_genes += 1
    except:
        non_translated += 1
print("number of translated genes: ",translated_genes," non_translated: ",non_translated)
pickle.dump(XlocID_to_AffyID, open("Testing/Xloc_to_Affy", "wb"))
'''

def gene_to_Affy_map(gene_to_symbol_file,gene_id_genes_file,data_file,save_file=None):
    #pickle.load(open("Testing/gene_ids_genes_fpkm", "rb"))
    gene_id_genes = pickle.load(open(gene_id_genes_file, "rb"))
    X = pickle.load(open(data_file, "rb")).astype(np.float)
    print(gene_id_genes[0:3])
    ID_to_symbol_map = pickle.load(open(gene_to_symbol_file, "rb"))
    print(list(ID_to_symbol_map.keys())[0:3])
    print(ID_to_symbol_map[list(ID_to_symbol_map.keys())[0]])
    print(gene_id_genes[0])
    print(ID_to_symbol_map[gene_id_genes[0]])
    symbol_to_affy = pickle.load(open("Testing/symbol_to_AffyID_map", "rb"))
    translated_genes = 0
    non_translated = 0
    ID_to_Affy = {}
    gene_expression = {} #keeps track of current average expression for gene
    symbols_seen = set()
    for gene in gene_id_genes:
        #print("symbol: ",ID_to_symbol_map[gene])
        #print(ID_to_symbol_map[gene] in symbol_to_affy)
        try:           
            if ID_to_symbol_map[gene] in symbols_seen:
                #choose the gene ID with the higher reported mean expression according to Tal
                #print("repeat symbol:", ID_to_symbol_map[gene],gene)
                if np.mean(X[gene_id_genes.index(gene),:]) > gene_expression[ID_to_symbol_map[gene]]:
                    ID_to_Affy[gene]=symbol_to_affy[ID_to_symbol_map[gene]]
                    gene_expression[ID_to_symbol_map[gene]] = np.mean(X[gene_id_genes.index(gene),:])
                    #print("replacing repeat symbol:", ID_to_symbol_map[gene],gene)
                else:
                    #print("not replaced")
                    pass
            else:
                ID_to_Affy[gene]=symbol_to_affy[ID_to_symbol_map[gene]]
                #print(gene_id_genes.index(gene))
                #print(X[gene_id_genes.index(gene),:].shape)
                gene_expression[ID_to_symbol_map[gene]] = np.mean(X[gene_id_genes.index(gene),:])
                translated_genes += 1
                symbols_seen.add(ID_to_symbol_map[gene])
        except:
            non_translated += 1
    print("number of translated genes: ",translated_genes," non_translated: ",non_translated)
    if save_file==None:
        pickle.dump(ID_to_Affy, open(gene_to_symbol_file+"_to_Affy", "wb"))
    else:
        pickle.dump(ID_to_Affy, open(save_file, "wb"))




#Need to construct an input matrix in the same form as the Affy X matrix, so for

def convert_Xloc_to_Affy():
    #Takes the input values from genes.fpkm file which has Xloc gene ID's, and converts it into the typical Affy gene ID setup
    #used by the model. Genes that cannot be converted (i.e. no equivalent Xloc gene for the necessary Affy gene) are replaced by a
    #mean value of training data.
    Affy_genes = pickle.load(open("Testing/gene_ids_GSE15907_series_matrix", "rb"))
    Xloc_genes = pickle.load(open("Testing/gene_ids_genes_fpkm", "rb"))
    X_affy = pickle.load(open("Testing/input_data_for_GSE15907_series_matrix", "rb")).astype(np.float)
    X_fpkm = pickle.load(open("Testing/input_data_for_genes_fpkm", "rb")).astype(np.float)
    Xloc_to_Affy = pickle.load(open("Testing/Xloc_to_Affy", "rb"))
    Affy_to_Xloc = {}
    d,n = X_fpkm.shape
    print("n: ",n," d: ",d)


    for entry in list(Xloc_to_Affy.keys()):
        #print("Xloc gene: ",entry," Affy gene: ",Xloc_to_Affy[entry])
        Affy_to_Xloc[Xloc_to_Affy[entry]] = entry

    X = []
    count = 0
    genes_converted = 0
    print(Affy_genes[0:3])
    for gene in Affy_genes:
        count+=1
        if count%500==0:
            print("count: ",count)
        if gene in Affy_to_Xloc:
            #print(gene)
            Xloc_gene = Affy_to_Xloc[gene]
            index = Xloc_genes.tolist().index(Xloc_gene)
            X.append(X_fpkm.tolist()[index])
            genes_converted+=1
        else:
            index = Affy_genes.index(gene)
            #print(X_affy[index,:].shape)
            X.append([np.mean(X_affy[index,:]) for _ in range(n)])
    X = np.array(X)
    print("shape of X: ",X.shape)
    print("genes converted: ",genes_converted)
    pickle.dump(X, open("Testing/fpkm_converted_to_Affy", "wb"))

#convert_Xloc_to_Affy()

#print("Test acc: ",test_acc)

#Baseline Methods:
#TCR expressed or not.
#for each gene, if in the tcr set, then add it's expression to the sum for that data point
def tcr_expression_baseline(filename,platform):
    X_affy = pickle.load(open("Testing/input_unscaled_data_for_"+filename, "rb")).astype(np.float) #pickle.load(open("Testing/input_data_for_"+filename, "rb")).astype(np.float)
    Affy_genes = pickle.load(open("Testing/gene_ids_"+filename, "rb")) #pickle.load(open("Testing/gene_ids_GSE15907_series_matrix", "rb"))
    txt_filename = "tcr_"+platform+"_genes.txt"#"tcr_affy_genes.txt"
    f = open(txt_filename, 'r')
    tcr_genes = set()
    line = f.readline()
    while(line != ""):
        tcr_genes.add(line[:-1])
        line = f.readline()
    #print(tcr_genes)
    
    #Affy_genes = ['10414793','10414796',10414799,10414802,10414805,10414807]
        
    #print(tcr_genes)
    tcr_expression = []
    for gene in Affy_genes:
        #print(gene)
        if gene in tcr_genes:
            index = Affy_genes.index(gene)
            tcr_expression.append(X_affy[index,:])
    tcr_expression = np.array(tcr_expression)
    print(tcr_expression.shape)
    pickle.dump(tcr_expression, open("Testing/tcr_expression_for_"+filename, "wb"))
        

def tcr_expression_classify(filename):
    tcr_expression = pickle.load(open("Testing/tcr_expression_for_"+filename, "rb")).astype(np.float)
    y= pickle.load(open("Testing/labels_for_"+filename, "rb"))
    d,n = tcr_expression.shape
    print("d: ",d)
    tcr_sums = np.sum(tcr_expression, axis = 0)/d
    print(tcr_sums.shape)
    print(tcr_sums[0:20])
    if filename=="GSE15907_series_matrix":
        threshold = 6
    elif filename=="genes_fpkm":
        threshold = 1
    else:
        threshold = 3
    wrong = 0
    correct = 0
    t_cells = ["T","preT","CD4","CD8","Tgd","NKT","T4","T8","Treg"]
    for i in range(len(tcr_sums.tolist())):
        
        if tcr_sums.tolist()[i]>threshold:
            
            if y[i] in t_cells:
                correct += 1
            else:
                wrong += 1
                print(y.tolist()[i],tcr_sums.tolist()[i])
        else:
            if y[i] not in t_cells:
                correct+= 1
            else:
                wrong += 1
                print(y.tolist()[i],tcr_sums.tolist()[i])
    print("correct: ",correct," wrong: ",wrong)

#tcr_expression_baseline("GSE15907_series_matrix","affy")
#tcr_expression_classify("GSE15907_series_matrix")
#tcr_expression_baseline("genes_fpkm","xloc")
#tcr_expression_classify("genes_fpkm")


def correlation_baseline(data,labels,threshold):   
    X_affy = data
    y = labels
    d,n = X_affy.shape
    print(X_affy.shape)
    print(y.shape)
    t_cells = ["T","preT","CD4","CD8","Tgd","NKT","T4","T8","Treg"]
    t_cell_sum = np.zeros(d)
    t_cell_num = 0
    for i in range(len(y.tolist())):
        if y.tolist()[i] in t_cells:
            t_cell_sum += X_affy[:,i]
            t_cell_num += 1
    t_cell_avg = t_cell_sum/t_cell_num
    print(t_cell_num)
    print(t_cell_avg.shape)
    correlations = []
    #threshold = 3.6
    wrong = 0
    correct = 0
    for i in range(n):
        #print("t_cell avg: ",t_cell_avg.T.shape)
        #print(X_affy[:,i].T.shape)
        correlation = np.cov(np.stack((t_cell_avg.T,X_affy[:,i].T),axis=0))[0,1]#np.correlate(t_cell_avg,X_affy[:,i])
        correlations.append(correlation)
        if correlation>threshold:            
            if y[i] in t_cells:
                correct += 1
            else:
                wrong += 1
                print(y.tolist()[i],correlation)
        else:
            if y[i] not in t_cells:
                correct+= 1
            else:
                wrong += 1
                print(y.tolist()[i],correlation)
    print("correct: ",correct," wrong: ",wrong,correct/(correct+wrong))
    return correct/(correct+wrong)
        


    
def pre_filter_var_and_expression(var_threshold,express_threshold,filename,scaled=False):
    #take in the data after +1/log2 transformation, but without scaling, remove feature if variance is below threshold/low expression, then run PCA
    X_raw = pickle.load(open(filename, "rb")).astype(np.float)
    print(X_raw.shape)
    d,n = X_raw.shape
    X = []
    indices_of_included_features = []
    removed_due_to_expression = 0
    removed_due_to_var = 0
    for i in range(d):
        row = X_raw[i,:]
        var = np.var(row)
        express = np.mean(row)
        if scaled:
            scale = np.max(row)
        else:
            scale=1
        if i==0 or i==1:
            print(row[0:4],' var: ',var,' expression: ',express,' scale: ',scale)
        if var>=var_threshold:
            if express>express_threshold:
                row = row/scale
                X.append(row.tolist())
                indices_of_included_features.append(i)
            else:
                removed_due_to_expression += 1
        else:
            removed_due_to_var += 1
    X = np.array(X)
    remaining = d-removed_due_to_var- removed_due_to_expression
    print(removed_due_to_var, removed_due_to_expression,remaining)
    if scaled:
        pickle.dump(X, open(filename+"scaled_input_data_variance_expression_filtered_"+str(remaining), "wb"))
    else:
        pickle.dump(X, open(filename+"unscaled_input_data_variance_expression_filtered_"+str(remaining), "wb"))
    pickle.dump(indices_of_included_features, open(filename+"indices_of_"+str(remaining)+"_included_features", "wb"))
    return remaining



#remaining = pre_filter_var_and_expression(3,6,"input_unscaled_data_for_genes_fpkm")
#X_affy = pickle.load(open("Testing/unscaled_input_data_variance_expression_filtered_input_unscaled_data_for_genes_fpkm", "rb")).astype(np.float)
#y= pickle.load(open("Testing/labels_for_genes_fpkm", "rb"))
#correlation_baseline(X_affy,y,2.2)

#remaining = pre_filter_var_and_expression(3,6,"input_unscaled_data_for_GSE15907_series_matrix")
#X_affy = pickle.load(open("Testing/unscaled_input_data_variance_expression_filtered_input_unscaled_data_for_GSE15907_series_matrix", "rb")).astype(np.float)
#y= pickle.load(open("Testing/labels_for_GSE15907_series_matrix", "rb"))
#correlation_baseline(X_affy,y,3)




"""
#FOR TESTING COVARIANCE BASELINE
acc = []
for i in [2.75,3,3.25,3.5,3.75]:
    remaining = pre_filter_var_and_expression(3,6,"input_unscaled_data_for_GSE15907_series_matrix")
    X_affy = pickle.load(open("Testing/unscaled_input_data_variance_expression_filtered_input_unscaled_data_for_GSE15907_series_matrix", "rb")).astype(np.float)
    y= pickle.load(open("Testing/labels_for_GSE15907_series_matrix", "rb"))
    acc.append(correlation_baseline(X_affy,y,i))
print(remaining,acc)
"""

def pca_for_filtered_data(new_dimension,filename,scaled=False):
    #Transform input data using pca, and save pca matrix for prediction purposes
    if scaled:
        X = pickle.load(open("Testing/scaled_input_data_variance_expression_filtered_"+filename, "rb")).astype(np.float)
    else:
        X = pickle.load(open("Testing/unscaled_input_data_variance_expression_filtered_"+filename, "rb")).astype(np.float)
    d,n = X.shape
    sklearn_pca = sklearnPCA(n_components=new_dimension)
    X=sklearn_pca.fit_transform(X.T).T
    if scaled:
        pickle.dump(sklearn_pca, open("Testing/pca_for_variance_expression_filtered_scaled_"+filename+"_"+str(d)+"_genes", "wb"))
        pickle.dump(X, open("Testing/pca_transformed_X_for_variance_expression_filtered_scaled_"+filename+"_"+str(d)+"_genes", "wb"))
    else:
        pickle.dump(sklearn_pca, open("Testing/pca_for_variance_expression_filtered_unscaled_"+filename+"_"+str(d)+"_genes", "wb"))
        pickle.dump(X, open("Testing/pca_transformed_X_for_variance_expression_filtered_unscaled_"+filename+"_"+str(d)+"_genes", "wb"))



    
#If we can delete feature through filtering before pca, this will make the knn process during prediction time much faster
#When a data point is missing, we want to replace it with the median of its k nearest neighbors. For this reason, we need to eliminate features to make this process much faster.
#Procedure: Convert input data to filtered Affy format. Then replace missing features with median of knn

def XLOC_to_filtered_Affy_with_KNN(k,num_genes,scaled=False):
    #When a data point is missing, we want to replace it with the median of its k nearest neighbors. For this reason, we need to eliminate features to make this process much faster.
    filename = 'X_unscaled_combo'
    Affy_genes = pickle.load(open("Testing/gene_ids_GSE15907_series_matrix", "rb"))
    Xloc_genes = pickle.load(open("Testing/gene_ids_genes_fpkm", "rb"))
    Xloc_genes = np.array(Xloc_genes)
    y_affy = pickle.load(open("Testing/y_combo", "rb")) #pickle.load(open("Testing/labels_for_GSE15907_series_matrix", "rb"))
    print(type(Xloc_genes))
    if scaled:
        X_affy = pickle.load(open("Testing/X_combo", "rb")).T.astype(np.float)#scaled #pickle.load(open("Testing/input_data_for_GSE15907_series_matrix", "rb")).astype(np.float) #scaled
        X_fpkm = pickle.load(open("Testing/input_data_for_genes_fpkm", "rb")).astype(np.float) #also scaled
    else:
        X_affy = pickle.load(open("Testing/X_unscaled_combo", "rb")).T.astype(np.float) #pickle.load(open("Testing/X_combo", "rb")).T.astype(np.float)#scaled #pickle.load(open("Testing/input_data_for_GSE15907_series_matrix", "rb")).astype(np.float) #scaled
        X_fpkm = pickle.load(open("Testing/input_unscaled_data_for_genes_fpkm", "rb")).astype(np.float) #pickle.load(open("Testing/input_data_for_genes_fpkm", "rb")).astype(np.float) #also scaled
    included_Affy_indices = pickle.load(open("Testing/indices_of_"+str(num_genes)+"_included_features_"+filename, "rb"))
    Xloc_to_Affy = pickle.load(open("Testing/Xloc_to_Affy", "rb"))
    Affy_to_Xloc = {}
    d,n = X_fpkm.shape
    print("n: ",n," d: ",d)

    #set up the conversions in both directions
    for entry in list(Xloc_to_Affy.keys()):
        Affy_to_Xloc[Xloc_to_Affy[entry]] = entry

    X = [] #newly formated array of fpkm data in Affy order
    missing_rows_affy = [] #when a gene can't be converted, we need to know which row of Affy data to use to fill it in
    missing_rows_X = [] #We also need to know where in the new X there is a gap
    X_affy_good_rows = [] #For distances, we only want to use features that were successfully converted
    input_X_good_rows = [] #X, but without the non-convertable rows
    count = 0
    genes_converted = 0
    print(Affy_genes[0:3])
    
    #for each gene we want to include
    for i in included_Affy_indices: 
        gene = Affy_genes[i]
            
        if count%300==0:
            print("count: ",count)
        if gene in Affy_to_Xloc:
            #convert if possible
            Xloc_gene = Affy_to_Xloc[gene]
            index = Xloc_genes.tolist().index(Xloc_gene)
            X.append(X_fpkm.tolist()[index])
            genes_converted+=1
            X_affy_good_rows.append(X_affy[i,:].tolist())
            input_X_good_rows.append(X_fpkm.tolist()[index])
        else:
            #otherwise mark where we have an empty row
            X.append([0 for _ in range(n)])
            missing_rows_affy.append(i)
            missing_rows_X.append(count)
        count+=1
    
                
    X = np.array(X)
    print("shape of X: ",X.shape)
    print("genes converted: ",genes_converted," out of ",count)
    
    X_affy_good_rows = np.array(X_affy_good_rows) #GSE data for rows that were successfully converted
    input_X_good_rows = np.array(input_X_good_rows) #fpkm data for rows that were successfully converted

    pickle.dump(X, open("Testing/Temp/X", "wb"))
    pickle.dump(X_affy_good_rows, open("Testing/Temp/X_affy_good_rows_"+filename, "wb"))
    pickle.dump(input_X_good_rows, open("Testing/Temp/input_X_good_rows_"+filename, "wb"))
    pickle.dump(missing_rows_affy, open("Testing/Temp/missing_rows_affy_"+filename, "wb"))
    pickle.dump(missing_rows_X, open("Testing/Temp/missing_rows_X_"+filename, "wb"))
    
    '''
    X = pickle.load(open("Testing/Temp/X", "rb"))
    X_affy_good_rows = pickle.load(open("Testing/Temp/X_affy_good_rows", "rb"))
    input_X_good_rows = pickle.load(open("Testing/Temp/input_X_good_rows", "rb"))
    missing_rows_affy = pickle.load(open("Testing/Temp/missing_rows_affy", "rb"))
    missing_rows_X = pickle.load(open("Testing/Temp/missing_rows_X", "rb"))
    '''
    print('missing rows: ',missing_rows_X[0:4])
    
    print(X_affy_good_rows.shape)
    d,n = input_X_good_rows.shape
    d,affy_n = X_affy_good_rows.shape
    for i in range(n): #for each point in X.fpkm
        if i%5 == 0:
            print("filling in missing genes for point: ",i)
        closest_examples = []
        for j in range(affy_n):
            dist = np.linalg.norm(X_affy_good_rows[:,j]-input_X_good_rows[:,i])
            closest_examples.append((j,dist))
        closest_examples.sort(key= lambda x:x[1])
        for m in range(3):
            #print(closest_examples[m])
            print("closest type:",y_affy[closest_examples[m][0]])
        k_neighbors = []
        for j in range(k):
            k_neighbors.append(X_affy[:,closest_examples[j][0]])
        k_neighbors = np.array(k_neighbors).T
        for index in range(len(missing_rows_affy)):
            X[missing_rows_X[index],i] = np.median(k_neighbors[missing_rows_affy[index],:])
            if i == 0:
                if index == 0:
                    print("median: ",np.median(k_neighbors[missing_rows_affy[index],:]))
    print(X)
            
    
    
    if scaled:
        pickle.dump(X, open("Testing/scaled_fpkm_converted_to_filtered_Affy_"+filename+str(num_genes), "wb"))
    else:
        pickle.dump(X, open("Testing/unscaled_fpkm_converted_to_filtered_Affy_"+filename+str(num_genes), "wb"))

    pass

'''
#TESTING VAR_FILTERED PCA
for i in [.5,.75,1,1.5,2]:
    remaining = pre_filter_var_and_expression(i,6,"X_unscaled_combo")
    pca_for_filtered_data(min(300,remaining),"X_unscaled_combo")
    filename = "X_unscaled_combo"
    X = pickle.load(open("Testing/pca_transformed_X_for_variance_expression_filtered_"+filename, "rb")).astype(np.float)
    y = pickle.load(open("Testing/y_combo", "rb"))
    fit_model(X,y)
    print(i,remaining)
    for _ in range(5):
        print('%%%%%%%%%%%%%%%%%%%%%%')
'''

'''
#Test scaled pca
remaining = pre_filter_var_and_expression(1.5,6,"X_unscaled_combo",scaled=True)
print("REMAINING: ",remaining)
pca_for_filtered_data(min(200,remaining),"X_unscaled_combo",scaled=True)
filename = "X_unscaled_combo"
X = pickle.load(open("Testing/pca_transformed_X_for_variance_expression_filtered_scaled_"+filename+"_"+str(remaining)+"_genes", "rb")).astype(np.float)
y = pickle.load(open("Testing/y_combo", "rb"))
pca = "Testing/pca_for_variance_expression_filtered_scaled_"+filename+"_"+str(remaining)+"_genes"
#XLOC_to_filtered_Affy_with_KNN(10,remaining,scaled=True)
save_file="scaled_pca_"
fit_model(X,y,save_file="scaled_pca_"+str(remaining))
int_to_type_file = "int_to_cell_type_for_"+ "scaled_pca_"+str(remaining) +"_" + str(12) + "_types"
test_10_types("Testing/scaled_fpkm_converted_to_filtered_Affy_X_unscaled_combo1421","Testing/labels_for_genes_fpkm","Testing/scaled_pca_"+str(remaining)+".h5",pca,int_to_type_file) #"Testing/GSE15907_series_matrix_10_types_model.h5")

#Test unscaled pca
remaining = pre_filter_var_and_expression(1.5,6,"X_unscaled_combo",scaled=False)
print("REMAINING: ",remaining)
pca_for_filtered_data(min(200,remaining),"X_unscaled_combo",scaled=False)
filename = "X_unscaled_combo"
X = pickle.load(open("Testing/pca_transformed_X_for_variance_expression_filtered_unscaled_"+filename+"_"+str(remaining)+"_genes", "rb")).astype(np.float)
y = pickle.load(open("Testing/y_combo", "rb"))
pca = "Testing/pca_for_variance_expression_filtered_unscaled_"+filename+"_"+str(remaining)+"_genes"
XLOC_to_filtered_Affy_with_KNN(10,remaining,scaled=False)
fit_model(X,y,save_file="unscaled_pca_"+str(remaining))
int_to_type_file = "int_to_cell_type_for_"+ "unscaled_pca_"+str(remaining) +"_" + str(12) + "_types"
test_10_types("Testing/unscaled_fpkm_converted_to_filtered_Affy_X_unscaled_combo1421","Testing/labels_for_genes_fpkm","Testing/unscaled_pca_"+str(remaining)+".h5",pca,int_to_type_file)
'''

#open(, "wb"))
#        pickle.dump(X, open(+filename+"_"+d+"_genes", "wb"))

#filename = "X_unscaled_combo"
#X = pickle.load(open("Testing/pca_transformed_X_for_variance_expression_filtered_"+filename, "rb")).astype(np.float)
#y = pickle.load(open("Testing/y_combo", "rb"))
#fit_model(X,y,filename = "X_unscaled_KNN_combo")
#XLOC_to_filtered_Affy_with_KNN(10,1421)
#fit_model_combo()
#test_10_types("Testing/fpkm_converted_to_filtered_Affy_X_unscaled_combo1421","Testing/labels_for_genes_fpkm")

def generate_plots(filename,filename2=None):
    #for each gene in a dataset, we want to find the average value of log(expression+1), discretize it into bins of 0-15 then plot number of genes per bin
    X = pickle.load(open(filename, "rb")).astype(np.float) #dataset - unscaled, but treated with log2(x+1)
    d,n = X.shape
    bin_mult = 3 #how many bins we want is 15 times the bin multiplier
    print(d)
    bins = [0 for i in range(16*bin_mult)]
    for row in range(d): #rows represent genes
        mean = np.mean(X[row,:])
        bins[math.floor(mean*bin_mult)] += 1
    pylab.plot([x/bin_mult for x in range(16*bin_mult)][1:],bins[1:])

    if filename2 != None:
        X = pickle.load(open(filename2, "rb")).astype(np.float) #dataset - unscaled, but treated with log2(x+1)
        d,n = X.shape
        bins = [0 for i in range(16*bin_mult)]
        for row in range(d): #rows represent genes
            mean = np.mean(X[row,:])
            bins[math.floor(mean*bin_mult)] += 1
        pylab.plot([x/bin_mult for x in range(16*bin_mult)][1:],bins[1:])
    
    pylab.xlabel("median of log(exp+1)")
    pylab.ylabel("Number of genes")
    pylab.title(filename)
    pylab.legend(('GSE37448 - Microarray','GSE15907 - Microarray'))
    pylab.show()

#('data/genes_fpkm.txt',1,2,68880,offset=0,var_threshold=2,feature_selector='pca',new_dimension = 100) 
#get_series_data('data/genes_fpkm.txt',1,2,68880,offset=0)
#generate_plots("unscaled_input_data_variance_expression_filtered_input_unscaled_data_for_GSE15907_series_matrix")
#generate_plots("input_unscaled_data_for_GSE15907_series_matrix","input_unscaled_data_for_genes_fpkm")  

#generate_plots("input_unscaled_data_for_genes_fpkm","input_unscaled_data_for_GSE15907_series_matrix")

#X = pickle.load(open("Testing/input_unscaled_data_for_genes_fpkm", "rb")).astype(np.float)
#print(X)

def pca_plots():
    X = pickle.load(open("Testing/pca_transformed_X_for_variance_expression_filtered_"+"X_unscaled_combo", "rb")).astype(np.float)
    y = pickle.load(open("Testing/y_combo", "rb"))
    pca = pickle.load(open("Testing/pca_for_variance_expression_filtered_"+"X_unscaled_combo", "rb"))
    gene_id_index = pickle.load(open("Testing/indices_of_"+str(1421)+"_included_features_"+"X_unscaled_combo", "rb"))
    gene_id = pickle.load(open("Testing/gene_ids_GSE15907_series_matrix", "rb"))
    id_to_symbol = pickle.load(open("Testing/AffyID_to_symbol_map", "rb"))
    print(pca.explained_variance_ratio_[0:2])
    components = pca.components_
    print(X.shape)
    d,n = X.shape
    print(components.shape)
    p,q = components.shape
    sort_comp = np.sort(components)
    print(sort_comp[0:2,:])
    #we need to get the indices of the maxes and mins to see the most relevant factors
    pc_1_genes = []
    pc_1_label = ""
    pc_2_label = ""
    for i in [0,1,2,-3,-2,-1]:
        amount_1 = sort_comp[0,i]
        index_1 = components[0,:].tolist().index(amount_1)
        pc_1_genes.append(id_to_symbol[gene_id[gene_id_index[index_1]]])
        amount_2 = sort_comp[1,i]
        index_2 = components[1,:].tolist().index(amount_2)
        print(amount_1)
        print("index: ", index_1)
        print(gene_id_index[index_1])
        print("gene_id: ",gene_id[gene_id_index[index_1]])
        print("gene symbol: ",id_to_symbol[gene_id[gene_id_index[index_1]]])
        if i<0:
            pc_1_label += " +"+id_to_symbol[gene_id[gene_id_index[index_1]]]
        else:
            pc_1_label += " -"+id_to_symbol[gene_id[gene_id_index[index_1]]]
        if i<0:
            pc_2_label += " +"+id_to_symbol[gene_id[gene_id_index[index_2]]]
        else:
            pc_2_label += " -"+id_to_symbol[gene_id[gene_id_index[index_2]]]
    
    #print(np.amax(components[0,:]), np.argmax(components[0,:]))
    #print(np.amin(components[0,:]), np.argmin(components[0,:]))
    type_map = {'ccd11b': 'other', 'b': 'b', 'cd19': 'b', 'sc': 'other', 'fi': 'other', 'frc': 'other', 'bec': 'other', 'lec': 'other',
            'ep': 'other', 'st': 'other', 't': 't4', 'nkt': 'nkt', 'prob': 'b', 'preb': 'b', 'pret': 't4', 'mo': 'other', 'b1b': 'b1ab',
            'b1a': 'b1ab', 'dc': 'dc', 'gn': 'gn', 'nk': 'nk', 'mf': 'mf', 'tgd': 'tgd', 'cd4': 't4', 'mlp': 'other', 'cd8': 't8',
            't8': 't8', 'b1ab': 'b1ab', 'treg': 'treg', 't4': 't4', 'dn': 'other', 'eo': 'other', 'ilc1': 'other', 'ilc2': 'other',
            'ilc3': 'other', 'ba': 'other', 'mechi': 'other', 'mc': 'other', 'ccd11b-': 'other'}
    cell_types_array = {}
    for i in range(n):
        if type_map[y[i]] not in cell_types_array:
            cell_types_array[type_map[y[i]]] = ([],[])
    for i in range(n):
        cell_types_array[type_map[y[i]]][0].append(X[0,i])
        cell_types_array[type_map[y[i]]][1].append(X[1,i])
    legend_list = []
    for cell_type in cell_types_array:
        pylab.plot(cell_types_array[cell_type][0],cell_types_array[cell_type][1],'o')
        legend_list.append(cell_type)
    #pylab.plot(X[0,:].tolist(),X[1,:].tolist(),'o')
    #"Testing/fpkm_converted_to_filtered_Affy_"+filename+str(num_genes)
    pylab.legend(tuple(legend_list))
    pylab.title("PCA - top two components on "+str(q)+" top variance genes")
    pylab.xlabel(pc_1_label)
    pylab.ylabel(pc_2_label)
    pylab.show()

#pca_plots()

#T or not, then specialize:
def T_or_not_specialize():
    X = pickle.load(open("Testing/pca_transformed_X_for_variance_expression_filtered_"+"X_unscaled_combo", "rb")).astype(np.float)
    d,n = X.shape
    y = pickle.load(open("Testing/y_combo", "rb"))
    acc_1 = fit_model(X,y,separate = "T",filename = "X_unscaled_combo")
    int_to_type = pickle.load(open("Testing/int_to_cell_type_for_caled_c_2_types", "rb"))
    
    #Now using that model, we'll take vectors if they're predicted to be "T", we keep them
    model = load_model("Testing/caled_c_2_types_model.h5")
    X_T_only = []
    y_T_only = []
    for i in range(n):
        X_T = X.T
        pred = model.predict(X_T[i:i+1,:])
        #print(int_to_type[np.argmax(pred,axis=1)[0]])
        if int_to_type[np.argmax(pred,axis=1)[0]]=="T":
            #print(X_T[i,:].shape)
            X_T_only.append(X_T[i,:].tolist())
            y_T_only.append(y[i])
    X_T_only = np.array(X_T_only).T
    y_T_only = np.array(y_T_only)
    acc_2 = fit_model(X_T_only,y_T_only,separate = "T_specific",filename = "X_unscaled_combo")
    print("overall accuracy: ",acc_1*acc_2)

def T_specialize():
    X = pickle.load(open("Testing/pca_transformed_X_for_variance_expression_filtered_"+"X_unscaled_combo", "rb")).astype(np.float)
    d,n = X.shape
    y = pickle.load(open("Testing/y_combo", "rb"))
    fit_model(X,y,separate = "T_specific",filename = "X_unscaled_combo")
    
    
#T_or_not_specialize()
#T_specialize()

def KNN_sort_filtered(train_data,train_labels,test_data,test_labels,included_affy_file,k=10,platform="affy",genes_list_file=None,check_expression=False,impute=False):
    #Takes in SCALED variance filtered training and test data, finds the mode of k closest neighbors for each point
    X_train = pickle.load(open(train_data, "rb")).astype(np.float)
    y_train = pickle.load(open(train_labels, "rb"))#[80:]
    X_test = pickle.load(open(test_data, "rb")).astype(np.float)
    y_test = pickle.load(open(test_labels, "rb"))#[:80]

    print("X_train shape: ",X_train.shape)
    print("X_test shape: ",X_test.shape)


    #If platform not affy we need to convert as much as possible
    if platform != "affy":
        Affy_genes = pickle.load(open("Testing/gene_ids_GSE15907_series_matrix", "rb"))
        
        included_Affy_indices = pickle.load(open(included_affy_file, "rb"))

        if platform=="xloc":
            ID_genes = pickle.load(open("Testing/gene_ids_genes_fpkm", "rb"))
            ID_genes = np.array(ID_genes)
            gene_to_Affy = pickle.load(open("Testing/Xloc_to_Affy", "rb"))
            Affy_to_gene = {}
            #set up the conversions in both directions
            for entry in list(gene_to_Affy.keys()):
                Affy_to_gene[gene_to_Affy[entry]] = entry
        else:
            gene_to_Affy = pickle.load(open(platform, "rb"))
            ID_genes = pickle.load(open(genes_list_file, "rb"))
            ID_genes = np.array(ID_genes)
            Affy_to_gene = {}
            #set up the conversions in both directions
            for entry in list(gene_to_Affy.keys()):
                Affy_to_gene[gene_to_Affy[entry]] = entry
            

    

        X = [] #newly formated array of fpkm data in Affy order
        missing_rows_affy = [] #when a gene can't be converted, we need to know which row of Affy data to use to fill it in
        missing_rows_X = [] #We also need to know where in the new X there is a gap
        X_affy_good_rows = [] #For distances, we only want to use features that were successfully converted
        input_X_good_rows = [] #X, but without the non-convertable rows
        count = 0
        genes_converted = 0
        #print(Affy_genes[0:3])
        
        #for each gene we want to include
        ID_genes_list = ID_genes.tolist()
        X_test_list = X_test.tolist()
        X_test_mean = np.mean(X_test,axis=0)
        print("shape of means: ",X_test_mean.shape)
        for i in included_Affy_indices: 
            gene = Affy_genes[i]
            
            if count%300==0:
                print("count: ",count)
            if gene in Affy_to_gene:
                #convert if possible
                
                ID_gene = Affy_to_gene[gene]
                index = ID_genes_list.index(ID_gene)
                if not check_expression or np.median(X_test[index,:])>.1:
                    X.append(X_test_list[index])
                    genes_converted+=1
                    input_X_good_rows.append(X_test_list[index]) #input_X_good_rows.append((X_test[index,:]+4).tolist()) # 
                    X_affy_good_rows.append(X_train[i,:].tolist())
                    
            elif impute:
                genes_converted+=1
                input_X_good_rows.append(X_test_mean) #input_X_good_rows.append((X_test[index,:]+4).tolist()) # 
                X_affy_good_rows.append(X_train[i,:].tolist())
            else:
                #eclude row
                pass
            count+=1
    
                
        X = np.array(X)
        print("shape of X: ",X.shape)
        print("genes converted: ",genes_converted," out of ",count)
    
        X_affy_good_rows = np.array(X_affy_good_rows) #GSE data for rows that were successfully converted
        input_X_good_rows = np.array(input_X_good_rows) #fpkm data for rows that were successfully converted

        #pickle.dump(X, open("Testing/Temp/X", "wb"))
        pickle.dump(X_affy_good_rows, open(train_data+"_good_rows_"+str(genes_converted), "wb"))
        pickle.dump(input_X_good_rows, open(test_data+"_good_rows_"+str(genes_converted), "wb"))
        #pickle.dump(X_affy_good_rows, open("Testing/X_affy_good_rows_1421", "wb"))
        #pickle.dump(input_X_good_rows, open("Testing/X_fpkm_good_rows_1421", "wb"))

        #print("Done dumping")
    
        '''
        X = pickle.load(open("Testing/Temp/X", "rb"))
        X_affy_good_rows = pickle.load(open("Testing/Temp/X_affy_good_rows", "rb"))
        input_X_good_rows = pickle.load(open("Testing/Temp/input_X_good_rows", "rb"))
        missing_rows_affy = pickle.load(open("Testing/Temp/missing_rows_affy", "rb"))
        missing_rows_X = pickle.load(open("Testing/Temp/missing_rows_X", "rb"))
        '''
        X_test = input_X_good_rows
        X_train = X_affy_good_rows


    type_map = {'ccd11b': 'cd4', 'b': 'b', 'cd19': 'b', 'sc': 'hsc', 'fi': 'stromal', 'frc': 'stromal', 'bec': 'stromal', 'lec': 'stromal',
            'ep': 'stromal', 'st': 'stromal', 't': 't4', 'nkt': 'nkt', 'prob': 'b', 'preb': 'b', 'pret': 't4', 'mo': 'mo', 'b1b': 'b1ab',
            'b1a': 'b1ab', 'dc': 'dc', 'gn': 'gn', 'nk': 'nk', 'mf': 'mf', 'tgd': 'tgd', 'cd4': 't4', 'mlp': 'other', 'cd8': 't8',
            't8': 't8', 'b1ab': 'b1ab', 'treg': 'treg', 't4': 't4', 'dn': 'stromal', 'eo': 'eo', 'ilc1': 'ilc', 'ilc2': 'ilc',
            'ilc3': 'ilc', 'ba': 'other', 'mechi': 'ep', 'mc': 'mc', 'ccd11b-': 't4', 'b-cells':'b','nucleated':'other','lt-hsc':'hsc',
            'monocytes':'mo','cd4+':'t4','cd8+':'t8','granulocytes':'gn', 'iap':'other', 'mmp4':'other','mmp3':'other', 'b1b':'other',
            'sthsc':'hsc', 'lthsc':'hsc','bec':'other', 'frc':'other', 'l1210':'other'}
    print(X_train.shape)
    print(X_test.shape)
    d,n = X_test.shape
    d,n_train = X_train.shape
    correct=0
    for i in range(n): #for each point in X.fpkm
        frequency = {}
        if i%50 == 0:
            print("classifying for point: ",i)
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
        #cell_type = max(frequency.iterkeys(), key =(lambda key: frequency[key]))
        #print("predicted: ",cell_type," actual: ",y_test[i])
        if cell_type==type_map[y_test[i]]:
            correct += 1
        else:
            print("predicted: ",cell_type," actual: ",type_map[y_test[i]])
    print("Accuracy: ",correct/n)


def column_scale(file,scale="mean"):
    X = pickle.load(open(file, "rb")).astype(np.float) #pickle.load(open("Testing/input_unscaled_data_for_genes_fpkm", "rb")).astype(np.float) #
    print("X ",X)
    d,n = X.shape
    column_maxes = np.amax(X,axis=0)
    column_medians = np.median(X,axis=0)
    column_means = np.mean(X,axis=0)
    #print("column maxes shape ",column_maxes.shape)
    #print("column means: ",column_means.shape)
    #print(column_means)
    for i in range(n):
        if scale=="max":
            X[:,i] = X[:,i]/column_maxes[i]
        elif scale=="median":
            X[:,i] = X[:,i]/column_medians[i]
        elif scale=="mean":
            X[:,i] = X[:,i]/column_means[i]
        elif scale=="none":
            pass
    print("X:",X)
    pickle.dump(X, open(file+"_colScaled_"+scale, "wb"))


#"Testing/X_unscaled_combo"

def shuffle(X_file,y_file,train_size):
    X = pickle.load(open(X_file, "rb")).astype(np.float)
    y = pickle.load(open(y_file, "rb"))
    d,n = X.shape
    data = X.tolist()
    data.append(y.tolist())
    data = np.array(data)
    print("datashape ",data.shape)
    data = data.T
    np.random.shuffle(data)
    print("datashape ",data.shape)
    data = data.T
    print("datashape ",data.shape)
    X_shuffle = data[0:d,:]
    print(X_shuffle.shape)
    y_shuffle = data[d,:]
    pickle.dump(X_shuffle[:,0:train_size], open(X_file+"_shuffle_train", "wb"))
    pickle.dump(y_shuffle[0:train_size], open(y_file+"_shuffle_train", "wb"))
    pickle.dump(X_shuffle[:,train_size:], open(X_file+"_shuffle_test", "wb"))
    pickle.dump(y_shuffle[train_size:], open(y_file+"_shuffle_test", "wb"))



'''
train_data = "Testing/X_unscaled_combo"#"Testing/X_colScaled_combo_shuffle_train_mean" #
train_labels="Testing/y_combo" #"Testing/y_combo_shuffle_train_mean"#_shuffle_train" #

test_data = "Testing/input_unscaled_data_for_genes_fpkm" #"Testing/X_colScaled_combo_shuffle_test_mean" #"Testing/X_colScaled_fpkmmean" #"Testing/X_colScaled_combo_shuffle_test" #"Testing/input_data_for_genes_fpkm" 
test_labels = "Testing/labels_for_genes_fpkm" #"Testing/y_combo_shuffle_test_mean"# "Testing/y_combo"#_shuffle_test" # "Testing/y_combo_shuffle_test" #
filename = 'X_unscaled_combo'
included_affy_file = "Testing/indices_of_"+str(1421)+"_included_features_"+filename
KNN_sort_filtered(train_data,train_labels,test_data,test_labels,included_affy_file,platform="xloc",k=3)
'''

#pre_filter_var_and_expression(1.5,6,"X_combo",scaled=True)

#need to test translating fpkm by 4 to right. For each good row, just add 4?

def match_dist(ref_file,query_file): #query file and reference file should already contain only good convertable genes, one-to-one scaling is ok
    #matching distributions.  Assemble converted fpkm. For each point, sort by expression, then copy over 
    #Average reference and Sort
    X_ref =  pickle.load(open(ref_file, "rb")).astype(np.float)
    X_query = pickle.load(open(query_file, "rb")).astype(np.float)

    #X_ref = np.array([[1,2,3],[2,1,1],[3,3,2]])
    #X_query = np.array([[.1,.2],[.2,.3],[.3,.1]])
    
    print("X_ref shape: ",X_ref.shape,"query shape: ", X_query.shape)
    #print(X_ref)
    X_ref_mean = np.median(X_ref,axis=1) #works same if not better (averages remove variance) # X_ref[:,200]#X_ref[:,0]#
    print("X_ref mean shape: ",X_ref_mean.shape)
    X_ref_sort = X_ref_mean.tolist()
    X_ref_sort.sort()
    print("X_ref_sort: ",X_ref_sort[0:5])
    d,n = X_query.shape
    X_out=[]
    for i in range(n):
        X_list = []
        for j in range(d):
            X_list.append((X_query[j,i],j))
        X_list.sort(key= lambda x:x[0])
        for j in range(len(X_list)):
            X_list[j] = (X_ref_sort[j],X_list[j][1])
        X_list.sort(key= lambda x:x[1])
        X_row = [x[0] for x in X_list]
        X_out.append(X_row)
    X_out = np.array(X_out).T
    print("X out: ",X_out)
    pickle.dump(X_out, open(query_file+"dist_matched", "wb"))

'''
train_data = "Testing/X_unscaled_combo"
train_labels="Testing/y_combo"
test_data = "Testing/input_unscaled_data_for_genes_fpkm"
test_labels = "Testing/labels_for_genes_fpkm" 
filename = 'X_unscaled_combo'
included_affy_file = "Testing/indices_of_"+str(1421)+"_included_features_"+filename
KNN_sort_filtered(train_data,train_labels,test_data,test_labels,included_affy_file,platform="xloc",k=3)
'''
'''
ref_file = "Testing/X_affy_good_rows_1421"
query_file = "Testing/X_fpkm_good_rows_1421"
match_dist(ref_file,query_file)

train_data = "Testing/X_affy_good_rows_1421dist_matched"
train_labels="Testing/y_combo"
test_data = "Testing/X_fpkm_good_rows_1421dist_matched_train" #"Testing/X_affy_good_rows_1421dist_matched_test" #
test_labels = "Testing/labels_for_genes_fpkm"  #"Testing/y_combo" #
filename = 'X_unscaled_combo'
included_affy_file = "Testing/indices_of_"+str(1421)+"_included_features_"+filename
KNN_sort_filtered(train_data,train_labels,test_data,test_labels,included_affy_file,k=3,platform="affy")
'''
'''
column_scale("Testing/X_unscaled_combo",scale="mean")
column_scale("Testing/input_unscaled_data_for_genes_fpkm",scale="mean")
train_data = "Testing/X_unscaled_combo_colScaled_mean"
train_labels="Testing/y_combo"
test_data = "Testing/input_unscaled_data_for_genes_fpkm_colScaled_mean"
test_labels = "Testing/labels_for_genes_fpkm"
filename = 'X_unscaled_combo'
included_affy_file = "Testing/indices_of_"+str(1421)+"_included_features_"+filename
KNN_sort_filtered(train_data,train_labels,test_data,test_labels,included_affy_file,k=3,platform="xloc")
'''
#filename = "data/GSE6506_gene_conversion.txt"
#gene_code_map(filename, 153, 45253, 10, 0,"GSE6506")
#filename = "data/GSE6506_series_matrix.txt"
#get_series_data(filename,47,76,45176,offset=1)

#filename = "data/GSE14833_gene_conversion.txt"
#gene_code_map(filename, 153, 10, 0)
#filename = "data/GSE14833_series_matrix.txt"
#get_series_data(filename,32,64,45164,offset=1)

#gene_to_Affy_map("Testing/GSE6506_ID_to_symbol_map","Testing/gene_ids_GSE6506_series_matrix","Testing/input_unscaled_data_for_GSE6506_series_matrix",save_file="Testing/GSE6505_ID_to_Affy")
#gene_to_Affy_map("Testing/GSE6506_ID_to_symbol_map","Testing/gene_ids_GSE14833_series_matrix","Testing/input_unscaled_data_for_GSE14833_series_matrix",save_file="Testing/GSE14833_ID_to_Affy")

#pre_filter_var_and_expression(1.5,6,"Testing/X_unscaled_combo",scaled=False)


#filename="Testing/X_unscaled_combo" #_filtered_1421"
#shuffle(filename,"Testing/y_combo",600)

'''
train_data = "Testing/X_unscaled_combo" #filename+"_shuffle_train"#"Testing/X_unscaled_combo"+"_shuffle_train" #
train_labels="Testing/y_combo"#+"_shuffle_train"
test_data = "Testing/input_unscaled_data_for_GSE6506_series_matrix" #filename+"_shuffle_test" #"Testing/X_unscaled_combo"+"_shuffle_test" #
test_labels = "Testing/labels_for_GSE6506_series_matrix" #"Testing/y_combo"+"_shuffle_test" #
filename = 'X_unscaled_combo'
included_affy_file = "Testing/indices_of_"+str(1421)+"_included_features_"+filename
KNN_sort_filtered(train_data,train_labels,test_data,test_labels,included_affy_file,k=3,platform="Testing/GSE6505_ID_to_Affy",genes_list_file="Testing/gene_ids_GSE6506_series_matrix")
'''

'''
match_dist("Testing/X_unscaled_combo_good_rows_1102","Testing/input_unscaled_data_for_GSE6506_series_matrix_good_rows_1102")
train_data = "Testing/X_unscaled_combo_good_rows_1102" #filename+"_shuffle_train"#"Testing/X_unscaled_combo"+"_shuffle_train" #
train_labels="Testing/y_combo"#+"_shuffle_train"
test_data = "Testing/input_unscaled_data_for_GSE6506_series_matrix_good_rows_1102dist_matched_train"
test_labels = "Testing/labels_for_GSE6506_series_matrix" #"Testing/y_combo"+"_shuffle_test" #
filename = 'X_unscaled_combo'
included_affy_file = "Testing/indices_of_"+str(1421)+"_included_features_"+filename
KNN_sort_filtered(train_data,train_labels,test_data,test_labels,included_affy_file,k=3,platform="affy") #"Testing/GSE6505_ID_to_Affy",genes_list_file="Testing/gene_ids_GSE6506_series_matrix")
'''
'''
column_scale("Testing/X_unscaled_combo",scale="max")
column_scale("Testing/input_unscaled_data_for_GSE6506_series_matrix",scale="max")
train_data = "Testing/X_unscaled_combo_colScaled_max" #filename+"_shuffle_train"#"Testing/X_unscaled_combo"+"_shuffle_train" #
train_labels="Testing/y_combo"#+"_shuffle_train"
test_data = "Testing/input_unscaled_data_for_GSE6506_series_matrix_colScaled_max" #filename+"_shuffle_test" #"Testing/X_unscaled_combo"+"_shuffle_test" #
test_labels = "Testing/labels_for_GSE6506_series_matrix" #"Testing/y_combo"+"_shuffle_test" #
filename = 'X_unscaled_combo'
included_affy_file = "Testing/indices_of_"+str(1421)+"_included_features_"+filename
KNN_sort_filtered(train_data,train_labels,test_data,test_labels,included_affy_file,k=3,platform="Testing/GSE6505_ID_to_Affy",genes_list_file="Testing/gene_ids_GSE6506_series_matrix")
'''

#Need to test column scaling for mean, max, dist-matching, also generate plots
#generate_plots("Testing/input_unscaled_data_for_GSE6506_series_matrix","Testing/input_unscaled_data_for_GSE15907_series_matrix")


'''
type_map = {'CCD11b':'Other','B':'B', 'CD19':'B','SC':'Other','Fi':'Other', 'FRC':'Other', 'BEC':'Other'
                    , 'LEC':'Other',  'Ep':'Other' , 'St':'Other', 'T':'T4', 'NKT':'NKT', 'proB':'B', 'preB':'B',
                    'preT':'T4','Mo':'Other','B1b':'B1ab'  , 'B1a':'B1ab','DC':'DC', 'GN':'GN', 'NK':'NK', 'MF':'MF',
                    'Tgd':'Tgd', 'CD4':'T4', 'MLP':'Other','CD8':'T8',"T8":"T8","B1ab":"B1ab","Treg":"Treg","Gn":"GN",
                    "T4":"T4","DN":"Other","EO":"Other","Eo":"Other","ILC1":"Other","ILC2":"Other","ILC3":"Other",
                    "BA":"Other","MEChi":"Other","MC":"Other","CCD11b-":"Other"}


type_map = {'ccd11b': 'other', 'b': 'b', 'cd19': 'b', 'sc': 'other', 'fi': 'other', 'frc': 'other', 'bec': 'other', 'lec': 'other',
            'ep': 'other', 'st': 'other', 't': 't4', 'nkt': 'nkt', 'prob': 'b', 'preb': 'b', 'pret': 't4', 'mo': 'other', 'b1b': 'b1ab',
            'b1a': 'b1ab', 'dc': 'dc', 'gn': 'gn', 'nk': 'nk', 'mf': 'mf', 'tgd': 'tgd', 'cd4': 't4', 'mlp': 'other', 'cd8': 't8',
            't8': 't8', 'b1ab': 'b1ab', 'treg': 'treg', 't4': 't4', 'dn': 'other', 'eo': 'other', 'ilc1': 'other', 'ilc2': 'other',
            'ilc3': 'other', 'ba': 'other', 'mechi': 'other', 'mc': 'other', 'ccd11b-': 'other','b-cells':'b','nucleated':'other','lt-hsc':'other',
            'monocytes':'other','cd4+':'t4','cd8+':'t8','granulocytes':'gn'}

type_map_updated = {'ccd11b': 'cd4', 'b': 'b', 'cd19': 'b', 'sc': 'hsc', 'fi': 'stromal', 'frc': 'stromal', 'bec': 'stromal', 'lec': 'stromal',
            'ep': 'stromal', 'st': 'stromal', 't': 't4', 'nkt': 'nkt', 'prob': 'b', 'preb': 'b', 'pret': 't4', 'mo': 'mo', 'b1b': 'b1ab',
            'b1a': 'b1ab', 'dc': 'dc', 'gn': 'gn', 'nk': 'nk', 'mf': 'mf', 'tgd': 'tgd', 'cd4': 't4', 'mlp': 'other', 'cd8': 't8',
            't8': 't8', 'b1ab': 'b1ab', 'treg': 'treg', 't4': 't4', 'dn': 'stromal', 'eo': 'eo', 'ilc1': 'ilc', 'ilc2': 'ilc',
            'ilc3': 'ilc', 'ba': 'other', 'mechi': 'ep', 'mc': 'mc', 'ccd11b-': 't4', 'b-cells':'b','nucleated':'other','lt-hsc':'hsc',
            'monocytes':'mo','cd4+':'t4','cd8+':'t8','granulocytes':'gn', 'iap':'other', 'mmp4':'other','mmp3':'other', 'b1b':'other',
            'sthsc':'hsc', 'lthsc':'hsc','bec':'other', 'frc':'other', 'l1210':'other'}

#{'ProE', 'LMPP', 'PreB', 'CFUE', 'GM', 'GMP', 'ProB', 'CD4', 'LTHSC', 'PreCFUE', 'NKmature', 'STHSC', 'MkE', 'ETP', 'CLP', 'MkP', 'IgM+SP'}
'''
'''
labels="Testing/y_combo"
label1="Testing/labels_for_GSE15907_series_matrix"
label2="Testing/labels_for_GSE37448_series_matrix"
y2 = pickle.load(open(label1, "rb"))
y1 = pickle.load(open(label2, "rb"))
y=y1.tolist()+y2.tolist()
y = np.array([x.lower() for x in y])
print(y)
pickle.dump(y, open(labels, "wb"))
'''

#Check that X_combo and y_combo not messed up
#y = pickle.load(open("Testing/y_combo", "rb"))
#X = pickle.load(open("Testing/X_unscaled_combo", "rb"))
#print("y: ",y)
#print("X: ",X)


#get_series_data('data/GSE37448_series_matrix.txt',26,65,24987,offset=1)
#get_series_data("data/fpkm_66.txt",1,2,68879,offset=0)
'''
train_data = "Testing/X_unscaled_combo"
train_labels="Testing/y_combo"
test_data = "Testing/input_unscaled_data_for_fpkm_66"
test_labels = "Testing/labels_for_fpkm_66" 
filename = 'X_unscaled_combo'
included_affy_file = "Testing/indices_of_"+str(1421)+"_included_features_"+filename
KNN_sort_filtered(train_data,train_labels,test_data,test_labels,included_affy_file,platform="xloc",k=3)
'''
'''
ref_file = "Testing/X_unscaled_combo_good_rows_1048"
query_file = "Testing/input_unscaled_data_for_fpkm_66_good_rows_1048"
match_dist(ref_file,query_file)

train_data = "Testing/X_unscaled_combo_good_rows_1048"
train_labels="Testing/y_combo"
test_data = "Testing/input_unscaled_data_for_fpkm_66_good_rows_1048dist_matched_train"
test_labels = "Testing/labels_for_fpkm_66"   #"Testing/y_combo" #
filename = 'X_unscaled_combo'
included_affy_file = "Testing/indices_of_"+str(1421)+"_included_features_"+filename
KNN_sort_filtered(train_data,train_labels,test_data,test_labels,included_affy_file,k=3,platform="affy")
'''
'''
column_scale("Testing/X_unscaled_combo",scale="mean")
column_scale("Testing/input_unscaled_data_for_fpkm_66",scale="mean")
train_data = "Testing/X_unscaled_combo_colScaled_mean"
train_labels="Testing/y_combo"
test_data = "Testing/input_unscaled_data_for_fpkm_66_colScaled_mean"
test_labels = "Testing/labels_for_fpkm_66"
filename = 'X_unscaled_combo'
included_affy_file = "Testing/indices_of_"+str(1421)+"_included_features_"+filename
KNN_sort_filtered(train_data,train_labels,test_data,test_labels,included_affy_file,k=3,platform="xloc")
'''

#get_series_data("data/MF.fixed.RNAseq.txt",1,2,17536,offset=0)
#"Testing/symbol_to_AffyID_map" #gene to affy map in this case


#gene_symbol_to_affy("MF.fixed.RNAseq")

'''
train_data = "Testing/X_unscaled_combo" #filename+"_shuffle_train"#"Testing/X_unscaled_combo"+"_shuffle_train" #
train_labels="Testing/y_combo"#+"_shuffle_train"
test_data = "Testing/input_unscaled_data_for_MF.fixed.RNAseq" #filename+"_shuffle_test" #"Testing/X_unscaled_combo"+"_shuffle_test" #
test_labels = "Testing/labels_for_MF.fixed.RNAseq" #"Testing/y_combo"+"_shuffle_test" #
filename = 'X_unscaled_combo'
included_affy_file = "Testing/indices_of_"+str(1421)+"_included_features_"+filename
KNN_sort_filtered(train_data,train_labels,test_data,test_labels,included_affy_file,k=3,platform="Testing/MF.fixed.RNAseq_to_affy",genes_list_file="Testing/gene_ids_MF.fixed.RNAseq")
'''
'''
ref_file = "Testing/X_unscaled_combo_good_rows_1159"
query_file = "Testing/input_unscaled_data_for_MF.fixed.RNAseq_good_rows_1159"
match_dist(ref_file,query_file)

train_data = "Testing/X_unscaled_combo_good_rows_1159"
train_labels="Testing/y_combo"
test_data = "Testing/input_unscaled_data_for_MF.fixed.RNAseq_good_rows_1159dist_matched_train"
test_labels = "Testing/labels_for_MF.fixed.RNAseq" 
filename = 'X_unscaled_combo'
included_affy_file = "Testing/indices_of_"+str(1421)+"_included_features_"+filename
KNN_sort_filtered(train_data,train_labels,test_data,test_labels,included_affy_file,k=3,platform="affy")
'''
'''
column_scale("Testing/X_unscaled_combo",scale="max")
column_scale("Testing/input_unscaled_data_for_MF.fixed.RNAseq",scale="max")
train_data = "Testing/X_unscaled_combo_colScaled_max"
train_labels="Testing/y_combo"
test_data = "Testing/input_unscaled_data_for_MF.fixed.RNAseq_colScaled_max"
test_labels = "Testing/labels_for_MF.fixed.RNAseq"
filename = 'X_unscaled_combo'
included_affy_file = "Testing/indices_of_"+str(1421)+"_included_features_"+filename
KNN_sort_filtered(train_data,train_labels,test_data,test_labels,included_affy_file,k=3,platform="Testing/MF.fixed.RNAseq_to_affy",genes_list_file="Testing/gene_ids_MF.fixed.RNAseq")
'''

#generate_plots("Testing/input_unscaled_data_for_GSE37448_series_matrix","Testing/input_unscaled_data_for_GSE15907_series_matrix")
'''
ref_file = "Testing/X_affy_good_rows_1421"
query_file = "Testing/X_affy_good_rows_1421"
match_dist(ref_file,query_file)

shuffle("Testing/X_affy_good_rows_1421dist_matched","Testing/y_combo",600)
train_data = "Testing/X_affy_good_rows_1421dist_matched"+"_shuffle_train"
train_labels= "Testing/y_combo"+"_shuffle_train"
test_data = "Testing/X_affy_good_rows_1421dist_matched"+"_shuffle_test"
test_labels = "Testing/y_combo"+"_shuffle_test"
filename = 'X_unscaled_combo'
included_affy_file = "Testing/indices_of_"+str(1421)+"_included_features_"+filename
KNN_sort_filtered(train_data,train_labels,test_data,test_labels,included_affy_file,k=3,platform="affy")
'''

#gene_symbol_to_affy("GSE74596")

'''
train_data = "Testing/X_unscaled_combo"
train_labels="Testing/y_combo"
test_data = "Testing/input_unscaled_data_for_GSE74596" 
test_labels = "Testing/labels_for_GSE74596" 
filename = 'X_unscaled_combo'
included_affy_file = "Testing/indices_of_"+str(2497)+"_included_features_"+filename
KNN_sort_filtered(train_data,train_labels,test_data,test_labels,included_affy_file,k=3,platform="Testing/GSE74596_to_affy",genes_list_file="Testing/gene_ids_GSE74596",check_expression=True)
'''
'''
column_scale("Testing/X_unscaled_combo",scale="mean")
column_scale("Testing/input_unscaled_data_for_GSE74596",scale="mean")
train_data = "Testing/X_unscaled_combo_colScaled_mean"
train_labels="Testing/y_combo"
test_data = "Testing/input_unscaled_data_for_GSE74596_colScaled_mean"
test_labels = "Testing/labels_for_GSE74596"
filename = 'X_unscaled_combo'
included_affy_file = "Testing/indices_of_"+str(2497)+"_included_features_"+filename
KNN_sort_filtered(train_data,train_labels,test_data,test_labels,included_affy_file,k=3,platform="Testing/GSE74596_to_affy",genes_list_file="Testing/gene_ids_GSE74596",check_expression=True)

'''

'''
ref_file = "Testing/X_unscaled_combo_good_rows_558"
query_file = "Testing/input_unscaled_data_for_GSE74596_good_rows_558"
match_dist(ref_file,query_file)


train_data = "Testing/X_unscaled_combo_good_rows_558"
train_labels="Testing/y_combo"
test_data = "Testing/input_unscaled_data_for_GSE74596_good_rows_558dist_matched"
test_labels = "Testing/labels_for_GSE74596" 
filename = 'X_unscaled_combo'
included_affy_file = "Testing/indices_of_"+str(1421)+"_included_features_"+filename
KNN_sort_filtered(train_data,train_labels,test_data,test_labels,included_affy_file,k=3,platform="affy")
'''

#get_series_data("data/GSE74923.txt",1,2,23421,offset=0)
'''
gene_symbol_to_affy("GSE74923")
train_data = "Testing/X_unscaled_combo"
train_labels="Testing/y_combo"
test_data = "Testing/input_unscaled_data_for_GSE74923" 
test_labels = "Testing/labels_for_GSE74923" 
filename = 'X_unscaled_combo'
included_affy_file = "Testing/indices_of_"+str(2497)+"_included_features_"+filename
KNN_sort_filtered(train_data,train_labels,test_data,test_labels,included_affy_file,k=3,platform="Testing/GSE74923_to_affy",genes_list_file="Testing/gene_ids_GSE74923",check_expression=True)
'''
'''
column_scale("Testing/X_unscaled_combo",scale="mean")
column_scale("Testing/input_unscaled_data_for_GSE74923",scale="mean")
train_data = "Testing/X_unscaled_combo_colScaled_mean"
train_labels="Testing/y_combo"
test_data = "Testing/input_unscaled_data_for_GSE74923_colScaled_mean"
test_labels = "Testing/labels_for_GSE74923"
filename = 'X_unscaled_combo'
included_affy_file = "Testing/indices_of_"+str(2497)+"_included_features_"+filename
KNN_sort_filtered(train_data,train_labels,test_data,test_labels,included_affy_file,k=3,platform="Testing/GSE74923_to_affy",genes_list_file="Testing/gene_ids_GSE74923",check_expression=True)
'''
'''
ref_file = "Testing/X_unscaled_combo_good_rows_1133"
query_file = "Testing/input_unscaled_data_for_GSE74923_good_rows_1133"
match_dist(ref_file,query_file)


train_data = "Testing/X_unscaled_combo_good_rows_1133"
train_labels="Testing/y_combo"
test_data = "Testing/input_unscaled_data_for_GSE74923_good_rows_1133dist_matched"
test_labels = "Testing/labels_for_GSE74923" 
filename = 'X_unscaled_combo'
included_affy_file = "Testing/indices_of_"+str(1421)+"_included_features_"+filename
KNN_sort_filtered(train_data,train_labels,test_data,test_labels,included_affy_file,k=3,platform="affy")
'''
'''
gene_symbol_to_affy("GSE74923")
train_data = "Testing/X_unscaled_combo"
train_labels="Testing/y_combo"
test_data = "Testing/input_unscaled_data_for_GSE74923" 
test_labels = "Testing/labels_for_GSE74923" 
filename = 'X_unscaled_combo'
included_affy_file = "Testing/indices_of_"+str(1421)+"_included_features_"+filename
KNN_sort_filtered(train_data,train_labels,test_data,test_labels,included_affy_file,k=3,platform="Testing/GSE74923_to_affy",genes_list_file="Testing/gene_ids_GSE74923",check_expression=False,impute=True)
'''
#generate_plots("Testing/input_unscaled_data_for_GSE74596_good_rows_558","Testing/X_unscaled_combo_good_rows_558")


