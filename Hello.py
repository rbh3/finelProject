import pdb
from flask import Flask, redirect, url_for, request, render_template
import Classifier as cs
import Magic
import numpy as np
import pickle
#IF ERROR - try importing command
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World \n Caleb Noble'

@app.route('/hello')
def hello_world_2():
   return render_template('hello.html') #'hello' #'<html><body><h1>'Hello World'</h1></body></html>'

@app.route('/hello/<name>')
def hello_name(name):
   return render_template('hello_name.html',name=name) #passes name into html file

#app.add_url_rule('/', 'hello2', hello_world)

@app.route('/blog/<int:postID>')
def show_blog(postID):
   return 'Blog Number %d' % postID

@app.route('/rev/<float:revNo>')
def revision(revNo):
   return 'Revision Number %f' % revNo

#USING LOGIC TO REDIRECT WEBPAGE
@app.route('/admin')
def hello_admin():
   return 'Hello Admin'

@app.route('/guest/<guest>')
def hello_guest(guest):
   return 'Hello %s as Guest' % guest

@app.route('/user/<name>')
def hello_user(name):
   if name =='admin':
      return redirect(url_for('hello_admin')) #hello_admin()#
   else:
      return redirect(url_for('hello_guest',guest = name))#hello_guest(name)#

#Getting and posting INPUTS!
@app.route('/success/<name>')
def success(name):
   return 'welcome %s' % name

@app.route('/login',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      user = request.form['nm']
      return redirect(url_for('success',name = user))
   else:
      user = request.args.get('nm')
      return redirect(url_for('success',name = user))

#FORM INPUTS ARE DICTIONARIES:
@app.route('/student')
def student():
   return render_template('student.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      return render_template("result.html",result = result)

#FILE UPLOADS
@app.route('/upload')
def upload_file():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def uploaded_file():
    if request.method == 'POST':
        f = request.files['file']
        start_row = int(request.form['start_row'])
        end_row = int(request.form['end_row'])
        #print(f.filename)
        f.save("data/"+f.filename)

        X,gene_ids = cs.get_series_data("data/"+f.filename,start_row,end_row)



        '''
        train_data = pickle.load(open('Testing/X_unscaled_combo', "rb")).astype(np.float)
        train_labels = pickle.load(open('Testing/y_combo', "rb"))
        filename = 'X_unscaled_combo'
        included_affy_file = "Testing/indices_of_"+str(1421)+"_included_features_"+filename   
        '''

        if request.form['platform']=="single_rna":
            #return "Single Cell RNA to be implemented."
            #then we need to run magic and then use colScaled by max KNN
            print("Running KNN classifier using single cell RNA data")
            if True: #request.form['convert']=='no':
                g = request.files['conversion_file']
                g.save("data/"+g.filename)
                conv_start = int(request.form['conv_start_row'])
                conv_end = int(request.form['conv_end_row'])
                id_col = int(request.form['id_col'])
                symbol_col = int(request.form['symbol_col'])
                id_to_sym = cs.gene_code_map("data/"+g.filename, conv_start, conv_end, symbol_col, id_col)

            else:
                id_to_sym = {}
                for gene in gene_ids:
                    id_to_sym[gene.lower()] = gene.lower()

            train_data = pickle.load(open('Testing/sc_X_combo', "rb")).astype(np.float)
            train_labels = pickle.load(open('Testing/sc_y_combo', "rb"))
            included_genes_file = "Testing/sc_X_comboindices_of_1388_included_features"
            train_genes_file = "Testing/sc_gene_list_combo"

            X, gene_ids = Magic.magic_process("data/"+f.filename,start_row)
            id_to_affy = cs.gene_symbol_to_affy(gene_ids)
            train_data = cs.column_scale(train_data,scale="max")
            test_data = cs.column_scale(X,scale="max")

            predicted,confidence =  cs.KNN_sort_filtered(train_data,train_labels,test_data,included_genes_file,train_genes_file,k=5,platform=id_to_sym,genes_list=gene_ids)

        else:
            #use dist matched KNN
            print("Running KNN classifier using Microarray data")
            if request.form['convert']=='no': #not labeled by gene symbol
                g = request.files['conversion_file']
                g.save("data/"+g.filename)
                conv_start = int(request.form['conv_start_row'])
                conv_end = int(request.form['conv_end_row'])
                id_col = int(request.form['id_col'])
                symbol_col = int(request.form['symbol_col'])
                id_symbol_map = cs.gene_code_map("data/"+g.filename, conv_start, conv_end, symbol_col, id_col)
                id_to_affy = cs.gene_to_Affy_map(id_symbol_map,gene_ids,X)
            else:
                id_to_affy = cs.gene_symbol_to_affy(gene_ids)

            train_data = pickle.load(open('Testing/X_unscaled_combo', "rb")).astype(np.float)
            train_labels = pickle.load(open('Testing/y_combo', "rb"))
            filename = 'X_unscaled_combo'
            included_affy_file = "Testing/indices_of_"+str(1421)+"_included_features_"+filename
            train_genes_file = "Testing/gene_ids_GSE15907_series_matrix"

            X_ref,X_query = cs.reduce_to_good_rows(train_data,X,included_affy_file,id_to_affy,gene_ids)
            X_query_dist_matched = cs.match_dist(X_ref,X_query)
            predicted,confidence = cs.KNN_sort_filtered(X_ref,train_labels,X_query_dist_matched,included_affy_file,train_genes_file,k=5,platform="affy")



        #d,n = X.shape

        output = ""
        output_dic = {}
        confidence_dic = {}
        for i in range(len(predicted)):
            output+= (str(predicted[i])+" ")
            output_dic[i]=predicted[i]
            confidence_dic[i] = confidence[i]
        return render_template('output_list_table.html',result=output_dic,result2=confidence_dic)#"Predictions: "+output

'''
if __name__ == '__main__':
    app.debug = True #allows for changes to be enacted without rerunning server
    app.run()
''
#f = request.files['file']
start_row = 2#request.form['start_row']
end_row = 68879#request.form['end_row']
#print(f.filename)
#f.save("data/"+f.filename)
X,gene_ids = cs.get_series_data("data/"+"genes_fpkm.txt",start_row,end_row)
train_data = pickle.load(open('Testing/X_unscaled_combo', "rb")).astype(np.float)
train_labels = pickle.load(open('Testing/y_combo', "rb"))
filename = 'X_unscaled_combo'
included_affy_file = "Testing/indices_of_"+str(1421)+"_included_features_"+filename        
X_ref,X_query = cs.reduce_to_good_rows(train_data,X,included_affy_file,platform="xloc")
X_query_dist_matched = cs.match_dist(X_ref,X_query)
predicted = cs.KNN_sort_filtered(X_ref,train_labels,X_query_dist_matched,included_affy_file,k=3,platform="affy")
print(predicted)
''

def test():
    if True:
        # = request.files['file']
        start_row = 2#int(request.form['start_row'])
        end_row = 68879#int(request.form['end_row'])
        #print(f.filename)
        #f.save("data/"+f.filename)

        X,gene_ids = cs.get_series_data("data/"+"genes_fpkm.txt",start_row,end_row)

        if True: #request.form['convert']=='no': #not labeled by gene symbol
            #g = request.files['conversion_file']
            #g.save("data/"+g.filename)
            conv_start = 2#int(request.form['conv_start_row'])
            conv_end = 68879#int(request.form['conv_end_row'])
            id_col = 1#int(request.form['id_col'])
            symbol_col = 5#int(request.form['symbol_col'])
            id_symbol_map = cs.gene_code_map("data/"+"genes_attr.txt", conv_start, conv_end, symbol_col, id_col)
            id_to_affy = cs.gene_to_Affy_map(id_symbol_map,gene_ids,X)
        else:
            id_to_affy = gene_symbol_to_affy(gene_ids)
        
        
        train_data = pickle.load(open('Testing/X_unscaled_combo', "rb")).astype(np.float)
        train_labels = pickle.load(open('Testing/y_combo', "rb"))
        filename = 'X_unscaled_combo'
        included_affy_file = "Testing/indices_of_"+str(1421)+"_included_features_"+filename        
        X_ref,X_query = cs.reduce_to_good_rows(train_data,X,included_affy_file,id_to_affy,gene_ids)
        X_query_dist_matched = cs.match_dist(X_ref,X_query)
        predicted = cs.KNN_sort_filtered(X_ref,train_labels,X_query_dist_matched,included_affy_file,k=3,platform="affy")
        #d,n = X.shape

        output = ""
        for elt in predicted:
            output+= (str(elt)+" ")
        print(output)

test()
'''
def test():
    if True:
        #f = open("data/GSE74596.txt",'r') #request.files['file']
        filename = "data/GSE74596.txt"
        start_row = 1 #int(request.form['start_row'])
        end_row = 23337 #int(request.form['end_row'])
        #print(f.filename)
        #f.save("data/"+f.filename)

        X,gene_ids = cs.get_series_data(filename,start_row,end_row)
        #print("type of gene list: ",type(gene_ids))
        #print("gene ids: ",gene_ids[0:5])

        


        '''
        train_data = pickle.load(open('Testing/X_unscaled_combo', "rb")).astype(np.float)
        train_labels = pickle.load(open('Testing/y_combo', "rb"))
        filename = 'X_unscaled_combo'
        included_affy_file = "Testing/indices_of_"+str(1421)+"_included_features_"+filename   
        '''
        
        if True: #request.form['platform']=="single_rna":
            print("Running correlation prediction using Microarray data")
            if False: #request.form['convert']=='no': #not labeled by gene symbol, so need to convert using second file
                g = request.files['conversion_file']
                g.save("data/"+g.filename)
                conv_start = int(request.form['conv_start_row'])
                conv_end = int(request.form['conv_end_row'])
                id_col = int(request.form['id_col'])
                symbol_col = int(request.form['symbol_col'])
                id_symbol_map = cs.gene_code_map("data/"+g.filename, conv_start, conv_end, symbol_col, id_col)
                id_to_affy = cs.gene_to_Affy_map(id_symbol_map,gene_ids,X)
            else:
                id_to_affy = cs.gene_symbol_to_affy(gene_ids)

            X, gene_ids = Magic.magic_process(filename,start_row)
            X = X.astype(np.float)
            test_data = X
            train_data = pickle.load(open('Testing/X_unscaled_combo', "rb")).astype(np.float)
            train_labels = pickle.load(open('Testing/y_combo', "rb"))
            included_genes_file = "Testing/indices_of_"+str(1421)+"_included_features_X_unscaled_combo"
            train_genes = pickle.load(open("Testing/gene_ids_GSE15907_series_matrix", "rb"))
            train_genes_file = "Testing/gene_ids_GSE15907_series_matrix"
            id_to_affy = cs.gene_symbol_to_affy(gene_ids)

            train_data = cs.column_scale(train_data,scale="median")
            test_data = cs.column_scale(X,scale="median")
            X_ref,X_query = cs.reduce_to_good_rows(train_data,test_data,included_genes_file,train_genes,id_to_affy,gene_ids)
            X_query_dist_matched = cs.match_dist(X_ref,X_query)
            cs.plot_two_genes(X_ref,train_labels,X_query_dist_matched)#_dist_matched)
            predicted,confidence = cs.KNN_sort_filtered(X_ref,train_labels,X_query_dist_matched,included_genes_file,train_genes_file,k=5,platform="affy")

            #predicted,confidence = cs.KNN_sort_filtered(X_ref,train_labels,X_query_dist_matched,included_genes_file,train_genes_file,k=5,platform="affy")
            
            #predicted,confidence = cs.covariance_predict(X_ref,train_labels,X_query)
            '''



            
            #then we need to run magic and then use colScaled by max KNN
            print("Running KNN classifier using single cell RNA data")
            train_data = pickle.load(open('Testing/sc_X_combo', "rb")).astype(np.float)
            train_labels = pickle.load(open('Testing/sc_y_combo', "rb"))

            
            included_genes_file = "Testing/sc_X_comboindices_of_1388_included_features"
            train_genes_file = "Testing/sc_gene_list_combo"
            train_genes = pickle.load(open('Testing/sc_gene_list_combo', "rb"))
            #X, gene_ids = Magic.magic_process(filename,start_row)
            #print(X)
            if False: #request.form['convert']=='no':
                g_filename = "ENSMUSG_conversion_table.txt"
                conv_start6 = 7 #int(request.form['conv_start_row'])
                conv_end = 17925 #int(request.form['conv_end_row'])
                id_col = 1 #int(request.form['id_col'])
                symbol_col = 2 #int(request.form['symbol_col'])
                id_to_sym = cs.gene_code_map("data/"+g_filename, conv_start, conv_end, symbol_col, id_col)
            
            else:
                id_to_sym = {}
                for gene in gene_ids:
                    id_to_sym[gene.lower()] = gene.lower()

            
            #train_data = train_data#/np.max(train_data) #cs.column_scale(train_data,scale="median")
            test_data = X# /np.max(X) #cs.column_scale(X,scale="median")
            
            
            #print("maxes: ",np.max(train_data),np.max(X))
            print("gene ids: ",gene_ids[0:5])
            #predicted,confidence,X_ref,X_query =  cs.KNN_sort_filtered(train_data,train_labels,test_data,included_genes_file,train_genes_file,k=5,platform=id_to_sym,genes_list=gene_ids)


            train_data = pickle.load(open('Testing/X_unscaled_combo', "rb")).astype(np.float)
            train_labels = pickle.load(open('Testing/y_combo', "rb"))
            included_genes_file = "Testing/indices_of_"+str(1421)+"_included_features_X_unscaled_combo"
            train_genes = pickle.load(open("Testing/gene_ids_GSE15907_series_matrix", "rb"))
            id_to_affy = cs.gene_symbol_to_affy(gene_ids)
            
            X_ref,X_query = cs.reduce_to_good_rows(train_data,test_data,included_genes_file,train_genes,id_to_affy,gene_ids)
            predicted,confidence = cs.covariance_predict(X_ref,train_labels,X_query)
            #X_ref = pickle.load(open('Testing/X_train_good_rows', "rb"))
            #X_query = pickle.load(open('Testing/X_test_good_rows', "rb"))
            
            #
            '''
            '''
            X_query_dist_matched = cs.match_dist(X_ref,X_query)
            predicted,confidence = cs.KNN_sort_filtered(X_ref,train_labels,X_query_dist_matched,included_genes_file,train_genes_file,k=5,platform="affy")
            '''
        else:
            if False: #request.form['convert']=='no': #not labeled by gene symbol
                #g = request.files['conversion_file']
                #g.save("data/"+g.filename)
                g_filename = "ENSMUSG_conversion_table.txt"
                conv_start = 7 #int(request.form['conv_start_row'])
                conv_end = 17925 #int(request.form['conv_end_row'])
                id_col = 1 #int(request.form['id_col'])
                symbol_col = 2 #int(request.form['symbol_col'])
                id_symbol_map = cs.gene_code_map("data/"+g_filename, conv_start, conv_end, symbol_col, id_col)
                id_to_affy = cs.gene_to_Affy_map(id_symbol_map,gene_ids,X)
            else:
                id_to_affy = cs.gene_symbol_to_affy(gene_ids)
            
            #use dist matched KNN
            print("Running KNN classifier using Microarray data")
            train_data = pickle.load(open('Testing/X_unscaled_combo', "rb")).astype(np.float)
            train_labels = pickle.load(open('Testing/y_combo', "rb"))
            filename = 'X_unscaled_combo'
            included_affy_file = "Testing/indices_of_"+str(1421)+"_included_features_"+filename
            train_genes_file = "Testing/gene_ids_GSE15907_series_matrix"
            X_ref,X_query = cs.reduce_to_good_rows(train_data,X,included_affy_file,id_to_affy,gene_ids)
            X_query_dist_matched = cs.match_dist(X_ref,X_query)
            predicted,confidence = cs.KNN_sort_filtered(X_ref,train_labels,X_query_dist_matched,included_affy_file,train_genes_file,k=5,platform="affy")
            
             
        
        #d,n = X.shape
        print(predicted)
        output = ""
        output_dic = {}
        confidence_dic = {}
        correct = 0
        for i in range(len(predicted)):
            if predicted[i]=='nkt':
                correct+=1
            output+= (str(predicted[i])+" ")
            output_dic[i]=predicted[i]
            confidence_dic[i] = confidence[i]
        print(correct/len(predicted))

test()#check

