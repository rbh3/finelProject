import logging
import pickle

import numpy as np
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS

import Classifier as cs
import Magic

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


# File upload page
@app.route('/', methods=["GET"])
def upload_file():
    return render_template('upload.html')


# Classification process and output page
@app.route('/uploader', methods=['GET', 'POST'])
def uploaded_file():
    if request.method == 'POST':
        f = request.files['file']
        start_row = int(request.form['start_row'])
        end_row = int(request.form['end_row'])
        isLabeled = request.form['isLabeled']
        isTitled = request.form['isTitled']
        algo = request.form['clfMethod']

        f.save("data/" + f.filename)

        try:
            X, gene_ids, labels, titles = cs.get_series_data("data/" + f.filename, start_row, end_row, isLabeled, isTitled)
        except Exception:
            return jsonify({'errMsg': 'The input file is empty'}), 416

        if request.form['platform'] == "single rna":
            # Code to run magic and then use distribution matched KNN

            print("Running classifier using single cell RNA data")
            if request.form['convert'] == 'no':  # not labeled by gene symbol, so need to convert using second file
                g = request.files['conversion_file']
                g.save("data/" + g.filename)
                conv_start = int(request.form['conv_start_row'])
                conv_end = int(request.form['conv_end_row'])
                id_col = int(request.form['id_col'])
                symbol_col = int(request.form['symbol_col'])
                try:
                    id_to_sym = cs.gene_code_map("data/" + g.filename, conv_start, conv_end, symbol_col, id_col)
                    if not id_to_sym:
                         return jsonify({'errMsg': 'Mapping file not in the correct format , check that the file isn\'t empty'}), 416
                except SyntaxError as exception:
                    return jsonify({'errMsg': 'Mapping file not in the correct format'}), 416
            else:
                id_to_sym = {}
                for gene in gene_ids:
                    id_to_sym[gene.lower()] = gene.lower()

            train_data = pickle.load(open('Testing/X_unscaled_combo', "rb")).astype(np.float)
            train_labels = pickle.load(open('Testing/y_combo', "rb"))
            train_labels = [element.lower().rstrip() for element in train_labels] ; train_labels
            included_genes_file = "Testing/indices_of_" + str(1421) + "_included_features_X_unscaled_combo"
            train_genes = pickle.load(open("Testing/gene_ids_GSE15907_series_matrix", "rb"))
            id_to_affy = cs.gene_symbol_to_affy(gene_ids)
            try:
                test_data, gene_ids = Magic.magic_process("data/" + f.filename, start_row)
                X_ref, X_query = cs.reduce_to_good_rows(train_data, test_data, included_genes_file, train_genes, id_to_affy,
                                                    gene_ids)
            except SyntaxError as exception:
                return jsonify({'errMsg': 'The input file does not contains cells'}), 416
            except EOFError as exception:
                return jsonify({'errMsg': 'File not on correct format'}), 416
            except IndexError as exception:
                return jsonify({'There is a problem in translating your genes, maybe you need conversion file'}), 416
            # try:
            X_query_dist_matched = cs.match_dist(X_ref, X_query)
            if X_query_dist_matched == 'File not on correct format':
                return jsonify({'errMsg': 'Mapping file is not on correct format'}), 416
            train_genes_file = "Testing/gene_ids_GSE15907_series_matrix"
            predicted, confidence = cs.classification(X_ref, train_labels, X_query_dist_matched, included_genes_file,
                                                         train_genes_file, algo, k=5, platform="affy")

        else:
            # Code to use dist matched KNN to classify
            print("Running classifier using Microarray data or BULK")
            if request.form['convert'] == 'no':  # not labeled by gene symbol, so need to convert using second file
                g = request.files['conversion_file']
                g.save("data/" + g.filename)
                conv_start = int(request.form['conv_start_row'])
                conv_end = int(request.form['conv_end_row'])
                id_col = int(request.form['id_col'])
                symbol_col = int(request.form['symbol_col'])
                try:
                    id_symbol_map = cs.gene_code_map("data/" + g.filename, conv_start, conv_end, symbol_col, id_col)
                    if not id_symbol_map:
                        return jsonify({'errMsg': 'Mapping file not in the correct format , check that the file isn\'t empty'}), 416
                except SyntaxError as exception:
                    return jsonify({'errMsg': 'Mapping file not in the correct format'}), 416

                id_to_affy = cs.gene_to_Affy_map(id_symbol_map, gene_ids, X)
            else:
                id_to_affy = cs.gene_symbol_to_affy(gene_ids)

            train_data = pickle.load(open('Testing/X_unscaled_combo', "rb")).astype(np.float)
            train_labels = pickle.load(open('Testing/y_combo', "rb"))
            train_labels = [element.lower().rstrip() for element in train_labels] ; train_labels
            filename = 'X_unscaled_combo'
            included_affy_file = "Testing/indices_of_" + str(1421) + "_included_features_" + filename
            train_genes_file = "Testing/gene_ids_GSE15907_series_matrix"

            Affy_genes = pickle.load(open("Testing/gene_ids_GSE15907_series_matrix", "rb"))
            try:
                X_ref, X_query = cs.reduce_to_good_rows(train_data, X, included_affy_file, Affy_genes, id_to_affy,
                                                        gene_ids)
            except Exception:
                return jsonify({'errMsg': 'The input file does not contains cells'}), 416
            X_query_dist_matched = cs.match_dist(X_ref,X_query)
            if X_query_dist_matched == 'File not on correct format':
                return jsonify({'errMsg': 'Mapping file is not on correct format'}), 416
            predicted,confidence = cs.classification(X_ref, train_labels, X_query_dist_matched, included_affy_file, train_genes_file, algo, k=5, platform="affy")

        output_dic = {}
        confidence_dic = {}
        for i in range(len(predicted)):
            output_dic[i] = predicted[i]
            confidence_dic[i] = confidence[i]

        precision = 0
        if len(labels) > 0:
            del labels[0]
            print('real labels: ', labels)
            for i in range(len(labels)):
                if labels[i] == output_dic[i]:
                    precision += 1
            precision = precision / len(labels)
            print('labels got: ', output_dic)
            print('Precision: ', precision * 100, '%')

        return jsonify(
            {'output': output_dic, 'confidence': confidence_dic, 'CellsNo': len(output_dic), 'actual': labels,
             'Titles': titles, 'Precision': precision})


if __name__ == '__main__':
    handler = logging.FileHandler('/var/log/wsgi/wsgi.log')
    handler.setLevel(logging.ERROR)
    app.debug = True  # allows for changes to be enacted without rerunning server
    app.logger.addHandler(handler)
    app.run(host="0.0.0.0")
