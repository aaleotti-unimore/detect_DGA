#import dask
#import dask.dataframe as dd

import importlib

import json
import os

from sklearn.externals import joblib

from sklearn.cluster import KMeans

# TODO debug
def initialize_dask_scheduler():
    dask.set_options(get=dask.multiprocessing.get)
    pass

# TODO debug : ok
def is_numeric(key):
    return key.lstrip('-+').isdigit()
    pass

# TODO debug : OK
def ensure_dir(file_path):
    '''
    Controlla l'esistenza della directory di file_path,
    se non esiste la crea (se necessario crea diverse directory e sottodirectory del percorso)
    :param file_path: percorso del file
    :return: None
    '''
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# TODO debug
def validate_dir(dir):
    if dir[-1] != '/':
        return dir + '/'
    return dir

# TODO debug
def get_subsets(dataset,feature,threshold):
    subset_1=dataset.query(str(feature)+' <= '+str(threshold))
    subset_2=dataset.query(str(feature)+' > '+str(threshold))
    return subset_1,subset_2
    pass



def get_labels_kmeans(df,n_classes=2):
    # TODO debug
    clust=KMeans(n_clusters=n_classes)
    return clust.fit_predict(df)
    pass

#OK
def get_x_y(df,y_column):
    columns=list(df.columns)
    columns.remove(y_column)
    x=df[columns]
    y=df[y_column]
    return x,y
    pass

#TODO debug
def to_numpy_array(df, is_dask=True):
    '''
    Restituisce un array numpy contenente i record del dataframe df.
    :param df: DataFrame in input
    :param is_dask: Boolean. True type(df) = dask.DataFrame; False type(df) = pandas.DataFrame
    :return: numpy array.
    '''
    if is_dask:
        return df.compute().values
    else:
        return df.values
    pass

#TODO debug
def delete_column(df,column_name):
    # type: (object, object) -> object
    columns=list(df.columns)
    columns.remove(column_name)
    return df[columns]
    pass

#TODO debug
def extract_column(df,column_names):
    '''
    Estrae dal dataframe df una lista di colonne.
    :param df: Dataframe in input.
    :param column_names: Nome di colonna o lista di nomi di colonne.
    :return: Una tupla formata da un nuovo DataFrame ottenuto eliminando le colonne
            contenute in column_names da df e un dict formato da coppie
             chiave = nome_colonna, valore = colonna (Dask Series)
    '''

    df_columns = list(df.columns)
    extracted_columns = {}
    result_df = df

    for column_name in column_names:
        if df_columns.count(column_name) > 0:
            extracted_columns[column_name] = result_df[column_name]
            result_df = delete(result_df, column_name)
            df_columns = list(df.columns)
            pass
        pass
    return result_df, extracted_columns

# TODO debug
def map_threshold(column):
    return column.map(lambda x: 1 if x == 0 else 0, meta=('x',int))
    pass

#TODO debug : OK
def _get_records_number(df):
    '''
    Restituisce il numero di record di df
    :param df: DataFrame in input
    :return: int
    '''
    #return int(df.size.compute() / len(df.columns))
    return df.count().compute()[0]
    pass

# # TODO debug
# def balance(X, n_records_0, n_records_1, labels_column = 'normal'):
#     '''
#         :param X: dataset in input con colonna di label
#         :param n_records_0: n di record della classe 0 da restituire
#         :param n_records_1: n di record della classe 1 da restituire
#         :param labels_column: colonna delle label
#         :return: restituisce un unico subset contenente rispettivamente n_records_0 e n_records_1 di sample
#                 delle classi 0 e 1
#         '''
#     # class_0 = X.query('normal == 0')
#     # class_1 = X.query('normal == 1')
#     #
#     # class_0 = class_0.head(n_records_0, npartitions=-1, compute=False)
#     # class_1 = class_1.head(n_records_1, npartitions=-1, compute=False)
#
#     class_0,class_1=get_samples(X,n_records_0,n_records_1,labels_column)
#
#     result = class_0.append(class_1)
#     result = result.repartition(npartitions=10, force=True)
#     result = result.sample(1)
#     return result
#     pass

# #TODO debug : OK nel caso in cui n_records entrambi == None
# #NOTA: meglio passare come argomenti le percentuali
# #di sample delle due classi da campionare
# def get_samples(X,n_records_0=None,n_records_1=None,labels_column='normal'):
#     '''
#     :param X: dataset in input con colonna di label
#     :param n_records_0: n di record della classe 0 da restituire (None = tutti)
#     :param n_records_1: n di record della classe 1 da restituire (None = tutti)
#     :param labels_column: colonna delle label
#     :return: restituisce due subdataset contenenti rispettivamente n_records_0 e n_records_1 di sample
#             delle classi 0 e 1
#     '''
#     class_0 = X.query(labels_column+' == 0')
#     class_1 = X.query(labels_column+' == 1')
#
#     if n_records_0 is not None:
#         class_0 = class_0.head(n_records_0, npartitions=-1, compute=False)
#         pass
#     if n_records_1 is not None:
#         class_1 = class_1.head(n_records_1, npartitions=-1, compute=False)
#         pass
#
#     # class_0 = class_0.repartition(npartitions=1, force=True)
#     # class_1 = class_1.repartition(npartitions=1, force=True)
#
#     return class_0,class_1
#     pass

#TODO debug
def sample(df,sample_0=1.0,sample_1=1.0,labels_column='normal'):
    '''
    Campiona una percentuale sample_0 di record della classe 0 e
    una percentuale sample_1 di record della classe 1. Se non viene fornita la percentuale
    corrispondente a una classe vengono restituiti tutti i record corrispondenti contenuti in df.
    :param df: dataframe in input con colonna di label
    :param sample_0: percentuale di sample della classe 0 da campionare
    :param sample_1: percentuale di sample della classe 1 da campionare
    :param labels_column: nome della colonna del dataframe che rappresenta le label
    :return:
    '''

    class_0 = df.query(labels_column+' == 0')
    class_1 = df.query(labels_column+' == 1')


    if sample_0 != 1.0:
        class_0 = class_0.repartition(npartitions=1, force=True)
        class_0=class_0.sample(sample_0)
        pass
    if sample_1 != 1.0:
        class_1 = class_1.repartition(npartitions=1, force=True)
        class_1=class_1.sample(sample_1)
        pass

    return class_0,class_1
    pass

#TODO debug
def sample_n_classes(df, sampling_dict, keep_absent_labels=False, labels_column='normal'):
    '''
    Esegue sampling senza reinserimento sui record di df sulla base della rispettiva label.
    :param df: DataFrame in input.
    :param sampling_dict: Dizionario che associa ad ogni valore di interesse per la label
            la frazione di esempi da campionare.
    :param keep_absent_labels: Boolean, se True i record corrispondenti a valori della label
            che non compaiono tra le chiavi del sampling dict vengono tenuti tutti, altrimenti vengono
            scartati tutti.
    :param labels_column: Nome della colonna di df che contiene le label.
    :return: Dataframe campionato.
    '''

    pass
