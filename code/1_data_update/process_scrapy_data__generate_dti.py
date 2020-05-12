import os
os.chdir('/home/chujunyi/2_Program/1_code/2_drug_feature')

import pandas as pd
import numpy as np
np.set_printoptions(threshold = np.inf)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

import linecache,socket,os,re
import urllib
import urllib.request
from urllib.request import urlretrieve
from urllib.request import urlopen
import requests
from bs4 import BeautifulSoup
socket.setdefaulttimeout(10)

from tqdm import tqdm
from rdkit import Chem
from copy import deepcopy
from collections import Counter
from itertools import product

###
def kegg_retrieve_drug(drug_list, dataset = 'NR'):
    error_list = []
    for drug_id in tqdm(drug_list):
        try:
            urlretrieve('http://www.kegg.jp/dbget-bin/www_bget?-f+m+compound+' + drug_id,
                        r'/home/chujunyi/2_Program/0_data_sets/' + dataset + '/Drug/' + drug_id + '.mol')
            
            smiles = Chem.MolToSmiles(Chem.MolFromMolFile(r'/home/chujunyi/2_Program/0_data_sets/' + dataset + '/Drug/' + drug_id + '.mol'))
            with open(r'/home/chujunyi/2_Program/0_data_sets/' + dataset + '/Drug/' + drug_id + '.txt', 'w') as f:
                f.write(smiles)
                
        except:
            error_list.append(drug_id)
#             print(drug_id)
    return error_list

def kegg_retrieve_target(target_list, dataset = 'NR'):
    error_list = []
    for hsa_id in tqdm(target_list):
        try:
            url = 'https://www.kegg.jp/dbget-bin/www_bget?-f+-n+a+hsa:' + str(hsa_id)
            html = requests.get(url).content.decode()
            soup = BeautifulSoup(html, 'lxml')

            content = soup.select('div > pre')[0].text
            fasta = content[1:]
            raw_sequence = fasta[fasta.index('\n')+1:]
#             print(hsa_id, raw_sequence)
            with open(r'/home/chujunyi/2_Program/0_data_sets/' + dataset + '/Target/' + hsa_id + '.fa', 'w') as f:
                f.write(fasta)

            with open(r'/home/chujunyi/2_Program/0_data_sets/' + dataset + '/Target/' + hsa_id + '.txt', 'w') as f:
                f.write(raw_sequence)
        except:
#             print(hsa_id)
            error_list.append(hsa_id)
    return error_list

def drugbank_retrieve_drug(drug_list, dataset = 'NR'):
    error_list = []
    for drug_id in tqdm(drug_list):
        try: # https://www.drugbank.ca/structures/small_molecule_drugs/DB02601.smiles
            urlretrieve('https://www.drugbank.ca/structures/small_molecule_drugs/' + drug_id + '.smiles',
            '/home/chujunyi/2_Program/0_data_sets/' + dataset + '/Drug/' + drug_id + '.smiles')
        except:
            error_list.append(drug_id)
#             print(drug_id)
    return error_list

def scrapy_kegg_nosmiles_id(drug_list, dataset = 'NR'):
    error_list = []
    for drug_id in tqdm(drug_list):
        url = 'https://www.kegg.jp/dbget-bin/www_bget?dr:' + drug_id
        html = requests.get(url).content.decode()
        soup = BeautifulSoup(html,'lxml')

        drugbank_id = []
        database_href = soup.select('td > table > tr > td > a')
        for jtem in database_href:
            try:
                ljnk = jtem.get('href')
                if ljnk[:29] == 'https://www.drugbank.ca/drugs':
                    drugbank_id.append(jtem.get('href')[30:])
            except:
                continue  

        if drugbank_id == []: 
            error_list.append(drug_id)
#             print(drug_id)

        for id_ in drugbank_id:
            try:
                urlretrieve('https://www.drugbank.ca/structures/small_molecule_drugs/' + id_ + '.smiles',
                '/home/chujunyi/2_Program/0_data_sets/' + dataset + '/Drug/' + drug_id + '.smiles')
            except:
                error_list.append(drug_id)
#                 print(drug_id, id_)
    return error_list
####### load data
def load_dti_data(pwd, filename):
    dti = np.load(pwd + filename, allow_pickle = True)
    
    dtis_keggid = pd.DataFrame(dti['DTIs_keggDid'], columns = ['drug_id', 'hsa_id'])
    dtis_dbid = pd.DataFrame(dti['DTIs_DrugBankDBid'], columns = ['drug_id', 'hsa_id'])
    dti_data = pd.concat([dtis_keggid, dtis_dbid], axis = 0).reset_index(drop = True)
    print('dtis_keggid.shape = {}, dtis_dbid.shape = {}, dti_data.shape = {}'.format(dtis_keggid.shape, dtis_dbid.shape, dti_data.shape))
    
    drugs_keggid = list(set(dti['DTIs_keggDid'][:,0]))
    drugs_dbid = list(set(dti['DTIs_DrugBankDBid'][:,0]))
    targets_all = dti['Targets_in_all_DTIs']
    print('# Target = {}; # Drug = {}: KEGG = {}, DB = {}'.format(len(targets_all), len(drugs_keggid) + len(drugs_dbid), len(drugs_keggid), len(drugs_dbid)))
    return dti_data, dtis_keggid, dtis_dbid, drugs_keggid, drugs_dbid, targets_all

####### 删掉重复的smiles
def obtain_drug_id_smiles_df(drug_list, data_set = 'NR', save_ = False):
    drug_dict = {}
    for drug_id in tqdm(drug_list):
        
        try:
            with open('/home/chujunyi/2_Program/0_data_sets/' + data_set + '/Drug/' + drug_id + '.txt', 'r') as f:
                    smiles = f.read()
        except:
            with open('/home/chujunyi/2_Program/0_data_sets/' + data_set + '/Drug/' + drug_id + '.smiles', 'r') as f:
                    smiles = f.read()
        drug_dict[drug_id] = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    assert len(drug_dict) == len(drug_list)
    
    drug_id_smiles = pd.DataFrame.from_dict(drug_dict, orient='index').reset_index()
    drug_id_smiles.columns = ['drug_id', 'smiles']
    
    if save_:
        pwd = '/home/chujunyi/2_Program/0_data_sets/' + data_set + '/'
        np.save(pwd + data_set + '_updated_drug_smiles_dict.npy', drug_dict)
        drug_id_smiles.to_csv(pwd + data_set + '_updated_drug_id_smiles.csv', index = False)
    return drug_dict, drug_id_smiles

def find_repeated_smi_id(drug_id_smiles_df):
    repeat_smi_id, all_repeat_id = [], []

    for smi, count_ in dict(Counter(drug_id_smiles_df['smiles'])).items():
        if count_ > 1:
            id_ = list(drug_id_smiles_df[drug_id_smiles_df['smiles'] == smi]['drug_id'])
            assert len(id_) == count_
            repeat_smi_id.append(id_)
            all_repeat_id.extend(id_)
    print('# repeat_smi = {}, # repeat_id = {}'.format(len(repeat_smi_id), len(all_repeat_id)))
    return repeat_smi_id, all_repeat_id 

def obtain_replaced_repeated_dtis(dtis_dbid, dtis_keggid, repeat_smi_id, print_ = False):
    all_replaced_dtis = []
    for ids in repeat_smi_id:
        hsa_list = []
        for item in ids:
            if item[:2] == 'DB':
                hsa_list.extend(list(dtis_dbid[dtis_dbid['drug_id'] == item]['hsa_id']))
            else: # kegg D id
                hsa_list.extend(list(dtis_keggid[dtis_keggid['drug_id'] == item]['hsa_id']))
                
        hsa_list = list(set(hsa_list))
        replaced_dtis = list(product([ids[0]],hsa_list))
        all_replaced_dtis.extend(replaced_dtis)
        
        if print_: print(ids, hsa_list, '\n', replaced_dtis)
    print('# all_replaced_dtis = ', len(all_replaced_dtis))
    return all_replaced_dtis
###### 删掉没找到_重复的smiles的dti
def delete_no_smiles_dtis(dti_data, non_smiles_drugs_id_list):
    print('Original data: dti_data.shape = {}'.format(dti_data.shape))
    dti_data = dti_data[~dti_data['drug_id'].isin(non_smiles_drugs_id_list)]
    drug_list = list(set(dti_data['drug_id']))
    target_list = list(set(dti_data['hsa_id']))
    print('After delete: dti_data.shape = {}, # drugs = {}, # targets = {}'.format(dti_data.shape, len(drug_list), len(target_list)))
    return dti_data, drug_list, target_list

def delete_impute_repeated_dtis(dti_data, all_repeat_id, all_replaced_dtis, drug_id_smiles):
    print('Original dti_data.shape = ', dti_data.shape)
    dti_data = dti_data[~dti_data['drug_id'].isin(all_repeat_id)]
    print('After delete, dti_data.shape = ', dti_data.shape)
    
    replaced_dti_data = pd.DataFrame(all_replaced_dtis, columns = ['drug_id', 'hsa_id'])
    final_dti_data = pd.concat([dti_data, replaced_dti_data],axis = 0).drop_duplicates().reset_index(drop = True)
    assert final_dti_data.shape == (dti_data.shape[0]+ replaced_dti_data.shape[0], 2)

    final_dti_data['hsa_id'] = ['hsa' + item for item in final_dti_data['hsa_id']]
    final_dti_data_with_smiles = pd.merge(drug_id_smiles, final_dti_data, on = 'drug_id').drop_duplicates().reset_index(drop = True)
    assert final_dti_data_with_smiles.shape[0] == final_dti_data.shape[0]
    print('After impute, final dti_data with smiles shape = ', final_dti_data_with_smiles.shape)
    return final_dti_data_with_smiles

def generate_all_target_fasta_file(target_id_list, data_set = 'GPCR'):
    save_pwd = '/home/chujunyi/2_Program/0_data_sets/' + data_set + '/'
    save_filename = data_set + '_updated_all_target.fasta'
    with open(save_pwd + save_filename, 'w') as f:
        
        for target_id in target_id_list:
            each_pwd = '/home/chujunyi/2_Program/0_data_sets/' +  data_set + '/Target/'
            each_filename = str(target_id) + '.fa'
            with open(each_pwd + each_filename, 'r') as ff:
                fasta = ff.read()
                
            f.write(fasta)
    return