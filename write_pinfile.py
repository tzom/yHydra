import pandas as pd
import sys,os 
os.environ['YHYDRA_CONFIG'] = sys.argv[1]
from load_config import CONFIG
import numpy as np
from tqdm import tqdm

OUTPUT_DIR = CONFIG['RESULTS_DIR']
SAVE_DB_AS_JSON = True#CONFIG['SAVE_DB_AS_JSON']

OUTPUT_TYPE = "PIN" # "PIN" or "PEPREC"

with pd.HDFStore(os.path.join(OUTPUT_DIR,'search_results_scored.h5')) as store:
    raw_files = store.keys()
    print(raw_files)
    for key in tqdm(raw_files):
        PIN_colnames_features = ['rank','best_score','best_distance','pepmass','delta_mass', 'peptide_length']
        print('writing PIN-file for %s'%key)
        search_results = store[key] 
        

        if SAVE_DB_AS_JSON:
            import json
            with open(os.path.join(OUTPUT_DIR+'/forward/db','db.json')) as fp:
                ncbi_peptide_protein = json.load(fp)
            #search_results['accession'] = list(map(lambda x: ncbi_peptide_protein[x],search_results.best_peptide))

        print(search_results.columns)

        #search_results = search_results[search_results.best_score <= 0.0]

        #search_results = search_results[search_results["raw_file"]=="20200110_QE1_DDA_1H25_T10_E9_R3"]
        
        #search_results['index'] = range(1,len(search_results)+1)

        #rename_yhydra2pin_dict = {'best_is_decoy':


        search_results['rank'] = 1
        search_results['peptide_length'] = search_results['best_peptide'].apply(lambda x: len(x))

        #search_results['charge'] = search_results['charge']

        charge_one_hot_df = pd.get_dummies(search_results['charge'],prefix='charge',prefix_sep="")
        charge_columns = charge_one_hot_df.columns.tolist()
        search_results[charge_columns] = charge_one_hot_df

        PIN_colnames_features = PIN_colnames_features + charge_columns

        pin_df = pd.DataFrame({})

        #### BUILD SpecId column

        #SpecId_blueprint = "{0}.{1}.{1}.{2}_{3}" # 0:raw_file, 1:id, 2:charge, 3:rank
        SpecId_blueprint = "{0}_{1}_{1}_{2}" # 0:raw_file, 1:id, 2:charge, 3:rank
        pin_df['SpecId'] = [SpecId_blueprint.format(*r) for r in search_results[['raw_file', 'index', 'charge','rank']].values.tolist()]

        #### BUILD ScanNr column

        pin_df['ScanNr'] = search_results['scan'].values
        pin_df['is_decoy'] = search_results['best_is_decoy'].values

        #### BUILD Label column

        pin_df['Label'] = pin_df['is_decoy'].apply(lambda x: -1 if x else 1)

        #### BUILD Peptide column

        pin_df['Peptide'] = search_results['best_peptide'].values
        pin_df['Peptide'] = pin_df['Peptide'].apply(lambda x: 'X.'+x+'.X')

        #### BUILD Protein column

        pin_df['Proteins'] = "Dummy_Protein%s"%int(np.random.randint(100))#search_results['accession'].values

        #pin_df['Reverse'] = pin_df['is_decoy'].apply(lambda x: "rev_" if x else "")

        #pin_df['Proteins'] = pin_df['Reverse'] + pin_df['Proteins']

        #### BUILD ExpMass column

        pin_df['ExpMass']  = search_results['precursorMZ'].values

        #### Build feature columns 
        for col in PIN_colnames_features:
            pin_df[col] = search_results[col].values

        PIN_colnames_a = ['SpecId',
                        'Label',
                        'ScanNr',
                        'ExpMass',]

        PIN_colnames_b = ['Peptide',
                        'Proteins']



        PIN_colnames = PIN_colnames_a + PIN_colnames_features + PIN_colnames_b#  + ['blah']     

        #print(pin_df[PIN_colnames][pin_df[PIN_colnames].isnull().any(axis=1)])

        if OUTPUT_TYPE == "PIN":
            # write as tsv without index column
            pin_df[PIN_colnames].to_csv(os.path.join(OUTPUT_DIR,'%s.pin'%key[1:]),sep='\t',index=False)

        if OUTPUT_TYPE == "PEPREC":
            peprec_df = pd.DataFrame()

            spec_id_blueprint = "controllerType=0 controllerNumber=1 scan={0}"
            peprec_df['spec_id'] = [spec_id_blueprint.format(*r) for r in search_results[['scan']].values.tolist()]
            peprec_df['modifications'] = '-'
            peprec_df['peptide'] = pin_df['Peptide'].map(lambda x: x[2:-2])
            peprec_df['charge'] = search_results['charge'].values

            peprec_df.to_csv(os.path.join(OUTPUT_DIR,'%s.PEPREC'%key[1:]),sep='\t',index=False)