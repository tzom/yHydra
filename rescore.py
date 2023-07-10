import mokapot
import pandas as pd


#python ./ms2rescore/ms2rescore -c example_tryptic_e/ms2rescore.json -m example_tryptic_e/X.mgf example_tryptic_e/search/X.pin

# psms = mokapot.read_pin("../example_tryptic_e/search/01075_A01_P010693_S00_N01_R1.pin")
# results, models = mokapot.brew(psms)
# print(results)

# psms = mokapot.read_pin("../example_tryptic_e/search/01075_A01_P010693_S00_N01_R1_searchengine_ms2pip_features.pin")
# results, models = mokapot.brew(psms)
# print(results)


# psms = mokapot.read_pin("../example_nontryptic_e/search/01781_D01_P018699_S00_N04_R1.pin")
# results, models = mokapot.brew(psms)
# print(results)

#pin_file = "../example_nontryptic_e/search/01781_D01_P018699_S00_N04_R1_searchengine_ms2pip_features.pin"
pin_file = "../example_nontryptic_e/search/01781_D01_P018699_S00_N04_R1.pin"

psms = mokapot.read_pin(pin_file)
results, models = mokapot.brew(psms,test_fdr=.01, folds=3, max_workers=64)

def to_pandas(x):
    return pd.DataFrame(dict([(col,x[col]) for col in x.columns]))

psms = to_pandas(psms.data)

est = to_pandas(results.confidence_estimates['psms'])
dec_est = to_pandas(results.decoy_confidence_estimates['psms'])

psms = pd.merge(psms, pd.concat([est,dec_est]), on=['SpecId', 'Label', 'ScanNr','ExpMass', 'Peptide', 'Proteins'],how='left')

psms.to_csv(pin_file[:-4]+".pout",sep='\t',index=False)
