"""Leave-one-district-out weak-label robustness diagnostic for CLIMORFA v8."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
ROOT=Path('.')
OUT=ROOT/'outputs/modeling/leave_district_out_v8_2026-06-19'; OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(ROOT/'data/03_processed/grid_250m_model_features_v8.csv')
manifest=json.loads((ROOT/'configs/climorfa_feature_sets_v1.json').read_text(encoding='utf-8'))
features=[]
for family in manifest['baseline_recipes']['baseline_d_full_proxy_context']:
    for field in manifest['feature_sets'][family]:
        if field not in features: features.append(field)
mask=df.lcz_weak_label.isin([3,6,8,9]) & (pd.to_numeric(df.lcz_weak_confidence,errors='coerce')>=.60)
a=df.loc[mask].copy(); a['target']=pd.to_numeric(a.lcz_weak_label).astype(int)
counts=a.district.value_counts(); held_out=[x for x in counts.index if counts[x]>=100]
rows=[]; preds=[]
for model_name in ['random_forest']:
    for idx,district in enumerate(held_out,1):
        test=a.district.eq(district); train=~test
        if model_name=='random_forest':
            model=RandomForestClassifier(n_estimators=100,min_samples_leaf=2,max_depth=18,max_features='sqrt',class_weight='balanced_subsample',n_jobs=1,random_state=100+idx)
        else:
            model=ExtraTreesClassifier(n_estimators=100,min_samples_leaf=2,max_depth=18,max_features='sqrt',class_weight='balanced',n_jobs=1,random_state=200+idx)
        pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('model',model)])
        pipe.fit(a.loc[train,features],a.loc[train,'target']); pred=pipe.predict(a.loc[test,features]); proba=pipe.predict_proba(a.loc[test,features])
        true=a.loc[test,'target'].to_numpy(); prob=np.max(proba,axis=1)
        rows.append({'model':model_name,'held_out_district':district,'train_rows':int(train.sum()),'test_rows':int(test.sum()),'test_classes':int(pd.Series(true).nunique()),'accuracy':accuracy_score(true,pred),'balanced_accuracy':balanced_accuracy_score(true,pred),'macro_f1_present_classes':f1_score(true,pred,average='macro',zero_division=0),'mean_max_probability':float(prob.mean())})
        for grid_id,t,p,q in zip(a.loc[test,'grid_id'],true,pred,prob): preds.append({'grid_id':grid_id,'model':model_name,'held_out_district':district,'true_label':int(t),'predicted_label':int(p),'predicted_probability':float(q)})
        print(model_name,district,int(test.sum()),rows[-1]['macro_f1_present_classes'],flush=True)
pd.DataFrame(rows).to_csv(OUT/'district_metrics.csv',index=False); pd.DataFrame(preds).to_csv(OUT/'predictions.csv',index=False)
summary=pd.DataFrame(rows).groupby('model',as_index=False).agg(districts=('held_out_district','count'),test_rows=('test_rows','sum'),accuracy_mean=('accuracy','mean'),balanced_accuracy_mean=('balanced_accuracy','mean'),macro_f1_mean=('macro_f1_present_classes','mean'),macro_f1_sd=('macro_f1_present_classes','std'),macro_f1_min=('macro_f1_present_classes','min'),macro_f1_max=('macro_f1_present_classes','max'))
summary.to_csv(OUT/'summary.csv',index=False)
(OUT/'run_manifest.json').write_text(json.dumps({'schema_version':'climorfa.leave_district_out.v1','claim_status':'exploratory weak-label robustness diagnostic; not audited validation','input':'data/03_processed/grid_250m_model_features_v8.csv','classes':[3,6,8,9],'confidence_min':.60,'minimum_test_rows_per_district':100,'districts':held_out,'features':features,'models':['random_forest']},indent=2,ensure_ascii=False),encoding='utf-8')
print(summary.to_string(index=False))


