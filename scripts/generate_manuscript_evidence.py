"""Generate evidence-backed manuscript figures, tables, and summary statistics."""
from __future__ import annotations
import json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import kruskal, spearmanr
from build_figure_1_texture_atlas import build_figure_1
from build_figure_2_methodology_multipanel import build_figure_2_methodology, build_graphical_abstract
from build_figure_3_label_audit_multipanel import build_figure_3_label_audit
from build_supplementary_texture_surface_canopy_atlas import build_texture_dsm_canopy_atlas
from build_supplementary_diagnostics_multipanel import build_supplementary_diagnostics
from build_figure_4_morphology_multipanel import build_figure_4_morphology
from build_figure_5_validation_multipanel import build_figure_5_validation
from build_figure_6_explainability_multipanel import build_figure_6_explainability
from build_figure_7_uncertainty_multipanel import build_figure_7_uncertainty
from build_figure_8_2sfca_multipanel import build_figure_8_2sfca
from build_figure_9_climate_validation_multipanel import build_figure_9_climate_validation
from build_figure_10_leave_district_multipanel import build_figure_10_leave_district
from build_figure_11_audited_validation_multipanel import build_figure_11_audited_validation

ROOT=Path('.')
DATA=ROOT/'data/03_processed/grid_250m_model_features_v8.csv'
MODEL=ROOT/'outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix'
LODO=ROOT/'outputs/modeling/leave_district_out_v8_2026-07-27_coastline_fix'
DIAG=ROOT/'outputs/diagnostics/feature_diagnostics_v8_2026-07-27_coastline_fix'
AUDIT=ROOT/'outputs/diagnostics/manual_audit_qa_2026-06-19/audit_qa_summary.json'
OUT=ROOT/'outputs/diagnostics/manuscript_evidence_2026-07-27_coastline_fix'; OUT.mkdir(parents=True,exist_ok=True)
MAIN=ROOT/'paper/figures/main'; SUPP=ROOT/'paper/figures/supplementary'; TABLES=ROOT/'paper/manuscript/src/tables/generated'
for p in (MAIN,SUPP,TABLES): p.mkdir(parents=True,exist_ok=True)

sns.set_theme(style='whitegrid',context='paper',font_scale=1.05)
COLORS={3:'#7B2CBF',6:'#277DA1',8:'#F8961E',9:'#43AA8B'}
LABELS={3:'LCZ 3 compact low-rise',6:'LCZ 6 open low-rise',8:'LCZ 8 large low-rise',9:'LCZ 9 sparsely built'}
RECIPE_LABELS={'baseline_a_morphology_only':'Morphology','baseline_b_morphology_plus_vegetation':'+ vegetation','baseline_c_morphology_plus_green_blue_context':'+ green-blue','baseline_d_no_green_2sfca':'Full, no 2SFCA','baseline_d_green_2sfca_400m':'Full + 400 m','baseline_d_full_proxy_context':'Full + 800 m','baseline_d_green_2sfca_1200m':'Full + 1200 m'}
MODEL_LABELS={'random_forest':'Random Forest','extra_trees':'Extra Trees','xgboost':'XGBoost','lightgbm':'LightGBM','dummy_majority':'Majority'}
PRIMARY_MODEL='lightgbm'

def save(fig,path):
    fig.savefig(path,dpi=300,bbox_inches='tight',facecolor='white'); plt.close(fig)

def esc(x):
    s=str(x); repl={'\\':'\\textbackslash{}','&':'\\&','%':'\\%','_':'\\_','#':'\\#','$':'\\$','{':'\\{','}':'\\}'}
    for a,b in repl.items(): s=s.replace(a,b)
    return s

def repair_mojibake(value):
    """Recover UTF-8 district names decoded once as Latin-1."""
    text = str(value)
    if not any(marker in text for marker in ('Ã', 'Ä', 'Å')):
        return text
    for encoding in ('cp1252', 'latin1'):
        try:
            return text.encode(encoding).decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return text

def write_table(
    name,
    caption,
    label,
    columns,
    rows,
    align=None,
    notes=None,
    size='\\footnotesize',
    env='tabular',
    raw_columns=None,
    raw_header_columns=None,
    tabcolsep=None,
):
    align=align or ('l'+'r'*(len(columns)-1)); path=TABLES/name
    raw_columns=set(raw_columns or [])
    raw_header_columns=set(raw_header_columns or [])
    begin = f'\\begin{{tabular}}{{{align}}}' if env == 'tabular' else f'\\begin{{tabularx}}{{\\textwidth}}{{{align}}}'
    end = '\\end{tabular}' if env == 'tabular' else '\\end{tabularx}'
    body=['\\begin{table*}[htbp]','\\centering',size,f'\\caption{{{caption}}}',f'\\label{{{label}}}']
    if tabcolsep:
        body.append(f'\\setlength{{\\tabcolsep}}{{{tabcolsep}}}')
    header_cells=[str(c) if j in raw_header_columns else esc(c) for j,c in enumerate(columns)]
    body.extend([begin,'\\toprule',' & '.join(header_cells)+' \\\\','\\midrule'])
    for row in rows:
        cells=[str(v) if j in raw_columns else esc(v) for j,v in enumerate(row)]
        body.append(' & '.join(cells)+' \\\\')
    body.extend(['\\bottomrule',end])
    if notes: body.append(f'\\begin{{minipage}}{{0.96\\textwidth}}\\vspace{{2pt}}\\scriptsize Note: {notes}\\end{{minipage}}')
    body.append('\\end{table*}'); path.write_text('\n'.join(body)+'\n',encoding='utf-8')

print('loading data',flush=True)
df=pd.read_csv(DATA)
summary=pd.read_csv(MODEL/'summary_metrics.csv'); fold=pd.read_csv(MODEL/'fold_metrics.csv'); pred=pd.read_csv(MODEL/'out_of_fold_predictions.csv')
imp=pd.read_csv(MODEL/'feature_importance_summary.csv'); deltas=pd.read_csv(MODEL/'2sfca_delta_summary.csv'); fold_delta=pd.read_csv(MODEL/'2sfca_fold_deltas.csv')
class_metrics=pd.read_csv(MODEL/'pooled_class_metrics.csv'); conf_long=pd.read_csv(MODEL/'pooled_confusion_matrix_long.csv')
family_missing=pd.read_csv(DIAG/'family_missingness.csv'); collin=pd.read_csv(DIAG/'collinearity_groups_corr95.csv')
lodo=pd.read_csv(LODO/'district_metrics.csv'); lodo_summary=pd.read_csv(LODO/'summary.csv')
audit=json.loads(Path(AUDIT).read_text(encoding='utf-8'))
audited_val_dir=ROOT/'outputs/diagnostics/audited_model_validation_2026-07-31'
audited_val_summary=pd.read_csv(audited_val_dir/'audited_validation_summary.csv')
mask=df.lcz_weak_label.isin([3,6,8,9]) & (pd.to_numeric(df.lcz_weak_confidence,errors='coerce')>=.60)
a=df.loc[mask].copy(); a['class']=pd.to_numeric(a.lcz_weak_label).astype(int)
counts = a["class"].value_counts().sort_index()

# Descriptive evidence
profile_fields={'building_coverage_exact':'Building coverage','height_proxy_aw_mean_m':'Height proxy','dsm_elevation_m_std':'Surface SD','canopy_cover_gt2m_share':'Canopy cover >2 m','s2_ndvi_mean':'NDVI','road_density_exact_m_per_km2':'Road density','green_2sfca_800m_access_log1p':'2SFCA log1p','lst_c_median_mean':'Summer LST'}
profiles=a.groupby('class')[list(profile_fields)].agg(['count','median',lambda x:x.quantile(.25),lambda x:x.quantile(.75)])
profiles.to_csv(OUT/'class_profiles.csv')
selected_corr=['building_coverage_exact','height_proxy_aw_mean_m','dsm_elevation_m_std','canopy_cover_gt2m_share','canopy_volume_gt2m_proxy_m3_per_ha','s2_ndvi_mean','s2_ndbi_mean','road_density_exact_m_per_km2','network_intersection_density_per_km2','green_2sfca_800m_access_log1p','coast_min_distance_m','morph_open_space_fragmentation_index']
corr_rows=[]
for f in selected_corr:
    z=a[[f,'lst_c_median_mean']].dropna(); rho,p=spearmanr(z[f],z.lst_c_median_mean)
    corr_rows.append({'feature':f,'rho':rho,'p_value':p,'n':len(z)})
corr=pd.DataFrame(corr_rows).sort_values('rho'); corr.to_csv(OUT/'lst_spearman_correlations.csv',index=False)
groups=[a.loc[a['class']==c,'lst_c_median_mean'].dropna().to_numpy() for c in [3,6,8,9]]
H,p_kw=kruskal(*groups); n=sum(map(len,groups)); k=len(groups); eps=max(0,(H-k+1)/(n-k))
climate={'kruskal_h':float(H),'p_value':float(p_kw),'epsilon_squared':float(eps),'n':int(n),'classes':[3,6,8,9]}
(OUT/'climate_validation.json').write_text(json.dumps(climate,indent=2),encoding='utf-8')

# Figure 1 study-area and texture atlas
build_figure_1(ROOT)
grid=gpd.read_file(ROOT/'data/03_processed/analysis_grids.gpkg',layer='grid_250m')

# Figure 2 workflow and graphical abstract
build_figure_2_methodology(ROOT)
build_graphical_abstract(ROOT)

# Figure 3 readiness
build_figure_3_label_audit(ROOT)

# Figure 4 class profiles heatmap
med=a.groupby('class')[list(profile_fields)].median(); z=(med-med.mean())/med.std(ddof=0); z.columns=[profile_fields[c] for c in z.columns]
fig,ax=plt.subplots(figsize=(10,4.8)); sns.heatmap(z,annot=med.rename(columns=profile_fields).round(2),fmt='g',cmap='vlag',center=0,linewidths=.5,cbar_kws={'label':'Standardized class median'},ax=ax); ax.set_yticklabels([f'LCZ {int(x.get_text())}' for x in ax.get_yticklabels()],rotation=0); ax.set_title('Class profiles: colour is standardized; annotation is the raw median'); ax.set_xlabel(''); ax.set_ylabel('Weak-label class'); save(fig,MAIN/'fig_4_class_profiles.png')
build_figure_4_morphology(ROOT)

# Figure 5 baseline performance. (A hardcoded 2-colour placeholder plot used
# to live here before build_figure_5_validation existed; it saved to the
# same path that build_figure_5_validation immediately overwrites below, so
# it was always dead output -- and with xgboost/lightgbm now present in
# summary_metrics.csv it would additionally crash (4 hue levels vs a
# 2-colour palette list). Removed rather than fixed.
build_figure_5_validation(ROOT)

# Figure 6 importance
build_figure_6_explainability(ROOT)

# Figure 7 uncertainty map
build_figure_7_uncertainty(ROOT)

# Figure 8 2SFCA sensitivity
fd=fold_delta.copy(); fd['threshold']=fd.comparison_recipe.map({'baseline_d_green_2sfca_400m':'400 m','baseline_d_full_proxy_context':'800 m','baseline_d_green_2sfca_1200m':'1200 m'}); fd['model_label']=fd.model.map(MODEL_LABELS)
fig,axs=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True); sns.barplot(data=fd,x='threshold',y='macro_f1_delta',hue='model_label',errorbar='sd',palette=['#277DA1','#F8961E'],ax=axs[0]); axs[0].axhline(0,color='black',lw=.8); axs[0].set_ylabel('Macro-F1 delta vs no 2SFCA'); axs[0].set_title('(a) Mean threshold contribution')
sns.stripplot(data=fd,x='threshold',y='macro_f1_delta',hue='model_label',dodge=True,palette=['#277DA1','#F8961E'],ax=axs[1]); axs[1].axhline(0,color='black',lw=.8); axs[1].set_ylabel('Fold-level macro-F1 delta'); axs[1].set_title('(b) Paired spatial-fold deltas'); handles,labels=axs[1].get_legend_handles_labels(); axs[1].legend(handles[:2],labels[:2],title=''); save(fig,MAIN/'fig_8_2sfca_sensitivity.png')
build_figure_8_2sfca(ROOT)

# Figure 9 climate validation
build_figure_9_climate_validation(ROOT)

# Figure 10 leave-district
build_figure_10_leave_district(ROOT)
build_figure_11_audited_validation(ROOT)

# Supplementary figures
build_texture_dsm_canopy_atlas(ROOT)
build_supplementary_diagnostics(ROOT)

# Tables
write_table('table_1_data_inventory.tex','Analysis-ready data inventory and provenance.','tab:data_inventory',['Dataset','Date/period','Resolution','Role'],[
['Study frame','Current operational boundary','vector','Izmir FUR and grid'],['Local surface model','local archive','5 m','surface elevation/roughness'],['Buildings and roads','local vector snapshot','vector','morphometrics; network'],['Global LCZ','RUB latest at extraction','100 m working','weak labels'],['Sentinel-2','summer 2025','20 m working','spectral indices'],['Landsat 8/9','summer 2021-2025','30 m','LST response'],['Dynamic World','summer 2025','20 m working','land-cover probabilities'],['ESA WorldCover','2021','10 m','land-cover shares'],['ETH canopy height','2020','10 m','vegetation height/volume proxies'],['OSM + 2024 population','current extract / 2024','vector + grid','2SFCA supply/demand']],align='@{}>{\\raggedright\\arraybackslash}p{0.18\\textwidth}>{\\raggedright\\arraybackslash}p{0.20\\textwidth}>{\\raggedright\\arraybackslash}p{0.12\\textwidth}>{\\raggedright\\arraybackslash}X@{}',env='tabularx',tabcolsep='3pt',size='\\scriptsize',notes='All analysis layers are harmonized to a common projection.')
write_table('table_2_analysis_frame.tex','Analysis frame, labels, and current evidence status.','tab:analysis_frame',['Quantity','Value','Interpretation'],[['FUR-intersecting 250 m cells','16,506','Full analysis frame'],['Eligible core cells','15,801','Boundary-coverage screen'],['Cells with weak LCZ label','16,233','Global product aggregated to grid'],['Mixed cells at 0.70','6,585','Transition/uncertain cells'],['Four-class diagnostic rows','5,339','LCZ 3/6/8/9, confidence at least 0.60'],['Manual audit sample','569','40 LCZ-by-intensity strata'],['Completed manual audits',f"{audit['completed_rows']:,}",'Audited validation available' if audit['ready_for_audited_model_validation'] else 'Audited validation unavailable'],['Model matrix','374 fields','v8, zero join losses']],align='lrl')
write_table('table_3_feature_families.tex','Feature families used in the evidence-locked baseline.','tab:feature_families',['Family','Representative variables','Primary role'],[['2D morphology','coverage, density, compactness, fragmentation','urban fabric'],['3D/Surface','Surface SD/range, floor-height proxies','surface roughness/vertical intensity'],['Vegetation','canopy cover, p95 height, volume, NDVI','3D green structure'],['Green-blue/coast','NDWI, water/tree shares, coast distance','climate context'],['Street network','road/intersection/dead-end density, entropy','connectivity proxy'],['Street interface','continuity, gap, open-buffer share','frontage proxy'],['Green access','network-cost 2SFCA at 400/800/1200 m','competition-adjusted access'],['Semantic context','Dynamic World, WorldCover, OSM shares','proxy land-use context']],align='@{}>{\\raggedright\\arraybackslash}p{0.17\\textwidth}>{\\raggedright\\arraybackslash}X>{\\raggedright\\arraybackslash}p{0.28\\textwidth}@{}',env='tabularx',notes='LST and label-quality fields were excluded from predictors. Surface variables are surface-elevation/roughness summaries rather than building-height ground truth. OSM, floor count, frontage, and endpoint topology remain explicitly named proxies.')
rows=[]
for c in [3,6,8,9]:
    x=a[a['class']==c]; rows.append([f'LCZ {c}',f'{len(x):,}',f"{x.lst_c_median_mean.median():.2f}",f"{100*x.building_coverage_exact.median():.1f}",f"{x.height_proxy_aw_mean_m.median():.2f}",f"{x.dsm_elevation_m_std.median():.2f}",f"{100*x.canopy_cover_gt2m_share.median():.1f}",f"{x.s2_ndvi_mean.median():.3f}",f"{x.green_2sfca_800m_access_log1p.median():.3f}"])
write_table('table_4_class_profiles.tex','Median climate-morphology profiles of the four retained weak-label classes.','tab:class_profiles',['Class','n','LST (deg C)','Building (%)','Height proxy (m)','Surface SD (m)','Canopy (%)','NDVI','2SFCA'],rows,align='lrrrrrrrr',size='\\scriptsize',tabcolsep='3pt',notes='Descriptive weak-label profiles; LST is in degrees C, cover and canopy are percentages, and 2SFCA is the log1p network-cost access value at 800 m. These are not manually audited local subtypes.')
rows=[]
recipe_rank={r:i for i,r in enumerate(RECIPE_LABELS)}
summary_5=summary[summary.model.ne('dummy_majority')].copy()
summary_5['recipe_rank']=summary_5.recipe.map(recipe_rank)
for _,r in summary_5.sort_values(['recipe_rank','macro_f1_mean'],ascending=[True,False]).iterrows(): rows.append([MODEL_LABELS[r.model],RECIPE_LABELS[r.recipe],int(r.features),f"{r.macro_f1_mean:.3f} $\\pm$ {r.macro_f1_sd:.3f}",f"{r.balanced_accuracy_mean:.3f}"])
write_table('table_5_baseline_performance.tex','Five-fold stratified spatial-block weak-label performance.','tab:baseline_performance',['Model','Recipe','Features','Macro-F1 (mean $\\pm$ SD)','Balanced accuracy'],rows,align='llrrr',raw_columns=[3],raw_header_columns=[3],notes='Blocks are nominally 1.25 km (5 by 5 cells). Preprocessing is fit within each training fold. Values are mean $\\pm$ standard deviation across five folds. These are exploratory weak-label diagnostics; within each recipe, models are ordered by descending macro-F1.')
rows=[]
model_rank={'lightgbm':0,'random_forest':1,'xgboost':2,'extra_trees':3}
cm6=class_metrics[(class_metrics.recipe=='baseline_d_full_proxy_context')].copy()
cm6['model_rank']=cm6.model.map(model_rank)
for _,r in cm6.sort_values(['model_rank','class']).iterrows(): rows.append([MODEL_LABELS[r.model],f'LCZ {int(r["class"])}',int(r.support),f"{r.precision:.3f}",f"{r.recall:.3f}",f"{r.f1:.3f}"])
write_table('table_6_class_metrics.tex','Pooled out-of-fold class performance for the primary 800 m recipe.','tab:class_metrics',['Model','Class','Support','Precision','Recall','F1'],rows,align='llrrrr')
rows=[]
lodo_primary=lodo[lodo.model.eq(PRIMARY_MODEL)]
for _,r in lodo_primary.sort_values('held_out_district').iterrows(): rows.append([repair_mojibake(r.held_out_district),int(r.test_rows),int(r.test_classes),f"{r.accuracy:.3f}",f"{r.balanced_accuracy:.3f}",f"{r.macro_f1_present_classes:.3f}"])
lodo_summary_primary=lodo_summary[lodo_summary.model.eq(PRIMARY_MODEL)]
lodo_mean=float(lodo_summary_primary.macro_f1_mean.iloc[0]); lodo_min=float(lodo_summary_primary.macro_f1_min.iloc[0]); lodo_max=float(lodo_summary_primary.macro_f1_max.iloc[0])
write_table('table_7_leave_district.tex',f'Leave-one-district-out {MODEL_LABELS[PRIMARY_MODEL]} weak-label robustness.','tab:leave_district',['Held-out district','n','Classes','Accuracy','Balanced accuracy','Macro-F1'],rows,align='lrrrrr',notes=f'Only districts with at least 100 diagnostic rows were evaluated. Macro-F1 uses classes present in the held-out district; mean {lodo_mean:.3f}, range {lodo_min:.3f}-{lodo_max:.3f}. Random Forest, evaluated identically as a comparison baseline, reached mean macro-F1 {float(lodo_summary[lodo_summary.model.eq("random_forest")].macro_f1_mean.iloc[0]):.3f}.')
rows=[]
for _,r in deltas.iterrows(): rows.append([MODEL_LABELS[r.model],{'baseline_d_green_2sfca_400m':'400 m','baseline_d_full_proxy_context':'800 m','baseline_d_green_2sfca_1200m':'1200 m'}[r.comparison_recipe],f"{r.macro_f1_delta_mean:+.4f}",f"{r.macro_f1_delta_sd:.4f}",f"{int(r.macro_f1_fold_wins)}/5",f"[{r.macro_f1_delta_min:+.4f}, {r.macro_f1_delta_max:+.4f}]"])
write_table('table_8_2sfca_sensitivity.tex','Paired 2SFCA threshold contribution relative to the same model without 2SFCA.','tab:2sfca',['Model','Threshold','Mean delta','SD','Fold wins','Range'],rows,align='llrrrr',notes='The 800 m threshold is primary by design; threshold performance is interpreted as sensitivity, not post-hoc model selection.')
CLIMATE_FEATURE_LABELS={'s2_ndbi_mean':'NDBI (Sentinel-2)','s2_ndvi_mean':'NDVI (Sentinel-2)','canopy_volume_gt2m_proxy_m3_per_ha':'Canopy volume >2 m (m3 per ha)','building_coverage_exact':'Building coverage share','road_density_exact_m_per_km2':'Road density (m per km2)','network_intersection_density_per_km2':'Intersection density (per km2)','dsm_elevation_m_std':'Surface elevation SD (m)','canopy_cover_gt2m_share':'Canopy cover >2 m share','morph_open_space_fragmentation_index':'Open-space fragmentation index','green_2sfca_800m_access_log1p':'2SFCA access (log1p, 800 m)'}
rows=[]
for _,r in corr.iloc[np.argsort(np.abs(corr.rho))[::-1]].head(10).iterrows(): rows.append([CLIMATE_FEATURE_LABELS.get(r.feature,r.feature.replace('_',' ')),int(r.n),f"{r.rho:+.3f}",'<0.001' if r.p_value<.001 else f"{r.p_value:.3f}"])
write_table('table_9_climate_validation.tex','Descriptive climate-response associations within the four-class weak-label subset.','tab:climate_validation',['Feature','n','Spearman rho with LST','p'],rows,align='lrrr',notes=f"Kruskal-Wallis across LCZ 3/6/8/9: H={H:.1f}, p<0.001, epsilon-squared={eps:.3f}. Associations are bivariate and non-causal.")

rows=[]
tier_labels={'primary_high':'High quality (primary)','sensitivity_high_medium':'High + medium (sensitivity)','all_quality':'All quality (reference)'}
tier_order=['primary_high','sensitivity_high_medium','all_quality']
av=audited_val_summary.set_index('tier')
for t in tier_order:
    r=av.loc[t]
    rows.append([tier_labels[t],int(r.n),f"{100*r.weak_label_agreement:.1f}",f"{100*r.lightgbm_agreement:.1f}",f"{100*r.random_forest_agreement:.1f}"])
primary_weak_pct=100*av.loc['primary_high','weak_label_agreement']
write_table('table_10_audited_validation.tex','Manual-audit exact-class agreement for the four-class modeling population (weak label in LCZ 3/6/8/9, confidence \\(\\ge 0.60\\)).','tab:audited_validation',['Audit-quality tier','n','Weak label (%)','LightGBM (%)','Random Forest (%)'],rows,align='lrrrr',notes=f"Primary (label\\_quality==high) and sensitivity (high+medium) tiers were specified before agreement was computed. For reference, the full audited sample (n=569, every LCZ and land-cover class, unrestricted by confidence) had exact weak-label agreement {100*audit['exact_weak_audit_agreement_share']:.1f}\\%; that broader mix includes easier land-cover distinctions and is not directly comparable to the built-subtype task above.")

write_table('table_11_evidence_status.tex','Claim-evidence-status matrix for the current manuscript.','tab:evidence_status',['Claim domain','Available evidence','Status'],[['250 m data integration','16,506 rows, 374 fields, no join loss','supported'],['Surface contribution','Surface summaries in ablation and importance','exploratory supported'],['Green-space accessibility','network-cost 2SFCA + reference macro audit','proxy supported'],['Spatial robustness','5 folds + 11 held-out districts','weak-label supported'],['Climate relevance','LST class contrast and correlations','descriptive supported'],['Audited LCZ refinement',f'manual audit 569/569 complete; primary-tier exact agreement {primary_weak_pct:.1f}% (weak label) vs {100*av.loc["primary_high","lightgbm_agreement"]:.1f}% (LightGBM)','tested: low agreement'],['Deep multimodal fusion','raster/figure-ground branches not trained','not testable'],['Planning-ready local subtypes','requires deep/XAI outputs on audited labels','not testable']],align='@{}>{\\raggedright\\arraybackslash}p{0.25\\textwidth}>{\\raggedright\\arraybackslash}X>{\\raggedright\\arraybackslash}p{0.18\\textwidth}@{}',env='tabularx',notes='Unsupported confirmatory claims are intentionally excluded from the abstract and conclusion.')

# Supplementary tables
write_table('table_s1_family_missingness.tex','Feature-family missingness diagnostics.','tab:s1_missingness',['Family','Columns','Mean missing (%)','Max missing (%)','Near constant'],[[str(r.family).replace('_',' '),int(r.columns),f"{100*r.avg_missing_share:.2f}",f"{100*r.max_missing_share:.2f}",int(r.near_constant_columns)] for _,r in family_missing.iterrows()],align='@{}>{\\raggedright\\arraybackslash}Xrrrr@{}',env='tabularx',size='\\scriptsize',tabcolsep='3pt')
# collinearity groups can have variable schema; keep largest groups
cols=collin.columns.tolist(); sizecol='group_size' if 'group_size' in cols else cols[1]
colrows=[]
for _,r in collin.head(20).iterrows(): colrows.append([r.get('group_id',r.iloc[0]),r.get(sizecol,''),str(r.get('members',r.iloc[-1]))[:130]])
write_table('table_s2_collinearity.tex','First 20 high-correlation feature groups at absolute \\(\\lvert r\\rvert \\ge 0.95\\).','tab:s2_collinearity',['Group','Size','Members'],colrows,align='@{}rr>{\\raggedright\\arraybackslash\\ttfamily\\scriptsize}X@{}',env='tabularx',size='\\scriptsize',tabcolsep='3pt',notes='Family-level ablations and representative-variable selection reduce redundancy.')
assign_full=pd.read_csv(MODEL/'spatial_fold_assignments.csv'); foldrows=[]
for f0,x in assign_full.groupby('fold'): foldrows.append([int(f0),len(x),x.spatial_block.nunique(),', '.join(f"LCZ {int(c)}: {n}" for c,n in x.lcz_weak_label.value_counts().sort_index().items())])
write_table('table_s3_fold_counts.tex','Spatial-fold composition.','tab:s3_folds',['Fold','Rows','Blocks','Weak-class counts'],foldrows,align='rrrl')
cm=conf_long[(conf_long.model==PRIMARY_MODEL)&(conf_long.recipe=='baseline_d_full_proxy_context')].pivot(index='true_class',columns='predicted_class',values='count').reindex(index=[3,6,8,9],columns=[3,6,8,9]); write_table('table_s4_confusion.tex',f'{MODEL_LABELS[PRIMARY_MODEL]} pooled confusion matrix, primary 800 m recipe.','tab:s4_confusion',['True / predicted','LCZ 3','LCZ 6','LCZ 8','LCZ 9'],[[f'LCZ {i}',*map(int,cm.loc[i].values)] for i in cm.index],align='lrrrr')
qman=json.loads((ROOT/'data/04_training_labels/audit_sample_v1_priority_queue_manifest.json').read_text(encoding='utf-8')); write_table('table_s5_audit_queue.tex','Manual-audit priority queue and completion state.','tab:s5_audit',['Pass','Cells','Purpose'],[['Pass 1',150,'highest uncertainty/mixed/context priority'],['Pass 2',200,'next priority block'],['Pass 3',219,'remaining stratified sample'],['Completed',audit['completed_rows'],'valid completed labels'],['Ready for audited validation',str(audit['ready_for_audited_model_validation']),'hard model gate']],align='lrl',notes='Priority changes review order only; it does not alter inclusion weights or sample design.')

included_figures = []
for tex in (ROOT / 'paper/manuscript/src/sections').glob('*.tex'):
    text = tex.read_text(encoding='utf-8')
    included_figures.extend(Path(name).name for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', text))
included_figures = sorted(dict.fromkeys(included_figures))
main_figures = sorted(name for name in included_figures if (MAIN / name).exists())
supplementary_figures = sorted(name for name in included_figures if (SUPP / name).exists())

manifest={'schema_version':'climorfa.manuscript_evidence.v1','date':'2026-06-19','input_matrix':str(DATA),'analysis_rows':len(a),'classes':counts.to_dict(),'manual_audit':audit,'climate_validation':climate,'main_figures':main_figures,'supplementary_figures':supplementary_figures,'tables':sorted(p.name for p in TABLES.glob('*.tex')),'claim_status':'Evidence-locked weak-label baseline; audited subtype and deep multimodal claims not available.'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'analysis_rows':len(a),'main_figures':len(manifest['main_figures']),'supplementary_figures':len(manifest['supplementary_figures']),'tables':len(manifest['tables']),'kruskal_H':H,'epsilon2':eps},indent=2),flush=True)
