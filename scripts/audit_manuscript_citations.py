"""Audit bibliography coverage and citation-key validity in the LaTeX manuscript."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
BIB_KEY_RE = re.compile(r"^@\w+\{\s*([^,\s]+)", re.MULTILINE)
CITE_RE = re.compile(r"\\(?:cite|citep|citet|citealp|citeauthor|citeyear)\s*\{([^}]+)\}")
MAPPING_RE = re.compile(r"`([A-Za-z0-9_:-]+)`")
def main():
    p=argparse.ArgumentParser(); p.add_argument('--manuscript',default='paper/manuscript/src'); p.add_argument('--bib',default='paper/manuscript/src/refs.bib'); p.add_argument('--mapping',default='references/reference_mapping.md'); p.add_argument('--out-dir',default='outputs/diagnostics/manuscript_citation_audit_2026-06-19'); a=p.parse_args()
    manuscript=Path(a.manuscript); bib_path=Path(a.bib); mapping_path=Path(a.mapping); out_dir=Path(a.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    bib_keys=sorted(set(BIB_KEY_RE.findall(bib_path.read_text(encoding='utf-8')))); tex_files=sorted(manuscript.rglob('*.tex')); cited_by_file={}; all_cited=set()
    for path in tex_files:
        keys=set()
        for group in CITE_RE.findall(path.read_text(encoding='utf-8')): keys.update(k.strip() for k in group.split(',') if k.strip())
        cited_by_file[path.as_posix()]=sorted(keys); all_cited.update(keys)
    mapping_keys=sorted(set(MAPPING_RE.findall(mapping_path.read_text(encoding='utf-8')))); mapped_bib_keys=sorted(set(mapping_keys).intersection(bib_keys))
    result={'schema_version':'climorfa.manuscript_citation_audit.v1','bibliography_entries':len(bib_keys),'cited_bibliography_entries':len(set(bib_keys).intersection(all_cited)),'mapped_bibliography_entries':len(mapped_bib_keys),'missing_in_text':sorted(set(bib_keys)-all_cited),'unknown_citation_keys':sorted(all_cited-set(bib_keys)),'bibliography_not_in_mapping':sorted(set(bib_keys)-set(mapped_bib_keys)),'cited_by_file':cited_by_file}
    (out_dir/'citation_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    report=['# Manuscript citation audit','',f"- Bibliography entries: **{result['bibliography_entries']}**",f"- Cited bibliography entries: **{result['cited_bibliography_entries']}**",f"- Mapping-covered bibliography entries: **{result['mapped_bibliography_entries']}**",f"- Missing in text: **{len(result['missing_in_text'])}**",f"- Unknown citation keys: **{len(result['unknown_citation_keys'])}**",'','## Missing in-text citations','',*(f'- `{k}`' for k in result['missing_in_text']),'','## Unknown citation keys','',*(f'- `{k}`' for k in result['unknown_citation_keys'])]
    (out_dir/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8'); print(json.dumps({k:v for k,v in result.items() if k!='cited_by_file'},indent=2))
    if result['missing_in_text'] or result['unknown_citation_keys']: raise SystemExit(2)
if __name__=='__main__': main()
