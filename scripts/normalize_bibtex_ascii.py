"""Convert DOI-verified UTF-8 BibTeX to classic BibTeX-safe LaTeX accents."""
from pathlib import Path
src=Path('references/refs_utf8_doi_verified.bib')
dst=Path('paper/manuscript/src/refs.bib')
text=src.read_text(encoding='utf-8')
repl={
'á':r"{\'a}",'é':r"{\'e}",'í':r"{\'i}",'ó':r"{\'o}",'ý':r"{\'y}",
'ñ':r"{\~n}",'ö':r'{\"o}','ü':r'{\"u}','č':r'{\v{c}}',
'Ç':r'{\c{C}}','ş':r'{\c{s}}','ğ':r'{\u{g}}','ı':r'{\i}','İ':r'{\.{I}}',
'‐':'-','–':'--','‘':"`",'’':"'"
}
for a,b in repl.items(): text=text.replace(a,b)
non_ascii=sorted(set(ch for ch in text if ord(ch)>127))
if non_ascii: raise SystemExit(f'unmapped characters: {non_ascii}')
dst.write_text(text,encoding='ascii',newline='\n')
print(f'wrote {dst} bytes={dst.stat().st_size}')
