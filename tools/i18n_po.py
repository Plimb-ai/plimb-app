"""Régénère lang/fr/LC_MESSAGES/plimb-studio.po avec les contextes Slint (nom du composant)."""
import re
po_path = 'lang/fr/LC_MESSAGES/plimb-studio.po'
# Lecture des traductions existantes (sans contexte)
txt = open(po_path).read()
tr = {}
plur = {}
for m in re.finditer(r'msgid "((?:[^"\\]|\\.)*)"\nmsgid_plural "((?:[^"\\]|\\.)*)"\nmsgstr\[0\] "((?:[^"\\]|\\.)*)"\nmsgstr\[1\] "((?:[^"\\]|\\.)*)"', txt):
    plur[m.group(1)] = (m.group(2), m.group(3), m.group(4))
for m in re.finditer(r'(?:^|\n)msgid "((?:[^"\\]|\\.)+)"\nmsgstr "((?:[^"\\]|\\.)*)"', txt):
    tr[m.group(1)] = m.group(2)

entries = {}  # (ctx, msgid) -> entry text
missing = []
for f in ['ui/app.slint', 'ui/components.slint']:
    comp = None
    for line in open(f):
        m = re.match(r'\s*(?:export\s+)?component\s+([\w-]+)', line)
        if m:
            comp = m.group(1)
        for t in re.finditer(r'@tr\("((?:[^"\\]|\\.)*)"(?:\s*\|\s*"((?:[^"\\]|\\.)*)"\s*%)?', line):
            mid, mpl = t.group(1), t.group(2)
            key = (comp, mid)
            if mpl is not None:
                p = plur.get(mid)
                if not p: missing.append(mid); continue
                entries[key] = f'msgctxt "{comp}"\nmsgid "{mid}"\nmsgid_plural "{mpl}"\nmsgstr[0] "{p[1]}"\nmsgstr[1] "{p[2]}"\n'
            else:
                if mid not in tr: missing.append(mid); continue
                entries[key] = f'msgctxt "{comp}"\nmsgid "{mid}"\nmsgstr "{tr[mid]}"\n'
head = 'msgid ""\nmsgstr ""\n"Content-Type: text/plain; charset=UTF-8\\n"\n"Language: fr\\n"\n"Plural-Forms: nplurals=2; plural=(n > 1);\\n"\n\n'
with open(po_path, 'w') as out:
    out.write(head + '\n'.join(entries[k] for k in sorted(entries, key=lambda k: (k[0] or '', k[1]))))
print(len(entries), 'entrées ; manquantes :', sorted(set(missing)))
