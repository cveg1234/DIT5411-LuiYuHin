"""Extract the Big5-encoded dataset zip with correct Chinese filenames."""
import os
import sys
import zipfile

ZIP = r'C:\Users\brend\OneDrive\Desktop\hw\DIT5411-LuiYuHin\data\handwritting_data_all\all_data.zip'
DEST = r'C:\Users\brend\OneDrive\Desktop\hw\DIT5411-LuiYuHin\data\cleaned_data'

os.makedirs(DEST, exist_ok=True)
z = zipfile.ZipFile(ZIP)
infos = z.infolist()
print(f'Extracting {len(infos)} entries...', flush=True)

for i, info in enumerate(infos):
    raw = info.filename
    try:
        name = raw.encode('cp437').decode('big5')
    except Exception:
        name = raw
    # Normalise separators, strip leading cleaned_data/
    name = name.replace('\\', '/')
    if name.startswith('cleaned_data/'):
        name = name[len('cleaned_data/'):]
    if not name:
        continue
    target = os.path.join(DEST, name.replace('/', os.sep))
    if info.is_dir():
        os.makedirs(target, exist_ok=True)
        continue
    os.makedirs(os.path.dirname(target), exist_ok=True)
    try:
        with z.open(info) as src, open(target, 'wb') as dst:
            dst.write(src.read())
    except Exception as e:
        print(f'Failed {name}: {e}', file=sys.stderr, flush=True)
    if (i + 1) % 50000 == 0:
        print(f'  {i+1}/{len(infos)}', flush=True)

print('Extraction complete.', flush=True)
