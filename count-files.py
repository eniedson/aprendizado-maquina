import os
print(len(list(filter(lambda x: x.startswith('p-'), os.listdir('./md_files')))))