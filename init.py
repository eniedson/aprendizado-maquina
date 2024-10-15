import os
import gdown
import zipfile

def download_zip_file(file_id, dest_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)

def extract_zip_file(zip_path):
    extract_to = ''
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Arquivos extraídos para: {extract_to}")

def main():
    file_id = '1DlyQQtmEJ-zK0huUPUyPxUajWHeIfDVf'
    zip_file_name = 'files.zip'

    download_zip_file(file_id, zip_file_name)
    extract_zip_file(zip_file_name)
    if os.path.exists(zip_file_name):
        os.remove(zip_file_name)
        print(f"Arquivo zip removido: {zip_file_name}")

if __name__ == "__main__":
    main()