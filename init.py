import os
import gdown
import zipfile

def download_zip_file(file_id, dest_path):
    """
    Baixa um arquivo ZIP do Google Drive utilizando o ID do arquivo.

    Parâmetros:
    file_id (str): O ID do arquivo no Google Drive.
    dest_path (str): O caminho de destino onde o arquivo ZIP será salvo.
    """

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)

def extract_zip_file(zip_path):
    """
    Extrai o conteúdo de um arquivo ZIP para o diretório atual.

    Parâmetros:
    zip_path (str): O caminho para o arquivo ZIP que será extraído.
    """

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()

def main():
    """
    Função principal que executa o processo de download e extração de um arquivo ZIP.
    
    Passos:
    1. Define o ID do arquivo do Google Drive e o nome do arquivo ZIP.
    2. Faz o download do arquivo ZIP para o diretório atual.
    3. Extrai o conteúdo do arquivo ZIP.
    4. Remove o arquivo ZIP após a extração.
    """

    file_id = '1DlyQQtmEJ-zK0huUPUyPxUajWHeIfDVf'
    zip_file_name = 'files.zip'

    download_zip_file(file_id, zip_file_name)
    extract_zip_file(zip_file_name)
    if os.path.exists(zip_file_name):
        os.remove(zip_file_name)
        print(f"Arquivo zip removido: {zip_file_name}")

if __name__ == "__main__":
    # Chama a função principal para executar o processo
    main()