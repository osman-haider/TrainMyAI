import zipfile

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def extract_zip(zip_file):
    """Extracts the zip file contents and returns the list of file paths."""
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall("extracted_folder")
        print("File is extracted successfully...")