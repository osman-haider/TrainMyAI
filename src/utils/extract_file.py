import zipfile

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def extract_zip(zip_file):
    """
    Extracts the contents of a zip file and saves them into a folder named 'extracted_folder'.

    Parameters:
    - zip_file: Path to the zip file to be extracted.

    Returns:
    - None. Prints a message upon successful extraction.
    """
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall("extracted_folder")
        print("File is extracted successfully...")