import gdown

# Define the folder URL
url = "https://drive.google.com/drive/u/2/folders/1mvLxi76rEmAZc8dd5GJ0XXS8gLUeP0U8"

# Execute the download_folder function
gdown.download_folder(url, quiet=True)
