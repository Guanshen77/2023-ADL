import gdown

# Define the folder URL
url = "https://drive.google.com/drive/folders/1wQznfVaNZ9pX06eKfHgp5dpfRsYvLO87?usp=share_link"

# Execute the download_folder function
gdown.download_folder(url, quiet=True)
