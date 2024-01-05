import gdown

# Define the folder URL
url = "https://drive.google.com/drive/folders/17Wo9ZfbXecL4lT0Q9DoerLzCrFM32BB8?usp=sharing"

# Execute the download_folder function
gdown.download_folder(url, quiet=True)
