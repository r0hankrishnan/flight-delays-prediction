import requests
import os
import zipfile
from pathlib import Path
from flight_delay_prediction.config import (
    ZIP_FILE_DIR, EXTERNAL_DATA_DIR,
    YEAR_START, YEAR_END, MONTH_CUTOFF,
    BASE_URL
)

def get_zips(year_start:int = YEAR_START, year_end:int = YEAR_END, month_cutoff:int = MONTH_CUTOFF) -> None:
    """Download each zip file for the set of months and years specified. Note that
    for the range of Jan 2020 - July 2025 this process took over an hour. Each zip
    file takes around 1-2 minutes to download.

    Args:
        year_start (int, optional): Lower bound of years to collect. Defaults to YEAR_START.
        year_end (int, optional): Upper bound of years to collect. Defaults to YEAR_END.
        month_cutoff (int, optional): Set if year_end doesn't have data for all 12 months. 
        If full year data exists, set to 12. Defaults to MONTH_CUTOFF.
    """
    ZIP_FILE_DIR.mkdir(parents=True, exist_ok=True)
    
    for year in range(year_start, year_end + 1):
        for month in range(1,13):
            if year == year_end and month > month_cutoff:
                break
            
            file_url = BASE_URL.format(year = year, month = month)
            zip_file_path = ZIP_FILE_DIR / f"{year}-{month:02d}.zip"
            
            print(f"Downloading {month}-{year} zip file...")
            response = requests.get(file_url)
            
            if response.status_code == 200:
                with open(zip_file_path, 'wb') as f:
                    f.write(response.content)
                print(f"Saved {month}-{year} zip file to {zip_file_path}")
                
            else:
                print(f"Failed to download {month}-{year} zip file. Something went wrong.\nStatus code:{response.status_code}")
                
def unpack_zips():
    """Unpack each zip file and move data CSV to raw
    data directory. This function relies on global 
    variables that you can change to your liking
    (see above). To edit what the CSV files are named,
    see the line with the inline comment.
    """
    EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for zip_file in os.listdir(ZIP_FILE_DIR):
        zip_path = ZIP_FILE_DIR / zip_file
        
        with zipfile.ZipFile(zip_path, "r") as z:
            for file in z.namelist():
                if file.lower().endswith(".csv"):
                    extracted_path = z.extract(file, EXTERNAL_DATA_DIR)

                    new_name = f"{zip_path.stem}.csv" # Define file name here
                    new_path = EXTERNAL_DATA_DIR / new_name
                    
                    Path(extracted_path).rename(new_path)
                    print(f"Extracted {file} from {zip_file} and renamed to {new_name}")