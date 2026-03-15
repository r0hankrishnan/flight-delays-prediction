"""
Script to automate downloading ZIP file of flight delay data
for each month between January 2020 and July 2025
"""
from flight_delay_prediction.data import get_zips, unpack_zips
from flight_delay_prediction.config import YEAR_START, YEAR_END, MONTH_CUTOFF

if __name__ == "__main__":
    print("Downloading flight delay data from The Bureau of Transportation Statistics.")
    print(f"Downloading data between {YEAR_START} to {YEAR_END} (cutoff at {MONTH_CUTOFF}).")
    
    get_zips()
    
    print("Finished downloading. Now unpacking zip files and moving to external data directory.")
    
    unpack_zips()
    
    print("Finished downloading and unpacking flight delay data.")
