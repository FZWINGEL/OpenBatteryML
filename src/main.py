
import argparse
import logging
import os

from parsers.calce_parser import CalceParser
from parsers.hnei_parser import HneiParser
from parsers.nasa_pcoe_parser import NasaPcoeParser

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    """Main function to parse command-line arguments and run the specified parser."""
    parser = argparse.ArgumentParser(description="Process battery datasets and save them in Parquet format.")
    parser.add_argument("dataset", choices=["nasa_pcoe", "calce", "hnei", "all"], help="The name of the dataset to process.")
    args = parser.parse_args()

    # Define the root directory of the project
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if args.dataset == "nasa_pcoe" or args.dataset == "all":
        logging.info("Processing NASA PCoE dataset...")
        nasa_parser = NasaPcoeParser(
            raw_data_dir=os.path.join(root_dir, "data", "raw", "nasa_pcoe"),
            processed_data_dir=os.path.join(root_dir, "data", "processed")
        )
        nasa_parser.execute()
        logging.info("Finished processing NASA PCoE dataset.")

    if args.dataset == "calce" or args.dataset == "all":
        logging.info("Processing CALCE dataset...")
        calce_parser = CalceParser(
            raw_data_dir=os.path.join(root_dir, "data", "raw", "calce"),
            processed_data_dir=os.path.join(root_dir, "data", "processed")
        )
        calce_parser.execute()
        logging.info("Finished processing CALCE dataset.")

    if args.dataset == "hnei" or args.dataset == "all":
        logging.info("Processing HNEI dataset...")
        hnei_parser = HneiParser(
            raw_data_dir=os.path.join(root_dir, "data", "raw", "hnei"),
            processed_data_dir=os.path.join(root_dir, "data", "processed")
        )
        hnei_parser.execute()
        logging.info("Finished processing HNEI dataset.")

if __name__ == "__main__":
    main()
