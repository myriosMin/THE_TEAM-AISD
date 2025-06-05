def load_csv(files: list) -> None:
    """
    Load a CSV file into a pandas DataFrame. Downloads the file if it does not exist locally.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    data_path = "olistbr/brazilian-ecommerce"
    save_path = Path("data/01_raw")
    os.makedirs(save_path,exist_ok=True)
    # Download the file if it does not exist (fixed to Olist dataset from Kaggle for demonstration)
    for file in files:
        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            data_path, 
            file,
        )
        file_path = save_path / file
        df.to_csv(file_path, index=False)
        logging.info(f"File {file} downloaded successfully and loaded into {file_path.parent}.")
        
    # Returns the pandas DataFrame    
    return None

files = ["olist_customers_dataset.csv", "olist_geolocation_dataset.csv", "olist_order_items_dataset.csv",
        "olist_order_payments_dataset.csv", "olist_order_reviews_dataset.csv", "olist_orders_dataset.csv", "olist_products_dataset.csv",
        "olist_sellers_dataset.csv", "product_category_name_translation.csv"]
