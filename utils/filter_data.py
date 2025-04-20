import pandas as pd
def filter_data(data_path, columns):
    """Filter the given data according to the given columns."""
    try:
        dframe = pd.read_excel(data_path)
        missing_cols = [col for col in columns if col not in dframe.columns]
        if missing_cols:
            raise ValueError(f"The following columns are not found in the file: {missing_cols}")
        
        extracted_df = dframe[columns]
    
    except Exception as err:
        print("An error occurred: {err}")
    
    return extracted_df