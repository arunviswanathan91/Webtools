import pandas as pd
from io import StringIO
from typing import Optional, Any

def load_data(file_obj: Any, header: Optional[int] = 0) -> pd.DataFrame:
    """
    Loads data from a file object (CSV or TSV).
    
    Args:
        file_obj: Uploaded file object from Streamlit.
        header: Row number to use as the column names, and the start of the data. 
                Default is 0. If None, columns will be generated.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if file_obj is None:
        return pd.DataFrame()
    
    try:
        # Determine separator based on extension if possible, else default to csv
        filename = getattr(file_obj, "name", "").lower()
        sep = "\t" if filename.endswith(".tsv") or filename.endswith(".txt") else ","
        
        # Reset pointer if needed (though usually handled by streamlit)
        file_obj.seek(0)
        
        df = pd.read_csv(
            file_obj, 
            sep=sep, 
            header=header, 
            comment="#"
        )
        
        # If header is None, generate column names
        if header is None:
            df.columns = [f"Col {i+1}" for i in range(df.shape[1])]
            
        return df
        
    except Exception as e:
        # Fallback to simple error handling; caller should catch or display
        raise ValueError(f"Error loading file: {str(e)}")
