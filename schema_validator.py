import pandas as pd
from typing import List, Dict, Tuple, Optional

class SchemaValidator:
    @staticmethod
    def validate(df: pd.DataFrame, required_columns: List[str], optional_columns: List[str] = []) -> Tuple[bool, List[str], List[str]]:
        """
        Validates if the DataFrame contains the required columns.
        
        Args:
            df: The DataFrame to validate.
            required_columns: List of column names that MUST be present (after mapping).
            optional_columns: List of column names that SHOULD be present (warn if missing).
            
        Returns:
            Tuple containing:
            - is_valid (bool): True if all required columns are present.
            - missing_required (List[str]): List of missing required columns.
            - missing_optional (List[str]): List of missing optional columns.
        """
        columns = set(df.columns)
        missing_required = [col for col in required_columns if col not in columns]
        missing_optional = [col for col in optional_columns if col not in columns]
        
        is_valid = len(missing_required) == 0
        return is_valid, missing_required, missing_optional
