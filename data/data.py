from data.molecules import MoleculeDataset

def LoadData(DATASET_NAME):
    """
        This function is called in the main.py file
        returns:
        ; dataset object
    """

    # Handling for (ZINC) molecule dataset
    if DATASET_NAME in ['ZINC', 'ZINC-full']:
        return MoleculeDataset(DATASET_NAME)

    # Fallback/Error Handling
    raise ValueError(f"Dataset {DATASET_NAME} is not supported in this minimal version. Please use 'ZINC'.")