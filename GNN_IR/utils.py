import pandas as pd

def build_features_dataframes(starting_df: pd.DataFrame, queries_df: pd.DataFrame, docs_df: pd.DataFrame, qrels: pd.DataFrame):
    """
    Processes and merges multiple DataFrames into a single DataFrame with additional operations.

    Args:
    - starting_df (pd.DataFrame): DataFrame with initial query results.
    - queries_df (pd.DataFrame): DataFrame containing queries and their vectors.
    - docs_df (pd.DataFrame): DataFrame containing documents and their vectors.
    - qrels (pd.DataFrame): DataFrame containing qrels (representation of relevance judgments).

    Returns:
    - Tuple[pd.DataFrame, Dict[str, int]]: A tuple containing the features DataFrame, a docno-to-index and a docno-to-docid mapping dictionaries.
    """
    # Merge with queries DataFrame on 'qid'
    merged_df = starting_df.merge(queries_df, on='qid', how='left')

    # Merge with docs DataFrame on 'docno'
    merged_df = merged_df.merge(docs_df, on='docno', how='left')

    # Extract the qid value (assuming it's the same for all rows in 'top_k_documents')
    qid_value = merged_df['qid'].iloc[0]

    # Filter the 'qrels' DataFrame to keep rows where 'qid' matches the extracted qid_value
    filtered_qrels = qrels[qrels['qid'] == qid_value]

    # Merge with the filtered 'qrels' DataFrame on 'docno'
    merged_df = merged_df.merge(filtered_qrels[['docno', 'label']], on='docno', how='left')

    # Fill missing values in 'label' column with -1 and rename to 'qrel'
    merged_df['qrels'] = merged_df['label'].fillna(-1).astype(int)
    merged_df.drop(columns=['label'], inplace=True)

    # Create a docno-to-index mapping
    docno_to_index = {docno: idx for idx, docno in enumerate(merged_df['docno'].unique())}
    merged_df['docno_index'] = merged_df['docno'].map(docno_to_index)


    return docno_to_index