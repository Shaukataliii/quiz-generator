import os

from src.module import Utils, DFToDocsConverter
from langchain_codebase.codebase import load_vectorstore, add_docs_to_chroma_vectorstore


DFS_DIR = "books_df_csvs"
BOOKS_VDB_PATH = "VDB-ALL-BOOKS"
UTILS = Utils()



def main():
    df_filepath = input("Enter book CSV file path: ")
    sure = input("Are you sure? (yes/no) ") == "yes"
    if sure:
        add_df_to_vdb(df_filepath)


def add_df_to_vdb(df_filepath: str):
    """Loads dataframe, optimizes its metadata and adds the docs to the vectorstore.

    Args:
        df_filepath (str) : A valid DataFrame CSV filepath.
    """
    df = UTILS.load_pandas_df(df_filepath)
    docs = DF_TO_DOCS_CONVERTER.convert_df_to_docs_with_proper_metadata(df)

    vectorstore = load_vectorstore(BOOKS_VDB_PATH)
    add_docs_to_chroma_vectorstore(docs, vectorstore)
    
    print(f"{df_filepath} docs added to {BOOKS_VDB_PATH}")


if __name__ == "__main__":
    DF_TO_DOCS_CONVERTER = DFToDocsConverter()
    main()