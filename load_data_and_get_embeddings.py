# imports
import os
import pandas as pd
import tiktoken
import openai
from openai.embeddings_utils import get_embedding

def load_data_and_get_embeddings():
    """
    Function load_data_and_get_embeddings():
    * loads a CSV file with food reviews, 
    * preprocesses the data, 
    * generates text embeddings using the OpenAI API
    * and saves them to a new CSV file.

    Raises
    ------
    Exception
        If no OpenAI API kei is set as environment variable in the system.
    """

    # embedding model parameters
    embedding_model = "text-embedding-ada-002"
    embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
    max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

    # check the system and add files to path
    input_file_name = "food_reviews_20.csv"
    output_file_name = "food_reviews_embeddings_20.csv"
    if os.name == "posix":
        datapath = "data/"
        print("Linux")
    elif os.name == "nt":
        datapath = "data\\"
        print("Windows")
    else:
        datapath = "data\\"
        print("other")
    input_datapath = datapath + input_file_name
    output_datapath = datapath + output_file_name

    # load & inspect dataset
    df = pd.read_csv(input_datapath, sep=';',index_col=0, encoding= 'ansi') #, encoding= 'unicode_escape')
    df = df[["Score", "Text"]] # remove unnecessary columns
    df = df.dropna() # remove missing values
    # df.head(2) # return the first n rows

    # add column with lenght of Text in tokens
    encoding = tiktoken.get_encoding(embedding_encoding)
    df["n_tokens"] = df.Text.apply(lambda x: len(encoding.encode(x)))
    # omit reviews that are too long to embed
    top_n = 1000
    df = df[df.n_tokens <= max_tokens].tail(top_n)

    # set your API key
    openai.api_key = os.getenv("OPENAI_API_KEY") # get API key from environment variable
    if not openai.api_key: raise Exception("There is no API kei!")

    # get embeddings and save them for future reuse
    print("Connection to AI started... this may take a few minutes")
    df["embedding"] = df.Text.apply(lambda x: get_embedding(x, engine=embedding_model))
    print("Successful connection to AI!")
    df.to_csv(output_datapath, sep=';', encoding= 'ansi')
    
    print(df.head(2))


if __name__ == "__main__":
    load_data_and_get_embeddings()