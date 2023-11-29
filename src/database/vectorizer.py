from typing import List, Tuple, Union
import json 
from langchain.schema.document import Document
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

from typing import List,Tuple ,Union, Any, Dict
import os 
import ast
from .document_generator import JsonToDocument



class VectorRetriever:
    """
    pass overwrite = True when creating the vector store for the first time and if you want to overwrite intentionally.
    pass overwrite = False otherwise.
    """
    def __init__(self, model_name: str, model_kwargs: dict, encode_kwargs: dict, overwrite: bool = False) -> None:

        
        # we can pass different vector stores instead of built-in implemented-Chroma option. but this may require further consideration.
        self.vector_store: Union[Chroma, None] = None
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.encode_kwargs = encode_kwargs
        self.embedder = self.initialize_embedding_func()
        self.overwrite = overwrite
        

    def initialize_embedding_func(self):
        """
        Initializes the embedding function.

        :return: The initialized HuggingFaceEmbeddings object.
        """
        hf = HuggingFaceEmbeddings(
        model_name=self.model_name,
        model_kwargs=self.model_kwargs,
        encode_kwargs=self.encode_kwargs)

        embedding_dimension = hf.dict()['client'][1].get_config_dict()["word_embedding_dimension"]
        print("embedder initialized with dimension: ", embedding_dimension)

        return hf

    def initialize_vector_store(self, persist_directory:str,collection_name:str, documents:List[Document]=None,):
        """
        Initializes a Chroma vector store with the given texts and collection name.
        if overwrite = True, the vector store will be overwritten or created if it does not exist.


        Args:
            persist_directory (str): The directory to persist the vector store.
            texts (List[str]): The list of texts to be stored in the vector store.
            collection_name (str): The name of the collection.

        Returns:
            Chroma: The initialized Chroma vector store.
        """
        if self.overwrite:
            if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
                print("persist directory already exists")
            else: 
                os.mkdir(persist_directory, exist_ok=True)
                print("persist directory created")
        
            self.vector_store = Chroma.from_documents(
                        documents =documents , 
                        embedding=self.embedder,
                        collection_name= collection_name,
                        persist_directory=persist_directory)

            self.vector_store.persist()
        
        else:
            # if user wants to use an existing vector store as retriever. user doesnot need to pass any documents.
            self.vector_store = Chroma(persist_directory=persist_directory, embedding_function=self.embedder, collection_name=collection_name)

        print("chroma vector store initialized successfully, and ready for retrieval!")





    @staticmethod
    def drop_duplicates(raw_text_list):
        """
        drops duplicates from a list of strings
        """
        return list(set(raw_text_list))



    def similarity_search(self, query: Union[str, List[str]], k:int = 3,vector_type:str="product")->List[Tuple[str, float]]:

        """
        Performs a similarity search on the given query and filters the results by metadata.

        Parameters:
        - query : The query/ies to search for.
        - k : The number of results to return.

        Returns:
        - A list of tuples containing the documents and their corresponding similarity scores.
        """
                
        if isinstance(query, list):
            query = "-".join(query) # concatenate list items to a string

        if isinstance(query, str):
            try:

                #print(f"query: {query} \n k: {k} \n filter: {filter} \n where: {where} \n where_document: {where_document}")
                results = self.vector_store.max_marginal_relevance_search(query, k=k,)
                #print("similarity search is performed successfully!")
                #print(results)
                #return results
                print(type(results))
                return self.post_process_results(results, vector_type=vector_type)
            except Exception as e:
                print("similarity search failed with error: ", e)
                return None 

        else: 
            raise ValueError("query must be a string or a list of strings")




    def add_new_documents(self,documents:List[Document])->List[str]:

        """Adds new documents: a list of langchain Document objects to the vector store.
           Its better to be consistent with the metadata of the documents.
        """
        # we need to ensure that the documents are unique by their specific metadata(e.g. href)

        ids = self.vector_store.add_documents(documents)
        self.vector_store.persist()# make sure the changes to database are persisted -> inherent behaviour from sqlite3
        print("new documents added to the vector store ids:", ids)
        return ids


    def post_process_results(self, search_results: List[Document], vector_type:str) -> List[Dict]:
        if search_results is None:
            return "No results found."
        #print("search results:\n", search_results )
        formatted_results = []

        for document in search_results:
            try:
                if vector_type == "product":
                    doc = {
                        "img_url": document.metadata["img_url"],
                        "description": document.page_content.split("|||")[0],
                    }
                elif vector_type == "user":
                    doc = {
                        "user_id": document.metadata["user_id"],
                    }
                #formatted_results.append(user)
                formatted_results.append(doc)
            except AttributeError:
                # Handle the case where the document does not have the expected attributes
                print("Missing attribute in document")
        

        return formatted_results








if __name__ == "__main__":


    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}


    persist_directory = r"C:\Users\ayhan\Desktop\Smart-Shopper-Recommendation-Engine\artifact\products_vector_store_updated"
    #persist_directory = r"C:\Users\ayhan\Desktop\Smart-Shopper-Recommendation-Engine\artifact\users_vector_store"
    os.makedirs(persist_directory, exist_ok=True)
    collection_name= "products"
    #collection_name = "users"
    

    
    

    #single document
    #json_file_path = r"C:\Users\ayhan\Desktop\ChefApp\artifacts\recipes\cusine\italian\italian-desserts.json"
    #documents = json_to_document.process_json_document(file_path=json_file_path)
    
    producst_path = r"C:\Users\ayhan\Desktop\Smart-Shopper-Recommendation-Engine\src\product_pool.json"
    users_path = r"C:\Users\ayhan\Desktop\Smart-Shopper-Recommendation-Engine\src\masked_user_profiles.json"
    updated_products_path = r"C:\Users\ayhan\Desktop\Smart-Shopper-Recommendation-Engine\src\updated_product_pool.json"
    json_to_document = JsonToDocument()
    #documents = json_to_document.process_json_document(producst_path)
    documents = json_to_document.process_json_document(updated_products_path)
    

    print(len(documents),"documents found!", type(documents)) # created database from 12456 documents in 165 seconds.

    ####################### VECTORSTORE & RETRIEVAL ###############################################

    vector_retriever = VectorRetriever(model_name = model_name, model_kwargs= model_kwargs, encode_kwargs=encode_kwargs, overwrite=True)

    #using CHROMA
    # since overwrite is set to False, it will initialize the vector store as retriever.pass documents=None
    vector_retriever.initialize_vector_store(persist_directory=persist_directory, documents=documents, collection_name=collection_name)
   



    
    description =  str({
        "description": "[{\"mouse_description\":\"F1 23 Standard PCWin | Downloading Code EA App - Origin | VideoGame | English\"}]",
        "img_url": "https://m.media-amazon.com/images/I/81mwZgaWbzL._AC_UY218_.jpg",
        "img_vector": str([
            0.0036406279541552067,
            0.004459529649466276,
            0.0003818089026026428,
            0.002264398382976651,
            0.0006581933121196926,
            0.003624215256422758,
            0.012133832089602947,
            0.0008295223815366626,
            0.042358968406915665,
            0.01204274594783783,
            0.004482813645154238,
            0.039213407784700394,
            0.015498315915465355,
            0.0,
            0.00015318997611757368,
            9.693340689409524e-05,
            0.0,
            0.0,
            0.002839979249984026,
            0.05105264484882355,
            0.0030705928802490234,
            0.008975325152277946,
            0.041931554675102234,
            0.0,
            0.01961524970829487,
            0.001195605262182653,
            0.04615384340286255,
            0.02058376371860504,
            0.003969556652009487,
            0.0,
            0.01925002597272396,
            0.0071684070862829685,
            0.0024123145267367363,
            0.02146882377564907,
            0.0003941525355912745,
            0.010601747781038284,
            0.0008717486052773893,
            9.94916699710302e-05,
            0.0009474712423980236,
            0.015846049413084984,
            0.0006513021653518081,
            1.285420421481831e-05,
            0.0,
            0.019187496975064278,
            0.001749716349877417,
            0.005703969858586788,
            0.001331739709712565,
            0.008312078192830086,
            0.0016811055829748511,
            0.006236277520656586,
            0.00300886039622128,
            0.0023947812151163816,
            0.004988392814993858,
            0.009163737297058105,
            0.017635338008403778,
            0.004511386156082153,
            6.537342414958403e-05,
            0.011068008840084076,
            0.006045845337212086,
            0.0,
            0.0005756748141720891,
            7.02491233823821e-05,
            0.00014090805780142546,
            0.003996310755610466,
            0.00017337033932562917,
            0.005996072199195623,
            0.014059174805879593,
            0.014405333437025547,
            0.0010313456878066063,
            0.06298103928565979,
            0.09643395990133286,
            0.0003091575053986162,
            0.03067564032971859,
            0.004693282768130302,
            0.018710464239120483,
            0.0,
            0.007172402460128069,
            0.0,
            0.009552507661283016,
            0.00149693398270756,
            0.00135329260956496,
            0.004562662914395332,])
            })




    # Perform a similarity search
    results = vector_retriever.similarity_search(query=description, k=3)


    
    for result in results:
        #print(result)
        #print(type(result))
        print(json.dumps(result, indent=4))
        print("\n\n")


    # Document: (page_content, metadata) where metadata: dict
    """ print(results[0].page_content)
    print(results[0].metadata.keys())
    print(results[0].metadata["recipe_name"]) """
    print("DONE")
