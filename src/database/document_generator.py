import json 

from langchain.schema.document import Document
from typing import List , Dict, Union
 


class JsonToDocument:
    """
    Takes a JSON file and returns a list of Langchain Document objects.
    
    """
    def __init__(self):

        print("JsonToDocument object initialized successfully!")# this should be logging instead

    
    
    def replace_null(self, obj:Union[Dict, List], fill_value = 'null')->Union[Dict, List]:
        """
        Recursively replaces all occurrences of `None` with the string `'null'` in a given JSON object.
        """
        if isinstance(obj, dict):
            return {k: self.replace_null(v) for k, v in obj.items()} # call replace_null for each item
        elif isinstance(obj, list):
            return [self.replace_null(v) for v in obj]
        elif obj is None:
            return fill_value
        else:
            return obj



    def process_json_document(self,file_path:str)->List[Document]:
        """
        Process a JSON document and return a list of Document objects.
        """ 
        # load the JSON data: list of recipe dictionaries derived from .xlsx file
        with open(file_path, 'r') as f:
            json_data = json.load(f) 
            # we should make the loaded json prettier via json.dumps(json_data, indent=4)
        json_data = self.replace_null(json_data)
        
        print("json document is loaded successfully!")
        documents = []


        """for user_id, user_mask in json_data.items(): 
            user_mask = str(user_mask)
            document_text = user_mask
            
            # Construct a new Document object
            new_document = Document(
                metadata={"user_id": user_id},
                page_content=document_text)
                # add additional fields, if necessary

            documents.append(new_document)"""

        
        # Process each recipe in the JSON data
        for product in json_data:
            img_url = product.get('img_url', 'None')
            img_vector = str(product.get('img_vector', 'None'))
            
            description = product.get('description', 'None').replace("mouse_description", "description")
            document_text = description + "|||" + img_vector
            # Construct a new Document object
            new_document = Document(
                metadata={"img_url": img_url},
                page_content=document_text)
                # add additional fields, if necessary

            documents.append(new_document) # we use append since we are adding documents one by one"
        
        print("Documents are generated successfully! # of documents: ", len(documents))
        return documents




if __name__ == "__main__":
    json_to_document = JsonToDocument()
    #documents = json_to_document.process_json_document("src/product_pool.json")
    #print(documents[0])
    documents = json_to_document.process_json_document("src/updated_product_pool.json")
    #documents = json_to_document.process_json_document("src/masked_user_profiles.json")
    print(documents[0])