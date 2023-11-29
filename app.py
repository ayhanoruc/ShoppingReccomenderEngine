from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA
from PIL import Image as PILImage
from urllib.request import urlretrieve

from src.database.vectorizer import VectorRetriever
from src.database.document_generator import JsonToDocument


model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

user_persist_directory = r"C:\Users\ayhan\Desktop\Smart-Shopper-Recommendation-Engine\artifact\products_vector_store_updated"
product_persist_directory = r"C:\Users\ayhan\Desktop\Smart-Shopper-Recommendation-Engine\artifact\users_vector_store"
os.makedirs(user_persist_directory, exist_ok=True)
os.makedirs(product_persist_directory, exist_ok=True)

user_collection_name= "users"
product_collection_name = "products"

user_vector_retriever = VectorRetriever(model_name = model_name, model_kwargs= model_kwargs, encode_kwargs=encode_kwargs, overwrite=False)
user_vector_retriever.initialize_vector_store(persist_directory=user_persist_directory, documents=None, collection_name=user_collection_name)

product_vector_retriever = VectorRetriever(model_name = model_name, model_kwargs= model_kwargs, encode_kwargs=encode_kwargs, overwrite=False)
product_vector_retriever.initialize_vector_store(persist_directory=product_persist_directory, documents=None, collection_name=product_collection_name)



app = FastAPI()

# Pydantic model definitions
class Product(BaseModel):
    description: str
    image_url: Optional[HttpUrl]  # Image URL is optional because it's not provided in case 2

class User(BaseModel):
    user_id: str
    user_masked_vector: str
    user_purchase_history: Optional[List[Product]]

# Step 1: Load Pre-trained CNN
base_model = ResNet50(weights='imagenet')
# We'll use the output of the layer just before the final dense layer (usually named 'avg_pool' for ResNet)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

# Step 2: Preprocess the Image
def preprocess_image_pillow(img_path):
    img = PILImage.open(img_path)
    img = img.resize((224, 224))  # Resize image to 224x224
    img_array = np.array(img)
    
    # If the image has an alpha channel, we should remove it
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    # Convert the image array to float and rescale it
    img_array = img_array.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # correct preprocess_input function for the model

    return img_array

# Step 3: Extract Features
def extract_features(img_path, model):
    preprocessed_image = preprocess_image_pillow(img_path)
    features = model.predict(preprocessed_image)
    flattened_features = features.flatten()  # Flatten the features to a 1D array
    return flattened_features

# Step 4: Normalize Features
def normalize_features(features):
    # Normalize feature vector (L2 norm)
    normalized_features = features / np.linalg.norm(features)
    return normalized_features

# Step 5: Reduce Dimensionality
def reduce_dimensionality(features, n_components=300):
    
    pca = PCA(n_components=n_components)
    features = np.array(features)
    pca.fit(features)
    reduced_features = pca.transform(features)
    return reduced_features

# Function to calculate cosine similarity
def cosine_similarity(vec_a, vec_b):
    similarity = np.dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))
    return similarity


def get_vector_from_image_url(image_url)->List:
    # Ensure the directory for uploaded images exists
    os.makedirs('uploaded_images', exist_ok=True)
    
    # Download the image
    filename = os.path.join('uploaded_images', image_url.split('/')[-1])
    urlretrieve(image_url, filename)
    
    # Extract and normalize features
    features = extract_features(filename, model)
    normalized_features = normalize_features(features)
    
    return normalized_features.tolist()


def product_concat(img_vector:List, desc:str)->str:
    return str(img_vector) + str(desc)

def get_similar_user_product_pool(user:User):
    """returns list of similar user id's"""
    global user_vector_retriever
    user_masked_vector = user.user_masked_vector
    result:List[tuple] = user_vector_retriever.similarity_search(user_masked_vector, k=3, vector_type="user") # (user_id, purchase_history)
    product_pool = []
    for user_id in result:
        for product in user_id[1]:
            product_pool.append(product) # extract purchase history from each user and add to product pool

    return product_pool # list of products



def find_similar_products_from_query(query:str, product_pool:List[Product]):
    temporary_persist_directory = r"C:\Users\ayhan\Desktop\Smart-Shopper-Recommendation-Engine\artifact\temporary_persist_directory"
    os.makedirs(temporary_persist_directory, exist_ok=True)
    collection_name= "temporary"
    #generate documents from product_pool or iterate thru and perform cosine similarity but we dont have a product as query, just the description.
    json_to_document = JsonToDocument()
    documents = json_to_document.process_json_document(product_pool)

    vector_retriever = VectorRetriever(model_name = model_name, model_kwargs= model_kwargs, encode_kwargs=encode_kwargs, overwrite=True)
    vector_retriever.initialize_vector_store(persist_directory=temporary_persist_directory, documents=documents, collection_name=collection_name)

    similar_products = vector_retriever.similarity_search(query, k=3, vector_type="product")
    return similar_products


def find_similar_products(product:Product):
    """returns list of similar product id's"""
    global product_vector_retriever
    img_vector = get_vector_from_image_url(product.image_url)
    combined_vector = product_concat(img_vector, product.description)
    result:List[tuple] = product_vector_retriever.similarity_search(combined_vector, k=3, vector_type="product") # (product_id, description)
    return result 



# Endpoint for case 1
@app.post("/case1/")
async def case1_endpoint(product: Product):
    if not product.image_url:
        raise HTTPException(status_code=400, detail="Image URL is required.")
    
    # Process image to vector
    img_vector = process_image_to_vector(product.image_url)
    
    # Concatenate img_vector and description
    combined_vector = concatenate_vectors(img_vector, product.description)
    
    # Search for similarity in vector store
    similar_products = find_similar_products(combined_vector)
    
    return {"similar_products": similar_products}

# Endpoint for case 2
@app.post("/case2/")
async def case2_endpoint(user:User,  user_request_product:str):
    # Find most similar users
    product_pool = get_similar_user_product_pool(user)
    similar_products = find_similar_products_from_query(user_request_product, product_pool)
    
    return {"similar_products": similar_products}




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)