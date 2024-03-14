# #
# import wikipedia
# from keras_vggface.utils import preprocess_input
# from keras_vggface.vggface import VGGFace
# import pickle
# from sklearn.metrics.pairwise import cosine_similarity
# import streamlit as st
# from PIL import Image
# import os
# import cv2
# from mtcnn import MTCNN
# import numpy as np
#
# detector = MTCNN()
# model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
# feature_list = pickle.load(open('embedding.pkl','rb'))
# filenames = pickle.load(open('filenames.pkl','rb'))
#
# def save_uploaded_image(uploaded_image):
#     try:
#         with open(os.path.join('uploads',uploaded_image.name),'wb') as f:
#             f.write(uploaded_image.getbuffer())
#         return True
#     except:
#         return False
#
# def extract_features(img_path,model,detector):
#     img = cv2.imread(img_path)
#     results = detector.detect_faces(img)
#
#     x, y, width, height = results[0]['box']
#
#     face = img[y:y + height, x:x + width]
#
#     #  extract its features
#     image = Image.fromarray(face)
#     image = image.resize((224, 224))
#
#     face_array = np.asarray(image)
#
#     face_array = face_array.astype('float32')
#
#     expanded_img = np.expand_dims(face_array, axis=0)
#     preprocessed_img = preprocess_input(expanded_img)
#     result = model.predict(preprocessed_img).flatten()
#     return result
# def search_wikipedia(query):
#     try:
#         # Search for the query on Wikipedia
#         search_results = wikipedia.search(query)
#
#         # Fetch the summary of the first search result
#         if search_results:
#             first_result = search_results[0]
#             summary = wikipedia.summary(first_result)
#             return summary
#         else:
#             return "No matching article found on Wikipedia."
#     except wikipedia.exceptions.DisambiguationError as e:
#         # Handle disambiguation pages
#         return f"Disambiguation page encountered. Options: {', '.join(e.options)}"
#     except wikipedia.exceptions.PageError:
#         # Handle page not found
#         return "No matching article found on Wikipedia."
# def recommend(feature_list,features):
#     similarity = []
#     for i in range(len(feature_list)):
#         similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])
#
#     index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
#     return index_pos
#
# st.title('Know More About the Artifacts?')
#
# uploaded_image = st.file_uploader('Choose an image')
#
# if uploaded_image is not None:
#     # save the image in a directory
#     if save_uploaded_image(uploaded_image):
#         # load the image
#         display_image = Image.open(uploaded_image)
#
#         # extract the features
#         features = extract_features(os.path.join('uploads',uploaded_image.name),model,detector)
#         # recommend
#         index_pos = recommend(feature_list,features)
#         predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
#         # display
#         col1,col2 = st.columns(2)
#
#         with col1:
#             st.header('Your uploaded image')
#             st.image(display_image)
#         with col2:
#             st.header("Seems like " + predicted_actor)
#             st.image(filenames[index_pos],width=300)
#         result = search_wikipedia(predicted_actor)
#         st.header("Infomation of "+ predicted_actor)
#         st.write(result)
#
# def main():
#         # Text input field for query
#     if st.button("Search"):
#         if query:
#             result = search_wikipedia(query)
#             st.header("Wikipedia Summary:")
#             st.write(result)
#         else:
#             st.write("Please enter a query.")
#
#
#
#
#
#
# # Function to search Wikipedia
#
#
#
# # Streamlit UI
# def main():
#     # Text input field for query
#     query = st.text_input("Enter your query:", "")
#
#     # Search button
#     if st.button("Search"):
#         if query:
#             result = search_wikipedia(query)
#             st.write(result)
#         else:
#             st.write("Please enter a query.")
#
# if __name__ == "__main__":
#     main()
#
#
#
#
#
# import wikipedia
# from keras_vggface.utils import preprocess_input
# from keras_vggface.vggface import VGGFace
# import pickle
# from sklearn.metrics.pairwise import cosine_similarity
# import streamlit as st
# from PIL import Image
# import os
# import cv2
# from mtcnn import MTCNN
# import numpy as np
# from gtts import gTTS
#
# language = 'en'
#
# def save_audio(mytext):
#     myobj = gTTS(text=mytext, lang=language, slow=False)
#     myobj.save("VOICE.mp3")
#
#     os.system("start VOICE.mp3")
#
#
#
# detector = MTCNN()
# model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
# feature_list = pickle.load(open('embedding.pkl','rb'))
# filenames = pickle.load(open('filenames.pkl','rb'))
#
# def save_uploaded_image(uploaded_image):
#     try:
#         with open(os.path.join('uploads',uploaded_image.name),'wb') as f:
#             f.write(uploaded_image.getbuffer())
#         return True
#     except:
#         return False
#
# def extract_features(img_path,model,detector):
#     img = cv2.imread(img_path)
#     results = detector.detect_faces(img)
#
#     x, y, width, height = results[0]['box']
#
#     face = img[y:y + height, x:x + width]
#
#     #  extract its features
#     image = Image.fromarray(face)
#     image = image.resize((224, 224))
#
#     face_array = np.asarray(image)
#
#     face_array = face_array.astype('float32')
#
#     expanded_img = np.expand_dims(face_array, axis=0)
#     preprocessed_img = preprocess_input(expanded_img)
#     result = model.predict(preprocessed_img).flatten()
#     return result
#
# def search_wikipedia(query):
#     try:
#         # Search for the query on Wikipedia
#         search_results = wikipedia.search(query)
#
#         # Fetch the summary of the first search result
#         if search_results:
#             first_result = search_results[0]
#             summary = wikipedia.summary(first_result)
#             return summary
#         else:
#             return "No matching article found on Wikipedia."
#     except wikipedia.exceptions.DisambiguationError as e:
#         # Handle disambiguation pages
#         return f"Disambiguation page encountered. Options: {', '.join(e.options)}"
#     except wikipedia.exceptions.PageError:
#         # Handle page not found
#         return "No matching article found on Wikipedia."
#
# def recommend(feature_list,features):
#     similarity = []
#     for i in range(len(feature_list)):
#         similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])
#
#     index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
#     return index_pos
#
# st.title('Know More About the Artifacts?')
#
# uploaded_image = st.file_uploader('Choose an image')
#
# if uploaded_image is not None:
#     # save the image in a directory
#     if save_uploaded_image(uploaded_image):
#         # load the image
#         display_image = Image.open(uploaded_image)
#
#         # extract the features
#         features = extract_features(os.path.join('uploads',uploaded_image.name),model,detector)
#         # recommend
#         index_pos = recommend(feature_list,features)
#         predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
#         # display
#         col1,col2 = st.columns(2)
#
#         with col1:
#             st.header('Your uploaded image')
#             st.image(display_image)
#         with col2:
#             st.header("Seems like " + predicted_actor)
#             st.image(filenames[index_pos],width=300)
#         result = search_wikipedia(predicted_actor)
#         st.header("Infomation of "+ predicted_actor)
#         st.write(result)
#         save_audio(result)
#
# # Text input field for query
# query = st.text_input("Enter your query:", "")
#
# # Search button
# if st.button("Search"):
#     if query:
#         result = search_wikipedia(query)
#         st.write(result)
#         save_audio(result)
#     else:
#         st.write("Please enter a query.")






import wikipedia
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np
from gtts import gTTS

language = 'en'

def save_audio(mytext):
    myobj = gTTS(text=mytext, lang=language, slow=False)
    myobj.save("VOICE.mp3")

    return "VOICE.mp3"

detector = MTCNN()
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
feature_list = pickle.load(open('embedding.pkl','rb'))
filenames = pickle.load(open('filenames.pkl','rb'))

def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads',uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

def extract_features(img_path,model,detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    x, y, width, height = results[0]['box']

    face = img[y:y + height, x:x + width]

    #  extract its features
    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)

    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

def search_wikipedia(query, max_words=200):
    try:
        # Search for the query on Wikipedia
        search_results = wikipedia.search(query)

        # Fetch the summary of the first search result
        if search_results:
            first_result = search_results[0]
            summary = wikipedia.summary(first_result)

            # Truncate the summary to the maximum number of words
            words = summary.split()
            truncated_summary = ' '.join(words[:max_words])

            return truncated_summary
        else:
            return "No matching article found on Wikipedia."
    except wikipedia.exceptions.DisambiguationError as e:
        # Handle disambiguation pages
        return f"Disambiguation page encountered. Options: {', '.join(e.options)}"
    except wikipedia.exceptions.PageError:
        # Handle page not found
        return "No matching article found on Wikipedia."


def recommend(feature_list,features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos


# Add logo at the top-left corner
logo_image_path = "Artifact jones.png"
st.image(logo_image_path, width=100)


st.title('Know More About the Artifacts..')
st.text("")
label = r'''
$\textsf{
    \large Choose an image 
}$
'''

uploaded_image = st.file_uploader(label)

if uploaded_image is not None:
    # save the image in a directory
    if save_uploaded_image(uploaded_image):
        # load the image
        display_image = Image.open(uploaded_image)

        # extract the features
        features = extract_features(os.path.join('uploads',uploaded_image.name),model,detector)
        # recommend
        index_pos = recommend(feature_list,features)
        predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
        # display
        col1,col2 = st.columns(2)

        with col1:
            st.header('Your uploaded image')
            st.image(display_image)
        with col2:
            st.header("Seems like " + predicted_actor)
            st.image(filenames[index_pos],width=300)
        result = search_wikipedia(predicted_actor)
        st.header("Infomation of "+ predicted_actor)
        st.write(result)
        audio_file = save_audio(result)
        st.audio(audio_file, format='audio/mp3', start_time=0)

# Text input field for query
# query =st.text_area("Enter your query:", "")

label = r'''
$\textsf{
    \large Enter your query: 
}$
'''
query = st.text_input(label, "")


# Search button
if st.button("Search"):
    if query:
        result = search_wikipedia(query)
        st.write(result)
        audio_file = save_audio(result)
        st.audio(audio_file, format='audio/mp3', start_time=0)
    else:
        st.write("Please enter a query.")
