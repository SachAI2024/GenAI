# from transformers import pipeline

# #Load the feature extrastion pipeline and model

# pipe = pipeline('feature-extraction', model = 'DarkWolf/kn-electra-small'
#                 ,tokenizer = 'DarkWolf/kn-electra-small')

# #Input sentence
# sentence = """ 
# AI is the future of technology. AI will change the way we think, AI will encompass a range of technologies
# including machine learning, natural language processing, and computer vision. AI will be used to develop new products and services
# that will revolutionize the way we live and work. AI will also have a significant impact on the global economy. AI will create new
# jobs and industries, and will drive innovation and growth in existing industries. AI will be a key driver of economic growth in the
# """
# #Get embeddings
# embedding = pipe(sentence)
# #The output is a list with one item per sentence
# sentence_embedding = embedding[0][0][0:10]
# print(sentence_embedding)

from transformers import pipeline

# Load the feature extraction pipeline and model
pipe = pipeline('feature-extraction',model='DarkWolf/kn-electra-small',tokenizer='DarkWolf/kn-electra-small')

# Input sentence
sentence = """
AI encompasses a range of technologies, including machine learning, 
natural language processing, robotics, and more.
"""
# Get embeddings
embedding = pipe(sentence)

# The output is a list with one item per sentence
sentence_embeddings = embedding[0][0][0:10]
print(sentence_embeddings)


#To make this one working I have to install pip3 install tensorflow==2.12.0 , pip3 install torch , pip3 install transformers