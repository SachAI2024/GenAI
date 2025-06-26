# Learn & Fun: Sentiment Analysis Demo

This short exercise introduces transformers by walking through a simple sentiment analysis example. You'll run a Jupyter notebook that classifies text as positive, neutral, or negative.

## Quick Start
1. Launch the provided notebook by clicking **Launch App**.
2. Wait for the kernel to show `Python 3 (ipykernel)` in the top-right corner.
3. Execute each cell in order.
4. Enter your own sentence when prompted. The notebook uses `pipeline("text-classification")` to predict the sentiment.
5. Experiment by changing the text or model parameters to see different results.

## Quiz
1. An online platform automatically tags uploaded articles as Sports, Politics, or Fashion. What technology is being used here?
   - A. Text summarization
   - B. Entity extraction
   - C. Text classification
   - D. Speech-to-text conversion

2. An online platform extracts names, dates, and locations from news invoices. What technology is being used here?
   - A. Entity extraction
   - B. Text classification
   - C. Text-to-speech conversion
   - D. Machine translation

3. Which statement is true for the transformer’s encoder-decoder architecture?
   - A. The encoder maps the input text to a vector representation, and the decoder translates this into an output sequence.
   - B. The encoder outputs the final prediction, and the decoder provides a probability distribution over the next possible word.
   - C. The encoder is only used during training, while the decoder is used during inference.
   - D. The decoder operates independently of the encoder and does not utilize its outputs.

4. What is the purpose of tokenization in the context of natural language processing and transformers?
   - A. To add semantic information to the tokens
   - B. To add the position of each token in the text
   - C. To divide the text into a list of tokens or words
   - D. To convert the tokens in the text to vectors

   **Tokenization is the process of converting text into words or tokens.**

5. What does embedding represent in a transformer model?
   - A. The frequency of each token in the text
   - B. The grammatical category of each token
   - C. The position of each token in the sentence
   - D. High-dimensional space representations of tokens that the model can interpret and process

6. Why is positional encoding added to embeddings in a transformer model?
   - A. To give the model information about the sequence order of the tokens
   - B. To reduce the dimensionality of the embeddings
   - C. To ensure all vectors have the same length
   - D. To increase the sentence complexity for better training
