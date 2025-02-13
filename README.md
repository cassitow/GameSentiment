# GameSentiment: Analyzing Player Emotions and Feedback

## Project Overview
This project aims to classify sentiments in social media posts related to video games using Natural Language Processing (NLP) and machine learning techniques. The dataset, consisting of tweets labeled with sentiments (positive, negative, neutral, irrelevant), was sourced from Twitter and preprocessed for analysis. The goal is to analyze these tweets to detect sentiment and understand the relationship between sentiment and the respective game brands mentioned in the posts. Various models, including Feedforward Neural Networks (FFNN), Gated Recurrent Units (GRU), and Bidirectional Long Short-Term Memory Networks (BiLSTM), were used to predict sentiment based on the cleaned and processed textual data.

## Data Preprocessing
Data preprocessing is essential to prepare raw text for machine learning models. The main steps involved:

### Text Cleaning:
- **Expanded contractions:** e.g., "won't" -> "will not".
- **Removed URLs and special characters** to focus on relevant content.
- **Normalized punctuation** (e.g., "!!" -> "!") for consistency.
- **Lowercased text** to reduce vocabulary complexity.
- **Removed emojis** to maintain focus on textual sentiment.

### SpaCy Preprocessing:
- **Lemmatization**: Words were converted to their base form to reduce vocabulary size.
- **Stopwords and punctuation removal**: Removed common words like "the", "and", etc., which do not contribute to sentiment analysis.

## Model Development

### 1. Feedforward Neural Networks (FFNN)
FFNN was trained as the initial model. Key components:
- **Embedding layer**: Converts words into dense vectors.
- **Flatten layer**: Transforms the 2D embedding output into a 1D vector.
- **Dense layers**: Fully connected layers with softmax activation for multi-class classification.

#### Results:
- **Best accuracy with Adam**: 94.62% (training), 84.58% (validation).
- **Best accuracy with SGD**: 32.74% (training), 32.63% (validation).
- **Best accuracy with Adagrad**: 31.96% (training), 31.36% (validation).

### 2. Gated Recurrent Unit (GRU)
GRU was designed to handle sequential data, capturing long-term dependencies:
- **Embedding layer**: Similar to FFNN.
- **GRU layer**: Captures sequential dependencies in text.
- **Dense output layer**: Sentiment classification.

#### Results:
- **Best accuracy with Adam**: 31.20% (training), 31.35% (validation).
- **Best accuracy with SGD**: 31.27% (training), 31.35% (validation).

### 3. Bidirectional Long Short-Term Memory (BiLSTM)
BiLSTM was implemented to handle sequence modeling more effectively:
- **Bidirectional LSTM layer**: Considers both past and future context.
- **Dense output layer**: Final sentiment classification.

#### Results:
- **Best accuracy with Adam**: 91.23% (training), 84.19% (validation).
- **Best accuracy with SGD**: 32.29% (training), 32.17% (validation).

## Model Comparison and Insights

### Key Insights:
- **Best Performance**: FFNN with Adam achieved 84.58% validation accuracy.
- **Underperformance**: GRU and BiLSTM struggled to match FFNN's performance, especially in validation accuracy.
- **Overfitting**: FFNN and BiLSTM showed significant overfitting, with training accuracy much higher than validation accuracy.

### Optimizer Comparison:
- **Adam** was the most effective optimizer, outperforming SGD and Adagrad in all models.
- **SGD** consistently underperformed, indicating it's not suitable for this NLP task.

## Future Directions
To address the overfitting issues and improve model performance, the following strategies will be considered:
- **Regularization techniques**: Using dropout layers or L2 regularization.
- **Hyperparameter tuning**: Adjusting layers, units, and learning rates for better generalization.
- **Data augmentation**: Paraphrasing or adding noise to the data to create more training samples.

## Conclusion
This project emphasizes the importance of data preprocessing and model selection in sentiment analysis. The FFNN with Adam optimizer showed the best results, although further improvements such as regularization techniques and model architecture adjustments are needed for better generalization and performance.
