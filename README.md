# Twitter Sentiment Analysis on Sentiment140 Dataset

A comprehensive NLP project for classifying Twitter sentiments using multiple machine learning approaches, including Logistic Regression and LSTM neural networks.

## Project Overview

This project performs sentiment analysis on tweets from the Sentiment140 dataset. The goal is to classify tweets as positive or negative sentiment using various NLP techniques and machine learning models.

## Dataset

- **Twitter_data.csv** (1.6M tweets): Raw dataset containing tweets with sentiment labels
- **cleaned_twitter_data.csv** (1.59M tweets): Preprocessed dataset with cleaned text
- **Tweet Sentimental Data.csv** (1,974 tweets): Test/validation subset

### Data Structure

Each record contains:
- `target`: Sentiment label (0 = Negative, 1 = Positive)
- `text`: Original tweet text
- `cleaned_text`: Preprocessed text (after tokenization, stemming, stopword removal)

## Project Files

### Notebooks

1. **data_preprocessing.ipynb**
   - Loads raw Twitter data
   - Performs text preprocessing: tokenization, stemming, stopword removal
   - Generates cleaned_twitter_data.csv
   - Uses NLTK for NLP operations

2. **lstm_train.ipynb**
   - LSTM neural network model implementation
   - Uses Keras/TensorFlow with bidirectional LSTM layers
   - Achieves **86% training accuracy** and **81% validation accuracy**
   - Output model: lstm_sentiment_model_train-accu-86_val-acc-81.h5

3. **train_lr.ipynb**
   - Logistic Regression baseline model
   - Traditional machine learning approach for comparison

4. **train.ipynb**
   - Initial training experiments
   - Development notebook

### Models

- **lstm_sentiment_model_train-accu-86_val-acc-81.h5**: Pre-trained LSTM model with best performance metrics

## Technologies & Libraries

- **Data Processing**: Pandas, NumPy
- **NLP**: NLTK (Natural Language Toolkit)
- **Deep Learning**: TensorFlow, Keras
- **Machine Learning**: Scikit-learn
- **Progress Tracking**: tqdm

## Key Features

- Text preprocessing pipeline with stemming and stopword removal
- LSTM-based deep learning model with bidirectional layers
- Dropout for regularization
- Word embedding layer for semantic representation
- Comparison with traditional ML approaches (Logistic Regression)

## Model Performance

**Best Model: Bidirectional LSTM**
- Training Accuracy: 86%
- Validation Accuracy: 81%

## Usage

### 1. Data Preprocessing
Run `data_preprocessing.ipynb` to clean and preprocess the raw Twitter data:
```
jupyter notebook data_preprocessing.ipynb
```

### 2. Model Training
Run `lstm_train.ipynb` to train the LSTM model:
```
jupyter notebook lstm_train.ipynb
```

### 3. Baseline Comparison
Run `train_lr.ipynb` to train and evaluate logistic regression:
```
jupyter notebook train_lr.ipynb
```

## Project Workflow

1. **Data Loading** → Load raw tweets from Twitter_data.csv
2. **Preprocessing** → Clean text, remove stopwords, apply stemming
3. **Tokenization** → Convert text to sequences for neural networks
4. **Model Training** → Train LSTM model on processed data
5. **Evaluation** → Assess model performance on validation set
6. **Comparison** → Compare with baseline models

## Requirements

Install required packages:
```bash
pip install pandas numpy nltk tensorflow keras scikit-learn tqdm
```

Download NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## Notes

- The dataset is balanced with positive and negative sentiments
- Preprocessing significantly improves model performance
- The LSTM model outperforms traditional ML baselines
- Consider ensemble methods or transfer learning for further improvements
