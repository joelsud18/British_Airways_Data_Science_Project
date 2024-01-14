import gensim
from gensim import corpora
from gensim.models import CoherenceModel
from gensim.models import TfidfModel
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import pycountry_convert as pc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class DataCleaning:

    '''
    This class is used to apply methods to clean the data.
    '''

    def replace_characters(self, DataFrame: pd.DataFrame, column_name: list, characters: list):

        '''
        This method is used to replace specified characters in a column of a DataFrame.

        Parameters:
            DataFrame (pd.DataFrame): The pandas DataFrame containing the data.
            column_name (str): The name of the column in which characters will be replaced.
            characters (list): The list of characters to be replaced in the specified column.

        Returns:
            pd.DataFrame: A modified DataFrame with the specified characters replaced in the specified column.
        '''

        def clean_text(text):

            '''
            Helper function to replace specified characters in a text.

            Parameters:
                text (str): The input text to be cleaned.

            Returns:
                str: The cleaned text with specified characters replaced.
            '''

            for string in characters: # For each unnecesary character.
                text = text.replace(string, '') # Remove the string.
            return text
        
        DataFrame.loc[:, column_name] = DataFrame[column_name].apply(clean_text) # For each row in the column, apply the text cleaning.
        return DataFrame

class DataAnalysis:

    '''
    This class is used to apply transformations required for analysis of data in a dataframe.
    '''

    def tokenise_data(self, DataFrame: pd.DataFrame, column_name: str, ):

        '''
        This method is used to tokenise text data in a specified column of a DataFrame, removing English stopwords.

        Parameters:
            DataFrame (pd.DataFrame): The pandas DataFrame containing the data.
            column_name (str): The name of the column containing the text data to be tokenised.

        Returns:
            list of list of str: A list of lists, where each inner list represents a document and contains tokenised words.
        '''

        # Save english stopwords into a variable:
        nltk.download('stopwords')
        stop_words = set(stopwords.words("english"))

        def gen_words(text):
            return gensim.utils.simple_preprocess(text) # tokenise the input text into a list of lowercase words.

        data_words = DataFrame[column_name].apply(gen_words) # Apply tokenisation to all rows in column.
        data_words = [[word for word in words if word not in stop_words] for words in data_words] # Filtering out stop words.
        return data_words

    def bigram_trigram(self, data_words):

        '''
        This method is used to create and apply bigram and trigram phrases to a list of tokenised texts.

        Parameters:
            data_words (list of list of str): A list of lists, where each inner list represents a document and contains tokenised words.

        Returns:
            list of list of str: A list of lists representing tokenised texts after applying both bigram and trigram phrases.
        '''

        bigram_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=50) # Create bigram phrases, which appear more than 5 times.
        trigram_phrases = gensim.models.Phrases(bigram_phrases[data_words], threshold=50) # Create trigram phrases based on bigram phrases.

        # Creating Phraser models for bigram and trigram phrases:
        bigram = gensim.models.phrases.Phraser(bigram_phrases)
        trigram = gensim.models.phrases.Phraser(trigram_phrases)

        # Function to apply bigram phrases to a list of texts:
        def make_bigrams(texts):
            return([bigram[doc] for doc in texts])

        # Function to apply trigram phrases to a list of texts:
        def make_trigrams(texts):
            return ([trigram[bigram[doc]] for doc in texts])

        data_bigrams = make_bigrams(data_words) # Applying bigram phrases to the original tokenised data.
        data_bigrams_trigrams = make_trigrams(data_bigrams) # Applying trigram phrases to the data that has already been processed with bigram phrases.
        return data_bigrams_trigrams

    def tf_idf_removal(self, data_bigrams_trigrams, low_value: float = 0.01):

        '''
        This method is used to remove low TF-IDF (Term Frequency-Inverse Document Frequency) score words and words missing in TF-IDF from a list of tokenised texts.

        Parameters:
            data_bigrams_trigrams (list of list of str): A list of lists, where each inner list represents a document and contains tokenised words.
            low_value (float, optional): The threshold below which words with TF-IDF scores will be considered low-value. Default is 0.01.

        Returns:
            tuple: A tuple containing two elements:
                - id2word (gensim.corpora.dictionary.Dictionary): A dictionary mapping word IDs to words for the bigrams and trigrams data.
                - corpus (list of list of tuple): A list of lists, where each inner list represents a document and contains tuples (word ID, TF-IDF score) for the remaining words.
        '''

        id2word = corpora.Dictionary(data_bigrams_trigrams) # Creating a dictionary mapping word IDs to words for the bigrams and trigrams data.
        texts = data_bigrams_trigrams # Assigning the list of bigrams and trigrams to the variable 'texts'.
        corpus = [id2word.doc2bow(text) for text in texts] # Creating a bag-of-words representation for each document using the dictionary.
        tfidf = TfidfModel(corpus, id2word=id2word) # Creating a TF-IDF model based on the bag-of-words representation

        # Initialising lists to store words with low TF-IDF scores and words missing in TF-IDF.
        words  = []
        words_missing_in_tfidf = []

        for i in range(0, len(corpus)): # Iterating over each document in the corpus.
            bow = corpus[i] # Getting the bag-of-words representation for the current document.
            low_value_words = [] 
            # Getting the word IDs and TF-IDF scores for the current document:
            tfidf_ids = [id for id, value in tfidf[bow]]
            bow_ids = [id for id, value in bow]
            # Identifying words with TF-IDF scores below the threshold:
            low_value_words = [id for id, value in tfidf[bow] if value < low_value]
            drops = low_value_words + words_missing_in_tfidf # Combining low-value words with words missing in TF-IDF.

            for item in drops: # Extracting the actual words corresponding to low-value word IDs.
                words.append(id2word[item])
            words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids] # Identifying words missing in TF-IDF (with TF-IDF score of 0).

            # Creating a new bag-of-words representation excluding low-value words and those missing in TF-IDF:
            new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
            corpus[i] = new_bow 
        
        return id2word, corpus

    def lda_model(self, id2word, corpus, num_topics = 30, alpha = 'auto', chunksize = 100, passes = 10):

        '''
        This method is used to create a Latent Dirichlet Allocation (LDA) model for topic modeling.

        Parameters:
            id2word (gensim.corpora.dictionary.Dictionary): A dictionary mapping word IDs to words for the bigrams and trigrams data.
            corpus (list of list of tuple): A list of lists, where each inner list represents a document and contains tuples (word ID, TF-IDF score) for the remaining words.
            num_topics (int, optional): The number of topics to identify. Default is 30.
            alpha (str or list of float, optional): The parameter controlling the document-topic density. It can be set to 'auto' or as a list of alpha values. Default is 'auto'.
            chunksize (int, optional): Number of documents to be used in each training chunk. Default is 100.
            passes (int, optional): Number of passes through the entire corpus during training. Default is 10.

        Returns:
            gensim.models.ldamodel.LdaModel: A trained LDA model for topic modeling.
        '''

        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=num_topics,
                                        random_state=123,
                                        update_every=1,
                                        chunksize=chunksize,
                                        passes=passes,
                                        alpha=alpha)
        
        return lda_model
    
    def vader_sentiment_analysis(self, DataFrame: pd.DataFrame, column_name: str):


        '''
        This method performs sentiment analysis using the VADER (Valence Aware Dictionary and sEntiment Reasoner) tool.

        Parameters:
            DataFrame (pd.DataFrame): The pandas DataFrame containing the text data for sentiment analysis.
            column_name (str): The name of the DataFrame column containing the text data.

        Dependencies:
            nltk.download('vader_lexicon'): This resource contains a list of words with associated sentiment scores.

        Returns:
            pd.DataFrame: The input DataFrame with additional columns for negative score, neutral score, positive score, and compound score.
        '''

        nltk.download('vader_lexicon') # This resoure contains a list of words with associated sentiment scores.

        sentiment_analyser = SentimentIntensityAnalyzer() # Initialising the sentiment analyser model.

        def get_polarity_scores(text):

            '''
            Helper function to calculate VADER polarity scores for a given text.

            Parameters:
                text (str): The input text for which sentiment scores are calculated.

            Returns:
                pd.Series: A pandas Series containing 'negative score', 'neutral score', 'positive score', and 'compound score'.
            '''
            return pd.Series(sentiment_analyser.polarity_scores(text)) # Return series containing sentiment scores.

        # Add columns with scores for each review to dataframe:
        DataFrameCopy = DataFrame.copy()
        DataFrameCopy[['negative score', 'neutral score', 'positive score', 'compound score']] = None # Create columns.
        DataFrameCopy[['negative score', 'neutral score', 'positive score', 'compound score']] = DataFrame[column_name].apply(get_polarity_scores) # Assign sentiment scores.

        return DataFrameCopy
    
    def determine_overall_sentiment(self, DataFrame: pd.DataFrame):

        '''
        This method determines overall sentiment labels based on compound scores and adds them as a new column to the DataFrame.

        Parameters:
            DataFrame (pd.DataFrame): The input DataFrame containing a column named 'compound score'.

        Returns:
            pd.DataFrame: The DataFrame with an additional column 'sentiment' indicating overall sentiment labels.
        '''

        def sentiment_thresholds(value):
            
            '''
            Assigns overall sentiment labels based on compound scores.

            Parameters:
                value (float): The compound score value.

            Returns:
                str: The overall sentiment label.
            '''

            if value <= -0.05:
                return 'Negative'
            elif value > -0.25 and value < 0.05:
                return 'Neutral'
            elif value >= 0.05:
                return 'Positive'
        
        DataFrame['sentiment'] = DataFrame['compound score'].apply(sentiment_thresholds) # Assign overall sentiment based on sentiment score to new column.
    
        return DataFrame

    def determine_specific_sentiment(self, DataFrame: pd.DataFrame):

        '''
        This method determines specific sentiment labels based on compound scores and adds them as a new column to the DataFrame.

        Parameters:
            DataFrame (pd.DataFrame): The input DataFrame containing a column named 'compound score'.

        Returns:
            pd.DataFrame: The DataFrame with an additional column 'specific sentiment' indicating specific sentiment labels.
        '''

        def specific_sentiment_thresholds(value):

            '''
            Assigns specific sentiment labels based on compound scores.

            Parameters:
                value (float): The compound score value.

            Returns:
                str: The specific sentiment label.
            '''

            if value <= -0.75:
                return 'Very Negative'
            elif value > -0.75 and value <= -0.25:
                return 'Negative'
            elif value > -0.25 and value <= -0.5:
                return 'Slightly Negative'
            elif value > -0.25 and value < 0.05:
                return 'Neutral'
            elif value >= 0.05 and value < 0.25:
                return 'Slightly Positive'
            elif value >= 0.25 and value < 0.75:
                return 'Positive'
            elif value >= 0.75:
                return 'Very Positive'
        
        DataFrame['specific sentiment'] = DataFrame['compound score'].apply(specific_sentiment_thresholds) # Assign specific sentiment based on sentiment score to new column.

class DataProcessing:

    '''
    This class is used to apply ethods to process the data for predictive modelling.
    '''

    def map_weekdays(self, DataFrame: pd.DataFrame, column_name: str):

        '''
        This method is used to map string weekdays to integer codes.

        Parameters:
            DataFrame (pd.DataFrame): The dataframe which contains the column with weekdays.
            column_name (str): The name of the column which contains weekdays.

        Returns:
            pd.DataFrame: The updated dataframe.
        '''

        mapping = {
        "Mon": 1,
        "Tue": 2,
        "Wed": 3,
        "Thu": 4,
        "Fri": 5,
        "Sat": 6,
        "Sun": 7,
        }

        DataFrame[column_name] = DataFrame[column_name].map(mapping)
        return DataFrame
    
    def encode_object_columns(self, DataFrame: pd.DataFrame):

        '''
        This method takes a dataframe and converts all object columns into category codes, to encode string data into numerical format.

        Parameters:
            DataFrame (pd.DataFrame): The dataframe to which this method will be applied.

        Returns: 
            pd.DataFrame: The updated encoded dataframe.

        '''
        
        object_columns = DataFrame.select_dtypes(include=['object']).columns.tolist() # Adds all columns with 'object' as their data type into a list.
        DataFrame[object_columns] = DataFrame[object_columns].astype('category') # Changes the data type of the columns in this list to 'category'.
        DataFrame[object_columns] = DataFrame[object_columns].apply(lambda x: x.cat.codes) # Converts these categories into numerical codes.
        return DataFrame
    
    def country_to_continent_column(self, DataFrame: pd.DataFrame, country_column_name: str):

        '''
        Add a new column to the DataFrame with continent information based on the given country column.

        Parameters:
            DataFrame (pd.DataFrame): The pandas DataFrame containing the data.
            country_column_name (str): The name of the column in DataFrame containing country names.

        Returns:
            pd.DataFrame: A DataFrame with an additional column representing the continent of each country.

        Notes:
            This method relies on the pycountry_convert library to map country names to continents.
            Some special cases are handled explicitly (e.g., 'Myanmar (Burma)', 'Timor-Leste', 'Svalbard & Jan Mayen').
            If a country is not recognized or an error occurs during the mapping process, the original country name is retained.
        '''

        def country_to_continent(country_name):

            '''
            Map a country name to its continent.

            Parameters:
                country_name (str): The name of the country.

            Returns:
                str or None: The continent name or None if the country is not recognized or an error occurs.
            '''

            try:
                if country_name == 'Myanmar (Burma)' or country_name == 'Timor-Leste':
                    return 'Asia'
                elif country_name == 'Svalbard & Jan Mayen':
                    return 'Europe'
                else:
                    country_alpha2 = pc.country_name_to_country_alpha2(country_name)
                    if country_alpha2 is None:
                        return None  # Country not recognized, you can choose to handle it as needed
                    country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
                    country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
                    return country_continent_name
            except Exception as e:
                return country_name

        DataFrame[f"{country_column_name}_continent"] = DataFrame[country_column_name].apply(lambda x: country_to_continent(x))
        return DataFrame