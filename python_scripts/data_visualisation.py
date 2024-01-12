import matplotlib.pyplot as plt
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim
import seaborn as sns
from wordcloud import WordCloud


class DataVisualisation:

    '''
    This class is used to create visualisations of the data.
    '''

    def word_cloud(self, DataFrame: pd.DataFrame, column_name: str):

        '''
        This method is used to generate a word cloud of a column within a dataframe.

        Parameters:
            DataFrame (pd.DataFrame): The dataframe to which this method is applied.
            column_name (str): The column to which this method is applied.

        Returns:
            PIL.Image.Image: An image object representing the word cloud.
        '''

        # Join the different reviews together.
        long_string = ','.join(list(DataFrame[column_name].values))
        # Create a WordCloud object.
        wordcloud = WordCloud(width=800, height=400, background_color="white", max_words=100, contour_width=3, contour_color='steelblue', colormap='seismic')
        # Generate a word cloud.
        wordcloud.generate(long_string)
        # Save the word cloud to a png file.
        wordcloud.to_file("visuals/wordcloud_image.png")
        # Visualise the word cloud.
        return wordcloud.to_image()
    
    def lda_visual(self, lda_model, corpus, id2word, mds="mmds", R=30):
        
        '''
        This method is used to generate a visual of the Latent Dirichlet Allocation (LDA) model topic clustering.

        Parameters:
            lda_model (gensim.models.ldamodel.LdaModel): The gensim LDA model for which the visual will be created.
            corpus (list): The list of bag-of-words representations for each document.
            id2word (gensim.corpora.dictionary.Dictionary): The dictionary mapping word IDs to words for the bigrams and trigrams data.
            mds (str, optional): The multi-dimensional scaling (MDS) method for visualizing topic distances. Default is "mmds".
            R (int, optional): The number of dimensions for MDS. Default is 30.

        Returns:
            pyLDAvis.PreparedData: A PyLDAvis PreparedData object representing the LDA model visualization.
        ''' 

        pyLDAvis.enable_notebook() # Enable notebook mode for PyLDAvis.
        visual = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds=mds, R=R) # Generate LDA Model visualisation.
        pyLDAvis.save_html(visual, 'visuals/lda.html') # Save html.
        return visual
    
    def sentiment_scores(self, DataFrame: pd.DataFrame, column_names: list):

        '''
        This method generates a visual representation of sentiment scores distribution for a DataFrame.

        Parameters:
            DataFrame (pd.DataFrame): The input DataFrame containing columns with sentiment scores.
            column_names (list): A list of column names in the DataFrame representing sentiment scores.
        '''

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6)) # Creating 1x3 grid.

        sns.histplot(DataFrame[column_names[0]], color='darkred', ax=axes[0]) # Assigning negative score to first plot.
        sns.histplot(DataFrame[column_names[1]], color='darkblue', ax=axes[1]) # Assigning neutral score to second plot.
        sns.histplot(DataFrame[column_names[2]], color='darkgreen', ax=axes[2]) # Assigning positive score to third plot.

        # Setting subplot titles:
        for i in range(3):
            axes[i].set_title(column_names[i])
        
        # Setting main plot title:
        plt.suptitle('Sentiment Scores', fontsize=16, y=1.02)
        
        plt.savefig('visuals/sentiment_scores.png') # Saving the visual as a .png.

        return plt.show()

    def histogram(self, DataFrame: pd.DataFrame, column_name: str):

        '''
        This method generates a histogram plot for a specified column in a DataFrame.

        Parameters:
            DataFrame (pd.DataFrame): The input DataFrame containing the specified column.
            column_name (str): The name of the column for which the histogram will be generated.
        '''

        sns.histplot(DataFrame[column_name], color='darkblue') # Generating the histogram plot.
        plt.title(column_name) # Setting title of plot.
        plt.savefig('visuals/compound_score.png') # Saving the visual as a .png.
        return plt.show()

    def sentiment_analysis(self, DataFrame: pd.DataFrame):

        '''
        This method generates a visual representation of overall and specific sentiment scores distribution.

        Parameters:
            DataFrame (pd.DataFrame): The input DataFrame containing sentiment-related columns.
        '''

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6)) # Creating 1x2 grid.

        # Set sub plot titles:
        axes[0].set_title('Overall Sentiment')
        axes[1].set_title('Specific Sentiment')

        color_shades = ['#00008B', '#0000CD', '#4169E1', '#ADD8E6', '#FF84A1', '#ff3333', '#B30000'] # Setting the pie chart colour scheme.

        overall_probabilities = DataFrame['sentiment'].value_counts(normalize=True) # Get the normalised value counts of all the overall sentimentssentiments.
        axes[0].pie(list(overall_probabilities.values), labels=list(overall_probabilities.index), colors=color_shades, autopct='%1.1f%%', startangle=180) # Generate pie chart.
        specific_probabilities = DataFrame['specific sentiment'].value_counts(normalize=True) # Get the normalised value counts of all the specific sentiments.
        sns.barplot(y=list(specific_probabilities.index), x=list(specific_probabilities.values), palette=reversed(color_shades), ax=axes[1]) # Generating the bar plot. 

        # Setting main plot title:
        plt.suptitle('Sentiment Analysis', fontsize=16, y=1.02)

        plt.savefig('visuals/sentiment_analysis.png') # Saving the visual as a .png.

        return plt.show() 