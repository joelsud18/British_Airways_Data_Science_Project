import matplotlib.pyplot as plt
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
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
    
    def confusion_matrix(self, y_test, y_predicted, version_number: int = None):

        '''
        This method returns a confusion matrix comparing the model predicted values to the true values in the data set.

        Parameters:
            y_test: The true output data.
            y_predicted: The model predicted output data.
        '''

        matrix = confusion_matrix(y_test, y_predicted) # Generate confusion matrix.
        plt.figure(figsize=(10,7))
        sns.heatmap(matrix, annot=True, fmt='g', cmap='seismic') # Map confusion matrix onto heatmap.
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        if version_number == None:
            plt.savefig('visuals/confusion_plot.png') # Save the visual.
        else:
            plt.savefig(f'visuals/confusion_plot{version_number}.png') # Save the visual.
        return plt.show()
    
    def feature_importance(self, DataFrame: pd.DataFrame, feature_importance, target_column_name: str):

        '''
        Visualise feature importances using a horizontal bar plot.

        Parameters:
            DataFrame (pd.DataFrame): The input DataFrame containing the features and target column.
            feature_importance: The array or list of feature importances corresponding to each feature.
            target_column_name (str): The name of the target column in the DataFrame.
        '''
        
        # Create a series object of the feature importances and feature names:
        feat_importances = pd.Series(feature_importance, index=DataFrame.drop([target_column_name], axis='columns').columns)
        # Get all the features in order:
        top_features = feat_importances.nlargest(len(DataFrame.drop([target_column_name], axis='columns').columns))

        # Create the bar plot:
        top_features.plot(kind='barh', colormap='brg')

        # Set plot title and axis labels if needed:
        plt.title('Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Features')

        # Save the plot as a PNG file:
        plt.savefig('visuals/feature_importances_plot.png')
        return plt.show()
    
    def box_plot(self, data, title: str, x_label: str):
        
        '''
        Generate and display a box plot for the given data.

        Parameters:
            data (array-like): The data to be visualized in the box plot.
            title (str): The title for the box plot.
            x_label (str): The label for the x-axis.
            version_number (int): The version of the plot being saved.
        '''

        sns.boxplot(x=data, color='red') # Creating the box-plot.
        plt.title(title) # Setting the title.
        plt.xlabel(x_label) # Setting the x_label.
        plt.savefig('visuals/box_plot.png') # Save the visual.
        return plt.show()