# British Airways Data Science Project
By **Joel Sud**
## Table of Contents:
- [Description](#description)
- [Installation Instructions](#installation_instructions)
- [Usage Instructions](#usage_instructions)
- [File Structure](#file_structure)
    - [File Description](#understanding-the-files)
- [Project Documentation:](#project-documentation)

## Description: 
A British Airways Data Science Project analysing customer reviews using data scraping and Machine Learning modelling techniques.

## Installation Instructions:
1. **Download and clone repository:**
- copy the repository URL by clicking '*<> Code*' above the list of files in GitHub Repo. Then copy and paste the '*HTTPS*' URL:
- in your CLI go to the location where you wish to clone your directory.
- Type the following '***git clone***' command with the '*HTTPS*' URL:

***<p style="text-align: center;">git clone https://github.com/joelsud18/British_Airways_Data_Science_Project.git***</p>

- Press 'Enter'.

2. **Ensure there is the '*environment.yaml*' file.** This will be used to clone the conda environment with all the packages and versions needed to run the code in this repository. Using conda on the CLI on your machine write the following command:

***<p style="text-align: center;">conda env create -f environment.yml***
</p>
    
- you can add the ***--name*** flag to give a name to this environment.

## Usage Instructions

1. Ensure the repository has been cloned and all the files in the [File Structure](#file_structure) are present.
2. 

## File Structure:
- data
  - BA_reviews.csv
- python_scripts
  - data_transformation.py
  - data_visualisation.py
- visuals
  - compound_score.png
  - lda.html
  - sentiment_analysis.png
  - sentiment_score.png
  - wordcloud_image.png
- data_science_task.ipynb
- data_scraping.ipynb
- Presenting_Insights.pptx

### Understanding the Files:
- **data**: This folder contains the raw data in .csv format.
- **data_transformation.py**: This script defines the class methods used for data cleaning and analysis transformations to the dataframe.
- **data_visualisation.py**: This script defines the class methods used for visualising data in the dataframe.
- **visuals**: This folder contains data visualisation.
- **lda.html**: This is a html interactive visualisation of the LDA topic modelling.
- **data_science_task.ipynb**: This interactive notebook contains the data cleaning and analysis of the scraped customer reviews including data cleaning, word cloud EDA, topic modelling and sentiment analysis.
- **data_scraping.ipynb**: This interactive notebook scrapes data using the BeautifulSoup library from a website and loads it into a .csv file.
- **Presenting_Insights.pptx**: This is a powerpoint presentation showcasing the key insights and visualisations from the *data_science_task.ipynb* analysis.
