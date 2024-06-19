# Anime Recommendation System (Honours Project)
A system for recommending anime focused on improving recommendations for new users by creating a hybrid recommendation system. It uses MyAnimeList for the user ratings data and for the anime data.

## Running the Project
This project requires Python. Python 3.10 is recommended since there is an issue with the way the Surprise package is setup. There is a file called requirements.txt which can be used to install all the required Python dependencies.

### Getting the Required Datasets
#### Downloading the Data Files
Downloaded files should go inside the data folder.

You can download the data from here: https://drive.google.com/drive/folders/1byfM21Q65Mn5gb2SVcutu_z-MspwJbr3, or you could download the appropriate files using apiDataCollector.py and jsonConverter.py. You also need to download the rating_complete.csv file from the Kaggle Dataset at: https://drive.google.com/drive/u/1/folders/1VHbxxhSLdK_g7ro7-3a7ntgL1YXdp-WT.

#### Getting the Data Yourself (Optional)
If you want, you can manually collect MyAnimeList data using the python files provided (apiDataCollector.py and jsonConverter.py). You will need a .env file with your MyAnimeList API Client ID:
```.env
CLIENT_ID=<MY ANIME LIST API CLIENT ID>
```

## Using the Streamlit Web App
Run the following commands to start the web app:
```
cd web-app
streamlit run main_page.py
``` 
You will need to run the RecommendationSystems Jupyter Notebook at least once so the relevant files are saved to the data folder before running the streamlit app.
