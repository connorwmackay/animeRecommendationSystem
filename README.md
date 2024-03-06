# animeRecommendationSystem
My RGU Honours Project: A system for recommending anime, making use of the MyAnimeList API.

If you want to collect the data using my python files, you will need a .env file with your MyAnimeList API Client ID:
```.env
CLIENT_ID=<MY ANIME LIST API CLIENT ID>
```

## Downloading the Data Files
Downloaded files should go inside the data folder.

You can download the data from here: https://drive.google.com/drive/folders/1byfM21Q65Mn5gb2SVcutu_z-MspwJbr3, or you could download the appropriate files using apiDataCollector.py and jsonConverter.py. You also need to download the rating_complete.csv file from the Kaggle Dataset at: https://drive.google.com/drive/u/1/folders/1VHbxxhSLdK_g7ro7-3a7ntgL1YXdp-WT.

## Using the Streamlit App
Run `streamlit run recommendation_systems_web.py` to access the streamlit app. You will need to run the RecommendationSystems Jupyter Notebook at least once so the relevant files are saved to the data folder before running the streamlit app.