# Rutgers---CS439
Final Project for Rutgers CS439 - Intro to Data Science
Scraping and analyzing NASAs Astronomy Picture Of the Day (APOD) page

## data

### databases
raw:
  apod_data.csv - raw data scraped directly from APOD

cleaned:
  apod_labeled_data.csv - apod_data.csv with category labels

Databases.docx - Document with every database used along with a short description and potential use cases

### metadata
apod_labeled_metadata.json - metadata for labeled APOD dataset, used to check if database needs rebuilding
apod_metadata.json - metadata for raw APOD dataset, used to check if database needs rebuilding

### vis
apod_yearly_pie_chart_pages - file containing all pages of category distribution pie charts:
  1 - 1995 to 2003
  2 - 2004 to 2012
  3 - 2013 to 2021
  4 - 2022 to 2026

Confident APOD Category Frequency Heatmap.png - Heatmap showing frequency of each category and how they change over time

## src

### apikey_manager
Helper script to create file with apikeys so user doesn't need to

### apod_scrapper
Script to scrape NASAs Astronomy Picture Of the Day website. 
Saves database to data/databases/raw folder. Creates metadata to use as check for corruption/manual edits to determine if dataset needs to be rebuilt or appending is better.

### apod_manager
Takes raw dataset created by apod_scrapper and creates labels using TF-IDF and Naive Bayes algorithm, saves as a new database under data/databases/cleaned.
Can do exploratory analysis if file run directly. 
Manually checked accuracy of 10 entries from the 2 lowest mean confidence categories (both ~60%) to be ~90% accurate. 3rd lowest mean confidence category ~77%

### apod_visualizer
Creates new "confident" APOD dataset by removing entries with <50% confidence, which increased average mean by 5%, average median by 5%, and decreased average standard deviation by 4%
Uses confident APOD dataset to create heatmap showing frequency of each category over time
Uses confident APOD dataset to create pie charts showing category distribution for each year

## requirements.txt - list of requirements to run all code, can be used to install everything with one line
