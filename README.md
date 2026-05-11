# 30 Year APOD Analysis

Below is a description of each file and script I used to scrape and analyze 30 years of daily posts, over 11,000 entries, on NASA's Astronomy Picture Of the Day (APOD) website, as well as a formal LaTeX report describing all methodology, and findings. 


## data

### databases
raw:  
  apod_data.csv - raw data scraped directly from APOD  

cleaned:  
  apod_labeled_data.csv - apod_data.csv with category labels  


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

apod_category_period_percentage_table.png - Table chart showing numerical percentage values for every category (using Cosmology as parent class to every Cosmos subcategory) over 5 evenly divided periods  

apod_distribution_over_time.png - Line chart showing distribution of APOD posts relative to all categories. Uses Cosmology as parent class to Cosmos>~  


## src

### apikey_manager
Helper script to create file with apikeys so user doesn't need to manually create  

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
Uses confident APOD dataset to create a line graph showing distribution relative to each category. Also makes Cosmology a parent class for each Cosmos subclass. Starts at 1996 since 1995 is an incomplete year
Uses confident APOD data set to create a table chart showing numerical percentage distribution over 5 evenly divided periods. Starts at 1996 since 1995 is an incomplete year  


## Space_Technology_to_Cosmology__A_30_Year_Analysis_of_NASA_s_Astronomy_Picture_Of_the_Day
Formal LaTeX report describing methodolgy and discoveries of all work done relating to this project  

## requirements.txt
list of requirements to run all code, can be used to install everything with one line
