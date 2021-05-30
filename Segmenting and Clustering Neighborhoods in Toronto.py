#!/usr/bin/env python
# coding: utf-8

# # Part 1 
# ## Importing necessary libraries

# In[1]:


import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import numpy as np


# ## Scraping data from Wikipedia

# In[2]:


url = requests.get('https://en.wikipedia.org/w/index.php?title=List_of_postal_codes_of_Canada:_M&oldid=1012118802')

#convert to a beautifulsoup object
soup = bs(url.content)


# ## Scraping target table

# In[3]:


#From HTLM script, table name = "wikitable sortable"
table = soup.find('table', {"class" : "wikitable sortable"})


# ## Converting a HTML table into a pandas dataframe

# In[4]:


#board_array = np.asarray(table)
df = pd.read_html(str(table))
neighborhood=pd.DataFrame(df[0])
neighborhood.head()


# ## Rows with "Not assigned" values
# ### Checking if column "Borough" contains "Not assigned" values and column "Neighbourhood" does not contain "Not assigned" values

# In[5]:


neighborhood2 = neighborhood[(neighborhood.Borough == 'Not assigned') & (neighborhood.Neighbourhood != 'Not assigned')]
print('Table contains', neighborhood2.shape[0], 'rows(s) with the above condition' )


# ## Droping rows from column "Borough" with 'Not assigned' values

# In[6]:


neighborhood = neighborhood[neighborhood['Borough'] != 'Not assigned']

print('Table contains', neighborhood.shape[0], 'rows and', neighborhood.shape[1], 'columns')
neighborhood.head()


# # Part 2

# ## Importing geographical data

# In[7]:


lat_long = pd.read_csv('Geospatial_Coordinates.csv')
lat_long.head(10)


# ## Joining importing data with dataframe from part 1 using a primary key

# In[8]:


merged_df = pd.merge(neighborhood, lat_long, on="Postal Code")
merged_df.head()


# # Part 3

# In[9]:


# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes ')
import folium # map rendering library
get_ipython().system('conda install -c conda-forge geopy --yes ')
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe
import json # library to handle JSON files


# ## Boroughs that contain the word Toronto

# In[10]:


import re
merged_df2 = merged_df[merged_df['Borough'].str.contains('Toronto')].reset_index().drop('index', axis=1)
merged_df2


# In[11]:


CLIENT_ID = 'VY2M1PLSLQIZVA1XAMRIFO1M0YSIKGQQGEIJAPAXE2MWA2K3' # your Foursquare ID
CLIENT_SECRET = 'WUVUBAYDOU2WZKUYPCZNDD4BJWN4ZWNMZTGUQI5124BV3EAJ' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version
LIMIT = 100 # A default Foursquare API limit value


# ## Explore Neighborhoods in Toronto

# In[13]:


address = 'Toronto, ON'

geolocator = Nominatim(user_agent="to_explorer")
location = geolocator.geocode(address)
neighborhood_latitude = location.latitude
neighborhood_longitude = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(neighborhood_latitude, neighborhood_longitude))


# In[14]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighbourhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[15]:


toronto_venues = getNearbyVenues(names=merged_df2['Neighbourhood'],
                                   latitudes=merged_df2['Latitude'],
                                   longitudes=merged_df2['Longitude']
                                  )


# In[17]:


# create map of Toronto using latitude and longitude values
map_toronto = folium.Map(location=[neighborhood_latitude, neighborhood_longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(merged_df2['Latitude'], merged_df2['Longitude'], merged_df2['Borough'], merged_df2['Neighbourhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# In[18]:


print(toronto_venues.shape)
toronto_venues.head()


# In[19]:


print('There are {} uniques categories.'.format(len(toronto_venues['Venue Category'].unique())))


# ## Analyze Each Neighborhood

# In[20]:


# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_onehot['Neighbourhood'] = toronto_venues['Neighbourhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]


toronto_onehot.head()


# In[21]:


toronto_onehot.shape


# ## Grouping rows by neighborhood and by taking the mean of the frequency of occurrence of each category

# In[22]:


toronto_grouped = toronto_onehot.groupby('Neighbourhood').mean().reset_index()
toronto_grouped


# In[23]:


toronto_grouped.shape


# ## Printing each neighborhood along with the top 3 most common venues

# In[24]:


num_top_venues = 3

for hood in toronto_grouped['Neighbourhood']:
    print("----"+hood+"----")
    temp = toronto_grouped[toronto_grouped['Neighbourhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# ## Function to sort the venues in descending order

# In[25]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# ## Creating the new dataframe and display the top 10 venues for each neighbourhood

# In[26]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighbourhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighbourhood'] = toronto_grouped['Neighbourhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# ## Clustering Neighborhoods

# In[27]:


# set number of clusters
kclusters = 3

toronto_grouped_clustering = toronto_grouped.drop('Neighbourhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# ## Dataframe that includes the cluster as well as the top 10 venues for each neighbourhood

# In[28]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = merged_df2

# merge manhattan_grouped with manhattan_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighbourhood'), on='Neighbourhood')

toronto_merged.head() # check the last columns!


# In[ ]:





# In[30]:


# create map
map_clusters = folium.Map(location=[neighborhood_latitude, neighborhood_longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighbourhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# ## Glympse of one of the clusters

# In[31]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 0, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]

