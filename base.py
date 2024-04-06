from IPython.core.display import display, HTML
import warnings
display(HTML("<style>.container{ widht:90% !important; }<\style>"))
display(HTML("<style>.container{ font-size:18px !important; }<\style>"))
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


#Libraries Python
import spiceypy as spy 
import numpy as np 
import datetime
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
import urllib.request
import statsmodels.api as sm
from scipy import stats
from astropy import units as u
from astropy.coordinates import SkyCoord
from sbpy.data import Ephem
from astropy.time import Time 
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body_barycentric, get_body, get_moon
from astroquery.jplsbdb import SBDB 

#Constants and parameters
au = 149597870.693 # [km], Link: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/de-403-masses.tpc
mu = 132712440023.31 # [km^3*s^-2] G*M_sun Solar gravitational parameter
ae = 23 + 26/60 + 12/3600 # Angle of obliquity of the elliptic from sexagesimal->decimal
deg = np.pi/180 # Parameters that will be used frequently in the codes
rad = 1/deg 

url = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls'
filename = 'naif0012.tls'
urllib.request.urlretrieve(url, filename)

N = 265


#Array filled with data of the first date of the three to use
time_arange1 = np.full((N,1), '2015-06', dtype = 'datetime64[D]')

#Array with data of successive dates from the first date identified in the previous line.
time_arange2 = np.arange('2015-06', '2017-06', dtype='datetime64[D]')
time_arange3 = np.arange('2015-06', '2018-06', dtype='datetime64[D]')
time_arange2 = time_arange2[0:N].reshape((N,1), order='C')

print(time_arange1.shape, time_arange2.shape, time_arange3.shape)


time_arangep = pd.DataFrame(time_arange3, columns = ['Times'])

time_arangep ['index'] = time_arangep.index

dates_index  = time_arangep[time_arangep['index'] % 2 == 0 ]

time_arangef = dates_index['Times']

time_arangef = time_arangef.to_numpy()

a = np.asarray(time_arangef).reshape((548,1))

time_arange_nd = a[0:N]

Dates_to_Gauss = np.concatenate((time_arange1, time_arange2, time_arange_nd), axis = 1)

Dates_to_Gauss = Dates_to_Gauss[1:N]

t= pd.to_datetime(Dates_to_Gauss,cache=False)

# To remove the hours and leave only year-month-day
Dates_to_Gauss = t.strftime('%Y-%m-%d')



#Data base 
table_JPL = pd.read_csv('sbdb_query_results.csv')



