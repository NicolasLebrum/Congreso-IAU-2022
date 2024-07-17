from IPython.core.display import display, HTML
import warnings
display(HTML("<style>.container{ widht:90% !important; }<\style>"))
display(HTML("<style>.container{ font-size:18px !important; }<\style>"))
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

#Libraries Python
import os
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
from hapsira.bodies import *
from hapsira.twobody import Orbit
from hapsira.plotting.orbit.backends import Plotly3D
from hapsira.frames import Planes

# Constants and parameters
au = 149597870.693  # [km], Link: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/de-403-masses.tpc
mu = 132712440023.31  # [km^3*s^-2] G*M_sun Solar gravitational parameter
ae = 23 + 26/60 + 12/3600  # Angle of obliquity of the elliptic from sexagesimal->decimal
deg = np.pi/180  # Parameters that will be used frequently in the codes
rad = 1/deg
N = 265  # Number of intervals to analyze


# archive downloads
url = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls'
filename = 'naif0012.tls'
urllib.request.urlretrieve(url, filename)

# Initial variables 
Asteroide = None
cuerpo = None



