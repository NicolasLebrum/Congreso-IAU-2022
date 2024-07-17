from constants import *

# Generate the date ranges
time_arange1 = np.full((N,), '2015-06', dtype='datetime64[D]')
time_arange2 = np.arange('2015-06', '2017-06', dtype='datetime64[D]')[:N]
time_arange3 = np.arange('2015-06', '2018-06', dtype='datetime64[D]')

# DataFrame for indexing
time_arangep = pd.DataFrame(time_arange3, columns=['Times'])
time_arangep['index'] = time_arangep.index

# Filter even indices and adjust the size if necessary
dates_index = time_arangep[time_arangep['index'] % 2 == 0]
time_arangef = dates_index['Times'].to_numpy()[:N]

# Ensure arrays are the same length before concatenation
min_len = min(len(time_arange1), len(time_arange2), len(time_arangef))
time_arange1 = time_arange1[:min_len].reshape((min_len, 1))
time_arange2 = time_arange2[:min_len].reshape((min_len, 1))
time_arangef = time_arangef[:min_len].reshape((min_len, 1))

# Reshape and concatenate arrays
Dates_to_Gauss = np.concatenate((time_arange1, time_arange2, time_arangef), axis=1)[1:min_len]


# Convert to datetime and format as string
t = pd.to_datetime(Dates_to_Gauss.flatten(), format='%Y-%m-%d', errors='coerce')
Dates_to_Gauss_str = t.strftime('%Y-%m-%d').values.reshape((-1, 3))


def astronomical_time(date):
    spy.furnsh("naif0012.tls")
    tu = spy.str2et(date)
    dt = spy.deltet(tu, "ET")
    return tu - dt

def ephemerides(epoch, cuerpo):
    return Ephem.from_mpc(cuerpo, epochs=epoch, location='568')[["RA", "DEC", "epoch"]]

def earth_position(P_earth):
    t = Time(P_earth)
    P = get_body_barycentric('earth', t, ephemeris='de432s')
    return P

def calculate_celestial_position(observation):
    return observation["RA"], observation["DEC"]

def method_gauss(alfa, delta, t, P1, P2, P3):
    # Unit vectors for the observations
    u = [np.array([np.cos(d * deg) * np.cos(a * deg),
                   np.cos(d * deg) * np.sin(a * deg),
                   np.sin(d * deg)]) for a, d in zip(alfa, delta)]
    u1, u2, u3 = u

    # Lagrangian coefficients (initial guess)
    r2 = 3 * au
    for _ in range(30):
        f3 = 1 - mu / (2 * r2**3) * (t[2] - t[1])**2
        g3 = (t[2] - t[1]) - mu / (6 * r2**3) * (t[2] - t[1])**3
        f1 = 1 - mu / (2 * r2**3) * (t[0] - t[1])**2
        g1 = (t[0] - t[1]) - mu / (6 * r2**3) * (t[0] - t[1])**3
        c1 = g3 / (f1 * g3 - f3 * g1)
        c3 = -g1 / (f1 * g3 - f3 * g1)
        rho2 = (np.dot(P2, np.cross(u1, u3)) - c1 * np.dot(P1, np.cross(u1, u3)) - c3 * np.dot(P3, np.cross(u1, u3))) / np.dot(u2, np.cross(u1, u3))
        r2 = np.sqrt(np.linalg.norm(P2)**2 + rho2**2 - 2 * rho2 * np.dot(P2, u2))

    # Calculate rho1 and rho3
    rho1 = (-np.dot(P2, np.cross(u2, u3)) + c1 * np.dot(P1, np.cross(u2, u3)) + c3 * np.dot(P3, np.cross(u2, u3))) / (c1 * np.dot(u1, np.cross(u2, u3)))
    rho3 = (-np.dot(P2, np.cross(u2, u1)) + c1 * np.dot(P1, np.cross(u2, u1)) + c3 * np.dot(P3, np.cross(u2, u1))) / (c3 * np.dot(u3, np.cross(u2, u1)))

    # Position vectors
    r1vec = rho1 * u1 - P1
    r2vec = rho2 * u2 - P2
    r3vec = rho3 * u3 - P3

    # Speed vector
    v2vec = (-f3 * r1vec + f1 * r3vec) / (f1 * g3 - f3 * g1)

    return r1vec, r2vec, r3vec, v2vec

def ecliptic_positions(ra):
    sx = ra[0]
    sy = ra[1] * np.cos(ae * deg) + ra[2] * np.sin(ae * deg)
    sz = ra[2] * np.cos(ae * deg) - ra[1] * np.sin(ae * deg)
    return sx, sy, sz

def ecliptic_velocity(va):
    vx = va[0]
    vy = va[1] * np.cos(ae * deg) + va[2] * np.sin(ae * deg)
    vz = va[2] * np.cos(ae * deg) - va[1] * np.sin(ae * deg)
    return vx, vy, vz

def orbital_elements(r1, v2):
    h = np.cross(r1, v2)
    k = np.array([0, 0, 1])
    N = np.cross(k, h)
    l = np.cross(v2, h) - mu * (r1 / np.linalg.norm(r1))
    e = l / mu
    i = np.arccos(h[2] / np.linalg.norm(h)) * rad
    ex = np.linalg.norm(e)

    #Node length
    if N[1]>=0:
        long_node = np.arccos(N[0]/np.linalg.norm(N))*rad 
        if long_node<0:
            long_node = long_node + 360
    else:
        long_node = 2 * np.pi-np.arccos(N[0]/np.linalg.norm(N))*rad
        if long_node<0:
            long_node = long_node + 360
        
    #Perihelion argument
    if e[2]>=0:
        arg_perh = np.arccos(np.dot(N,e)/(np.linalg.norm(N)*np.linalg.norm(e)))*rad
        if arg_perh<0:
            arg_perh = arg_perh + 360
    else:
        arg_perh = 2 * np.pi-np.arccos(np.dot(N,e)/(np.linalg.norm(N)*np.linalg.norm(e)))*rad
        if arg_perh<0:
            arg_perh = arg_perh + 360


    #True anomaly
    A = np.dot(e,r2)
    B = ex*np.linalg.norm(r2)
    anom_true = np.arccos(np.linalg.norm(A/B))*rad

    
    #semi-latus rectum
    q = np.linalg.norm(h**2/mu)/au
    
    #Semi-major axis
    a = q /(1-ex**2)

    return a, i, ex, long_node, arg_perh, anom_true

# Data structures for results
list_dates = []
data_orbital_elements = {
    "a": [],
    "e": [],
    "i": [],
    "long_node": [],
    "arg_perh": [],
    "anom_true": []
}

# Main loop
for i in range(len(Dates_to_Gauss)):
    dates = Dates_to_Gauss_str[i]
    
    # Convert dates to strings
    dates_str = [str(date) for date in dates]
    t = [astronomical_time(date) for date in dates_str]

    epochs = [Time(date, scale='utc') for date in dates_str]
    observations = [ephemerides(epoch, cuerpo) for epoch in epochs]
    earth_positions = [earth_position(date) for date in dates_str]
    P = [np.array([-ep.x/u.km, -ep.y/u.km, -ep.z/u.km]) for ep in earth_positions]

    alfa, delta = zip(*[calculate_celestial_position(obs) for obs in observations])
    alfa = np.array(alfa).flatten()
    delta = np.array(delta).flatten()

    r1vec, r2vec, r3vec, v2vec = method_gauss(alfa, delta, t, P[0], P[1], P[2])

    r1 = ecliptic_positions(r1vec)
    r2 = ecliptic_positions(r2vec)
    r3 = ecliptic_positions(r3vec)
    v2 = ecliptic_velocity(v2vec)

    a, i, ex, long_node, arg_perh, anom_true = orbital_elements(r1, v2)

    list_dates.append(dates)
    data_orbital_elements["a"].append(a)
    data_orbital_elements["e"].append(ex)
    data_orbital_elements["i"].append(i)
    data_orbital_elements["long_node"].append(long_node)
    data_orbital_elements["arg_perh"].append(arg_perh)
    data_orbital_elements["anom_true"].append(anom_true)
    
Data_orbital_element = pd.DataFrame(data_orbital_elements)