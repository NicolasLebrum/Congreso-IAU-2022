# Gauss method for preliminary determination of orbits

def astronomical_time(date):
    spy.furnsh("naif0012.tls")
    tu=spy.str2et(date) 
    dt=spy.deltet(tu,"ET") 
    t=tu-dt
    return (t)


def ephemerides(epoch,cuerpo):
    eph = Ephem.from_mpc(cuerpo, epochs=epoch,location='568')["RA","DEC","epoch"] #cuerpo es una variable tipo str
    return(eph)




def earth_position(P_earth):
    t = Time(P_earth)
    P=get_body_barycentric('earth',t,ephemeris='de432s')
    return P


def calculate_celestial_position(observation):
    alfa = observation["RA"]
    delta = observation["DEC"]
    return alfa,delta

def method_gauss(alfa,delta,t):
        
    #Unit vector for the first observation
    u1 = np.array([np.cos(delta[0]*deg)*np.cos(alfa[0]*deg),np.cos(delta[0]*deg)*np.sin(alfa[0]*deg),np.sin(delta[0]*deg)])
      
    #Unit vector for the second observation
    u2 = np.array([np.cos(delta[1]*deg)*np.cos(alfa[1]*deg),np.cos(delta[1]*deg)* np.sin(alfa[1]*deg),np.sin(delta[1]*deg)])
      
    #Unit vector for the third observation
    u3 = np.array([np.cos(delta[2]*deg)*np.cos(alfa[2]*deg),np.cos(delta[2]*deg)*np.sin(alfa[2]*deg),np.sin(delta[2]*deg)])
    #print("u1=",u1)
    #print ("u2=",u2)
    #print("u3=",u3)
    #print("\n")

    #Lagrangian coefficients
    r2 = 3*au #Arbitrary value
      
    f3 = 1-mu/(2*r2**3)*(t[2]-t[1])**2
    g3 = (t[2]-t[1])-mu/(6*r2**3)*(t[2]-t[1])**3
      
    f1 = 1-mu/(2*r2**3)*(t[0]-t[1])**2
    g1 = (t[0]-t[1])-mu/(6*r2**3)*(t[0]-t[1])**3
      
      
    #Constants  
    c1 = g3/(f1*g3-f3*g1)
    c3 = -g1/(f1*g3-f3*g1)
      
    #pho2 value
    rho2 = (np.dot(P2,np.cross(u1,u3))-c1*np.dot(P1,np.cross(u1,u3))-c3*np.dot(P3,np.cross(u1,u3)))/np.dot(u2,np.cross(u1,u3))
      
      
    #Loop to find the true value of r2
    r2=3*au
    for i in range(30):
        f3 = 1-mu/(2*r2**3)*(t[2]-t[1])**2
        g3 = (t[2]-t[1])-mu/(6*r2**3)*(t[2]-t[1])**3

        f1=1-mu/(2*r2**3)*(t[0]-t[1])**2
        g1=(t[0]-t[1])-mu/(6*r2**3)*(t[0]-t[1])**3

        c1 = g3/(f1*g3-f3*g1)
        c3 = -g1/(f1*g3-f3*g1)

        rho2 = (np.dot(P2,np.cross(u1,u3))-c1*np.dot(P1,np.cross(u1,u3))-c3*np.dot(P3,np.cross(u1,u3)))/np.dot(u2,np.cross(u1,u3))

        r2 = (np.linalg.norm(P2)**2+rho2**2-2*rho2*np.dot(P2,u2))**0.5
        
        
    #pho1 and pho3 values
    rho1 = (-np.dot(P2,np.cross(u2,u3))+c1*np.dot(P1,np.cross(u2,u3))+c3*np.dot(P3,np.cross(u2,u3)))/(c1*np.dot(u1,np.cross(u2,u3)))
    rho3 = (-np.dot(P2,np.cross(u2,u1))+c1*np.dot(P1,np.cross(u2,u1))+c3*np.dot(P3,np.cross(u2,u1)))/(c3*np.dot(u3,np.cross(u2,u1)))
    
    
      
    #r1 and r3 values
    r1 = np.sqrt(np.linalg.norm(P1)**2+rho1**2-2*rho1*np.dot(P1,u1))
    r3 = np.sqrt(np.linalg.norm(P3)**2+rho3**2-2*rho3*np.dot(P3,u3))
      
      
    """Vectors in their rectangular components""" 
      
    #Position vectors
    r1vec = (rho1*u1-P1)
    r2vec = (rho2*u2-P2)
    r3vec = (rho3*u3-P3)

    #Speed vector
    v2vec=(-f3*r1vec+f1*r3vec)/(f1*g3-f3*g1)
      
    #print("Geocentric state vector values",r1vec,r2vec,r3vec,v2vec)
    return(r1vec,r2vec,r3vec,v2vec)

def ecliptic_positions(ra):
    sx = ra[0]
    sy = ra[1]*np.cos(ae*deg)+ra[2]*np.sin(ae*deg) 
    sz = ra[2]*np.cos(ae*deg)-ra[1]*np.sin(ae*deg)
    return(sx,sy,sz)

def ecliptic_velocity(va):
    vx = va[0]
    vy = va[1]*np.cos(ae*deg)+va[2]*np.sin(ae*deg) 
    vz = va[2]*np.cos(ae*deg)-va[1]*np.sin(ae*deg)
    return(vx,vy,vz)
      

def orbital_elements(r1,v2):
    h = np.cross(r1,v2)

    #Node Vector    
    k = np.array([0,0,1])
    N = np.cross(k,h)

    #Eccentricity vector
    l = np.cross(v2,h)-mu*(r1/np.linalg.norm(r1))
    e = l/mu


    #Inclination
    i = np.arccos(h[2]/np.linalg.norm(h))*rad

    #Eccentricity
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

    return(a,i,ex,long_node,arg_perh,anom_true)



list_dates = []
data_orbital_elements = {
    "a":[],
    "e":[],
    "i":[],
    "long_node":[],
    "arg_perh":[],
    "anom_true":[]    
}
for i in range(len(Dates_to_Gauss)):
    
    date1 = Dates_to_Gauss[i][0]
    date2 = Dates_to_Gauss[i][1]
    date3 = Dates_to_Gauss[i][2]

    t1 = astronomical_time(date1)
    t2 = astronomical_time(date2)
    t3 = astronomical_time(date3)

    t = [t1,t2,t3]
    list_three_dates = [date1,date2,date3]
    #delta_tiempos = [t[0]-t[1],t[1]-t[2]]

    epoch1 = Time(Dates_to_Gauss[i][0], scale='utc')
    epoch2 = Time(Dates_to_Gauss[i][1], scale='utc')
    epoch3 = Time(Dates_to_Gauss[i][2], scale='utc')

    observation1 = ephemerides(epoch1,cuerpo)
    observation2 = ephemerides(epoch2,cuerpo)
    observation3 = ephemerides(epoch3,ccuerpo)
    
    #print("ob1=",observation1)
    #print("ob2=",observation2)
    #print("ob3=",observation3)


    P_earth1 = earth_position(date1)
    P_earth2 = earth_position(date2)
    P_earth3 = earth_position(date3)

    P1 = np.array([-P_earth1.x/u.km,-P_earth1.y/u.km,-P_earth1.z/u.km])
    P2 = np.array([-P_earth2.x/u.km,-P_earth2.y/u.km,-P_earth2.z/u.km])
    P3 = np.array([-P_earth3.x/u.km,-P_earth3.y/u.km,-P_earth3.z/u.km])
    
    
    #First observation
    alfa1,delta1 = calculate_celestial_position(observation1)  
        
    #second observation
    alfa2,delta2 = calculate_celestial_position(observation2)
        
    #Third observation
    alfa3,delta3 = calculate_celestial_position(observation3)

  
    alfa = np.array([alfa1,alfa2,alfa3])
    delta = np.array([delta1,delta2,delta3])

    #To convert multilist to list
    alfa = alfa.flatten()
    delta = delta.flatten()

    
    r1vec,r2vec,r3vec,v2vec = method_gauss(alfa,delta,t)
    
    
  
    r1 = ecliptic_positions(r1vec)
    r2 = ecliptic_positions(r2vec)
    r3 = ecliptic_positions(r3vec)
    
    v2 = ecliptic_velocity(v2vec)
  
    
    a,i,ex,long_node,arg_perh,anom_true = orbital_elements(r1,v2)
    

#ADDING ORBITAL ELEMENTS TO THE DICTIONARY AND DATES
    list_dates.append(list_three_dates)
    data_orbital_elements["a"].append(a)
    data_orbital_elements["e"].append(ex)
    data_orbital_elements["i"].append(i)
    data_orbital_elements["long_node"].append(long_node)
    data_orbital_elements["arg_perh"].append(arg_perh)
    data_orbital_elements["anom_true"].append(anom_true)



Data_orbital_elements = pd.DataFrame(data_orbital_elements)
a = Data_orbital_elements[Data_orbital_elements["a"]>0]["a"] #Semi major axis
e = Data_orbital_elements[(Data_orbital_elements["e"]>0)&(Data_orbital_elements["e"]<1)]["e"] #Eccentricity
i=Data_orbital_elements["i"] #Inclination
long_node = Data_orbital_elements["long_node"] #Ascending node longitude
arg_perh = Data_orbital_elements["arg_perh"] #Perihelion argument
anom_true = Data_orbital_elements["anom_true"] #True anomaly
times = np.arange(0,N-1)