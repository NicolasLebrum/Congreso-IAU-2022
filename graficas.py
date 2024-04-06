fag,ax = plt.subplots(5,1, figsize=(6,36)) 

ax[0].hist(a,bins=3000,color="skyblue");
#ax[0].set_xlim(a_asteroid-1, a_asteroid+1)
ax[0].set_xlim(1, 3)
ax[0].axvline(a_asteroid, color='red',markersize=50)
ax[0].set_xlabel('a [AU]', size=20)
ax[0].set_ylabel('Number bodies', size=20)
ax[0].set_title('Semi-major axis', size = 20)
ax[0].grid(color='black')
ax[0].legend(["Theoric value","Obtained values"],loc='best')
ax[0].minorticks_on()



ax[1].hist(e,bins=8,color="skyblue");
ax[1].axvline(e_asteroid, color='red',markersize=50) 
ax[1].set_xlabel('e', size=20)
ax[1].set_ylabel('Number bodies', size=20)
ax[1].set_title('Eccentricity', size = 20)
ax[1].grid(color='black')
#ex.axhline(y=e_asteroid, color='red',markersize=50)
ax[1].legend(["Theoric value","Obtained values"],loc='best')
ax[1].minorticks_on()



ax[2].set_ylabel('Number bodies', size=20)
ax[2].hist(i,bins=25,color="skyblue");
ax[2].set_xlim(i_asteroid-20, i_asteroid+20)
ax[2].set_xlabel('i [deg]', size=20)
ax[2].set_title('Inclination', size = 20)
ax[2].axvline(i_asteroid, color='red',markersize=50) 
ax[2].grid(color='black')
ax[2].legend(["Theoric value","Obtained values"],loc='best')
ax[2].minorticks_on()



ax[3].hist(long_node,bins=23,color="skyblue");
ax[3].set_xlim(long_node_asteroid-50, long_node_asteroid+50)
#ax[3].set_xlim(180, 198)
ax[3].axvline(long_node_asteroid, color='red',markersize=50) 
ax[3].set_xlabel('Node [deg]', size=20)
ax[3].set_ylabel('Number bodies', size=20)
ax[3].set_title('Longitude of the ascending node', size = 20)
ax[3].grid(color='black')
ax[3].legend(["Theoric value","Obtained values"],loc='best')
ax[3].minorticks_on()



ax[4].hist(arg_perh,bins=50,color="skyblue");
ax[4].set_xlim(arg_perh_asteroid-15, arg_perh_asteroid+15)
ax[4].axvline(arg_perh_asteroid, color='red',markersize=50) 
#hx.axhline(y= arg_perh_asteroid, xmin=0, xmax=50, color='blue')
ax[4].set_xlabel('Peri [deg]', size=20)
ax[4].set_ylabel('Number bodies', size=20)
ax[4].set_title('Argument of perihelion', size = 20)
ax[4].grid(color='black')
ax[4].legend(["Theoric value","Obtained values"],loc='best')
ax[4].minorticks_on()