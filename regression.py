from constants import *
from method import *


# Create directories for saving plots
asteroid_name = cuerpo  # Replace with actual asteroid name
base_dir = "RESULTS"  #base directory
asteroid_dir = os.path.join(base_dir, asteroid_name)

# Create directory if it doesn't exist
# os.makedirs(asteroid_dir, exist_ok=True)


# Theorical elements
a = Data_orbital_element[Data_orbital_element["a"]>0]["a"] #Semi major axis
e = Data_orbital_element[(Data_orbital_element["e"]>0)&(Data_orbital_element["e"]<1)]["e"] #Eccentricity
i=Data_orbital_element["i"] #Inclination
long_node = Data_orbital_element["long_node"] #Ascending node longitude
arg_perh = Data_orbital_element["arg_perh"] #Perihelion argument
anom_true = Data_orbital_element["anom_true"] #True anomaly

#Table of asteroids referenced in the JPL Horizons NASA
"""Filtering the theoretical orbital elements of the chosen asteroid""" 
a_asteroid = np.mean(Asteroide['a']) #Theoretical Semi major axis
e_asteroid = np.mean(Asteroide['e']) #Theoretical eccentricity
i_asteroid = np.mean(Asteroide['i']) #Theoretical inclination
long_node_asteroid = np.mean(Asteroide['om']) #Theoretical ascending node longitude
arg_perh_asteroid = np.mean(Asteroide['w']) #Theoretical perihelion argument
anom_true_asteroid = np.mean(Asteroide['ma']) #Theoretical true anomaly

# Grap orbital elements
'''
fag,ax = plt.subplots(5,1, figsize=(6,36)) 

ax[0].hist(a,bins='auto',color="skyblue");
ax[0].set_xlim(a_asteroid-1, a_asteroid+1)
ax[0].set_xlim(1, 3)
ax[0].axvline(a_asteroid, color='red',markersize=50)
ax[0].set_xlabel('a [AU]', size=20)
ax[0].set_ylabel('Number bodies', size=20)
ax[0].set_title('Semi-major axis', size = 20)
ax[0].grid(color='black')
ax[0].legend(["Theoric value","Obtained values"],loc='best')
ax[0].minorticks_on()

ax[1].hist(e,bins='auto',color="skyblue");
ax[1].axvline(e_asteroid, color='red',markersize=50) 
ax[1].set_xlabel('e', size=20)
ax[1].set_ylabel('Number bodies', size=20)
ax[1].set_title('Eccentricity', size = 20)
ax[1].grid(color='black')
#ex.axhline(y=e_asteroid, color='red',markersize=50)
ax[1].legend(["Theoric value","Obtained values"],loc='best')
ax[1].minorticks_on()

ax[2].set_ylabel('Number bodies', size=20)
ax[2].hist(i,bins='auto',color="skyblue");
ax[2].set_xlim(i_asteroid-20, i_asteroid+20)
ax[2].set_xlabel('i [deg]', size=20)
ax[2].set_title('Inclination', size = 20)
ax[2].axvline(i_asteroid, color='red',markersize=50) 
ax[2].grid(color='black')
ax[2].legend(["Theoric value","Obtained values"],loc='best')
ax[2].minorticks_on()

ax[3].hist(long_node,bins='auto',color="skyblue");
ax[3].set_xlim(long_node_asteroid-50, long_node_asteroid+50)
#ax[3].set_xlim(180, 198)
ax[3].axvline(long_node_asteroid, color='red',markersize=50) 
ax[3].set_xlabel('Node [deg]', size=20)
ax[3].set_ylabel('Number bodies', size=20)
ax[3].set_title('Longitude of the ascending node', size = 20)
ax[3].grid(color='black')
ax[3].legend(["Theoric value","Obtained values"],loc='best')
ax[3].minorticks_on()

ax[4].hist(arg_perh,bins='auto',color="skyblue");
ax[4].set_xlim(arg_perh_asteroid-15, arg_perh_asteroid+15)
ax[4].axvline(arg_perh_asteroid, color='red',markersize=50) 
#hx.axhline(y= arg_perh_asteroid, xmin=0, xmax=50, color='blue')
ax[4].set_xlabel('Peri [deg]', size=20)
ax[4].set_ylabel('Number bodies', size=20)
ax[4].set_title('Argument of perihelion', size = 20)
ax[4].grid(color='black')
ax[4].legend(["Theoric value","Obtained values"],loc='best')
ax[4].minorticks_on()

plot_path = os.path.join(asteroid_dir, 'orbital_elements_distribution.png')
plt.savefig(plot_path, bbox_inches='tight')
'''
#Delta of theoretical value and value obtained for each orbital element
Data_orbital_element['delta_a'] = np.abs(Data_orbital_element['a'] - a_asteroid) #semi-major axis
Data_orbital_element['delta_e'] = np.abs(Data_orbital_element['e'] - e_asteroid) #eccentricity
Data_orbital_element['delta_i'] = np.abs(Data_orbital_element['i'] - i_asteroid) #inclination
Data_orbital_element['delta_node'] = np.abs(Data_orbital_element['long_node'] - long_node_asteroid) #ascending node longitude 
Data_orbital_element['delta_perh'] = np.abs(Data_orbital_element['arg_perh'] - arg_perh_asteroid) #perihelion argument


#Index of the minimum delta value of each orbital element
Index_a = np.argmin(Data_orbital_element['delta_a']) 
Index_e = np.argmin(Data_orbital_element['delta_e'])
Index_i = np.argmin(Data_orbital_element['delta_i'])
Index_node = np.argmin(Data_orbital_element['delta_node'])
Index_perh = np.argmin(Data_orbital_element['delta_perh'])


# Ordering of 5 orbital elements with the smallest delta respect to them theoretical values.
def sort_and_select(data, column):
    sorted_data = data.sort_values(column)
    return np.array(sorted_data.head())

# Ordenar y seleccionar las primeras 5 filas para cada elemento orbital
arr_sort_a = sort_and_select(Data_orbital_element, 'a')  # Semi-major axis
arr_sort_e = sort_and_select(Data_orbital_element, 'e')  # Eccentricity
arr_sort_i = sort_and_select(Data_orbital_element, 'i')  # Inclination
arr_sort_long_node = sort_and_select(Data_orbital_element, 'long_node')  # Ascending node longitude
arr_sort_arg_perh = sort_and_select(Data_orbital_element, 'arg_perh')  # Perihelion argument

#This part of code show the orbital elements with smallest delta respect to them theoretical values

a_ref = np.array(Data_orbital_element[np.argmin(Data_orbital_element['delta_a']):
                                       np.argmin(Data_orbital_element['delta_a'])+1])

e_ref = np.array(Data_orbital_element[np.argmin(Data_orbital_element['delta_e']):
                                       np.argmin(Data_orbital_element['delta_e'])+1])

i_ref = np.array(Data_orbital_element[np.argmin(Data_orbital_element['delta_i']):
                                       np.argmin(Data_orbital_element['delta_i'])+1])

long_node_ref = np.array(Data_orbital_element[np.argmin(Data_orbital_element['delta_node']):
                                               np.argmin(Data_orbital_element['delta_node'])+1])

arg_perh_ref = np.array(Data_orbital_element[np.argmin(Data_orbital_element['delta_perh']):
                                              np.argmin(Data_orbital_element['delta_perh'])+1])

theoretical_table = pd.DataFrame(Asteroide)
filtered_values = pd.DataFrame(np.concatenate([a_ref, e_ref, i_ref, long_node_ref, arg_perh_ref]),
                           columns = ['a','e','i','long_node','arg_perh','mean_anomaly','da','de','di',
                                      'd(long_node)','d(arg_perh)'])


"""Table with the orbital data obtained and filtered, as well as their 
corresponding observation dates used for their calculation and the data number 
to which the orbital element corresponds"""

Dates_obs = pd.DataFrame([list_dates[Index_a],list_dates[Index_e],list_dates[Index_i],
                          list_dates[Index_node],list_dates[Index_perh]],columns = ['Date1','Date2','Date3'])

Data_number = pd.DataFrame([Index_a,Index_e,Index_i,Index_node,Index_perh],columns = ['Data number'])

filtered_values_table = pd.concat([Data_number,Dates_obs, filtered_values.iloc[:,0:6]], axis=1)

#Eliminating negative values of the semi-axis
filtered_values_table = filtered_values_table[filtered_values_table['a']>0]
filtered_values_table = filtered_values_table.reset_index(drop=True) #Resetting dataframe index

# Applying logarithm to the orbital elements to be able to graph
theoretical_matrix_a = np.log10(np.full((len(filtered_values_table)), theoretical_table["a"]))
theoretical_matrix_e = np.log10(np.full((len(filtered_values_table)), theoretical_table["e"]))
theoretical_matrix_i = np.log10(np.full((len(filtered_values_table)), theoretical_table["i"]))
theoretical_matrix_long_node = np.log10(np.full((len(filtered_values_table)), theoretical_table["om"]))
theoretical_matrix_arg_perh = np.log10(np.full((len(filtered_values_table)), theoretical_table["w"]))

Filtered_values_table = np.log10(np.abs(filtered_values_table.iloc[:,4:9]))

#Calculating the errors of each orbital element set
error_e = np.array(np.abs((Filtered_values_table['e'] - theoretical_matrix_e)/theoretical_matrix_e))
error_a = np.array(np.abs((Filtered_values_table['a'] - theoretical_matrix_a)/theoretical_matrix_a))
error_i = np.array(np.abs((Filtered_values_table['i'] - theoretical_matrix_i)/theoretical_matrix_i))
error_node = np.array(np.abs((Filtered_values_table['long_node'] - theoretical_matrix_long_node)/theoretical_matrix_long_node))
error_arg = np.array(np.abs((Filtered_values_table['arg_perh'] - theoretical_matrix_arg_perh)/theoretical_matrix_arg_perh))

# Computing the linear least squares regression for the orbital element dataset.
y = np.array([theoretical_matrix_a, theoretical_matrix_e, theoretical_matrix_i,
     theoretical_matrix_long_node, theoretical_matrix_arg_perh])

y = y.reshape(((len(theoretical_matrix_a)*5)), order='C')    

x = np.array([Filtered_values_table['a'],Filtered_values_table['e'],Filtered_values_table['i'], 
    Filtered_values_table['long_node'], Filtered_values_table['arg_perh']])

x = x.reshape(((len(theoretical_matrix_a)*5)), order='C')  

slope,intercept, r_value, p_value, std_err = stats.linregress(x,y)

print(slope, intercept, r_value, p_value, std_err)

def lineal_function(x):
    return slope*x + intercept 

Y = lineal_function(x)

#linear least squares regression plot
'''
fig = plt.figure(figsize=(7,5))
ejes = fig.add_axes([0,0,1,1])

ejes.scatter(theoretical_matrix_e, Filtered_values_table['e'])
ejes.scatter(theoretical_matrix_a, Filtered_values_table['a'])
ejes.scatter(theoretical_matrix_i, Filtered_values_table['i'])
ejes.scatter(theoretical_matrix_long_node, Filtered_values_table['long_node'])
ejes.scatter(theoretical_matrix_arg_perh, Filtered_values_table['arg_perh'])

ejes.axvline(theoretical_matrix_e[0], color = 'blue', label = 'theoretical a', lw = 1)
ejes.axvline(theoretical_matrix_a[0], color = 'orange', label = 'theoretical a', lw = 1)
ejes.axvline(theoretical_matrix_i[0], color = 'green', label = 'theoretical a', lw = 1)
ejes.axvline(theoretical_matrix_long_node[0], color = 'red', label = 'theoretical a', lw = 1)
ejes.axvline(theoretical_matrix_arg_perh[0], color = 'purple', label = 'theoretical a', lw = 1)

for i in range (len(Filtered_values_table)):
    ejes.axhline(Filtered_values_table['e'][i], color = 'blue', label = 'exp a', lw = 1)
    ejes.axhline(Filtered_values_table['a'][i], color = 'orange', label = 'exp e', lw = 1)
    ejes.axhline(Filtered_values_table['i'][i], color = 'green', label = 'exp i', lw = 1)
    ejes.axhline(Filtered_values_table['long_node'][i], color = 'red', label = 'exp node', lw = 1)
    ejes.axhline(Filtered_values_table['arg_perh'][i], color = 'purple', label = 'exp peri', lw = 1)




ejes.legend(["Eccentricity" , "Semi-major axis","inclination",
             "Longitude of the ascending node","Argument of perihelion"],loc='center left', bbox_to_anchor=(1, 0.5))


plt.grid()
fig.set_size_inches(5, 5)
ejes.plot(x,Y, color = 'black')
ejes.set_xlabel(' Log(Theorical orbital elements data)', size = 10)
ejes.set_ylabel(' Log(Calculated orbital elements data)', size = 10)
ejes.set_title('Lineal squares minimum aproximation', size = 20)

plot_path = os.path.join(asteroid_dir, 'Lineal square.png')
plt.savefig(plot_path, bbox_inches='tight')

#Error bar plot
fig = plt.figure(figsize=(7,5))
ejes = fig.add_axes([0,0,1,1])
ecolor= ['black', 'red', 'blue', 'white']  


ejes.scatter(theoretical_matrix_e, Filtered_values_table['e'])
ejes.scatter(theoretical_matrix_a, Filtered_values_table['a'])
ejes.scatter(theoretical_matrix_i, Filtered_values_table['i'])
ejes.scatter(theoretical_matrix_long_node, Filtered_values_table['long_node'])
ejes.scatter(theoretical_matrix_arg_perh, Filtered_values_table['arg_perh'])


for i in range (len(Filtered_values_table)):
    ejes.errorbar(theoretical_matrix_e[i],Filtered_values_table['e'][i],yerr = error_e[i])
    ejes.errorbar(theoretical_matrix_a[i],Filtered_values_table['a'][i],yerr = error_a[i])
    ejes.errorbar(theoretical_matrix_i[i],Filtered_values_table['i'][i],yerr = error_i[i])
    ejes.errorbar(theoretical_matrix_long_node[i],Filtered_values_table['long_node'][i],yerr = error_node[i])
    ejes.errorbar(theoretical_matrix_arg_perh[i],Filtered_values_table['arg_perh'][i],yerr = error_arg[i])



ejes.legend(["Eccentricity" , "Semi-major axis","inclination",
             "Longitude of the ascending node","Argument of perihelion"],loc='center left', bbox_to_anchor=(1, 0.5))


plt.grid()
fig.set_size_inches(5, 5)
ejes.plot(x,Y, color = 'black')
ejes.set_xlabel(' Log(Theorical orbital elements data)', size = 10)
ejes.set_ylabel(' Log(Calculated orbital elements data)', size = 10)
ejes.set_title('Error bar', size = 20)

plot_path = os.path.join(asteroid_dir, 'Error.png')
plt.savefig(plot_path, bbox_inches='tight')
'''
#Generate artificial data (2 regressors + constant)
#Fit regression model
results = sm.OLS(y, x).fit()
results_table = results.summary()
#(print(results_table))

filepath1 = os.path.join(asteroid_dir, 'arr_sort_a.txt')
np.savetxt('filepath1', arr_sort_a, delimiter=';', comments='')
filepath2 = os.path.join(asteroid_dir, 'arr_sort_e.txt')
np.savetxt('filepath2', arr_sort_e, delimiter=';', comments='')
filepath3 = os.path.join(asteroid_dir, 'arr_sort_i.txt')
np.savetxt('filepath3', arr_sort_i, delimiter=';', comments='')
filepath4 = os.path.join(asteroid_dir, 'arr_sort_long_node.txt')
np.savetxt('filepath4', arr_sort_long_node, delimiter=';', comments='')
filepath5 = os.path.join(asteroid_dir, 'arr_sort_arg_perh.txt')
np.savetxt('filepath5', arr_sort_arg_perh, delimiter=';', comments='')



