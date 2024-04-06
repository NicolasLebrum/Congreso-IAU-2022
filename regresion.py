
#Table of asteroids referenced in the JPL Horizons NASA
table_JPL = pd.read_csv('sbdb_query_results.csv')

"""Filtering the theoretical orbital elements of the chosen asteroid"""
asteroid_data = table_JPL[table_JPL['spkid'] == identificador] #Tiene que ser el spkid de cada cuerpo
a_asteroid = np.mean(asteroid_data['a']) #Theoretical Semi major axis
e_asteroid = np.mean(asteroid_data['e']) #Theoretical eccentricity
i_asteroid = np.mean(asteroid_data['i']) #Theoretical inclination
long_node_asteroid = np.mean(asteroid_data['om']) #Theoretical ascending node longitude
arg_perh_asteroid = np.mean(asteroid_data['w']) #Theoretical perihelion argument
anom_true_asteroid = np.mean(asteroid_data['ma']) #Theoretical true anomaly




#Delta of theoretical value and value obtained for each orbital element
Data_orbital_elements['delta_a'] = np.abs(Data_orbital_elements['a'] - a_asteroid) #semi-major axis
Data_orbital_elements['delta_e'] = np.abs(Data_orbital_elements['e'] - e_asteroid) #eccentricity
Data_orbital_elements['delta_i'] = np.abs(Data_orbital_elements['i'] - i_asteroid) #inclination
Data_orbital_elements['delta_node'] = np.abs(Data_orbital_elements['long_node'] - long_node_asteroid) #ascending node longitude 
Data_orbital_elements['delta_perh'] = np.abs(Data_orbital_elements['arg_perh'] - arg_perh_asteroid) #perihelion argument


#Index of the minimum delta value of each orbital element
Index_a = np.argmin(Data_orbital_elements['delta_a']) 
Index_e = np.argmin(Data_orbital_elements['delta_e'])
Index_i = np.argmin(Data_orbital_elements['delta_i'])
Index_node = np.argmin(Data_orbital_elements['delta_node'])
Index_perh = np.argmin(Data_orbital_elements['delta_perh'])

# Ordering of 5 orbital elements with the smallest delta respect to them theoretical values.

#semi-major axis
sort_a = Data_orbital_elements.sort_values('a')
df_sort_a  = sort_a.head()
arr_sort_a = np.array(df_sort_a)


#eccentricity
sort_e = Data_orbital_elements.sort_values('e')
df_sort_e  = sort_e.head()
arr_sort_e = np.array(df_sort_e)


#inclination
sort_i = Data_orbital_elements.sort_values('i')
df_sort_i  = sort_i.head()
arr_sort_i = np.array(df_sort_i)


#ascending node longitude
sort_long_node = Data_orbital_elements.sort_values('long_node')
df_sort_long_node  = sort_long_node.head()
arr_sort_long_node = np.array(df_sort_long_node)


#perihelion argument
sort_arg_perh = Data_orbital_elements.sort_values('arg_perh')
df_sort_arg_perh  = sort_arg_perh.head()
arr_sort_arg_perh = np.array(df_sort_arg_perh)

#This part of code show the orbital elements with smallest delta respect to them theoretical values

a_ref = np.array(Data_orbital_elements[np.argmin(Data_orbital_elements['delta_a']):
                                       np.argmin(Data_orbital_elements['delta_a'])+1])

e_ref = np.array(Data_orbital_elements[np.argmin(Data_orbital_elements['delta_e']):
                                       np.argmin(Data_orbital_elements['delta_e'])+1])

i_ref = np.array(Data_orbital_elements[np.argmin(Data_orbital_elements['delta_i']):
                                       np.argmin(Data_orbital_elements['delta_i'])+1])

long_node_ref = np.array(Data_orbital_elements[np.argmin(Data_orbital_elements['delta_node']):
                                               np.argmin(Data_orbital_elements['delta_node'])+1])

arg_perh_ref = np.array(Data_orbital_elements[np.argmin(Data_orbital_elements['delta_perh']):
                                              np.argmin(Data_orbital_elements['delta_perh'])+1])

theoretical_table = pd.DataFrame(asteroid_data)
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

#Generate artificial data (2 regressors + constant)
#Fit regression model
results = sm.OLS(y, x).fit()

#Inspect the results
print(results.summary())

