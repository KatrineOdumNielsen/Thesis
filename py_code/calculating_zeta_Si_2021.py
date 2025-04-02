from sympy import *

S, z = symbols("S z")

eq1 = Eq(sqrt((7.5/5.5)*S+((2*7.5**2)/((5.5**2)*(7.5-4)))*z**2) , 0.76) 
eq2 = Eq(((2*z*sqrt(7.5*(7.5-4)))/((sqrt(S)*(2*7.5*z**2))/S+5.5*3.5)**(3/2))*(3*5.5+(8*7.5*z**2)/S*1.5), 4.27)

# output = solve([eq1,eq2],S,z,dict=True)
# print(output) 


initial_guess = (0.1, 0.1)  # Adjust these based on your intuition
solution = nsolve([eq1, eq2], (S, z), initial_guess)
print(solution)