# SETS
param n;
param t = 10; 
set N := 1..n; #houses 
set A := {N,N}; #arcs that link the houses 
# PARAMS
 
param Cx{N};  
param Cy{N}; 
param Dc{N}; 
param usable{N}; 
param range; 
param Fc;
param Vc; 
param capacity;

#distance between i and j 
param sp{(i, j) in A} := sqrt((Cx[i]-Cx[j])**2 + (Cy[i]-Cy[j])**2);
param distOk{(i, j) in A} := 
	if sp[i, j] <= range then 1 else 0;

# VARS

var built{N} binary;

# OBJECTIVE FUNCTION

minimize obj:
	sum{i in N} built[i] * Dc[i];

#CONSTRAINTS
	  
s.t. building_allowed{i in N}:
	built[i] <= usable[i];
	
s.t. distance_ok{i in N}:	
	sum{j in N} distOk[i, j] * built[j] + built[i] >= 1;

s.t. main_branch: built[1] = 1;
	
