# Software_Effort_Estimation
Estimating effort using deep learning techniques on the COCOMO dataset
This project demonstrates how to estimate software development effort using a deep learning model trained on a historical dataset (likely based on COCOMO81). The model uses various project features to predict the required effort and evaluates its performance using standard regression metrics.

## Dataset
Used COCOMO.arff dataset that contains the features
Rely - rely       data       cplx       time      stor       virt  \
count  63.000000  63.000000  63.000000  63.000000  63.00000  63.000000   
mean    1.036349   1.003968   1.091429   1.113810   1.14381   1.008413   
std     0.193477   0.073431   0.202563   0.161639   0.17942   0.120593   
min     0.750000   0.940000   0.700000   1.000000   1.00000   0.870000   
25%     0.880000   0.940000   1.000000   1.000000   1.00000   0.870000   
50%     1.000000   1.000000   1.070000   1.060000   1.06000   1.000000   
75%     1.150000   1.040000   1.300000   1.110000   1.21000   1.150000   
max     1.400000   1.160000   1.650000   1.660000   1.56000   1.300000   

            turn       acap       aexp      pcap       vexp       lexp  \
count  63.000000  63.000000  63.000000  63.00000  63.000000  63.000000   
mean    0.971746   0.905238   0.948571   0.93746   1.005238   1.001429   
std     0.080973   0.151507   0.119243   0.16651   0.093375   0.051988   
min     0.870000   0.710000   0.820000   0.70000   0.900000   0.950000   
25%     0.870000   0.860000   0.820000   0.86000   0.900000   0.950000   
50%     1.000000   0.860000   1.000000   0.86000   1.000000   1.000000   
75%     1.000000   1.000000   1.000000   1.00000   1.100000   1.000000   
max     1.150000   1.460000   1.290000   1.42000   1.210000   1.140000   

            modp       tool       sced          loc        actual  
count  63.000000  63.000000  63.000000    63.000000     63.000000  
mean    1.004127   1.016984   1.048889    77.209841    683.320635  
std     0.130935   0.085735   0.075586   168.509374   1821.582348  
min     0.820000   0.830000   1.000000     1.980000      5.900000  
25%     0.910000   1.000000   1.000000     8.650000     40.500000  
50%     1.000000   1.000000   1.000000    25.000000     98.000000  
75%     1.100000   1.100000   1.080000    60.000000    438.000000  
max     1.240000   1.240000   1.230000  1150.000000  11400.000000  



## Dependencies
`pip install pandas numpy matplotlib seaborn scikit-learn tensorflow`

## Notes

The model uses early stopping to prevent overfitting.

The feature importance is derived from the weights of the first dense layer.

Performance can vary based on data quality, size, and feature engineering.
