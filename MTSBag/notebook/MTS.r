install.packages("devtools")
devtools::install_github("okayaa/MTSYS")

library(MTSYS)

# 40 data for versicolor in the iris dataset
iris_versicolor <- iris[61:100, -5]

unit_space_MT <- MT(unit_space_data = unit)

# 10 data for each kind (setosa, versicolor, virginica) in the iris dataset
iris_test <- iris[c(1:10, 51:60, 101:110), -5]

diagnosis_MT <- diagnosis(unit_space = unit, newdata = test, 
                          threshold = 4)

(diagnosis_MT$le_threshold)

#>     1     2     3     4     5     6     7     8     9    10
#> FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
#>   51    52    53    54    55    56    57    58    59    60   
#> TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
#>  101   102   103   104   105   106   107   108   109   110 
#> TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE 