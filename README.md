# Deconfounding Temporal Autoencoder (DTA)

This is an anonymous git repository for the paper "Deconfounding Temporal Autoencoder: Estimating Treatment Effects over Time Using Noisy Proxies". 

# Python scripts

The following scripts are used to reproduce the results:

main.py      (main script containing the implementation code)

sim_exp.py   (reproducing the simulation experiments)

mimic_exp.py (reproducing the results with MIMIC-III)

The first two scripts are self contained. The last one uses the pre-processed MIMIC-III data set. The pre-processing pipeline is adopted from Wang et. al. (2020). See the references below.

# Requirements

python 3.6
pytorch 1.7

# Reproducing simulation results

Run the script sim_exp.py. The script contains synthetic data simulation and code that produces the results reported.

# Reproducing MIMIC-III results

Run the script mimic_exp.py. The script requires the pre-processed MIMIC-III data as input (see references below).

MIMIC-III is a freely accessible database. However, access must be requested at https://physionet.org/content/mimiciii/1.4/. When MIMIC-III access is granted, the pre-processed data by Wang et. al. (2020) is accessible with instructions in the respective paper. We use this data set to study the effect of vasopressors and mechanical ventilation on two outcome variables: (diastolic) blood pressure and oxygen saturation. 

We extract 2313 patients with 30 time steps each. We use the following covariates: heart rate, red blood cell count, sodium, mean blood pressure, systemic vascular resistence, glucose, chloride urine, glascow coma scale total, hematrocit, positive end-pressure set, respiratory rate, prothrombin time pt, cholesterol, hemoglobin, creatinine, blood urea nitrogen, bicarbonate, calcium ionized, partial pressure of carbon dioxide, magnesium, anion gap, phosphorous, venous pvo2, platelets, calcium urine.

# References

Wang, S., McDermott, M. B., Chauhan, G., Ghassemi, M., Hughes, M. C., & Naumann, T. (2020, April). MIMIC-extract: A data extraction, preprocessing, and representation pipeline for MIMIC-III. In Proceedings of the ACM Conference on Health, Inference, and Learning (pp. 222-235).

Johnson, A. E., Pollard, T. J., Shen, L., Li-Wei, H. L., Feng, M., Ghassemi, M., ... & Mark, R. G. (2016). MIMIC-III, a freely accessible critical care database. Scientific data, 3(1), 1-9.

