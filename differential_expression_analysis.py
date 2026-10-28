import numpy as np
import csv
import random
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import seaborn as sns

genes = []
data = []

with open("GSE150910_gene-level_count_file.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        
        genes.append(row[0])
        data.append(row[1:])

samples = data[0]
data = data[1:]
genes = genes[1:]

data = np.array(data).astype(float)# use numPy to convert the data to a float array(numerical data format) for numerical analysis
genes = np.array(genes)
samples = np.array(samples)

print(genes.shape)
print(samples.shape)
print(data.shape)

######checking the data making sure that it is correctly formatted
print("genes:")
print(genes[0:5])
print()

print("samples:")
print(samples[0:4])
print()


print("data:")
print(data[0:5,0:4])
print()

#making new array that contains control and treatment samples

labels = []

for i in range(len(samples)):
    tmp = samples[i].split("_")
    labels.append(tmp[0])
    
labels = np.array(labels)

#checking the labels
print(labels[0:5])

print(np.unique(labels))

#eliminating chp samples using boolean mask
data = data[:  ,labels != "chp"]
samples = samples[labels != "chp"]
labels = labels[labels != "chp"]

print(data.shape)
print(samples.shape)
print(labels.shape)

#data normalization (counts per million normalization(cpm))
for j in range(data.shape[1]):
    column_sum = sum(data[:,j])
    
    data[:,j] = data[:,j] / column_sum * 1000000
    print(data)

#checking mean values of control and ipf samples
mean_CPM_control = data[:,labels == "control"].mean(axis=1)
mean_CPM_ipf = data[:,labels == "ipf"].mean(axis=1)

print (mean_CPM_control)
print (mean_CPM_control.shape)

#eliminating genes with low expression with boolean mask
to_keep = (mean_CPM_ipf >= 5) | (mean_CPM_control >= 5)

data = data[to_keep,:]
genes = genes[to_keep]

print(data.shape)
print(len(genes))

# quick check of the data starting with just the first gene

control = data[0, labels == "control"]
ipf = data[0, labels == "ipf"]

print(control.shape)
print(ipf.shape)

#Kolmogorov-Smirnov (KS) test

ks_statistic, p_value = ks_2samp(control, ipf)

print(f"ks_statistic: {ks_statistic}")
print(f"p_value: {p_value}")

#calculating log2 fold change for the first gene
epsilon = 1  # small constant to avoid division by zero - optional (0.1, 0.01)

control_mean = np.mean(control)
ipf_mean = np.mean(ipf)

# calculate log2(fold change), adding epsilon to avoid division by 0
FC = (ipf_mean + epsilon) / (control_mean + epsilon)
log2_FC = np.log2(FC)

print(log2_FC)

## getting p-value and log2FC, now for every gene

p_values = []
log2_FCs = []

epsilon = 1  # small constant to avoid division by zero

for i in range(data.shape[0]):
    
    control = data[i, labels == "control"]
    ipf = data[i, labels == "ipf"]

    # perform ks-test
    ks_statistic, p_value = ks_2samp(control, ipf)

    # save p_value for this gene
    p_values.append(p_value)


    # calculate means for each condition
    control_mean = np.mean(control)
    ipf_mean = np.mean(ipf)

    # calculate log2(fold change), adding epsilon to avoid division by 0
    FC = (ipf_mean + epsilon) / (control_mean + epsilon)
    log2_FC = np.log2(FC)
    
    # save log2_FC for this gene
    log2_FCs.append(log2_FC)


# cast results to numpy arrays
p_values = np.array(p_values)
log2_FCs = np.array(log2_FCs)

#Bonferroni correction for multiple testing
p_values_bonf = p_values * len(genes)

#Differentially expressed genes (DEGs) selection based on p-value and log2 fold change thresholds
to_keep = (p_values_bonf <= 0.05) & (np.abs(log2_FCs) >= 2)

sig_genes = genes[to_keep]
sig_log2_FCs = log2_FCs[to_keep]
sig_p_values_bonf = p_values_bonf[to_keep]
sig_data = data[to_keep, :]

##########################Summary of findings###################################################
print(f"Number of DEGs: {len(sig_genes)}")
print(f"Total number of genes: {len(genes)}")
print(f"DEG percentage: {len(sig_genes) / len(genes) * 100}")
print()
print(f"Number of upregulated DEGs: {np.sum(sig_log2_FCs>0)}")
print(f"Number of downregulated DEGs: {np.sum(sig_log2_FCs<0)}")

#Printing the upregulated genes
for i in range(len(sig_genes)):
    if sig_log2_FCs[i] > 0: #genes that are upregulated in ipf samples compared to control samples
        print(sig_genes[i])

#expression matrix of the DEGs
sig_data_log2fc = np.zeros_like(sig_data)

print(sig_data.shape)

epsilon = 1 


for i in range(len(sig_genes)):
    
    control_mean =  np.mean(sig_data[i,labels == "control"])
    
    sig_data_log2fc[i,:] = np.log2((sig_data[i,:]+epsilon) / (control_mean +epsilon))

##Heatmap of the DEGs
sns_plot = sns.clustermap(sig_data_log2fc, xticklabels=samples, yticklabels= sig_genes, cmap="coolwarm", vmin=-6, vmax=6)

# changing font sizes for x and y labels
sns_plot.ax_heatmap.set_xticklabels(sns_plot.ax_heatmap.get_xmajorticklabels(), fontsize=2)
sns_plot.ax_heatmap.set_yticklabels(sns_plot.ax_heatmap.get_ymajorticklabels(), fontsize=4)


sns_plot.savefig("output/heatmap.pdf")

## expression bar plot##

gene_to_plot = "POSTN" #select any genes from the list of DEGs to plot its expression levels in control and ipf samples

gene_matches = np.flatnonzero(genes == gene_to_plot)

if len(gene_matches) == 0:
    raise ValueError(f"{gene_to_plot} was not found after filtering")

gene_index = gene_matches[0]

control_expression = data[gene_index, labels == "control"]
ipf_expression = data[gene_index, labels == "ipf"]

control_mean = np.mean(control_expression)
control_sem = np.std(control_expression, ddof=1) / np.sqrt(len(control_expression))

ipf_mean = np.mean(ipf_expression)
ipf_sem = np.std(ipf_expression, ddof=1) / np.sqrt(len(ipf_expression))

plt.figure()
plt.bar(
    ["Control", "IPF"],
    [control_mean, ipf_mean],
    yerr=[control_sem, ipf_sem],
    capsize=5,
    color=["blue", "red"],
)
plt.ylabel("Expression Level (CPM)") 
plt.title(f"{gene_to_plot} Expression")
plt.savefig(
    f"output/{gene_to_plot}_expression_bar.pdf",
    format="pdf",
    bbox_inches="tight", #save the figure
)
plt.show()

## expression scatter plot

gene_to_plot = "POSTN"

control_expression = data[genes == gene_to_plot, labels == "control"]
ipf_expression = data[genes == gene_to_plot, labels == "ipf"]

control_mean = np.mean(control_expression)
control_sem = np.std(control_expression) / np.sqrt(len(control_expression))
ipf_mean = np.mean(ipf_expression)
ipf_sem = np.std(ipf_expression) / np.sqrt(len(ipf_expression))

def add_jitter(values, jitter_amount=0.05):
    return values + np.random.uniform(-jitter_amount, jitter_amount, size=values.shape)

control_x = add_jitter(np.ones_like(control_expression) * 1)
ipf_x = add_jitter(np.ones_like(ipf_expression) * 2)

plt.scatter(control_x, control_expression, color='blue', label='Control', s=10)  # smaller dots with s=10
plt.scatter(ipf_x, ipf_expression, color='red', label='IPF', s=10)  # smaller dots with s=10

plt.errorbar(1, control_mean, yerr=control_sem, fmt='o', color='black', capsize=5)
plt.errorbar(2, ipf_mean, yerr=ipf_sem, fmt='o', color='black', capsize=5)

plt.xticks([1, 2], ['Control', 'IPF'])

plt.ylabel('Expression Level (CPM)')
plt.title(f'{gene_to_plot} Expression')
plt.legend()

plt.savefig(f"output/{gene_to_plot}_expression_scatter.pdf") #save the figure
plt.show()

# DEG expression correlation

correlation_matrix = np.corrcoef(sig_data)

sns_plot = sns.clustermap(correlation_matrix, 
                          xticklabels=sig_genes, 
                          yticklabels=sig_genes, 
                          cmap="coolwarm", 
                          annot=False, 
                          linewidths=.5, 
                          figsize=(12, 10))

sns_plot.ax_heatmap.set_xticklabels(sns_plot.ax_heatmap.get_xmajorticklabels(), fontsize=4)
sns_plot.ax_heatmap.set_yticklabels(sns_plot.ax_heatmap.get_ymajorticklabels(), fontsize=4)

sns_plot.savefig("output/degs_correlation_heatmap.pdf") #save the figure
plt.show()