#%%
import matplotlib.pyplot as plt
import numpy as np
nucleotide = "mono"
norm_method = "from_mono"
# ref_raw = np.load(f"first_var_CTCF_refscores_raw_{nucleotide}.npy")
# ref_normalized = np.load(f"first_var_CTCF_refscores_normalized_{nucleotide}_{norm_method}.npy")
# alt_raw = np.load(f"first_var_CTCF_altscores_raw_{nucleotide}.npy")
# alt_normalized = np.load(f"first_var_CTCF_altscores_normalized_{nucleotide}_{norm_method}.npy")
dir = "/home/seh197/rezwan/research/Maria/PADIT-seq/HOXD13/probNorm_preds/single_snps"
snp = "rs1262183AtoT_HUMAN"
ws = 50 # if it is mouse (HOXD13) ws is 31
ref_raw = np.load(f"{dir}/raw_scores/{snp}_refscores_raw_{ws}.npy")
alt_raw = np.load(f"{dir}/raw_scores/{snp}_altscores_raw_{ws}.npy")
_
ref_normalized = np.load(f"{dir}/normalized_scores/{snp}_refscores_normalized_{ws}.npy")
alt_normalized = np.load(f"{dir}/normalized_scores/{snp}_altscores_normalized_{ws}.npy")
print(len(ref_raw))
print(len(ref_normalized))

x_range = min(len(ref_raw), len(ref_normalized))
print(x_range)
plt.plot(range(0,x_range), ref_raw[:x_range], marker='o', label="ref")
plt.plot(range(0,x_range), alt_raw[:x_range], marker='o', label="alt")
plt.title(f"raw convolution scores on {nucleotide} nucleotide \n {snp.replace('to', '>')} - ws = {ws}")
plt.legend()
# plt.savefig(f"{dir}/plots/{snp}_HUMAN_raw.png", dpi=300, bbox_inches="tight")
plt.show()

plt.plot(range(0,x_range), ref_normalized[:x_range], marker='o', label="ref")
plt.plot(range(0,x_range), alt_normalized[:x_range], marker='o', label="alt")
plt.title(f"normalized scores on {nucleotide} nucleotide \n {snp.replace('to', '>')} - ws = {ws} - FWD")
plt.legend(loc="upper left")
# plt.savefig(f"{dir}/plots/{snp}_HUMAN_normalized_fwd.png", dpi=300, bbox_inches="tight")
plt.show()

plt.plot(range(0,x_range), ref_normalized[x_range:], marker='o', label="ref")
plt.plot(range(0,x_range), alt_normalized[x_range:], marker='o', label="alt")
plt.title(f"normalized scores on {nucleotide} nucleotide \n {snp.replace('to', '>')} - ws = {ws} - RC")
plt.legend(loc="upper left")
# plt.savefig(f"{dir}/plots/{snp}_HUMAN_normalized_rc.png", dpi=300, bbox_inches="tight")
plt.show()

ref_max = np.maximum(ref_normalized[:x_range], ref_normalized[x_range:])
alt_max = np.maximum(alt_normalized[:x_range], alt_normalized[x_range:])
plt.plot(range(0,x_range), ref_max, marker='o', label="ref")
plt.plot(range(0,x_range), alt_max, marker='o', label="alt")
plt.title(f"normalized scores on {nucleotide} nucleotide \n {snp.replace('to', '>')} - ws = {ws} - Max")
plt.legend(loc="upper left")
# plt.savefig(f"{dir}/plots/{snp}_normalized_max.png", dpi=300, bbox_inches="tight")
plt.show()

ref_mean = (ref_normalized[:x_range]+ref_normalized[x_range:])/2
alt_mean = (alt_normalized[:x_range]+alt_normalized[x_range:])/2
plt.plot(range(0,x_range), ref_mean, marker='o', label="ref")
plt.plot(range(0,x_range), alt_mean, marker='o', label="alt")
plt.title(f"normalized scores on {nucleotide} nucleotide \n {snp.replace('to', '>')} - ws = {ws} - Avg")
plt.legend(loc="upper left")
# plt.savefig(f"{dir}/plots/{snp}_normalized_mean.png", dpi=300, bbox_inches="tight")
plt.show()

print("ref mean:", np.mean(ref_normalized))
print("alt mean", np.mean(alt_normalized))
print("ref max:", np.max(ref_normalized))
print("alt max", np.max(alt_normalized))
#%%

refseq = "TCCTCCCGGCCCCCGCCCTCCCACAGCCCTTTGCAGGACGTGCAACAGG"
altseq = "TCCTCCCGGCCCCCGCCCTCCCACCGCCCTTTGCAGGACGTGCAACAGG"

# %%
import seaborn as sns
import pandas as pd
PPM = np.load(f"{dir}/for_heatmaps/EGR1_PPM.npy")
PPM = PPM[:, PPM.sum(axis=0) <= 1]

plt.figure(figsize=(PPM.shape[1],PPM.shape[0]))
ax = sns.heatmap(np.round(PPM, 2),
            vmin=0,
            vmax=1,
            #center=1,
            cmap='Blues_r',
            annot=True,
            annot_kws={"size": 13},
            fmt='.2f',
            xticklabels=False,
            yticklabels=False)
ax.set_title("EGR1 - PPM", fontsize=20, pad=20)
plt.show()
# #%%
# ## plot raw scores
# plt.figure(figsize=(len(ref_raw),1))
# ax = sns.heatmap(ref_raw[None, :],
#             # vmin=0,
#             # vmax=1,
#             #center=1,
#             cmap='Blues_r',
#             annot=True,
#             annot_kws={"size": 13},
#             fmt='.2f',
#             xticklabels=False,
#             yticklabels=False)
# ax.set_title("raw scores - ref", fontsize=20, pad=20)
# plt.show()
# plt.figure(figsize=(len(alt_raw),1))
# ax = sns.heatmap(alt_raw[None, :],
#             # vmin=0,
#             # vmax=1,
#             #center=1,
#             cmap='Blues_r',
#             annot=True,
#             annot_kws={"size": 13},
#             fmt='.2f',
#             xticklabels=False,
#             yticklabels=False)
# ax.set_title("raw scores - alt", fontsize=20, pad=20)
# plt.show()
#%%
# plt.figure(figsize=(len(ref_normalized),1))
# ax = sns.heatmap(ref_normalized[None, :x_range],
#             vmin=0,
#             vmax=1,
#             center=1,
#             cmap='Blues_r',
#             annot=True,
#             annot_kws={"size": 13},
#             fmt='.2f',
#             xticklabels=False,
#             yticklabels=False)
# ax.set_title("normalized scores - ref", fontsize=20, pad=20)
# plt.show()
# plt.figure(figsize=(len(alt_normalized),1))
# ax = sns.heatmap(alt_normalized[None, :x_range],
#             vmin=0,
#             vmax=1,
#             center=1,
#             cmap='Blues_r',
#             annot=True,
#             annot_kws={"size": 13},
#             fmt='.2f',
#             xticklabels=False,
#             yticklabels=False)
# ax.set_title("normalized scores - alt", fontsize=20, pad=20)
# plt.show()
#%%
plt.figure(figsize=(len(ref_normalized)/2,1))
ax = sns.heatmap(ref_normalized[None, x_range:],
            vmin=0,
            vmax=1,
            center=1,
            cmap='Blues_r',
            annot=True,
            annot_kws={"size": 13},
            fmt='.2f',
            xticklabels=False,
            yticklabels=False)
ax.set_title("normalized scores - ref", fontsize=20, pad=20)
plt.show()
plt.figure(figsize=(len(alt_normalized)/2,1))
ax = sns.heatmap(alt_normalized[None, x_range:],
            vmin=0,
            vmax=1,
            center=1,
            cmap='Blues_r',
            annot=True,
            annot_kws={"size": 13},
            fmt='.2f',
            xticklabels=False,
            yticklabels=False)
ax.set_title("normalized scores - alt", fontsize=20, pad=20)
plt.show()
#%%
import logomaker
def reverse_complement_ppm(ppm, alphabet=("A", "C", "G", "T")):
    """
    Reverse-complement a Position Probability Matrix (PPM).

    Parameters
    ----------
    ppm : np.ndarray
        Shape (L, 4) where columns correspond to A, C, G, T in that order.
    alphabet : tuple
        Order of nucleotides in columns (default: A, C, G, T)

    Returns
    -------
    np.ndarray
        Reverse-complemented PPM with same shape (L, 4)
    """
    ppm = np.asarray(ppm)
    
    # Reverse positions
    ppm_rev = ppm[::-1, :]
    
    # Complement column indices
    complement = {"A": "T", "C": "G", "G": "C", "T": "A"}
    col_index = {nt: i for i, nt in enumerate(alphabet)}
    rc_order = [col_index[complement[nt]] for nt in alphabet]
    
    # Reorder columns to complement bases
    ppm_rc = ppm_rev[:, rc_order]
    
    return ppm_rc

PPM = np.load(f"{dir}/for_heatmaps/EGR1_PPM.npy")
new_motif = PPM.T#reverse_complement_ppm(PPM.T)
motif_ppm = pd.DataFrame(new_motif, columns=['A', 'C', 'G', 'T'])

info_matrix = logomaker.transform_matrix(motif_ppm, from_type='probability', to_type='information')

logo = logomaker.Logo(info_matrix,
                      color_scheme='classic',
                      stack_order='big_on_top',
                      baseline_width=0.01)

logo.style_spines(visible=False)
logo.style_spines(spines=['left', 'bottom'], visible=True)
logo.ax.set_ylabel("bits", fontsize=12)
logo.ax.set_xlabel("")  # optional
logo.ax.set_title("EGR1 motif Information Content", fontsize=10)
# %%
