#%%
import matplotlib.pyplot as plt
import numpy as np

def style_genome_axis(ax, chrom_label, x_start, x_end, y_label=None):
    # Limits
    ax.set_xlim(x_start, x_end)

    # Only show start/end ticks like the example
    ax.set_xticks([x_start, x_end])
    ax.set_xticklabels([str(x_start), str(x_end)])

    # Centered chromosome label
    ax.set_xlabel(chrom_label)

    # Clean track-like look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=3, width=0.8, labelsize=10)
    ax.tick_params(axis="y", length=3, width=0.8, labelsize=10)

    # Optional: keep y axis minimal (often genome tracks do)
    if y_label is None:
        ax.set_ylabel("")
    else:
        ax.set_ylabel(y_label)

    # A little padding so tick labels don't touch plot
    ax.margins(x=0)

#%%
nucleotide = "mono"
motif = "HOXD13"
dir = f"/home/seh197/rezwan/research/Maria/PADIT-seq/{motif}/probNorm_preds/single_snps"
snp = "fig2a"

raw = np.load(f"{dir}/raw_scores/{snp}_scores_raw.npy")

normalized = np.load(f"{dir}/normalized_scores/{snp}_scores_normalized.npy")
print(raw)
print(normalized)

print(len(raw), len(normalized))
x_range = min(len(raw), len(normalized))

genomic_start = 12823034
x_range = min(len(raw), len(normalized))
x_vals = np.arange(genomic_start, genomic_start + x_range)
x_end = int(x_vals[-1])

# 1) raw
fig, ax = plt.subplots(figsize=(8, 2))
ax.plot(x_vals, raw[:x_range], marker='o', markersize=2.5, linewidth=0.8, color="black")
ax.set_title(f"raw convolution scores on {nucleotide} nucleotide\n{snp.replace('to','>')}", fontsize=11)
style_genome_axis(ax, "Chr. 14 (mm10)", genomic_start, x_end)
fig.savefig(f"{dir}/plots/{snp}_raw.png", dpi=300, bbox_inches="tight")
plt.show()

# 2) normalized fwd
fig, ax = plt.subplots(figsize=(8, 2))
ax.plot(x_vals, normalized[:x_range], marker='o', markersize=2.5, linewidth=0.8, color="black")
ax.set_title(f"normalized scores on {nucleotide} nucleotide\n{snp.replace('to','>')} - FWD", fontsize=11)
style_genome_axis(ax, "Chr. 14 (mm10)", genomic_start, x_end)
fig.savefig(f"{dir}/plots/{snp}_normalized_fwd.png", dpi=300, bbox_inches="tight")
plt.show()

# 3) normalized rc
fig, ax = plt.subplots(figsize=(8, 2))
ax.plot(x_vals, normalized[x_range:], marker='o', markersize=2.5, linewidth=0.8, color="black")
ax.set_title(f"normalized scores on {nucleotide} nucleotide\n{snp.replace('to','>')} - RC", fontsize=11)
style_genome_axis(ax, "Chr. 14 (mm10)", genomic_start, x_end)
fig.savefig(f"{dir}/plots/{snp}_normalized_rc.png", dpi=300, bbox_inches="tight")
plt.show()

print("ref mean:", np.mean(normalized))
print("ref max:", np.max(normalized))
#%%

# refseq = "TCCTCCCGGCCCCCGCCCTCCCACAGCCCTTTGCAGGACGTGCAACAGG"
# altseq = "TCCTCCCGGCCCCCGCCCTCCCACCGCCCTTTGCAGGACGTGCAACAGG"

# %%
import seaborn as sns
import pandas as pd
PPM = np.load(f"{dir}/for_heatmaps/HOXD13_PPM.npy")
# PPM = PPM[:, PPM.sum(axis=0) <= 1]

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
ax.set_title("HOXD13 - PPM", fontsize=20, pad=20)
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
import pandas as pd
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

PPM = np.load(f"{dir}/for_heatmaps/HXD13_PPM.npy")
new_motif = PPM.T #reverse_complement_ppm(PPM.T)
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
logo.ax.set_title("HOXD13 FWD motif Information Content", fontsize=10)
# %%
