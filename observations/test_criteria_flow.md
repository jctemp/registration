Training Loss (train_loss_epoch):

The models with criteria_flow set to [["gl2d","0.25"]] have the lowest mean training loss.
Models with criteria_flow set to [["bel","0.5"]] have a slightly higher mean training loss compared to others.
SSIM Mean (ssim_mean_epoch):

The highest mean SSIM is achieved by the models with criteria_flow set to [["gl2d","0.25"]].
Models with criteria_flow set to [["zero","0.0"]] have the lowest mean SSIM.
Percentage of Negative Jacobian Determinants (perc_neg_jac_det_mean_epoch):

The models with criteria_flow set to [["gl2d","0.25"]] have the highest mean percentage of negative Jacobian determinants.
All models with criteria_flow set to bel or zero have zero mean percentage of negative Jacobian determinants.
Mean Squared Error (mse_mean_epoch):

The models with criteria_flow set to [["gl2d","0.25"]] have the lowest mean MSE.
Models with criteria_flow set to [["zero","0.0"]] have the highest mean MSE.

---

SSIM Improvement:

The most significant SSIM improvement is observed with [["gl2d","0.25"]], which improves SSIM by 0.050898.
Other gl2d settings also show notable improvements: 0.046038 for [["gl2d","0.5"]] and 0.043684 for [["gl2d","1.0"]].
bel settings show smaller improvements, with [["bel","0.125"]] leading at 0.018241.
Percentage of Negative Jacobian Determinants Improvement:

Only gl2d settings show changes, with [["gl2d","0.25"]] having an increase of 0.001909.

MSE Improvement:

The best MSE improvement is also with [["gl2d","0.25"]], which reduces MSE by 0.001009.
All other gl2d settings and bel settings show lesser reductions in MSE, ranging from -0.000984 to -0.000309.
Overall, the gl2d criteria, particularly [["gl2d","0.25"]], show the most significant improvements in SSIM and MSE compared to the baseline, despite a slight increase in the percentage of negative Jacobian determinants.

---

Strength of BEL Regularization:

The BEL (Boundary Equilibrium Loss) regularization term is strong and effective in controlling model behavior. This is evident from its balanced performance across SSIM, MSE, and the low percentage of negative Jacobian determinants.
Impact of Over-Regularization:

Over-regularization can result in models with lower displacement magnitudes, which may not be sufficient for handling large deformations or displacements. This is likely to affect performance in tasks requiring significant flexibility or adaptability.
Effect of Little Regularization:

When regularization is minimized, there can be noticeable improvements in MSE and SSIM. However, this comes at the cost of a dramatic increase in the percentage of negative Jacobian determinants (|NEG_JAC|%). This indicates a potential instability in the model, as a high |NEG_JAC|% can suggest local distortions or folding in the transformed space.
Summary of Findings:
Criteria Flow - [["bel","0.125"]]:

Performance: Shows a balanced performance with moderate improvements in SSIM and MSE.
|NEG_JAC|%: Maintains a very low percentage of negative Jacobian determinants.
Criteria Flow - [["gl2d","1.0"]] and [["gl2d","0.5"]]:

Performance: Achieves the highest scores in overall performance.
|NEG_JAC|%: Although |NEG_JAC|% is low, it is higher than BEL regularization.
Criteria Flow - [["gl2d","0.25"]]:

Performance: Provides the best improvements in SSIM and MSE but at the cost of a significant increase in |NEG_JAC|%.
Baseline (Zero Regularization):

Performance: Lowest overall performance score, indicating the need for some regularization for optimal performance.
Recommendations:
Moderate Regularization: Using moderate levels of BEL or GL2D regularization appears to provide a balanced approach, improving SSIM and MSE without excessively increasing |NEG_JAC|%.
Avoid Over-Regularization: Excessive regularization can hinder the model’s ability to handle large displacements, potentially impacting performance in more challenging scenarios.
Monitor |NEG_JAC|%: While optimizing for SSIM and MSE, it is crucial to monitor the percentage of negative Jacobian determinants to ensure model stability.

---

Qualitative around 90th frame, do not see a significant benefit if lower regularisation