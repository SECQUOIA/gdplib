# Multiperiod Blending Problem
This model is a GDP formulation for the Multiperiod Blending Problem. This model was originally formulated by Lotero et. al. and Ovalle et. al. developed 60 test instances which have can be found in `instances_json/`. If you want to learn more about the problem and instances please see the following github repo: https://github.com/arshb11/mpbp-instances. 

If you decide to use these instances or model, please cite the following papers:

> Ovalle, D., Bhatia, A., Laird, C. D., & Grossmann, I. E. (2026). A logic-based decomposition for the global optimization of the multiperiod blending problem using symmetry-breaking cuts. Industrial & Engineering Chemistry Research, 65(7), 3981–3998. https://doi.org/10.1021/acs.iecr.5c02853
> 
> Lotero, I., Trespalacios, F., Grossmann, I. E., Papageorgiou, D. J., & Cheon, M.-S. (2016). An MILP-MINLP decomposition method for the global optimization of a source based model of the multiperiod blending problem. Computers & Chemical Engineering, 87, 13–35. https://doi.org/10.1016/j.compchemeng.2015.12.017 

## Problem Details

### Solution

Best known objective value for the default instance: 337.155000916326 (maximization).

Best found with GAMS/BARON on the `gdp.bigm` reformulation (120 s,
`optcr=1e-6`), 2026-08-18; not proven optimal. The solution point passes all
six GAMS/Examiner checks (primal and dual bounds and constraints at 1e-6,
complementary slackness at 1e-7) with the discrete variables fixed. See #152.

Note that this package ships 60 instances under `instances_json/`; the value
above is for the instance built by the default `build_model()`.
