# Medium-term Purchasing Contracts Problem

Medium-term Purchasing Contracts problem from https://www.minlp.org/library/problem/index.php?i=129

This model maximizes profit in a short-term horizon in which various contracts are available for purchasing raw materials. The model decides inventory levels, amounts to purchase, amount sold, and flows through the process nodes while maximizing profit. The four different contracts available are:
1. **FIXED PRICE CONTRACT**: buy as much as you want at constant price 
2. **DISCOUNT CONTRACT**: quantities below minimum amount cost RegPrice. Any additional quantity above min amount costs DiscoutPrice.
3. **BULK CONTRACT**: If more than min amount is purchased, whole purchase is at discount price.
4. **FIXED DURATION CONTRACT**: Depending on length of time contract is valid, there is a purchase price during that time and min quantity that must be purchased

## Problem Details

### Solution


### Size
| Component             |   Number |
|:----------------------|---------:|
| Variables             |     1165 |
| Binary variables      |      216 |
| Integer variables     |        0 |
| Continuous variables  |      949 |
| Disjunctions          |       72 |
| Disjuncts             |      216 |
| Constraints           |      762 |
| Nonlinear constraints |        0 |


## References
> [1] Vecchietti, A., & Grossmann, I. (2004). Computational experience with logmip solving linear and nonlinear disjunctive programming problems. Proc. of FOCAPD, 587-590.
> 
> [2] Park, M., Park, S., Mele, F. D., & Grossmann, I. E. (2006). Modeling of purchase and sales contracts in supply chain optimization. Industrial and Engineering Chemistry Research, 45(14), 5013-5026. DOI: 10.1021/ie0513144