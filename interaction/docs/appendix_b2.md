# Appendix B.2: CH2 Diagnostics

The Appendix B.2 code isolates where the signed approximation error of CH2 arises.

1. A functional untied-readout sanity check establishes that the CH1 readout identity is algebraically exact.
2. The final-block factorization ladder compares exact block gradients, exact module factorization, the shared readout-force approximation, and the implemented CH2 block term.
3. The per-layer diagnostic compares exact and approximate CH2 contributions at every transformer layer, reconstructs the full backbone interaction, and evaluates top-down cumulative sums.
4. The restricted single-block sweep applies an actual SGD update to one transformer block at a time. It distinguishes finite-step Taylor error from the structural CH2 approximation error.

The diagnostics are designed to support the refined interpretation that the final block can be locally well approximated, while middle/lower layer signed errors and cross-layer cancellation degrade the full summed CH2 prediction.
