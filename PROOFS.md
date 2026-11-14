# Proofs

## Setting
A session produces emission-time information X (the slate, the provider groups, the reranker's
claim) and a violation indicator Y in {0,1} that is revealed only later (for example because the
exposure a slate really delivered depends on the user's browsing behaviour). A score
s(X) in R is fixed before calibration (a closed-form margin, or a model fitted on data that is
independent of the calibration and test sessions). The verifier accepts a slate iff s(X) > lambda
and escalates it to repair otherwise. The loss of threshold lambda on session i is

    L_i(lambda) = Y_i * 1[s(X_i) > lambda]        (accepted and violating).

Given calibration sessions 1..n, write Rhat_n(lambda) = (1/n) sum_i L_i(lambda) and choose

    lambda_hat = inf { lambda in R ∪ {+inf} : (n * Rhat_n(lambda) + 1) / (n + 1) <= alpha }.

## Theorem 1 (accepted-but-violating rate)
Assume (X_1,Y_1), ..., (X_{n+1},Y_{n+1}) are exchangeable and alpha >= 1/(n+1). Then

    P( s(X_{n+1}) > lambda_hat  and  Y_{n+1} = 1 ) <= alpha.

If in addition the scores of violating sessions are almost surely distinct, the probability is at
least alpha - 2/(n+1).

Proof. We verify the conditions of Theorem 1 of Angelopoulos et al. (Conformal Risk Control,
ICLR 2024) and repeat its short argument for completeness.
(i) Monotone: for lambda <= lambda', 1[s > lambda'] <= 1[s > lambda], hence L_i is non-increasing.
(ii) Right-continuous: lambda -> 1[s > lambda] is right-continuous (it equals 1 on (-inf, s) and 0
on [s, +inf)), hence so is L_i.
(iii) Bounded: 0 <= L_i <= B = 1.
(iv) Safe point: L_i(+inf) = 0 <= alpha. The feasible set of the infimum is therefore non-empty,
because (n*0 + 1)/(n+1) <= alpha by the assumption alpha >= 1/(n+1).
Define the oracle threshold that also uses the test loss,
    lambda' = inf { lambda : Rhat_{n+1}(lambda) <= alpha },   Rhat_{n+1} = (1/(n+1)) sum_{i<=n+1} L_i.
Since L_{n+1} <= 1, Rhat_{n+1}(lambda) <= (n Rhat_n(lambda) + 1)/(n+1) for every lambda, so every
lambda feasible for lambda_hat is feasible for lambda', which gives lambda' <= lambda_hat and, by
monotonicity, L_{n+1}(lambda_hat) <= L_{n+1}(lambda'). The threshold lambda' is a symmetric
function of the n+1 exchangeable pairs, so conditionally on the multiset of the n+1 loss functions
the test index is uniform over 1..n+1 and
    E[ L_{n+1}(lambda') | multiset ] = Rhat_{n+1}(lambda') <= alpha,
where the last inequality uses right-continuity of Rhat_{n+1} at its infimum. Taking expectations,
E[L_{n+1}(lambda_hat)] <= E[L_{n+1}(lambda')] <= alpha. The left-hand side equals the stated
probability because L is an indicator.
For the lower bound, Theorem 2 of the same paper gives E[L_{n+1}(lambda_hat)] >= alpha - 2B/(n+1)
when the jump of each L_i is at a point that is almost surely distinct across i, which holds when
violating sessions have almost surely distinct scores (non-violating sessions have L_i = 0 and no
jump). QED.

Caveat on the lower bound. The margins used in the experiments take finitely many values (they
depend only on the group pattern of the slate), so violating sessions share scores and the lower
bound does not apply to the reported experiments; only the upper bound is claimed there. The
synthetic check in `code/sim_crc.py` uses continuous scores, where both bounds apply.

Remarks.
1. The guarantee is marginal over calibration and test draws. It is a statement about the joint
   event "accepted and violating", which is the quantity an operator pays for. The conditional rate
   among accepted slates is at most alpha / P(accept) by definition of conditional probability.
2. End to end: if every escalated slate is replaced by a repair that satisfies the constraint for
   every admissible attention model, the served violation rate is at most
   alpha + P(escalated and no such repair exists).
3. Exchangeability fails if calibration and deployment differ in model, constraint tightness or
   audience; `results/verifier_shift.csv` measures one such case (calibrate with uniform patience,
   deploy to impatient users).
4. The bound is on the joint event. The violation rate among accepted slates is the false-discovery
   quantity that conformal selection controls (Jin and Candes 2023; Gui, Jin and Ren 2024);
   `results/verifier.csv` reports it as `viol_among_accepted`.

## Proposition 1 (a constant claim forces full escalation)
Let the score be a claim c(X) taking a single value c0 with probability q, and suppose
P(Y = 1, c = c0) > alpha. Any threshold rule "accept iff c > lambda" whose accepted-but-violating
rate is at most alpha escalates at least a fraction q of the slates.

Proof. A threshold rule either accepts every slate with c = c0 (when lambda < c0) or none of them.
In the first case its accepted-but-violating rate is at least P(Y = 1, c = c0) > alpha. Hence a
rule meeting the bound rejects all slates with c = c0, a fraction q. QED.

In our measurements the rerankers answer "satisfied, confidence 100" on almost every slate, so q is
close to one and the claim alone can only meet a risk target by escalating nearly everything.

## Numeric check
`code/sim_crc.py` draws exchangeable synthetic sessions (SYNTHETIC, continuous scores) with scores of varying informativeness
and compares the empirical accepted-but-violating rate over 100,000 trials with alpha and with
alpha - 2/(n+1); results are in `results/sim_crc.csv`. `code/verifier.py` repeats the check on the
real reranker outputs over random calibration/test splits (`results/verifier.csv`, column `risk`).

## Dropped
The cumulative-violation bound O(sqrt(mT)) for amortised exposure that appeared in the project
brief is not claimed: we did not produce a proof for it.
