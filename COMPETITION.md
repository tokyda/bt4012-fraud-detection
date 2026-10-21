# BT4012 In-class Kaggle Competition [2026]

## Description

Bitcoin transactions can be used to move funds for illicit purposes, from money laundering to funding illegal marketplaces. In this competition, you'll build a machine learning model that classifies transactions on the Bitcoin network as licit or illicit, using a graph-structured dataset where each transaction is described by a set of anonymized features and connected to other transactions through payment flows.

Only a fraction of transactions carry a known label, so part of the challenge is making the most of a large pool of unlabeled data alongside a highly imbalanced set of confirmed illicit cases. Strong solutions will need thoughtful feature engineering, sound handling of class imbalance, and a model that generalizes to transactions from future time periods it has never seen. This task mirrors a real-world anti-money-laundering problem: catching bad actors early, without drowning investigators in false alarms.

## What will the competition be like?

- The competition will end on **10/24/2026 11:55 PM (SGT)**
- Given a training dataset, make predictions on a test dataset
- You can make submissions in `.csv` format up to 3x per day
- Your top ROC-AUC score will be ranked on a public leaderboard
- The public leaderboard is based on 30% of your submission
- At the end of the competition, you will be ranked on a private leaderboard
- The private leaderboard is based on the other 70% of your submission
- You may select 2 submissions for the final private leaderboard ranking

## How will you be graded?

20% of final grade. 3 components: improvement over baseline, report, and ranking.

### Improvement Over Baseline (5 points)

5 points will be awarded if your model's performance exceeds the baseline model's performance. This includes:

- **Working Code**: Your submitted code should be functional and run without errors.
- **Reproducibility**: Ensure that your results can be reproduced using the submitted code and data.
- **Clear Documentation**: Your code should be well-documented, with comments explaining your approach, feature engineering, model selection, and any other critical steps.

### Report (5 points)

You need to submit a short report by **25th Oct 2026 23:59 Singapore time**. Name your report with your NUS Student Number, i.e. `e1234567.pdf`.

Your code must be hosted on GitHub, and the report must include a link to the repository.

The report should be no more than 4 pages and must clearly document the end-to-end process of your approach to the Fraudulent Bitcoin Transaction Detection task. It should be understandable to a reader unfamiliar with the competition and include the following core sections:

#### Introduction

Briefly introduce the competition's objective and its relevance to fraud analytics. Explain the nature of the fraudulent Bitcoin transaction detection problem and why it's important. Outline your high-level approach, goals, and key strategies. Aim to provide enough background for readers to grasp the context and significance of the task.

#### Methodology

This is the main section of your report. Organize it into the following subsections:

**2.1 Data Preprocessing and Feature Engineering**
Describe how you cleaned, transformed, and prepared the data. Include your decisions around handling missing values, engineering graph-based features (e.g. node degree, neighborhood aggregates), feature selection/extraction, and any augmentation techniques used. Justify your choices using domain insights or experimental evidence.

**2.2 Model Development and Validation**
Explain the models you implemented, how you trained them, and how you validated their performance. Mention techniques like cross-validation, hyperparameter tuning, and how you handled class imbalance or overfitting. Clearly describe your evaluation approach.

**2.3 Model Exploration and Final Strategy**
Summarize the range of models you explored (e.g., logistic regression, decision trees, ensemble methods, graph neural networks) and how you compared them. Discuss why you selected your final submission model(s), referencing performance metrics, validation results, and insights from experimentation.

#### Results and Discussion

Present the performance of your model(s) using appropriate metrics such as accuracy, precision, recall, F1-score, or ROC AUC. Include tables or charts where helpful. Discuss the strengths and limitations of your approach and reflect on possible causes for model behavior.

#### Conclusion and Reflection

Summarize your main findings and what you learned from the project. Reflect on what worked well, what could be improved, and how this experience informs your understanding of AI-driven fraud detection.

You are free to design the layout of the report as long as it covers the essential elements outlined above. Aim for clarity, conciseness, and coherence throughout.

### Final Ranking (10 points)

10 points will be based on your final ranking in the competition. The highest-ranked participant will receive the full 10 points, and the lowest-ranked participant will receive 0 points.

Your score will be computed using a scaled (non-linear) transformation, similar to the approach used for Piazza participation:

```
Final Score = 10 x (AUC score / Highest AUC score)^a
```

Where:
- `AUC score` is your model's private leaderboard score
- `Highest AUC score` is the best score achieved in the competition
- `a` is a scaling parameter (e.g., `a=5`)

Example:

If:
- Your AUC = 0.92
- Highest AUC = 0.95
- a = 5

`Final Score = 10 x (0.92/0.95)^5 ~ 8.5177`

### Deliverables

Finally, put all of your code in a GitHub repo, and include the link to the repo in your report. The repo should contain a Jupyter notebook (`.ipynb`) for your top submission's final model, named with your NUS Student Number, i.e. `e1234567.ipynb`, covering data preprocessing, feature engineering, prediction validation, and generating your `.csv` submission. Also include a `requirements.txt` file listing your dependencies.

## Important

- Upon signing up for Kaggle and joining the competition, kindly change the Display Name in your Kaggle profile to your NUS student number, e.g. `e1234567`.
- Do not sign up using more than one Kaggle account.
- This is a competition between individual participants. You should work on this problem on your own.
- Please read the Overview, Data, and Rules pages carefully.

---

## Understanding the Bitcoin Transaction Graph

### Why this matters

The competition dataset represents Bitcoin activity as a graph: each node is a transaction, and each edge is a payment flow between two transactions. Before doing any feature engineering or modeling, it helps to understand exactly what a "transaction" is in Bitcoin and why the edges are drawn the way they are — this directly explains where the graph's structure and the aggregated features come from.

### Bitcoin doesn't have account balances

In systems like a bank account, your balance is just a number that goes up and down. Bitcoin doesn't work that way. Instead, money exists as discrete chunks called **unspent transaction outputs (UTXOs)**. Each UTXO is created by some transaction and sits there, unspent, until a later transaction consumes it as an input.

An ordinary Bitcoin transaction does two things:

1. **Consumes** one or more existing UTXOs as inputs (this is "spending" money someone already received).
2. **Creates** one or more new outputs, sending specific amounts to specific recipients — and often one output going right back to the sender as "change," the same way you'd get change back from a cash payment.

A wallet's "balance" is really just the sum of all the UTXOs it currently controls, scattered across many past transactions. A UTXO is atomic — you can't partially spend it; you must consume it in full and, if you don't need the whole amount, route the leftover back to yourself as a new "change" output.

Normally: `sum(inputs) = sum(outputs) + transaction fee` (the worked example below ignores fees for simplicity).

### A worked example

Say Alice, Bob, Carol, and Dave are transacting. Here's a small, realistic chain of events:

- **TX_A**: Alice spends a 1.0 BTC output she owned. She pays Bob 0.6 BTC and sends 0.4 BTC back to herself as change. TX_A now has two outputs.
- **TX_X**: An unrelated, earlier transaction where Alice separately owns a 0.2 BTC output.
- **TX_B**: Bob takes the 0.6 BTC he just received (TX_A's output 0) and spends it, paying Carol 0.5 BTC and keeping 0.1 BTC change.
- **TX_C**: Alice later combines her 0.4 BTC change (TX_A's output 1) with the older 0.2 BTC (TX_X's output) and spends both together to pay Dave 0.55 BTC, keeping 0.05 BTC change.

`TX_A` breakdown as a concrete example of the input/output mechanic:

```
TX_A
input: 1.0 BTC (older tx)
output 0 · 0.6 BTC → Bob's wallet
output 1 · 0.4 BTC → Alice's wallet (change)
initiated by: Alice's wallet
```

- **Input** = a pointer to an existing UTXO being consumed (here, a single 1.0 BTC UTXO Alice already owned). Using it as an input destroys it — it's removed from the unspent set permanently, in full, all at once.
- **Output** = a brand-new UTXO being created (here, 0.6 BTC to Bob and 0.4 BTC back to Alice as change). Each output is independently spendable later, by a potentially different transaction.
- Sanity check: `sum(inputs) = sum(outputs)` → 1.0 = 0.6 + 0.4 ✓.

Note that `TX_A` does **not** include the 0.2 BTC from `TX_X` — that's a separate, unrelated UTXO Alice happens to also own from an earlier transaction. A wallet can hold multiple UTXOs simultaneously from different sources; a transaction only bundles together whichever specific UTXOs its creator chooses to spend at that moment (a "coin selection" choice), not everything the wallet owns. Alice only combines the `TX_A` change and the `TX_X` UTXO later, in `TX_C`.

![UTXO transaction graph example: TX_A and TX_X as source transactions, TX_B and TX_C as destination transactions, with edges TX_A→TX_B, TX_A→TX_C, and TX_X→TX_C](docs/images/utxo_graph.png)

### How this becomes the edge list

An edge is drawn from transaction A to transaction B whenever an output of A is spent as an input of B. Wallets and addresses (Alice, Bob, Carol, Dave) never appear in the graph at all — only transactions do.

From the example above, three edges are generated:

| txId1 | txId2 |
|-------|-------|
| TX_A  | TX_B  |
| TX_A  | TX_C  |
| TX_X  | TX_C  |

This is exactly the format of `txs_edgelist.csv` in the competition data: each edge indicates a payment-flow dependency between two transactions — an output of `txId1` is consumed by `txId2`. `txId1` is always the earlier/source transaction (whose output is being spent); `txId2` is always the later/destination transaction (the one spending it).

Notice that `TX_A` produced two edges out of it (because it had two outputs, each spent separately), and `TX_C` has two edges into it (because it had two inputs, drawn from two different earlier transactions). This is why some transactions in the real dataset show up dozens or even hundreds of times in the edge list — a transaction with many inputs (e.g. an exchange sweeping up many customer deposits into one transaction) or many outputs (e.g. a payout distributing funds to many recipients) will have correspondingly many edges.

### Why this matters

A few practical takeaways follow directly from this mechanism:

- **Edges always point forward in time.** A transaction can only spend outputs that already exist, so an edge's source transaction always occurred at or before its destination transaction. Edges respect Bitcoin's causal ordering: a transaction can only spend an output that has already been created. However, `time_step` is a coarse temporal snapshot rather than an exact transaction timestamp — in the Elliptic dataset, connected transactions belong to the same time step, so `time_step` itself should not be used to order the two endpoints of an edge.
- **The graph has no cycles.** Since you can't spend money that doesn't exist yet, a transaction can never (directly or indirectly) point back to itself — it's a DAG.
- **Degree reflects real transaction behavior, not noise.** Graph degree reflects transaction connectivity, but it is not the same as the raw number of Bitcoin inputs or outputs. In-degree measures how many predecessor transactions connect to the transaction in the observed graph; out-degree measures how many later transactions it connects to. The actual number of inputs and outputs is captured separately among the transaction's local features (`feat_1`...`feat_165`).
- **Wallet identity is hidden by design.** You can't reconstruct who Alice, Bob, Carol, or Dave are from the anonymized data — only the transaction-level structure and its features are provided. Any modeling or feature engineering has to work with that transaction graph, not with wallet-level information.

### Dataset reality check (from exploring `data/raw/`)

- `train.csv`: 141,772 rows × 168 cols — `txId`, `time_step` (1–35), `feat_1`...`feat_165`, `label`. `label`: `1` = illicit (3,644 rows), `0` = licit (27,591 rows), `NaN` = unlabeled (110,537 rows) — heavily imbalanced *and* mostly unlabeled.
- `test.csv`: 15,329 rows × 168 cols — same features, `time_step` 36–49 (strictly future periods, no overlap with train) — this is the "generalize to the future" challenge called out in the competition description.
- `txs_edgelist.csv`: 234,355 edges — `txId1 → txId2`, the payment-flow graph described above.
- `sample_submission.csv`: `index, target` — `index` matches `test.csv`'s `index` column; `target` is the predicted probability of illicit, scored by ROC-AUC.
