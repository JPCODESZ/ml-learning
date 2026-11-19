# Who is worth calling

A bank phoned people and asked them to open a term deposit. About **11.7%** said yes. This sample builds a call list before the phone rings. It is the same job as scoring leads for a small sales team.

Call length was removed. You only learn the length after the conversation, so a score that uses it cannot decide who to dial.

## What we cleaned

The file has 45,211 people and no duplicate rows.

Some fields were blank: job (288), education (1,857), contact method (13,020), and previous campaign result (36,959). Those blanks were labeled "unknown" rather than deleting the person. Most people had never been in an earlier campaign, so throwing out blank previous results would delete most of the list.

The file names one column `day_of_week`, but the values run from 1 to 31. We treated it as the day of the month.

## How accurate

We held out 20% of the people and did not use them to fit the models. On the other 80%, five-fold cross-validation ranked the models by ROC-AUC (1.00 is a perfect ranking, 0.50 is no better than chance):

- Logistic regression: **0.766** (the five folds stayed within about 0.005)
- Gradient boosting: **0.799** (within about 0.006)

On the held-out people, the ranking scores were:

- Call at random: **0.500**
- Call only people whose previous campaign succeeded: **0.587**
- Logistic regression: **0.771**
- Gradient boosting: **0.796**

Boosting beat both the simple rule and logistic regression. A random call is a yes about **11.7%** of the time.

## Where to set the bar

Calling only previous successes is a strict list: **67.4%** of those 291 holdout calls were a yes, but the list reached only **18.5%** of the people who would have said yes.

We chose the boosting cutoff on the training people, not on the holdout. The rule was: still reach at least 40% of the yeses, and among cutoffs that do that, take the one where the highest share of calls are yeses. That cutoff was **0.75**.

On the held-out people, that list made **778** calls. **54.9%** of them were a yes, and the list still reached **40.4%** of all yeses. That is a worse hit rate than "previous success only," and a much better hit rate than calling at random, while covering more of the yeses than the strict rule.

If staff time is scarce, keep this higher bar. If missing a possible yes is more painful than an extra call, lower the bar. At 0.50 the same model reached **63.7%** of the yeses, but only **34.0%** of its 1,984 calls were a yes.

## What drives the score

Hiding one fact at a time, the ranking slipped most when we hid:

1. Month of the call
2. Whether we reached them by mobile or landline
3. Day of the month
4. What happened in the previous campaign
5. Whether they have a housing loan
6. Account balance
7. How many times they were already contacted in this campaign
8. Age

Month and channel are choices you make, not just facts about the person. Part of the score is "when and how you call," not only "who they are."

## What to do next

Build next week's list from people the boosting score marks at 0.75 or above. Do not dial the whole file. After the campaign, compare the yes rate on that list with the 54.9% from this test before you freeze the rule. And keep call length out of the score.
