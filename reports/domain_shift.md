# Domain-shift check

The model is trained on Reddit comments. Here it is scored on YouTube comments that I labelled by hand (all from Hinglish-language videos).

Labelled comments scored: **105** (of 105 sampled; blanks skipped).

## Headline

| | value |
|---|---|
| Accuracy | 0.590 |
| Macro F1 | 0.396 |
| Majority-class accuracy (baseline) | 0.686 |
| Reddit test macro F1 (for comparison) | 0.864 |

## Label distribution

| class | my labels | model predictions |
|---|---|---|
| negative | 6 | 2 |
| neutral | 27 | 54 |
| positive | 72 | 49 |

## Per-class F1

| class | F1 | support |
|---|---|---|
| negative | 0.000 | 6 |
| neutral | 0.494 | 27 |
| positive | 0.694 | 72 |

## Confusion matrix (rows = my label, columns = predicted)

| mine \ predicted | negative | neutral | positive |
|---|---|---|---|
| negative | 0 | 5 | 1 |
| neutral | 1 | 20 | 6 |
| positive | 1 | 29 | 42 |

## Per video

| video | comments | accuracy | macro F1 |
|---|---|---|---|
| 2dH_qjc9mFg | 35 | 0.571 | 0.395 |
| Ty8gcCKuwNI | 35 | 0.629 | 0.428 |
| nlz9j-r0U9U | 35 | 0.571 | 0.353 |

## Comments with no usable text after cleaning

2 of 105 comments are empty after cleaning (emoji-only, or written in Devanagari / other non-Latin script). The model can only guess from the bias term for these.

Accuracy on those: 0.000. Accuracy on the rest: 0.602.

## Accuracy by model confidence

| confidence | comments | accuracy |
|---|---|---|
| <0.5 | 4 | 0.250 |
| 0.5-0.7 | 3 | 0.333 |
| 0.7-0.9 | 37 | 0.486 |
| 0.9+ | 61 | 0.689 |

## Most confident mistakes

- mine positive, predicted neutral (0.96): "didi aap bhut acha pdhati hai aisa aapko lgta hai apko bhut galat lgta hai"
- mine negative, predicted neutral (0.96): "Aapki video..vaibhav ke tarah AI generated hai kya?"
- mine neutral, predicted negative (0.95): "QUESTION ? is this for agentic AI too ? mean someone doing agentic ai , can they take this playlist for langchain ?"
- mine positive, predicted neutral (0.94): "Sir ji kya apke udemy ya coursera pe course hai i want the certifications ❤"
- mine positive, predicted neutral (0.94): "mam bohot bolte ho aap, kuch samajh nahi aaya...."
- mine positive, predicted neutral (0.93): "shabbash bachhe! kitni asaan bhasha mein samjhaya aapne. honhaar bitiya."
- mine neutral, predicted positive (0.93): "Mam kindly use full english while teaching becoz we are in other language."
- mine positive, predicted neutral (0.92): "I hope , kuch naya tech stack seekhane mein introduction ke time pre requisites dena chahiye."
- mine positive, predicted neutral (0.92): "All I can say is Thank you ma'am🙌🙌"
- mine positive, predicted neutral (0.92): "Hi Sir, do you have any plans to have Langchain v1.3 refresher as it is architecturally different than v.3"
