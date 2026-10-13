# Error analysis

Test comments: 7333. Mistakes: 921 (accuracy 0.8744). Examples show the cleaned text (stopwords removed).

## Confusion matrix (rows = true, columns = predicted)

| true \ predicted | negative | neutral | positive |
|---|---|---|---|
| negative | 1321 (80%) | 139 (8%) | 190 (12%) |
| neutral | 87 (3%) | 2411 (95%) | 31 (1%) |
| positive | 266 (8%) | 208 (7%) | 2680 (85%) |

## Most common mistakes

- true positive, predicted negative: 266
- true positive, predicted neutral: 208
- true negative, predicted positive: 190
- true negative, predicted neutral: 139
- true neutral, predicted negative: 87
- true neutral, predicted positive: 31

## Accuracy by comment length

| length | comments | accuracy |
|---|---|---|
| 1-5 words | 2293 | 0.936 |
| 6-15 words | 2680 | 0.867 |
| 16-40 words | 1544 | 0.815 |
| 41+ words | 816 | 0.837 |

## Accuracy by model confidence

| confidence | comments | accuracy |
|---|---|---|
| <0.5 | 158 | 0.468 |
| 0.5-0.7 | 756 | 0.624 |
| 0.7-0.9 | 2214 | 0.843 |
| 0.9+ | 4205 | 0.951 |

## Example mistakes

### True positive, predicted negative
- "get called paki time sindhi imagine much worse would person muslim also noting not derogative pakistani but derogative say belong offended rhetoric back pakistan even mildest criticism show divisive becoming even reddit ..." (confidence 0.55)
- "sushma swaraj speech islamic nation oic slam india kashmir 2001785 pakistan yet amazing oic longer irrelevant india hype created pakistan nothing statement attend" (confidence 0.60)
- "ehhhh yes firstly big deal bjp call shot not weak coalition dependent crazy regional party give insanity magic nuclear deal given platter usa cohesion state support party actually execute plan way better optic time make ..." (confidence 0.78)
- "every party ideal one cannot expect every single political party equivalently secular seem like representing single community but end day revolting others india biggest democracy still surviving peacefully blessing" (confidence 0.55)
- "absolutely stupid think nothing narrative pakistan put far indian jet entered territory got beaten back jet dropped payload but missed nothing hit anyways show infiltrate india sent jet none f16s dropped minor payload op..." (confidence 0.65)
- "bjp try change way actually veer towards direction sabka saath sabka vikaas try congress lose state thing get banned phenomenon people returning award reach crescendo major nationwide event gajendra chauhan still chairma..." (confidence 0.71)
- "jinnah made first india pakistan created point people took decision congress dead gone democratic undemocratic five" (confidence 0.77)
- "liberal leftist alliance islamic world mean american liberal obligated ally pakistan dispute india tough situation not mention avg abcd sort push narrative modi basically eastern trump one reason american medium overwhel..." (confidence 0.93)

### True positive, predicted neutral
- "expect man abandoned wife" (confidence 0.90)
- "banned plastic bag 10ish year ago went hilariously tit opposition medium cried murder bag came back let see go" (confidence 0.61)
- "bitch want" (confidence 0.91)
- "nnnngh heart level adorable" (confidence 0.91)
- "incazzare per che gente comporti come fosse una specie complotto per bruciare governo sticazzi del governo sono informazioni che devono sapere non fosse roba far uscire non uscirebbe anche perch stelle sono ormai mainstr..." (confidence 0.40)
- "well dielemma solved brother decided get married voting day city marriage another city vote even managed figure vote" (confidence 0.42)
- "look like stray puppy want kept blame" (confidence 0.88)
- "sound logic folk" (confidence 0.54)

### True negative, predicted positive
- "everything fine except churchill stuff look unfair unproven" (confidence 0.54)
- "not whether bjp properly governing party not but people believe party something curse moan demonetization actually worked big time modi favor strengthening cult bit fearful bjp controlled country opposition capable unner..." (confidence 0.86)
- "remember vajpayee addressed conference hindi say use english leader country like china russia japan etc generally use native language interpreter speaking foreign leader audience" (confidence 0.89)
- "fucking goldmine thanks sharing" (confidence 0.63)
- "actual event not worth story someone think story deleted paranoid say lot quality journalism today quality government fear government" (confidence 0.57)
- "much grinding poverty india billion gold sitting around temple place half people even access clean water wtf man sell gold invest infrastructure" (confidence 0.52)
- "thread farce mod rather friend alt account biggest political propgandists bigot troll india yet account never face action instead reported mod bend backwards defend complainant attack latter" (confidence 0.87)
- "probably check hardcore hindutva page facebook way find reason justify shortcoming current government typical brainwashed shill follower thousand agree said maybe wholly aware fact action directly indirectly responsible ..." (confidence 0.75)

## Most confident mistakes

- true positive, predicted negative (0.99): "sorry deity delicious"
- true neutral, predicted negative (0.99): "fucking traore taking player"
- true positive, predicted negative (0.98): "something wrong patriotic country"
- true positive, predicted negative (0.98): "completely understand frustration family need answer also sincerely hope life lost search remote location bad weather condition"
- true neutral, predicted negative (0.98): "would agree meet influence anymore slug pushed least would waste time"
- true positive, predicted negative (0.98): "let face indian least one vote prioritize caste religion freedom speech fucking low keep going example"
- true positive, predicted negative (0.98): "nonsense least expected remove favored nation status pakistan trade economic sanction least"
- true positive, predicted negative (0.98): "came entertainment comment not disappointed"
- true positive, predicted negative (0.98): "but not long shot"
- true positive, predicted negative (0.98): "situation unfortunate ancestor fault not leave religion desert semite behind try returning tradition kind leave islam marry islam indoctrinate kid islam rather atheist muslim"

## Top words per class (logistic regression coefficients)

- **negative**: bad, hate, stupid, fuck, fucking, idiot, wrong, shit, worst, sad, least, behind, sorry, evil, fucked
- **neutral**: anyone know, chaliye, hai, sarkar, cancer, nda, 2016, find way, setting, morning walk, deleted, rajnath, people like, sending, solved
- **positive**: good, best, lol, great, many, better, win, right, first, thanks, new, love, top, free, sure
