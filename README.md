# FER2013 — Facial Expression Recognition
 
Challenges in Representation Learning: Facial Expression Recognition Challenge (Kaggle).
 
ეს რეპოზიტორია შეიცავს FER2013 dataset-ზე emotion recognition-ის ექსპერიმენტებს. მთავარი მიზანი იყო PyTorch-ში neural network-ებთან მუშაობის პრაქტიკული გამოცდილების მიღება, სხვადასხვა architecture-ისა და hyperparameter-ის გატესტვა, შედეგების ანალიზი და ყველა ექსპერიმენტის Weights & Biases-ზე (W&B) დალოგვა.
 
ექსპერიმენტებს ვაშენებდი იტერატიულად — პატარა architecture-იდან დავიწყე და თანდათან ვამატებდი layer-ებსა და regularization-ს. ჩემი მთავარი ფოკუსი იყო არა მხოლოდ მაღალი accuracy, არამედ ის, რომ ცხადად მეჩვენებინა underfitting და overfitting და გამეგო, რა იწვევდა მათ.
 
- **W&B project:** [fer2013-experiments](https://wandb.ai/njvar23-free-university-of-tbilisi-/fer2013-experiments)
- **W&B report:** _(report-ის link-ი publish-ის შემდეგ ჩასვი აქ)_
---
 
## 1. Dataset და EDA
 
FER2013 შედგება 48×48 grayscale სახის სურათებისგან, 7 emotion კლასით: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral. თითოეული სურათი CSV-ში ინახება space-separated pixel string-ად.
 
- Train: 28,709 sample
- Test: 7,178 sample (labels-ის გარეშე)
რადგან test set-ს labels არ აქვს, local-ად test accuracy-ს ვერ ვითვლი — ამიტომ ქვემოთ ყველა მოყვანილი accuracy არის **validation accuracy**.
 
**მთავარი დაკვირვება — class imbalance.** კლასები ძალიან არათანაბრად არის გადანაწილებული:
 
| Emotion | Train samples |
|---|---|
| Happy | 7,215 |
| Neutral | 4,965 |
| Sad | 4,830 |
| Fear | 4,097 |
| Angry | 3,995 |
| Surprise | 3,171 |
| **Disgust** | **436** |
 
Disgust დანარჩენებზე ~16-ჯერ ნაკლებია. ეს დისბალანსი მთელი პროექტის განმავლობაში გავლენას ახდენდა შედეგებზე (იხ. ქვემოთ confusion matrix-ების ანალიზი) — ამიტომ accuracy-ს გვერდით ყოველთვის ვლოგავ **F1-macro**-საც, რომელიც minority კლასებზე performance-ს უკეთ ასახავს.
 
---
 
## 2. Data pipeline
 
- **`FERDataset`** — custom `Dataset`, რომელიც pixel string-ს კითხულობს, 48×48 array-ად აქცევს და transform-ებს ადებს. test set-ს `has_labels=False`-ით ამუშავებს.
- **Train/val split** — 90/10, stratified by emotion, `random_state=42`. გამოდის 25,838 train / 2,871 val. Stratify-ს იმიტომ ვიყენებ, რომ დაუბალანსებელი კლასების proportion-ები ორივე split-ში შენარჩუნდეს. fixed seed კი იმას უზრუნველყოფს, რომ **ყველა ექსპერიმენტი ერთსა და იმავე split-ზე** შედარდეს.
- **Normalization** — mean 0.5 / std 0.5 (ანუ pixel-ები ~[-1, 1] შუალედში).
- **Augmentation (მხოლოდ train-ზე)** — RandomHorizontalFlip, RandomRotation(±10°), ColorJitter, RandomErasing. Validation-ზე augmentation არ ვადებ. horizontal flip სახეებისთვის უსაფრთხოა (გადაბრუნებული სახე იგივე emotion-ია), rotation/jitter pose/lighting-ის variation-ს ბაძავს, RandomErasing კი occlusion-ისადმი robustness-ს ზრდის.
---
 
## 3. Architecture-ები
 
architecture-ები იტერატიულად ავაშენე — სამი განსხვავებული "ოჯახი", თანდათან მზარდი capacity-თა და quality-თ.
 
### TinyCNN (~149K params)
ყველაზე პატარა baseline: 2 conv layer (8, 16 channel) + ReLU + MaxPool, შემდეგ flatten და 2 fully-connected layer. **არ აქვს** BatchNorm-ი და dropout-ი. ეს განზრახ მინიმალური მოდელია — მისი მიზანი capacity-ის ქვედა ზღვრის დადგენა და underfitting-ის ჩვენებაა.
 
### MediumCNN (~174K params)
უფრო ღრმა: 5 conv block, თითო = Conv → BatchNorm → ReLU, channel-ები 32→64→128. ორ ადგილას MaxPool + Dropout2d, ბოლოს **AdaptiveAvgPool (global average pooling)** flatten-ის ნაცვლად, შემდეგ FC + Dropout + FC.
 
გადაწყვეტილებები და მათი მიზეზი:
- **BatchNorm** — training-ს ასტაბილურებს და აჩქარებს (TinyCNN-ს ეს არ ჰქონდა).
- **Global average pooling** flatten-ის ნაცვლად — ამის გამო მოდელ გაცილებით ღრმაა, მაგრამ param count თითქმის იგივეა, რაც TinyCNN-ს (149K vs 174K). ანუ performance-ის ზრდა capacity quality-დან მოდის და არა უბრალოდ parameter-ების რაოდენობიდან.
- **Dropout** (`dropout_rate` როგორც argument) — სწორედ ეს არის regularization-ის "ჩამრთველი", რომელიც overfit vs regularized შედარების საშუალებას იძლევა.
### EfficientNetFER (~4.34M total / ~330K trainable frozen-ის დროს)
Pretrained EfficientNet-B0 (ImageNet weights) backbone-ად, ახალი classifier head-ით (Dropout → FC → ReLU → Dropout → FC). `freeze_backbone` flag და `unfreeze_backbone()` method two-stage transfer learning-ს უზრუნველყოფს.
 
რადგან FER2013 არის 48×48 grayscale, EfficientNet კი 224×224 RGB-ს ელოდება, `forward`-ში single channel 3-მდე ვამრავლებ და bilinear-ად 224×224-მდე ვზრდი.
 
---
 
## 4. Forward/Backward sanity check-ები
 
training-ამდე სამივე მოდელზე გავუშვი sanity check-ები (ლექციებზე განხილული forward/backward შემოწმება):
- **Forward** — input [8,1,48,48] → output [8,7], NaN/Inf-ის გარეშე. სამივე გადის.
- **Backward / gradient flow** — gradient-ი ყველა trainable layer-ს აღწევს, dead (zero-gradient) layer არ არის. EfficientNet-ს frozen-ის დროს მხოლოდ 4 layer-ს აქვს gradient — ეს ადასტურებს, რომ freezing მუშაობს.
- **Tiny-batch overfit test** (50 step, 8 sample) — TinyCNN და EfficientNet head loss-ს 0-მდე აჩერებენ. **MediumCNN ამ test-ს ვერ აკმაყოფილებს** და warning-ს იძლევა.
MediumCNN-ის "warning" აქ **არ** ნიშნავს, რომ მოდელ გატეხილია (ის შემდეგ საუკეთესო CNN აღმოჩნდა). მიზეზი methodological-ია: BatchNorm 8 sample-იან batch-ზე არასტაბილურია, dropout კი training mode-ში noise-ს ამატებს — random noise input-ის 50 step-ში დამახსოვრება regularized მოდელ-ისგან ლოგიკურად რთულია. ანუ ეს test BatchNorm + dropout მოდელ-ისთვის არასაიმედო heuristic-ია.
 
---
 
## 5. ექსპერიმენტები
 
ყველა ექსპერიმენტი ერთ `train_model` ფუნქციაში გადის — ერთი code path ნიშნავს, რომ ყველა run იდენტურად ლოგავს W&B-ზე (`train/loss`, `train/accuracy`, `val/loss`, `val/accuracy`, `learning_rate`, `gradient_norm`, `train_val_gap`, ბოლოს `best_val_accuracy`, `confusion_matrix`, `val_f1_macro`). ყოველი architecture/config ცალკე run-ია, ცალკე tag-ით.
 
`train_val_gap = train_acc − val_acc` არის მთავარი diagnostic, რომელსაც overfit/underfit-ის შესაფასებლად ვიყენებ.
 
### Exp 1 — TinyCNN baseline
`lr=1e-3, 30 epochs, Adam, augment on`
**Val 55.4% | F1 0.519 | gap +0.019**
 
train (0.573) და val (0.554) ერთმანეთთან ახლოს არიან და ორივე დაბალია — პატარა gap-თან ერთად ეს არის **underfitting**-ის ნიშანი (high bias). მოდელს უბრალოდ არ აქვს საკმარისი capacity. (თავიდან ~40%-ს ველოდი; რეალური 55% უფრო მაღალი გამოვიდა, მაგრამ underfitting-ის დასკვნა accuracy-ის absolute რიცხვიდან კი არა, არამედ პატარა gap-იდან გამომდინარეობს.) ეს ამართლებს შემდეგ ნაბიჯს — მეტი capacity.
 
### Exp 2 — TinyCNN high LR
`lr=1e-2` (10× მეტი), დანარჩენი იგივე
**Val 40.4% | F1 0.312 | gap −0.006**
 
baseline-ზე 15 ქულით ცუდი. loss პირველ ~5 epoch-ში სწრაფად ეცემა და მერე "ჩერდება" ~1.54-ზე, val curve კი ხმაურიანია. ძალიან მაღალი learning rate-ის გამო optimizer ნაბიჯები იმდენად დიდია, რომ კარგ minimum-ში ვერ ჯდება (overshooting). ეს ისევ underfitting-ია, მაგრამ **სხვა მიზეზით** — baseline capacity-ის გამო underfit-დება, ეს კი optimization-ის ჩავარდნის გამო. ამან cosine scheduler-ის გამოყენების მოტივაცია მომცა შემდეგ მოდელებში.
 
### Exp 3 — MediumCNN, no dropout / no augmentation (overfit demo)
`dropout=0.0, augment off, 50 epochs`
**Val 62.8% | F1 0.570 | gap +0.049 (იზრდება)**
 
regularization-ის გარეშე gap თანდათან ფართოვდება (≈0 epoch 25-ზე → +0.049 epoch 50-ზე), train loss ეცემა, val loss კი ფლატდება — ეს overfitting-ის დაწყების ნიშანია. gap არ არის უზარმაზარი, მაგრამ მნიშვნელოვანია, რომ ის **მზარდია** — მეტ epoch-ზე უფრო გაიზრდებოდა.
 
აქ უნდა აღინიშნოს: ამ overfit მოდელის val accuracy (62.8%) უფრო მაღალია, ვიდრე regularized version-ის (60.1%). მიზეზი ის არის, რომ augmentation-ის გარეშე მოდელი ნამდვილ train distribution-ს ერგება (val-იც augmentation-ის გარეშეა), ამიტომ val accuracy ერთ კონკრეტულ epoch-ზე მაღალია — მაგრამ ამ accuracy-ს memorization-ის ხარჯზე ყიდულობს, რასაც მზარდი gap ცხადყოფს.
 
### Exp 4 — MediumCNN, regularized (balanced)
`dropout=0.5, augment on, weight_decay=1e-4, cosine, 60 epochs`
**Val 60.1% | F1 0.489 | gap −0.022 (სტაბილური)**
 
Exp 3 — იგივე architecture, regularization ჩართული. ყველაზე მნიშვნელოვანი შედეგი: gap-ის ნიშანი იცვლება. **+0.049 (მზარდი) → −0.022 (სტაბილური).** dropout + augmentation + weight decay-მ overfitting-ი არა მხოლოდ შეამცირა, არამედ შეაბრუნა — model val-ზე (augmentation-ის გარეშე) უკეთ მუშაობს, ვიდრე train-ზე (augmentation-ით).
 
regularized მოდელს val accuracy ცოტა დაბალი აქვს, მაგრამ ეს იმიტომ, რომ 60 epoch-ზე ჯერ არ დასრულებულა convergence (val ბოლო epoch-ზეც იზრდებოდა) და negative gap headroom-ს მიუთითებს. ანუ regularization-მა accuracy-ს ცოტა "მსხვერპლის" სანაცვლოდ მოდელი უსაფრთხოდ generalizable გახადა.
 
შენიშვნა: F1-macro (0.489) აქ no-dropout-ზე (0.570) დაბალია — ძლიერი augmentation + dropout მოდელს უფრო "კონსერვატიულს" ხდის და majority კლასებზე ეყრდნობა. ანუ `balanced` tag bias-variance-ს ეხება და არა class balance-ს.
 
### Exp 5 — MediumCNN, SGD vs Adam
`optimizer=sgd, lr=1e-1`, დანარჩენი regularized config-ის იგივე
**Val 59.8% | F1 0.482 | gap −0.026**
 
SGD (lr 0.1) და Adam (lr 0.001) თითქმის ერთსა და იმავე ადგილას მიდიან (59.8% vs 60.1%, F1-იც პრაქტიკულად იგივე). განსხვავება მხოლოდ ისაა, რომ **SGD ნელა იწყებს** — პირველ ~5–10 epoch-ში Adam უსწრებს, მაგრამ ბოლოს SGD ეწევა. ეს Adam-ის adaptive learning rate-ის კლასიკური ეფექტია. დასკვნა: ამ task-ისთვის optimizer არ არის bottleneck — ceiling-ს architecture/regularization განსაზღვრავს.
 
### Exp 6 — Batch size sweep (MediumCNN regularized)
მხოლოდ batch size იცვლება, დანარჩენი იგივეა.
 
| Batch size | Val Acc | F1-macro | gap |
|---|---|---|---|
| 32 | **61.8%** | 0.504 | −0.027 |
| 64 | 60.1% | 0.489 | −0.022 |
| 128 | 59.4% | 0.477 | −0.020 |
 
უფრო პატარა batch → უფრო მაღალი val accuracy. პატარა batch მეტ update step-ს და უფრო "ხმაურიან" gradient-ს ნიშნავს, რაც აქ mild regularization-ად მუშაობს. მაგრამ ეს ნაწილობრივ **confounded**-ია: learning rate ფიქსირებული იყო (1e-3) სამივესთვის, ხოლო linear scaling rule-ის მიხედვით bs=128-ს proportional-ად მაღალი lr სჭირდება. ანუ bs=128-ის ჩამორჩენის ნაწილი სავარაუდოდ un-scaled learning rate-ია და არა დიდი batch-ის "თანდაყოლილი" ნაკლი. ასევე: bs=32 wall-clock-ში ყველაზე ნელი იყო (4× მეტი step).
 
### Exp 7 — EfficientNet, frozen backbone
`backbone frozen, 30 epochs, AdamW lr=1e-3, cosine`
**Val 48.4% | F1 0.406 | gap −0.060**
 
frozen EfficientNet **ყველაზე ცუდია** ყველა "ნამდვილ" მოდელს შორის — TinyCNN-საც კი ჩამორჩება. მიზეზი **domain mismatch**-ია: EfficientNet-ის frozen feature-ები ImageNet-ზე (ფერადი ბუნებრივი ობიექტები) ისწავლა, FER2013 კი დაბალი გარჩევადობის grayscale სახეებია. რადგან backbone ვერ ადაპტირდება, მოდელი generic feature-ებითაა "ჩარჩენილი". დიდი negative gap (−0.060) და დაბალი train accuracy (0.42) ცხადყოფს, რომ ეს **underfitting**-ია (model train set-საც კი ვერ ერგება). სწორედ ამიტომ შემდეგი ნაბიჯია — unfreezing.
 
### Exp 8 — EfficientNet, two-stage full fine-tune (საუკეთესო მოდელი)
Stage 1: head-only warmup, 10 epoch, AdamW lr 1e-3. Stage 2: unfreeze backbone, AdamW lr 1e-4, cosine, patience 6.
**Val 69.9% | F1 0.679 | gap +0.109 (managed)**
 
ეს არის საუკეთესო მოდელ. curve-ის ისტორია ცხადია:
- **Stage 1 (frozen head):** val 0.59 → 0.69, head ერგება frozen feature-ებს.
- **Unfreeze moment (epoch 11):** `gradient_norm` მკვეთრად ხტება, val loss ერთ ნაბიჯში ეცემა — backbone იწყებს ადაპტაციას.
- **Stage 2:** train 0.81-მდე იზრდება, val ~0.70-ზე რჩება, gap +0.109-მდე ფართოვდება. early stopping epoch 14-ზე ირთვება და overfitting-ის დასაწყისს იჭერს.
მთავარი შედეგი: **frozen → fine-tuned: 48.4% → 69.9%, +21.5 ქულა.** იგივე architecture, იგივე head — ერთადერთი განსხვავება ისაა, ადაპტირდება თუ არა backbone. ეს არის პროექტის ყველაზე ძლიერი დემონსტრაცია იმისა, თუ რას აძლევს fine-tuning. ბოლოს გამოჩენილი mild overfitting რეალურია, მაგრამ early stopping-მა, dropout-მა და weight decay-მ კონტროლქვეშ დაიჭირა — ამით განსხვავდება Exp 3-ის "უკონტროლო" მზარდი gap-ისგან.
 
---
 
## 6. შედეგების შეჯამება
 
| Experiment | Val Acc | F1-macro | Gap | დასკვნა |
|---|---|---|---|---|
| EfficientNet full finetune | **69.9%** | 0.679 | +0.109 | საუკეთესო / managed overfit |
| MediumCNN bs=32 | 61.8% | 0.504 | −0.027 | საუკეთესო CNN |
| MediumCNN no-dropout | 62.8% | 0.570 | +0.049 | overfit (მზარდი gap) |
| MediumCNN regularized | 60.1% | 0.489 | −0.022 | balanced |
| MediumCNN SGD | 59.8% | 0.482 | −0.026 | balanced |
| MediumCNN bs=128 | 59.4% | 0.477 | −0.020 | batch sweep |
| TinyCNN baseline | 55.4% | 0.519 | +0.019 | mild underfit |
| EfficientNet frozen | 48.4% | 0.406 | −0.060 | underfit (frozen feats) |
| TinyCNN high-LR | 40.4% | 0.312 | −0.006 | underfit / bad optimization |
 
---
 
## 7. Overfitting / Underfitting ანალიზი
 
ექსპერიმენტები bias-variance spectrum-ზე ასე განლაგდა:
 
- **Underfitting:** TinyCNN high-LR (optimization-ის ჩავარდნა), EfficientNet frozen (domain mismatch, ვერ ადაპტირებადი feature-ები), TinyCNN baseline (capacity ნაკლებობა). სამივეს დაბალი accuracy + ≈0 ან negative gap.
- **კარგად regularized შუა:** MediumCNN-ის ოჯახი (59–62%, negative gap-ები) — overfitting-ი regularization-მა გააკონტროლა.
- **Overfitting:** MediumCNN no-dropout (+0.049, მზარდი, უკონტროლო) და EfficientNet fine-tune (+0.109, მაგრამ early stopping-ით კონტროლირებული).
მნიშვნელოვანი დაკვირვება: val accuracy ცალკე აღებული შეცდომაში შეგვიყვანდა (no-dropout model-ს მაღალი accuracy აქვს, მაგრამ overfit-დება). სწორედ ამიტომ gap-ი და F1-macro საჭიროა სრული სურათისთვის.
 
---
 
## 8. Class imbalance — confusion matrix-ების ანალიზი
 
**MediumCNN (regularized):**
- მთელი **Disgust column ნულია** — მოდელი არც ერთ validation sample-ს არ აფრედიქთებს Disgust-ად. ეს იყო `UndefinedMetricWarning`-ის წყარო ყველა CNN run-ში. 436 Disgust sample-ით (vs 7,215 Happy) მოდელმა "ისწავლა", რომ Disgust-ის predict არასდროს ღირს.
- Happy ყველაზე ძლიერია (619 სწორი). Fear ყველაზე სუსტი — მხოლოდ 62 სწორი, დანარჩენი Sad/Surprise/Neutral-ში იფანტება. Sad↔Neutral confusion მძიმეა. ეს ის emotion წყვილებია, რომლებიც ადამიანებსაც კი ერევათ დაბალი გარჩევადობის სახეებზე.
**EfficientNet (fine-tuned):**
- **Disgust აღარ არის "მკვდარი":** ~26 სწორი predict (vs 0 MediumCNN-თან). fine-tuned backbone-ს საკმარისად მდიდარი feature-ები აქვს პატარა კლასისთვისაც. სწორედ ეს ხსნის F1-macro-ის ნახტომს 0.679-მდე.
- ყველა კლასზე გაუმჯობესება: Fear 220 (vs 62), Angry 255 (vs 213), Sad 295 (vs 259). residual confusion (Fear→Sad/Neutral, Sad↔Neutral) რჩება, მაგრამ უფრო რბილია — ანუ ეს confusion-ები task-ის თანდაყოლილი სირთულეა და არა მხოლოდ მოდელ-ის სისუსტე.
დასკვნა: accuracy imbalance-ის ისტორიას მალავს. ყველა CNN-მა F1-macro ~0.48–0.52 მიიღო ~60% accuracy-ის მიუხედავად, რადგან Disgust-ს თმობდნენ. მხოლოდ fine-tuned EfficientNet-მა აღადგინა ის (F1 0.679).
 
---
 
## 9. დასკვნები
 
- Architecture quality > raw parameter count: MediumCNN-მა TinyCNN-ს აჯობა თითქმის იგივე param count-ით (BatchNorm, depth, GAP).
- Regularization-ი (dropout + augmentation + weight decay) overfitting-ს აკონტროლებს — gap-ის ნიშანი იცვლება.
- Hyperparameter-ები მნიშვნელოვანია: learning rate-მა იგივე მოდელ-ზე 15 ქულა შეცვალა; batch size-მა ~2.4 ქულა.
- Optimizer (SGD vs Adam) ამ task-ისთვის bottleneck არ იყო.
- Transfer learning-ის ნამდვილი ეფექტი fine-tuning-ში იყო (backbone-ის unfreezing), არა frozen feature-ებში: +21.5 ქულა.
- Class imbalance F1-macro-სა და confusion matrix-ით ჩანს და არა accuracy-თ.
---
```
 
