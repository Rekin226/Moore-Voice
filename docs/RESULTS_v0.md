# Mooré-Voice v0 results

_Generated 2026-08-13T03:12:24+00:00 by `scripts/make_results_doc.py`. Do not edit the tables by hand._

## Translation — FLORES-200 devtest (1,012 sentences/direction)

chrF++ is the primary metric for Mooré; BLEU shown for comparability.

### chrF++

| Model | mos→fra | mos→eng | fra→mos | eng→mos |
|---|---:|---:|---:|---:|
| NLLB-600M zero-shot | 26.49 | 29.01 | 20.37 | 20.5 |
| NLLB-600M + LoRA v0 | 29.16 | 30.13 | 22.01 | 21.82 |

### BLEU

| Model | mos→fra | mos→eng | fra→mos | eng→mos |
|---|---:|---:|---:|---:|
| NLLB-600M zero-shot | 7.04 | 8.71 | 2.54 | 2.94 |
| NLLB-600M + LoRA v0 | 8.32 | 8.92 | 3.1 | 3.19 |

## Speech recognition — held-out test split

| Model | WER ↓ | CER ↓ | n |
|---|---:|---:|---:|

## Samples (fine-tuned)

**eng_Latn→mos_Latn**
- src: "We now have 4-month-old mice that are non-diabetic that used to be diabetic," he added.
- hyp: A paasame: " Rũndã-rũndã, d tara bõn-bil sẽn tar kiis a naas sẽn pa tar diabɛt sẽn da tar diabɛtã. "
- ref: "Ed tara moasã kiuugã 4 yõense sẽn n ka tare sɩkra bãaga sẽn n reeg dag n tare bãaga," a n paase.

**eng_Latn→mos_Latn**
- src: Dr. Ehud Ur, professor of medicine at Dalhousie University in Halifax, Nova Scotia and chair of the clinical and scientific division of the Canadian Diabetes Association cautioned that the research is still in its early days.
- hyp: Doktɛɛr Ehud Ur sẽn yaa logtor sẽn zãmsd tɩɩm Yunivɛrsite a Dalhousie sẽn be Halifax, Nova Scotia wã, la sẽn yaa Kanada diabɛt sull sẽn get bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag bãag
- ref: Logtor a Ehud Ur, sẽn n yaa Logtoɛɛmb karen-saambã Dalhousie University Halifax pʋgẽ, Nouvelle-Ecosse la Kanada Diabɛt siãnsã tẽnga sullã taoor soaba keoogame tɩ vɛɛsgra na yaa paalm.

**fra_Latn→mos_Latn**
- src: «Nous avons à présent des souris de 4 mois qui ne sont pas diabétiques alors qu'elles l'étaient auparavant», a-t-il ajouté.
- hyp: A paasame: " Tõnd tara kols sẽn tar kiuug a 4 sẽn pa tar diabɛt wa b sẽn da tar-a pĩndã.
- ref: "Ed tara moasã kiuugã 4 yõense sẽn n ka tare sɩkra bãaga sẽn n reeg dag n tare bãaga," a n paase.

**fra_Latn→mos_Latn**
- src: Le Dr Ehud Ur, professeur de médecine à l'Université Dalhousie de Halifax (Nouvelle-Écosse) et président de la division clinique et scientifique de l'Association canadienne du diabète, a averti que la recherche en était encore à ses débuts.
- hyp: Dr Ehud Ur sẽn yaa logtor sẽn be Dalhousie Yunivɛrsite wã sẽn be Halifax, Nova Scotia wã, n yaa Kanada Diabɛt Sosiete wã logtoɛɛg la bãngd sull taoor soaba, keoogame tɩ b ra ket n baoodame n na n bãng b sẽn na n maan to-to wã.
- ref: Logtor a Ehud Ur, sẽn n yaa Logtoɛɛmb karen-saambã Dalhousie University Halifax pʋgẽ, Nouvelle-Ecosse la Kanada Diabɛt siãnsã tẽnga sullã taoor soaba keoogame tɩ vɛɛsgra na yaa paalm.

**mos_Latn→eng_Latn**
- src: "Ed tara moasã kiuugã 4 yõense sẽn n ka tare sɩkra bãaga sẽn n reeg dag n tare bãaga," a n paase.
- hyp: "Ed is now 4 months pregnant with her first child, who was diagnosed with diabetes", she adds.
- ref: "We now have 4-month-old mice that are non-diabetic that used to be diabetic," he added.

**mos_Latn→eng_Latn**
- src: Logtor a Ehud Ur, sẽn n yaa Logtoɛɛmb karen-saambã Dalhousie University Halifax pʋgẽ, Nouvelle-Ecosse la Kanada Diabɛt siãnsã tẽnga sullã taoor soaba keoogame tɩ vɛɛsgra na yaa paalm.
- hyp: Dr Ehud Ur, Professor of Medicine at Dalhousie University in Halifax, Nova Scotia and head of the National Diabetes Science Board of Canada, warned that the explanation would be new.
- ref: Dr. Ehud Ur, professor of medicine at Dalhousie University in Halifax, Nova Scotia and chair of the clinical and scientific division of the Canadian Diabetes Association cautioned that the research is still in its early days.

**mos_Latn→fra_Latn**
- src: "Ed tara moasã kiuugã 4 yõense sẽn n ka tare sɩkra bãaga sẽn n reeg dag n tare bãaga," a n paase.
- hyp: " Ed a maintenant 4 mois de maladie du diabète, " ajoute-t-il.
- ref: «Nous avons à présent des souris de 4 mois qui ne sont pas diabétiques alors qu'elles l'étaient auparavant», a-t-il ajouté.

**mos_Latn→fra_Latn**
- src: Logtor a Ehud Ur, sẽn n yaa Logtoɛɛmb karen-saambã Dalhousie University Halifax pʋgẽ, Nouvelle-Ecosse la Kanada Diabɛt siãnsã tẽnga sullã taoor soaba keoogame tɩ vɛɛsgra na yaa paalm.
- hyp: Le Dr Ehud Ur, professeur de médecine à l'Université Dalhousie à Halifax, chef du National Diabetes Science Board de Nouvelle-Écosse et du Canada, a mis en garde contre une nouvelle version.
- ref: Le Dr Ehud Ur, professeur de médecine à l'Université Dalhousie de Halifax (Nouvelle-Écosse) et président de la division clinique et scientifique de l'Association canadienne du diabète, a averti que la recherche en était encore à ses débuts.

## Native-speaker rating sheet (to fill)

Rate each fine-tuned output 1–5 for Fluency (natural Mooré?) and
Adequacy (meaning preserved?). Draw 20 sentences from the samples
above plus everyday domains (market, clinic, agriculture).

| # | Direction | Source | Model output | Fluency | Adequacy |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
