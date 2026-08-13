# Mooré-Voice v0 results

_Generated 2026-08-13T09:06:04+00:00 by `scripts/make_results_doc.py`. Do not edit the tables by hand._

## Translation — FLORES-200 devtest (1,012 sentences/direction)

chrF++ is the primary metric for Mooré; BLEU shown for comparability.

### chrF++

| Model | mos→fra | mos→eng | fra→mos | eng→mos |
|---|---:|---:|---:|---:|
| NLLB-600M zero-shot | 26.49 | 29.01 | 20.37 | 20.5 |
| NLLB-600M + LoRA v0 | 29.16 | 30.13 | 22.01 | 21.82 |
| NLLB-3.3B zero-shot | 29.86 | 32.01 | 22.82 | 23.77 |
| NLLB-3.3B + LoRA v0 | 32.66 | 33.06 | 23.32 | 23.41 |

### BLEU

| Model | mos→fra | mos→eng | fra→mos | eng→mos |
|---|---:|---:|---:|---:|
| NLLB-600M zero-shot | 7.04 | 8.71 | 2.54 | 2.94 |
| NLLB-600M + LoRA v0 | 8.32 | 8.92 | 3.1 | 3.19 |
| NLLB-3.3B zero-shot | 9.14 | 10.77 | 3.18 | 3.72 |
| NLLB-3.3B + LoRA v0 | 11.24 | 11.86 | 3.5 | 3.88 |

## Speech recognition — held-out test split

| Model | WER ↓ | CER ↓ | n |
|---|---:|---:|---:|
| MMS-1b-all (mos adapter, zero-shot) | 0.3111 | 0.086 | 564 |
| Whisper-small fine-tuned v0 | 0.3408 | 0.1136 | 564 |

## Samples (fine-tuned)

**eng_Latn→mos_Latn**
- src: "We now have 4-month-old mice that are non-diabetic that used to be diabetic," he added.
- hyp: A paasame: " Masã, d tara piis sẽn tar kiis a naas sẽn pa tar sik-m - meng sẽn da tar sik-m - meng pĩndã. "
- ref: "Ed tara moasã kiuugã 4 yõense sẽn n ka tare sɩkra bãaga sẽn n reeg dag n tare bãaga," a n paase.

**eng_Latn→mos_Latn**
- src: Dr. Ehud Ur, professor of medicine at Dalhousie University in Halifax, Nova Scotia and chair of the clinical and scientific division of the Canadian Diabetes Association cautioned that the research is still in its early days.
- hyp: Doktɛɛr a Ehud Ur sẽn yaa logtoɛɛmbã profesɛɛr Dalhousie Inivɛrsite sẽn be Halifax, Nova Scotia, la Kanada Sãn-kɛglã sull taoor soab keoogame tɩ vaeesgã nan pa sɩng ye.
- ref: Logtor a Ehud Ur, sẽn n yaa Logtoɛɛmb karen-saambã Dalhousie University Halifax pʋgẽ, Nouvelle-Ecosse la Kanada Diabɛt siãnsã tẽnga sullã taoor soaba keoogame tɩ vɛɛsgra na yaa paalm.

**fra_Latn→mos_Latn**
- src: «Nous avons à présent des souris de 4 mois qui ne sont pas diabétiques alors qu'elles l'étaient auparavant», a-t-il ajouté.
- hyp: " Tõnd tara rũms sẽn tar kiis a 4 sẽn pa tar diabɛt n yaool n da tar-a pĩnda, " a paasame.
- ref: "Ed tara moasã kiuugã 4 yõense sẽn n ka tare sɩkra bãaga sẽn n reeg dag n tare bãaga," a n paase.

**fra_Latn→mos_Latn**
- src: Le Dr Ehud Ur, professeur de médecine à l'Université Dalhousie de Halifax (Nouvelle-Écosse) et président de la division clinique et scientifique de l'Association canadienne du diabète, a averti que la recherche en était encore à ses débuts.
- hyp: A Dr Ehud Ur sẽn yaa logtoɛɛmbã profesɛɛr Inivɛrsite Dalhousie sẽn be Halifax, Nova Scotia, la Kanada Sãn-kɛglã sull taoor soaba, keooga nebã tɩ vaeesgã nan ket n sɩngdame.
- ref: Logtor a Ehud Ur, sẽn n yaa Logtoɛɛmb karen-saambã Dalhousie University Halifax pʋgẽ, Nouvelle-Ecosse la Kanada Diabɛt siãnsã tẽnga sullã taoor soaba keoogame tɩ vɛɛsgra na yaa paalm.

**mos_Latn→eng_Latn**
- src: "Ed tara moasã kiuugã 4 yõense sẽn n ka tare sɩkra bãaga sẽn n reeg dag n tare bãaga," a n paase.
- hyp: " Ed is now 4 months pregnant with non-diabetic breast cancer, " he added.
- ref: "We now have 4-month-old mice that are non-diabetic that used to be diabetic," he added.

**mos_Latn→eng_Latn**
- src: Logtor a Ehud Ur, sẽn n yaa Logtoɛɛmb karen-saambã Dalhousie University Halifax pʋgẽ, Nouvelle-Ecosse la Kanada Diabɛt siãnsã tẽnga sullã taoor soaba keoogame tɩ vɛɛsgra na yaa paalm.
- hyp: Dr. Ehud Ur, professor of medicine at Dalhousie University in Halifax, Nova Scotia and president of the Canadian Diabetes Science Association, warned that the virus could be new.
- ref: Dr. Ehud Ur, professor of medicine at Dalhousie University in Halifax, Nova Scotia and chair of the clinical and scientific division of the Canadian Diabetes Association cautioned that the research is still in its early days.

**mos_Latn→fra_Latn**
- src: "Ed tara moasã kiuugã 4 yõense sẽn n ka tare sɩkra bãaga sẽn n reeg dag n tare bãaga," a n paase.
- hyp: " Ed a maintenant 4 mois de diabète non contrôlé, " ajoute-t-il.
- ref: «Nous avons à présent des souris de 4 mois qui ne sont pas diabétiques alors qu'elles l'étaient auparavant», a-t-il ajouté.

**mos_Latn→fra_Latn**
- src: Logtor a Ehud Ur, sẽn n yaa Logtoɛɛmb karen-saambã Dalhousie University Halifax pʋgẽ, Nouvelle-Ecosse la Kanada Diabɛt siãnsã tẽnga sullã taoor soaba keoogame tɩ vɛɛsgra na yaa paalm.
- hyp: Le Dr Ehud Ur, professeur de médecine à l'Université Dalhousie à Halifax, Nouvelle-Écosse et président de l'Association nationale canadienne de la science du diabète, a mis en garde contre la propagation du virus.
- ref: Le Dr Ehud Ur, professeur de médecine à l'Université Dalhousie de Halifax (Nouvelle-Écosse) et président de la division clinique et scientifique de l'Association canadienne du diabète, a averti que la recherche en était encore à ses débuts.

## Native-speaker rating sheet (to fill)

Rate each fine-tuned output 1–5 for Fluency (natural Mooré?) and
Adequacy (meaning preserved?). Draw 20 sentences from the samples
above plus everyday domains (market, clinic, agriculture).

| # | Direction | Source | Model output | Fluency | Adequacy |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
