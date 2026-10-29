# Replacement Unanswerable Candidates

These five questions are proposals for human review. They are not added to the
candidate JSON, have no assigned split, and have not been used in retrieval
experiments.

## 1. Vayyar Imaging revenue

**Question:** What was Vayyar Imaging Ltd.'s total revenue for calendar year 2019, when the Walabot-HOME Reseller Agreement was effective?

**Relevant corpus file:**

- `data/raw/corpus/cuad/HealthcareIntegratedTechnologiesInc_20190812_8-K_EX-10.1_11776966_EX-10.1_Reseller Agreement.txt`

**Evidence searched:** The full corpus was searched for `Vayyar Imaging`, `Vayyar`, `Walabot-HOME`, `revenue`, `annual revenue`, `financial statements`, and `sales`. Vayyar and Walabot-HOME occur only in the reseller agreement. In that agreement, Vayyar is identified as the Supplier and the text discusses reseller sales meetings and product sales. The only revenue hit in the lead file is in a limitation-of-liability exclusion for loss of revenue, not a financial result.

**Why unanswerable:** The agreement contains no Vayyar revenue figure or financial statement. Revenue hits elsewhere in the corpus belong to unrelated contracts and do not establish Vayyar's revenue.

**Confidence:** High. This is a specific external financial fact, distinct from the existing Flotek quarterly-revenue question.

## 2. Bon-Ton exchange rate

**Question:** What was the USD/CAD exchange rate on April 18, 2018, the execution date of the Bon-Ton Agency Agreement?

**Relevant corpus file:**

- `data/raw/corpus/cuad/BONTONSTORESINC_04_20_2018-EX-99.3-AGENCY AGREEMENT.txt`

**Evidence searched:** The full corpus was searched for `Bon-Ton`, `Bon Ton`, `exchange rate`, `foreign exchange`, `USD`, `CAD`, `currency`, and `conversion`. The agreement is dated April 18, 2018. Section 15.6 states that references to dollars in the agreement and its schedules and exhibits mean U.S. dollars. It does not state an exchange rate or a Canadian-dollar conversion.

**Why unanswerable:** The requested historical foreign-exchange rate is external market data. The contract's currency-definition clause does not supply that rate.

**Confidence:** High. This is distinct from the existing natural-gas price question and does not ask for a contract term.

## 3. PCQuote subscriber count

**Question:** How many AB Wately subscribers accessed the PCQ software and Hyperfeed on October 11, 1996, the effective date of the PCQuote co-branding agreement?

**Relevant corpus file:**

- `data/raw/corpus/cuad/PcquoteComInc_19990721_S-1A_EX-10.11_6377149_EX-10.11_Co-Branding Agreement1.txt`

**Evidence searched:** The full corpus was searched for `PC Quote`, `PCQ`, `AB Wately`, `ABW`, `subscriber count`, `number of subscribers`, `subscribers`, `customers`, `users`, and `installed base`. The agreement identifies PCQ and ABW and says the server will be accessed by “all ABW subscribers” to download the software and receive the Hyperfeed. No subscriber count is provided.

**Why unanswerable:** The contract establishes that ABW subscribers have access, but supplies no number of subscribers on the effective date. The question is not a rewording of the existing product-launch question.

**Confidence:** High. The exact parties and effective date are in the contract; the requested operational count is absent.

## 4. On2 Technologies revenue

**Question:** What was On2 Technologies, Inc.'s total revenue for calendar year 2005, when the Support and Maintenance Agreement with Wildform was effective?

**Relevant corpus file:**

- `data/raw/corpus/cuad/ON2TECHNOLOGIES,INC_11_17_2006-EX-10.3-SUPPORT AND MAINTENANCE AGREEMENT.txt`

**Evidence searched:** The full corpus was searched for `On2 Technologies`, `ON2`, `annual revenue`, `revenue`, `financial statements`, `sales`, and `income`. The agreement identifies On2 and Wildform and is a 2005 Support and Maintenance Agreement. The lead file has no revenue or annual-revenue figure; its sales-related language concerns sale of payment shares and securities-law restrictions.

**Why unanswerable:** On2's annual corporate revenue is external financial information, not a support obligation or contract value in the agreement.

**Confidence:** High. This is distinct from the existing Flotek quarterly-revenue question because it concerns a different entity and annual total.

## 5. Premier Biomedical treatment outcome

**Question:** How many patients had been treated with the COVID-19 treatment contemplated by the Premier Biomedical and Marv Enterprises agreement by May 12, 2020?

**Relevant corpus file:**

- `data/raw/corpus/cuad/PREMIERBIOMEDICALINC_05_14_2020-EX-10.2-INTELLECTUAL PROPERTY AGREEMENT.txt`

**Evidence searched:** The full corpus was searched for `Premier Biomedical`, `Premier`, `Marv Enterprises`, `patients treated`, `clinical outcome`, `efficacy`, `approval`, and related treatment terms. The agreement is effective May 12, 2020 and concerns the collective development, implementation, and commercialization of a potential COVID-19 treatment. It contains licensing and development language, but no patient-treatment count or clinical outcome. The one full-corpus hit for the phrase `clinical efficacy` is in an unrelated Innoviva collaboration agreement, not this file.

**Why unanswerable:** A later patient-treatment count is an external or post-execution clinical outcome. The agreement describes a potential treatment and does not establish that patients had been treated by the stated date.

**Confidence:** High. This is distinct from the existing FDA-approval and sales-increase questions and is not answered by the agreement's licensing provisions.

## Duplicate and ambiguity check

The five proposals were compared against all 39 current questions. They do not duplicate the existing questions:

- Proposal 1 asks Vayyar's annual revenue, not Flotek's quarterly revenue.
- Proposal 2 asks a foreign-exchange rate, not a gas price.
- Proposal 3 asks a subscriber count, not whether a product launched.
- Proposal 4 asks On2's annual revenue, not Flotek's quarterly revenue.
- Proposal 5 asks a patient-treatment outcome, not FDA approval or sales increase.

The questions intentionally ask facts outside the supplied contract text. They should still receive human review before being added to the evaluation JSON, especially for the date interpretation in Proposal 5.