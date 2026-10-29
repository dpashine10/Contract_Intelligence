#!/usr/bin/env python3
"""Create and validate the unanswerable query set for the CUAD mini benchmark."""

import json
import hashlib
from pathlib import Path
from typing import Any


def assign_split(query: str) -> str:
    """Assign a query to train or test using the prescribed SHA-256 rule."""
    h = int(hashlib.sha256(query.strip().encode("utf-8")).hexdigest(), 16)
    return "test" if h % 10 < 3 else "train"


def normalize_question(q: str) -> str:
    """Normalize question for deduplication."""
    return " ".join(q.strip().lower().split())


def load_benchmark_contracts() -> set[str]:
    """Load the set of contract file paths in the mini benchmark."""
    from app.data import load_benchmark
    tests = load_benchmark()
    contracts = set()
    for test in tests:
        for snippet in test["snippets"]:
            contracts.add(snippet["file_path"])
    return contracts


CANDIDATE_QUESTIONS = [
    # Category A: Entities/events NOT in any indexed contract
    {
        "question": "What are the terms of the licensing agreement between Microsoft Corporation and OpenAI Inc.?",
        "rationale": "Neither Microsoft nor OpenAI appear as parties in any of the 29 indexed contracts."
    },
    {
        "question": "What is the expiration date of the merger agreement between Tesla Inc. and SolarCity Corporation?",
        "rationale": "Tesla and SolarCity do not appear in any indexed contract; the corpus contains no merger agreements."
    },
    {
        "question": "Does the collaboration agreement between Pfizer Inc. and BioNTech SE contain a force majeure clause?",
        "rationale": "Pfizer and BioNTech are not parties to any indexed contract."
    },
    {
        "question": "What royalty rate does Apple Inc. pay to Qualcomm Incorporated under their patent license?",
        "rationale": "Apple and Qualcomm do not appear in the indexed contracts."
    },
    {
        "question": "What are the data protection obligations in the cloud services agreement between Amazon Web Services and Netflix Inc.?",
        "rationale": "AWS and Netflix are not parties to any indexed contract; no cloud services agreements are present."
    },
    {
        "question": "What is the termination notice period in the distribution agreement between Coca-Cola Company and Monster Beverage Corporation?",
        "rationale": "Coca-Cola and Monster Beverage are not parties to any indexed contract."
    },

    # Category B: Commercial terms NOT covered in these specific contracts
    {
        "question": "What is the employee stock option pool percentage in the joint venture between Veoneer AB and Nissin Kogyo Co., Ltd.?",
        "rationale": "The Veoneer/Nissin JV amendment (VEONEER,INC_02_21_2020) covers termination terms, not equity structure or employee stock options."
    },
    {
        "question": "What are the clinical trial milestone payments in the strategic alliance between Sucampo Pharmaceuticals and R-Tech Ueno?",
        "rationale": "The Sucampo/R-Tech strategic alliance (SUCAMPOPHARMACEUTICALS,INC_11_04_2015) is indexed but the annotated snippets cover definitions and transaction terms, not clinical milestone payments."
    },
    {
        "question": "What is the minimum purchase commitment for the reseller in the Walabot-HOME Reseller Agreement?",
        "rationale": "The Walabot-HOME Reseller Agreement (HealthcareIntegratedTechnologiesInc) is indexed but the annotated snippets do not cover purchase commitments."
    },
    {
        "question": "What are the indemnification caps in the Sponsorship Agreement between Jacksonville Jaguars LLC and The ARC Group Inc.?",
        "rationale": "The Jaguars/ARC sponsorship (ArcGroupInc_20171211) is indexed but the snippets focus on sponsorship benefits, not indemnification caps."
    },
    {
        "question": "What is the revenue share percentage in the Co-Branding Agreement between RSL COM PrimeCall and deltathree.com?",
        "rationale": "The PrimeCall/DeltaThree co-branding agreement (DeltathreeInc_19991102) is indexed but annotated snippets cover term and services, not revenue sharing."
    },
    {
        "question": "What are the service level availability targets in the Support and Maintenance Agreement between On2 Technologies and Wildform Inc.?",
        "rationale": "The On2/Wildform support agreement (ON2TECHNOLOGIES,INC_11_17_2006) is indexed but annotated snippets do not cover SLA availability targets."
    },
    {
        "question": "What is the geographic territory for the exclusive license in the Intellectual Property Agreement between Marv Enterprises and Premier Biomedical?",
        "rationale": "The Premier Biomedical IP agreement (PREMIERBIOMEDICALINC_05_14_2020) is indexed but the snippets cover COVID-19 treatment development, not geographic license territory."
    },
    {
        "question": "What are the minimum annual marketing spend requirements in the Marketing Agreement between Loop Industries and Indorama Loop Technologies?",
        "rationale": "The Loop/Indorama marketing agreement (Loop Industries, Inc. - Marketing Agreement) is indexed but annotated snippets do not cover marketing spend minimums."
    },
    {
        "question": "What is the earn-out formula in the Joint Venture Agreement between Tate & Lyle Fermentation Products and IGene Biotechnology?",
        "rationale": "The Tate & Lyle/IGene JV (IGENEBIOTECHNOLOGYINC_05_13_2003) is indexed but annotated snippets do not cover earn-out provisions."
    },

    # Category C: Cross-contract knowledge (requires information across multiple contracts)
    {
        "question": "Do any of the indexed strategic alliance agreements contain identical force majeure language?",
        "rationale": "This requires comparing force majeure clauses across DUOSTECHNOLOGIES and SUCAMPOPHARMACEUTICALS strategic alliances, which is not answerable from any single contract's annotated snippets."
    },
    {
        "question": "Which of the indexed joint venture agreements has the longest initial term?",
        "rationale": "Requires comparing term lengths across VEONEER, IGENE, MJBIOTECH, and PREMIERBIOMEDICAL JV agreements; no single contract answers this comparative question."
    },
    {
        "question": "Are the governing law provisions consistent across all indexed sponsorship agreements?",
        "rationale": "Requires comparing governing law across ArcGroup, CANOPETROLEUM, and VITAMINSHOPPE sponsorship agreements; not answerable from any single contract."
    },
    {
        "question": "Do the indexed agency agreements share a common definition of 'Affiliate'?",
        "rationale": "Requires comparing 'Affiliate' definitions across ALCOSTORES and BONTONSTORES agency agreements; not answerable from any single contract's annotations."
    },

    # Category D: Operational facts unrelated to contract terms
    {
        "question": "What was the stock price of Veoneer AB on the effective date of the JV amendment with Nissin Kogyo?",
        "rationale": "Stock price history is external financial data not contained in any contract."
    },
    {
        "question": "How many employees did BicycleTX Ltd have when the service agreement with Nigel Crockett was signed?",
        "rationale": "Employee headcount is operational HR data not contained in the service agreement."
    },
    {
        "question": "What was the quarterly revenue of Flotek Chemistry, LLC in Q1 2019 when the supply agreement was executed?",
        "rationale": "Financial performance data is not contained in the supply agreement."
    },
    {
        "question": "Did the Jacksonville Jaguars make the playoffs in the 2017 NFL season when the ARC sponsorship was executed?",
        "rationale": "Sports team performance is external factual data not contained in the sponsorship agreement."
    },
    {
        "question": "What was the market capitalization of Papa John's International on March 15, 2019, the effective date of the Shaq endorsement agreement?",
        "rationale": "Market cap is external financial data not contained in the endorsement agreement."
    },
    {
        "question": "How many retail locations did Bon-Ton Stores operate when the agency agreement was executed in April 2018?",
        "rationale": "Retail footprint is operational business data not contained in the agency agreement."
    },
    {
        "question": "What was the price of natural gas per MMBtu on April 14, 2015, the effective date of the PennTex gas transportation agreement?",
        "rationale": "Commodity pricing is external market data not contained in the transportation agreement."
    },
    {
        "question": "Did R-Tech Ueno, Ltd. receive FDA approval for any product during the term of the Sucampo strategic alliance?",
        "rationale": "Regulatory approval history is external factual data not contained in the strategic alliance agreement."
    },

    # Category E: Post-contract events/outcomes
    {
        "question": "Was the Joint Venture between MJ Syndicated and Simply Herbals successfully terminated or is it still active?",
        "rationale": "The MJ/Simply Herbals JV agreement (MJBIOTECH,INC_12_06_2018) is indexed but post-execution outcomes are not in the contract text."
    },
    {
        "question": "Did the non-competition agreement between Glamis Gold and Western Copper result in any litigation?",
        "rationale": "The Western Copper non-compete (WESTERN COPPER - NON-COMPETITION AGREEMENT) is indexed but litigation outcomes are not in the contract."
    },
    {
        "question": "Did the strategic alliance between Information System Associates and Rubicon Software Group achieve its revenue targets?",
        "rationale": "The ISA/Rubicon strategic alliance (DUOSTECHNOLOGIESGROUP,INC_04_21_2009) is indexed but performance outcomes are not in the contract."
    },
    {
        "question": "Was the exclusive reseller territory for Vayyar Imaging's Walabot-HOME expanded after July 2019?",
        "rationale": "The Walabot-HOME Reseller Agreement (HealthcareIntegratedTechnologiesInc) is indexed but post-execution modifications are not in the contract."
    },
    {
        "question": "Did the co-branding agreement between PCQuote and its partner lead to a joint product launch?",
        "rationale": "The PCQuote co-branding agreement (PcquoteComInc_19990721) is indexed but post-execution business outcomes are not in the contract."
    },
    {
        "question": "Did the endorsement agreement between Holiday RV Superstores and Affinity Group result in a measurable sales increase?",
        "rationale": "The Holiday RV/Affinity endorsement (HOLIDAYRVSUPERSTORESINC_04_15_2002) is indexed but business outcomes are not in the contract."
    },
    {
        "question": "Was the transportation services agreement between Purolator and Parcelway renewed after its initial term?",
        "rationale": "The Purolator/Parcelway agreement (DYNAMEXINC_06_06_1996) is indexed but renewal outcomes are not in the contract."
    },

    # Additional Category F: Specific contract clauses not annotated in the benchmark
    {
        "question": "What are the confidentiality obligations in the Cooperation Agreement between HPIL ENERGYTECH and GINARES GROUP AG?",
        "rationale": "The HPIL/Ginares cooperation agreement (HPILHOLDING_01_07_2015) is indexed but the annotated snippets do not cover confidentiality."
    },
    {
        "question": "What is the dispute resolution mechanism in the Franchise Agreement between the Town of Vinton and Roanoke Gas Company?",
        "rationale": "The Vinton/Roanoke Gas franchise (RgcResourcesInc_20151216) is indexed but annotated snippets do not cover dispute resolution."
    },
    {
        "question": "What insurance requirements apply to the Service Agreement between Theravance Biopharma UK and Brett Haumann?",
        "rationale": "The Theravance/Haumann service agreement (THERAVANCEBIOPHARMA,INC_05_08_2020) is indexed but annotated snippets do not cover insurance requirements."
    },
    {
        "question": "What are the audit rights in the Agency Agreement between General Electric Capital Corporation and Duckwall-Alco Stores?",
        "rationale": "The GE Capital/Alco Stores agency agreement (ALCOSTORESINC_12_14_2005) is indexed but annotated snippets do not cover audit rights."
    },
    {
        "question": "What is the assignment restriction in the Endorsement Agreement between ABG-Shaq and Papa John's entities?",
        "rationale": "The Papa John's/Shaq endorsement (PapaJohnsInternationalInc_20190617) is indexed but annotated snippets do not cover assignment restrictions."
    },
]


def main():
    print("Loading benchmark contracts...")
    benchmark_contracts = load_benchmark_contracts()
    print(f"Indexed contracts: {len(benchmark_contracts)}")

    # Build records with IDs, splits, and validation
    records = []
    seen_normalized = set()
    seen_ids = set()

    for i, candidate in enumerate(CANDIDATE_QUESTIONS):
        q = candidate["question"]
        norm = normalize_question(q)

        # Check uniqueness
        if norm in seen_normalized:
            print(f"WARNING: Duplicate normalized question at index {i}: {q[:80]}...")
            continue
        seen_normalized.add(norm)

        qid = f"unans_{i+1:03d}"
        if qid in seen_ids:
            print(f"WARNING: Duplicate ID {qid}")
            continue
        seen_ids.add(qid)

        split = assign_split(q)

        record = {
            "id": qid,
            "question": q,
            "label": "unanswerable",
            "split": split,
            "rationale": candidate["rationale"],
        }
        records.append(record)

    # Validation
    print(f"\nTotal candidate questions: {len(CANDIDATE_QUESTIONS)}")
    print(f"Valid records after dedup: {len(records)}")

    if len(records) < 30:
        print(f"ERROR: Need at least 30 records, got {len(records)}")
        return 1

    # Check unique IDs
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        print("ERROR: Duplicate IDs found")
        return 1

    # Check unique normalized questions
    norms = [normalize_question(r["question"]) for r in records]
    if len(set(norms)) != len(norms):
        print("ERROR: Duplicate normalized questions found")
        return 1

    # Check labels
    labels = set(r["label"] for r in records)
    if labels != {"unanswerable"}:
        print(f"ERROR: Unexpected labels: {labels}")
        return 1

    # Check splits
    splits = set(r["split"] for r in records)
    if not splits.issubset({"train", "test"}):
        print(f"ERROR: Unexpected splits: {splits}")
        return 1

    # Count splits
    train_count = sum(1 for r in records if r["split"] == "train")
    test_count = sum(1 for r in records if r["split"] == "test")
    print(f"\nSplit distribution: train={train_count}, test={test_count}, total={len(records)}")

    # Show all records for review
    print("\n=== ALL UNANSWERABLE QUESTIONS FOR REVIEW ===")
    for r in records:
        print(f"\n[{r['id']}] Split: {r['split']}")
        print(f"  Q: {r['question']}")
        print(f"  Rationale: {r['rationale']}")

    # Save to file
    output_path = Path("data/unanswerable_queries.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\nSaved to {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())