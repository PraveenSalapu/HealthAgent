"""
Document metadata configuration for clinical documents.

Maps document filenames to rich metadata including:
- Source organization
- Publication date
- Evidence level
- Document type
- Relevant topics
"""

from datetime import datetime

# ADA Standards of Care 2024 - Clinical Practice Guidelines
# These are the official American Diabetes Association guidelines
# Evidence Level: A (Strong evidence from well-conducted RCTs or meta-analyses)

ADA_2024_METADATA = {
    "dc24s001.pdf": {
        "source": "American Diabetes Association",
        "title": "Standards of Care in Diabetes—2024: Introduction",
        "publication_date": "2024-01-01",
        "evidence_level": "A",
        "document_type": "clinical_guideline",
        "doi": "10.2337/dc24-S001",
        "guideline_version": "2024",
        "section": "Introduction and Methodology",
        "topics": ["diabetes_overview", "evidence_grading", "methodology"],
        "target_audience": "healthcare_professional",
        "geographic_scope": "USA",
        "language": "en"
    },
    "dc24s002.pdf": {
        "source": "American Diabetes Association",
        "title": "Standards of Care in Diabetes—2024: Diagnosis and Classification",
        "publication_date": "2024-01-01",
        "evidence_level": "A",
        "document_type": "clinical_guideline",
        "doi": "10.2337/dc24-S002",
        "guideline_version": "2024",
        "section": "Classification and Diagnosis",
        "topics": ["diagnosis", "classification", "type1_diabetes", "type2_diabetes", "prediabetes", "screening"],
        "target_audience": "healthcare_professional",
        "geographic_scope": "USA",
        "language": "en"
    },
    "dc24s003.pdf": {
        "source": "American Diabetes Association",
        "title": "Standards of Care in Diabetes—2024: Prevention and Delay of Type 2 Diabetes",
        "publication_date": "2024-01-01",
        "evidence_level": "A",
        "document_type": "clinical_guideline",
        "doi": "10.2337/dc24-S003",
        "guideline_version": "2024",
        "section": "Prevention",
        "topics": ["prevention", "prediabetes", "lifestyle_intervention", "risk_reduction", "screening"],
        "target_audience": "healthcare_professional",
        "geographic_scope": "USA",
        "language": "en"
    },
    "dc24s004.pdf": {
        "source": "American Diabetes Association",
        "title": "Standards of Care in Diabetes—2024: Comprehensive Medical Evaluation",
        "publication_date": "2024-01-01",
        "evidence_level": "A",
        "document_type": "clinical_guideline",
        "doi": "10.2337/dc24-S004",
        "guideline_version": "2024",
        "section": "Comprehensive Medical Evaluation",
        "topics": ["medical_evaluation", "initial_assessment", "continuing_care", "screening"],
        "target_audience": "healthcare_professional",
        "geographic_scope": "USA",
        "language": "en"
    },
    "dc24s005.pdf": {
        "source": "American Diabetes Association",
        "title": "Standards of Care in Diabetes—2024: Facilitating Positive Health Behaviors",
        "publication_date": "2024-01-01",
        "evidence_level": "A",
        "document_type": "clinical_guideline",
        "doi": "10.2337/dc24-S005",
        "guideline_version": "2024",
        "section": "Lifestyle and Behavior Change",
        "topics": ["behavior_change", "lifestyle_modification", "self_management", "education"],
        "target_audience": "healthcare_professional",
        "geographic_scope": "USA",
        "language": "en"
    },
    "dc24s006.pdf": {
        "source": "American Diabetes Association",
        "title": "Standards of Care in Diabetes—2024: Glycemic Goals and Hypoglycemia",
        "publication_date": "2024-01-01",
        "evidence_level": "A",
        "document_type": "clinical_guideline",
        "doi": "10.2337/dc24-S006",
        "guideline_version": "2024",
        "section": "Glycemic Targets",
        "topics": ["glycemic_control", "HbA1c", "hypoglycemia", "treatment_goals"],
        "target_audience": "healthcare_professional",
        "geographic_scope": "USA",
        "language": "en"
    },
    "dc24s007.pdf": {
        "source": "American Diabetes Association",
        "title": "Standards of Care in Diabetes—2024: Diabetes Technology",
        "publication_date": "2024-01-01",
        "evidence_level": "A",
        "document_type": "clinical_guideline",
        "doi": "10.2337/dc24-S007",
        "guideline_version": "2024",
        "section": "Diabetes Technology",
        "topics": ["diabetes_technology", "CGM", "insulin_pumps", "monitoring"],
        "target_audience": "healthcare_professional",
        "geographic_scope": "USA",
        "language": "en"
    },
    "dc24s008.pdf": {
        "source": "American Diabetes Association",
        "title": "Standards of Care in Diabetes—2024: Obesity and Weight Management",
        "publication_date": "2024-01-01",
        "evidence_level": "A",
        "document_type": "clinical_guideline",
        "doi": "10.2337/dc24-S008",
        "guideline_version": "2024",
        "section": "Obesity Management",
        "topics": ["obesity", "weight_management", "BMI", "metabolic_surgery", "weight_loss"],
        "target_audience": "healthcare_professional",
        "geographic_scope": "USA",
        "language": "en"
    },
    "dc24s009.pdf": {
        "source": "American Diabetes Association",
        "title": "Standards of Care in Diabetes—2024: Pharmacologic Approaches to Glycemic Treatment",
        "publication_date": "2024-01-01",
        "evidence_level": "A",
        "document_type": "clinical_guideline",
        "doi": "10.2337/dc24-S009",
        "guideline_version": "2024",
        "section": "Pharmacologic Treatment",
        "topics": ["medication", "pharmacology", "metformin", "insulin", "treatment"],
        "target_audience": "healthcare_professional",
        "geographic_scope": "USA",
        "language": "en"
    },
    "dc24s010.pdf": {
        "source": "American Diabetes Association",
        "title": "Standards of Care in Diabetes—2024: Cardiovascular Disease and Risk Management",
        "publication_date": "2024-01-01",
        "evidence_level": "A",
        "document_type": "clinical_guideline",
        "doi": "10.2337/dc24-S010",
        "guideline_version": "2024",
        "section": "Cardiovascular Disease",
        "topics": ["cardiovascular_disease", "heart_disease", "risk_management", "complications"],
        "target_audience": "healthcare_professional",
        "geographic_scope": "USA",
        "language": "en"
    },
    "dc24s011.pdf": {
        "source": "American Diabetes Association",
        "title": "Standards of Care in Diabetes—2024: Chronic Kidney Disease and Risk Management",
        "publication_date": "2024-01-01",
        "evidence_level": "A",
        "document_type": "clinical_guideline",
        "doi": "10.2337/dc24-S011",
        "guideline_version": "2024",
        "section": "Kidney Disease",
        "topics": ["kidney_disease", "nephropathy", "complications", "screening"],
        "target_audience": "healthcare_professional",
        "geographic_scope": "USA",
        "language": "en"
    },
    "dc24s012.pdf": {
        "source": "American Diabetes Association",
        "title": "Standards of Care in Diabetes—2024: Retinopathy, Neuropathy, and Foot Care",
        "publication_date": "2024-01-01",
        "evidence_level": "A",
        "document_type": "clinical_guideline",
        "doi": "10.2337/dc24-S012",
        "guideline_version": "2024",
        "section": "Microvascular Complications",
        "topics": ["retinopathy", "neuropathy", "foot_care", "complications"],
        "target_audience": "healthcare_professional",
        "geographic_scope": "USA",
        "language": "en"
    },
    "dc24s013.pdf": {
        "source": "American Diabetes Association",
        "title": "Standards of Care in Diabetes—2024: Older Adults",
        "publication_date": "2024-01-01",
        "evidence_level": "A",
        "document_type": "clinical_guideline",
        "doi": "10.2337/dc24-S013",
        "guideline_version": "2024",
        "section": "Older Adults",
        "topics": ["older_adults", "elderly", "geriatric", "age_considerations"],
        "target_audience": "healthcare_professional",
        "geographic_scope": "USA",
        "language": "en"
    },
    "dc24s014.pdf": {
        "source": "American Diabetes Association",
        "title": "Standards of Care in Diabetes—2024: Children and Adolescents",
        "publication_date": "2024-01-01",
        "evidence_level": "A",
        "document_type": "clinical_guideline",
        "doi": "10.2337/dc24-S014",
        "guideline_version": "2024",
        "section": "Pediatrics",
        "topics": ["children", "adolescents", "pediatric", "youth"],
        "target_audience": "healthcare_professional",
        "geographic_scope": "USA",
        "language": "en"
    },
    "dc24s015.pdf": {
        "source": "American Diabetes Association",
        "title": "Standards of Care in Diabetes—2024: Diabetes and Pregnancy",
        "publication_date": "2024-01-01",
        "evidence_level": "A",
        "document_type": "clinical_guideline",
        "doi": "10.2337/dc24-S015",
        "guideline_version": "2024",
        "section": "Pregnancy",
        "topics": ["pregnancy", "gestational_diabetes", "preconception", "maternal_health"],
        "target_audience": "healthcare_professional",
        "geographic_scope": "USA",
        "language": "en"
    },
    "dc24s016.pdf": {
        "source": "American Diabetes Association",
        "title": "Standards of Care in Diabetes—2024: Hospital and Surgical Care",
        "publication_date": "2024-01-01",
        "evidence_level": "A",
        "document_type": "clinical_guideline",
        "doi": "10.2337/dc24-S016",
        "guideline_version": "2024",
        "section": "Inpatient Care",
        "topics": ["hospital_care", "surgery", "inpatient", "perioperative"],
        "target_audience": "healthcare_professional",
        "geographic_scope": "USA",
        "language": "en"
    },
    "dc24s017.pdf": {
        "source": "American Diabetes Association",
        "title": "Standards of Care in Diabetes—2024: Diabetes Advocacy",
        "publication_date": "2024-01-01",
        "evidence_level": "A",
        "document_type": "clinical_guideline",
        "doi": "10.2337/dc24-S017",
        "guideline_version": "2024",
        "section": "Advocacy",
        "topics": ["advocacy", "policy", "access_to_care"],
        "target_audience": "healthcare_professional",
        "geographic_scope": "USA",
        "language": "en"
    },
    "dc24sin01.pdf": {
        "source": "American Diabetes Association",
        "title": "Standards of Care in Diabetes—2024: Introduction",
        "publication_date": "2024-01-01",
        "evidence_level": "A",
        "document_type": "clinical_guideline",
        "doi": "10.2337/dc24-SIN01",
        "guideline_version": "2024",
        "section": "Introduction",
        "topics": ["introduction", "overview"],
        "target_audience": "healthcare_professional",
        "geographic_scope": "USA",
        "language": "en"
    },
    "dc24sint.pdf": {
        "source": "American Diabetes Association",
        "title": "Standards of Care in Diabetes—2024: Summary of Changes",
        "publication_date": "2024-01-01",
        "evidence_level": "A",
        "document_type": "clinical_guideline",
        "doi": "10.2337/dc24-SINT",
        "guideline_version": "2024",
        "section": "Summary of Changes",
        "topics": ["updates", "changes", "revisions"],
        "target_audience": "healthcare_professional",
        "geographic_scope": "USA",
        "language": "en"
    },
    "dc24srev.pdf": {
        "source": "American Diabetes Association",
        "title": "Standards of Care in Diabetes—2024: Summary of Revisions",
        "publication_date": "2024-01-01",
        "evidence_level": "A",
        "document_type": "clinical_guideline",
        "doi": "10.2337/dc24-SREV",
        "guideline_version": "2024",
        "section": "Revisions Summary",
        "topics": ["updates", "revisions", "changes"],
        "target_audience": "healthcare_professional",
        "geographic_scope": "USA",
        "language": "en"
    },
    "dc24sdis.pdf": {
        "source": "American Diabetes Association",
        "title": "Standards of Care in Diabetes—2024: Disclosure",
        "publication_date": "2024-01-01",
        "evidence_level": "A",
        "document_type": "clinical_guideline",
        "doi": "10.2337/dc24-SDIS",
        "guideline_version": "2024",
        "section": "Disclosure",
        "topics": ["disclosure", "conflicts_of_interest"],
        "target_audience": "healthcare_professional",
        "geographic_scope": "USA",
        "language": "en"
    }
}

# BRFSS Codebook (data source for diabetes surveillance)
OTHER_METADATA = {
    "codebook15_llcp.pdf": {
        "source": "CDC Behavioral Risk Factor Surveillance System (BRFSS)",
        "title": "BRFSS 2015 Codebook",
        "publication_date": "2015-01-01",
        "evidence_level": "B",  # Surveillance data, not clinical trial
        "document_type": "data_codebook",
        "doi": None,
        "guideline_version": "2015",
        "section": "Data Dictionary",
        "topics": ["epidemiology", "surveillance", "risk_factors", "data_source"],
        "target_audience": "researcher",
        "geographic_scope": "USA",
        "language": "en"
    }
}

# Combine all metadata
DOCUMENT_METADATA = {**ADA_2024_METADATA, **OTHER_METADATA}


def get_metadata_for_file(filename: str) -> dict:
    """
    Get metadata for a given filename.

    Args:
        filename: Name of the file (e.g., "dc24s003.pdf")

    Returns:
        dict: Metadata dictionary or default metadata if not found
    """
    # Handle duplicates (e.g., "dc24s013 - Copy.pdf" -> "dc24s013.pdf")
    clean_filename = filename.replace(" - Copy", "")

    if clean_filename in DOCUMENT_METADATA:
        return DOCUMENT_METADATA[clean_filename].copy()
    else:
        # Return default metadata for unknown files
        return {
            "source": "Unknown",
            "title": filename,
            "publication_date": "2024-01-01",
            "evidence_level": "C",  # Unknown = lowest evidence level
            "document_type": "unknown",
            "doi": None,
            "guideline_version": "unknown",
            "section": "unknown",
            "topics": [],
            "target_audience": "general",
            "geographic_scope": "unknown",
            "language": "en"
        }


def get_evidence_rank(level: str) -> int:
    """
    Convert evidence level to numeric rank for filtering.

    A = 3 (Highest - RCTs, meta-analyses)
    B = 2 (Moderate - cohort studies, case-control)
    C = 1 (Lowest - expert opinion, case reports)

    Args:
        level: Evidence level (A, B, or C)

    Returns:
        int: Numeric rank
    """
    ranks = {'A': 3, 'B': 2, 'C': 1}
    return ranks.get(level.upper(), 0)


def is_recent_guideline(pub_date: str, max_age_years: int = 5) -> bool:
    """
    Check if a guideline is recent (within max_age_years).

    Args:
        pub_date: Publication date in ISO format (YYYY-MM-DD)
        max_age_years: Maximum age in years to consider "recent"

    Returns:
        bool: True if recent, False otherwise
    """
    from datetime import datetime, timedelta

    try:
        if isinstance(pub_date, str):
            pub_datetime = datetime.fromisoformat(pub_date)
        else:
            pub_datetime = pub_date

        age = datetime.now() - pub_datetime
        return age <= timedelta(days=365 * max_age_years)
    except Exception:
        return False
