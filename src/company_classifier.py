"""
Company Classifier for the Redrob AI Candidate Ranking System.

Classifies companies as consulting/services, product, or unknown.
The JD explicitly penalizes consulting-only career histories:
  "People who have only worked at consulting firms (TCS, Infosys,
   Wipro, Accenture, Cognizant, Capgemini, etc.) in their entire career."
"""

from __future__ import annotations
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Known company classifications
# ---------------------------------------------------------------------------

# IT Services / Consulting firms explicitly named in the JD
CONSULTING_COMPANIES = {
    # Explicitly named in JD
    "tcs", "tata consultancy services",
    "infosys",
    "wipro",
    "accenture",
    "cognizant", "cognizant technology solutions",
    "capgemini",
    # Other major IT services / consulting
    "hcl", "hcl technologies",
    "tech mahindra",
    "ltimindtree", "mindtree", "lti",
    "mphasis",
    "hexaware",
    "niit technologies", "coforge",
    "persistent systems",
    "zensar",
    "l&t infotech", "larsen & toubro infotech",
    "cyient",
    "birlasoft",
    "sonata software",
    "mastek",
    # Global consulting
    "deloitte",
    "pwc", "pricewaterhousecoopers",
    "ey", "ernst & young", "ernst and young",
    "kpmg",
    "mckinsey", "mckinsey & company",
    "bcg", "boston consulting group",
    "bain", "bain & company",
    # Other services
    "ibm global services",
    "dxc technology",
    "atos",
    "ntt data",
    "fujitsu",
    "cgi", "cgi group",
}

# Known product companies (positive signal for the JD)
PRODUCT_COMPANIES = {
    # FAANG / Big Tech
    "google", "alphabet",
    "meta", "facebook",
    "amazon", "aws",
    "apple",
    "microsoft",
    "netflix",
    # Tech giants
    "uber", "lyft",
    "airbnb",
    "spotify",
    "twitter", "x",
    "linkedin",
    "salesforce",
    "adobe",
    "oracle",
    "snap", "snapchat",
    "pinterest",
    "stripe",
    "dropbox",
    "slack",
    "zoom",
    "shopify",
    "atlassian",
    "databricks",
    "snowflake",
    "palantir",
    "datadog",
    "cloudflare",
    "twilio",
    "square", "block",
    "intuit",
    "servicenow",
    # AI-specific
    "openai",
    "anthropic",
    "cohere",
    "hugging face", "huggingface",
    "stability ai",
    "midjourney",
    "deepmind",
    "nvidia",
    # Indian product companies
    "flipkart",
    "swiggy",
    "zomato",
    "razorpay",
    "phonepe",
    "cred",
    "meesho",
    "ola",
    "paytm",
    "dream11",
    "freshworks",
    "zoho",
    "browserstack",
    "postman",
    "chargebee",
    "clevertap",
    "unacademy",
    "byju's", "byjus",
    "sharechat",
    "dunzo",
    "urban company", "urbanclap",
    "lenskart",
    "nykaa",
    "policybazaar",
    "groww",
    "zerodha",
    "upstox",
    "bigbasket",
    "myntra",
    "jio", "reliance jio",
    "makemytrip",
    "oyo",
    "cure.fit", "cultfit",
    "vedantu",
    "practo",
    "1mg",
    # Global startups / scale-ups
    "stripe",
    "figma",
    "notion",
    "vercel",
    "supabase",
    "railway",
}

# Industries that strongly suggest non-product (from dataset)
NON_TECH_INDUSTRIES = {
    "paper products",
    "manufacturing",
    "construction",
    "agriculture",
    "mining",
    "oil & gas", "oil and gas",
    "textiles",
    "food & beverage", "food and beverage",
    "hospitality",
    "real estate",
    "transportation",
    "utilities",
    "government",
}

# Industries that suggest product / tech
TECH_INDUSTRIES = {
    "software", "technology", "information technology",
    "internet", "saas", "ai", "artificial intelligence",
    "machine learning", "data analytics",
    "e-commerce", "ecommerce",
    "fintech", "edtech", "healthtech",
    "cloud computing", "cybersecurity",
}


def _normalize(name: str) -> str:
    """Normalize company/industry name for matching."""
    return name.lower().strip()


def classify_company(
    company_name: str,
    industry: str = "",
    company_size: str = "",
) -> str:
    """
    Classify a company as 'consulting', 'product', or 'unknown'.

    Args:
        company_name: The company name string
        industry: The industry string (from career_history)
        company_size: Company size band

    Returns:
        One of: 'consulting', 'product', 'unknown'
    """
    norm_company = _normalize(company_name)
    norm_industry = _normalize(industry)

    # Check direct company match
    for consulting in CONSULTING_COMPANIES:
        if consulting in norm_company or norm_company in consulting:
            return "consulting"

    for product in PRODUCT_COMPANIES:
        if product in norm_company or norm_company in product:
            return "product"

    # Check industry signals
    if norm_industry in NON_TECH_INDUSTRIES:
        return "unknown"  # Not consulting, but not product tech either

    if norm_industry in TECH_INDUSTRIES:
        return "product"

    if norm_industry == "it services":
        # IT Services is the consulting bucket
        return "consulting"

    return "unknown"


def analyze_career_companies(career_history: List[dict]) -> dict:
    """
    Analyze a candidate's full career history for company types.

    Returns:
        Dict with:
        - company_types: list of (company, classification) tuples
        - consulting_ratio: fraction of career spent at consulting firms
        - has_product_experience: bool
        - consulting_only: bool (JD disqualifier)
        - consulting_months: total months at consulting
        - product_months: total months at product companies
    """
    company_types: List[Tuple[str, str]] = []
    consulting_months = 0
    product_months = 0
    total_months = 0

    for role in career_history:
        company = role.get("company", "")
        industry = role.get("industry", "")
        company_size = role.get("company_size", "")
        duration = role.get("duration_months", 0)

        classification = classify_company(company, industry, company_size)
        company_types.append((company, classification))

        total_months += duration
        if classification == "consulting":
            consulting_months += duration
        elif classification == "product":
            product_months += duration

    consulting_ratio = consulting_months / max(total_months, 1)
    has_product = product_months > 0
    consulting_only = (
        consulting_ratio > 0.9 and
        not has_product and
        len(career_history) >= 2
    )

    return {
        "company_types": company_types,
        "consulting_ratio": round(consulting_ratio, 3),
        "has_product_experience": has_product,
        "consulting_only": consulting_only,
        "consulting_months": consulting_months,
        "product_months": product_months,
        "total_months": total_months,
    }
