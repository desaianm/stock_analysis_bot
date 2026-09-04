"""Public 13F portfolio context used as a supporting research signal.

The SEC filing is delayed and may include option positions. It is therefore
never treated as a standalone buy signal; the funnel only gives a small boost
to direct long positions that also pass the scarcity-lane fundamentals.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import requests


SITUATIONAL_AWARENESS_CIK = "0002045724"
SEC_SUBMISSIONS_URL = (
    f"https://data.sec.gov/submissions/CIK{SITUATIONAL_AWARENESS_CIK}.json"
)
SEC_ARCHIVES_ROOT = "https://www.sec.gov/Archives/edgar/data"
SEC_HEADERS = {
    "User-Agent": os.getenv(
        "SEC_USER_AGENT", "stock-analysis-bot/1.0 stock-analysis@example.com"
    )
}


@dataclass(frozen=True)
class InstitutionalHolding:
    issuer: str
    cusip: str
    value_usd: float
    shares: float
    option_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InstitutionalPortfolio:
    manager: str = "Situational Awareness LP"
    cik: str = SITUATIONAL_AWARENESS_CIK
    report_date: Optional[str] = None
    filing_date: Optional[str] = None
    accession_number: Optional[str] = None
    holdings: List[InstitutionalHolding] = field(default_factory=list)
    error: Optional[str] = None

    def direct_long_value(self, company_name: Optional[str]) -> float:
        """Return disclosed direct-long value matched by normalized issuer name."""
        if not company_name:
            return 0.0
        company_key = normalize_issuer_name(company_name)
        return sum(
            holding.value_usd
            for holding in self.holdings
            if holding.option_type is None
            and normalize_issuer_name(holding.issuer) == company_key
        )

    def positions_for(self, company_name: Optional[str]) -> List[Dict[str, Any]]:
        if not company_name:
            return []
        company_key = normalize_issuer_name(company_name)
        return [
            holding.to_dict()
            for holding in self.holdings
            if normalize_issuer_name(holding.issuer) == company_key
        ]


_ISSUER_SUFFIXES = {
    "class",
    "co",
    "company",
    "corp",
    "corporation",
    "group",
    "holdings",
    "inc",
    "limited",
    "ltd",
    "new",
    "nv",
    "plc",
    "sa",
    "the",
}


def normalize_issuer_name(value: str) -> str:
    """Normalize Yahoo and 13F issuer names for conservative exact matching."""
    words = re.findall(r"[a-z0-9]+", value.lower())
    return " ".join(word for word in words if word not in _ISSUER_SUFFIXES)


def _latest_13f_metadata(payload: Dict[str, Any]) -> Dict[str, str]:
    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form") or []
    for index, form in enumerate(forms):
        if form == "13F-HR":
            return {
                "accession_number": recent["accessionNumber"][index],
                "filing_date": recent["filingDate"][index],
                "report_date": recent["reportDate"][index],
            }
    raise ValueError("SEC submissions contain no 13F-HR filing")


def _parse_information_table(xml_text: str) -> List[InstitutionalHolding]:
    root = ET.fromstring(xml_text)
    holdings: List[InstitutionalHolding] = []
    for row in root.findall(".//{*}infoTable"):
        issuer = row.findtext("{*}nameOfIssuer")
        cusip = row.findtext("{*}cusip")
        value = row.findtext("{*}value")
        shares = row.findtext(".//{*}sshPrnamt")
        option_type = row.findtext("{*}putCall")
        if not issuer or not cusip or value is None or shares is None:
            continue
        holdings.append(
            InstitutionalHolding(
                issuer=issuer.strip(),
                cusip=cusip.strip(),
                value_usd=float(value),
                shares=float(shares),
                option_type=option_type.strip().lower() if option_type else None,
            )
        )
    if not holdings:
        raise ValueError("SEC 13F information table contains no holdings")
    return holdings


def fetch_situational_awareness_portfolio(
    *, session: Any = requests, timeout: float = 20.0
) -> InstitutionalPortfolio:
    """Fetch the manager's latest public 13F directly from SEC EDGAR."""
    portfolio = InstitutionalPortfolio()
    try:
        submissions_response = session.get(
            SEC_SUBMISSIONS_URL, headers=SEC_HEADERS, timeout=timeout
        )
        submissions_response.raise_for_status()
        metadata = _latest_13f_metadata(submissions_response.json())

        accession_path = metadata["accession_number"].replace("-", "")
        filing_root = (
            f"{SEC_ARCHIVES_ROOT}/{int(SITUATIONAL_AWARENESS_CIK)}/"
            f"{accession_path}"
        )
        index_response = session.get(
            f"{filing_root}/index.json", headers=SEC_HEADERS, timeout=timeout
        )
        index_response.raise_for_status()
        filenames = [
            item.get("name", "")
            for item in index_response.json().get("directory", {}).get("item", [])
        ]
        information_tables = [
            name
            for name in filenames
            if name.lower().endswith(".xml")
            and "primary_doc" not in name.lower()
        ]
        if not information_tables:
            raise ValueError("SEC filing contains no 13F information table")

        table_response = session.get(
            f"{filing_root}/{information_tables[0]}",
            headers=SEC_HEADERS,
            timeout=timeout,
        )
        table_response.raise_for_status()
        portfolio.report_date = metadata["report_date"]
        portfolio.filing_date = metadata["filing_date"]
        portfolio.accession_number = metadata["accession_number"]
        portfolio.holdings = _parse_information_table(table_response.text)
    except Exception as exc:  # noqa: BLE001 - optional overlay must fail open
        portfolio.error = str(exc)
    return portfolio
