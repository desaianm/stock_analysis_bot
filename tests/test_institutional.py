from stockbot.tools.institutional import (
    InstitutionalHolding,
    InstitutionalPortfolio,
    fetch_situational_awareness_portfolio,
    normalize_issuer_name,
)


class _Response:
    def __init__(self, *, json_data=None, text=""):
        self._json_data = json_data
        self.text = text

    def raise_for_status(self):
        return None

    def json(self):
        return self._json_data


class _Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.urls = []

    def get(self, url, **_kwargs):
        self.urls.append(url)
        return next(self.responses)


def test_fetches_latest_13f_and_preserves_option_type():
    submissions = {
        "filings": {
            "recent": {
                "form": ["13G", "13F-HR"],
                "accessionNumber": ["ignored", "0000000000-26-000001"],
                "filingDate": ["2026-08-15", "2026-08-14"],
                "reportDate": ["", "2026-06-30"],
            }
        }
    }
    index = {
        "directory": {
            "item": [
                {"name": "primary_doc.xml"},
                {"name": "form13fInfoTable.xml"},
            ]
        }
    }
    table = """<informationTable xmlns="urn:test">
      <infoTable><nameOfIssuer>SANDISK CORP</nameOfIssuer><cusip>80004C200</cusip>
      <value>5700000000</value><shrsOrPrnAmt><sshPrnamt>2500000</sshPrnamt>
      </shrsOrPrnAmt></infoTable>
      <infoTable><nameOfIssuer>NVIDIA CORPORATION</nameOfIssuer><cusip>67066G104</cusip>
      <value>1000000</value><shrsOrPrnAmt><sshPrnamt>5000</sshPrnamt></shrsOrPrnAmt>
      <putCall>Put</putCall></infoTable></informationTable>"""
    session = _Session(
        [_Response(json_data=submissions), _Response(json_data=index), _Response(text=table)]
    )

    portfolio = fetch_situational_awareness_portfolio(session=session)

    assert portfolio.error is None
    assert portfolio.report_date == "2026-06-30"
    assert portfolio.direct_long_value("SanDisk Corporation") == 5_700_000_000
    assert portfolio.direct_long_value("NVIDIA Corporation") == 0
    assert portfolio.positions_for("NVIDIA Corp")[0]["option_type"] == "put"
    assert session.urls[-1].endswith("/form13fInfoTable.xml")


def test_sec_failure_is_nonfatal():
    class FailingSession:
        def get(self, *_args, **_kwargs):
            raise TimeoutError("SEC unavailable")

    portfolio = fetch_situational_awareness_portfolio(session=FailingSession())

    assert portfolio.holdings == []
    assert portfolio.error == "SEC unavailable"


def test_issuer_matching_is_conservative_but_ignores_legal_suffixes():
    assert normalize_issuer_name("Taiwan Semiconductor Manufacturing Co. Ltd.") == (
        normalize_issuer_name("TAIWAN SEMICONDUCTOR MANUFACTURING COMPANY LIMITED")
    )
    portfolio = InstitutionalPortfolio(
        holdings=[
            InstitutionalHolding("SANDISK CORP", "x", 10, 1),
            InstitutionalHolding("SANDISK CORP", "x", 20, 2, "call"),
        ]
    )

    assert portfolio.direct_long_value("SanDisk Corporation") == 10
    assert len(portfolio.positions_for("SanDisk Corporation")) == 2
