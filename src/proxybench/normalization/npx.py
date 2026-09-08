"""Conservative N-PX mappings. Mechanical success never grants training admission."""

from decimal import Decimal
import re

from proxybench.normalization.values import scalar, ACTIVE_GUIDE
from proxybench.sources.npx_xml import FORM_NS, VOTE_NS, XMLRejected, parse_xml, pointer

RULE_VERSION = "npx-mapping-v2"
ROW_FIELDS = {"issuerName", "cusip", "isin", "figi", "meetingDate", "voteDescription",
              "voteCategories", "otherVoteDescription", "voteSource", "sharesVoted",
              "sharesOnLoan", "vote", "voteManager", "voteSeries", "voteOtherInfo"}


def leaf(node):
    if node.children or not node.text.strip():
        raise XMLRejected(f"Empty or structured scalar {node.name} at byte {node.start}")
    return node.text.strip()


def amount(node):
    value = leaf(node)
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value) or len(value.replace('.', '')) > 24:
        raise XMLRejected(f"Unsupported shares amount at byte {node.start}")
    return value


def map_filing(primary, votes, *, primary_id, votes_id, selected=None):
    """Return selected row outcomes and counts for the complete bounded attachment."""
    root = parse_xml(primary, namespace=FORM_NS, root_name="edgarSubmission")
    table = parse_xml(votes, namespace=VOTE_NS, root_name="proxyVoteTable")
    if any(n.name != "proxyTable" for n in table.children) or table.text.strip():
        raise XMLRejected("Unsupported vote-table structure")
    rows = table.children
    if selected is None:
        selected = list(range(min(24, len(rows))))
    if len(selected) > 24 or len(set(selected)) != len(selected) or any(
            type(i) is not int or not 0 <= i < len(rows) for i in selected):
        raise XMLRejected("Invalid selected row indices or selection cap exceeded")
    form_type = leaf(root.at("headerData/submissionType"))
    registrant = leaf(root.at("headerData/filerInfo/registrantType"))
    report_type = leaf(root.at("formData/coverPage/reportInfo/reportType"))
    if form_type not in {"N-PX", "N-PX/A"} or registrant != "RMIC" or report_type != "FUND VOTING REPORT":
        raise XMLRejected(f"Unsupported report class: {form_type}/{registrant}/{report_type}")
    series = {}
    page = root.at("formData").one("seriesPage", optional=True)
    if page:
        for entry in page.at("seriesDetails").all("seriesReports"):
            key = leaf(entry.one("idOfSeries"))
            if key in series:
                raise XMLRejected("Duplicate or conflicting series join")
            series[key] = entry
    outcomes = []
    for index in selected:
        row = rows[index]
        location = pointer(row, votes, votes_id)
        try:
            if row.text.strip() or any(n.name not in ROW_FIELDS for n in row.children):
                raise XMLRejected("Unknown row structure")
            for name in ROW_FIELDS:
                row.one(name, optional=True)  # Reject duplicate singleton elements.
            cells = []

            def add(key, label, node, *, source=votes, document_id=votes_id, text=None):
                cells.append({"key": key, "label": label, "text": leaf(node) if text is None else text,
                              "source": pointer(node, source, document_id)})

            series_node = row.one("voteSeries")
            series_id = leaf(series_node)
            if series_id not in series or not re.fullmatch(r"[Ss][0-9]{9}", series_id):
                raise XMLRejected("Unresolved scope join; no fund expansion is permitted")
            scope = series[series_id].one("nameOfSeries")
            add("scope", "Reporting fund", scope, source=primary, document_id=primary_id)
            add("series", "Series ID", series_node)
            add("issuer", "Issuer", row.one("issuerName"))
            for key, label in (("cusip", "CUSIP"), ("isin", "ISIN"), ("figi", "FIGI")):
                node = row.one(key, optional=True)
                if node:
                    if leaf(node) == "N/A":
                        raise XMLRejected("Identifier N/A needs a separate reviewed absence rule")
                    add(key, label, node)
            date = row.one("meetingDate")
            scalar(leaf(date), "date", guide_version=ACTIVE_GUIDE)
            add("date", "Meeting date", date)
            add("description", "Proposal", row.one("voteDescription"))
            categories = row.one("voteCategories")
            if any(n.name != "voteCategory" for n in categories.children) or not categories.children:
                raise XMLRejected("Unsupported categories structure")
            category_values = [leaf(n.one("categoryType")) for n in categories.children]
            if any(len(n.children) != 1 or n.text.strip() for n in categories.children):
                raise XMLRejected("Unsupported category structure")
            if "DIRECTOR ELECTIONS" in category_values:
                raise XMLRejected("Director subject boundaries need a reviewed mapping rule")
            # Free text can modify attribution or vote interpretation. Do not silently omit it.
            if row.one("voteOtherInfo", optional=True) or row.one("otherVoteDescription", optional=True):
                raise XMLRejected("Supplemental voting text needs source review")
            proponent = row.one("voteSource", optional=True)
            if proponent:
                if leaf(proponent) not in {"ISSUER", "SECURITY HOLDER"}:
                    raise XMLRejected("Unsupported proposal source")
                add("proponent", "Proposed by", proponent)
            total = amount(row.one("sharesVoted"))
            loan = amount(row.one("sharesOnLoan"))
            managers = row.one("voteManager", optional=True)
            if managers:
                if not 1 <= len(managers.children) <= 25 or any(n.name != "otherManagers" for n in managers.children):
                    raise XMLRejected("Unsupported manager relationship structure")
                for entry in managers.children:
                    if len(entry.children) != 1 or not re.fullmatch(r"[0-9]{1,3}", leaf(entry.one("otherManager"))):
                        raise XMLRejected("Unsupported manager relationship")
            vote = row.one("vote", optional=True)
            if not vote or not vote.children or any(n.name != "voteRecord" for n in vote.children):
                raise XMLRejected("Zero or missing voting components do not establish nonvoting")
            if len(vote.children) > 999:
                raise XMLRejected("Vote component limit exceeded")
            component_amounts = []
            for j, component in enumerate(vote.children):
                if {n.name for n in component.children} != {"howVoted", "sharesVoted", "managementRecommendation"}:
                    raise XMLRejected("Unsupported vote component structure")
                direction = component.one("howVoted")
                quantity = component.one("sharesVoted")
                alignment = component.one("managementRecommendation")
                value = amount(quantity)
                if Decimal(value) == 0:
                    raise XMLRejected("Zero component is not a cast vote; quarantine the complete row")
                if leaf(alignment) not in {"FOR", "AGAINST", "NONE"}:
                    raise XMLRejected("Unsupported management alignment")
                leaf(direction)
                component_amounts.append(Decimal(value))
                add(f"direction:{j}", "Vote", direction)
                add(f"quantity:{j}", "Shares voted", quantity)
                # The official stylesheet calls this FOR OR AGAINST MANAGEMENT.
                add(f"alignment:{j}", "For or against management", alignment)
            if sum(component_amounts) != Decimal(total):
                raise XMLRejected("Component quantities conflict with disclosed total")
            add("loan", "Shares on loan", row.one("sharesOnLoan"))
            outcomes.append({"index": index, "status": "PROVISIONAL_GENERATED", "rule_version": RULE_VERSION,
                             "mechanical_mapping": "PASS", "human_review": "PENDING",
                             "training_admitted": False, "source": location, "cells": cells,
                             "components": len(vote.children), "scope_type": "INDIVIDUAL_FUND",
                             "source_audit": {"shares_voted_total": total, "shares_on_loan": loan,
                                              "categories": category_values,
                                              "management_recommendation": "NOT_MAPPED_FROM_ALIGNMENT"}})
        except (XMLRejected, ValueError) as error:
            outcomes.append({"index": index, "status": "REJECTED", "reason": str(error),
                             "source": location, "training_admitted": False, "rule_version": RULE_VERSION})
    passed = sum(r["status"] == "PROVISIONAL_GENERATED" for r in outcomes)
    return {"rule_version": RULE_VERSION, "submission_type": form_type, "registrant_type": registrant,
            "discovered": len(rows), "parsed": len(rows), "selected": len(selected),
            "unselected": len(rows) - len(selected), "provisional": passed,
            "rejected": len(selected) - passed, "admitted": 0, "records": outcomes}
