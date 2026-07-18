"""
CCLD Facility Report Scraper + NLP Pipeline  —  high-level sketch
================================================================
For each facility: probe the API visit-by-visit, parse each report
into structured fields, score its severity with zero-shot NLP, and
stream rows to CSV. Full code has been stripped down.
"""

BASE_URL        = "https://www.ccld.dss.ca.gov/transparencyapi/api/FacilityReports"
REQUEST_DELAY   = 0.5     # politeness to server
CHECKPOINT_EVERY = 200    # flush rows to disk this often
SEVERITY_LABELS = ["serious safety risk", "minor admin issue", "fully compliant"]


# ── Fetch one report page ──
def fetch_report(facility_number, index):
    response = http_get(f"{BASE_URL}?facNum={facility_number}&inx={index}")

    if response.status == 200:
        return response.text, "ok"
    if response.status == 400:
        return None, "exhausted"     # this index has no report → stop probing
    return None, "error"             # timeout / 5xx → transient, retry later


# ── One facility: probe visits 0, 1, 2, ... until the API says "no more" ──
def scrape_facility(facility_number):
    consecutive_errors = 0

    for index in count(start=0):
        html, status = fetch_report(facility_number, index)
        sleep(REQUEST_DELAY)

        if status == "exhausted":
            break                    # clean end of this facility's visits

        if status == "error":
            consecutive_errors += 1
            if consecutive_errors >= 3:
                log("giving up — facility may be under-scraped")
                break
            continue                 # skip index, keep probing

        consecutive_errors = 0
        yield parse_report(html, facility_number, index)


# ── One report: raw HTML → one flat row of structured fields ──
def parse_report(html, facility_number, index):
    text = strip_html_to_plain_text(html)

    row = {
        "facility_number": facility_number,
        "visit_index":     index,
        "facility_name":   find(r"FACILITY NAME: (...)", text),
        "census":          find(r"CENSUS: (\d+)", text),
        "visit_type":      "complaint" if "COMPLAINT" in text else "evaluation",
        "visit_subtype":   find(r"TYPE OF VISIT: (...)", text),
    }
    row["incident_driven"] = is_incident_or_complaint(row["visit_type"], row["visit_subtype"])

    # Citations = the regulator's own Type A/B coding → PRIMARY severity signal.
    # Anchored on the literal phrase "Section Cited" to avoid boilerplate matches.
    row["citations"]        = find_all_citations_after("Section Cited", text)
    row["citation_count"]   = len(row["citations"])
    row["deficiency_types"] = collect_type_letters(text)     # A / B / C / D

    # Outcome: substantiated? clean? Structured citation count beats language guessing, so it is checked BEFORE the "no deficiencies found" phrase-matching fallback.
    row["outcome"] = decide_outcome(text, row["citation_count"])

    # Complaint reports carry allegations + verdicts; evaluations leave these blank.
    if row["visit_type"] == "complaint":
        row["allegations"]     = split_allegations(text)
        row["any_substantiated"] = count_substantiated(text) > 0

    # Pull the two prose sections that actually describe what went wrong, kept separate so the classifier can prioritize the deficiency text.
    narrative, deficiencies = extract_findings_text(text)

    row["keyword_score"]  = keyword_severity(narrative + deficiencies)   # fast sanity-check
    row["severity_label"] = classify_severity(deficiencies, narrative)   # the real measure

    return row


# ── Fast, negation-aware keyword score (secondary / diagnostic only) ──
def keyword_severity(text):
    score = 0
    for word in tokenize(text):
        if word in HIGH_SEVERITY_WORDS and not negated_nearby(word):
            score += 3
        elif word in MED_SEVERITY_WORDS and not negated_nearby(word):
            score += 2
        elif word in LOW_SEVERITY_WORDS and not negated_nearby(word):
            score += 1
    return score


# ── Severity via zero-shot NLP (understands context, not just keywords) ──
def classify_severity(deficiencies, narrative):
    if not deficiencies and not narrative:
        return "no_narrative"

    text = truncate(deficiencies + " " + narrative, limit=1000)
    result = zero_shot_model(text, candidate_labels=SEVERITY_LABELS)

    return result.top_label      # e.g. "serious safety risk", confidence attached


# ── Drive everything: read facilities, scrape each, stream to CSV ──
def main(input_csv, output_csv):
    facilities = read_csv(input_csv)
    facilities = drop_closed(facilities)                     # skip non-operating homes
    facilities = pad_facility_numbers_to_9_digits(facilities)  # Excel strips leading zeros
    facilities = skip_already_done(facilities, output_csv)   # resumable across runs

    warm_up_nlp_model()          # load the ~1.6GB model once, before the loop

    buffer = []
    for facility_number in facilities:
        buffer += scrape_facility(facility_number)

        if len(buffer) >= CHECKPOINT_EVERY:     # flush periodically → memory stays flat,
            append_to_csv(buffer, output_csv)   # a crash only loses the unwritten buffer
            buffer = []

    append_to_csv(buffer, output_csv)           # flush the remainder
    log("done")