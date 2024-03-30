# Write AACT ctgov schema into a file
# TODO: currently old db schema format.
# Update to produce langchain or llamaindex like schema

import os
import re

import psycopg2
from dotenv import load_dotenv

load_dotenv(".env")
AACT_USER = os.getenv("AACT_USER")
AACT_PWD = os.getenv("AACT_PWD")

# fmt: off
enumerated_fields = {
    "Status": ["ACTIVE_NOT_RECRUITING", "COMPLETED", "ENROLLING_BY_INVITATION", "NOT_YET_RECRUITING", "RECRUITING", "SUSPENDED", "TERMINATED", "WITHDRAWN", "AVAILABLE", "NO_LONGER_AVAILABLE", "TEMPORARILY_NOT_AVAILABLE", "APPROVED_FOR_MARKETING", "WITHHELD", "UNKNOWN"],
    "StudyType": ["EXPANDED_ACCESS", "INTERVENTIONAL", "OBSERVATIONAL"],
    "Phase": ["NA", "EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4"],
    "Sex": ["FEMALE", "MALE", "ALL"],
    "StandardAge": ["CHILD", "ADULT", "OLDER_ADULT"],
    "SamplingMethod": ["PROBABILITY_SAMPLE", "NON_PROBABILITY_SAMPLE"],
    "IpdSharing": ["YES", "NO", "UNDECIDED"],
    "IpdSharingInfoType": ["STUDY_PROTOCOL", "SAP", "ICF", "CSR", "ANALYTIC_CODE"],
    "OrgStudyIdType": ["NIH", "FDA", "VA", "CDC", "AHRQ", "SAMHSA"],
    "SecondaryIdType": ["NIH", "FDA", "VA", "CDC", "AHRQ", "SAMHSA", "OTHER_GRANT", "EUDRACT_NUMBER", "CTIS", "REGISTRY", "OTHER"],
    "AgencyClass": ["NIH", "FED", "OTHER_GOV", "INDIV", "INDUSTRY", "NETWORK", "AMBIG", "OTHER", "UNKNOWN"],
    "ExpandedAccessStatus": ["AVAILABLE", "NO_LONGER_AVAILABLE", "TEMPORARILY_NOT_AVAILABLE", "APPROVED_FOR_MARKETING"],
    "DateType": ["ACTUAL", "ESTIMATED"],
    "ResponsiblePartyType": ["SPONSOR", "PRINCIPAL_INVESTIGATOR", "SPONSOR_INVESTIGATOR"],
    "DesignAllocation": ["RANDOMIZED", "NON_RANDOMIZED", "NA"],
    "InterventionalAssignment": ["SINGLE_GROUP", "PARALLEL", "CROSSOVER", "FACTORIAL", "SEQUENTIAL"],
    "PrimaryPurpose": ["TREATMENT", "PREVENTION", "DIAGNOSTIC", "ECT", "SUPPORTIVE_CARE", "SCREENING", "HEALTH_SERVICES_RESEARCH", "BASIC_SCIENCE", "DEVICE_FEASIBILITY", "OTHER"],
    "ObservationalModel": ["COHORT", "CASE_CONTROL", "CASE_ONLY", "CASE_CROSSOVER", "ECOLOGIC_OR_COMMUNITY", "FAMILY_BASED", "DEFINED_POPULATION", "NATURAL_HISTORY", "OTHER"],
    "DesignTimePerspective": ["RETROSPECTIVE", "PROSPECTIVE", "CROSS_SECTIONAL", "OTHER"],
    "BioSpecRetention": ["NONE_RETAINED", "SAMPLES_WITH_DNA", "SAMPLES_WITHOUT_DNA"],
    "EnrollmentType": ["ACTUAL", "ESTIMATED"],
    "ArmGroupType": ["EXPERIMENTAL", "ACTIVE_COMPARATOR", "PLACEBO_COMPARATOR", "SHAM_COMPARATOR", "NO_INTERVENTION", "OTHER"],
    "InterventionType": ["BEHAVIORAL", "BIOLOGICAL", "COMBINATION_PRODUCT", "DEVICE", "DIAGNOSTIC_TEST", "DIETARY_SUPPLEMENT", "DRUG", "GENETIC", "PROCEDURE", "RADIATION", "OTHER"],
    "ContactRole": ["STUDY_CHAIR", "STUDY_DIRECTOR", "PRINCIPAL_INVESTIGATOR", "SUB_INVESTIGATOR", "CONTACT"],
    "OfficialRole": ["STUDY_CHAIR", "STUDY_DIRECTOR", "PRINCIPAL_INVESTIGATOR", "SUB_INVESTIGATOR"],
    "RecruitmentStatus": ["ACTIVE_NOT_RECRUITING", "COMPLETED", "ENROLLING_BY_INVITATION", "NOT_YET_RECRUITING", "RECRUITING", "SUSPENDED", "TERMINATED", "WITHDRAWN", "AVAILABLE"],
    "ReferenceType": ["BACKGROUND", "RESULT", "DERIVED"],
    "MeasureParam": ["GEOMETRIC_MEAN", "GEOMETRIC_LEAST_SQUARES_MEAN", "LEAST_SQUARES_MEAN", "LOG_MEAN", "MEAN", "MEDIAN", "NUMBER", "COUNT_OF_PARTICIPANTS", "COUNT_OF_UNITS"],
    "MeasureDispersionType": ["NA", "STANDARD_DEVIATION", "STANDARD_ERROR", "INTER_QUARTILE_RANGE", "FULL_RANGE", "CONFIDENCE_80", "CONFIDENCE_90", "CONFIDENCE_95", "CONFIDENCE_975", "CONFIDENCE_99", "CONFIDENCE_OTHER", "GEOMETRIC_COEFFICIENT"],
    "OutcomeMeasureType": ["PRIMARY", "SECONDARY", "OTHER_PRE_SPECIFIED", "POST_HOC"],
    "ReportingStatus": ["NOT_POSTED", "POSTED"],
    "EventAssessment": ["NON_SYSTEMATIC_ASSESSMENT", "SYSTEMATIC_ASSESSMENT"],
    "AgreementRestrictionType": ["LTE60", "GT60", "OTHER"],
    "BrowseLeafRelevance": ["LOW", "HIGH"],
    "DesignMasking": ["NONE", "SINGLE", "DOUBLE", "TRIPLE", "QUADRUPLE"],
    "WhoMasked": ["PARTICIPANT", "CARE_PROVIDER", "INVESTIGATOR", "OUTCOMES_ASSESSOR"],
    "AnalysisDispersionType": ["STANDARD_DEVIATION", "STANDARD_ERROR_OF_MEAN"],
    "ConfidenceIntervalNumSides": ["ONE_SIDED", "TWO_SIDED"],
    "NonInferiorityType": ["SUPERIORITY", "NON_INFERIORITY", "EQUIVALENCE", "OTHER", "NON_INFERIORITY_OR_EQUIVALENCE", "SUPERIORITY_OR_OTHER", "NON_INFERIORITY_OR_EQUIVALENCE_LEGACY", "SUPERIORITY_OR_OTHER_LEGACY"],
    "UnpostedEventType": ["RESET", "RELEASE", "UNRELEASE"],
    "ViolationEventType": ["VIOLATION_IDENTIFIED", "CORRECTION_CONFIRMED", "PENALTY_IMPOSED"]
}

def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

tmp = {}
for k,v in enumerated_fields.items():
    k = camel_to_snake(k)
    v = [x.capitalize() for x in v]
    tmp[k] = v
enumerated_fields = tmp

trimmed_tables = [
    "browse_interventions",
    "sponsors",
    "outcome_analysis_groups",
    #  'reported_event_totals',
    #  'ipd_information_types',
    #  'search_results',
    #  'documents',
    #  'links',
    #  'result_agreements',
    # #  'pending_results',
    # "detailed_descriptions",
    # "interventions",
    "facilities",
    "studies",
    "outcomes",
    "browse_conditions",
    #  'outcome_counts',
    #  'milestones',
    #  'overall_officials',
    "outcome_analyses",
    "keywords",
    #  'intervention_other_names',
    #  'conditions',
    #  'calculated_values',
    "eligibilities",
    #  'mesh_headings',
    "id_information",
    "design_group_interventions",
    #  'participant_flows',
    #  'result_contacts',
    "reported_events",
    #  'responsible_parties',
    #  'study_references',
    #  'result_groups',
    #  'mesh_terms',
    #  'provided_documents',
    #  'study_searches',
    #  'retractions',
    "brief_summaries",
    #  'baseline_measurements',
    #  'central_contacts',
    #  'baseline_counts',
    #  'facility_contacts',
    "designs",
    "drop_withdrawals",
    #  'facility_investigators',
    "outcome_measurements",
    #  'all_interventions',
    #  'all_keywords',
    #  'all_overall_official_affiliations',
    #  'all_overall_officials',
    #  'all_primary_outcome_measures',
    #  'all_secondary_outcome_measures',
    #  'all_sponsors',
    #  'all_states',
    #  'categories',
    #  'covid_19_studies',
    #  'design_outcomes',
    #  'design_groups',
    "countries",
    #  'all_browse_conditions',
    #  'all_browse_interventions',
    #  'all_cities',
    #  'all_conditions',
    #  'all_countries',
    #  'all_design_outcomes',
    #  'all_facilities',
    #  'all_group_types',
    #  'all_id_information',
    #  'all_intervention_types'
]

trimmed_study_columns = [
    "nct_id",
    "brief_title",
    "study_type",
    "overall_status",
    "why_stopped",
]


def connect_to_aact():
    conn = psycopg2.connect(
        database="aact",
        host="aact-db.ctti-clinicaltrials.org",
        user=AACT_USER,
        password=AACT_PWD,
        port=5432,
    )
    return conn


def aact_run_sql_query(conn, sql_query: str) -> list:
    cur = conn.cursor()
    cur.execute(sql_query)
    rows = cur.fetchall()
    cur.close()
    return rows


def get_list_tables(conn):
    sql_query = "SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE table_schema = 'ctgov'"
    response = aact_run_sql_query(conn, sql_query)
    tables = [r[2] for r in response]
    return tables


def get_columns_table(conn, table_name) -> list:
    sql_query = f"""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = '{table_name}';
    """
    response = aact_run_sql_query(conn, sql_query)
    return response


def get_enumerated_col_values(conn, table, col):
    sql_query = f"""
    SELECT distinct {col} FROM {table}
    """
    response = aact_run_sql_query(conn, sql_query)
    response = set([x[0] for x in response]) - set([" ", "", None])
    response = list(response)
    return response

def main(trimmed: bool = True):
    conn = connect_to_aact()
    tables = get_list_tables(conn)

    schema = '{"tables": [\n'

    if trimmed:
        tables = set(tables).intersection(set(trimmed_tables))

    I = len(tables) - 1
    for i, table in enumerate(tables):
        columns = get_columns_table(conn, table)

        columns = set(columns) - set(
            "id",
        )

        if (table == "studies") and trimmed:
            columns = set(columns).intersection(set(trimmed_study_columns))
        schema += "\t{\n"
        schema += f'\t\t"name":"{table}",\n'
        schema += '\t\t"columns":[\n'
        J = len(columns) - 1
        for j, (col, t) in enumerate(columns):
            
            if col in enumerated_fields.keys():
                print(f"{table} {col}")
                values = get_enumerated_col_values(conn, table, col)
                t = f'ENUM({", ".join(values)})'
            
            t = t.replace("character varying", "text")
            schema += '\t\t\t{"' + f'name":"{col}", "type":"{t}"' + "}"
            if j < J:
                schema += ","
            schema += "\n"
        schema += "\t\t]\n"
        schema += "\t}"
        if i < I:
            schema += ","
        schema += "\n"
    schema += "]\n}"

    with open("./data/aact_ctgov_schema.json", "w") as f:
        f.write(schema)


if __name__ == "__main__":
    main(trimmed=True)
    # BUG: currently when generating the trimmed version it add 2
    # # extra commas due to skipping rows and tables
