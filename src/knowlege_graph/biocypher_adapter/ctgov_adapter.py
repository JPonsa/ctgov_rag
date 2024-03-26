#
# Biocypher adaptor for clinicalTrials.gov
# Adapted from https://github.com/biocypher/igan/blob/main/igan/adapters/clinicaltrials_adapter.py
#

import json
from enum import Enum, auto
from itertools import chain
from typing import Optional

import yaml
from biocypher._logger import logger
from tqdm import tqdm

logger.debug(f"Loading module {__name__}.")

import requests

QUERY_PARAMS = {
    "format": "json",
    "query.cond": "(heart failure OR asthma)",
    "filter.overallStatus": ["COMPLETED"],
    "query.term": "AREA[LastUpdatePostDate]RANGE[MIN,2023-02-15]",
    # "query.cond": "iga nephropathy",
    # "query.parser": "advanced",
    # "query.term": "AREA[LastUpdatePostDate]RANGE[2023-01-15,MAX]",
    # "query.locn": "",
    # "query.titles": "",
    # "query.intr": "",
    # "query.outc": "",
    # "query.spons": "",
    # "query.lead": "",
    # "query.id": "",
    # "query.patient": "",
    # "filter.overallStatus": ["NOT_YET_RECRUITING", "RECRUITING"],
    # "filter.geo": "",
    # "filter.ids": ["NCT04852770", "NCT01728545", "NCT02109302"],
    # "filter.advanced": "",
    # "filter.synonyms": "",
    # "postFilter.overallStatus": ["NOT_YET_RECRUITING", "RECRUITING"],
    # "postFilter.geo": "",
    # "postFilter.ids": ["NCT04852770", "NCT01728545", "NCT02109302"],
    # "postFilter.advanced": "",
    # "postFilter.synonyms": "",
    # "aggFilters": "",
    # "geoDecay": "",
    # "fields": ["NCTId", "BriefTitle", "OverallStatus", "HasResults"],
    # "sort": ["@relevance"],
    # "countTotal": False,
    # "pageSize": 10,
    # "pageToken": "",
}


class ctGovAdapterNodeType(Enum):
    """
    Define types of nodes the adapter can provide.
    """

    STUDY = auto()
    ORGANISATION = auto()
    SPONSOR = auto()
    OUTCOME = auto()
    OUTCOME_MEASURES = auto()
    INTERVENTION = auto()
    CONDITION = auto()
    LOCATION = auto()
    ELIGIBILITY = auto()
    BIOSPEC = auto()
    ARM_GROUP = auto()
    INTERVENTION_PROTOCOL = auto()
    OBSERVATION_PROTOCOL = auto()
    ADVERSE_EVENT = auto()
    ADVERSE_EVENT_GROUP = auto()
    ADVERSE_EVENT_PROTOCOL = auto()


class ctGovAdapterStudyField(Enum):
    """
    Define possible fields the adapter can provide for studies.
    """

    ID = "identificationModule/nctId"
    BRIEF_TITLE = "identificationModule/briefTitle"
    OFFICIAL_TITLE = "identificationModule/officialTitle"
    STATUS = "statusModule/overallStatus"
    WHYSTOPPED = "statusModule/whyStopped"
    BRIEF_SUMMARY = "descriptionModule/briefSummary"
    TYPE = "designModule/studyType"
    # ALLOCATION = "designModule/designInfo/allocation"
    PHASES = "designModule/phases"
    MODEL = "designModule/designInfo/interventionModel"
    PRIMARY_PURPOSE = "designModule/designInfo/primaryPurpose"
    NUMBER_OF_PATIENTS = "designModule/enrollmentInfo/count"
    # ELIGIBILITY_CRITERIA = "eligibilityModule/eligibilityCriteria"
    # HEALTHY_VOLUNTEERS = "eligibilityModule/healthyVolunteers"
    # SEX = "eligibilityModule/sex"
    # MINIMUM_AGE = "eligibilityModule/minimumAge"
    # MAXIMUM_AGE = "eligibilityModule/maximumAge"
    # STANDARDISED_AGES = "eligibilityModule/stdAges"


class ctGovAdapterDiseaseField(Enum):
    """
    Define possible fields the adapter can provide for diseases.
    """

    ID = "id"
    NAME = "name"
    DESCRIPTION = "description"


class ctGovAdapterEdgeType(Enum):
    """
    Enum for the types of the protein adapter.
    """

    STUDY_TO_ORGANISATION = auto()
    STUDY_TO_SPONSOR = auto()
    STUDY_TO_OUTCOME = auto()
    STUDY_TO_OUTCOME_MEASURE = auto()
    STUDY_TO_INTERVENTION = auto()
    STUDY_TO_CONDITION = auto()
    STUDY_TO_LOCATION = auto()
    STUDY_TO_ELIGIBILITY = auto()
    STUDY_TO_ARM_GROUP = auto()
    STUDY_TO_BIOSPEC = auto()
    STUDY_TO_OBSERVATION_PROTOCOL = auto()
    STUDY_TO_INTERVENTION_PROTOCOL = auto()

    STUDY_TO_ADVERSE_EVENT = auto()
    STUDY_TO_ADVESE_EVENT_GROUP = auto()
    STUDY_TO_ADVERSE_EVENT_PROTOCOL = auto()


class ctGovAdapter:
    """
    ClinicalTrials BioCypher adapter. Generates nodes and edges for creating a
    knowledge graph.

    Args:
        node_types: List of node types to include in the result.
        node_fields: List of node fields to include in the result.
        edge_types: List of edge types to include in the result.
        edge_fields: List of edge fields to include in the result.
    """

    def __init__(
        self,
        node_types: Optional[list] = None,
        node_fields: Optional[list] = None,
        edge_types: Optional[list] = None,
        edge_fields: Optional[list] = None,
    ):
        self._set_types_and_fields(node_types, node_fields, edge_types, edge_fields)

        self.base_url = "https://clinicaltrials.gov/api/v2"

        self._api_response = self._get_studies(QUERY_PARAMS)

        self._preprocess()

    def _get_studies(self, query_params):
        """
        Get all studies fitting the parameters from the API.

        Args:
            query_params: Dictionary of query parameters to pass to the API.

        Returns:
            A list of studies (dictionaries).
        """
        url = f"{self.base_url}/studies"
        response = requests.get(url, params=query_params)
        result = response.json()
        # append pages until empty
        while result.get("nextPageToken"):
            query_params["pageToken"] = result.get("nextPageToken")
            response = requests.get(url, params=query_params)
            result.get("studies").extend(response.json().get("studies"))
            result["nextPageToken"] = response.json().get("nextPageToken")

        return result.get("studies")

    def _preprocess(self):
        """
        Preprocess raw API results into node and edge types.
        """
        # Nodes
        self._studies = {}
        self._organisations = {}
        self._sponsors = {}
        self._interventions = {}
        self._conditions = {}
        self._outcomes = {}
        self._locations = {}
        self._eligibility = {}
        self._biospec = {}
        self._arm_groups = {}
        # adverse events
        self._adverse_events = {}
        self._adverse_event_group = {}
        self._adverse_event_protocol = {}
        # intervention or observational
        self._intervention_protocols = {}
        self._observation_protocols = {}
        # results
        self._baseline = {}
        self._outcome_measures = {}

        # BUG: outcome measure not getting exported to file

        # Edges
        self._study_to_org_edges = []
        self._study_to_sponsor_edges = []
        self._study_to_intervention_edges = []
        self._study_to_condition_edges = []
        self._study_to_outcome_edges = []
        self._study_to_location_edges = []
        self._study_to_biospec_edges = []
        self._study_to_eligibility_edges = []
        self._study_to_arm_group_edges = []
        self._study_to_adverse_event_edges = []
        self._study_to_adverse_event_group_edges = []
        self._study_to_adverse_event_protocol_edges = []
        self._study_to_intervention_protocol_edges = []
        self._study_to_observation_protocol_edges = []
        self._study_to_outcome_measure_edges = []

        self._intervention_to_arm_group_edges = []

        for study in tqdm(
            self._api_response, desc="Pre-precessing clinical trial protocols"
        ):
            self._preprocess_study(study)

    def _preprocess_study(self, study: dict):

        nct_id = get_recursive(study, "protocolSection.identificationModule.nctId")

        if not nct_id:
            return

        protocol = study.get("protocolSection")
        results = study.get("resultsSection")
        derived = study.get("derivedSection")
        # the derived module has interesting info about conditions and
        # interventions, linking to MeSH terms; could use for diseases and
        # drugs

        # study
        if ctGovAdapterNodeType.STUDY in self.node_types:
            brief_title = get_recursive(protocol, "identificationModule.briefTitle")
            official_title = get_recursive(
                protocol, "identificationModule.officialTitle"
            )
            overall_status = get_recursive(protocol, "statusModule.overallStatus")

            brief_summary = get_recursive(protocol, "descriptionModule.briefSummary")
            detailed_description = get_recursive(
                protocol, "descriptionModule.detailedDescription"
            )
            keywords = get_recursive(protocol, "conditionsModule.keywords")
            phases = get_recursive(protocol, "designModule.phases")
            study_type = get_recursive(protocol, "designModule.studyType")
            why_stopped = get_recursive(protocol, "statusModule.whyStopped")

            if nct_id not in self._studies.keys():
                self._studies.update(
                    {
                        nct_id: {
                            "brief_title": brief_title or "N/A",
                            "official_title": official_title or "N/A",
                            "overall_status": overall_status or "N/A",
                            "brief_summary": brief_summary or "N/A",
                            "keywords": keywords or [],
                            "detailed_description": detailed_description or "N/A",
                            "phases": phases
                            or [
                                "N/A",
                            ],
                            "study_type": study_type or "N/A",
                            "why_stopped": why_stopped or "N/A",
                        }
                    }
                )

        # organisations
        if ctGovAdapterNodeType.ORGANISATION in self.node_types:
            name = get_recursive(protocol, "identificationModule.organization.fullName")
            oclass = get_recursive(protocol, "identificationModule.organization.class")

            if name and name not in self._organisations.keys():
                # org node
                self._organisations.update(
                    {
                        name: {"class": oclass or "N/A"},
                    }
                )

                # study to org edges
                self._study_to_org_edges.append(
                    (
                        None,
                        nct_id,
                        _check_str_format(name),
                        "study_has_org",
                        {},
                    )
                )

        # sponsor
        if ctGovAdapterNodeType.SPONSOR in self.node_types:
            name = get_recursive(
                protocol, "sponsorCollaboratorsModule.leadSponsor.name"
            )
            lclass = get_recursive(
                protocol, "sponsorCollaboratorsModule.leadSponsor.class"
            )

            if name and name not in self._sponsors.keys():
                # sponsor node
                self._sponsors.update(
                    {
                        name: {
                            "class": lclass or "N/A",
                        },
                    }
                )

                # study to sponsor edges
                self._study_to_sponsor_edges.append(
                    (
                        None,
                        nct_id,
                        _check_str_format(name),
                        "study_has_sponsor",
                        {},
                    )
                )

        # outcomes
        if ctGovAdapterNodeType.OUTCOME in self.node_types:
            primary = get_recursive(protocol, "outcomesModule.primaryOutcomes")
            secondary = get_recursive(protocol, "outcomesModule.secondaryOutcomes")
            i = 0
            if primary:
                for outcome in primary:
                    # outcome node
                    # study to outcome edge
                    i += 1
                    self._add_outcome(i, nct_id, outcome, True)

            if secondary:
                for outcome in secondary:
                    # outcome node
                    # study to outcome edge
                    i += 1
                    self._add_outcome(i, nct_id, outcome, False)

        # arm group
        if ctGovAdapterNodeType.ARM_GROUP in self.node_types:
            arms = get_recursive(protocol, "armsInterventionsModule.armGroups")
            if arms:
                for i, arm in enumerate(arms):
                    label = get_recursive(arm, "label")
                    atype = get_recursive(arm, "type")
                    description = get_recursive(arm, "description")
                    if description:
                        description = replace_quote(description)
                        description = replace_newline(description)

                    id = f"{nct_id}_{i}"
                    if id not in self._arm_groups.keys():

                        # arm group node
                        self._arm_groups.update(
                            {
                                id: {
                                    "label": label or "N/A",
                                    "type": atype or "N/A",
                                    "description": description or "N/A",
                                }
                            }
                        )

                        # study to arm group edge
                        self._study_to_arm_group_edges.append(
                            (None, nct_id, id, "study_has_arm_group", {})
                        )

        # intervention
        if ctGovAdapterNodeType.INTERVENTION in self.node_types:
            interventions = get_recursive(
                protocol, "armsInterventionsModule.interventions"
            )

            if interventions:
                for intervention in interventions:
                    name = get_recursive(intervention, "name")
                    other_names = get_recursive(intervention, "otherNames")
                    intervention_type = get_recursive(intervention, "type")
                    description = get_recursive(intervention, "description")
                    arm_labels = get_recursive(intervention, "armGroupLabels")

                    if name:
                        # intervention node
                        name = name.capitalize()
                        if name not in self._interventions.keys():
                            self._interventions.update(
                                {
                                    name: {
                                        "type": intervention_type or "N/A",
                                        "description": description or "N/A",
                                        "other_names": other_names or "N/A",
                                    },
                                }
                            )

                        # study to intervention edge
                        self._study_to_intervention_edges.append(
                            (
                                None,
                                nct_id,
                                _check_str_format(name),
                                "study_has_intervention",
                                {},
                            )
                        )
                        # intervention to arm group edge
                        if arm_labels:
                            for arm in arm_labels:
                                arm = f"{nct_id}_{arm}"
                                if arm in self._arm_groups.keys():
                                    self._intervention_to_arm_group_edges.append(
                                        (
                                            None,
                                            _check_str_format(name),
                                            arm,
                                            "intervention_has_arm",
                                            {},
                                        )
                                    )

        # condition aka disease
        if ctGovAdapterNodeType.CONDITION in self.node_types:

            meshes = get_recursive(derived, "conditionBrowseModule.meshes")

            if meshes:
                for mesh in meshes:
                    condition = mesh.get("term")
                    mesh_id = mesh.get("id")
                    if condition not in self._conditions.keys():
                        # condition node
                        self._conditions.update(
                            {
                                condition: {"mesh_id": mesh_id},
                            }
                        )

                        # study to condition edges
                        self._study_to_condition_edges.append(
                            (
                                None,
                                nct_id,
                                condition,
                                "study_has_condition",
                                {},
                            )
                        )

        # locations
        if ctGovAdapterNodeType.LOCATION in self.node_types:

            locations = get_recursive(protocol, "contactsLocationsModule.locations")

            if locations:
                for location in locations:
                    facility = get_recursive(location, "facility")
                    city = get_recursive(location, "city")
                    state = get_recursive(location, "state")
                    country = get_recursive(location, "country")

                    try:
                        name = ", ".join([facility, city, country])
                        name = _check_str_format(name)
                    except TypeError:
                        name = None

                    if name:
                        if name not in self._locations.keys():
                            self._locations.update(
                                {
                                    name: {
                                        "city": city or "N/A",
                                        "state": state or "N/A",
                                        "country": country or "N/A",
                                    },
                                }
                            )

                        # study to location edges
                        self._study_to_location_edges.append(
                            (
                                None,
                                nct_id,
                                _check_str_format(name),
                                "study_has_location",
                                {},
                            )
                        )

        # eligibility
        if ctGovAdapterNodeType.ELIGIBILITY in self.node_types:
            eligibility = get_recursive(protocol, "eligibilityModule")

            if eligibility:
                sex = get_recursive(eligibility, "sex")
                healthy_volunteers = get_recursive(eligibility, "healthyVolunteers")
                min_age = get_recursive(eligibility, "minimumAge")
                max_age = get_recursive(eligibility, "maximumAge")
                standarised_ages = get_recursive(eligibility, "stdAges")
                eligibility_criteria = get_recursive(eligibility, "eligibilityCriteria")

                id = f"{nct_id}_eligibility"

                if id not in self._eligibility.keys():
                    # create node
                    self._eligibility.update(
                        {
                            id: {
                                "sex": sex or "N/A",
                                "healthy_volunteers": healthy_volunteers or False,
                                "minimum_age": min_age or None,
                                "maximum_age": max_age or None,
                                "standarised_ages": standarised_ages or "N/A",
                                "eligibility_criteria": eligibility_criteria or "N/A",
                            },
                        }
                    )
                    # create study to eligibity edges
                    self._study_to_eligibility_edges.append(
                        (None, nct_id, id, "study_has_eligibility", {})
                    )

        # biospeciment (aka biospec)
        if ctGovAdapterNodeType.BIOSPEC in self.node_types:
            retention = get_recursive(protocol, "designModule.bioSpec.retention")
            description = get_recursive(protocol, "designModule.bioSpec.description")
            if retention:
                id = f"{nct_id}_biospec"
                if id not in self._biospec.keys():
                    self._biospec.update(
                        {
                            id: {
                                "retention": retention or "N/A",
                                "description": description.capitalize() or "N/A",
                            },
                        }
                    )

                    self._study_to_biospec_edges.append(
                        (
                            None,
                            nct_id,
                            id,
                            "study_has_biospec",
                            {"retention": retention},
                        )
                    )

        # intervention protocol
        if ctGovAdapterNodeType.INTERVENTION_PROTOCOL in self.node_types:
            model = get_recursive(protocol, "designModule.designInfo.interventionModel")
            description = get_recursive(
                protocol, "designModule.designInfo.interventionModelDescription"
            )
            allocation = get_recursive(protocol, "designModule.designInfo.allocation")

            masking = get_recursive(
                protocol, "designModule.designInfo.maskingInfo.whoMasked"
            )

            masking_desc = get_recursive(
                protocol,
                "designModule.designInfo.maskingInfo.maskingDescription",
            )

            if model:
                id = f"{nct_id}_int_protocol"
                if id not in self._intervention_protocols.keys():
                    # intervention protocol node
                    self._intervention_protocols.update(
                        {
                            id: {
                                "model": model or "N/A",
                                "description": description or "N/A",
                                "allocation": allocation or "N/A",
                                "masking": masking or [],
                                "masking_description": masking_desc or "N/A",
                            }
                        }
                    )

                    # study to intervention protocol edges
                    self._study_to_intervention_protocol_edges.append(
                        (
                            None,
                            nct_id,
                            id,
                            "study_has_intervention_protocol",
                            {},
                        )
                    )

        # observation protocol
        if ctGovAdapterNodeType.OBSERVATION_PROTOCOL in self.node_types:
            model = get_recursive(
                protocol, "designModule.designInfo.observationalModel"
            )
            time_perspective = get_recursive(
                protocol, "designModule.designInfo.timePerspective"
            )
            population = get_recursive(protocol, "eligibilityModule.studyPopulation")
            patient_registry = get_recursive(protocol, "designModule.patientRegistry")
            sampling_method = get_recursive(
                protocol, "eligibilityModule.samplingMethod"
            )

            if model:
                id = f"{nct_id}_obs_protocol"
                if nct_id not in self._observation_protocols.keys():
                    # observation protocol node
                    self._observation_protocols.update(
                        {
                            id: {
                                "model": model or "N/A",
                                "time_perspective": time_perspective or "N/A",
                                "population": population or "N/A",
                                "patient_registry": patient_registry or False,
                                "sampling_method": sampling_method or "N/A",
                            },
                        }
                    )

                    # study to observation protocol edges
                    self._study_to_observation_protocol_edges.append(
                        (
                            None,
                            nct_id,
                            id,
                            "study_has_observation_protocol",
                            {},
                        )
                    )

        # results -  outcome measures
        if results and ctGovAdapterNodeType.OUTCOME_MEASURES in self.node_types:
            outcome_m = get_recursive(results, ".outcomeMeasuresModule.outcomeMeasures")

            if outcome_m:
                for i, outcome in enumerate(outcome_m):
                    outcome_type = get_recursive(outcome_m, "type")
                    id = f"{nct_id}_outcome_measure_{i}"
                    if id not in self._outcome_measures.keys():

                        # outcome measure nodes
                        self._outcome_measures.update(
                            {
                                id: {"description": outcome},
                            }
                        )

                        # study to outcome measure edges
                        self._study_to_outcome_measure_edges.append(
                            (
                                None,
                                nct_id,
                                id,
                                "study_has_outcome_measure",
                                {"type:": outcome_type or "N/A"},
                            )
                        )

        # Adverse Event protocol
        if results and ctGovAdapterNodeType.ADVERSE_EVENT_PROTOCOL in self.node_types:
            description = get_recursive(results, "adverseEventsModule.description")
            timeframe = get_recursive(results, "adverseEventsModule.timeFrame")

            if description:
                id = f"{nct_id}_adverse_event_protocol"
                if id not in self._adverse_event_protocol.keys():
                    # adverser event protocol node
                    self._adverse_event_protocol.update(
                        {
                            id: {
                                "description": description or "N/A",
                                "timeframe": timeframe or "N/A",
                            },
                        }
                    )

                    # study to adverse event protocol edges
                    self._study_to_adverse_event_protocol_edges.append(
                        (None, nct_id, id, "study_has_adverse_event_protocol", {})
                    )
        if results and ctGovAdapterNodeType.ADVERSE_EVENT_GROUP in self.node_types:
            groups = get_recursive(results, "adverseEventsModule.eventGroups")
            if groups:
                for group in groups:
                    id = get_recursive(group, "id")
                    title = get_recursive(group, "title")
                    description = get_recursive(group, "description")
                    g_id = f"{nct_id}_{id}"
                    if g_id not in self._adverse_event_group.keys():
                        # event group node

                        self._adverse_event_group.update(
                            {
                                g_id: {
                                    "id": id,
                                    "title": title or "N/A",
                                    "description": description or "N/A",
                                },
                            }
                        )
                        # study to event group edges
                        self._study_to_adverse_event_group_edges.append(
                            (
                                None,
                                nct_id,
                                g_id,
                                "study_has_adverse_event_group",
                                {},
                            )
                        )

        # adverse events
        if results and ctGovAdapterNodeType.ADVERSE_EVENT in self.node_types:

            serious = get_recursive(results, "adverseEventsModule.seriousEvents")
            other = get_recursive(results, "adverseEventsModule.otherEvents")
            i = 0
            if serious:
                for event in serious:
                    i += 1
                    self._add_adverse_event(i, nct_id, event, True)

            if other:
                for event in other:
                    i += 1
                    self._add_adverse_event(i, nct_id, event, False)

    def _add_outcome(self, i: int, nct_id: str, outcome: dict, primary: bool) -> None:
        """Add an outcome to the internal data structures.

        Args:
            i (int): The index of the outcome.
            nct_id (str): study id.
            outcome (dict): A dictionary containing information about the outcome.
            primary (bool): A flag indicating whether is primary (True) or secondary (False) outcome.

        Returns:
            None
        """

        measure = get_recursive(outcome, "measure")
        time_frame = get_recursive(outcome, "timeFrame")
        description = get_recursive(outcome, "description")

        if measure:
            name = f"{nct_id}_outcome_{i}"
            if name not in self._outcomes.keys():
                self._outcomes.update(
                    {
                        name: {
                            "measure": measure,
                            "time_frame": time_frame or "N/A",
                            "description": description or "N/A",
                            "primary": primary,
                        },
                    }
                )

                self._study_to_outcome_edges.append(
                    (
                        None,
                        nct_id,
                        _check_str_format(name),
                        "study_has_outcome",
                        {},
                    )
                )

    def _add_adverse_event(
        self, i: int, nct_id: str, adverse_event: dict, serious: bool
    ) -> None:
        """Add an adverse event to the internal data structures.

        Args:
            i (int): The index of the adverse event.
            nct_id (str): study id.
            adverse_event (dict): A dictionary containing information about the adverse event.
            serious (bool): A flag indicating whether the adverse event is serious.

        Returns:
            None
        """

        term = get_recursive(adverse_event, "term")
        organ_system = get_recursive(adverse_event, "organSystem")
        source_vocabulary = get_recursive(adverse_event, "sourceVocabulary")
        assessment_type = get_recursive(adverse_event, "assessmentType")
        notes = get_recursive(adverse_event, "notes")
        stats = get_recursive(adverse_event, "stats")

        if term:
            id = f"{nct_id}_AE_{i}"
            if id not in self._adverse_events.keys():
                self._adverse_events.update(
                    {
                        id: {
                            "term": term.capitalize(),
                            "serious_event": serious,
                            "organ_system": organ_system or "N/A",
                            "source_vocabulary": source_vocabulary or "N/A",
                            "assessment_type": assessment_type or "N/A",
                            "notes": notes or "N/A",
                            "stats": stats or "N/A",
                        },
                    }
                )
                self._study_to_adverse_event_edges.append(
                    (None, nct_id, id, "study_has_adverse_event", {})
                )

    def get_nodes(self):
        """
        Returns a generator of node tuples for node types specified in the
        adapter constructor.
        """

        logger.info("Generating nodes.")

        if ctGovAdapterNodeType.STUDY in self.node_types:
            for name, props in self._studies.items():
                name = _check_str_format(name)
                formatted_props = check_node_props(props)
                yield (name, "study", formatted_props)

        if ctGovAdapterNodeType.ORGANISATION in self.node_types:
            for name, props in self._organisations.items():
                name = _check_str_format(name)
                formatted_props = check_node_props(props)
                yield (name, "organisation", formatted_props)

        if ctGovAdapterNodeType.SPONSOR in self.node_types:
            for name, props in self._sponsors.items():
                name = _check_str_format(name)
                formatted_props = check_node_props(props)
                yield (name, "sponsor", formatted_props)

        if ctGovAdapterNodeType.OUTCOME in self.node_types:
            for name, props in self._outcomes.items():
                name = _check_str_format(name)
                formatted_props = check_node_props(props)
                yield (name, "outcome", formatted_props)

        if ctGovAdapterNodeType.OUTCOME_MEASURES in self.node_types:
            for name, props in self._outcome_measures.items():
                name = _check_str_format(name)
                formatted_props = check_node_props(props)
                yield (name, "outcome_measure", formatted_props)

        if ctGovAdapterNodeType.INTERVENTION in self.node_types:
            for name, props in self._interventions.items():
                name = _check_str_format(name)
                formatted_props = check_node_props(props)
                yield (name, "intervention", formatted_props)

        if ctGovAdapterNodeType.CONDITION in self.node_types:
            for name, props in self._conditions.items():
                name = _check_str_format(name)
                formatted_props = check_node_props(props)
                yield (name, "condition", formatted_props)

        if ctGovAdapterNodeType.LOCATION in self.node_types:
            for name, props in self._locations.items():
                name = _check_str_format(name)
                formatted_props = check_node_props(props)
                yield (name, "location", formatted_props)

        if ctGovAdapterNodeType.ELIGIBILITY in self.node_types:
            for name, props in self._eligibility.items():
                name = _check_str_format(name)
                formatted_props = check_node_props(props)

                yield (name, "eligibility", formatted_props)

        if ctGovAdapterNodeType.BIOSPEC in self.node_types:
            for name, props in self._biospec.items():
                name = _check_str_format(name)
                formatted_props = check_node_props(props)
                yield (name, "biospec", formatted_props)

        if ctGovAdapterNodeType.ARM_GROUP in self.node_types:
            for name, props in self._arm_groups.items():
                name = _check_str_format(name)
                formatted_props = check_node_props(props)
                yield (name, "arm_group", formatted_props)

        if ctGovAdapterNodeType.INTERVENTION_PROTOCOL in self.node_types:
            for name, props in self._intervention_protocols.items():
                name = _check_str_format(name)
                formatted_props = check_node_props(props)
                yield (name, "intervention_protocol", formatted_props)

        if ctGovAdapterNodeType.OBSERVATION_PROTOCOL in self.node_types:
            for name, props in self._observation_protocols.items():
                name = _check_str_format(name)
                formatted_props = check_node_props(props)
                yield (name, "observation_protocol", formatted_props)

        if ctGovAdapterNodeType.ADVERSE_EVENT in self.node_types:
            for name, props in self._adverse_events.items():
                name = _check_str_format(name)
                formatted_props = check_node_props(props)
                yield (name, "adverse_event", formatted_props)

        if ctGovAdapterNodeType.ADVERSE_EVENT_GROUP in self.node_types:
            for name, props in self._adverse_event_group.items():
                name = _check_str_format(name)
                formatted_props = check_node_props(props)
                yield (name, "adverse_event_group", formatted_props)

        if ctGovAdapterNodeType.ADVERSE_EVENT_PROTOCOL in self.node_types:
            for name, props in self._adverse_event_protocol.items():
                name = _check_str_format(name)
                formatted_props = check_node_props(props)
                yield (name, "adverse_event_protocol", formatted_props)

    def get_edges(self):
        """
        Returns a generator of edge tuples for edge types specified in the
        adapter constructor.
        """

        logger.info("Generating edges.")

        if ctGovAdapterEdgeType.STUDY_TO_INTERVENTION in self.edge_types:
            yield from self._study_to_intervention_edges

        if ctGovAdapterEdgeType.STUDY_TO_CONDITION in self.edge_types:
            yield from self._study_to_condition_edges

        if ctGovAdapterEdgeType.STUDY_TO_OUTCOME in self.edge_types:
            yield from self._study_to_outcome_edges

        if ctGovAdapterEdgeType.STUDY_TO_OUTCOME_MEASURE in self.edge_types:
            yield from self._study_to_outcome_measure_edges

        if ctGovAdapterEdgeType.STUDY_TO_LOCATION in self.edge_types:
            yield from self._study_to_location_edges

        if ctGovAdapterEdgeType.STUDY_TO_SPONSOR in self.edge_types:
            yield from self._study_to_sponsor_edges

        if ctGovAdapterEdgeType.STUDY_TO_BIOSPEC in self.edge_types:
            yield from self._study_to_biospec_edges

        if ctGovAdapterEdgeType.STUDY_TO_ADVERSE_EVENT in self.edge_types:
            yield from self._study_to_adverse_event_edges

        if ctGovAdapterEdgeType.STUDY_TO_ADVESE_EVENT_GROUP in self.edge_types:
            yield from self._study_to_adverse_event_group_edges

        if ctGovAdapterEdgeType.STUDY_TO_ADVERSE_EVENT_PROTOCOL in self.edge_types:
            yield from self._study_to_adverse_event_protocol_edges

        if ctGovAdapterEdgeType.STUDY_TO_ARM_GROUP in self.edge_types:
            yield from self._study_to_arm_group_edges

        if ctGovAdapterEdgeType.STUDY_TO_ELIGIBILITY in self.edge_types:
            yield from self._study_to_eligibility_edges

        if ctGovAdapterEdgeType.STUDY_TO_INTERVENTION_PROTOCOL in self.edge_types:
            yield from self._study_to_intervention_protocol_edges

        if ctGovAdapterEdgeType.STUDY_TO_OBSERVATION_PROTOCOL in self.edge_types:
            yield from self._study_to_observation_protocol_edges

    def _set_types_and_fields(self, node_types, node_fields, edge_types, edge_fields):
        if node_types:
            self.node_types = node_types
        else:
            self.node_types = [type for type in ctGovAdapterNodeType]

        if node_fields:
            self.node_fields = node_fields
        else:
            self.node_fields = [
                field
                for field in chain(
                    ctGovAdapterStudyField,
                    ctGovAdapterDiseaseField,
                )
            ]

        if edge_types:
            self.edge_types = edge_types
        else:
            self.edge_types = [type for type in ctGovAdapterEdgeType]

        if edge_fields:
            self.edge_fields = edge_fields
        else:
            self.edge_fields = [field for field in chain()]


def replace_quote(string):
    return string.replace('"', "'")


def replace_newline(string):
    return string.replace("\n", " | ")


def _get_recursive(data, keys):
    if "." in keys:
        current_key, remaining_keys = keys.split(".", 1)
        return get_recursive(data.get(current_key, {}), remaining_keys)
    else:
        return data.get(keys)


def get_recursive(data, key):
    try:
        x = _get_recursive(data, key)
    except AttributeError:
        x = None

    return x


def _check_str_format(x):

    if x is None:
        return "N/A"

    if isinstance(x, str):
        x = replace_quote(x)
        return x

    if isinstance(x, list):
        return [_check_str_format(i) for i in x]

    # Nested dictionaries are not supported
    if isinstance(x, dict):
        return _check_str_format(yaml.dump(x))

    return x


def check_node_props(props: dict) -> dict:
    formatted_props = {}
    for k, v in props.items():
        formatted_props[k] = _check_str_format(v)
    return formatted_props
