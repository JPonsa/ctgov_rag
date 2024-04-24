# Takes a collection for Clinical Trials from clinicalTrial.gov
# and create a set of question-context-answer triplets.


__author__ = "Joan Ponsa"

import argparse
import json
import os
import sys

import dspy
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

####### Add src folder to the system path so it can call utils
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the current script
parent_dir = os.path.dirname(script_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from utils.utils import connect_to_mongoDB

load_dotenv(".env")


class QAwithContext(dspy.Signature):
    """Give a question and a context, produce an answer"""

    question: str = dspy.InputField(prefix="Question:")
    context: str = dspy.InputField(
        prefix="Context:", desc="May contain useful information to answer the question."
    )
    answer: str = dspy.OutputField(prefix="Answer:", desc="final response")


qa_with_context = dspy.Predict(QAwithContext)


def _get_recursive(data, keys):
    if "." in keys:
        current_key, remaining_keys = keys.split(".", 1)

        if isinstance(data.get(current_key, {}), list):
            x = []
            for i in data.get(current_key, {}):
                x.append(get_recursive(i, remaining_keys))
            return x

        return get_recursive(data.get(current_key, {}), remaining_keys)
    else:
        return data.get(keys)


def get_recursive(data, key):
    try:
        x = _get_recursive(data, key)
    except AttributeError:
        x = None

    return x


class CtGovStudyQuestioner:

    # Total number of questions
    N = 25

    GENERIC_QUESTIONS_IDX = [1, 2, 3, 4, 5, 8, 9, 10, 11, 24]
    INTERVENTIONAL_STUDIES_QUESTIONS_IDX = [6, 7, 13, 15, 16, 17, 18]
    OBSERVATIONAL_STUDIES_QUESTIONS_IDX = [19, 20, 21, 22]
    MISSING_QUESTIODS_IDX = [12, 14]

    KEY_ENTITIES = {
        "nctId": "protocolSection.identificationModule.nctId",
        "study_type": "protocolSection.designModule.studyType",
        "conditions": "derivedSection.conditionBrowseModule.meshes.term",
        "interventions": "protocolSection.armsInterventionsModule.interventions.name",
    }

    GENERIC_CONDITIONS = [
        "Heart Failure",
        "Diabetes Mellitus",
        "Hypertension",
        "Asthma",
        "Chronic Obstructive Pulmonary Disease (COPD)",
        "Coronary Artery Disease",
        "Stroke",
        "Depression",
        "Anxiety Disorders",
        "Rheumatoid Arthritis",
        "Osteoarthritis",
        "Breast Cancer",
        "Prostate Cancer",
        "Colorectal Cancer",
        "HIV/AIDS",
        "Alzheimer's Disease",
        "Parkinson's Disease",
        "Schizophrenia",
        "Bipolar Disorder",
        "Chronic Kidney Disease",
        "Chronic Liver Disease",
    ]

    def __init__(self, study):
        self.study = study
        self.nctId = get_recursive(study, self.KEY_ENTITIES["nctId"]) or None
        self.study_type = get_recursive(study, self.KEY_ENTITIES["study_type"]) or None
        self.conditions = get_recursive(study, self.KEY_ENTITIES["conditions"]) or [
            None,
        ]
        self.interventions = get_recursive(
            study, self.KEY_ENTITIES["interventions"]
        ) or [
            None,
        ]
        self.other_conditions = list(
            set(self.GENERIC_CONDITIONS) - set(self.conditions)
        )

        if self.study_type == "INTERVENTIONAL":
            self.study_question_space = sorted(
                self.GENERIC_QUESTIONS_IDX + self.INTERVENTIONAL_STUDIES_QUESTIONS_IDX
            )
        elif self.study_type == "OBSERVATIONAL":
            self.study_question_space = sorted(
                self.GENERIC_QUESTIONS_IDX + self.OBSERVATIONAL_STUDIES_QUESTIONS_IDX
            )
        elif self.study_type == "EXPANDED_ACCESS":
            return None
        else:
            raise NotImplementedError(f"{self.study_type} not implemented")

    # Question N - Template
    def question_n(self):
        study = self.study
        nctId = self.nctId
        # Question
        # TODO : add question
        questions = []
        question = np.random.choice(questions, 1)
        # Context
        # TODO: add context
        key = ""
        context = get_recursive(study, key) or "NA"
        # Answer
        # TODO : add answer
        key = ""
        answer = get_recursive(study, key) or "NA"
        return question, context, answer

    # Question 1 - What is the title in clinical trial (a.k.a. study) {nctId}?
    def question_1(self):
        study = self.study
        nctId = self.nctId
        # Question
        questions = [
            f"What is the title in clinical trial (a.k.a. study) {nctId}?",
        ]
        question = np.random.choice(questions, 1)
        # Context
        key = "protocolSection.identificationModule"
        context = get_recursive(study, key)
        # Answer
        title_key = "protocolSection.identificationModule.briefTitle"
        offical_title_key = "protocolSection.identificationModule.officialTitle"
        title = get_recursive(study, title_key)
        offical_title = get_recursive(study, offical_title_key)
        answer = offical_title or title or "UNKNOWN"

        return question, context, answer

    # Question 2 - Summarise clinical trial (a.k.a. study) {nctId}
    def question_2(self):
        study = self.study
        nctId = self.nctId
        # Question
        questions = [
            f"Summarise clinical trial (a.k.a. study) {nctId}.",
        ]
        question = np.random.choice(questions, 1)
        # Context
        key_desc = "protocolSection.descriptionModule.detailedDescription"
        key_sum = "protocolSection.descriptionModule.briefSummary"
        context = (
            get_recursive(study, key_desc) or get_recursive(study, key_sum) or "NA"
        )

        # Answer
        answer = get_recursive(study, key_sum) or "NA"

        return question, context, answer

    # Question 3 - Is clinical trial (a.k.a. study) {nctId} and interventional or observational study?
    def question_3(self):
        study = self.study
        nctId = self.nctId
        # Question
        questions = [
            f"Is clinical trial (a.k.a. study) {nctId} and interventional or observational study? Select from [INTERVENTIONAL, OBSERVATIONAL, NA]",
        ]
        question = np.random.choice(questions, 1)
        # Context
        key = "protocolSection.designModule"
        context = get_recursive(study, key) or "/NA"
        # Answer
        study_type_key = "protocolSection.designModule.studyType"
        answer = get_recursive(study, study_type_key) or "NA"
        return question, context, answer

    # Question 4 - What condition is studied in clinical trial (a.k.a. study) {nctId}?
    def question_4(self):
        study = self.study
        nctId = self.nctId
        # Question
        questions = [
            f"What condition is studied in clinical trial (a.k.a. study) {nctId}?",
        ]
        question = np.random.choice(questions, 1)
        # Context
        key_desc = "protocolSection.descriptionModule.detailedDescription"
        key_sum = "protocolSection.descriptionModule.briefSummary"
        context = (
            get_recursive(study, key_desc) or get_recursive(study, key_sum) or "NA"
        )
        # Answer
        answer = self.conditions
        return question, context, answer

    # Question 5 - Is {condition} studied in clinical trial (a.k.a. study) {nctId}?
    def question_5(self):
        study = self.study
        nctId = self.nctId
        conditions = self.conditions
        other_conditions = self.other_conditions
        # Answer
        answer = np.random.choice(["TRUE", "FALSE"])
        if answer == "TRUE":
            condition = np.random.choice(conditions, 1)
        else:
            condition = np.random.choice(other_conditions, 1)

        # Question
        questions = [
            f"Is {condition[0]} studied in clinical trial (a.k.a. study) {nctId}? Select from [TRUE, FALSE, NA]",
        ]
        question = np.random.choice(questions, 1)
        # Context
        key = key = "derivedSection.conditionBrowseModule"
        context = get_recursive(study, key) or "NA"
        return question, context, answer

    # Question 6 - What drugs / treatments is studied in clinical trial (a.k.a. study) {nctId}?
    def question_6(self):
        study = self.study
        nctId = self.nctId
        study_type = self.study_type
        interventions = self.interventions
        # Question
        questions = [
            f"What drugs / treatments is studied in clinical trial (a.k.a. study) {nctId}?",
        ]
        question = np.random.choice(questions, 1)
        # Context
        key = "protocolSection.armsInterventionsModule.interventions"
        context = get_recursive(study, key) or "NA"
        # Answer
        if study_type == "INTERVENTIONAL":
            answer = interventions
        elif study_type == "OBSERVATIONAL":
            answer = interventions or "Observation"
        else:
            raise ValueError(f"{study_type} is not a right study type")
        return question, context, answer

    # Question 7 - In what phase is clinical trial (a.k.a. study) {nctId}?
    def question_7(self):
        study = self.study
        nctId = self.nctId
        # Question
        questions = [
            f"In what phase is clinical trial (a.k.a. study) {nctId}? Select from [Early Phase 1, Phase 1, Phase 2, Phase 3, Phase 4, NA]",
        ]
        question = np.random.choice(questions, 1)
        # Context
        key = "protocolSection.designModule"
        context = get_recursive(study, key) or "NA"
        # Answer
        phases_key = "protocolSection.designModule.phases"
        phases_dict = {
            "NA": "NA - NA",
            "EARLY_PHASE1": "Early Phase 1",
            "PHASE1": "Phase 1",
            "PHASE2": "Phase 2",
            "PHASE3": "Phase 3",
            "PHASE4": "Phase 4",
        }
        answer = get_recursive(study, phases_key) or "NA"
        answer = [phases_dict[x] for x in answer]

        return question, context, answer

    # Question 8 - Describe clinical trial (a.k.a. study) {nctId} as a PICO question.
    def question_8(self):
        study = self.study
        nctId = self.nctId
        # Question
        questions = [
            f"Describe clinical trial (a.k.a. study) {nctId} as a PICO question. Where P stands for Population/Patient/Problem. I for Intervention. C for Comparison and O for Outcome. For example: In adult patients with total hip replacements (Population), how effective is pain medication (Intervention) compared to aerobic stretching (Comparison) in controlling post operative pain (Outcome) during the perioperative and recovery time (Time)?",
        ]
        question = np.random.choice(questions, 1)
        # Context
        key1 = "protocolSection.descriptionModule.briefSummary"
        key2 = "protocolSection.descriptionModule.detailedDescription"
        context = get_recursive(study, key1) or get_recursive(study, key2) or "NA"
        # Answer
        # Generate response with LLM
        answer = qa_with_context(question=question, context=context).answer or "NA"
        return question, context, answer

    # Question 9 - How many patients to be enrolled in clinical trial (a.k.a. study) {nctId}?
    def question_9(self):
        study = self.study
        nctId = self.nctId
        # Question
        questions = [
            f"How many patients to be enrolled in clinical trial (a.k.a. study) {nctId}?",
            f"How many patients are recruited in clinical trial (a.k.a. study) {nctId}?",
        ]
        question = np.random.choice(questions, 1)
        # Context
        key = "protocolSection.designModule.enrollmentInfo"
        context = get_recursive(study, key) or {"count": "NA", "type": "NA"}

        # Answer
        key = "protocolSection.designModule.enrollmentInfo.count"
        answer = get_recursive(study, key) or "UNKNOWN"
        return question, context, answer

    # Question 10 - What is the eligibility criteria for clinical trial (a.k.a. study) {nctId}?
    def question_10(self):
        study = self.study
        nctId = self.nctId
        # Question
        questions = [
            f"What is the eligibility criteria for clinical trial (a.k.a. study) {nctId}?",
        ]
        question = np.random.choice(questions, 1)
        # Context
        key = "protocolSection.eligibilityModule"
        context = get_recursive(study, key) or "NA"
        # Answer
        key = "protocolSection.eligibilityModule.eligibilityCriteria"
        answer = get_recursive(study, key) or "NA"
        return question, context, answer

    # Question 11 - What is what is the min. and max. age range in clinical trial (a.k.a. study) {nctId}?
    def question_11(self):
        study = self.study
        nctId = self.nctId
        # Question
        questions = [
            f"What is what is the min. and max. age range in clinical trial (a.k.a. study) {nctId}?. Example: Age Range: [18 - 55]",
        ]
        question = np.random.choice(questions, 1)
        # Context
        key = "protocolSection.eligibilityModule"
        context = get_recursive(study, key) or "NA"
        # context = yaml.dump(context)
        # Answer
        key = "protocolSection.eligibilityModule.minimumAge"
        _min = get_recursive(study, key) or "NA"
        key = "	protocolSection.eligibilityModule.maximumAge"
        _max = get_recursive(study, key) or "NA"
        answer = f"Age Range: [{_min} - {_max}]"
        return question, context, answer

    # Question 12 - Describe the method of participant recruitment utilized in clinical trial (a.k.a. study) {nctId}
    def question_12(self):
        study = self.study
        nctId = self.nctId
        # Question
        questions = [
            f"Describe the method of participant recruitment utilized in clinical trial (a.k.a. study) {nctId}.",
        ]
        question = np.random.choice(questions, 1)
        # Context
        key = ""
        context = get_recursive(study, key) or "TODO"
        # Answer
        key = ""
        answer = get_recursive(study, key) or "TODO"
        return question, context, answer

    # Question 13 - Describe the intervention model in clinical trial (a.k.a. study) {nctId}
    def question_13(self):
        study = self.study
        nctId = self.nctId
        # Question
        questions = [
            f"Describe the intervention model in clinical trial (a.k.a. study) {nctId}",
        ]
        question = np.random.choice(questions, 1)
        # Context
        key1 = "protocolSection.designModule.designInfo.interventionModelDescription"
        key2 = "protocolSection.designModule.designInfo"
        context = get_recursive(study, key1) or get_recursive(study, key2) or "NA"
        # Answer

        answer = qa_with_context(question=question, context=context).answer or "NA"
        return question, context, answer

    # Question 14 - Describe the primary purpose of clinical trial (a.k.a. study) {nctId}
    def question_14(self):
        study = self.study
        nctId = self.nctId
        # Question
        questions = [
            f"Describe the primary purpose of clinical trial (a.k.a. study) {nctId}",
        ]
        question = np.random.choice(questions, 1)
        # Context
        key = ""
        context = get_recursive(study, key) or "NA"
        # Answer
        key = ""
        answer = get_recursive(study, key) or "NA"
        return question, context, answer

    # Question 15 - What is the purpose of clinical trial (a.k.a. study) {nctId}?
    def question_15(self):
        study = self.study
        nctId = self.nctId
        # Question
        questions = [
            f"What is the purpose of clinical trial (a.k.a. study) {nctId}? Select one from [Treatment, Prevention, Diagnostic, Educational/Counseling/Training, Supportive Care, Screening, Health Services Research, Basic Science, Device Feasibility, Other, NA]"
        ]
        question = np.random.choice(questions, 1)
        # Context
        key = "protocolSection.designModule.designInfo"
        context = get_recursive(study, key) or "NA"
        # context = yaml.dump(context)
        # Answer
        key = "protocolSection.designModule.designInfo.primaryPurpose"
        answer = get_recursive(study, key) or "NA"
        purpose_dict = {
            "TREATMENT": "Treatment",
            "PREVENTION": "Prevention",
            "DIAGNOSTIC": "Diagnostic",
            "ECT": "Educational/Counseling/Training",
            "SUPPORTIVE_CARE": "Supportive Care",
            "SCREENING": "Screening",
            "HEALTH_SERVICES_RESEARCH": "Health Services Research",
            "BASIC_SCIENCE": "Basic Science",
            "DEVICE_FEASIBILITY": "Device Feasibility",
            "OTHER": "Other",
            "NA": "NA",
        }
        answer = purpose_dict[answer]
        return question, context, answer

    # Question 16 - What intervention types are used in this study?
    def question_16(self):
        study = self.study
        nctId = self.nctId
        # Question
        questions = [
            f"What intervention types are used in clinical trial (a.k.a. study) {nctId}? Select one from [BEHAVIORAL, BIOLOGICAL, COMBINATION_PRODUCT, DEVICE, DIAGNOSTIC_TEST, DIETARY_SUPPLEMENT, DRUG, GENETIC, PROCEDURE, RADIATION, OTHER, NA]",
        ]
        question = np.random.choice(questions, 1)
        # Context
        key = "protocolSection.armsInterventionsModule"
        context = get_recursive(study, key) or {"NA"}
        # context = yaml.dump(context)
        # Answer
        key = "protocolSection.armsInterventionsModule.interventions.type"
        answer = get_recursive(study, key) or "NA"
        return question, context, answer

    # Question 17 - Are there any blinding procedures implemented in clinical trial (a.k.a. study) {nctId}? If so, describe them.
    def question_17(self):
        study = self.study
        nctId = self.nctId
        # Question
        questions = [
            # f"Are there any blinding procedures implemented in clinical trial (a.k.a. study) {nctId}? If so, describe them.",
            f"What blinding (a.k.a. masking) is implemented in clinical trial (a.k.a. study) {nctId}? Select relevant from [PARTICIPANT, CARE_PROVIDER, INVESTIGATOR, OUTCOMES_ASSESSOR, NA]",
        ]
        question = np.random.choice(questions, 1)
        # Context
        key = "	protocolSection.designModule.designInfo"
        context = get_recursive(study, key) or "NA"
        # context = yaml.dump(context)
        # Answer
        key = "protocolSection.designModule.designInfo.maskingInfo.whoMasked"
        answer = get_recursive(study, key) or "NA"
        return question, context, answer

    # Question 18 - What is the allocation strategy employed in clinical trial (a.k.a. study) {nctId}?
    def question_18(self):
        study = self.study
        nctId = self.nctId
        # Question
        questions = [
            f"What is the allocation strategy employed in clinical trial (a.k.a. study) {nctId}? Select from [RANDOMIZED, NON_RANDOMIZED, NA]",
        ]
        question = np.random.choice(questions, 1)
        # Context
        key = "protocolSection.designModule.designInfo"
        context = get_recursive(study, key) or "NA"
        # Answer
        key = "protocolSection.designModule.designInfo.allocation"
        answer = get_recursive(study, key) or "NA"
        return question, context, answer

    # Question 19 - Does clinical trial (a.k.a. study) {nctId} uses a patient registry? [TRUE, FALSE]
    def question_19(self):
        study = self.study
        nctId = self.nctId
        # Question
        questions = [
            f"Does clinical trial (a.k.a. study) {nctId} uses a patient registry? Select from [TRUE, FALSE, NA]",
        ]
        question = np.random.choice(questions, 1)
        # Context
        key = "protocolSection.designModule"
        context = get_recursive(study, key) or "NA"
        # Answer
        key = "protocolSection.designModule.patientRegistry"
        answer = get_recursive(study, key) or False
        return question, context, answer

    # Question 20 - What is the study population in clinical trial (a.k.a. study) {nctId}?
    def question_20(self):
        study = self.study
        nctId = self.nctId
        # Question
        questions = [
            f"What is the study population in clinical trial (a.k.a. study) {nctId}?",
        ]
        question = np.random.choice(questions, 1)
        # Context
        key = "protocolSection.eligibilityModule"
        context = get_recursive(study, key) or "NA"
        # Answer
        key = "protocolSection.eligibilityModule.studyPopulation"
        answer = get_recursive(study, key) or "NA"
        return question, context, answer

    # Question 21 - What type of observational model is used in {nctId}?
    def question_21(self):
        study = self.study
        nctId = self.nctId
        # Question
        questions = [
            f"What type of observational model is used in {nctId}? [Cohort, Case-Control, Case-Only, Case-CrossOver, Ecologic, Family-based, Defined Population, Natural History, Other, NA]"
        ]
        question = np.random.choice(questions, 1)
        # Context
        key = "protocolSection.designModule"
        context = get_recursive(study, key) or "UNKNOWN"
        # Answer
        key = "protocolSection.designModule.designInfo.observationalModel"
        answer = get_recursive(study, key) or "NA"
        if answer != "NA":
            answer = answer.capitalize()
        return question, context, answer

    # Question 22 - What is the design time frame in clinical trial (a.k.a. study) {nctId}?
    def question_22(self):
        study = self.study
        nctId = self.nctId
        # Question
        questions = [
            f"What is the design time frame in clinical trial (a.k.a. study) {nctId}? [Retrospective, Prospective, Cross-Sectional, Other, NA]"
        ]
        question = np.random.choice(questions, 1)
        # Context
        key = "protocolSection.designModule"
        context = get_recursive(study, key) or "UNKNOWN"
        # Answer
        key = "protocolSection.designModule.designInfo.timePerspective"
        answer = get_recursive(study, key) or "NA"
        if answer != "NA":
            answer = answer.capitalize()
        return question, context, answer

    # Question 23 - Describe the primary outcomes of clinical trial (a.k.a. study) {nctId}
    def question_23(self):
        study = self.study
        nctId = self.nctId
        # Question
        questions = [
            f"Describe the primary outcomes of clinical trial (a.k.a. study) {nctId}",
            f"What are the outcomes of clinical trial (a.k.a. study) {nctId}?",
            f"Outline the primary outcome measure(s) assessed in clinical {nctId}",
        ]
        question = np.random.choice(questions, 1)
        # Context
        key = "protocolSection.outcomesModule"
        context = get_recursive(study, key) or "NA"
        # Answer
        key = "protocolSection.outcomesModule.primaryOutcomes.description"
        answer = get_recursive(study, key) or "NA"
        return question, context, answer

    # Question 24 - Is there a results section for clinical trial (a.k.a. study) {nctId}?
    def question_24(self):
        study = self.study
        nctId = self.nctId
        # Question
        questions = [
            f"Is there a results section for clinical trial (a.k.a. study) {nctId}?",
        ]
        question = np.random.choice(questions, 1)
        # Context
        key = "resultsSection.outcomeMeasuresModule.outcomeMeasures"
        context = get_recursive(study, key) or "NA"
        # Answer
        key = "hasResults"
        answer = get_recursive(study, key) or False
        answer = str(answer)
        return question, context, answer

    # Question 25 - Where there any adverse events described in clinical trial (a.k.a. study) {nctId}?
    def question_n(self):
        study = self.study
        nctId = self.nctId
        # Question
        questions = [
            f"Where there any adverse events described in clinical trial (a.k.a. study) {nctId}?",
        ]
        question = np.random.choice(questions, 1)
        # Context
        key = "resultsSection.adverseEventsModule"
        context = get_recursive(study, key) or "NA"
        # Answer
        key = "resultsSection.adverseEventsModule.seriousEvents"
        serious = get_recursive(study, key) or "NA"

        key = "resultsSection.adverseEventsModule.otherEvents"
        other = get_recursive(study, key) or "NA"

        return question, context, [serious, other]

    def get_single_question(self, nctId, idx):
        question_func = getattr(self, f"question_{idx}")
        question, context, answer = question_func()

        def output_formatting(x):

            if isinstance(x, int) or isinstance(x, bool):
                x = str(x)

            if isinstance(x, dict):
                x = yaml.dump(x)

            if isinstance(x, list):
                tmp = []
                for i in x:
                    if isinstance(i, dict):
                        i = yaml.dump(i)
                    tmp.append(i)
                x = tmp
                x = set(x) - set([None])
                x = ", ".join(set(x))
                if not isinstance(x, str):
                    raise ValueError(f"{x} not a string")

            formatting_dict = {
                "\t": "",  # remove tabulation
                "   ": " ",  # amend triple whitespace
                "  ": " ",  # amend double whitespace
                "\n\n\n": "\n",  # remove triple newlines
                "\n\n": "\n",  # remove double newlines
                "\n": " | ",  # replace newline character
                '"': "",  # remove quotation
                "'": "",  # remove quotation
            }
            for k, v in formatting_dict.items():
                x = x.replace(k, v)

            return x

        question = output_formatting(question[0])
        context = output_formatting(context)
        answer = output_formatting(answer)

        formatted_triplet = {
            "idx": f"{nctId}_{idx:03}",
            "nctId": nctId,
            "question": question,
            "context": context,
            "answer": answer,
        }
        return formatted_triplet

    def question_randomizer(self, n: int = 5):
        nctId = self.nctId
        study_seed = int(nctId.replace("NCT", ""))
        np.random.seed(study_seed)

        questions_idx = np.random.choice(
            self.study_question_space, size=n, replace=False
        )
        questions_idx = np.sort(questions_idx)

        questions = []
        for idx in questions_idx:
            formatted_triplet = self.get_single_question(nctId, idx)
            questions.append(formatted_triplet)
        return questions

    def get_all_questions(self):
        nctId = self.nctId
        questions = []
        for idx in self.study_question_space:
            print(idx)
            formatted_triplet = self.get_single_question(nctId, idx)
            questions.append(formatted_triplet)
        return questions


def main(args):

    # Set LLM
    lm = dspy.HFClientVLLM(
        model=args.vllm,
        port=args.port,
        url=args.host,
        max_tokens=1_000,
        timeout_s=2_000,
    )
    dspy.settings.configure(lm=lm, temperature=0.7)

    # Connect to Mongo DB and pull a collection of Clinical Studies protocols
    # NOTE: Originally I would have pulled the CTs from ct.gov
    # but as as I am working with a subset (based on the trialgpt paper)
    # I had to put them on MongoDB. Moreover, I decided to trim them.
    MONGODB_USER = os.getenv("MONGODB_USER")
    MONGODB_PWD = os.getenv("MONGODB_PWD")

    client = connect_to_mongoDB(MONGODB_USER, MONGODB_PWD)
    db = client["ctGov"]
    collection = db["trialgpt"]
    studies = collection.find({})

    # Generate dataset
    np.random.seed(123)
    output_df = pd.DataFrame([])
    i = 0
    o = 0
    for study in tqdm(studies, desc="Generating ctGov questioner..."):
        questioner = CtGovStudyQuestioner(study)

        if questioner.study_type == "INTERVENTIONAL":
            i += 1
            if i > 5:
                continue
        elif questioner.study_type == "OBSERVATIONAL":
            o += 1
            if o > 5:
                continue
        elif questioner.study_type == "EXPANDED_ACCESS":
            continue
        else:
            raise NotImplementedError(f"{questioner.study_type } not implemented")

        study_questions = questioner.get_all_questions()
        output_df = pd.concat([output_df, pd.DataFrame(study_questions)])

        if (i > 5) and (o > 5):
            break

    # # Generate dataset
    # np.random.seed(123)
    # output_df = pd.DataFrame([])
    # for study in tqdm(studies):
    #     questioner = CtGovStudyQuestioner(study)
    #     study_questions = questioner.question_randomizer(args.n)
    #     study_questions_df = pd.DataFrame(study_questions)
    #     output_df = pd.concat(output_df, pd.DataFrame())

    output_df.to_csv(
        os.path.join(args.output_dir, args.output_file),
        sep=args.delimiter,
        index=False,
        # quotechar="'",
    )


if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(
        description="Given a collection of clinical trials studies from clinicalTrials.gov, generate question-context-answer triplets."
    )
    parser.add_argument(
        "--output_dir", type=str, help="output directory", default="./data/"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="output filename",
        default="ctGov.questioner.tsv",
    )
    parser.add_argument("--delimiter", type=str, help="delimiter", default="\t")
    parser.add_argument(
        "--n", metavar="n", type=int, help="number of questions per study", default=5
    )
    parser.add_argument("--l", type=int, help="Total number of questions", default=5)

    parser.add_argument(
        "-vllm",
        type=str,
        default="TheBloke/meditron-7B-GPTQ",
        help="Large Language Model name using HF nomenclature. E.g. 'TheBloke/meditron-7B-GPTQ'.",
    )

    parser.add_argument(
        "-host",
        type=str,
        default="http://0.0.0.0",
        help="LLM server host.",
    )

    parser.add_argument(
        "-port",
        type=int,
        default=8_000,
        help="LLM server port.",
    )

    # Parse the arguments
    args = parser.parse_args()
    # print(args.accumulate(args.Integers))

    main(args)
